"""
core/impact_detector.py

Detects hammer strikes on the excitation channel for the experimental
modal analysis mode - see `data/models.py::ModalAnalysisConfig`,
`analysis/modal_analysis.py::ModalAverager`.

Architecture context:

    Ring Buffer -> ModalLiveView (per-tick reader, ~15ms) -> ImpactDetector.feed()
                                                           -> ImpactDetector.check_double_hit()/check_overload()
                                                           -> ModalAverager.add_impact()

`ImpactDetector` is deliberately Qt- AND ring-buffer-free: it only ever
sees plain numpy arrays and absolute sample indices that its caller
(`gui/modal_live_view.py::ModalLiveView`) reads from
`core/controller.py::MeasurementController`. That keeps it trivially
unit-testable (no QApplication, no fake hardware) and keeps the actual
reader lifecycle - registering a normal live reader, and later a
throwaway RETROACTIVE reader with `back_samples` to extract the
captured window - entirely in `ModalLiveView`, which is the one place
that already owns that lifecycle for the standard live view too.

Why this exists alongside the standard trigger (`gui/live_view.py`,
`data/models.py::TriggerCondition`): that trigger checks only the LAST
sample of each ~15-25ms GUI tick
(`LiveView._check_threshold_trigger`), which is enough to start/stop a
recording but would miss most 1-5ms hammer pulses entirely. This
detector scans the WHOLE block every tick instead
(`data.models.evaluate_threshold_crossings`), and - unlike the standard
trigger, which fires once per measurement (see
`gui/main_window.py::_on_trigger_fired`) - is designed to fire
repeatedly within one continuous, uninterrupted recording, one hammer
strike at a time.

All sample values this module receives MUST already be in the
channel's physical unit (scaled, e.g. via
`core.measurement.apply_scaling`) - exactly like `LiveView` scales data
before evaluating its own trigger. `ModalAnalysisConfig.impact_condition
.threshold_value` and `.overload_fraction` are both expressed in
physical units (Volts, g, N, ... depending on the channel), not raw
ADC counts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from data.models import ModalAnalysisConfig, evaluate_threshold_crossings


def _ms_to_samples(milliseconds: float, sample_rate_hz: float) -> int:
    return round(milliseconds / 1000.0 * sample_rate_hz)


class ImpactDetector:
    """Stateful, per-sample-accurate hammer-strike detector.

    A small state machine with three states:

        ARMED       - scanning every new block for a rising edge over
                      `config.impact_condition`.
        PENDING     - a strike was detected; waiting for enough
                      subsequent samples to exist so the full
                      pretrigger+posttrigger window can be extracted
                      (see `is_ready_to_capture`/`window_start_index`).
        REFRACTORY  - a window was just resolved (accepted OR
                      rejected - see `finish_capture`); ignoring new
                      strikes for `config.min_rest_time_ms` so the
                      still-decaying response of THIS strike is never
                      mistaken for a new one.

    The caller (`ModalLiveView`) drives the transitions PENDING ->
    (extract window) -> REFRACTORY explicitly via
    `is_ready_to_capture()`/`window_start_index()`/`finish_capture()`;
    ARMED <-> REFRACTORY is handled automatically inside `feed()`.
    """

    def __init__(
        self,
        config: ModalAnalysisConfig,
        sample_rate_hz: float,
        block_size: int,
    ) -> None:
        self._config = config
        self._sample_rate_hz = sample_rate_hz
        self._block_size = block_size
        self._pretrigger_samples = _ms_to_samples(config.pretrigger_ms, sample_rate_hz)
        self._min_rest_samples = _ms_to_samples(config.min_rest_time_ms, sample_rate_hz)
        self._double_hit_window_samples = _ms_to_samples(
            config.double_hit_window_ms, sample_rate_hz
        )
        if self._pretrigger_samples >= block_size:
            raise ValueError(
                "pretrigger_ms ist so groß, dass für den Block keine Samples nach dem "
                "Schlag mehr übrig blieben - pretrigger_ms verkleinern oder "
                "frequency_resolution_hz vergrößern (kleinerer Block)."
            )

        self._state = "armed"
        # None = "no prior sample known yet" - deliberately three-valued
        # rather than starting at False: an excitation channel that
        # already reads above threshold the INSTANT the detector arms
        # must not immediately fire (there is no genuine rising edge,
        # just an unknown prior state) - see the None-handling in
        # `feed()`. Exactly the same reasoning as
        # `gui/live_view.py::LiveView.enter_armed_state` resetting its
        # own edge state to `None` rather than `False` on every arm.
        self._last_above_threshold: Optional[bool] = None
        self._pending_trigger_index: Optional[int] = None
        self._refractory_until_index: Optional[int] = None

    def reset(self) -> None:
        """Back to a fresh, armed detector - no pending capture, no
        refractory period, edge-detection state cleared. Used when the
        operator presses "Zurücksetzen" (see `gui/modal_live_view.py`)."""
        self._state = "armed"
        self._last_above_threshold = None
        self._pending_trigger_index = None
        self._refractory_until_index = None

    def feed(self, excitation_block: np.ndarray, absolute_start_index: int) -> Optional[int]:
        """Scans one new block of excitation samples for a hammer
        strike - the whole block, not just its last sample (see the
        module docstring for why that distinction matters here).

        Args:
            excitation_block: New excitation-channel samples, already
                in physical units, in acquisition order.
            absolute_start_index: The ring buffer's absolute sample
                index (`RingBuffer`'s monotonic write counter) that
                `excitation_block[0]` corresponds to - needed to report
                the strike's position in that same absolute space, and
                to detect an edge that straddles two blocks.

        Returns:
            The absolute sample index of the detected rising edge, if
            one occurred AND the detector was ARMED at the time -
            `None` otherwise (nothing crossed the threshold, or a
            capture is already PENDING/REFRACTORY). At most one index
            is ever returned per call - a second crossing within the
            same block is caught on a later call, once the current one
            has been resolved via `finish_capture()`.
        """
        excitation_block = np.asarray(excitation_block, dtype=float)
        if excitation_block.size == 0:
            return None

        block_end_index = absolute_start_index + excitation_block.size

        if self._state == "refractory":
            if self._refractory_until_index is not None and block_end_index >= self._refractory_until_index:
                self._state = "armed"
                self._refractory_until_index = None
                # A fresh arm - see the `None` rationale in `__init__` -
                # deliberately discards whatever level was tracked
                # during refractory, then falls through to scan THIS
                # SAME block below rather than waiting for the next
                # tick.
                self._last_above_threshold = None
            else:
                self._track_edge_state_only(excitation_block)
                return None

        if self._state == "pending":
            self._track_edge_state_only(excitation_block)
            return None

        # state == "armed"
        mask = evaluate_threshold_crossings(excitation_block, self._config.impact_condition)
        if self._last_above_threshold is None:
            # No prior sample to compare the very first one against -
            # seed with the block's own first value so THAT sample can
            # never register as a rising edge (an already-exceeded
            # channel must not fire the instant we start watching it),
            # while a genuine later edge within this same block is
            # still caught normally.
            seed = bool(mask[0])
        else:
            seed = self._last_above_threshold
        extended = np.concatenate(([seed], mask))
        rising_edges = np.flatnonzero((~extended[:-1]) & extended[1:])
        self._last_above_threshold = bool(mask[-1])
        if rising_edges.size == 0:
            return None

        trigger_index = absolute_start_index + int(rising_edges[0])
        self._pending_trigger_index = trigger_index
        self._state = "pending"
        return trigger_index

    def _track_edge_state_only(self, excitation_block: np.ndarray) -> None:
        """Keeps the edge-detection carry state current while PENDING/
        REFRACTORY, without acting on it - a strike occurring while the
        detector cannot respond to it must not fake a false edge (a
        transition from "already above threshold" to "still above
        threshold") once it becomes ARMED again."""
        mask = evaluate_threshold_crossings(excitation_block, self._config.impact_condition)
        if mask.size:
            self._last_above_threshold = bool(mask[-1])

    def is_ready_to_capture(self, current_absolute_index: int) -> bool:
        """Whether enough samples now exist (from a PENDING trigger
        onward) to extract the full `block_size`-sample window -
        `current_absolute_index` is the number of samples written to
        the ring buffer so far (its `_total_written`, exposed e.g. via
        the count `read_live_data` has advanced past)."""
        if self._pending_trigger_index is None:
            return False
        needed_until = self._pending_trigger_index + (self._block_size - self._pretrigger_samples)
        return current_absolute_index >= needed_until

    def window_start_index(self) -> Optional[int]:
        """Absolute sample index the captured window should START at
        (pretrigger samples before the detected strike), or `None` if
        no capture is currently pending. Used by `ModalLiveView` to
        compute the `back_samples` argument of a retroactive
        `MeasurementController.register_reader(back_samples=...)`."""
        if self._pending_trigger_index is None:
            return None
        return self._pending_trigger_index - self._pretrigger_samples

    def finish_capture(self) -> None:
        """Resolves the PENDING capture and starts the refractory
        period - called once the window has been extracted, REGARDLESS
        of whether it was accepted or rejected (double hit, overload):
        a rejected strike's ringdown must still be waited out before
        the detector arms again, exactly like an accepted one's.

        Raises:
            ValueError: if no capture is currently pending.
        """
        if self._pending_trigger_index is None:
            raise ValueError("Es ist gerade keine Erfassung offen, die abgeschlossen werden könnte.")
        capture_end_index = self._pending_trigger_index + (
            self._block_size - self._pretrigger_samples
        )
        self._pending_trigger_index = None
        self._state = "refractory"
        self._refractory_until_index = capture_end_index + self._min_rest_samples

    def is_in_refractory(self, now_absolute_index: int) -> bool:
        """Read-only status query (no state change) - e.g. for a "still
        settling" indicator in `ModalLiveView`. The actual ARMED
        transition happens lazily inside `feed()`, not here."""
        return (
            self._state == "refractory"
            and self._refractory_until_index is not None
            and now_absolute_index < self._refractory_until_index
        )

    def check_double_hit(self, excitation_window: np.ndarray) -> bool:
        """Checks a captured excitation window for a double hit - a
        second, unintended bounce of the hammer shortly after the main
        strike, which distorts the excitation spectrum and must not be
        averaged in as if it were a single clean impact.

        Finds the window's single largest-magnitude sample (the main
        strike - normally located `pretrigger_ms` into the window) and
        looks for a second peak within `double_hit_window_ms`
        AFTERWARDS that reaches at least
        `double_hit_relative_threshold` of the main peak's magnitude.

        Returns:
            True if a double hit was found.
        """
        excitation_window = np.asarray(excitation_window, dtype=float)
        if excitation_window.size == 0:
            return False

        main_peak_index = int(np.argmax(np.abs(excitation_window)))
        main_peak_value = abs(excitation_window[main_peak_index])
        if main_peak_value == 0.0:
            return False

        search_start = main_peak_index + 1
        search_end = min(
            excitation_window.size, main_peak_index + 1 + self._double_hit_window_samples
        )
        if search_start >= search_end:
            return False

        secondary_peak_value = np.max(np.abs(excitation_window[search_start:search_end]))
        # bool(...): comparing numpy scalars yields numpy.bool_, not a
        # Python bool - `numpy.bool_(True) is True` is False, which
        # would silently break any `is True`/`is False` check on the
        # result (as well as e.g. JSON serialization of it).
        return bool(secondary_peak_value >= self._config.double_hit_relative_threshold * main_peak_value)

    def check_overload(
        self,
        excitation_window: np.ndarray,
        response_window: np.ndarray,
        excitation_range: float,
        response_range: float,
    ) -> bool:
        """Checks whether either channel's captured window reaches
        `config.overload_fraction` of its configured measurement range
        (`data.models.Channel.max_range`) - a clipped/overdriven
        channel produces a physically meaningless spectrum and must not
        be averaged in.

        Args:
            excitation_range/response_range: The respective channel's
                `max_range` (assumed symmetric, as every currently
                supported IEPE/voltage channel's range is).

        Returns:
            True if either channel is at or above its overload
            fraction.
        """
        excitation_peak = np.max(np.abs(excitation_window)) if excitation_window.size else 0.0
        response_peak = np.max(np.abs(response_window)) if response_window.size else 0.0
        # bool(...): see the comment in check_double_hit - the operands
        # here are numpy scalars, so the raw comparison would otherwise
        # be numpy.bool_ rather than a Python bool.
        return bool(
            excitation_peak >= self._config.overload_fraction * abs(excitation_range)
            or response_peak >= self._config.overload_fraction * abs(response_range)
        )
