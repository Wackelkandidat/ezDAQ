"""
tests/test_impact_detector.py

Tests for `core/impact_detector.py::ImpactDetector` - the per-sample-
accurate hammer-strike detector behind the experimental modal analysis
mode.

Deliberately does NOT need a `QApplication`/Qt or a `MeasurementController`
/ring buffer: `ImpactDetector` only ever sees plain numpy arrays and
absolute sample indices, exactly what its caller
(`gui/modal_live_view.py::ModalLiveView`, not built yet) would read
from those - see the module docstring for why that separation exists
(mirrors `tests/test_naming.py`/`tests/test_rate_groups.py`).
"""

from __future__ import annotations

import unittest

import numpy as np

from core.impact_detector import ImpactDetector
from data.models import ModalAnalysisConfig, TriggerCondition, TriggerDirection


def _make_config(**overrides) -> ModalAnalysisConfig:
    condition = TriggerCondition(
        threshold_value=5.0, threshold_direction=TriggerDirection.RISES_ABOVE
    )
    defaults = dict(
        impact_condition=condition,
        pretrigger_ms=5.0,
        min_rest_time_ms=50.0,
        double_hit_window_ms=5.0,
        double_hit_relative_threshold=0.5,
        overload_fraction=0.95,
    )
    defaults.update(overrides)
    return ModalAnalysisConfig(**defaults)


_SAMPLE_RATE_HZ = 10_000.0
_BLOCK_SIZE = 1000  # pretrigger_ms=5.0 -> 50 samples at 10 kHz, well under this


def _trigger_at(detector: ImpactDetector, absolute_index: int):
    """Feeds a below-threshold sample immediately before `absolute_index`
    and then an above-threshold one AT it - the minimal realistic way
    to simulate "a strike happens here" against a freshly armed
    detector.

    A detector's very FIRST observed sample can never by itself
    register as a rising edge, deliberately - see the `None`-seed
    rationale in `ImpactDetector.__init__` (mirrors
    `gui/live_view.py::LiveView` never firing on the very first check
    after arming). Tests that only care about what happens AFTER a
    trigger use this helper rather than re-deriving that priming step
    themselves.
    """
    detector.feed(np.array([0.0]), absolute_start_index=absolute_index - 1)
    return detector.feed(np.array([10.0]), absolute_start_index=absolute_index)


class EdgeDetectionTest(unittest.TestCase):
    """`feed()` must scan the WHOLE block (the gap this detector exists
    to close - see the module docstring), not just its last sample."""

    def test_quiet_block_produces_no_trigger(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)

        self.assertIsNone(detector.feed(np.zeros(200), absolute_start_index=0))

    def test_a_crossing_anywhere_in_the_block_is_found(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        block = np.zeros(200)
        block[50] = 10.0  # not the last sample - would be missed by a last-sample-only check

        self.assertEqual(detector.feed(block, absolute_start_index=1000), 1050)

    def test_edge_spanning_a_block_boundary_is_still_detected(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        below_threshold = np.zeros(100)
        below_threshold[-1] = 2.0
        self.assertIsNone(detector.feed(below_threshold, absolute_start_index=0))

        above_threshold = np.full(100, 10.0)
        # The edge is between the two blocks, at the very first sample
        # of this second one - only detectable by carrying the previous
        # block's threshold state forward.
        self.assertEqual(detector.feed(above_threshold, absolute_start_index=100), 100)

    def test_level_already_above_threshold_at_construction_does_not_fire(self) -> None:
        """No RISING edge without a prior below-threshold sample to rise
        from - mirrors `LiveView`'s own edge-triggered trigger."""
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        already_high = np.full(50, 10.0)

        self.assertIsNone(detector.feed(already_high, absolute_start_index=0))

    def test_no_second_trigger_while_a_capture_is_pending(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        first = np.zeros(200)
        first[10] = 10.0
        self.assertEqual(detector.feed(first, absolute_start_index=0), 10)

        still_high = np.full(50, 10.0)
        self.assertIsNone(detector.feed(still_high, absolute_start_index=200))


class CaptureWindowTest(unittest.TestCase):
    """`is_ready_to_capture()`/`window_start_index()` - the interface
    `ModalLiveView` uses to know WHEN and WHERE to extract the captured
    block from the ring buffer."""

    def test_window_starts_pretrigger_samples_before_the_strike(self) -> None:
        detector = ImpactDetector(_make_config(pretrigger_ms=5.0), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        block = np.zeros(10)
        block[5] = 10.0

        trigger_index = detector.feed(block, absolute_start_index=1000)

        self.assertEqual(trigger_index, 1005)
        self.assertEqual(detector.window_start_index(), 1005 - 50)  # 5ms @ 10kHz = 50 samples

    def test_not_ready_until_enough_post_trigger_samples_exist(self) -> None:
        detector = ImpactDetector(_make_config(pretrigger_ms=5.0), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        _trigger_at(detector, 1005)
        needed_until = 1005 + (_BLOCK_SIZE - 50)

        self.assertFalse(detector.is_ready_to_capture(needed_until - 1))
        self.assertTrue(detector.is_ready_to_capture(needed_until))

    def test_no_pending_window_reports_none_and_not_ready(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)

        self.assertIsNone(detector.window_start_index())
        self.assertFalse(detector.is_ready_to_capture(1_000_000))


class RefractoryPeriodTest(unittest.TestCase):
    """`finish_capture()`/`is_in_refractory()` - the rest period that
    stops a strike's own ringdown from being mistaken for a new one."""

    def test_finish_capture_without_a_pending_trigger_raises(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)

        with self.assertRaises(ValueError):
            detector.finish_capture()

    def test_refractory_window_matches_min_rest_time_after_capture(self) -> None:
        detector = ImpactDetector(
            _make_config(min_rest_time_ms=10.0), _SAMPLE_RATE_HZ, _BLOCK_SIZE
        )
        _trigger_at(detector, 0)
        detector.finish_capture()
        # capture_end = 0 + (1000 - 50) = 950; rest = 10ms @ 10kHz = 100
        # samples -> refractory_until = 1050.

        self.assertTrue(detector.is_in_refractory(950))
        self.assertTrue(detector.is_in_refractory(1049))
        self.assertFalse(detector.is_in_refractory(1050))

    def test_no_trigger_fires_while_refractory_even_above_threshold(self) -> None:
        detector = ImpactDetector(
            _make_config(min_rest_time_ms=10.0), _SAMPLE_RATE_HZ, _BLOCK_SIZE
        )
        _trigger_at(detector, 0)
        detector.finish_capture()

        still_high = np.full(10, 10.0)
        self.assertIsNone(detector.feed(still_high, absolute_start_index=950))

    def test_detector_re_arms_and_fires_again_once_rest_time_has_passed(self) -> None:
        detector = ImpactDetector(
            _make_config(min_rest_time_ms=10.0), _SAMPLE_RATE_HZ, _BLOCK_SIZE
        )
        _trigger_at(detector, 0)
        detector.finish_capture()  # refractory_until = 1050

        block = np.zeros(20)
        block[15] = 10.0
        trigger_index = detector.feed(block, absolute_start_index=1050)

        self.assertEqual(trigger_index, 1065)

    def test_a_rejected_capture_still_starts_the_refractory_period(self) -> None:
        """finish_capture() must be called - and its refractory period
        respected - for a REJECTED strike too (double hit/overload):
        its ringdown must still be waited out, exactly like an accepted
        strike's (see the module docstring)."""
        detector = ImpactDetector(
            _make_config(min_rest_time_ms=10.0), _SAMPLE_RATE_HZ, _BLOCK_SIZE
        )
        _trigger_at(detector, 0)
        detector.finish_capture()

        self.assertTrue(detector.is_in_refractory(1000))

    def test_reset_clears_pending_and_refractory_state(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        _trigger_at(detector, 0)

        detector.reset()

        self.assertIsNone(detector.window_start_index())
        # reset() puts the detector back into a fresh arm (see
        # __init__'s None-seed rationale), so re-triggering it again
        # afterwards needs the same priming as the very first arm.
        self.assertEqual(_trigger_at(detector, 100), 100)


class DoubleHitDetectionTest(unittest.TestCase):
    def test_secondary_peak_above_relative_threshold_is_a_double_hit(self) -> None:
        detector = ImpactDetector(
            _make_config(double_hit_window_ms=5.0, double_hit_relative_threshold=0.5),
            _SAMPLE_RATE_HZ,
            _BLOCK_SIZE,
        )
        window = np.zeros(500)
        window[50] = 10.0  # main peak
        window[70] = 6.0  # 20 samples later (< 50-sample window), 60% of the main peak

        self.assertIs(detector.check_double_hit(window), True)

    def test_secondary_peak_below_relative_threshold_is_not_a_double_hit(self) -> None:
        detector = ImpactDetector(
            _make_config(double_hit_window_ms=5.0, double_hit_relative_threshold=0.5),
            _SAMPLE_RATE_HZ,
            _BLOCK_SIZE,
        )
        window = np.zeros(500)
        window[50] = 10.0
        window[70] = 2.0  # only 20% of the main peak

        self.assertIs(detector.check_double_hit(window), False)

    def test_secondary_peak_outside_the_double_hit_window_is_ignored(self) -> None:
        detector = ImpactDetector(
            _make_config(double_hit_window_ms=5.0, double_hit_relative_threshold=0.5),
            _SAMPLE_RATE_HZ,
            _BLOCK_SIZE,
        )
        window = np.zeros(500)
        window[50] = 10.0
        window[200] = 8.0  # far outside the 50-sample double-hit window

        self.assertIs(detector.check_double_hit(window), False)

    def test_no_secondary_peak_at_all_is_not_a_double_hit(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        window = np.zeros(500)
        window[50] = 10.0

        self.assertIs(detector.check_double_hit(window), False)

    def test_result_is_a_real_python_bool_not_a_numpy_bool(self) -> None:
        """`numpy.bool_(True) is True` is False - a caller comparing the
        result with `is True`/`is False` (or serializing it) would
        silently misbehave if this ever regressed."""
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)
        window = np.zeros(500)
        window[50] = 10.0

        self.assertIsInstance(detector.check_double_hit(window), bool)


class OverloadDetectionTest(unittest.TestCase):
    def test_excitation_over_its_fraction_of_range_is_an_overload(self) -> None:
        detector = ImpactDetector(_make_config(overload_fraction=0.9), _SAMPLE_RATE_HZ, _BLOCK_SIZE)

        self.assertIs(
            detector.check_overload(
                np.array([1.0, 2.0, 9.5]), np.array([1.0, 2.0]),
                excitation_range=10.0, response_range=50.0,
            ),
            True,
        )

    def test_response_over_its_fraction_of_range_is_an_overload(self) -> None:
        detector = ImpactDetector(_make_config(overload_fraction=0.9), _SAMPLE_RATE_HZ, _BLOCK_SIZE)

        self.assertIs(
            detector.check_overload(
                np.array([1.0, 8.0]), np.array([1.0, 48.0]),
                excitation_range=10.0, response_range=50.0,
            ),
            True,
        )

    def test_both_channels_within_range_is_not_an_overload(self) -> None:
        detector = ImpactDetector(_make_config(overload_fraction=0.9), _SAMPLE_RATE_HZ, _BLOCK_SIZE)

        self.assertIs(
            detector.check_overload(
                np.array([1.0, 8.0]), np.array([1.0, 40.0]),
                excitation_range=10.0, response_range=50.0,
            ),
            False,
        )

    def test_result_is_a_real_python_bool_not_a_numpy_bool(self) -> None:
        detector = ImpactDetector(_make_config(), _SAMPLE_RATE_HZ, _BLOCK_SIZE)

        result = detector.check_overload(
            np.array([1.0]), np.array([1.0]), excitation_range=10.0, response_range=50.0
        )
        self.assertIsInstance(result, bool)


class ConstructorValidationTest(unittest.TestCase):
    def test_pretrigger_longer_than_the_block_raises(self) -> None:
        with self.assertRaises(ValueError):
            ImpactDetector(_make_config(pretrigger_ms=1000.0), _SAMPLE_RATE_HZ, _BLOCK_SIZE)


if __name__ == "__main__":
    unittest.main()
