"""
analysis/modal_analysis.py

Signal processing for the experimental modal analysis mode (impact
hammer excitation + accelerometer response) - see
`data/models.py::ModalAnalysisConfig`, `core/impact_detector.py`, and
`gui/modal_live_view.py`.

Deliberately independent of the GUI and of pandas (unlike
`analysis/basic_analysis.py`, which works on `pandas.DataFrame`s of
already-recorded data): the impact detector feeds this module raw numpy
blocks straight from the ring buffer, one hammer strike at a time, so
this module stays a pure numpy/scipy layer and is individually testable
without a QApplication or any recorded file.

Currently implemented:
    * force_window(...)/exponential_window(...)/get_window(...): the
      time windows conventionally used for impact-hammer testing.
    * resolve_block_size(...): turns the user-facing "frequency
      resolution" into the FFT block size actually used.
    * ModalAverager: accumulates cross-/auto-spectra across accepted
      impacts and exposes the running H1/H2 frequency response function
      plus coherence, with an exact undo of the last impact.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows as scipy_windows


def resolve_block_size(sample_rate_hz: float, frequency_resolution_hz: float) -> int:
    """Converts the fachlich meaningful "desired frequency spacing" Δf
    into the FFT block size actually needed to achieve it (Δf = fs/N).

    Rounds to the nearest sample count rather than snapping to a
    power-of-two grid like `data/models.py`'s sample-rate grids do -
    unlike the acquisition rate, the FFT block size has no hardware
    constraint here; `numpy.fft.rfft` handles any length.

    Clamped to at least 2 samples, so a pathologically large requested
    resolution (or a caller passing 0) cannot produce a degenerate,
    unusable block.
    """
    return max(2, round(sample_rate_hz / frequency_resolution_hz))


def force_window(n: int, plateau_fraction: float = 0.1, taper_fraction: float = 0.05) -> np.ndarray:
    """The window conventionally used for the EXCITATION (hammer)
    channel in impact testing.

    A hammer strike itself lasts only a few samples at the very start
    of the captured block; everything after it is not signal but noise
    (a second, unintended bounce; electrical noise on an otherwise
    quiet channel). The force window keeps a short plateau of 1.0 at
    the start (`plateau_fraction` of the block - wide enough to hold
    the pulse regardless of its exact width) and then tapers smoothly
    to exactly 0.0 with a half-cosine ramp (`taper_fraction` of the
    block), so nothing beyond the pulse contributes to the FFT.

    This is NOT the same shape as `scipy.signal.windows.tukey`: a Tukey
    window ramps up AND down around a centered plateau - this window
    starts at its plateau (sample 0) and only ramps down once, which is
    what a pulse sitting at the start of the block needs.
    """
    window = np.zeros(n, dtype=float)
    plateau_end = int(round(n * plateau_fraction))
    plateau_end = min(max(plateau_end, 0), n)
    taper_end = int(round(n * (plateau_fraction + taper_fraction)))
    taper_end = min(max(taper_end, plateau_end + 1), n)

    window[:plateau_end] = 1.0
    taper_len = taper_end - plateau_end
    if taper_len > 0:
        ramp = np.linspace(0.0, np.pi, taper_len)
        window[plateau_end:taper_end] = (1.0 + np.cos(ramp)) / 2.0
    return window


def exponential_window(n: int, decay_to: float = 0.01) -> np.ndarray:
    """The window conventionally used for the RESPONSE (accelerometer)
    channel in impact testing.

    Unlike the excitation, the response signal is the actual
    measurement of interest for its whole length - a hard cutoff would
    discard real information. Instead, `exponential_window` decays
    smoothly from 1.0 at the start to `decay_to` at the last sample,
    damping a ringdown that has not yet fully decayed by the end of the
    block before the FFT wraps it around (leakage) - a standard remedy
    for lightly damped structures whose response outlasts the capture
    window.
    """
    decay_to = max(decay_to, 1e-6)  # guards log(0) for a degenerate decay_to=0
    tau = -(n - 1) / np.log(decay_to)
    return np.exp(-np.arange(n) / tau)


def get_window(name: str, n: int) -> np.ndarray:
    """Resolves a window name (as stored in
    `ModalAnalysisConfig.excitation_window`/`.response_window`) to the
    actual window array.

    Raises:
        ValueError: for an unknown window name.
    """
    if name == "force":
        return force_window(n)
    if name == "exponential":
        return exponential_window(n)
    if name == "rectangular":
        return scipy_windows.boxcar(n)
    if name == "hann":
        return scipy_windows.hann(n)
    raise ValueError(f"Unbekannter Fenstertyp: '{name}'.")


class ModalAverager:
    """Accumulates cross-/auto-spectra across accepted hammer strikes
    and exposes the running H1/H2 frequency response function plus
    coherence.

    One segment per impact (rather than the usual overlapping segments
    of one long continuous record): each hammer strike naturally
    supplies exactly one windowed excitation/response pair, so this is
    Welch-style averaging with a segment count of one impact each.

    Only the running SUMS of Gff/Gxx/Gxf are kept, not their averages -
    the factor 1/N cancels out of H1, H2 AND the coherence formula
    (each is a ratio of two quantities that both carry the same 1/N),
    so keeping sums instead of averages is both simpler and exact
    regardless of how many impacts have been added. `undo_last()`
    therefore only needs the single most recent impact's contribution
    to subtract it back out - kept in `_history` for exactly that.
    """

    def __init__(
        self,
        block_size: int,
        sample_rate_hz: float,
        excitation_window: str = "force",
        response_window: str = "exponential",
        estimator: str = "h1",
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size muss mindestens 2 betragen.")
        if estimator not in ("h1", "h2"):
            raise ValueError(f"Unbekannter Schätzer: '{estimator}' (erwartet 'h1' oder 'h2').")

        self._block_size = block_size
        self._sample_rate_hz = sample_rate_hz
        self._estimator = estimator
        self._excitation_window = get_window(excitation_window, block_size)
        self._response_window = get_window(response_window, block_size)

        num_bins = block_size // 2 + 1
        self._Gff_sum = np.zeros(num_bins, dtype=float)
        self._Gxx_sum = np.zeros(num_bins, dtype=float)
        self._Gxf_sum = np.zeros(num_bins, dtype=complex)
        # One (Gff_i, Gxx_i, Gxf_i) triple per accepted impact, in order
        # - only ever appended to/popped from the end, so `undo_last()`
        # is an exact O(1) inverse of the most recent `add_impact()`.
        self._history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_averages(self) -> int:
        """Number of impacts currently contributing to the average."""
        return len(self._history)

    def frequency_hz(self) -> np.ndarray:
        """Frequency axis matching `frf()`/`coherence()` - one entry per
        FFT bin, from 0 Hz to the Nyquist frequency."""
        return np.fft.rfftfreq(self._block_size, d=1.0 / self._sample_rate_hz)

    def add_impact(self, excitation: np.ndarray, response: np.ndarray) -> None:
        """Adds one accepted hammer strike to the running average.

        Args:
            excitation: Force-channel time block, exactly `block_size`
                samples, force channel first in the block (the impact
                near its start - see `core/impact_detector.py`).
            response: Response-channel time block, same length.

        Raises:
            ValueError: if either block is not exactly `block_size`
                samples long.
        """
        excitation = np.asarray(excitation, dtype=float)
        response = np.asarray(response, dtype=float)
        if excitation.shape != (self._block_size,) or response.shape != (self._block_size,):
            raise ValueError(
                f"Erwartet {self._block_size} Samples je Kanal, erhalten "
                f"{excitation.shape[0] if excitation.ndim else 'ungültig'}/"
                f"{response.shape[0] if response.ndim else 'ungültig'}."
            )

        force_spectrum = np.fft.rfft(excitation * self._excitation_window)
        response_spectrum = np.fft.rfft(response * self._response_window)

        gff = np.abs(force_spectrum) ** 2
        gxx = np.abs(response_spectrum) ** 2
        gxf = response_spectrum * np.conj(force_spectrum)

        self._history.append((gff, gxx, gxf))
        self._Gff_sum += gff
        self._Gxx_sum += gxx
        self._Gxf_sum += gxf

    def undo_last(self) -> None:
        """Removes the most recently added impact from the average -
        exact, not an approximation: the subtracted contribution is the
        very one `add_impact()` added.

        Raises:
            ValueError: if no impact has been added yet.
        """
        if not self._history:
            raise ValueError("Es wurde noch kein Schlag erfasst, der zurückgenommen werden könnte.")
        gff, gxx, gxf = self._history.pop()
        self._Gff_sum -= gff
        self._Gxx_sum -= gxx
        self._Gxf_sum -= gxf

    def reset(self) -> None:
        """Discards all accepted impacts, back to a fresh average."""
        self._history.clear()
        self._Gff_sum[:] = 0.0
        self._Gxx_sum[:] = 0.0
        self._Gxf_sum[:] = 0.0

    def frf(self, quantity: str = "accelerance") -> np.ndarray:
        """The running frequency response function estimate.

        Args:
            quantity: Which mechanical quantity to express the FRF in -
                "accelerance" (response/force, exactly what
                `add_impact()` measured, no further conversion),
                "mobility" (velocity/force, one frequency-domain
                integration) or "receptance" (displacement/force, two
                integrations) - see `data.models.ModalAnalysisConfig.
                frf_quantity`. All three describe the SAME underlying
                measurement; only the post-processing differs.

        Returns:
            Complex array, one value per bin of `frequency_hz()`. For
            "mobility"/"receptance", bin 0 (0 Hz) is `nan+nanj` - both
            conversions divide by `j*omega`, which is exactly zero at
            0 Hz (a receptance/mobility value at DC is undefined, not
            zero).

        Raises:
            ValueError: if no impact has been added yet, or `quantity`
                is not one of the three above.
        """
        if self.num_averages == 0:
            raise ValueError("Es wurde noch kein Schlag erfasst.")

        with np.errstate(divide="ignore", invalid="ignore"):
            if self._estimator == "h1":
                accelerance = self._Gxf_sum / self._Gff_sum
            else:  # "h2"
                accelerance = self._Gxx_sum / np.conj(self._Gxf_sum)

            if quantity == "accelerance":
                return accelerance
            omega = 2.0 * np.pi * self.frequency_hz()
            if quantity == "mobility":
                result = accelerance / (1j * omega)
            elif quantity == "receptance":
                result = accelerance / (1j * omega) ** 2
            else:
                raise ValueError(
                    f"Unbekannte FRF-Größe: '{quantity}' "
                    "(erwartet 'accelerance', 'mobility' oder 'receptance')."
                )
        # Explicit rather than relying on the division's own inf/nan
        # behavior at omega[0] == 0: correctness here must not depend on
        # numpy's (version-specific) floating-point edge-case handling.
        result[0] = complex(np.nan, np.nan)
        return result

    def coherence(self) -> np.ndarray:
        """The running (ordinary) coherence γ² between excitation and
        response, one value per bin of `frequency_hz()`.

        Always exactly 1.0 after the FIRST impact, for any signal -
        that is a correct property of the formula (a single data point
        cannot show scatter/incoherence), not a bug; coherence becomes
        meaningful only once several impacts have been averaged.

        Returns:
            Real array in [0, 1] (clipped - floating-point rounding can
            push the raw ratio fractionally above 1.0).

        Raises:
            ValueError: if no impact has been added yet.
        """
        if self.num_averages == 0:
            raise ValueError("Es wurde noch kein Schlag erfasst.")
        with np.errstate(divide="ignore", invalid="ignore"):
            coherence = np.abs(self._Gxf_sum) ** 2 / (self._Gff_sum * self._Gxx_sum)
        return np.clip(coherence, 0.0, 1.0)
