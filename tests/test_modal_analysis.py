"""
tests/test_modal_analysis.py

Tests for `analysis/modal_analysis.py` - the signal processing behind
the experimental modal analysis mode (H1/H2 frequency response
function, coherence, impact-testing windows).

Deliberately does NOT need a `QApplication`/Qt: the module is pure
numpy/scipy, fed raw time blocks directly - see the module docstring
for why (mirrors `tests/test_naming.py`/`tests/test_rate_groups.py`).

The core correctness check (`FrfAndCoherenceTest`) is built around an
EXACT closed-form case rather than an approximate physical simulation:
feeding a unit impulse as the excitation makes its spectrum exactly
1+0j at every bin (the DFT of a unit impulse is flat), so the response
spectrum alone - synthesized directly from a chosen "true" analytic
frequency response function via `numpy.fft.irfft` - passes through
`ModalAverager` completely unchanged after a single impact. That lets
the estimator's math be checked against an exact reference (down to
floating-point precision) instead of a noisy approximation, without
needing to simulate an actual impulse response.

An ODD block size is used throughout for this: `numpy.fft.irfft`
requires a real-valued Nyquist bin for a signal of EVEN length (a
mathematical requirement of any real-valued FFT of even length, not a
`ModalAverager` concern) - an arbitrary synthetic "true" FRF generally
has a complex value there, which `irfft` would silently alter. An odd
length simply has no such bin.
"""

from __future__ import annotations

import unittest

import numpy as np

from analysis.modal_analysis import (
    ModalAverager,
    exponential_window,
    force_window,
    get_window,
    resolve_block_size,
)


class ResolveBlockSizeTest(unittest.TestCase):
    def test_matches_fs_over_df(self) -> None:
        self.assertEqual(resolve_block_size(51200.0, 100.0), 512)
        self.assertEqual(resolve_block_size(1000.0, 1.0), 1000)

    def test_clamped_to_at_least_two_samples(self) -> None:
        self.assertGreaterEqual(resolve_block_size(1000.0, 1_000_000.0), 2)


class WindowFunctionTest(unittest.TestCase):
    def test_force_window_starts_at_plateau_and_decays_to_zero(self) -> None:
        window = force_window(1000)
        self.assertEqual(window[0], 1.0)
        self.assertEqual(window[-1], 0.0)
        # Non-increasing once past the plateau (allow float noise).
        tail = window[100:]
        self.assertTrue(np.all(np.diff(tail) <= 1e-9))

    def test_exponential_window_decays_from_one_to_decay_to(self) -> None:
        window = exponential_window(1000, decay_to=0.01)
        self.assertEqual(window[0], 1.0)
        self.assertAlmostEqual(window[-1], 0.01, places=9)

    def test_get_window_dispatches_known_names(self) -> None:
        self.assertTrue(np.allclose(get_window("rectangular", 100), np.ones(100)))
        self.assertEqual(len(get_window("hann", 100)), 100)
        self.assertEqual(len(get_window("force", 100)), 100)
        self.assertEqual(len(get_window("exponential", 100)), 100)

    def test_get_window_raises_for_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            get_window("nonsense", 10)


def _sdof_accelerance(omega: np.ndarray, fn_hz: float, zeta: float) -> np.ndarray:
    """Closed-form accelerance (response/force = acceleration/force) of
    a single-degree-of-freedom mass-spring-damper system - the standard
    textbook reference FRF used to validate a modal estimator.

    H(omega) = -omega^2 / (k - m*omega^2 + i*c*omega), expressed via the
    natural frequency and damping ratio rather than raw mass/stiffness/
    damping so the test reads in the same terms an operator would use.
    """
    omega_n = 2 * np.pi * fn_hz
    stiffness = 1.0e6  # arbitrary absolute scale - only shape/peak location matter here
    mass = stiffness / omega_n**2
    damping = 2 * zeta * np.sqrt(stiffness * mass)
    return -(omega**2) / (stiffness - mass * omega**2 + 1j * damping * omega)


class FrfAndCoherenceTest(unittest.TestCase):
    """`ModalAverager` against the exact closed-form case described in
    the module docstring above."""

    def setUp(self) -> None:
        self.sample_rate_hz = 10_000.0
        self.block_size = 2049  # odd - see module docstring
        self.frequency_hz = np.fft.rfftfreq(self.block_size, d=1.0 / self.sample_rate_hz)
        self.omega = 2 * np.pi * self.frequency_hz
        self.natural_frequency_hz = 500.0
        self.h_true = _sdof_accelerance(self.omega, self.natural_frequency_hz, zeta=0.02)

        self.excitation = np.zeros(self.block_size)
        self.excitation[0] = 1.0  # unit impulse: rfft is exactly 1+0j at every bin
        self.response = np.fft.irfft(self.h_true, n=self.block_size)

    def _rectangular_averager(self, estimator: str = "h1") -> ModalAverager:
        return ModalAverager(
            block_size=self.block_size,
            sample_rate_hz=self.sample_rate_hz,
            excitation_window="rectangular",
            response_window="rectangular",
            estimator=estimator,
        )

    def test_h1_recovers_the_exact_frf_from_one_noiseless_impact(self) -> None:
        averager = self._rectangular_averager()
        self.assertEqual(averager.num_averages, 0)

        averager.add_impact(self.excitation, self.response)

        self.assertEqual(averager.num_averages, 1)
        np.testing.assert_allclose(averager.frf("accelerance"), self.h_true, atol=1e-8)

    def test_coherence_is_exactly_one_after_a_single_impact(self) -> None:
        """A mathematically correct property of the formula (one data
        point cannot show scatter), not a bug - see the module
        docstring of `ModalAverager.coherence`."""
        averager = self._rectangular_averager()
        averager.add_impact(self.excitation, self.response)

        np.testing.assert_allclose(averager.coherence(), 1.0, atol=1e-6)

    def test_frf_peak_lands_on_the_configured_natural_frequency(self) -> None:
        averager = self._rectangular_averager()
        averager.add_impact(self.excitation, self.response)

        peak_index = np.argmax(np.abs(averager.frf()))
        self.assertLess(abs(self.frequency_hz[peak_index] - self.natural_frequency_hz), 5.0)

    def test_receptance_matches_hand_computed_conversion_and_guards_dc(self) -> None:
        averager = self._rectangular_averager()
        averager.add_impact(self.excitation, self.response)

        receptance = averager.frf("receptance")

        self.assertTrue(np.isnan(receptance[0].real))
        self.assertTrue(np.isnan(receptance[0].imag))
        expected = self.h_true[1:] / (1j * self.omega[1:]) ** 2
        np.testing.assert_allclose(receptance[1:], expected, atol=1e-12)

    def test_mobility_matches_hand_computed_conversion_and_guards_dc(self) -> None:
        averager = self._rectangular_averager()
        averager.add_impact(self.excitation, self.response)

        mobility = averager.frf("mobility")

        self.assertTrue(np.isnan(mobility[0].real))
        expected = self.h_true[1:] / (1j * self.omega[1:])
        np.testing.assert_allclose(mobility[1:], expected, atol=1e-10)

    def test_unknown_frf_quantity_raises(self) -> None:
        averager = self._rectangular_averager()
        averager.add_impact(self.excitation, self.response)

        with self.assertRaises(ValueError):
            averager.frf("nonsense")

    def test_h2_is_more_robust_than_h1_to_noise_on_the_excitation(self) -> None:
        """The textbook property that motivates offering both
        estimators: H2 is preferred when the excitation (input) is the
        noisier channel, H1 when the response (output) is."""
        rng = np.random.default_rng(1)
        h1 = self._rectangular_averager("h1")
        h2 = self._rectangular_averager("h2")
        for _ in range(20):
            noisy_excitation = self.excitation + rng.normal(scale=0.05, size=self.block_size)
            h1.add_impact(noisy_excitation, self.response)
            h2.add_impact(noisy_excitation, self.response)

        error_h1 = np.mean(np.abs(h1.frf() - self.h_true))
        error_h2 = np.mean(np.abs(h2.frf() - self.h_true))
        self.assertLess(error_h2, error_h1)


class ModalAveragerStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.block_size = 256
        self.sample_rate_hz = 5000.0
        self.excitation = np.zeros(self.block_size)
        self.excitation[0] = 1.0
        self.response = np.fft.irfft(
            _sdof_accelerance(
                2 * np.pi * np.fft.rfftfreq(self.block_size, d=1.0 / self.sample_rate_hz),
                fn_hz=800.0,
                zeta=0.03,
            ),
            n=self.block_size,
        )

    def _averager(self) -> ModalAverager:
        return ModalAverager(
            block_size=self.block_size,
            sample_rate_hz=self.sample_rate_hz,
            excitation_window="rectangular",
            response_window="rectangular",
        )

    def test_undo_last_is_an_exact_inverse_of_add_impact(self) -> None:
        averager = self._averager()
        averager.add_impact(self.excitation, self.response)
        frf_after_one = averager.frf().copy()
        coherence_after_one = averager.coherence().copy()

        rng = np.random.default_rng(0)
        averager.add_impact(self.excitation, self.response + rng.normal(scale=0.01, size=self.block_size))
        self.assertEqual(averager.num_averages, 2)

        averager.undo_last()

        self.assertEqual(averager.num_averages, 1)
        np.testing.assert_allclose(averager.frf(), frf_after_one)
        np.testing.assert_allclose(averager.coherence(), coherence_after_one)

    def test_reset_clears_all_accepted_impacts(self) -> None:
        averager = self._averager()
        averager.add_impact(self.excitation, self.response)
        averager.add_impact(self.excitation, self.response)

        averager.reset()

        self.assertEqual(averager.num_averages, 0)
        with self.assertRaises(ValueError):
            averager.frf()

    def test_frf_and_coherence_raise_before_any_impact(self) -> None:
        averager = self._averager()
        with self.assertRaises(ValueError):
            averager.frf()
        with self.assertRaises(ValueError):
            averager.coherence()

    def test_undo_last_raises_when_nothing_has_been_added(self) -> None:
        averager = self._averager()
        with self.assertRaises(ValueError):
            averager.undo_last()

    def test_add_impact_rejects_wrong_block_size(self) -> None:
        averager = self._averager()
        with self.assertRaises(ValueError):
            averager.add_impact(np.zeros(10), np.zeros(10))

    def test_constructor_rejects_unknown_estimator(self) -> None:
        with self.assertRaises(ValueError):
            ModalAverager(block_size=64, sample_rate_hz=1000.0, estimator="nonsense")

    def test_constructor_rejects_too_small_block_size(self) -> None:
        with self.assertRaises(ValueError):
            ModalAverager(block_size=1, sample_rate_hz=1000.0)


if __name__ == "__main__":
    unittest.main()
