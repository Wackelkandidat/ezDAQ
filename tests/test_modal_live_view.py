"""
tests/test_modal_live_view.py

Tests for `gui/modal_live_view.py::ModalLiveView` - the impact-triggered
capture pipeline (edge detection -> pretrigger/posttrigger window
extraction -> double-hit/overload check -> averaging -> plots/hit
table), driven by a fake ring buffer rather than real hardware or the
real `QTimer`/event loop.

`_FakeRingBuffer` stands in for the parts of `MeasurementController`
that talk to the real `core.ringbuffer.RingBuffer`
(`register_reader`/`unregister_reader`/`read_live_data`/
`total_samples_acquired`) - `ModalLiveView` only ever calls these, so
faking them here is enough to exercise the full pipeline without a
running acquisition thread. `_on_timer_tick()` is invoked directly
(never via the real `QTimer`) so each simulated "tick" is deterministic
and the test does not depend on real wall-clock timing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

from config.configuration_manager import ConfigurationManager
from core.controller import MeasurementController
from data.models import (
    Channel,
    ModalAnalysisConfig,
    ModalAxis,
    ModalResponseChannel,
    ModuleType,
    SignalType,
    StorageFormat,
    TriggerCondition,
    TriggerDirection,
)
from gui.modal_live_view import ModalLiveView

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _make_controller() -> tuple[MeasurementController, tempfile.TemporaryDirectory]:
    """A `MeasurementController` backed by a throwaway settings
    directory - `ConfigurationManager()` without an explicit
    `config_dir` would otherwise read/write the REAL user's
    `%APPDATA%/ezDAQ` settings. Callers must `cleanup()` the returned
    `TemporaryDirectory` once done (see each test's `tearDown`/
    `finally`)."""
    tmp_dir = tempfile.TemporaryDirectory()
    controller = MeasurementController(ConfigurationManager(Path(tmp_dir.name)))
    return controller, tmp_dir


_SAMPLE_RATE_HZ = 10_000.0
_BLOCK_SIZE = 2048
_TICK_SAMPLES = 150  # ~15ms at 10 kHz, matches _UI_UPDATE_INTERVAL_MS


def _make_channels() -> list[Channel]:
    return [
        Channel(
            hardware_channel="cDAQ1Mod1/ai0",
            display_name="Anregung (Hammer)",
            unit="N",
            signal_type=SignalType.IEPE_ACCELERATION,
            module_type=ModuleType.NI9234,
            sensitivity_mv_per_unit=11.2,
            min_range=None,
            max_range=100.0,  # generous - keeps the small synthetic amplitudes below overload
        ),
        Channel(
            hardware_channel="cDAQ1Mod1/ai1",
            display_name="Antwort X",
            unit="g",
            signal_type=SignalType.IEPE_ACCELERATION,
            module_type=ModuleType.NI9234,
            sensitivity_mv_per_unit=10.5,
            min_range=None,
            max_range=100.0,
        ),
    ]


def _make_modal_config(**overrides) -> ModalAnalysisConfig:
    defaults = dict(
        excitation_channel_hardware_id="cDAQ1Mod1/ai0",
        response_channels=[ModalResponseChannel(ModalAxis.X, "cDAQ1Mod1/ai1")],
        excitation_window="rectangular",
        response_window="rectangular",
        frequency_resolution_hz=_SAMPLE_RATE_HZ / _BLOCK_SIZE,  # -> exactly _BLOCK_SIZE
        num_averages_target=3,
        impact_condition=TriggerCondition(
            threshold_value=1.0, threshold_direction=TriggerDirection.RISES_ABOVE
        ),
        pretrigger_ms=10.0,
        min_rest_time_ms=50.0,
        double_hit_window_ms=5.0,
        double_hit_relative_threshold=0.5,
    )
    defaults.update(overrides)
    return ModalAnalysisConfig(**defaults)


class _FakeRingBuffer:
    """Stands in for `core.ringbuffer.RingBuffer`, driven through the
    exact same four calls `ModalLiveView` makes on
    `MeasurementController` - see the module docstring."""

    def __init__(self, num_samples: int = 20_000, num_channels: int = 2) -> None:
        self.total_written = 0
        self.data = np.zeros((num_channels, num_samples))
        self._next_reader_id = 0
        self._readers: dict[int, int] = {}

    def register_reader(self, back_samples: int = 0) -> int:
        reader_id = self._next_reader_id
        self._next_reader_id += 1
        self._readers[reader_id] = max(0, self.total_written - back_samples)
        return reader_id

    def unregister_reader(self, reader_id: int) -> None:
        self._readers.pop(reader_id, None)

    def read_live_data(self, reader_id: int, max_samples: int | None = None) -> np.ndarray:
        start = self._readers[reader_id]
        end = min(self.total_written, start + (max_samples or 10**9))
        block = self.data[:, start:end]
        self._readers[reader_id] = end
        return block

    def advance(self, num_samples: int) -> None:
        self.total_written = min(self.data.shape[1], self.total_written + num_samples)


def _attach_fake_ring_buffer(controller: MeasurementController, fake: _FakeRingBuffer, channels: list[Channel]) -> None:
    controller.register_reader = fake.register_reader
    controller.unregister_reader = fake.unregister_reader
    controller.read_live_data = fake.read_live_data
    type(controller).active_channels = property(lambda self: channels)
    type(controller).total_samples_acquired = property(lambda self: fake.total_written)


def _run_ticks(fake: _FakeRingBuffer, view: ModalLiveView, until, max_ticks: int = 400) -> None:
    """Advances the fake ring buffer in fixed-size ticks and calls
    `_on_timer_tick()` directly until `until(view)` is truthy or
    `max_ticks` is reached (whichever first) - the deterministic
    stand-in for letting the real QTimer run."""
    for _ in range(max_ticks):
        fake.advance(_TICK_SAMPLES)
        view._on_timer_tick()
        if until(view):
            return


def _click_button_with_role(role) -> None:
    """Monkeypatches `QMessageBox.exec` for the duration of the CALLER's
    control (module-level patch, restored by the caller) to
    synchronously click whichever button has the given role - Qt's
    `QPushButton.click()` triggers `QMessageBox`'s own real internal
    `clickedButton()`/`done()` wiring, so this exercises the actual
    dialog machinery rather than re-implementing it."""

    def fake_exec(self):
        for button in self.buttons():
            if self.buttonRole(button) == role:
                button.click()
                break
        return self.result()

    QMessageBox.exec = fake_exec


class CaptureTest(unittest.TestCase):
    """The end-to-end pipeline: a single clean impact all the way
    through to an averaged FRF and an updated hit table row."""

    def setUp(self) -> None:
        _app()
        self._original_message_box_exec = QMessageBox.exec
        self._channels = _make_channels()
        self._controller, self._tmp_dir = _make_controller()
        self._fake = _FakeRingBuffer()
        _attach_fake_ring_buffer(self._controller, self._fake, self._channels)

        # A quiet excitation with one impulse, and a decaying sinusoid
        # response starting at the same instant - not physically exact
        # (no attempt to match a real transducer), just enough shape to
        # exercise windowing/FFT without being all zeros.
        impulse_index = 5000
        self._fake.data[0, impulse_index] = 3.0
        t = np.arange(self._fake.data.shape[1]) / _SAMPLE_RATE_HZ
        decay = np.where(t >= impulse_index / _SAMPLE_RATE_HZ, np.exp(-(t - impulse_index / _SAMPLE_RATE_HZ) * 50.0), 0.0)
        self._fake.data[1] = 2.0 * decay * np.sin(2 * np.pi * 440.0 * (t - impulse_index / _SAMPLE_RATE_HZ))

        self._view = ModalLiveView(self._controller)

    def tearDown(self) -> None:
        QMessageBox.exec = self._original_message_box_exec
        self._view.close()
        self._view.deleteLater()
        _app().processEvents()
        self._tmp_dir.cleanup()

    def test_clean_impact_is_captured_and_averaged(self) -> None:
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")
        self.assertEqual(self._view._block_size, _BLOCK_SIZE)

        _run_ticks(self._fake, self._view, until=lambda v: v._averagers["x"].num_averages >= 1)

        self.assertEqual(self._view._averagers["x"].num_averages, 1)
        self.assertEqual(self._view._row_kinds, ["captured"])

    def test_excitation_plot_is_populated_with_the_full_block(self) -> None:
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")

        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)

        x_data, _y_data = self._view._excitation_curve.getData()
        self.assertEqual(len(x_data), _BLOCK_SIZE)

    def test_frf_and_coherence_plots_are_populated_after_a_capture(self) -> None:
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")

        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)

        freq, _amplitude = self._view._amplitude_curve.getData()
        self.assertGreater(len(freq), 0)
        freq_coh, _coherence = self._view._coherence_curve.getData()
        self.assertGreater(len(freq_coh), 0)

    def test_undo_reverts_the_capture_and_the_row(self) -> None:
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")
        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)

        self._view._on_undo_clicked()

        self.assertEqual(self._view._averagers["x"].num_averages, 0)
        self.assertEqual(self._view._row_kinds, [])

    def test_reset_clears_averages_and_rebuilds_the_table(self) -> None:
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")
        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)

        self._view._on_reset_clicked()

        self.assertEqual(self._view._averagers["x"].num_averages, 0)
        self.assertEqual(self._view._row_kinds, [])
        self.assertEqual(self._view._hit_table.rowCount(), 3)

    def test_detection_pauses_once_every_table_row_is_filled(self) -> None:
        """See `ModalLiveView._is_fully_captured` - once the fixed-size
        hit table is full, further ticks must not silently keep
        averaging strikes with nowhere to show them."""
        config = _make_modal_config(num_averages_target=1)
        self._view.start_display(config, _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")
        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)
        self.assertEqual(self._view._averagers["x"].num_averages, 1)

        # Feed a SECOND impulse further into the (still-quiet-otherwise)
        # recording - with the table already full, it must be ignored.
        second_impulse_index = self._fake.total_written + 2000
        self._fake.data[0, second_impulse_index] = 3.0
        _run_ticks(self._fake, self._view, until=lambda v: False, max_ticks=100)

        self.assertEqual(self._view._averagers["x"].num_averages, 1)


class OverloadRejectionTest(unittest.TestCase):
    def test_excitation_over_its_range_is_rejected_without_averaging(self) -> None:
        _app()
        channels = _make_channels()
        controller, tmp_dir = _make_controller()
        fake = _FakeRingBuffer()
        _attach_fake_ring_buffer(controller, fake, channels)
        # 150 N against the channel's configured ±100 range - well over.
        fake.data[0, 5000] = 150.0

        view = ModalLiveView(controller)
        try:
            view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")
            _run_ticks(fake, view, until=lambda v: v._status_label.text() != "" or v._row_kinds)

            self.assertEqual(view._averagers["x"].num_averages, 0)
            self.assertEqual(view._row_kinds, [])
        finally:
            view.close()
            view.deleteLater()
            _app().processEvents()
            tmp_dir.cleanup()


class DoubleHitDialogTest(unittest.TestCase):
    """The double-hit dialog's two real outcomes - see
    `_click_button_with_role`, which exercises Qt's actual
    `QMessageBox` button-click wiring rather than simulating it."""

    def setUp(self) -> None:
        _app()
        self._original_message_box_exec = QMessageBox.exec
        self._channels = _make_channels()
        self._controller, self._tmp_dir = _make_controller()
        self._fake = _FakeRingBuffer()
        _attach_fake_ring_buffer(self._controller, self._fake, self._channels)
        # Main peak plus a secondary one well within the double-hit
        # window and above the relative threshold.
        self._fake.data[0, 3000] = 3.0
        self._fake.data[0, 3020] = 2.0
        self._view = ModalLiveView(self._controller)

    def tearDown(self) -> None:
        QMessageBox.exec = self._original_message_box_exec
        self._view.close()
        self._view.deleteLater()
        _app().processEvents()
        self._tmp_dir.cleanup()

    def test_skip_marks_the_row_skipped_without_averaging(self) -> None:
        _click_button_with_role(QMessageBox.ButtonRole.DestructiveRole)
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")

        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)

        self.assertEqual(self._view._row_kinds, ["skipped"])
        self.assertEqual(self._view._averagers["x"].num_averages, 0)

    def test_retry_leaves_the_slot_pending(self) -> None:
        _click_button_with_role(QMessageBox.ButtonRole.AcceptRole)
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")

        # "Retry" records nothing - run a fixed number of ticks rather
        # than waiting for a state that, by design, never changes.
        _run_ticks(self._fake, self._view, until=lambda v: False, max_ticks=200)

        self.assertEqual(self._view._row_kinds, [])
        self.assertEqual(self._view._averagers["x"].num_averages, 0)

    def test_a_rejected_double_hit_still_starts_the_refractory_period(self) -> None:
        """Mirrors `tests/test_impact_detector.py`'s equivalent check -
        `finish_capture()` must run for a REJECTED strike too, so its
        ringdown does not fake a second, immediate trigger."""
        _click_button_with_role(QMessageBox.ButtonRole.DestructiveRole)
        self._view.start_display(_make_modal_config(min_rest_time_ms=500.0), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")

        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)

        self.assertEqual(self._view._detector._state, "refractory")


class MultiAxisTest(unittest.TestCase):
    """Triaxial response: one excitation feeds an independent
    `ModalAverager` per configured axis - see
    `data/models.py::ModalAnalysisConfig`'s docstring on why
    `ImpactDetector` itself stays axis-count-agnostic."""

    def _make_triaxial_channels(self) -> list[Channel]:
        channels = _make_channels()
        for axis_letter, hw_id in (("Y", "cDAQ1Mod1/ai2"), ("Z", "cDAQ1Mod1/ai3")):
            channels.append(
                Channel(
                    hardware_channel=hw_id,
                    display_name=f"Antwort {axis_letter}",
                    unit="g",
                    signal_type=SignalType.IEPE_ACCELERATION,
                    module_type=ModuleType.NI9234,
                    sensitivity_mv_per_unit=10.0,
                    min_range=None,
                    max_range=100.0,
                )
            )
        return channels

    def test_one_impact_feeds_every_configured_axis(self) -> None:
        _app()
        channels = self._make_triaxial_channels()
        controller, tmp_dir = _make_controller()
        fake = _FakeRingBuffer(num_channels=4)
        _attach_fake_ring_buffer(controller, fake, channels)
        fake.data[0, 5000] = 3.0
        fake.data[1, 5000:5100] = 1.0
        fake.data[2, 5000:5100] = 2.0
        fake.data[3, 5000:5100] = 3.0

        view = ModalLiveView(controller)
        try:
            config = _make_modal_config(
                response_channels=[
                    ModalResponseChannel(ModalAxis.X, "cDAQ1Mod1/ai1"),
                    ModalResponseChannel(ModalAxis.Y, "cDAQ1Mod1/ai2"),
                    ModalResponseChannel(ModalAxis.Z, "cDAQ1Mod1/ai3"),
                ]
            )
            view.start_display(config, _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")
            self.assertTrue(view._axis_combo.isEnabled())

            _run_ticks(fake, view, until=lambda v: v._row_kinds)

            for axis in ("x", "y", "z"):
                self.assertEqual(view._averagers[axis].num_averages, 1)
        finally:
            view.close()
            view.deleteLater()
            _app().processEvents()
            tmp_dir.cleanup()

    def test_single_axis_disables_the_axis_selector(self) -> None:
        _app()
        channels = _make_channels()
        controller, tmp_dir = _make_controller()
        fake = _FakeRingBuffer()
        _attach_fake_ring_buffer(controller, fake, channels)

        view = ModalLiveView(controller)
        try:
            view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")
            self.assertFalse(view._axis_combo.isEnabled())
        finally:
            view.close()
            view.deleteLater()
            _app().processEvents()
            tmp_dir.cleanup()


class RetranslateTest(unittest.TestCase):
    def test_retranslate_ui_does_not_raise_before_or_after_start_display(self) -> None:
        _app()
        controller, tmp_dir = _make_controller()
        view = ModalLiveView(controller)
        try:
            view.retranslate_ui()  # before any measurement - must not assume state exists

            channels = _make_channels()
            fake = _FakeRingBuffer()
            _attach_fake_ring_buffer(controller, fake, channels)
            view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, Path("unused"), "TestMessung")

            view.retranslate_ui()
        finally:
            view.close()
            view.deleteLater()
            _app().processEvents()
            tmp_dir.cleanup()


class ExportResultsTest(unittest.TestCase):
    """`_on_export_clicked` - the FRF/coherence sidecar written next to
    the raw recording, additive and independent of it (see the method's
    own docstring)."""

    def setUp(self) -> None:
        _app()
        self._controller, self._config_tmp_dir = _make_controller()
        self._export_tmp_dir = tempfile.TemporaryDirectory()
        self._channels = _make_channels()
        self._fake = _FakeRingBuffer()
        _attach_fake_ring_buffer(self._controller, self._fake, self._channels)
        self._fake.data[0, 5000] = 3.0
        self._fake.data[1, 5000:5100] = 1.0
        self._view = ModalLiveView(self._controller)

    def tearDown(self) -> None:
        self._view.close()
        self._view.deleteLater()
        _app().processEvents()
        self._config_tmp_dir.cleanup()
        self._export_tmp_dir.cleanup()

    def test_export_without_any_average_shows_an_error_and_writes_nothing(self) -> None:
        storage_path = Path(self._export_tmp_dir.name)
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, storage_path, "Test")
        errors: list[str] = []
        self._view._show_error = errors.append

        self._view._on_export_clicked()

        self.assertEqual(len(errors), 1)
        self.assertEqual(list(storage_path.iterdir()), [])

    def test_export_writes_one_frf_file_per_axis_with_data_plus_metadata(self) -> None:
        storage_path = Path(self._export_tmp_dir.name)
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, storage_path, "Test")
        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)
        self.assertEqual(self._view._averagers["x"].num_averages, 1)  # sanity check on the fixture

        self._view._on_export_clicked()

        frf_path = storage_path / "Test_frf_x.parquet"
        metadata_path = storage_path / "Test_modal_info.json"
        self.assertTrue(frf_path.exists())
        self.assertTrue(metadata_path.exists())

        import json

        import pandas as pd

        frame = pd.read_parquet(frf_path)
        self.assertEqual(
            list(frame.columns),
            [
                "frequency_hz",
                "accelerance_real",
                "accelerance_imag",
                "receptance_real",
                "receptance_imag",
                "coherence",
            ],
        )
        self.assertTrue(frame["receptance_real"].iloc[0] != frame["receptance_real"].iloc[0])  # NaN

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["num_averages_actual"], {"x": 1})

    def test_export_respects_the_configured_result_storage_format(self) -> None:
        storage_path = Path(self._export_tmp_dir.name)
        self._view.start_display(
            _make_modal_config(result_storage_format=StorageFormat.CSV),
            _SAMPLE_RATE_HZ,
            storage_path,
            "Test",
        )
        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)

        self._view._on_export_clicked()

        self.assertTrue((storage_path / "Test_frf_x.csv").exists())
        self.assertFalse((storage_path / "Test_frf_x.parquet").exists())

    def test_export_failure_is_reported_rather_than_raised(self) -> None:
        # A storage path that does not exist -> writing fails, but the
        # method must not propagate the exception to the button click.
        storage_path = Path(self._export_tmp_dir.name) / "does_not_exist"
        self._view.start_display(_make_modal_config(), _SAMPLE_RATE_HZ, storage_path, "Test")
        _run_ticks(self._fake, self._view, until=lambda v: v._row_kinds)
        errors: list[str] = []
        self._view._show_error = errors.append

        self._view._on_export_clicked()

        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
