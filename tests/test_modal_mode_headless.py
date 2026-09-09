"""
tests/test_modal_mode_headless.py

Tests for the switchable application mode (Settings -> Mode, "Standard"
/ "Modalanalyse") introduced for the experimental modal analysis
feature - see `gui/main_window.py::MainWindow._resolve_workspace_index`/
`_on_mode_action_triggered`, `data/models.py::ModalAnalysisConfig`, and
`config/configuration_manager.py::update_app_mode`/
`update_last_modal_config`.

Covers three things that are easy to get wrong when a second pair of
pages is spliced into the same `QStackedWidget` as the standard
Setup/Live views:
    - The nav-tile-row -> workspace-page translation
      (`_resolve_workspace_index`) picks the right page for BOTH modes,
      and Analysis (shared by both modes) is unaffected by the mode.
    - The mode switch is refused while a measurement is running, and
      the menu's checked state reverts to reflect that refusal rather
      than silently drifting out of sync with `_app_mode`.
    - `ModalAnalysisConfig`/`AppSettings.app_mode` round-trip through
      persistence exactly like the existing `TriggerConfig`/
      `last_trigger_config` pattern they were modeled on.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from config.configuration_manager import ConfigurationManager
from core.controller import MeasurementController
from data.models import (
    ModalAnalysisConfig,
    ModalAxis,
    ModalResponseChannel,
    TriggerCondition,
    TriggerDirection,
)
from gui.main_window import (
    MainWindow,
    _VIEW_ANALYSIS,
    _VIEW_LIVE,
    _VIEW_MODAL_LIVE,
    _VIEW_MODAL_SETUP,
    _VIEW_SETUP,
)

# Module-level reference on purpose - see tests/test_axis_labels.py for
# why: without one, the QApplication created here is garbage-collected
# once this file's tests return, and Qt tears down every remaining
# QObject with it, including the lazily built i18n signal singleton.
_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _build_window(tmp_dir: str) -> tuple[MainWindow, MeasurementController, ConfigurationManager]:
    _app()
    config_manager = ConfigurationManager(Path(tmp_dir))
    controller = MeasurementController(config_manager)
    window = MainWindow(controller, config_manager)
    return window, controller, config_manager


class ModeSwitchWorkspaceResolutionTest(unittest.TestCase):
    """`_resolve_workspace_index` - the nav-row -> page translation."""

    def test_starts_in_standard_mode_on_the_setup_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            window, _controller, _config = _build_window(tmp_dir)
            try:
                self.assertEqual(window._app_mode, "standard")
                self.assertEqual(window._workspace.currentIndex(), _VIEW_SETUP)
            finally:
                window.close()
                window.deleteLater()
                _app().processEvents()

    def test_modal_mode_swaps_setup_and_live_but_not_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            window, _controller, _config = _build_window(tmp_dir)
            try:
                window._mode_modal_action.trigger()

                self.assertEqual(window._resolve_workspace_index(_VIEW_SETUP), _VIEW_MODAL_SETUP)
                self.assertEqual(window._resolve_workspace_index(_VIEW_LIVE), _VIEW_MODAL_LIVE)
                self.assertEqual(window._resolve_workspace_index(_VIEW_ANALYSIS), _VIEW_ANALYSIS)

                window._set_nav_index(_VIEW_LIVE)
                self.assertEqual(window._workspace.currentIndex(), _VIEW_MODAL_LIVE)
            finally:
                window.close()
                window.deleteLater()
                _app().processEvents()

    def test_switching_back_to_standard_stays_on_the_same_nav_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            window, _controller, _config = _build_window(tmp_dir)
            try:
                window._set_nav_index(_VIEW_LIVE)
                window._mode_modal_action.trigger()
                window._mode_standard_action.trigger()

                self.assertEqual(window._app_mode, "standard")
                self.assertEqual(window._workspace.currentIndex(), _VIEW_LIVE)
            finally:
                window.close()
                window.deleteLater()
                _app().processEvents()


class ModeSwitchBlockedWhileRunningTest(unittest.TestCase):
    """The mode menu must not change anything while a measurement runs."""

    def test_mode_switch_is_refused_and_checked_state_reverts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            window, controller, _config = _build_window(tmp_dir)
            try:

                class _FakeRunningThread:
                    is_running = True

                controller._acquisition_thread = _FakeRunningThread()

                window._mode_modal_action.trigger()

                self.assertEqual(window._app_mode, "standard")
                self.assertEqual(window._workspace.currentIndex(), _VIEW_SETUP)
                self.assertTrue(window._mode_standard_action.isChecked())
                self.assertFalse(window._mode_modal_action.isChecked())
            finally:
                controller._acquisition_thread = None
                window.close()
                window.deleteLater()
                _app().processEvents()


class ModeSettingsPersistenceTest(unittest.TestCase):
    """`ConfigurationManager.update_app_mode` - same pattern as
    `update_theme`/`update_last_trigger_settings`."""

    def test_app_mode_round_trips_through_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_manager = ConfigurationManager(Path(tmp_dir))
            config_manager.update_app_mode("modal")

            reloaded = ConfigurationManager(Path(tmp_dir))
            self.assertEqual(reloaded.settings.app_mode, "modal")

    def test_unknown_stored_mode_falls_back_to_standard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            window, _controller, config_manager = _build_window(tmp_dir)
            window.close()
            window.deleteLater()
            _app().processEvents()

            # Simulate a corrupted/future settings file.
            config_manager.settings.app_mode = "not-a-real-mode"
            config_manager.save_settings()

            window2, _controller2, _config2 = _build_window(tmp_dir)
            try:
                self.assertEqual(window2._app_mode, "standard")
            finally:
                window2.close()
                window2.deleteLater()
                _app().processEvents()


class ModalAnalysisConfigPersistenceTest(unittest.TestCase):
    """`ModalAnalysisConfig.to_dict`/`from_dict` and
    `ConfigurationManager.update_last_modal_config` - same pattern as
    `TriggerConfig`/`update_last_trigger_settings`."""

    def test_round_trip_preserves_the_nested_impact_condition(self) -> None:
        config = ModalAnalysisConfig(
            excitation_channel_hardware_id="cDAQ1Mod1/ai0",
            response_channels=[
                ModalResponseChannel(ModalAxis.X, "cDAQ1Mod1/ai1"),
                ModalResponseChannel(ModalAxis.Y, "cDAQ1Mod1/ai2"),
                ModalResponseChannel(ModalAxis.Z, "cDAQ1Mod1/ai3"),
            ],
            num_averages_target=8,
            frf_quantity="receptance",
            impact_condition=TriggerCondition(
                threshold_channel_hardware_id="cDAQ1Mod1/ai0",
                threshold_value=12.5,
                threshold_direction=TriggerDirection.RISES_ABOVE,
            ),
        )

        restored = ModalAnalysisConfig.from_dict(config.to_dict())

        self.assertEqual(restored, config)

    def test_configuration_manager_persists_the_modal_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_manager = ConfigurationManager(Path(tmp_dir))
            config = ModalAnalysisConfig(excitation_channel_hardware_id="cDAQ1Mod1/ai0")

            config_manager.update_last_modal_config(config)

            reloaded = ConfigurationManager(Path(tmp_dir))
            restored = ModalAnalysisConfig.from_dict(reloaded.settings.last_modal_config)
            self.assertEqual(restored, config)


if __name__ == "__main__":
    unittest.main()
