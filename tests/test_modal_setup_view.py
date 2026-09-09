"""
tests/test_modal_setup_view.py

Tests for `gui/modal_setup_view.py::ModalSetupView` - channel role
assignment (excitation + X/Y/Z response axes), the modal analysis
parameter form, and the sensitivity-unit conversion in
`_ModalChannelParameterDialog`.

QApplication pattern mirrors `tests/test_axis_labels.py` (module-level
singleton, see the comment there for why one is needed at all).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from config.configuration_manager import ConfigurationManager
from data.models import DeviceInfo, ModuleType
from gui.modal_setup_view import (
    _ROLE_EXCITATION,
    _ChannelRowState,
    _ModalChannelParameterDialog,
    ModalSetupView,
)

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


_TEST_DEVICE = DeviceInfo(
    device_name="cDAQ1Mod1",
    product_type="NI 9234",
    module_type=ModuleType.NI9234,
    num_channels=4,
    has_any_channels=True,
    physical_channels=["cDAQ1Mod1/ai0", "cDAQ1Mod1/ai1", "cDAQ1Mod1/ai2", "cDAQ1Mod1/ai3"],
    is_connected=True,
    connection_probed=True,
)


class ChannelAssignmentValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        _app()
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._config_manager = ConfigurationManager(Path(self._tmp_dir.name))
        self._view = ModalSetupView(self._config_manager)
        self._view.set_discovered_devices([_TEST_DEVICE])
        # build_current_config() calls show_error(), which for real
        # would open a genuinely blocking native QMessageBox - replaced
        # with a no-op that records the message, so the validation
        # tests below stay deterministic and headless.
        self._errors_shown: list[str] = []
        self._view.show_error = self._errors_shown.append

    def tearDown(self) -> None:
        self._view.close()
        self._view.deleteLater()
        _app().processEvents()
        self._tmp_dir.cleanup()

    def test_missing_excitation_channel_refuses_to_build_a_config(self) -> None:
        self._view._channel_states["x"] = _ChannelRowState(hardware_channel_id="cDAQ1Mod1/ai1")

        self.assertIsNone(self._view.build_current_config())
        self.assertEqual(len(self._errors_shown), 1)

    def test_missing_any_response_channel_refuses_to_build_a_config(self) -> None:
        self._view._channel_states[_ROLE_EXCITATION] = _ChannelRowState(
            hardware_channel_id="cDAQ1Mod1/ai0"
        )

        self.assertIsNone(self._view.build_current_config())
        self.assertEqual(len(self._errors_shown), 1)

    def test_excitation_and_one_response_axis_is_enough_to_start(self) -> None:
        self._view._channel_states[_ROLE_EXCITATION] = _ChannelRowState(
            hardware_channel_id="cDAQ1Mod1/ai0", sensitivity_mv_per_unit=11.2
        )
        self._view._channel_states["y"] = _ChannelRowState(
            hardware_channel_id="cDAQ1Mod1/ai1", sensitivity_mv_per_unit=10.5
        )

        config = self._view.build_current_config()

        self.assertIsNotNone(config)
        self.assertEqual(len(config.channels), 2)


class BuiltChannelPropertiesTest(unittest.TestCase):
    """The `Channel` objects `build_current_config()` produces - unit
    labeling, signal type, and the min/max range convention."""

    def setUp(self) -> None:
        _app()
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._view = ModalSetupView(ConfigurationManager(Path(self._tmp_dir.name)))
        self._view.set_discovered_devices([_TEST_DEVICE])
        self._view._channel_states[_ROLE_EXCITATION] = _ChannelRowState(
            hardware_channel_id="cDAQ1Mod1/ai0", sensitivity_mv_per_unit=11.2
        )
        self._view._channel_states["x"] = _ChannelRowState(
            hardware_channel_id="cDAQ1Mod1/ai1", sensitivity_mv_per_unit=10.5
        )
        self._view._channel_states["z"] = _ChannelRowState(
            hardware_channel_id="cDAQ1Mod1/ai2", sensitivity_mv_per_unit=9.8
        )
        self._view._name_edit.setText("Testmessung")
        self._config = self._view.build_current_config()

    def tearDown(self) -> None:
        self._view.close()
        self._view.deleteLater()
        _app().processEvents()
        self._tmp_dir.cleanup()

    def test_excitation_channel_is_labeled_in_newton_despite_accel_signal_type(self) -> None:
        """The documented shortcut (see the module docstring): the
        excitation channel is configured as SignalType.IEPE_ACCELERATION
        (electrically identical to an accelerometer), but ezDAQ's own
        `Channel.unit` correctly says "N" - the only unit tag that ever
        reaches ezDAQ's display/storage."""
        excitation = self._config.channels[0]

        self.assertEqual(excitation.hardware_channel, "cDAQ1Mod1/ai0")
        self.assertEqual(excitation.unit, "N")
        self.assertEqual(excitation.signal_type.value, "iepe_acceleration")
        self.assertEqual(excitation.sensitivity_mv_per_unit, 11.2)

    def test_response_channels_are_labeled_in_g(self) -> None:
        response_channels = self._config.channels[1:]

        self.assertEqual({c.unit for c in response_channels}, {"g"})
        self.assertEqual(
            {c.hardware_channel for c in response_channels},
            {"cDAQ1Mod1/ai1", "cDAQ1Mod1/ai2"},
        )

    def test_min_max_range_are_none_not_the_channel_default(self) -> None:
        """Mirrors `gui/widgets/channel_table.py`'s identical choice -
        leaves the hardware layer's own correct module fallback range
        in charge instead of the Channel-dataclass-wide ±10V default,
        which the NI9234 hardware rejects."""
        for channel in self._config.channels:
            self.assertIsNone(channel.min_range)
            self.assertIsNone(channel.max_range)

    def test_modal_mode_always_records_without_the_standard_trigger(self) -> None:
        from data.models import TriggerKind

        self.assertEqual(self._config.trigger.start.kind, TriggerKind.NONE)
        self.assertTrue(self._config.save_to_disk)
        self.assertTrue(self._config.recording_unlimited)


class CurrentModalConfigAndPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        _app()
        self._tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_current_modal_config_reflects_assigned_axes_only(self) -> None:
        view = ModalSetupView(ConfigurationManager(Path(self._tmp_dir.name)))
        try:
            view._channel_states[_ROLE_EXCITATION] = _ChannelRowState(
                hardware_channel_id="cDAQ1Mod1/ai0"
            )
            view._channel_states["x"] = _ChannelRowState(hardware_channel_id="cDAQ1Mod1/ai1")
            # "y"/"z" left unassigned - a valid, common uniaxial setup.

            modal_config = view.current_modal_config()

            self.assertEqual(modal_config.excitation_channel_hardware_id, "cDAQ1Mod1/ai0")
            self.assertEqual(len(modal_config.response_channels), 1)
            self.assertEqual(modal_config.response_channels[0].axis.value, "x")
        finally:
            view.close()
            view.deleteLater()
            _app().processEvents()

    def test_last_modal_config_persists_and_reloads_channel_assignment(self) -> None:
        config_manager = ConfigurationManager(Path(self._tmp_dir.name))
        view = ModalSetupView(config_manager)
        try:
            view._channel_states[_ROLE_EXCITATION] = _ChannelRowState(
                hardware_channel_id="cDAQ1Mod1/ai0"
            )
            view._channel_states["z"] = _ChannelRowState(hardware_channel_id="cDAQ1Mod1/ai3")
            config_manager.update_last_modal_config(view.current_modal_config())
        finally:
            view.close()
            view.deleteLater()
            _app().processEvents()

        reloaded_view = ModalSetupView(ConfigurationManager(Path(self._tmp_dir.name)))
        try:
            self.assertEqual(
                reloaded_view._channel_states[_ROLE_EXCITATION].hardware_channel_id,
                "cDAQ1Mod1/ai0",
            )
            self.assertEqual(
                reloaded_view._channel_states["z"].hardware_channel_id, "cDAQ1Mod1/ai3"
            )
            self.assertEqual(reloaded_view._channel_states["y"].hardware_channel_id, "")
        finally:
            reloaded_view.close()
            reloaded_view.deleteLater()
            _app().processEvents()


class SensitivityUnitDialogTest(unittest.TestCase):
    """`_ModalChannelParameterDialog` - the sensitivity-unit picker and
    its rescale-on-switch behavior."""

    def setUp(self) -> None:
        _app()

    def test_response_dialog_offers_g_and_metric_units(self) -> None:
        dialog = _ModalChannelParameterDialog(
            is_excitation=False, scale=1.0, offset=0.0, sensitivity_mv_per_unit=1.0
        )
        try:
            self.assertEqual(dialog._unit_combo.count(), 2)
            self.assertEqual(dialog._unit_combo.itemText(0), "mV/g")
        finally:
            dialog.close()
            dialog.deleteLater()
            _app().processEvents()

    def test_switching_unit_rescales_the_displayed_number_not_its_meaning(self) -> None:
        """98.0665 mV/g re-expressed in mV/(m/s²) is 10.0 - the DISPLAYED
        number must change so the underlying physical sensitivity
        (98.0665 mV/g) stays the same, rather than reinterpreting the
        same digits in the new unit."""
        dialog = _ModalChannelParameterDialog(
            is_excitation=False, scale=1.0, offset=0.0, sensitivity_mv_per_unit=98.0665
        )
        try:
            dialog._unit_combo.setCurrentIndex(1)  # mV/(m/s^2)

            self.assertAlmostEqual(dialog._sensitivity_spin.value(), 10.0, places=6)
            self.assertAlmostEqual(dialog.sensitivity_mv_per_unit(), 98.0665, places=4)

            dialog._unit_combo.setCurrentIndex(0)  # back to mV/g

            self.assertAlmostEqual(dialog._sensitivity_spin.value(), 98.0665, places=6)
        finally:
            dialog.close()
            dialog.deleteLater()
            _app().processEvents()

    def test_excitation_dialog_converts_between_newton_and_pound_force(self) -> None:
        """1 lbf = 4.4482216153 N - a sensor rated 1.0 mV/N reads as
        4.4482216153 mV/lbf (more mV per the larger force unit), same
        physical sensitivity re-expressed."""
        dialog = _ModalChannelParameterDialog(
            is_excitation=True, scale=1.0, offset=0.0, sensitivity_mv_per_unit=1.0
        )
        try:
            self.assertEqual(dialog._unit_combo.itemText(0), "mV/N")

            dialog._unit_combo.setCurrentIndex(1)  # mV/lbf

            self.assertAlmostEqual(dialog._sensitivity_spin.value(), 4.4482216153, places=6)
            self.assertAlmostEqual(dialog.sensitivity_mv_per_unit(), 1.0, places=6)
        finally:
            dialog.close()
            dialog.deleteLater()
            _app().processEvents()

    def test_scale_and_offset_round_trip_unchanged(self) -> None:
        dialog = _ModalChannelParameterDialog(
            is_excitation=False, scale=2.5, offset=-1.5, sensitivity_mv_per_unit=10.0
        )
        try:
            self.assertEqual(dialog.scale(), 2.5)
            self.assertEqual(dialog.offset(), -1.5)
        finally:
            dialog.close()
            dialog.deleteLater()
            _app().processEvents()


class RetranslateTest(unittest.TestCase):
    def test_retranslate_ui_does_not_raise_and_keeps_selections(self) -> None:
        _app()
        tmp_dir = tempfile.TemporaryDirectory()
        try:
            view = ModalSetupView(ConfigurationManager(Path(tmp_dir.name)))
            try:
                view._estimator_combo.setCurrentIndex(1)  # "h2"
                selected_estimator = view._estimator_combo.currentData()

                view.retranslate_ui()

                self.assertEqual(view._estimator_combo.currentData(), selected_estimator)
            finally:
                view.close()
                view.deleteLater()
                _app().processEvents()
        finally:
            tmp_dir.cleanup()


class ChannelNameAndExcitationAxisTest(unittest.TestCase):
    """The channel's own name/formula symbol (freely editable, same as
    the standard configuration view) and the excitation channel's
    direction - see `data/models.py::ModalAnalysisConfig.
    excitation_axis`/`excitation_display_name`."""

    def setUp(self) -> None:
        _app()
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._view = ModalSetupView(ConfigurationManager(Path(self._tmp_dir.name)))
        self._view.set_discovered_devices([_TEST_DEVICE])
        self._view._channel_states[_ROLE_EXCITATION] = _ChannelRowState(
            hardware_channel_id="cDAQ1Mod1/ai0"
        )
        self._view._channel_states["x"] = _ChannelRowState(hardware_channel_id="cDAQ1Mod1/ai1")

    def tearDown(self) -> None:
        self._view.close()
        self._view.deleteLater()
        _app().processEvents()
        self._tmp_dir.cleanup()

    def test_the_word_hammer_no_longer_appears_in_the_excitation_label(self) -> None:
        from gui.i18n import t

        self.assertNotIn("Hammer", t("modal_excitation_label"))

    def test_empty_display_name_falls_back_to_the_axis_derived_default(self) -> None:
        config = self._view.build_current_config()

        excitation, response_x = config.channels
        self.assertEqual(excitation.display_name, "F")
        self.assertEqual(response_x.display_name, "a_x")

    def test_custom_display_name_is_used_when_entered(self) -> None:
        self._view._display_name_edits[_ROLE_EXCITATION].setText("F1")
        self._view._display_name_edits["x"].setText("a_horizontal")

        config = self._view.build_current_config()

        excitation, response_x = config.channels
        self.assertEqual(excitation.display_name, "F1")
        self.assertEqual(response_x.display_name, "a_horizontal")

    def test_excitation_axis_defaults_to_x_and_is_configurable(self) -> None:
        self.assertEqual(self._view.current_modal_config().excitation_axis.value, "x")

        index = self._view._excitation_axis_combo.findData("y")
        self._view._excitation_axis_combo.setCurrentIndex(index)

        self.assertEqual(self._view.current_modal_config().excitation_axis.value, "y")

    def test_display_names_and_excitation_axis_persist_and_reload(self) -> None:
        self._view._display_name_edits[_ROLE_EXCITATION].setText("F1")
        self._view._display_name_edits["x"].setText("a_x1")
        index = self._view._excitation_axis_combo.findData("z")
        self._view._excitation_axis_combo.setCurrentIndex(index)
        config_manager = self._view._configuration_manager
        config_manager.update_last_modal_config(self._view.current_modal_config())

        reloaded_view = ModalSetupView(ConfigurationManager(Path(self._tmp_dir.name)))
        try:
            self.assertEqual(reloaded_view._display_name_edits[_ROLE_EXCITATION].text(), "F1")
            self.assertEqual(reloaded_view._display_name_edits["x"].text(), "a_x1")
            self.assertEqual(reloaded_view._excitation_axis_combo.currentData(), "z")
        finally:
            reloaded_view.close()
            reloaded_view.deleteLater()
            _app().processEvents()


class SectionHeaderStyleTest(unittest.TestCase):
    """Section headers should carry the same emphasis as
    `gui/setup_view.py::SetupView` - see
    `ModalSetupView._apply_section_header_emphasis`."""

    def test_section_headers_are_bold_and_enlarged(self) -> None:
        _app()
        tmp_dir = tempfile.TemporaryDirectory()
        try:
            view = ModalSetupView(ConfigurationManager(Path(tmp_dir.name)))
            try:
                base_point_size = view.font().pointSize()
                for header in (
                    view._device_header,
                    view._channel_header,
                    view._parameter_header,
                    view._measurement_header,
                    view._storage_header,
                ):
                    self.assertTrue(header.font().bold())
                    if base_point_size > 0:
                        self.assertGreater(header.font().pointSize(), base_point_size)
            finally:
                view.close()
                view.deleteLater()
                _app().processEvents()
        finally:
            tmp_dir.cleanup()


class DeviceTreeTest(unittest.TestCase):
    """`set_discovered_devices`/`show_discovery_error` - same rendering
    style as `gui/setup_view.py::SetupView`'s device tree (see
    `ModalSetupView._build_device_section`)."""

    def setUp(self) -> None:
        _app()
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._view = ModalSetupView(ConfigurationManager(Path(self._tmp_dir.name)))

    def tearDown(self) -> None:
        self._view.close()
        self._view.deleteLater()
        _app().processEvents()
        self._tmp_dir.cleanup()

    def test_a_device_is_rendered_with_its_module_and_channels(self) -> None:
        self._view.set_discovered_devices([_TEST_DEVICE])

        self.assertEqual(self._view._device_list.topLevelItemCount(), 1)
        top_item = self._view._device_list.topLevelItem(0)
        self.assertIn("NI9234", top_item.text(0))
        self.assertEqual(top_item.childCount(), 4)

    def test_no_devices_shows_a_hint_instead_of_an_empty_tree(self) -> None:
        self._view.set_discovered_devices([])

        self.assertEqual(self._view._device_list.topLevelItemCount(), 1)

    def test_discovery_error_replaces_the_tree_with_the_message(self) -> None:
        self._view.set_discovered_devices([_TEST_DEVICE])

        self._view.show_discovery_error("kein Treiber gefunden")

        self.assertEqual(self._view._device_list.topLevelItemCount(), 1)
        self.assertIn("kein Treiber gefunden", self._view._device_list.topLevelItem(0).text(0))
        self.assertIsNone(self._view.get_discovered_devices())


class ChannelTableTest(unittest.TestCase):
    """The channel table itself (`_channel_table`) - fixed 4-row/5-column
    structure, `_PickerCell` reuse for the hardware-channel column."""

    def setUp(self) -> None:
        _app()
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._view = ModalSetupView(ConfigurationManager(Path(self._tmp_dir.name)))
        self._view.set_discovered_devices([_TEST_DEVICE])

    def tearDown(self) -> None:
        self._view.close()
        self._view.deleteLater()
        _app().processEvents()
        self._tmp_dir.cleanup()

    def test_table_has_one_row_per_role_and_five_columns(self) -> None:
        self.assertEqual(self._view._channel_table.rowCount(), 4)
        self.assertEqual(self._view._channel_table.columnCount(), 5)

    def test_picker_cell_reflects_the_assigned_hardware_channel(self) -> None:
        self._view._channel_states[_ROLE_EXCITATION] = _ChannelRowState(
            hardware_channel_id="cDAQ1Mod1/ai0"
        )
        self._view._refresh_channel_row_label(_ROLE_EXCITATION)

        # `_PickerCell` exposes no text getter (see its own docstring:
        # setText/setToolTip/setIcon/setIconSize/clicked only) - reading
        # its internal label is the only way to verify the displayed
        # text, same as elsewhere in this codebase's tests reaching into
        # a widget's private internals.
        self.assertEqual(
            self._view._channel_labels[_ROLE_EXCITATION]._label.text(), "cDAQ1Mod1/ai0"
        )

    def test_response_rows_show_their_fixed_axis_read_only(self) -> None:
        y_axis_widget = self._view._channel_table.cellWidget(2, ModalSetupView._COL_AXIS)
        self.assertEqual(y_axis_widget.text(), "Y")

    def test_excitation_row_axis_cell_is_the_editable_combo(self) -> None:
        axis_widget = self._view._channel_table.cellWidget(0, ModalSetupView._COL_AXIS)
        self.assertIs(axis_widget, self._view._excitation_axis_combo)


class StartStopButtonStyleTest(unittest.TestCase):
    """Record/Stop icon buttons, same style as
    `gui/setup_view.py::SetupView` - see
    `ModalSetupView._build_start_stop_section`."""

    def test_buttons_have_icons_and_toggle_enabled_state_oppositely(self) -> None:
        _app()
        tmp_dir = tempfile.TemporaryDirectory()
        try:
            view = ModalSetupView(ConfigurationManager(Path(tmp_dir.name)))
            try:
                self.assertFalse(view._start_button.icon().isNull())
                self.assertFalse(view._stop_button.icon().isNull())

                view.set_start_enabled(False, "measurement_running")
                self.assertFalse(view._start_button.isEnabled())
                self.assertTrue(view._stop_button.isEnabled())

                view.set_start_enabled(True)
                self.assertTrue(view._start_button.isEnabled())
                self.assertFalse(view._stop_button.isEnabled())
            finally:
                view.close()
                view.deleteLater()
                _app().processEvents()
        finally:
            tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
