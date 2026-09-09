"""
gui/modal_setup_view.py

Configuration view for the experimental modal analysis mode (impact
hammer + accelerometer) - see `gui/main_window.py::MainWindow` for how
this replaces `gui/setup_view.py::SetupView` in the workspace stack
while "Modalanalyse" mode is active.

Deliberately a separate, dedicated class rather than an extension of
`SetupView`: the standard configuration view is already large, and
keeping the modal-specific UI in its own file means the standard
measurement path (`SetupView`, `LiveView`) is never touched by this
work - see the design plan for the full rationale.

Channel model: always exactly ONE excitation (hammer) channel and 1-3
response (accelerometer) channels, one per axis (`data.models.ModalAxis`)
- a hammer test excites in a single direction but a triaxial
accelerometer measures the response in all three at once. Both roles
are configured as `SignalType.IEPE_ACCELERATION` (see
`hardware/ni9234.py`) - including the excitation channel, which is
electrically identical to an accelerometer (both are two-wire,
constant-current-excited IEPE sensors whose output voltage is simply
divided by a "sensitivity" number to get the physical value; only WHICH
physical value that is - force vs. acceleration - differs). Using the
accelerometer channel type for the hammer is a deliberate, documented
shortcut (see the design plan's "Zurückgestellt" section) rather than
adding `SignalType.IEPE_FORCE`/`add_ai_force_iepe_chan` support - it
produces numerically identical, correct results (DAQmx just divides by
the sensitivity value regardless of which physical unit it is
labeled), only DAQmx's OWN internal unit tag for the channel says "g"
instead of "N"; that tag never reaches ezDAQ's own display or storage,
both of which use `Channel.unit` (set correctly to "N" here).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QFont, QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.configuration_manager import ConfigurationManager
from data.models import (
    Channel,
    DeviceInfo,
    MeasurementConfig,
    ModalAnalysisConfig,
    ModalAxis,
    ModalResponseChannel,
    ModuleType,
    SignalType,
    StorageFormat,
    TriggerCondition,
    TriggerDirection,
)
from gui.i18n import connect_language_changed, t
from gui.theme import (
    PLAY_ICON_COLOR,
    action_button_style,
    connect_theme_changed,
    disabled_text_color,
    draw_ellipsis_icon,
    draw_play_icon,
    draw_stop_icon,
    repolish,
)
# `_PickerCell` is reused verbatim here for the SAME look as the
# standard configuration view's channel table (label + "..." button
# that opens `HardwareChannelPickerDialog`) - its own docstring
# describes it as a small, self-contained, reusable widget (not an
# implementation detail specific to `ChannelTableWidget`), which is
# exactly the case here.
from gui.widgets.channel_table import HardwareChannelPickerDialog, _PickerCell
from gui.widgets.spinbox import NoWheelSpinBox, PrecisionDoubleSpinBox

# Mirrors `gui/setup_view.py::_STORAGE_FORMAT_LABEL_KEYS` - same i18n
# keys, small enough to duplicate rather than import a private name
# from another view.
_STORAGE_FORMAT_LABEL_KEYS: dict[StorageFormat, str] = {
    StorageFormat.PARQUET: "storage_format_parquet",
    StorageFormat.CSV: "storage_format_csv",
}

# Mirrors `gui/trigger_settings_dialog.py::_TRIGGER_DIRECTION_LABEL_KEYS`
# - same reasoning as above.
_TRIGGER_DIRECTION_LABEL_KEYS: dict[TriggerDirection, str] = {
    TriggerDirection.RISES_ABOVE: "trigger_direction_rises_above",
    TriggerDirection.FALLS_BELOW: "trigger_direction_falls_below",
    TriggerDirection.ABS_EXCEEDS: "trigger_direction_abs_exceeds",
}

_EXCITATION_WINDOW_LABEL_KEYS = {"force": "modal_window_force", "rectangular": "modal_window_rectangular"}
_RESPONSE_WINDOW_LABEL_KEYS = {
    "exponential": "modal_window_exponential",
    "hann": "modal_window_hann",
    "rectangular": "modal_window_rectangular",
}
_ESTIMATOR_LABEL_KEYS = {"h1": "modal_estimator_h1", "h2": "modal_estimator_h2"}
_FRF_QUANTITY_LABEL_KEYS = {
    "accelerance": "modal_frf_quantity_accelerance",
    "mobility": "modal_frf_quantity_mobility",
    "receptance": "modal_frf_quantity_receptance",
}

# Sensitivity unit choices offered in `_ModalChannelParameterDialog`,
# each mapped to the factor that converts a value FROM that unit TO the
# internal unit DAQmx/`hardware/ni9234.py` actually wants (mV/g for
# every IEPE_ACCELERATION channel - including the excitation channel,
# see the module docstring). Kept here rather than as a general
# feature of `gui/widgets/channel_table.py::ChannelParameterDialog`
# (which has no unit picker for its sensitivity field at all): that
# dialog is shared by the standard, well-tested channel configuration
# path, and widening it is a separate, general improvement outside the
# scope of this experimental mode.
_RESPONSE_SENSITIVITY_UNITS: list[tuple[str, float]] = [
    ("mV/g", 1.0),
    ("mV/(m/s²)", 9.80665),  # 1 g = 9.80665 m/s^2
]
# A force sensor's sensitivity is only ever RELABELED, not actually
# converted by DAQmx (see the module docstring) - "mV/N" IS the
# internal unit here, "mV/lbf" is offered because some hammer
# datasheets (particularly US-made ones) specify it that way.
_EXCITATION_SENSITIVITY_UNITS: list[tuple[str, float]] = [
    ("mV/N", 1.0),
    ("mV/lbf", 1.0 / 4.4482216153),  # 1 lbf = 4.4482216153 N
]

# Internal role identifiers for the four fixed channel rows - "x"/"y"/
# "z" match `ModalAxis.value` directly, "excitation" does not have a
# `ModalAxis` counterpart (the excitation channel is not an axis of the
# response sensor).
_ROLE_EXCITATION = "excitation"
_RESPONSE_ROLES = [ModalAxis.X.value, ModalAxis.Y.value, ModalAxis.Z.value]

# Shown as placeholder text (not a persisted value, see
# `ModalAnalysisConfig.excitation_display_name`/
# `ModalResponseChannel.display_name`) when the operator has not typed
# their own name/formula symbol for a channel yet - matches the
# customary symbols (force, acceleration) rather than a generic
# "Kanal 1"-style placeholder.
_DEFAULT_DISPLAY_NAMES = {_ROLE_EXCITATION: "F", "x": "a_x", "y": "a_y", "z": "a_z"}

# Default sample rate for a new modal-analysis measurement - 51200/4, a
# valid point on the NI9234's fixed rate grid (see
# `data/models.py::resolve_rate_groups`) with reasonable bandwidth
# (Nyquist 6400 Hz) for typical structural impact testing.
# Deliberately NOT `AppSettings.default_sample_rate_hz` (1000.0): every
# modal channel sits on the NI9234, whose grid does not include 1000 Hz
# at all (lowest possible is ~1651.6 Hz) - reusing the generic app-wide
# default would fail the very first "Start" click.
_DEFAULT_MODAL_SAMPLE_RATE_HZ = 12_800.0


@dataclass
class _ChannelRowState:
    """Per-row state for one channel assignment slot (excitation or one
    response axis) - deliberately separate from `Channel`/
    `ModalResponseChannel`: those describe a fully configured channel,
    this also represents "not assigned yet" (`hardware_channel_id ==
    ""`) without needing an Optional wrapper everywhere."""

    hardware_channel_id: str = ""
    scale: float = 1.0
    offset: float = 0.0
    sensitivity_mv_per_unit: float = 1.0


class _ModalChannelParameterDialog(QDialog):
    """Small parameter dialog for one modal-analysis channel (excitation
    or one response axis) - scale, offset, and sensitivity WITH a unit
    picker.

    Deliberately its own small dialog rather than reusing
    `gui/widgets/channel_table.py::ChannelParameterDialog` (which
    handles every signal type in one dialog and has no unit picker for
    its bare sensitivity field - see the module-level comment on
    `_RESPONSE_SENSITIVITY_UNITS`): every modal channel is always
    `SignalType.IEPE_ACCELERATION`, so a small, focused dialog is both
    simpler than reusing the multi-type one and, with the added unit
    picker, more correct - sensor datasheets specify sensitivity in
    different units (mV/g vs mV/(m/s²) for accelerometers, mV/N vs
    mV/lbf for force sensors), and typing e.g. a mV/(m/s²) number
    straight into a field that silently means mV/g would apply a wrong
    scale (differing by almost a factor of 10) to every sample without
    any indication something is off.

    The value passed in/read back (`sensitivity_mv_per_unit`) is always
    in the INTERNAL unit (first entry of the relevant unit list) -
    which unit the value was originally entered in is not persisted, so
    reopening this dialog always shows it re-expressed in that internal
    unit. Switching the unit combo AFTER entering a value rescales the
    displayed number to keep the underlying physical sensitivity
    unchanged, rather than silently reinterpreting the same digits in a
    different unit.
    """

    def __init__(
        self,
        is_excitation: bool,
        scale: float,
        offset: float,
        sensitivity_mv_per_unit: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("modal_channel_parameter_dialog_title"))
        self._units = _EXCITATION_SENSITIVITY_UNITS if is_excitation else _RESPONSE_SENSITIVITY_UNITS

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._scale_spin = PrecisionDoubleSpinBox()
        self._scale_spin.setRange(-1e9, 1e9)
        self._scale_spin.setValue(scale)
        form.addRow(t("param_scale_label"), self._scale_spin)

        self._offset_spin = PrecisionDoubleSpinBox()
        self._offset_spin.setRange(-1e9, 1e9)
        self._offset_spin.setValue(offset)
        form.addRow(t("param_offset_label"), self._offset_spin)

        sensitivity_row = QHBoxLayout()
        self._sensitivity_spin = PrecisionDoubleSpinBox()
        self._sensitivity_spin.setRange(0.0, 1e6)
        self._unit_combo = QComboBox()
        for label, _factor in self._units:
            self._unit_combo.addItem(label)
        # The value passed in is already in the internal unit (first
        # entry) - initialize the combo to that before computing the
        # displayed raw number, so `_on_unit_changed` is never fed a
        # stale `_current_factor`.
        self._current_factor = self._units[0][1]
        self._sensitivity_spin.setValue(sensitivity_mv_per_unit / self._current_factor)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        sensitivity_row.addWidget(self._sensitivity_spin, stretch=1)
        sensitivity_row.addWidget(self._unit_combo)
        self._sensitivity_label = QLabel(t("modal_sensitivity_label"))
        form.addRow(self._sensitivity_label, sensitivity_row)

        layout.addLayout(form)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_unit_changed(self, index: int) -> None:
        """Rescales the displayed raw number so the underlying physical
        sensitivity stays the same across a unit switch - see the class
        docstring for why this matters."""
        _, new_factor = self._units[index]
        physical_value = self._sensitivity_spin.value() * self._current_factor
        self._current_factor = new_factor
        self._sensitivity_spin.blockSignals(True)
        self._sensitivity_spin.setValue(physical_value / new_factor)
        self._sensitivity_spin.blockSignals(False)

    def scale(self) -> float:
        return self._scale_spin.value()

    def offset(self) -> float:
        return self._offset_spin.value()

    def sensitivity_mv_per_unit(self) -> float:
        """Always in the INTERNAL unit (mV/g for a response channel,
        mV/N-equivalent for the excitation channel) - see the class
        docstring."""
        return self._sensitivity_spin.value() * self._current_factor

    def retranslate_ui(self) -> None:
        self.setWindowTitle(t("modal_channel_parameter_dialog_title"))
        self._sensitivity_label.setText(t("modal_sensitivity_label"))


class ModalSetupView(QWidget):
    """Configuration view for the experimental modal analysis mode.

    Signals:
        discover_hardware_requested: Same meaning as
            `gui/setup_view.py::SetupView.discover_hardware_requested`.
        open_ni_max_requested: Same meaning as `SetupView`'s.
        storage_path_requested: Same meaning as `SetupView`'s - the
            storage location is shared app-wide, not per-mode.
        start_measurement_requested: User wants to start a modal
            analysis measurement with the given `MeasurementConfig`
            (always `save_to_disk=True`, `trigger=TriggerConfig()` -
            modal mode always records continuously without the
            standard start/stop trigger machinery, see
            `data/models.py::ModalAnalysisConfig`'s docstring on
            `impact_condition`).
        stop_requested: Same meaning as `SetupView`'s.
    """

    discover_hardware_requested = pyqtSignal()
    open_ni_max_requested = pyqtSignal()
    storage_path_requested = pyqtSignal()
    start_measurement_requested = pyqtSignal(object)  # MeasurementConfig
    stop_requested = pyqtSignal()

    def __init__(
        self,
        configuration_manager: ConfigurationManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._configuration_manager = configuration_manager
        self._discovered_devices: list[DeviceInfo] | None = None
        self._storage_path_is_set = False

        # One row of state per role - see `_ROLE_EXCITATION`/
        # `_RESPONSE_ROLES`. Seeded from the last used configuration so
        # the view reopens exactly as it was left, same pattern as
        # `SetupView`'s `last_trigger_config`.
        self._channel_states: dict[str, _ChannelRowState] = {
            role: _ChannelRowState() for role in [_ROLE_EXCITATION, *_RESPONSE_ROLES]
        }
        # Populated in `_build_channel_section` (needs widgets to exist).
        self._channel_labels: dict[str, _PickerCell] = {}
        self._impact_condition = TriggerCondition(threshold_direction=TriggerDirection.RISES_ABOVE)
        self._load_last_modal_config()

        # Same scroll-area wrapping as `SetupView` - several sections
        # (devices, channels, parameters, storage) can easily exceed the
        # window height.
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll_area, stretch=1)

        content = QWidget()
        scroll_area.setWidget(content)
        layout = QVBoxLayout(content)

        self._build_device_section(layout)
        self._build_channel_section(layout)
        self._build_parameter_section(layout)
        self._build_measurement_section(layout)
        self._build_storage_section(layout)
        self._build_start_stop_section(layout)
        self._apply_section_header_emphasis()

        connect_language_changed(self.retranslate_ui)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _build_device_section(self, layout: QVBoxLayout) -> None:
        # Same device TREE as `gui/setup_view.py::SetupView` (see
        # `set_discovered_devices` below) - kept visually consistent
        # with the standard configuration view rather than the earlier,
        # more compact status-line version.
        self._device_header = QLabel(t("connected_devices"))
        layout.addWidget(self._device_header)
        device_group = QGroupBox()
        device_layout = QVBoxLayout(device_group)
        discover_row = QHBoxLayout()
        self._discover_button = QPushButton(t("search_devices"))
        self._discover_button.clicked.connect(self.discover_hardware_requested.emit)
        self._open_ni_max_button = QPushButton(t("open_ni_max_button"))
        self._open_ni_max_button.clicked.connect(self.open_ni_max_requested.emit)
        discover_row.addWidget(self._discover_button)
        discover_row.addWidget(self._open_ni_max_button)
        self._device_list = QTreeWidget()
        self._device_list.setHeaderHidden(True)
        self._device_list.setMinimumHeight(120)
        device_layout.addLayout(discover_row)
        device_layout.addWidget(self._device_list)
        layout.addWidget(device_group)

    # Column order of `_channel_table` - named so the retranslation code
    # below does not repeat the raw indices.
    _COL_DISPLAY_NAME, _COL_AXIS, _COL_HW_CHANNEL, _COL_PARAMETERS, _COL_CLEAR = range(5)

    def _build_channel_section(self, layout: QVBoxLayout) -> None:
        self._channel_header = QLabel(t("modal_channel_assignment_header"))
        layout.addWidget(self._channel_header)
        channel_group = QGroupBox()
        channel_layout = QVBoxLayout(channel_group)

        self._display_name_edits: dict[str, QLineEdit] = {}
        self._channel_labels: dict[str, _PickerCell] = {}
        self._row_header_items: dict[str, QTableWidgetItem] = {}

        # Direction of the EXCITATION itself - a response channel's axis
        # is its identity (see `_RESPONSE_ROLES`), but the excitation
        # has so far only ever had a hardware channel, not a direction;
        # which cross-axis FRF a measurement represents depends on it
        # (see `data/models.py::ModalAnalysisConfig.excitation_axis`).
        self._excitation_axis_combo = QComboBox()
        for axis in ModalAxis:
            self._excitation_axis_combo.addItem(axis.value.upper(), axis.value)
        self._set_combo_by_data(self._excitation_axis_combo, self._modal_config.excitation_axis.value)

        # One table, styled like the standard configuration view's
        # channel table (`gui/widgets/channel_table.py::
        # ChannelTableWidget`) - reusing its `_PickerCell` cell widget
        # directly - rather than the earlier per-role QHBoxLayout rows,
        # which looked visually inconsistent with the rest of the app.
        # A genuinely dynamic, add/remove-row table (like the standard
        # one) does not fit here though: the four rows - one excitation,
        # up to three response axes - are a fixed structural part of
        # `data/models.py::ModalAnalysisConfig`, not a free list.
        roles = [_ROLE_EXCITATION, *_RESPONSE_ROLES]
        self._channel_table = QTableWidget(len(roles), 5)
        self._channel_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._channel_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._channel_table.horizontalHeader().setSectionResizeMode(
            self._COL_HW_CHANNEL, QHeaderView.ResizeMode.Stretch
        )
        self._channel_table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        for row, role in enumerate(roles):
            row_header = QTableWidgetItem(self._role_label_text(role))
            self._channel_table.setVerticalHeaderItem(row, row_header)
            self._row_header_items[role] = row_header
            self._build_channel_table_row(row, role)

        self._retranslate_channel_table_header()
        self._fix_channel_table_height()
        channel_layout.addWidget(self._channel_table)
        layout.addWidget(channel_group)

    def _fix_channel_table_height(self) -> None:
        """Sizes `_channel_table` to fit exactly its 4 rows plus the
        header - a `QTableWidget` left at its default size policy
        otherwise stretches to fill whatever space the layout offers,
        which reads as a fifth, empty row below "Antwort Z" rather than
        as the fixed 4-row table it actually is."""
        header_height = self._channel_table.horizontalHeader().height()
        rows_height = self._channel_table.verticalHeader().length()
        frame = 2 * self._channel_table.frameWidth()
        self._channel_table.setFixedHeight(header_height + rows_height + frame)

    @staticmethod
    def _role_label_text(role: str) -> str:
        return t("modal_excitation_label") if role == _ROLE_EXCITATION else t(f"modal_response_{role}_label")

    def _build_channel_table_row(self, row: int, role: str) -> None:
        # The channel's own name/formula symbol (e.g. "F", "a_x") -
        # freely editable, exactly like naming a channel in the
        # standard configuration view (`Channel.display_name`). Empty
        # is a valid, deliberate state (see `_build_channel`/
        # `current_modal_config`): the placeholder shows what will be
        # used if the operator leaves it blank, without that default
        # being persisted as if it had been chosen on purpose.
        name_edit = QLineEdit(self._initial_display_names.get(role, ""))
        name_edit.setPlaceholderText(_DEFAULT_DISPLAY_NAMES[role])
        name_edit.setToolTip(t("modal_display_name_tooltip"))
        self._display_name_edits[role] = name_edit
        self._channel_table.setCellWidget(row, self._COL_DISPLAY_NAME, name_edit)

        if role == _ROLE_EXCITATION:
            self._channel_table.setCellWidget(row, self._COL_AXIS, self._excitation_axis_combo)
        else:
            axis_label = QLabel(role.upper())
            axis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._channel_table.setCellWidget(row, self._COL_AXIS, axis_label)

        picker_cell = _PickerCell()
        picker_cell.setText(t("modal_no_channel_assigned"))
        picker_cell.setIcon(QIcon(draw_ellipsis_icon(14)))
        picker_cell.setIconSize(QSize(14, 14))
        picker_cell.clicked.connect(lambda r=role: self._on_pick_channel_clicked(r))
        self._channel_labels[role] = picker_cell
        self._channel_table.setCellWidget(row, self._COL_HW_CHANNEL, picker_cell)

        parameters_button = QPushButton(t("modal_edit_parameters_button"))
        parameters_button.setIcon(QIcon(draw_ellipsis_icon(14)))
        parameters_button.clicked.connect(
            lambda _checked=False, r=role: self._on_edit_parameters_clicked(r)
        )
        self._channel_table.setCellWidget(row, self._COL_PARAMETERS, parameters_button)

        clear_button = QPushButton(t("modal_clear_channel_button"))
        clear_button.clicked.connect(lambda _checked=False, r=role: self._on_clear_channel_clicked(r))
        self._channel_table.setCellWidget(row, self._COL_CLEAR, clear_button)

    def _retranslate_channel_table_header(self) -> None:
        self._channel_table.setHorizontalHeaderLabels(
            [
                t("modal_column_display_name"),
                t("modal_column_axis"),
                t("modal_column_hardware_channel"),
                t("modal_edit_parameters_button"),
                t("modal_clear_channel_button"),
            ]
        )

    def _build_parameter_section(self, layout: QVBoxLayout) -> None:
        self._parameter_header = QLabel(t("modal_parameters_header"))
        layout.addWidget(self._parameter_header)
        parameter_group = QGroupBox()
        form = QFormLayout(parameter_group)

        self._excitation_window_combo = QComboBox()
        for value, key in _EXCITATION_WINDOW_LABEL_KEYS.items():
            self._excitation_window_combo.addItem(t(key), value)
        self._set_combo_by_data(self._excitation_window_combo, self._modal_config.excitation_window)
        self._excitation_window_row_label = QLabel(f"{t('modal_excitation_window_label')}:")
        form.addRow(self._excitation_window_row_label, self._excitation_window_combo)

        self._response_window_combo = QComboBox()
        for value, key in _RESPONSE_WINDOW_LABEL_KEYS.items():
            self._response_window_combo.addItem(t(key), value)
        self._set_combo_by_data(self._response_window_combo, self._modal_config.response_window)
        self._response_window_row_label = QLabel(f"{t('modal_response_window_label')}:")
        form.addRow(self._response_window_row_label, self._response_window_combo)

        self._frequency_resolution_spin = PrecisionDoubleSpinBox()
        self._frequency_resolution_spin.setRange(0.01, 10_000.0)
        self._frequency_resolution_spin.setValue(self._modal_config.frequency_resolution_hz)
        self._frequency_resolution_row_label = QLabel(f"{t('modal_frequency_resolution_label')}:")
        form.addRow(self._frequency_resolution_row_label, self._frequency_resolution_spin)

        self._num_averages_spin = NoWheelSpinBox()
        self._num_averages_spin.setRange(1, 1000)
        self._num_averages_spin.setValue(self._modal_config.num_averages_target)
        self._num_averages_row_label = QLabel(f"{t('modal_num_averages_label')}:")
        form.addRow(self._num_averages_row_label, self._num_averages_spin)

        self._estimator_combo = QComboBox()
        for value, key in _ESTIMATOR_LABEL_KEYS.items():
            self._estimator_combo.addItem(t(key), value)
        self._set_combo_by_data(self._estimator_combo, self._modal_config.estimator)
        self._estimator_row_label = QLabel(f"{t('modal_estimator_label')}:")
        form.addRow(self._estimator_row_label, self._estimator_combo)

        self._frf_quantity_combo = QComboBox()
        for value, key in _FRF_QUANTITY_LABEL_KEYS.items():
            self._frf_quantity_combo.addItem(t(key), value)
        self._set_combo_by_data(self._frf_quantity_combo, self._modal_config.frf_quantity)
        self._frf_quantity_row_label = QLabel(f"{t('modal_frf_quantity_label')}:")
        form.addRow(self._frf_quantity_row_label, self._frf_quantity_combo)

        threshold_row = QHBoxLayout()
        self._impact_threshold_spin = PrecisionDoubleSpinBox()
        self._impact_threshold_spin.setRange(0.0, 1e6)
        self._impact_threshold_spin.setValue(self._impact_condition.threshold_value)
        self._impact_direction_combo = QComboBox()
        for direction in TriggerDirection:
            self._impact_direction_combo.addItem(
                t(_TRIGGER_DIRECTION_LABEL_KEYS[direction]), direction.value
            )
        self._set_combo_by_data(
            self._impact_direction_combo, self._impact_condition.threshold_direction.value
        )
        threshold_row.addWidget(self._impact_threshold_spin, stretch=1)
        threshold_row.addWidget(self._impact_direction_combo)
        self._impact_threshold_row_label = QLabel(f"{t('modal_impact_threshold_label')}:")
        form.addRow(self._impact_threshold_row_label, threshold_row)

        self._pretrigger_spin = PrecisionDoubleSpinBox()
        self._pretrigger_spin.setRange(0.0, 10_000.0)
        self._pretrigger_spin.setValue(self._modal_config.pretrigger_ms)
        self._pretrigger_row_label = QLabel(f"{t('modal_pretrigger_label')}:")
        form.addRow(self._pretrigger_row_label, self._pretrigger_spin)

        self._double_hit_window_spin = PrecisionDoubleSpinBox()
        self._double_hit_window_spin.setRange(0.0, 10_000.0)
        self._double_hit_window_spin.setValue(self._modal_config.double_hit_window_ms)
        self._double_hit_window_row_label = QLabel(f"{t('modal_double_hit_window_label')}:")
        form.addRow(self._double_hit_window_row_label, self._double_hit_window_spin)

        self._double_hit_threshold_spin = PrecisionDoubleSpinBox()
        self._double_hit_threshold_spin.setRange(0.0, 1.0)
        self._double_hit_threshold_spin.setValue(self._modal_config.double_hit_relative_threshold)
        self._double_hit_threshold_row_label = QLabel(f"{t('modal_double_hit_relative_threshold_label')}:")
        form.addRow(self._double_hit_threshold_row_label, self._double_hit_threshold_spin)

        self._min_rest_time_spin = PrecisionDoubleSpinBox()
        self._min_rest_time_spin.setRange(0.0, 60_000.0)
        self._min_rest_time_spin.setValue(self._modal_config.min_rest_time_ms)
        self._min_rest_time_row_label = QLabel(f"{t('modal_min_rest_time_label')}:")
        form.addRow(self._min_rest_time_row_label, self._min_rest_time_spin)

        self._overload_fraction_spin = PrecisionDoubleSpinBox()
        self._overload_fraction_spin.setRange(0.0, 1.0)
        self._overload_fraction_spin.setValue(self._modal_config.overload_fraction)
        self._overload_fraction_row_label = QLabel(f"{t('modal_overload_fraction_label')}:")
        form.addRow(self._overload_fraction_row_label, self._overload_fraction_spin)

        layout.addWidget(parameter_group)

    def _build_measurement_section(self, layout: QVBoxLayout) -> None:
        # Only the sample rate - mirrors `gui/setup_view.py::SetupView`'s
        # OWN split exactly: there too, "Messeinstellungen" holds just
        # the sample rate, while name/storage format/location all sit
        # under "Speichereinstellungen" (`_build_storage_section`) - both
        # are, after all, statements about how/where the file is
        # written, not about the measurement itself.
        self._measurement_header = QLabel(t("measurement_settings"))
        layout.addWidget(self._measurement_header)
        measurement_group = QGroupBox()
        form = QFormLayout(measurement_group)

        self._sample_rate_spin = PrecisionDoubleSpinBox()
        self._sample_rate_spin.setRange(1.0, 1_000_000.0)
        self._sample_rate_spin.setValue(_DEFAULT_MODAL_SAMPLE_RATE_HZ)
        self._sample_rate_row_label = QLabel(f"{t('sample_rate_hz')}:")
        form.addRow(self._sample_rate_row_label, self._sample_rate_spin)

        layout.addWidget(measurement_group)

    def _build_storage_section(self, layout: QVBoxLayout) -> None:
        self._storage_header = QLabel(t("storage_settings"))
        layout.addWidget(self._storage_header)
        storage_group = QGroupBox()
        form = QFormLayout(storage_group)

        self._name_edit = QLineEdit(self._configuration_manager.settings.last_measurement_name)
        self._name_row_label = QLabel(f"{t('measurement_name')}:")
        form.addRow(self._name_row_label, self._name_edit)

        self._storage_format_combo = QComboBox()
        for storage_format, key in _STORAGE_FORMAT_LABEL_KEYS.items():
            self._storage_format_combo.addItem(t(key), storage_format.value)
        self._set_combo_by_data(
            self._storage_format_combo, self._configuration_manager.settings.default_storage_format
        )
        self._storage_format_row_label = QLabel(f"{t('storage_format')}:")
        form.addRow(self._storage_format_row_label, self._storage_format_combo)

        # Deliberately SEPARATE from the raw-data format above - the
        # exported analysis results (FRF/coherence sidecar, see the
        # design plan's export step) are a different, much smaller file
        # an operator may reasonably want in a different format (e.g.
        # bulky raw Parquet, but a small, human-inspectable CSV result).
        self._result_storage_format_combo = QComboBox()
        for storage_format, key in _STORAGE_FORMAT_LABEL_KEYS.items():
            self._result_storage_format_combo.addItem(t(key), storage_format.value)
        self._set_combo_by_data(
            self._result_storage_format_combo, self._modal_config.result_storage_format.value
        )
        self._result_storage_format_row_label = QLabel(f"{t('modal_result_storage_format_label')}:")
        form.addRow(self._result_storage_format_row_label, self._result_storage_format_combo)

        self._storage_path_label = QLabel(t("no_storage_location"))
        self._storage_button = QPushButton(t("choose_storage_location"))
        self._storage_button.clicked.connect(self.storage_path_requested.emit)
        self._storage_location_row_label = QLabel(f"{t('storage_location')}:")
        form.addRow(self._storage_location_row_label, self._storage_path_label)
        form.addRow("", self._storage_button)
        layout.addWidget(storage_group)

    def _build_start_stop_section(self, layout: QVBoxLayout) -> None:
        # Play/Stop icon pair, same drawing functions as
        # `gui/setup_view.py::SetupView` - only one start button here
        # (modal mode has no separate "live view only" vs. "record"
        # choice, it always both records AND runs the impact-triggered
        # analysis, see `build_current_config`'s `save_to_disk=True`),
        # so Play is the natural icon for "start" rather than Record.
        row = QHBoxLayout()
        self._start_button = QPushButton()
        self._start_button.setIconSize(QSize(24, 24))
        self._start_button.setStyleSheet(action_button_style())
        self._start_button.clicked.connect(self._on_start_clicked)
        row.addWidget(self._start_button)

        self._stop_button = QPushButton()
        self._stop_button.setIconSize(QSize(24, 24))
        self._stop_button.setStyleSheet(action_button_style())
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop_requested.emit)
        row.addWidget(self._stop_button)

        self._retheme_start_stop_button_icons()
        self._update_start_stop_button_labels()
        connect_theme_changed(self._retheme_start_stop_button_icons)
        layout.addLayout(row)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

    def _retheme_start_stop_button_icons(self) -> None:
        """Mirrors `gui/setup_view.py::SetupView._retheme_start_button_icons`
        for the play/stop pair."""
        self._start_button.setIcon(QIcon(draw_play_icon(24, y_offset=0.6, color=PLAY_ICON_COLOR)))
        self._stop_button.setIcon(QIcon(draw_stop_icon(24, y_offset=0.6)))
        for button in (self._start_button, self._stop_button):
            repolish(button)

    def _update_start_stop_button_labels(self) -> None:
        # Own label ("Analyse starten") rather than the standard mode's
        # "Aufnahme" (`record_button_label`) - modal mode always
        # records (see `build_current_config`'s `save_to_disk=True`),
        # but what actually starts is the impact-triggered analysis,
        # not "a recording" in the generic sense the standard label
        # implies.
        self._start_button.setText(f"  {t('modal_start_button_label')}")
        self._start_button.setToolTip(t("modal_start_button_label"))
        self._stop_button.setText(f"  {t('stop_button_label')}")
        self._stop_button.setToolTip(t("stop_measurement"))

    def _apply_section_header_emphasis(self) -> None:
        """Emphasizes only section labels and stays fully theme-safe -
        identical to `gui/setup_view.py::SetupView.
        _apply_section_header_emphasis`, duplicated rather than shared
        so that file stays untouched by this feature (see the module
        docstring)."""
        header_font = QFont(self.font())
        if header_font.pointSize() > 0:
            header_font.setPointSize(header_font.pointSize() + 2)
        header_font.setBold(True)

        for header in (
            self._device_header,
            self._channel_header,
            self._parameter_header,
            self._measurement_header,
            self._storage_header,
        ):
            header.setFont(header_font)
            margins = header.contentsMargins()
            header.setContentsMargins(margins.left(), 8, margins.right(), 4)

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    # ------------------------------------------------------------------ #
    # Device discovery
    # ------------------------------------------------------------------ #

    def set_discovered_devices(self, devices: list[DeviceInfo]) -> None:
        """Renders the discovery result into `_device_list` - the same
        one-line-per-device format as
        `gui/setup_view.py::SetupView.set_discovered_devices` (product
        type, module/connection status, channel count; grayed out when
        offline or unsupported), WITHOUT that method's warning-dialog
        memory: with only 4 fixed channel slots here, a problem device
        is already visible right where it matters (grayed out in this
        tree, and again inline in `HardwareChannelPickerDialog` when
        actually picking a channel) - a modal interruption on every
        affected discovery run would be redundant on top of that.
        """
        self._discovered_devices = devices
        self._device_list.clear()
        devices_with_channels = [d for d in devices if d.num_channels > 0 or d.has_any_channels]
        if not devices_with_channels:
            self._device_list.addTopLevelItem(QTreeWidgetItem([t("no_devices_found")]))
            return
        for device in devices_with_channels:
            if device.connection_probed and not device.is_connected:
                module_info = f" [{t('device_not_connected')}]"
            elif device.module_type is None:
                module_info = f" [{t('device_module_unsupported')}]"
            else:
                module_info = f" [{device.module_type.value}]"
            device_item = QTreeWidgetItem(
                [
                    f"{device.device_name} - {device.product_type}{module_info} "
                    f"({t('device_channel_count', count=device.num_channels)})"
                ]
            )
            channels = device.physical_channels or [
                f"{device.device_name}/ai{i}" for i in range(device.num_channels)
            ]
            for channel in channels:
                device_item.addChild(QTreeWidgetItem([channel]))
            if device.module_type is None or (
                device.connection_probed and not device.is_connected
            ):
                brush = QBrush(disabled_text_color())
                device_item.setForeground(0, brush)
                for i in range(device_item.childCount()):
                    device_item.child(i).setForeground(0, brush)
            self._device_list.addTopLevelItem(device_item)

    def get_discovered_devices(self) -> list[DeviceInfo] | None:
        return self._discovered_devices

    def show_discovery_error(self, message: str) -> None:
        """Mirrors `SetupView.show_discovery_error` - the cause shown
        directly in the device tree rather than only in the log."""
        self._device_list.clear()
        self._discovered_devices = None
        self._device_list.addTopLevelItem(
            QTreeWidgetItem([f"{t('device_discovery_failed')}: {message}"])
        )

    def _find_device_for_channel(self, hardware_channel_id: str) -> DeviceInfo | None:
        for device in self._discovered_devices or []:
            if hardware_channel_id in device.physical_channels:
                return device
        return None

    def _used_channels(self, exclude_role: str) -> set[str]:
        return {
            state.hardware_channel_id
            for role, state in self._channel_states.items()
            if role != exclude_role and state.hardware_channel_id
        }

    # ------------------------------------------------------------------ #
    # Channel assignment
    # ------------------------------------------------------------------ #

    def _on_pick_channel_clicked(self, role: str) -> None:
        state = self._channel_states[role]
        dialog = HardwareChannelPickerDialog(
            self._discovered_devices or [],
            used_channels=self._used_channels(exclude_role=role),
            current_channel=state.hardware_channel_id,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_channel()
        if selected is None:
            return
        state.hardware_channel_id = selected
        self._refresh_channel_row_label(role)

    def _on_edit_parameters_clicked(self, role: str) -> None:
        state = self._channel_states[role]
        dialog = _ModalChannelParameterDialog(
            is_excitation=(role == _ROLE_EXCITATION),
            scale=state.scale,
            offset=state.offset,
            sensitivity_mv_per_unit=state.sensitivity_mv_per_unit,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        state.scale = dialog.scale()
        state.offset = dialog.offset()
        state.sensitivity_mv_per_unit = dialog.sensitivity_mv_per_unit()

    def _on_clear_channel_clicked(self, role: str) -> None:
        self._channel_states[role] = _ChannelRowState()
        self._refresh_channel_row_label(role)

    def _refresh_channel_row_label(self, role: str) -> None:
        hardware_channel_id = self._channel_states[role].hardware_channel_id
        self._channel_labels[role].setText(hardware_channel_id or t("modal_no_channel_assigned"))

    # ------------------------------------------------------------------ #
    # Storage location
    # ------------------------------------------------------------------ #

    def set_storage_path(self, path: str | None) -> None:
        self._storage_path_is_set = bool(path)
        self._storage_path_label.setText(path or t("no_storage_location"))

    # ------------------------------------------------------------------ #
    # Start/stop
    # ------------------------------------------------------------------ #

    def set_start_enabled(self, enabled: bool, reason: str = "") -> None:
        self._start_button.setEnabled(enabled)
        self._stop_button.setEnabled(not enabled)
        if reason:
            self._status_label.setText(t(reason))
        elif enabled:
            self._status_label.setText("")

    def show_error(self, message: str) -> None:
        QMessageBox.warning(self, t("error"), message)

    def _on_start_clicked(self) -> None:
        config = self.build_current_config()
        if config is not None:
            self.start_measurement_requested.emit(config)

    def build_current_config(self) -> MeasurementConfig | None:
        """Builds the `MeasurementConfig` for the currently configured
        channels/parameters, or `None` (after showing an explanatory
        error) if the configuration is incomplete.

        `MeasurementConfig.trigger` is left at its default
        (`TriggerKind.NONE`) - modal mode always records continuously,
        see the class docstring.
        """
        excitation_state = self._channel_states[_ROLE_EXCITATION]
        if not excitation_state.hardware_channel_id:
            self.show_error(t("modal_error_no_excitation_channel"))
            return None

        response_roles_assigned = [
            role for role in _RESPONSE_ROLES if self._channel_states[role].hardware_channel_id
        ]
        if not response_roles_assigned:
            self.show_error(t("modal_error_no_response_channel"))
            return None

        channels: list[Channel] = [self._build_channel(_ROLE_EXCITATION, "N")]
        for role in response_roles_assigned:
            channels.append(self._build_channel(role, "g"))

        return MeasurementConfig(
            name=self._name_edit.text().strip() or "Messung",
            sample_rate_hz=self._sample_rate_spin.value(),
            channels=channels,
            storage_format=StorageFormat(self._storage_format_combo.currentData()),
            save_to_disk=True,
            recording_unlimited=True,
        )

    def _resolved_display_name(self, role: str) -> str:
        """The channel's name/formula symbol - the operator's own text
        if they entered one, otherwise the same axis-derived default
        shown as the field's placeholder (see `_DEFAULT_DISPLAY_NAMES`)."""
        return self._display_name_edits[role].text().strip() or _DEFAULT_DISPLAY_NAMES[role]

    def _build_channel(self, role: str, unit: str) -> Channel:
        state = self._channel_states[role]
        display_name = self._resolved_display_name(role)
        device = self._find_device_for_channel(state.hardware_channel_id)
        # Falls back to NI9234 if the channel's device isn't in the
        # current discovery result (e.g. a saved configuration is
        # loaded before this session has discovered anything yet) -
        # NOT a guess: NI9234 is the ONLY module in this codebase that
        # supports SignalType.IEPE_ACCELERATION at all (see
        # `hardware/ni9234.py`), so it is the only value that could ever
        # be correct here regardless. `core/measurement.py::create_devices`
        # takes `module_type` from the `Channel` object as-is (it does
        # NOT cross-check it against a fresh discovery) - if the
        # assigned hardware channel turns out to sit on different
        # hardware after all, that surfaces as a normal
        # `MeasurementConfigError`/`AcquisitionError` at measurement
        # start, same as anywhere else in the app.
        module_type = device.module_type if device is not None else ModuleType.NI9234
        return Channel(
            hardware_channel=state.hardware_channel_id,
            display_name=display_name,
            unit=unit,
            scale=state.scale,
            offset=state.offset,
            signal_type=SignalType.IEPE_ACCELERATION,
            module_type=module_type,
            sensitivity_mv_per_unit=state.sensitivity_mv_per_unit,
            # Deliberately None, not the Channel-dataclass-wide ±10V
            # default - see `gui/widgets/channel_table.py`'s identical
            # choice: the hardware layer (`hardware/ni9234.py`) then
            # applies its own correct module fallback range instead of
            # a blanket ±10, which the NI9234 hardware hard-rejects.
            min_range=None,
            max_range=None,
        )

    def current_modal_config(self) -> ModalAnalysisConfig:
        """The current `ModalAnalysisConfig` (channel roles + all modal
        parameters) - used by `MainWindow` to persist it via
        `ConfigurationManager.update_last_modal_config` and, later, by
        `ModalLiveView` to drive the impact detector/averager."""
        response_channels = [
            ModalResponseChannel(
                axis=ModalAxis(role),
                hardware_channel_id=self._channel_states[role].hardware_channel_id,
                display_name=self._display_name_edits[role].text().strip(),
            )
            for role in _RESPONSE_ROLES
            if self._channel_states[role].hardware_channel_id
        ]
        return ModalAnalysisConfig(
            excitation_channel_hardware_id=self._channel_states[_ROLE_EXCITATION].hardware_channel_id,
            excitation_axis=ModalAxis(self._excitation_axis_combo.currentData()),
            excitation_display_name=self._display_name_edits[_ROLE_EXCITATION].text().strip(),
            result_storage_format=StorageFormat(self._result_storage_format_combo.currentData()),
            response_channels=response_channels,
            excitation_window=self._excitation_window_combo.currentData(),
            response_window=self._response_window_combo.currentData(),
            frequency_resolution_hz=self._frequency_resolution_spin.value(),
            num_averages_target=self._num_averages_spin.value(),
            estimator=self._estimator_combo.currentData(),
            frf_quantity=self._frf_quantity_combo.currentData(),
            impact_condition=replace(
                self._impact_condition,
                threshold_value=self._impact_threshold_spin.value(),
                threshold_direction=TriggerDirection(self._impact_direction_combo.currentData()),
            ),
            pretrigger_ms=self._pretrigger_spin.value(),
            double_hit_window_ms=self._double_hit_window_spin.value(),
            double_hit_relative_threshold=self._double_hit_threshold_spin.value(),
            min_rest_time_ms=self._min_rest_time_spin.value(),
            overload_fraction=self._overload_fraction_spin.value(),
        )

    def _load_last_modal_config(self) -> None:
        """Seeds `_channel_states`/`_impact_condition` from the last
        used configuration (`AppSettings.last_modal_config`) - called
        BEFORE the widgets exist, the widgets themselves are seeded from
        `self._modal_config` while being built (see
        `_build_parameter_section`)."""
        self._modal_config = ModalAnalysisConfig.from_dict(
            self._configuration_manager.settings.last_modal_config
        )
        self._channel_states[_ROLE_EXCITATION] = _ChannelRowState(
            hardware_channel_id=self._modal_config.excitation_channel_hardware_id,
            sensitivity_mv_per_unit=1.0,
        )
        self._initial_display_names: dict[str, str] = {
            _ROLE_EXCITATION: self._modal_config.excitation_display_name
        }
        for response_channel in self._modal_config.response_channels:
            self._channel_states[response_channel.axis.value] = _ChannelRowState(
                hardware_channel_id=response_channel.hardware_channel_id,
                sensitivity_mv_per_unit=1.0,
            )
            self._initial_display_names[response_channel.axis.value] = response_channel.display_name
        self._impact_condition = self._modal_config.impact_condition

    # ------------------------------------------------------------------ #
    # i18n
    # ------------------------------------------------------------------ #

    def retranslate_ui(self) -> None:
        self._device_header.setText(t("connected_devices"))
        self._discover_button.setText(t("search_devices"))
        self._open_ni_max_button.setText(t("open_ni_max_button"))

        self._channel_header.setText(t("modal_channel_assignment_header"))
        for role, row_header in self._row_header_items.items():
            row_header.setText(self._role_label_text(role))
        self._retranslate_channel_table_header()
        for role in self._channel_labels:
            self._refresh_channel_row_label(role)

        self._parameter_header.setText(t("modal_parameters_header"))
        self._excitation_window_row_label.setText(f"{t('modal_excitation_window_label')}:")
        self._response_window_row_label.setText(f"{t('modal_response_window_label')}:")
        self._frequency_resolution_row_label.setText(f"{t('modal_frequency_resolution_label')}:")
        self._num_averages_row_label.setText(f"{t('modal_num_averages_label')}:")
        self._estimator_row_label.setText(f"{t('modal_estimator_label')}:")
        self._frf_quantity_row_label.setText(f"{t('modal_frf_quantity_label')}:")
        self._impact_threshold_row_label.setText(f"{t('modal_impact_threshold_label')}:")
        self._pretrigger_row_label.setText(f"{t('modal_pretrigger_label')}:")
        self._double_hit_window_row_label.setText(f"{t('modal_double_hit_window_label')}:")
        self._double_hit_threshold_row_label.setText(f"{t('modal_double_hit_relative_threshold_label')}:")
        self._min_rest_time_row_label.setText(f"{t('modal_min_rest_time_label')}:")
        self._overload_fraction_row_label.setText(f"{t('modal_overload_fraction_label')}:")
        self._retranslate_combo(self._excitation_window_combo, _EXCITATION_WINDOW_LABEL_KEYS)
        self._retranslate_combo(self._response_window_combo, _RESPONSE_WINDOW_LABEL_KEYS)
        self._retranslate_combo(self._estimator_combo, _ESTIMATOR_LABEL_KEYS)
        self._retranslate_combo(self._frf_quantity_combo, _FRF_QUANTITY_LABEL_KEYS)
        self._retranslate_direction_combo()

        self._measurement_header.setText(t("measurement_settings"))
        self._sample_rate_row_label.setText(f"{t('sample_rate_hz')}:")

        self._storage_header.setText(t("storage_settings"))
        self._name_row_label.setText(f"{t('measurement_name')}:")
        self._storage_format_row_label.setText(f"{t('storage_format')}:")
        self._retranslate_storage_format_combo(self._storage_format_combo)
        self._result_storage_format_row_label.setText(f"{t('modal_result_storage_format_label')}:")
        self._retranslate_storage_format_combo(self._result_storage_format_combo)
        self._storage_location_row_label.setText(f"{t('storage_location')}:")
        self._storage_button.setText(t("choose_storage_location"))
        if not self._storage_path_is_set:
            self._storage_path_label.setText(t("no_storage_location"))

        self._update_start_stop_button_labels()

    @staticmethod
    def _retranslate_combo(combo: QComboBox, label_keys: dict) -> None:
        current_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for value, key in label_keys.items():
            combo.addItem(t(key), value)
        ModalSetupView._set_combo_by_data(combo, current_data)
        combo.blockSignals(False)

    def _retranslate_direction_combo(self) -> None:
        current_data = self._impact_direction_combo.currentData()
        self._impact_direction_combo.blockSignals(True)
        self._impact_direction_combo.clear()
        for direction in TriggerDirection:
            self._impact_direction_combo.addItem(
                t(_TRIGGER_DIRECTION_LABEL_KEYS[direction]), direction.value
            )
        self._set_combo_by_data(self._impact_direction_combo, current_data)
        self._impact_direction_combo.blockSignals(False)

    @staticmethod
    def _retranslate_storage_format_combo(combo: QComboBox) -> None:
        current_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for storage_format, key in _STORAGE_FORMAT_LABEL_KEYS.items():
            combo.addItem(t(key), storage_format.value)
        ModalSetupView._set_combo_by_data(combo, current_data)
        combo.blockSignals(False)
