"""
gui/modal_live_view.py

Live view for the experimental modal analysis mode (impact hammer +
accelerometer) - see `gui/main_window.py::MainWindow` for how this
replaces `gui/live_view.py::LiveView` in the workspace stack while
"Modalanalyse" mode is active.

Deliberately a separate, dedicated class rather than an extension of
`LiveView`: the standard live view is already very large (3000+ lines)
and built around continuous scrolling display, whereas modal analysis
needs a fundamentally different interaction model (block capture per
hammer strike, running averages per response axis, undo/redo of
individual strikes) - see the design plan for the full rationale.
`gui/live_view.py` is not touched by this file at all.

Architecture, per GUI tick (`_UI_UPDATE_INTERVAL_MS`, same rate as
`LiveView`):
    1. Read new raw samples via the normal live-reader API
       (`MeasurementController.read_live_data`, unchanged - the same
       mechanism `LiveView` uses).
    2. Feed the excitation channel's samples to `ImpactDetector.feed()`
       - it scans the WHOLE block for a hammer strike, not just the
       last sample (see `core/impact_detector.py`'s module docstring
       for why that distinction matters).
    3. Once `ImpactDetector.is_ready_to_capture()` says enough
       post-trigger samples exist, register a THROWAWAY, RETROACTIVE
       reader (`MeasurementController.register_reader(back_samples=...)`)
       to extract the exact pretrigger+posttrigger window, then
       immediately unregister it - the same pre-roll mechanism the
       standard recording trigger already uses
       (`gui/main_window.py::_on_trigger_fired`).
    4. Check the window for a double hit / overload
       (`ImpactDetector.check_double_hit`/`check_overload`); on a
       double hit, ask the operator whether to retry or skip (a real
       dialog, not a silent rejection - see `_handle_double_hit`).
    5. On acceptance, feed the window into ONE `ModalAverager` PER
       response axis (all fed the SAME excitation block - see
       `data/models.py::ModalAnalysisConfig`'s docstring on why
       `ImpactDetector` itself never needs to know how many response
       axes exist) and redraw.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from analysis.modal_analysis import ModalAverager, resolve_block_size
from core.controller import MeasurementController
from core.impact_detector import ImpactDetector
from core.measurement import apply_scaling
from data.models import Channel, ModalAnalysisConfig
from gui.i18n import connect_language_changed, t
from gui.modal_setup_view import _FRF_QUANTITY_LABEL_KEYS
from gui.theme import (
    axis_tick_point_size,
    channel_curve_color,
    connect_theme_changed,
    plot_background_color,
    style_plot_container,
    style_plot_item,
)

# Same tick rate as `gui/live_view.py::_UI_UPDATE_INTERVAL_MS` -
# duplicated rather than imported, see that module's docstring
# guarantee that it stays untouched by this feature.
_UI_UPDATE_INTERVAL_MS = 15

# Fallback measurement range (in g, symmetric) used for the overload
# check when a channel's own `max_range` is `None` - mirrors
# `hardware/ni9234.py::NI9234_MIN_VOLTAGE`/`MAX_VOLTAGE` (±5), which is
# exactly the fallback the hardware layer itself applies in that case
# (see `gui/modal_setup_view.py`'s identical min_range=None/
# max_range=None choice). Kept as an independent constant rather than
# importing from `hardware/` - this is a SOFTWARE safety check on
# already-scaled values, unrelated to the DAQmx task's own input-range
# configuration, so `gui/` has no need to depend on `hardware/` for it.
_FALLBACK_CHANNEL_RANGE = 5.0

_ROW_STATUS_LABEL_KEYS = {
    "pending": "modal_row_pending",
    "captured": "modal_row_captured",
    "skipped": "modal_row_skipped",
}


def _axis_label_style() -> dict[str, str]:
    """Mirrors `gui/live_view.py::_axis_label_style` - see there for why
    the font size must be set before `style_plot_item()`."""
    return {"font-size": f"{axis_tick_point_size()}pt"}


class ModalLiveView(QWidget):
    """Live view for the experimental modal analysis mode.

    Signals:
        stop_requested: User wants to stop the running measurement -
            same meaning as `gui/live_view.py::LiveView.stop_requested`,
            offered here too so the operator does not have to switch
            back to the (hidden) Setup page just to stop.
    """

    stop_requested = pyqtSignal()

    def __init__(self, controller: MeasurementController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller

        self._modal_config: Optional[ModalAnalysisConfig] = None
        self._detector: Optional[ImpactDetector] = None
        self._averagers: dict[str, ModalAverager] = {}
        self._reader_id: Optional[int] = None
        self._block_size: int = 0
        self._excitation_row_index: int = 0
        self._excitation_range: float = _FALLBACK_CHANNEL_RANGE
        self._response_row_indices: dict[str, int] = {}
        self._response_ranges: dict[str, float] = {}
        self._current_axis: Optional[str] = None
        # One entry per FILLED table row, in order - "captured" or
        # "skipped" - so `_on_undo_clicked` knows whether the row it is
        # about to reopen ever added anything to the averagers (see the
        # class-level note on `_on_undo_clicked`).
        self._row_kinds: list[str] = []
        # Guards the entire per-tick processing pipeline while the
        # double-hit dialog is open (see `_handle_double_hit`) - a modal
        # `QMessageBox.exec()` still pumps this widget's own QTimer
        # (Qt's exec() runs a nested event loop, it does not suspend
        # timers), so a strike happening WHILE the operator is being
        # asked "retry or skip?" must not be silently processed too.
        self._awaiting_double_hit_decision = False

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

        connect_language_changed(self.retranslate_ui)
        connect_theme_changed(self._retheme_plots)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)

        plot_column = QVBoxLayout()
        plot_column.addLayout(self._build_plot_header())
        self._plot_widget = pg.GraphicsLayoutWidget()
        style_plot_container(self._plot_widget)
        plot_column.addWidget(self._plot_widget, stretch=1)
        root_layout.addLayout(plot_column, stretch=1)

        self._excitation_plot = self._plot_widget.addPlot(row=0, col=0)
        self._response_plot = self._plot_widget.addPlot(row=0, col=1)
        self._amplitude_plot = self._plot_widget.addPlot(row=1, col=0, colspan=2)
        self._phase_plot = self._plot_widget.addPlot(row=2, col=0, colspan=2)
        self._coherence_plot = self._plot_widget.addPlot(row=3, col=0, colspan=2)
        # The operator reads amplitude, phase and coherence together at
        # the same frequency - `LiveView` deliberately does NOT link its
        # per-channel time plots (each channel keeps its own time
        # window), but that reasoning does not apply here: these three
        # panels always share the same frequency axis by construction.
        self._phase_plot.setXLink(self._amplitude_plot)
        self._coherence_plot.setXLink(self._amplitude_plot)

        for plot_item in (
            self._excitation_plot,
            self._response_plot,
            self._amplitude_plot,
            self._phase_plot,
            self._coherence_plot,
        ):
            plot_item.showGrid(x=True, y=True, alpha=0.3)

        self._excitation_curve = self._excitation_plot.plot(pen=pg.mkPen(channel_curve_color(0), width=1.2))
        self._response_curve = self._response_plot.plot(pen=pg.mkPen(channel_curve_color(1), width=1.2))
        self._amplitude_curve = self._amplitude_plot.plot(pen=pg.mkPen(channel_curve_color(2), width=1.2))
        self._phase_curve = self._phase_plot.plot(pen=pg.mkPen(channel_curve_color(2), width=1.2))
        self._coherence_curve = self._coherence_plot.plot(pen=pg.mkPen(channel_curve_color(3), width=1.2))
        self._phase_plot.setYRange(-180.0, 180.0, padding=0)
        self._coherence_plot.setYRange(0.0, 1.0, padding=0.05)

        self._retheme_plots()
        self._retranslate_plot_labels()

        side_column = QVBoxLayout()
        side_column.setSpacing(10)

        self._progress_label = QLabel("")
        font = self._progress_label.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        self._progress_label.setFont(font)
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_column.addWidget(self._progress_label)

        self._hit_table = QTableWidget(0, 2)
        self._hit_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._hit_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._hit_table.verticalHeader().setVisible(False)
        side_column.addWidget(self._hit_table, stretch=1)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        side_column.addWidget(self._status_label)

        self._undo_button = QPushButton(t("modal_undo_button"))
        self._undo_button.clicked.connect(self._on_undo_clicked)
        side_column.addWidget(self._undo_button)

        self._reset_button = QPushButton(t("modal_reset_button"))
        self._reset_button.clicked.connect(self._on_reset_clicked)
        side_column.addWidget(self._reset_button)

        self._stop_button = QPushButton(t("stop_measurement"))
        self._stop_button.clicked.connect(self.stop_requested.emit)
        side_column.addWidget(self._stop_button)

        side_container = QWidget()
        side_container.setLayout(side_column)
        side_container.setFixedWidth(260)
        root_layout.addWidget(side_container)

        self._retranslate_hit_table_header()

    def _build_plot_header(self) -> QHBoxLayout:
        header = QHBoxLayout()

        self._axis_label = QLabel(t("modal_axis_selector_label"))
        header.addWidget(self._axis_label)
        self._axis_combo = QComboBox()
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        header.addWidget(self._axis_combo)

        header.addStretch(1)

        self._frf_quantity_label = QLabel(t("modal_frf_quantity_label"))
        header.addWidget(self._frf_quantity_label)
        self._frf_quantity_combo = QComboBox()
        for value, key in _FRF_QUANTITY_LABEL_KEYS.items():
            self._frf_quantity_combo.addItem(t(key), value)
        self._frf_quantity_combo.currentIndexChanged.connect(self._redraw_frf_plots)
        header.addWidget(self._frf_quantity_combo)

        self._db_checkbox = QPushButton(t("modal_db_toggle"))
        self._db_checkbox.setCheckable(True)
        self._db_checkbox.toggled.connect(self._redraw_frf_plots)
        header.addWidget(self._db_checkbox)

        return header

    # ------------------------------------------------------------------ #
    # Start/stop
    # ------------------------------------------------------------------ #

    def start_display(self, modal_config: ModalAnalysisConfig, sample_rate_hz: float) -> None:
        """Begins the impact-triggered live display for a running modal
        measurement - called once by `MainWindow` right after
        `MeasurementController.start_measurement()` succeeds.

        Args:
            modal_config: The configuration `ModalSetupView` was showing
                when the measurement was started (see
                `ModalSetupView.current_modal_config`).
            sample_rate_hz: The EFFECTIVE tick rate of the running
                measurement (`resolve_rate_groups(...).
                resolved_sample_rate_hz`, not the raw requested rate -
                same distinction `LiveView.start_display` makes), since
                that is what the FFT block size must be derived from.
        """
        self._modal_config = modal_config
        channels = self._controller.active_channels
        hardware_id_to_index = {channel.hardware_channel: i for i, channel in enumerate(channels)}

        self._excitation_row_index = hardware_id_to_index[modal_config.excitation_channel_hardware_id]
        self._excitation_range = self._resolve_range(channels[self._excitation_row_index])

        self._response_row_indices = {}
        self._response_ranges = {}
        for response_channel in modal_config.response_channels:
            index = hardware_id_to_index[response_channel.hardware_channel_id]
            axis = response_channel.axis.value
            self._response_row_indices[axis] = index
            self._response_ranges[axis] = self._resolve_range(channels[index])

        self._block_size = resolve_block_size(sample_rate_hz, modal_config.frequency_resolution_hz)
        self._detector = ImpactDetector(modal_config, sample_rate_hz, self._block_size)
        self._averagers = {
            axis: ModalAverager(
                self._block_size,
                sample_rate_hz,
                modal_config.excitation_window,
                modal_config.response_window,
                modal_config.estimator,
            )
            for axis in self._response_row_indices
        }
        self._current_axis = next(iter(self._response_row_indices))
        self._row_kinds = []
        self._awaiting_double_hit_decision = False

        self._populate_axis_combo()
        self._build_hit_table_rows(modal_config.num_averages_target)
        self._clear_plots()
        self._update_progress_label()
        self._status_label.setText("")

        self._reader_id = self._controller.register_reader()
        self._timer.start(_UI_UPDATE_INTERVAL_MS)

    def stop_display(self) -> None:
        """Ends the live display (measurement has stopped) - the last
        captured signals/FRF stay on screen for review; only the
        capture pipeline itself shuts down."""
        self._timer.stop()
        if self._reader_id is not None:
            self._controller.unregister_reader(self._reader_id)
            self._reader_id = None
        self._detector = None

    @staticmethod
    def _resolve_range(channel: Channel) -> float:
        return abs(channel.max_range) if channel.max_range is not None else _FALLBACK_CHANNEL_RANGE

    # ------------------------------------------------------------------ #
    # Per-tick capture pipeline
    # ------------------------------------------------------------------ #

    def _on_timer_tick(self) -> None:
        if self._reader_id is None or self._awaiting_double_hit_decision or self._detector is None:
            return
        if self._modal_config is not None and self._is_fully_captured():
            return

        raw = self._controller.read_live_data(self._reader_id)
        if raw.size == 0:
            return
        current_total = self._controller.total_samples_acquired
        block_start_index = current_total - raw.shape[1]
        scaled = apply_scaling(raw, self._controller.active_channels)

        self._detector.feed(scaled[self._excitation_row_index], block_start_index)

        if self._detector.is_ready_to_capture(current_total):
            self._extract_and_process_capture()

    def _is_fully_captured(self) -> bool:
        """Whether every row of the hit table has been filled (captured
        or skipped) - detection then pauses rather than silently
        continuing to average strikes with no table row left to show
        them in (see the design plan on why the table has a fixed
        number of rows, `num_averages_target`)."""
        return self._modal_config is not None and len(self._row_kinds) >= self._modal_config.num_averages_target

    def _extract_and_process_capture(self) -> None:
        window_start_index = self._detector.window_start_index()
        current_total = self._controller.total_samples_acquired
        back_samples = max(0, current_total - window_start_index)

        reader_id = self._controller.register_reader(back_samples=back_samples)
        try:
            raw_window = self._controller.read_live_data(reader_id, max_samples=self._block_size)
        finally:
            self._controller.unregister_reader(reader_id)

        if raw_window.shape[1] < self._block_size:
            # Not enough history was actually available (e.g. the
            # requested pretrigger exceeded the ring buffer's capacity) -
            # nothing usable to average; re-arm without touching the
            # hit table.
            self._detector.finish_capture()
            return

        scaled_window = apply_scaling(raw_window, self._controller.active_channels)
        excitation_window = scaled_window[self._excitation_row_index]
        response_windows = {
            axis: scaled_window[index] for axis, index in self._response_row_indices.items()
        }

        if any(
            self._detector.check_overload(
                excitation_window, response_windows[axis], self._excitation_range, self._response_ranges[axis]
            )
            for axis in response_windows
        ):
            self._detector.finish_capture()
            self._status_label.setText(t("modal_overload_rejected"))
            return

        if self._detector.check_double_hit(excitation_window):
            self._handle_double_hit()
            self._detector.finish_capture()
            return

        self._accept_capture(excitation_window, response_windows)
        self._detector.finish_capture()

    def _handle_double_hit(self) -> None:
        """Asks the operator whether to retry the same table slot or
        give up on it - see the module/class docstrings for why this is
        a real dialog rather than a silent rejection, and why
        `_awaiting_double_hit_decision` guards the tick handler while it
        is open."""
        self._awaiting_double_hit_decision = True
        try:
            dialog = QMessageBox(self)
            dialog.setWindowTitle(t("modal_double_hit_title"))
            dialog.setText(t("modal_double_hit_body"))
            retry_button = dialog.addButton(t("modal_double_hit_retry"), QMessageBox.ButtonRole.AcceptRole)
            skip_button = dialog.addButton(t("modal_double_hit_skip"), QMessageBox.ButtonRole.DestructiveRole)
            dialog.setDefaultButton(retry_button)
            dialog.exec()
            if dialog.clickedButton() is skip_button:
                self._row_kinds.append("skipped")
                self._set_row_status(len(self._row_kinds) - 1, "skipped")
                self._update_progress_label()
            # "retry": nothing recorded - the same table slot stays
            # "pending", the very next accepted (or skipped) strike
            # fills it.
        finally:
            self._awaiting_double_hit_decision = False

    def _accept_capture(self, excitation_window: np.ndarray, response_windows: dict[str, np.ndarray]) -> None:
        for axis, averager in self._averagers.items():
            averager.add_impact(excitation_window, response_windows[axis])
        self._row_kinds.append("captured")
        self._set_row_status(len(self._row_kinds) - 1, "captured")
        self._update_progress_label()
        self._redraw_time_signals(excitation_window, response_windows.get(self._current_axis))
        self._redraw_frf_plots()

    # ------------------------------------------------------------------ #
    # Undo / reset
    # ------------------------------------------------------------------ #

    def _on_undo_clicked(self) -> None:
        """Undoes whatever happened at the most recently filled table
        row - a capture (via `ModalAverager.undo_last()`, exact per the
        class's own guarantee) or a skip (nothing to undo in the
        averagers, the row simply reopens). Deliberately the MOST
        RECENT row regardless of kind, not specifically the last
        CAPTURED one - simpler to reason about (exactly one counter),
        and matches what an operator pressing "undo" right after
        something just happened actually means."""
        if not self._row_kinds:
            return
        kind = self._row_kinds.pop()
        if kind == "captured":
            for averager in self._averagers.values():
                averager.undo_last()
            self._redraw_frf_plots()
        self._set_row_status(len(self._row_kinds), "pending")
        self._update_progress_label()

    def _on_reset_clicked(self) -> None:
        for averager in self._averagers.values():
            averager.reset()
        self._row_kinds = []
        if self._detector is not None:
            self._detector.reset()
        if self._modal_config is not None:
            self._build_hit_table_rows(self._modal_config.num_averages_target)
        self._clear_plots()
        self._update_progress_label()
        self._status_label.setText("")

    # ------------------------------------------------------------------ #
    # Hit table / progress
    # ------------------------------------------------------------------ #

    def _build_hit_table_rows(self, num_averages_target: int) -> None:
        self._hit_table.setRowCount(num_averages_target)
        for row in range(num_averages_target):
            number_item = QTableWidgetItem(str(row + 1))
            number_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._hit_table.setItem(row, 0, number_item)
            self._hit_table.setItem(row, 1, QTableWidgetItem(t(_ROW_STATUS_LABEL_KEYS["pending"])))

    def _set_row_status(self, row: int, status: str) -> None:
        item = self._hit_table.item(row, 1)
        if item is not None:
            item.setText(t(_ROW_STATUS_LABEL_KEYS[status]))

    def _update_progress_label(self) -> None:
        target = self._modal_config.num_averages_target if self._modal_config is not None else 0
        captured = sum(1 for kind in self._row_kinds if kind == "captured")
        self._progress_label.setText(f"{captured} / {target}")

    # ------------------------------------------------------------------ #
    # Axis selection
    # ------------------------------------------------------------------ #

    def _populate_axis_combo(self) -> None:
        self._axis_combo.blockSignals(True)
        self._axis_combo.clear()
        for axis in self._response_row_indices:
            self._axis_combo.addItem(t(f"modal_response_{axis}_label"), axis)
        # A single configured axis needs no picker - avoids an
        # interactive-looking control with nothing to switch between.
        self._axis_combo.setEnabled(len(self._response_row_indices) > 1)
        self._axis_combo.blockSignals(False)
        if self._response_row_indices:
            self._current_axis = next(iter(self._response_row_indices))

    def _on_axis_changed(self, index: int) -> None:
        if index < 0:
            return
        self._current_axis = self._axis_combo.itemData(index)
        self._redraw_frf_plots()
        self._response_curve.setData([], [])

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def _clear_plots(self) -> None:
        for curve in (
            self._excitation_curve,
            self._response_curve,
            self._amplitude_curve,
            self._phase_curve,
            self._coherence_curve,
        ):
            curve.setData([], [])

    def _redraw_time_signals(self, excitation_window: np.ndarray, response_window: Optional[np.ndarray]) -> None:
        sample_rate_hz = self._averager_sample_rate_hz()
        time_axis = np.arange(len(excitation_window)) / sample_rate_hz
        self._excitation_curve.setData(time_axis, excitation_window)
        if response_window is not None:
            self._response_curve.setData(time_axis, response_window)

    def _averager_sample_rate_hz(self) -> float:
        averager = next(iter(self._averagers.values()), None)
        if averager is None:
            return 1.0
        frequency_hz = averager.frequency_hz()
        # Nyquist * 2 = sample rate; frequency_hz's last bin IS Nyquist.
        return float(frequency_hz[-1] * 2.0) if len(frequency_hz) else 1.0

    def _redraw_frf_plots(self) -> None:
        if self._current_axis is None or self._current_axis not in self._averagers:
            return
        averager = self._averagers[self._current_axis]
        if averager.num_averages == 0:
            return

        quantity = self._frf_quantity_combo.currentData() or "accelerance"
        frf = averager.frf(quantity)
        frequency_hz = averager.frequency_hz()
        coherence = averager.coherence()

        magnitude = np.abs(frf)
        if self._db_checkbox.isChecked():
            with np.errstate(divide="ignore"):
                magnitude = 20.0 * np.log10(np.where(magnitude > 0, magnitude, np.nan))
        phase_deg = np.degrees(np.angle(frf))

        # Bin 0 is NaN for mobility/receptance (see
        # `ModalAverager.frf`'s own DC guard) - pyqtgraph simply skips
        # NaN points, no separate handling needed here.
        self._amplitude_curve.setData(frequency_hz, magnitude)
        self._phase_curve.setData(frequency_hz, phase_deg)
        self._coherence_curve.setData(frequency_hz, coherence)

    def _retheme_plots(self) -> None:
        style_plot_container(self._plot_widget)
        background = plot_background_color()
        for plot_item in (
            self._excitation_plot,
            self._response_plot,
            self._amplitude_plot,
            self._phase_plot,
            self._coherence_plot,
        ):
            style_plot_item(plot_item)
            plot_item.getViewBox().setBackgroundColor(background)

    # ------------------------------------------------------------------ #
    # i18n
    # ------------------------------------------------------------------ #

    def _retranslate_plot_labels(self) -> None:
        label_style = _axis_label_style()
        self._excitation_plot.setLabel("bottom", f"{t('axis_time')} [s]", **label_style)
        self._excitation_plot.setLabel("left", t("modal_excitation_label"), **label_style)
        self._response_plot.setLabel("bottom", f"{t('axis_time')} [s]", **label_style)
        self._response_plot.setLabel("left", t("modal_response_plot_label"), **label_style)
        self._amplitude_plot.setLabel("bottom", f"{t('axis_frequency')} [Hz]", **label_style)
        self._amplitude_plot.setLabel("left", t("modal_amplitude_axis_label"), **label_style)
        self._phase_plot.setLabel("bottom", f"{t('axis_frequency')} [Hz]", **label_style)
        self._phase_plot.setLabel("left", t("modal_phase_axis_label"), **label_style)
        self._coherence_plot.setLabel("bottom", f"{t('axis_frequency')} [Hz]", **label_style)
        self._coherence_plot.setLabel("left", t("modal_coherence_axis_label"), **label_style)
        for plot_item in (
            self._excitation_plot,
            self._response_plot,
            self._amplitude_plot,
            self._phase_plot,
            self._coherence_plot,
        ):
            style_plot_item(plot_item)

    def _retranslate_hit_table_header(self) -> None:
        self._hit_table.setHorizontalHeaderLabels([t("modal_hit_table_number"), t("modal_hit_table_status")])

    def retranslate_ui(self) -> None:
        self._axis_label.setText(t("modal_axis_selector_label"))
        current_axis_data = self._axis_combo.currentData()
        self._axis_combo.blockSignals(True)
        self._axis_combo.clear()
        for axis in self._response_row_indices:
            self._axis_combo.addItem(t(f"modal_response_{axis}_label"), axis)
        if current_axis_data is not None:
            index = self._axis_combo.findData(current_axis_data)
            self._axis_combo.setCurrentIndex(index if index >= 0 else 0)
        self._axis_combo.blockSignals(False)

        self._frf_quantity_label.setText(t("modal_frf_quantity_label"))
        current_quantity = self._frf_quantity_combo.currentData()
        self._frf_quantity_combo.blockSignals(True)
        self._frf_quantity_combo.clear()
        for value, key in _FRF_QUANTITY_LABEL_KEYS.items():
            self._frf_quantity_combo.addItem(t(key), value)
        index = self._frf_quantity_combo.findData(current_quantity)
        self._frf_quantity_combo.setCurrentIndex(index if index >= 0 else 0)
        self._frf_quantity_combo.blockSignals(False)

        self._db_checkbox.setText(t("modal_db_toggle"))
        self._undo_button.setText(t("modal_undo_button"))
        self._reset_button.setText(t("modal_reset_button"))
        self._stop_button.setText(t("stop_measurement"))

        self._retranslate_plot_labels()
        self._retranslate_hit_table_header()
        for row, kind in enumerate(self._row_kinds):
            self._set_row_status(row, kind)
        for row in range(len(self._row_kinds), self._hit_table.rowCount()):
            self._set_row_status(row, "pending")
