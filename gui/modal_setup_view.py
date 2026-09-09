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

This file currently holds only the placeholder shown while the mode
switch (menu entry, workspace stack wiring) is being built - the real
channel-role pickers and `data.models.ModalAnalysisConfig` form follow
in a later step.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.i18n import connect_language_changed, t


class ModalSetupView(QWidget):
    """Placeholder for the modal analysis configuration view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self._placeholder_label = QLabel(t("modal_setup_placeholder"))
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setWordWrap(True)
        layout.addWidget(self._placeholder_label)

        connect_language_changed(self.retranslate_ui)

    def retranslate_ui(self) -> None:
        self._placeholder_label.setText(t("modal_setup_placeholder"))
