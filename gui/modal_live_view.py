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
hammer strike, running averages, undo/redo of individual strikes) - see
the design plan for the full rationale.

This file currently holds only the placeholder shown while the mode
switch (menu entry, workspace stack wiring) is being built - the
2x2 plot grid, impact detection and averaging wiring follow in a later
step.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.i18n import connect_language_changed, t


class ModalLiveView(QWidget):
    """Placeholder for the modal analysis live view."""

    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller

        layout = QVBoxLayout(self)
        self._placeholder_label = QLabel(t("modal_live_placeholder"))
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setWordWrap(True)
        layout.addWidget(self._placeholder_label)

        connect_language_changed(self.retranslate_ui)

    def retranslate_ui(self) -> None:
        self._placeholder_label.setText(t("modal_live_placeholder"))
