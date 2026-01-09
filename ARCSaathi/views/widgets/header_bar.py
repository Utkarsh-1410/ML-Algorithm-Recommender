"""Header / Top navigation bar widget."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)


class HeaderBar(QWidget):
    project_name_changed = Signal(str)
    mode_changed = Signal(str)

    help_clicked = Signal()
    settings_clicked = Signal()
    export_clicked = Signal()
    profile_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("HeaderBar")

        self.lbl_logo = QLabel("ARCSaathi")
        self.lbl_logo.setObjectName("AppLogo")

        self.txt_project = QLineEdit("Untitled Project")
        self.txt_project.setObjectName("ProjectName")
        self.txt_project.setClearButtonEnabled(True)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["General ML", "Automobile Analytics"])
        self.cmb_mode.setObjectName("ModeSelector")

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setObjectName("StatusIndicator")
        self.set_status("Ready")

        self.btn_help = QPushButton("Help/Docs")
        self.btn_settings = QPushButton("Settings")
        self.btn_export = QPushButton("Export Project")
        self.btn_profile = QPushButton("User Profile")

        for b in (self.btn_help, self.btn_settings, self.btn_export, self.btn_profile):
            b.setObjectName("HeaderButton")

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        layout.addWidget(self.lbl_logo)
        layout.addWidget(QLabel("Project:"))
        layout.addWidget(self.txt_project, 1)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.cmb_mode)
        layout.addStretch(1)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.btn_help)
        layout.addWidget(self.btn_settings)
        layout.addWidget(self.btn_export)
        layout.addWidget(self.btn_profile)
        self.setLayout(layout)

        self.txt_project.editingFinished.connect(self._emit_project_name)
        self.cmb_mode.currentTextChanged.connect(self.mode_changed)
        self.btn_help.clicked.connect(self.help_clicked)
        self.btn_settings.clicked.connect(self.settings_clicked)
        self.btn_export.clicked.connect(self.export_clicked)
        self.btn_profile.clicked.connect(self.profile_clicked)

    def _emit_project_name(self) -> None:
        self.project_name_changed.emit(self.txt_project.text().strip())

    def set_status(self, state: str) -> None:
        """Set status text and styling state: Ready / Processing / Error."""
        state_norm = (state or "Ready").strip().lower()
        if state_norm.startswith("proc"):
            text = "Processing"
            css_state = "processing"
        elif state_norm.startswith("err"):
            text = "Error"
            css_state = "error"
        else:
            text = "Ready"
            css_state = "ready"

        self.lbl_status.setText(text)
        self.lbl_status.setProperty("status", css_state)
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)
