"""Wrapper that provides a consistent tab header (title, help, actions toolbar)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class TabPage(QWidget):
    help_clicked = Signal()

    def __init__(self, title: str, inner: QWidget, parent=None):
        super().__init__(parent)
        self.setObjectName("TabPage")

        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("TabTitle")

        self.btn_help = QPushButton("Help")
        self.btn_help.setObjectName("TabHelpButton")

        self.toolbar = QToolBar()
        self.toolbar.setObjectName("TabToolbar")
        self.toolbar.setMovable(False)

        header = QHBoxLayout()
        header.setContentsMargins(10, 10, 10, 6)
        header.setSpacing(10)
        header.addWidget(self.lbl_title)
        header.addStretch(1)
        header.addWidget(self.btn_help)
        header.addWidget(self.toolbar)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(inner, 1)
        self.setLayout(layout)

        self.btn_help.clicked.connect(self.help_clicked)
