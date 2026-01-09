"""Base view classes."""

from PySide6.QtWidgets import QWidget


class BaseView(QWidget):
    """Base QWidget for all views."""

    def __init__(self, parent=None):
        super().__init__(parent)
