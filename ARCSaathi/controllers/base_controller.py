"""Base controller classes."""

from PySide6.QtCore import QObject


class BaseController(QObject):
    """Base QObject controller."""

    def __init__(self, parent=None):
        super().__init__(parent)
