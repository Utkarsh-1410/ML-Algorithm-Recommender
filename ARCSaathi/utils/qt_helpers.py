"""Qt helper utilities (message boxes, etc.)."""

from PySide6.QtWidgets import QMessageBox, QWidget


def show_error(parent: QWidget | None, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)


def show_info(parent: QWidget | None, title: str, message: str) -> None:
    QMessageBox.information(parent, title, message)


def show_confirm(parent: QWidget | None, title: str, message: str) -> bool:
    res = QMessageBox.question(parent, title, message, QMessageBox.Yes | QMessageBox.No)
    return res == QMessageBox.Yes
