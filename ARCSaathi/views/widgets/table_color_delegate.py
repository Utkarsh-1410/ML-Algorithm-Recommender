"""Table delegate for lightweight color coding of best/worst values."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QStyledItemDelegate


class BestWorstColorDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._best_rows: set[int] = set()
        self._worst_rows: set[int] = set()
        self._metric_col: int = -1

    def set_best_worst(self, *, metric_col: int, best_rows: set[int], worst_rows: set[int]) -> None:
        self._metric_col = int(metric_col)
        self._best_rows = set(best_rows or set())
        self._worst_rows = set(worst_rows or set())

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.column() != self._metric_col:
            return
        r = int(index.row())
        if r in self._best_rows:
            option.backgroundBrush = QColor(198, 239, 206)  # soft green
            option.font.setBold(True)
        elif r in self._worst_rows:
            option.backgroundBrush = QColor(255, 199, 206)  # soft red

        option.displayAlignment = Qt.AlignCenter
