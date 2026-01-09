"""Reusable Matplotlib canvas widget for PySide6."""

from __future__ import annotations

import io

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QWidget, QVBoxLayout


class _RenderProxy:
    def __init__(self, owner: "MatplotlibCanvas"):
        self._owner = owner

    def draw_idle(self) -> None:
        self._owner._render_to_label()

    def draw(self) -> None:
        self._owner._render_to_label()


class MatplotlibCanvas(QWidget):
    def __init__(self, parent=None, *, width: float = 5, height: float = 3.5, dpi: int = 100):
        super().__init__(parent)

        # NOTE: We intentionally avoid embedding Matplotlib's QtAgg canvas here because it can
        # cause native crashes on some Windows setups (heap corruption). Instead we render via
        # Agg into a PNG and show it in a QLabel. This keeps plots enabled while remaining stable.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self._agg_canvas = FigureCanvasAgg(self.figure)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumHeight(140)
        self.canvas = _RenderProxy(self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self.setLayout(layout)

        self._render_to_label()

    def _render_to_label(self) -> None:
        try:
            self._agg_canvas.draw()
            buf = io.BytesIO()
            self._agg_canvas.print_png(buf)
            pix = QPixmap()
            pix.loadFromData(buf.getvalue(), "PNG")
            if not pix.isNull() and self._label.width() > 0 and self._label.height() > 0:
                pix = pix.scaled(self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._label.setPixmap(pix)
        except Exception:
            # Best-effort: don't crash the UI if plotting fails.
            pass

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Re-scale the existing pixmap (fast). If none, render.
        try:
            pix = self._label.pixmap()
            if pix is not None and not pix.isNull():
                self._label.setPixmap(pix.scaled(self._label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self._render_to_label()
        except Exception:
            pass

    def clear(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()
