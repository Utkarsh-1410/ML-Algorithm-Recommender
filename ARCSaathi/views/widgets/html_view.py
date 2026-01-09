"""Simple HTML view widget.

Uses QtWebEngine if available; otherwise falls back to QTextBrowser.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget


class HtmlView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._engine = None
        # QtWebEngine is powerful but can be unstable on some Windows setups (GPU/driver).
        # Keep the app reliable by requiring explicit opt-in.
        use_webengine = str(os.environ.get("ARCSAATHI_USE_WEBENGINE", "0")).strip() in ("1", "true", "True")
        if use_webengine:
            try:
                from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore

                self._engine = QWebEngineView()
                w = self._engine
            except Exception:
                self._engine = None
                w = QTextBrowser()
        else:
            w = QTextBrowser()

        self._widget = w
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget)
        self.setLayout(layout)

    def set_html(self, html: str) -> None:
        if self._engine is not None:
            self._engine.setHtml(html)
        else:
            # QTextBrowser
            try:
                self._widget.setHtml(html)
            except Exception:
                self._widget.setPlainText(str(html))

    def clear(self) -> None:
        self.set_html("")
