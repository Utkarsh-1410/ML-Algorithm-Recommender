"""Simple radar chart widget for displaying score breakdowns.

Uses matplotlib (already a dependency) embedded in PySide6.
"""

from __future__ import annotations

from typing import Dict, List

from PySide6.QtWidgets import QWidget, QVBoxLayout

from .mpl_canvas import MatplotlibCanvas


class RadarChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._plot = MatplotlibCanvas(width=3.2, height=3.2, dpi=110)
        self._fig = self._plot.figure
        self._canvas = self._plot.canvas

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def set_scores(self, scores: Dict[str, int], *, title: str = "") -> None:
        # Expect keys: accuracy/speed/interpretability/robustness
        labels = ["Accuracy", "Speed", "Interpretability", "Robustness"]
        keys = ["accuracy", "speed", "interpretability", "robustness"]
        values = [max(0, min(100, int(scores.get(k, 0)))) for k in keys]

        # Radar needs a closed loop
        values_loop = values + values[:1]

        import numpy as np

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles_loop = angles + angles[:1]

        self._fig.clear()
        ax = self._fig.add_subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_thetagrids(np.degrees(angles), labels)
        ax.set_ylim(0, 100)

        ax.plot(angles_loop, values_loop, linewidth=2)
        ax.fill(angles_loop, values_loop, alpha=0.15)

        if title:
            ax.set_title(title, pad=12)

        ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()
