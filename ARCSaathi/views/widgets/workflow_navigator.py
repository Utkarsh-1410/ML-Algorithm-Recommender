"""Left sidebar workflow navigator with 7 steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class WorkflowStep:
    index: int
    title: str


class WorkflowStepButton(QPushButton):
    def __init__(self, step: WorkflowStep, parent=None):
        super().__init__(parent)
        self.step = step
        self.setCheckable(True)
        self.setObjectName("WorkflowStepButton")
        self._completed = False
        self._locked = False
        self._render()

    def set_completed(self, completed: bool) -> None:
        self._completed = bool(completed)
        self._render()

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked)
        self.setEnabled(not self._locked)
        self._render()

    def _render(self) -> None:
        mark = "✓" if self._completed else "○"
        self.setText(f"{self.step.index}. {self.step.title}   {mark}")
        self.setProperty("completed", "1" if self._completed else "0")
        self.setProperty("locked", "1" if self._locked else "0")
        self.style().unpolish(self)
        self.style().polish(self)


class WorkflowNavigator(QWidget):
    step_selected = Signal(int)  # 1..7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WorkflowNavigator")

        self.steps: List[WorkflowStep] = [
            WorkflowStep(1, "Dataset Upload"),
            WorkflowStep(2, "Data Profiling"),
            WorkflowStep(3, "Task Detection"),
            WorkflowStep(4, "Preprocessing"),
            WorkflowStep(5, "Model Training"),
            WorkflowStep(6, "Evaluation"),
            WorkflowStep(7, "Deployment"),
        ]

        self.lbl_title = QLabel("Workflow")
        self.lbl_title.setObjectName("SidebarTitle")

        self.progress = QProgressBar()
        self.progress.setOrientation(Qt.Vertical)
        self.progress.setRange(0, len(self.steps))
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setObjectName("WorkflowProgress")

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self._buttons: List[WorkflowStepButton] = []
        for s in self.steps:
            b = WorkflowStepButton(s)
            self.btn_group.addButton(b, s.index)
            self._buttons.append(b)

        # Layout: progress bar + vertical steps
        steps_layout = QVBoxLayout()
        steps_layout.setSpacing(6)
        for b in self._buttons:
            steps_layout.addWidget(b)
        steps_layout.addStretch(1)

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.progress)
        row.addLayout(steps_layout, 1)

        outer = QVBoxLayout()
        outer.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(self.lbl_title)
        outer.addLayout(row, 1)

        self.setLayout(outer)

        self.btn_group.idClicked.connect(self.step_selected)

        # Default locking: only step 1 unlocked
        for i in range(2, 8):
            self.set_step_locked(i, True)
        self.set_current_step(1)

    def set_current_step(self, step_index: int) -> None:
        btn = self.btn_group.button(step_index)
        if btn:
            btn.setChecked(True)
            btn.setProperty("current", "1")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        for b in self._buttons:
            if b.step.index != step_index:
                b.setProperty("current", "0")
                b.style().unpolish(b)
                b.style().polish(b)

    def set_step_completed(self, step_index: int, completed: bool) -> None:
        btn = self.btn_group.button(step_index)
        if isinstance(btn, WorkflowStepButton):
            btn.set_completed(completed)
        self._recompute_progress()

    def set_step_locked(self, step_index: int, locked: bool) -> None:
        btn = self.btn_group.button(step_index)
        if isinstance(btn, WorkflowStepButton):
            btn.set_locked(locked)

    def unlock_next(self, step_index: int) -> None:
        """Unlock the step after step_index (if exists)."""
        nxt = step_index + 1
        if 1 <= nxt <= len(self.steps):
            self.set_step_locked(nxt, False)

    def _recompute_progress(self) -> None:
        count = 0
        for b in self._buttons:
            if b.property("completed") == "1":
                count += 1
        self.progress.setValue(count)
