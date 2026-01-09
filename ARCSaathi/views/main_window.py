"""Main window for ARCSaathi with header, workflow sidebar, tabs, and status bar."""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .tabs import (
    DataLoadingProfilingTab,
    PreprocessingPipelineTab,
    ModelTrainingTuningTab,
    ResultsComparisonTab,
    ModelRecommenderTab,
    ExplainabilityTab,
)
from .widgets import HeaderBar, WorkflowNavigator, TabPage


class MainWindow(QMainWindow):
    """Professional main window with synchronized workflow navigation and tabs."""

    # Surface key interactions to controllers
    theme_toggle_requested = Signal()
    help_requested = Signal(str)  # context
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("ARCSaathi")
        self.setMinimumSize(1024, 768)

        # Header
        self.header = HeaderBar()
        self.header.help_clicked.connect(lambda: self.help_requested.emit("app"))
        self.header.settings_clicked.connect(self.theme_toggle_requested)
        self.header.export_clicked.connect(self.export_requested)

        # Sidebar workflow
        self.workflow = WorkflowNavigator()
        self.workflow.step_selected.connect(self._on_workflow_step_selected)

        # Tabs (core workspace)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.tabs.setMovable(False)
        self.tabs.setTabsClosable(False)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Inner tab contents
        self.tab_data = DataLoadingProfilingTab()
        self.tab_preprocess = PreprocessingPipelineTab()
        self.tab_training = ModelTrainingTuningTab()
        self.tab_results = ResultsComparisonTab()
        self.tab_recommender = ModelRecommenderTab()
        self.tab_explainability = ExplainabilityTab()

        # Wrap each in a consistent header container
        self.page_data = TabPage("Data Loading & Profiling", self.tab_data)
        self.page_preprocess = TabPage("Preprocessing Pipeline", self.tab_preprocess)
        self.page_training = TabPage("Model Training & Tuning", self.tab_training)
        self.page_results = TabPage("Results & Comparison", self.tab_results)
        self.page_recommender = TabPage("Model Recommender", self.tab_recommender)
        self.page_explainability = TabPage("Explainability & Monitoring", self.tab_explainability)

        self.page_data.help_clicked.connect(lambda: self.help_requested.emit("tab:data"))
        self.page_preprocess.help_clicked.connect(lambda: self.help_requested.emit("tab:preprocessing"))
        self.page_training.help_clicked.connect(lambda: self.help_requested.emit("tab:training"))
        self.page_results.help_clicked.connect(lambda: self.help_requested.emit("tab:results"))
        self.page_recommender.help_clicked.connect(lambda: self.help_requested.emit("tab:recommender"))
        self.page_explainability.help_clicked.connect(lambda: self.help_requested.emit("tab:explainability"))

        self.tabs.addTab(self.page_data, "Data Loading & Profiling")
        self.tabs.addTab(self.page_preprocess, "Preprocessing Pipeline")
        self.tabs.addTab(self.page_training, "Model Training & Tuning")
        self.tabs.addTab(self.page_results, "Results & Comparison")
        self.tabs.addTab(self.page_recommender, "Model Recommender")
        self.tabs.addTab(self.page_explainability, "Explainability")

        # Layout: header on top, splitter for sidebar + tabs
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.workflow)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1120])

        central = QWidget()
        v = QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.header)
        v.addWidget(splitter, 1)
        central.setLayout(v)
        self.setCentralWidget(central)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        self.lbl_operation = QLabel("Ready")
        self.lbl_operation.setObjectName("StatusOperation")

        self.lbl_memory = QLabel("Memory: -")
        self.lbl_memory.setObjectName("StatusMemory")

        self.lbl_dataset = QLabel("Dataset: -")
        self.lbl_dataset.setObjectName("StatusDataset")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setFixedWidth(220)

        self._status.addWidget(self.lbl_operation, 1)
        self._status.addPermanentWidget(self.lbl_memory)
        self._status.addPermanentWidget(self.progress)
        self._status.addPermanentWidget(self.lbl_dataset)

        # Memory usage timer
        self._mem_timer = QTimer(self)
        self._mem_timer.setInterval(1000)
        self._mem_timer.timeout.connect(self._update_memory)
        self._mem_timer.start()

        # Default: current step reflects initial tab
        self._sync_workflow_to_tab(self.tabs.currentIndex())

        # Lock tabs to match workflow gating
        self._update_tab_locks()

    # ---- Public API for controllers ----
    def set_app_status(self, state: str) -> None:
        self.header.set_status(state)
        self.lbl_operation.setText(state)

    def set_dataset_info(self, rows: int, cols: int) -> None:
        self.lbl_dataset.setText(f"Dataset: {rows} × {cols}")

    def set_progress(self, visible: bool, value: int = 0) -> None:
        self.progress.setVisible(visible)
        self.progress.setValue(max(0, min(100, int(value))))

    def set_step_completed(self, step_index: int, completed: bool = True) -> None:
        self.workflow.set_step_completed(step_index, completed)
        if completed:
            self.workflow.unlock_next(step_index)
        self._update_tab_locks()

    def set_step_locked(self, step_index: int, locked: bool) -> None:
        self.workflow.set_step_locked(step_index, locked)
        self._update_tab_locks()

    def navigate_to_step(self, step_index: int) -> None:
        tab_index = self._tab_for_step(step_index)
        self.tabs.setCurrentIndex(tab_index)
        self.workflow.set_current_step(step_index)

    # ---- Internal wiring ----
    def _on_workflow_step_selected(self, step_index: int) -> None:
        self.navigate_to_step(step_index)

    def _on_tab_changed(self, index: int) -> None:
        self._sync_workflow_to_tab(index)

    def _sync_workflow_to_tab(self, tab_index: int) -> None:
        # Highlight the earliest step mapped to this tab.
        step = self._step_for_tab(tab_index)
        self.workflow.set_current_step(step)

    def _tab_for_step(self, step_index: int) -> int:
        # 7 workflow steps mapped into 6 tabs
        if step_index in (1, 2, 3):
            return 0
        if step_index == 4:
            return 1
        if step_index == 5:
            return 2
        if step_index == 6:
            return 3
        # Step 7 unlocks both tabs
        return 4

    def _update_tab_locks(self) -> None:
        """Disable future tabs until corresponding workflow steps are unlocked."""
        # Tab 0 always enabled
        self.tabs.setTabEnabled(0, True)

        # Tab enablement follows whether the *first step* mapped to that tab is unlocked.
        step_for_tab = {
            1: 4,  # Preprocessing tab gated by Step 4
            2: 5,  # Training tab gated by Step 5
            3: 6,  # Results tab gated by Step 6
            4: 7,  # Recommender/Deployment gated by Step 7
            5: 7,  # Explainability gated by Step 7
        }
        for tab_index, step_index in step_for_tab.items():
            btn = self.workflow.btn_group.button(step_index)
            self.tabs.setTabEnabled(tab_index, bool(btn and btn.isEnabled()))

    def _step_for_tab(self, tab_index: int) -> int:
        # Default highlight per tab
        if tab_index == 0:
            return 1
        if tab_index == 1:
            return 4
        if tab_index == 2:
            return 5
        if tab_index == 3:
            return 6
        return 7

    def _update_memory(self) -> None:
        # Prefer psutil, fall back gracefully.
        try:
            try:
                import psutil  # type: ignore

                proc = psutil.Process()
                rss = proc.memory_info().rss
                self.lbl_memory.setText(f"Memory: {rss / (1024 ** 2):.0f} MB")
                return
            except Exception:
                self.lbl_memory.setText("Memory: -")
        except BaseException:
            # Never let periodic UI timers crash the app.
            return
