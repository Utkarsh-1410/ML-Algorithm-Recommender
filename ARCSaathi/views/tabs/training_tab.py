"""Tab 3: Model Repository + Training Control Panel."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class ModelTrainingTuningTab(QWidget):
    # Legacy signals (kept for compatibility with older controller wiring)
    create_model_requested = Signal(str, str, dict)
    train_model_requested = Signal(str)

    # New signals
    enqueue_jobs_requested = Signal(list, dict)  # jobs, config
    pause_queue_requested = Signal()
    resume_queue_requested = Signal()
    cancel_all_requested = Signal()
    parallelism_changed = Signal(int)
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._registry: Dict[str, Any] = {}
        self._params_by_key: Dict[str, Dict[str, Any]] = {}
        self._selected_keys: List[str] = []

        # ---- Repository panel (left) ----
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Search models...")

        self.cmb_task_filter = QComboBox()
        self.cmb_task_filter.addItems(["All", "Regression", "Classification", "Clustering", "DimRed"])

        self.list_models = QListWidget()
        self.list_models.setSelectionMode(QListWidget.ExtendedSelection)

        self.btn_select_all = QPushButton("Select All")
        self.btn_clear_selection = QPushButton("Clear")

        left_controls = QHBoxLayout()
        left_controls.addWidget(self.btn_select_all)
        left_controls.addWidget(self.btn_clear_selection)
        left_controls.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(QLabel("Model Repository"))
        left.addWidget(self.txt_search)
        left.addWidget(self.cmb_task_filter)
        left.addWidget(self.list_models, 1)
        left.addLayout(left_controls)

        # ---- Details panel (right) ----
        self.lbl_model_name = QLabel("Select a model")
        self.lbl_model_meta = QLabel("")
        self.lbl_model_meta.setWordWrap(True)
        self.txt_model_card = QTextEdit()
        self.txt_model_card.setReadOnly(True)

        self.tbl_params = QTableWidget(0, 2)
        self.tbl_params.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.tbl_params.horizontalHeader().setStretchLastSection(True)

        card_group = QGroupBox("Model Card")
        card_layout = QVBoxLayout()
        card_layout.addWidget(self.lbl_model_name)
        card_layout.addWidget(self.lbl_model_meta)
        card_layout.addWidget(self.txt_model_card, 1)
        card_group.setLayout(card_layout)

        params_group = QGroupBox("Hyperparameters (for active model)")
        params_layout = QVBoxLayout()
        params_layout.addWidget(self.tbl_params, 1)
        params_group.setLayout(params_layout)

        # ---- Training config ----
        cfg_group = QGroupBox("Training Configuration")
        cfg_form = QFormLayout()

        self.cmb_task_type = QComboBox()
        self.cmb_task_type.addItems(["classification", "regression", "clustering", "dimred"])

        self.txt_target = QLineEdit()
        self.txt_target.setPlaceholderText("Target column (for supervised)")

        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.05, 0.95)
        self.spin_test_size.setSingleStep(0.05)
        self.spin_test_size.setValue(0.2)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000)
        self.spin_seed.setValue(42)

        self.cmb_tuning = QComboBox()
        self.cmb_tuning.addItems(["none", "grid", "random", "optuna", "hyperopt"])

        self.spin_cv = QSpinBox()
        self.spin_cv.setRange(2, 10)
        self.spin_cv.setValue(3)

        self.spin_n_iter = QSpinBox()
        self.spin_n_iter.setRange(5, 200)
        self.spin_n_iter.setValue(20)

        self.spin_parallel = QSpinBox()
        self.spin_parallel.setRange(1, 32)
        self.spin_parallel.setValue(2)

        cfg_form.addRow("Task type:", self.cmb_task_type)
        cfg_form.addRow("Target:", self.txt_target)
        cfg_form.addRow("Test size:", self.spin_test_size)
        cfg_form.addRow("Random seed:", self.spin_seed)
        cfg_form.addRow("Tuning:", self.cmb_tuning)
        cfg_form.addRow("CV folds:", self.spin_cv)
        cfg_form.addRow("Random iters:", self.spin_n_iter)
        cfg_form.addRow("Parallel jobs:", self.spin_parallel)
        cfg_group.setLayout(cfg_form)

        # ---- Queue controls + status ----
        queue_group = QGroupBox("Batch Queue")
        self.tbl_queue = QTableWidget(0, 4)
        self.tbl_queue.setHorizontalHeaderLabels(["Run ID", "Model", "Status", "Progress"])
        self.tbl_queue.horizontalHeader().setStretchLastSection(True)

        self.btn_add_queue = QPushButton("Add Selected to Queue")
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_resume = QPushButton("Resume")
        self.btn_cancel = QPushButton("Cancel All")
        self.btn_refresh = QPushButton("Refresh")

        qrow = QHBoxLayout()
        qrow.addWidget(self.btn_add_queue)
        qrow.addStretch(1)
        qrow.addWidget(self.btn_start)
        qrow.addWidget(self.btn_pause)
        qrow.addWidget(self.btn_resume)
        qrow.addWidget(self.btn_cancel)
        qrow.addWidget(self.btn_refresh)

        queue_layout = QVBoxLayout()
        queue_layout.addLayout(qrow)
        queue_layout.addWidget(self.tbl_queue, 1)
        queue_group.setLayout(queue_layout)

        # ---- Recent runs ----
        runs_group = QGroupBox("Recent Runs")
        self.tbl_runs = QTableWidget(0, 5)
        self.tbl_runs.setHorizontalHeaderLabels(["Run ID", "Task", "Model", "Status", "Metrics"])
        self.tbl_runs.horizontalHeader().setStretchLastSection(True)
        runs_layout = QVBoxLayout()
        runs_layout.addWidget(self.tbl_runs, 1)
        runs_group.setLayout(runs_layout)

        # ---- 3-way layout (sub-tabs) to reduce clutter ----
        self.subtabs = QTabWidget()

        # Repository: model list + model card + params
        tab_repo = QWidget()
        tab_repo_l = QHBoxLayout()
        tab_repo_l.setContentsMargins(10, 10, 10, 10)
        tab_repo_l.setSpacing(10)

        repo_left = QWidget()
        repo_left.setLayout(left)

        repo_right = QWidget()
        repo_right_l = QVBoxLayout()
        repo_right_l.setContentsMargins(0, 0, 0, 0)
        repo_right_l.setSpacing(10)
        repo_right_l.addWidget(card_group, 2)
        repo_right_l.addWidget(params_group, 2)
        repo_right.setLayout(repo_right_l)

        tab_repo_l.addWidget(repo_left, 2)
        tab_repo_l.addWidget(repo_right, 5)
        tab_repo.setLayout(tab_repo_l)

        # Training: config + queue
        tab_train = QWidget()
        tab_train_l = QVBoxLayout()
        tab_train_l.setContentsMargins(10, 10, 10, 10)
        tab_train_l.setSpacing(10)
        tab_train_l.addWidget(cfg_group)
        tab_train_l.addWidget(queue_group, 1)
        tab_train.setLayout(tab_train_l)

        # Runs: recent runs table
        tab_runs = QWidget()
        tab_runs_l = QVBoxLayout()
        tab_runs_l.setContentsMargins(10, 10, 10, 10)
        tab_runs_l.setSpacing(10)
        tab_runs_l.addWidget(runs_group, 1)
        tab_runs.setLayout(tab_runs_l)

        self.subtabs.addTab(tab_repo, "Repository")
        self.subtabs.addTab(tab_train, "Training")
        self.subtabs.addTab(tab_runs, "Runs")

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.subtabs, 1)
        self.setLayout(root)

        # ---- Wiring ----
        self.txt_search.textChanged.connect(self._refresh_model_list)
        self.cmb_task_filter.currentTextChanged.connect(self._refresh_model_list)
        self.list_models.itemSelectionChanged.connect(self._on_model_selection_changed)
        self.tbl_params.itemChanged.connect(self._on_param_changed)

        self.btn_select_all.clicked.connect(self._select_all_visible)
        self.btn_clear_selection.clicked.connect(self._clear_selection)

        self.btn_add_queue.clicked.connect(self._on_add_to_queue)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self.pause_queue_requested)
        self.btn_resume.clicked.connect(self.resume_queue_requested)
        self.btn_cancel.clicked.connect(self.cancel_all_requested)
        self.btn_refresh.clicked.connect(self.refresh_requested)
        self.spin_parallel.valueChanged.connect(self.parallelism_changed)

        # Legacy buttons are not present; legacy signals are unused by this UI.

    # ---- Public API (called by controller/main wiring) ----
    def set_registry(self, registry: Dict[str, Any]) -> None:
        self._registry = dict(registry or {})
        self._params_by_key.clear()
        for k, v in self._registry.items():
            card = v.get("card")
            params: Dict[str, Any] = {}
            for p in getattr(card, "params", []) or []:
                params[p.name] = p.default
            self._params_by_key[str(k)] = params
        self._refresh_model_list()

    def set_queue_rows(self, rows: List[Tuple[str, str, str, int]]) -> None:
        self.tbl_queue.setRowCount(0)
        for run_id, model_name, status, pct in rows or []:
            r = self.tbl_queue.rowCount()
            self.tbl_queue.insertRow(r)
            self.tbl_queue.setItem(r, 0, QTableWidgetItem(str(run_id)))
            self.tbl_queue.setItem(r, 1, QTableWidgetItem(str(model_name)))
            self.tbl_queue.setItem(r, 2, QTableWidgetItem(str(status)))
            self.tbl_queue.setItem(r, 3, QTableWidgetItem(f"{int(pct)}%"))

    def upsert_queue_status(self, run_id: str, status: str, progress: Optional[int] = None) -> None:
        # Find row; if absent, create
        row_idx = None
        for r in range(self.tbl_queue.rowCount()):
            item = self.tbl_queue.item(r, 0)
            if item and item.text() == str(run_id):
                row_idx = r
                break
        if row_idx is None:
            row_idx = self.tbl_queue.rowCount()
            self.tbl_queue.insertRow(row_idx)
            self.tbl_queue.setItem(row_idx, 0, QTableWidgetItem(str(run_id)))
            self.tbl_queue.setItem(row_idx, 1, QTableWidgetItem(""))
            self.tbl_queue.setItem(row_idx, 2, QTableWidgetItem(""))
            self.tbl_queue.setItem(row_idx, 3, QTableWidgetItem(""))

        self.tbl_queue.setItem(row_idx, 2, QTableWidgetItem(str(status)))
        if progress is not None:
            self.tbl_queue.setItem(row_idx, 3, QTableWidgetItem(f"{int(progress)}%"))

    def set_recent_runs(self, runs: List[Any]) -> None:
        self.tbl_runs.setRowCount(0)
        for run in runs or []:
            r = self.tbl_runs.rowCount()
            self.tbl_runs.insertRow(r)
            self.tbl_runs.setItem(r, 0, QTableWidgetItem(getattr(run, "run_id", "")))
            self.tbl_runs.setItem(r, 1, QTableWidgetItem(getattr(run, "task_type", "")))
            self.tbl_runs.setItem(r, 2, QTableWidgetItem(getattr(run, "model_name", "")))
            self.tbl_runs.setItem(r, 3, QTableWidgetItem(getattr(run, "status", "")))
            metrics_json = getattr(run, "metrics_json", "{}")
            try:
                mj = json.loads(metrics_json) if isinstance(metrics_json, str) else (metrics_json or {})
                metrics_txt = ", ".join([f"{k}={v}" for k, v in list(mj.items())[:6]])
            except Exception:
                metrics_txt = str(metrics_json)
            self.tbl_runs.setItem(r, 4, QTableWidgetItem(metrics_txt))

    def set_target(self, target: str) -> None:
        if target:
            self.txt_target.setText(str(target))

    # Legacy compatibility
    def set_models(self, model_names: List[str]) -> None:
        # Old UI listed created models. New UI is registry-driven; ignore.
        _ = model_names

    # ---- Internal helpers ----
    def _refresh_model_list(self) -> None:
        search = (self.txt_search.text() or "").strip().lower()
        task_filter = self.cmb_task_filter.currentText()
        task_map = {
            "All": None,
            "Regression": "regression",
            "Classification": "classification",
            "Clustering": "clustering",
            "DimRed": "dimred",
        }
        want_task = task_map.get(task_filter)

        self.list_models.blockSignals(True)
        self.list_models.clear()

        items: List[Tuple[str, str]] = []
        for key, v in (self._registry or {}).items():
            card = v.get("card")
            available = bool(v.get("available", True))
            reason = str(v.get("unavailable_reason", ""))
            name = getattr(card, "name", str(key))
            task = getattr(card, "task_type", "")

            if want_task and task != want_task:
                continue

            hay = f"{key} {name} {task} {getattr(card, 'family', '')}".lower()
            if search and search not in hay:
                continue

            label = f"{name}  [{task}]"
            if not available:
                label += " (unavailable)"
            items.append((str(key), label))

        for key, label in sorted(items, key=lambda x: x[1].lower()):
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if key in self._selected_keys else Qt.Unchecked)
            self.list_models.addItem(it)

        self.list_models.blockSignals(False)

    def _select_all_visible(self) -> None:
        self._selected_keys = []
        for i in range(self.list_models.count()):
            it = self.list_models.item(i)
            key = it.data(Qt.UserRole)
            it.setCheckState(Qt.Checked)
            self._selected_keys.append(str(key))

    def _clear_selection(self) -> None:
        self._selected_keys = []
        for i in range(self.list_models.count()):
            it = self.list_models.item(i)
            it.setCheckState(Qt.Unchecked)
        self._set_active_model(None)

    def _on_model_selection_changed(self) -> None:
        # Active model is the currently selected item; selection may differ from checked items.
        item = self.list_models.currentItem()
        if not item:
            self._set_active_model(None)
            return
        key = str(item.data(Qt.UserRole) or "")
        self._set_active_model(key)

    def _set_active_model(self, algorithm_key: Optional[str]) -> None:
        if not algorithm_key or algorithm_key not in self._registry:
            self.lbl_model_name.setText("Select a model")
            self.lbl_model_meta.setText("")
            self.txt_model_card.setPlainText("")
            self.tbl_params.blockSignals(True)
            self.tbl_params.setRowCount(0)
            self.tbl_params.blockSignals(False)
            return

        v = self._registry.get(algorithm_key, {})
        card = v.get("card")
        available = bool(v.get("available", True))
        reason = str(v.get("unavailable_reason", ""))

        self.lbl_model_name.setText(getattr(card, "name", algorithm_key))
        meta = f"Key: {algorithm_key} | Task: {getattr(card, 'task_type', '')} | Family: {getattr(card, 'family', '')}"
        if not available:
            meta += f" | Unavailable: {reason}"
        self.lbl_model_meta.setText(meta)

        card_text = (
            f"Description: {getattr(card, 'description', '')}\n\n"
            f"Best for: {getattr(card, 'best_for', '')}\n\n"
            f"Pros: {getattr(card, 'pros', '')}\n\n"
            f"Cons: {getattr(card, 'cons', '')}\n\n"
            f"Time: {getattr(card, 'time_complexity', '')} | Memory: {getattr(card, 'memory_usage', '')}\n\n"
            f"Expected: {getattr(card, 'expected_performance', '')}"
        )
        self.txt_model_card.setPlainText(card_text)

        params = self._params_by_key.get(algorithm_key, {})
        self.tbl_params.blockSignals(True)
        self.tbl_params.setRowCount(0)
        for pname, pval in params.items():
            r = self.tbl_params.rowCount()
            self.tbl_params.insertRow(r)
            self.tbl_params.setItem(r, 0, QTableWidgetItem(str(pname)))
            val_item = QTableWidgetItem(str(pval))
            self.tbl_params.setItem(r, 1, val_item)
        self.tbl_params.blockSignals(False)

    def _on_param_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        active_item = self.list_models.currentItem()
        if not active_item:
            return
        key = str(active_item.data(Qt.UserRole) or "")
        if not key:
            return
        pname_item = self.tbl_params.item(item.row(), 0)
        if not pname_item:
            return
        pname = pname_item.text()
        raw = item.text()
        # Best-effort parse
        val: Any = raw
        try:
            if raw.lower() in ("true", "false"):
                val = raw.lower() == "true"
            elif raw.isdigit():
                val = int(raw)
            else:
                val = float(raw)
        except Exception:
            val = raw
        self._params_by_key.setdefault(key, {})[pname] = val

    def _checked_algorithm_keys(self) -> List[str]:
        keys: List[str] = []
        for i in range(self.list_models.count()):
            it = self.list_models.item(i)
            if it.checkState() == Qt.Checked:
                keys.append(str(it.data(Qt.UserRole)))
        self._selected_keys = keys
        return keys

    def _build_config(self) -> Dict[str, Any]:
        return {
            "task_type": self.cmb_task_type.currentText(),
            "target": (self.txt_target.text() or "").strip(),
            "test_size": float(self.spin_test_size.value()),
            "random_state": int(self.spin_seed.value()),
            "tuning_mode": self.cmb_tuning.currentText(),
            "cv_folds": int(self.spin_cv.value()),
            "n_iter": int(self.spin_n_iter.value()),
        }

    def _on_add_to_queue(self) -> None:
        keys = self._checked_algorithm_keys()
        jobs: List[Dict[str, Any]] = []
        for k in keys:
            v = self._registry.get(k, {})
            if not bool(v.get("available", True)):
                continue
            jobs.append({
                "algorithm_key": k,
                "params": dict(self._params_by_key.get(k, {})),
            })
        self.enqueue_jobs_requested.emit(jobs, self._build_config())

    def _on_start(self) -> None:
        # Start is equivalent to resuming + pumping from controller side.
        self.resume_queue_requested.emit()

