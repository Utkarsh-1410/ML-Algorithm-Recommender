"""Tab 2: Task detection + preprocessing configuration + pipeline builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass
class TaskConfig:
    task_type: str = "regression"  # regression|classification|clustering|time_series
    target: str = ""
    classification_type: str = "binary"  # binary|multiclass
    n_clusters: int = 3


class PreprocessingPipelineTab(QWidget):
    # Existing pipeline actions
    apply_pipeline_requested = Signal()
    add_step_requested = Signal(str, dict)  # operation, params
    remove_step_requested = Signal(int)  # index

    # Task detection
    detect_task_requested = Signal()
    task_config_changed = Signal(object)  # TaskConfig

    # Pipeline support
    preview_steps_requested = Signal(object, int)  # list[step_dict], step_index (-1 means full)
    validate_steps_requested = Signal(object)  # list[step_dict]
    steps_replaced_requested = Signal(object)  # list[step_dict]
    smart_preprocess_requested = Signal(object)  # TaskConfig
    clear_steps_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._task = TaskConfig()
        self._dataset_columns: list[str] = []
        self._steps: list[Dict[str, Any]] = []

        # Build panels once so they can be placed into sub-tabs.
        task_panel = self._build_task_detection_panel()
        missing_panel = self._build_missing_values_panel()
        encoding_panel = self._build_encoding_panel()
        scaling_panel = self._build_scaling_panel()
        feat_panel = self._build_feature_selection_panel()
        outlier_panel = self._build_outlier_panel()
        pipeline_panel = self._build_pipeline_builder()

        self.subtabs = QTabWidget()

        tab_task = QWidget()
        tab_task_l = QVBoxLayout()
        tab_task_l.setContentsMargins(10, 10, 10, 10)
        tab_task_l.setSpacing(10)
        tab_task_l.addWidget(task_panel, 1)
        tab_task.setLayout(tab_task_l)

        tab_transforms = QWidget()
        tab_transforms_l = QVBoxLayout()
        tab_transforms_l.setContentsMargins(10, 10, 10, 10)
        tab_transforms_l.setSpacing(10)

        # Inner tabs so the transforms page isn't cramped.
        self.transforms_tabs = QTabWidget()

        t_missing = QWidget()
        t_missing_l = QVBoxLayout()
        t_missing_l.setContentsMargins(0, 0, 0, 0)
        t_missing_l.setSpacing(10)
        t_missing_l.addWidget(missing_panel, 1)
        t_missing.setLayout(t_missing_l)

        t_encode_scale = QWidget()
        t_encode_scale_l = QVBoxLayout()
        t_encode_scale_l.setContentsMargins(0, 0, 0, 0)
        t_encode_scale_l.setSpacing(10)
        t_encode_scale_l.addWidget(encoding_panel)
        t_encode_scale_l.addWidget(scaling_panel)
        t_encode_scale_l.addStretch(1)
        t_encode_scale.setLayout(t_encode_scale_l)

        t_feature_outliers = QWidget()
        t_feature_outliers_l = QVBoxLayout()
        t_feature_outliers_l.setContentsMargins(0, 0, 0, 0)
        t_feature_outliers_l.setSpacing(10)
        t_feature_outliers_l.addWidget(feat_panel)
        t_feature_outliers_l.addWidget(outlier_panel)
        t_feature_outliers_l.addStretch(1)
        t_feature_outliers.setLayout(t_feature_outliers_l)

        self.transforms_tabs.addTab(t_missing, "Missing")
        self.transforms_tabs.addTab(t_encode_scale, "Encode + Scale")
        self.transforms_tabs.addTab(t_feature_outliers, "Features + Outliers")

        tab_transforms_l.addWidget(self.transforms_tabs, 1)
        tab_transforms.setLayout(tab_transforms_l)

        tab_pipeline = QWidget()
        tab_pipeline_l = QVBoxLayout()
        tab_pipeline_l.setContentsMargins(10, 10, 10, 10)
        tab_pipeline_l.setSpacing(10)
        tab_pipeline_l.addWidget(pipeline_panel, 1)
        tab_pipeline.setLayout(tab_pipeline_l)

        self.subtabs.addTab(tab_task, "Task")
        self.subtabs.addTab(tab_transforms, "Transforms")
        self.subtabs.addTab(tab_pipeline, "Pipeline")

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.subtabs, 1)
        self.setLayout(root)

    # ---- Public API for controllers ----
    def set_dataset_columns(self, columns: list[str], *, missing_pct: Optional[Dict[str, float]] = None) -> None:
        self._dataset_columns = [str(c) for c in (columns or [])]
        missing_pct = missing_pct or {}

        # Target selectors
        for cmb in [self.cmb_target_reg, self.cmb_target_cls, self.cmb_target_ts]:
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem("")
            cmb.addItems(self._dataset_columns)
            cmb.blockSignals(False)

        # Missing table
        self.tbl_missing.setRowCount(0)
        for col in self._dataset_columns:
            r = self.tbl_missing.rowCount()
            self.tbl_missing.insertRow(r)
            self.tbl_missing.setItem(r, 0, QTableWidgetItem(col))
            self.tbl_missing.setItem(r, 1, QTableWidgetItem(f"{float(missing_pct.get(col, 0.0)):.1f}%"))

            cmb = QComboBox()
            cmb.addItems(["Mean", "Median", "Mode", "Constant", "Drop rows", "Interpolation"])
            self.tbl_missing.setCellWidget(r, 2, cmb)

            const = QLineEdit()
            const.setPlaceholderText("(for Constant)")
            self.tbl_missing.setCellWidget(r, 3, const)

            use = QComboBox()
            use.addItems(["Skip", "Apply"])
            self.tbl_missing.setCellWidget(r, 4, use)

        self._update_pipeline_preview_choices()

    def set_task_detection(self, *, task_type: str, target: Optional[str], metrics: list[str], reasoning: list[str], classification_type: Optional[str] = None) -> None:
        task_type = (task_type or "regression").lower()
        if task_type == "time_series":
            self.rb_task_ts.setChecked(True)
        elif task_type == "clustering":
            self.rb_task_clust.setChecked(True)
        elif task_type == "classification":
            self.rb_task_cls.setChecked(True)
        else:
            self.rb_task_reg.setChecked(True)

        self.lbl_suggested_task_value.setText(task_type.replace("_", " ").title())
        self.lbl_suggested_target_value.setText(target or "-")
        self.lbl_metrics_value.setText(", ".join(metrics or []) if metrics else "-")
        self.txt_detection.setPlainText("\n".join(reasoning or []))

        if target:
            for cmb in [self.cmb_target_reg, self.cmb_target_cls, self.cmb_target_ts]:
                idx = cmb.findText(target)
                if idx >= 0:
                    cmb.setCurrentIndex(idx)

        if classification_type:
            idx = self.cmb_cls_type.findText(classification_type.title())
            if idx >= 0:
                self.cmb_cls_type.setCurrentIndex(idx)

        self._sync_task_config_from_ui(emit=False)

    def set_steps(self, steps: List[Dict[str, Any]]) -> None:
        self._steps = list(steps or [])
        self.steps_list.clear()
        for step in self._steps:
            item = QListWidgetItem(self._format_step(step))
            item.setData(Qt.UserRole, step)
            self.steps_list.addItem(item)
        self._refresh_pipeline_graph()
        self._update_pipeline_preview_choices()

    def set_preview(self, headers: list[str], rows: list[list[str]], summary: str = "") -> None:
        self.lbl_preview_summary.setText(summary or "")
        self.tbl_preview.clear()
        self.tbl_preview.setRowCount(0)
        self.tbl_preview.setColumnCount(len(headers))
        self.tbl_preview.setHorizontalHeaderLabels([str(h) for h in headers])
        for r in rows:
            ri = self.tbl_preview.rowCount()
            self.tbl_preview.insertRow(ri)
            for ci, v in enumerate(r):
                self.tbl_preview.setItem(ri, ci, QTableWidgetItem(str(v)))

    def set_validation_result(self, ok: bool, message: str) -> None:
        self.lbl_validate_result.setText(("Valid" if ok else "Invalid") + (f": {message}" if message else ""))

    # ---- Task Detection Panel ----
    def _build_task_detection_panel(self) -> QGroupBox:
        gb = QGroupBox("Task Detection")
        v = QVBoxLayout()

        top = QHBoxLayout()
        self.btn_detect = QPushButton("Auto-detect")
        self.btn_detect.clicked.connect(self.detect_task_requested)
        top.addWidget(self.btn_detect)
        top.addStretch(1)

        self.lbl_suggested_task_value = QLabel("-")
        self.lbl_suggested_target_value = QLabel("-")
        self.lbl_metrics_value = QLabel("-")

        summary = QFormLayout()
        summary.addRow("Suggested task:", self.lbl_suggested_task_value)
        summary.addRow("Suggested target:", self.lbl_suggested_target_value)
        summary.addRow("Suggested metrics:", self.lbl_metrics_value)

        self.txt_detection = QTextEdit()
        self.txt_detection.setReadOnly(True)
        self.txt_detection.setPlaceholderText("Detection reasoning will appear here…")

        self.rb_task_reg = QRadioButton("Regression")
        self.rb_task_cls = QRadioButton("Classification")
        self.rb_task_clust = QRadioButton("Clustering")
        self.rb_task_ts = QRadioButton("Time Series")
        self.rb_task_reg.setChecked(True)

        self.task_group = QButtonGroup(self)
        for rb in [self.rb_task_reg, self.rb_task_cls, self.rb_task_clust, self.rb_task_ts]:
            self.task_group.addButton(rb)
            rb.toggled.connect(self._on_task_override_changed)

        override_row = QHBoxLayout()
        override_row.addWidget(QLabel("Manual override:"))
        override_row.addWidget(self.rb_task_reg)
        override_row.addWidget(self.rb_task_cls)
        override_row.addWidget(self.rb_task_clust)
        override_row.addWidget(self.rb_task_ts)
        override_row.addStretch(1)

        self.task_stack = QStackedWidget()

        # Regression
        w_reg = QWidget()
        f_reg = QFormLayout()
        self.cmb_target_reg = QComboBox()
        self.cmb_target_reg.currentTextChanged.connect(lambda _t: self._sync_task_config_from_ui())
        f_reg.addRow("Target column:", self.cmb_target_reg)
        w_reg.setLayout(f_reg)

        # Classification
        w_cls = QWidget()
        f_cls = QFormLayout()
        self.cmb_target_cls = QComboBox()
        self.cmb_target_cls.currentTextChanged.connect(lambda _t: self._sync_task_config_from_ui())
        self.cmb_cls_type = QComboBox()
        self.cmb_cls_type.addItems(["Binary", "Multi-class"])
        self.cmb_cls_type.currentTextChanged.connect(lambda _t: self._sync_task_config_from_ui())
        f_cls.addRow("Target column:", self.cmb_target_cls)
        f_cls.addRow("Classification:", self.cmb_cls_type)
        w_cls.setLayout(f_cls)

        # Clustering
        w_cl = QWidget()
        f_cl = QFormLayout()
        self.spn_clusters = QSpinBox()
        self.spn_clusters.setRange(2, 50)
        self.spn_clusters.setValue(3)
        self.spn_clusters.valueChanged.connect(lambda _v: self._sync_task_config_from_ui())
        self.sld_clusters = QSlider(Qt.Horizontal)
        self.sld_clusters.setRange(2, 50)
        self.sld_clusters.setValue(3)
        self.sld_clusters.valueChanged.connect(self.spn_clusters.setValue)
        self.spn_clusters.valueChanged.connect(self.sld_clusters.setValue)
        f_cl.addRow("# Clusters:", self.spn_clusters)
        f_cl.addRow("", self.sld_clusters)
        w_cl.setLayout(f_cl)

        # Time series
        w_ts = QWidget()
        f_ts = QFormLayout()
        self.cmb_target_ts = QComboBox()
        self.cmb_target_ts.currentTextChanged.connect(lambda _t: self._sync_task_config_from_ui())
        f_ts.addRow("Target column:", self.cmb_target_ts)
        w_ts.setLayout(f_ts)

        self.task_stack.addWidget(w_reg)
        self.task_stack.addWidget(w_cls)
        self.task_stack.addWidget(w_cl)
        self.task_stack.addWidget(w_ts)
        self.task_stack.setCurrentIndex(0)

        v.addLayout(top)
        v.addLayout(summary)
        v.addWidget(QLabel("Reasoning"))
        v.addWidget(self.txt_detection)
        v.addLayout(override_row)
        v.addWidget(self.task_stack)
        gb.setLayout(v)
        return gb

    def _on_task_override_changed(self) -> None:
        if self.rb_task_reg.isChecked():
            self.task_stack.setCurrentIndex(0)
        elif self.rb_task_cls.isChecked():
            self.task_stack.setCurrentIndex(1)
        elif self.rb_task_clust.isChecked():
            self.task_stack.setCurrentIndex(2)
        else:
            self.task_stack.setCurrentIndex(3)
        self._sync_task_config_from_ui()

    def _sync_task_config_from_ui(self, emit: bool = True) -> None:
        if self.rb_task_reg.isChecked():
            self._task.task_type = "regression"
            self._task.target = self.cmb_target_reg.currentText().strip()
        elif self.rb_task_cls.isChecked():
            self._task.task_type = "classification"
            self._task.target = self.cmb_target_cls.currentText().strip()
            self._task.classification_type = "binary" if self.cmb_cls_type.currentText().lower().startswith("binary") else "multiclass"
        elif self.rb_task_clust.isChecked():
            self._task.task_type = "clustering"
            self._task.target = ""
            self._task.n_clusters = int(self.spn_clusters.value())
        else:
            self._task.task_type = "time_series"
            self._task.target = self.cmb_target_ts.currentText().strip()

        if emit:
            self.task_config_changed.emit(self._task)

    # ---- Preprocessing panels ----
    def _build_missing_values_panel(self) -> QGroupBox:
        gb = QGroupBox("Missing Value Handling")
        v = QVBoxLayout()
        self.tbl_missing = QTableWidget(0, 5)
        self.tbl_missing.setHorizontalHeaderLabels(["Column", "Missing %", "Strategy", "Constant", "Use"])
        self.tbl_missing.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_missing.setAlternatingRowColors(True)

        btns = QHBoxLayout()
        self.btn_preview_missing = QPushButton("Preview")
        self.btn_add_missing = QPushButton("Add to Pipeline")
        self.btn_preview_missing.clicked.connect(self._preview_missing)
        self.btn_add_missing.clicked.connect(self._add_missing_to_pipeline)
        btns.addWidget(self.btn_preview_missing)
        btns.addWidget(self.btn_add_missing)
        btns.addStretch(1)

        v.addWidget(self.tbl_missing)
        v.addLayout(btns)
        gb.setLayout(v)
        return gb

    def _build_encoding_panel(self) -> QGroupBox:
        gb = QGroupBox("Categorical Encoding")
        f = QFormLayout()
        self.cmb_encoding = QComboBox()
        self.cmb_encoding.addItems(["One-Hot", "Label", "Ordinal", "Target", "Frequency"])
        self.spn_max_categories = QSpinBox()
        self.spn_max_categories.setRange(2, 100000)
        self.spn_max_categories.setValue(50)
        self.cmb_unknown = QComboBox()
        self.cmb_unknown.addItems(["Ignore", "Error"])
        f.addRow("Method:", self.cmb_encoding)
        f.addRow("Max categories:", self.spn_max_categories)
        f.addRow("Unknown categories:", self.cmb_unknown)

        btns = QHBoxLayout()
        self.btn_preview_enc = QPushButton("Preview")
        self.btn_add_enc = QPushButton("Add to Pipeline")
        self.btn_preview_enc.clicked.connect(self._preview_encoding)
        self.btn_add_enc.clicked.connect(self._add_encoding_to_pipeline)
        btns.addWidget(self.btn_preview_enc)
        btns.addWidget(self.btn_add_enc)
        btns.addStretch(1)

        v = QVBoxLayout()
        v.addLayout(f)
        v.addLayout(btns)
        gb.setLayout(v)
        return gb

    def _build_scaling_panel(self) -> QGroupBox:
        gb = QGroupBox("Feature Scaling")
        f = QFormLayout()
        self.cmb_scaling = QComboBox()
        self.cmb_scaling.addItems(["Standard", "MinMax", "Robust", "MaxAbs"])

        self.rb_scale_all = QRadioButton("All features")
        self.rb_scale_num = QRadioButton("Only numerical")
        self.rb_scale_excl_bin = QRadioButton("Exclude binary")
        self.rb_scale_num.setChecked(True)

        scope = QHBoxLayout()
        scope.addWidget(self.rb_scale_all)
        scope.addWidget(self.rb_scale_num)
        scope.addWidget(self.rb_scale_excl_bin)
        scope.addStretch(1)

        f.addRow("Method:", self.cmb_scaling)
        f.addRow("Apply to:", scope)

        btns = QHBoxLayout()
        self.btn_preview_scale = QPushButton("Preview")
        self.btn_add_scale = QPushButton("Add to Pipeline")
        self.btn_preview_scale.clicked.connect(self._preview_scaling)
        self.btn_add_scale.clicked.connect(self._add_scaling_to_pipeline)
        btns.addWidget(self.btn_preview_scale)
        btns.addWidget(self.btn_add_scale)
        btns.addStretch(1)

        v = QVBoxLayout()
        v.addLayout(f)
        v.addLayout(btns)
        gb.setLayout(v)
        return gb

    def _build_feature_selection_panel(self) -> QGroupBox:
        gb = QGroupBox("Feature Selection")
        f = QFormLayout()

        self.sld_corr = QSlider(Qt.Horizontal)
        self.sld_corr.setRange(0, 100)
        self.sld_corr.setValue(90)
        self.spn_corr = QDoubleSpinBox()
        self.spn_corr.setRange(0.0, 1.0)
        self.spn_corr.setSingleStep(0.01)
        self.spn_corr.setValue(0.90)
        self.sld_corr.valueChanged.connect(lambda v: self.spn_corr.setValue(v / 100.0))
        self.spn_corr.valueChanged.connect(lambda v: self.sld_corr.setValue(int(v * 100)))
        corr_row = QHBoxLayout()
        corr_row.addWidget(self.sld_corr)
        corr_row.addWidget(self.spn_corr)

        self.spn_var = QDoubleSpinBox()
        self.spn_var.setRange(0.0, 1000.0)
        self.spn_var.setSingleStep(0.01)
        self.spn_var.setValue(0.0)

        self.chk_rfe = QRadioButton("Enable RFE")
        self.spn_rfe_k = QSpinBox()
        self.spn_rfe_k.setRange(1, 5000)
        self.spn_rfe_k.setValue(10)
        rfe_row = QHBoxLayout()
        rfe_row.addWidget(self.chk_rfe)
        rfe_row.addWidget(QLabel("k="))
        rfe_row.addWidget(self.spn_rfe_k)
        rfe_row.addStretch(1)

        self.chk_kbest = QRadioButton("Enable SelectKBest")
        self.spn_kbest_k = QSpinBox()
        self.spn_kbest_k.setRange(1, 5000)
        self.spn_kbest_k.setValue(20)
        self.cmb_kbest = QComboBox()
        self.cmb_kbest.addItems(["f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression", "chi2"])
        kbest_row = QHBoxLayout()
        kbest_row.addWidget(self.chk_kbest)
        kbest_row.addWidget(QLabel("k="))
        kbest_row.addWidget(self.spn_kbest_k)
        kbest_row.addWidget(self.cmb_kbest)
        kbest_row.addStretch(1)

        f.addRow("Correlation threshold:", corr_row)
        f.addRow("Variance threshold:", self.spn_var)
        f.addRow("RFE:", rfe_row)
        f.addRow("SelectKBest:", kbest_row)

        btns = QHBoxLayout()
        self.btn_preview_fs = QPushButton("Preview")
        self.btn_add_fs = QPushButton("Add to Pipeline")
        self.btn_preview_fs.clicked.connect(self._preview_feature_selection)
        self.btn_add_fs.clicked.connect(self._add_feature_selection_to_pipeline)
        btns.addWidget(self.btn_preview_fs)
        btns.addWidget(self.btn_add_fs)
        btns.addStretch(1)

        v = QVBoxLayout()
        v.addLayout(f)
        v.addLayout(btns)
        gb.setLayout(v)
        return gb

    def _build_outlier_panel(self) -> QGroupBox:
        gb = QGroupBox("Outlier Handling")
        f = QFormLayout()
        self.cmb_outlier = QComboBox()
        self.cmb_outlier.addItems(["Z-score", "IQR", "Isolation Forest", "Winsorization"])
        self.spn_z = QDoubleSpinBox()
        self.spn_z.setRange(0.5, 10.0)
        self.spn_z.setValue(3.0)
        self.spn_iqr = QDoubleSpinBox()
        self.spn_iqr.setRange(0.5, 10.0)
        self.spn_iqr.setValue(1.5)
        self.spn_cont = QDoubleSpinBox()
        self.spn_cont.setRange(0.001, 0.5)
        self.spn_cont.setDecimals(3)
        self.spn_cont.setSingleStep(0.01)
        self.spn_cont.setValue(0.05)
        self.spn_win = QDoubleSpinBox()
        self.spn_win.setRange(0.0, 0.49)
        self.spn_win.setDecimals(2)
        self.spn_win.setSingleStep(0.01)
        self.spn_win.setValue(0.01)

        f.addRow("Method:", self.cmb_outlier)
        f.addRow("Z-score threshold:", self.spn_z)
        f.addRow("IQR multiplier:", self.spn_iqr)
        f.addRow("IsolationForest contamination:", self.spn_cont)
        f.addRow("Winsorization tail:", self.spn_win)

        btns = QHBoxLayout()
        self.btn_preview_out = QPushButton("Preview")
        self.btn_add_out = QPushButton("Add to Pipeline")
        self.btn_preview_out.clicked.connect(self._preview_outliers)
        self.btn_add_out.clicked.connect(self._add_outliers_to_pipeline)
        btns.addWidget(self.btn_preview_out)
        btns.addWidget(self.btn_add_out)
        btns.addStretch(1)

        v = QVBoxLayout()
        v.addLayout(f)
        v.addLayout(btns)
        gb.setLayout(v)
        return gb

    # ---- Pipeline Builder ----
    def _build_pipeline_builder(self) -> QGroupBox:
        gb = QGroupBox("Pipeline Builder")
        v = QVBoxLayout()

        self.steps_list = QListWidget()
        self.steps_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.steps_list.setDefaultDropAction(Qt.MoveAction)
        self.steps_list.model().rowsMoved.connect(lambda *_: self._on_steps_reordered())

        self.scene = QGraphicsScene(self)
        self.graph = QGraphicsView(self.scene)
        self.graph.setMinimumHeight(140)

        btns = QHBoxLayout()
        self.btn_validate = QPushButton("Validate")
        self.btn_preview = QPushButton("Preview intermediate")
        self.btn_save = QPushButton("Save pipeline")
        self.btn_load = QPushButton("Load pipeline")
        self.btn_smart = QPushButton("Smart Preprocess")
        self.btn_remove = QPushButton("Remove selected")
        self.btn_clear = QPushButton("Clear")
        self.btn_apply = QPushButton("Apply Pipeline")

        self.btn_validate.clicked.connect(self._validate_pipeline)
        self.btn_preview.clicked.connect(self._preview_pipeline)
        self.btn_save.clicked.connect(self._save_pipeline)
        self.btn_load.clicked.connect(self._load_pipeline)
        self.btn_smart.clicked.connect(lambda: self.smart_preprocess_requested.emit(self._task))
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self.clear_steps_requested)
        self.btn_apply.clicked.connect(self.apply_pipeline_requested)

        for b in [self.btn_validate, self.btn_preview, self.btn_save, self.btn_load, self.btn_smart, self.btn_remove, self.btn_clear]:
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(self.btn_apply)

        self.lbl_validate_result = QLabel("")

        sel = QHBoxLayout()
        self.cmb_preview_stage = QComboBox()
        sel.addWidget(QLabel("Preview:"))
        sel.addWidget(self.cmb_preview_stage)
        sel.addStretch(1)

        self.lbl_preview_summary = QLabel("")
        self.tbl_preview = QTableWidget(0, 0)
        self.tbl_preview.setAlternatingRowColors(True)
        self.tbl_preview.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_preview.setMinimumHeight(180)

        v.addWidget(QLabel("Steps (drag & drop to reorder)"))
        v.addWidget(self.steps_list, 1)
        v.addWidget(QLabel("Visual pipeline flow"))
        v.addWidget(self.graph)
        v.addLayout(btns)
        v.addWidget(self.lbl_validate_result)
        v.addLayout(sel)
        v.addWidget(self.lbl_preview_summary)
        v.addWidget(self.tbl_preview)
        gb.setLayout(v)
        return gb

    # ---- Step list helpers ----
    def _format_step(self, step: Dict[str, Any]) -> str:
        name = step.get("name") or step.get("operation")
        params = step.get("params", {})
        return f"{name}: {params}"

    def _current_steps_from_list(self) -> list[Dict[str, Any]]:
        steps: list[Dict[str, Any]] = []
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            step = item.data(Qt.UserRole) or {}
            steps.append(step)
        return steps

    def _on_steps_reordered(self) -> None:
        steps = self._current_steps_from_list()
        self.steps_replaced_requested.emit(steps)
        self._steps = steps
        self._refresh_pipeline_graph()
        self._update_pipeline_preview_choices()

    def _update_pipeline_preview_choices(self) -> None:
        self.cmb_preview_stage.blockSignals(True)
        self.cmb_preview_stage.clear()
        self.cmb_preview_stage.addItem("After all steps", -1)
        for idx in range(len(self._steps)):
            self.cmb_preview_stage.addItem(f"After step {idx + 1}", idx)
        self.cmb_preview_stage.blockSignals(False)

    def _refresh_pipeline_graph(self) -> None:
        self.scene.clear()
        x, y = 10, 10
        w, h = 260, 44
        pad = 12

        if not self._steps:
            t = self.scene.addText("No steps in pipeline")
            t.setDefaultTextColor(Qt.gray)
            t.setPos(x, y)
            return

        for i, step in enumerate(self._steps):
            rect = QGraphicsRectItem(x, y + i * (h + pad), w, h)
            rect.setBrush(Qt.GlobalColor.white)
            rect.setPen(QPen(Qt.GlobalColor.black))
            self.scene.addItem(rect)
            txt = QGraphicsTextItem(self._format_step(step))
            txt.setTextWidth(w - 10)
            txt.setPos(x + 5, y + i * (h + pad) + 6)
            self.scene.addItem(txt)

        self.scene.setSceneRect(0, 0, w + 40, y + len(self._steps) * (h + pad) + 10)

    # ---- Section actions ----
    def _preview_missing(self) -> None:
        steps = self._build_missing_steps()
        if not steps:
            return
        self.preview_steps_requested.emit(self._current_steps_from_list() + steps, -1)

    def _add_missing_to_pipeline(self) -> None:
        steps = self._build_missing_steps()
        if not steps:
            return
        for step in steps:
            self.add_step_requested.emit(step["operation"], step.get("params", {}))

    def _build_missing_steps(self) -> list[Dict[str, Any]]:
        # Group per-column strategies into separate steps.
        groups: Dict[str, list[str]] = {}
        constants: Dict[str, Any] = {}

        for r in range(self.tbl_missing.rowCount()):
            col_item = self.tbl_missing.item(r, 0)
            if col_item is None:
                continue
            col = col_item.text()

            use = self.tbl_missing.cellWidget(r, 4)
            if isinstance(use, QComboBox) and use.currentText() != "Apply":
                continue

            cmb = self.tbl_missing.cellWidget(r, 2)
            const = self.tbl_missing.cellWidget(r, 3)
            if not isinstance(cmb, QComboBox) or not isinstance(const, QLineEdit):
                continue

            s = cmb.currentText().lower()
            groups.setdefault(s, []).append(col)
            if s == "constant":
                constants[col] = const.text().strip()

        steps: list[Dict[str, Any]] = []
        for strat, cols in groups.items():
            if strat == "drop rows":
                steps.append({"name": "Drop rows (missing)", "operation": "drop_missing", "params": {"columns": cols}})
            elif strat == "interpolation":
                steps.append({"name": "Interpolation", "operation": "interpolate", "params": {"columns": cols}})
            else:
                strategy = {
                    "mean": "mean",
                    "median": "median",
                    "mode": "most_frequent",
                    "constant": "constant",
                }.get(strat, "mean")
                params: Dict[str, Any] = {"columns": cols, "strategy": strategy}
                if strategy == "constant":
                    # If multiple columns, use a simple scalar if all equal; else default to 0.
                    vals = [constants.get(c, "") for c in cols]
                    params["fill_value"] = vals[0] if vals and all(v == vals[0] for v in vals) and vals[0] != "" else 0
                steps.append({"name": f"Impute ({strategy})", "operation": "impute", "params": params})

        return steps

    def _preview_encoding(self) -> None:
        step = self._build_encoding_step()
        if step is None:
            return
        self.preview_steps_requested.emit(self._current_steps_from_list() + [step], -1)

    def _add_encoding_to_pipeline(self) -> None:
        step = self._build_encoding_step()
        if step is None:
            return
        self.add_step_requested.emit(step["operation"], step.get("params", {}))

    def _build_encoding_step(self) -> Optional[Dict[str, Any]]:
        method_map = {"one-hot": "onehot", "label": "label", "ordinal": "ordinal", "target": "target", "frequency": "frequency"}
        method = method_map.get(self.cmb_encoding.currentText().lower(), "onehot")
        handle_unknown = "ignore" if self.cmb_unknown.currentText().lower() == "ignore" else "error"
        return {
            "name": f"Encode ({method})",
            "operation": "encode",
            "params": {
                "columns": [],
                "method": method,
                "max_categories": int(self.spn_max_categories.value()),
                "handle_unknown": handle_unknown,
                "target": self._task.target,
            },
        }

    def _preview_scaling(self) -> None:
        step = self._build_scaling_step()
        if step is None:
            return
        self.preview_steps_requested.emit(self._current_steps_from_list() + [step], -1)

    def _add_scaling_to_pipeline(self) -> None:
        step = self._build_scaling_step()
        if step is None:
            return
        self.add_step_requested.emit(step["operation"], step.get("params", {}))

    def _build_scaling_step(self) -> Optional[Dict[str, Any]]:
        method_map = {"standard": "standard", "minmax": "minmax", "robust": "robust", "maxabs": "maxabs"}
        method = method_map.get(self.cmb_scaling.currentText().lower(), "standard")
        scope = "numeric"
        if self.rb_scale_all.isChecked():
            scope = "all"
        elif self.rb_scale_excl_bin.isChecked():
            scope = "exclude_binary"
        return {"name": f"Scale ({method})", "operation": "scale", "params": {"columns": [], "method": method, "scope": scope}}

    def _preview_feature_selection(self) -> None:
        steps = self._build_feature_selection_steps()
        if not steps:
            return
        self.preview_steps_requested.emit(self._current_steps_from_list() + steps, -1)

    def _add_feature_selection_to_pipeline(self) -> None:
        steps = self._build_feature_selection_steps()
        if not steps:
            return
        for s in steps:
            self.add_step_requested.emit(s["operation"], s.get("params", {}))

    def _build_feature_selection_steps(self) -> list[Dict[str, Any]]:
        steps: list[Dict[str, Any]] = []
        corr = float(self.spn_corr.value())
        if corr > 0:
            steps.append({"name": "Correlation filter", "operation": "feature_selection", "params": {"method": "correlation", "threshold": corr}})
        var = float(self.spn_var.value())
        if var > 0:
            steps.append({"name": "Variance threshold", "operation": "feature_selection", "params": {"method": "variance", "threshold": var}})
        if self.chk_kbest.isChecked():
            steps.append({"name": "SelectKBest", "operation": "feature_selection", "params": {"method": "select_k_best", "k": int(self.spn_kbest_k.value()), "score_func": self.cmb_kbest.currentText(), "target": self._task.target, "task_type": self._task.task_type}})
        if self.chk_rfe.isChecked():
            steps.append({"name": "RFE", "operation": "feature_selection", "params": {"method": "rfe", "k": int(self.spn_rfe_k.value()), "target": self._task.target, "task_type": self._task.task_type}})
        return steps

    def _preview_outliers(self) -> None:
        step = self._build_outlier_step()
        if step is None:
            return
        self.preview_steps_requested.emit(self._current_steps_from_list() + [step], -1)

    def _add_outliers_to_pipeline(self) -> None:
        step = self._build_outlier_step()
        if step is None:
            return
        self.add_step_requested.emit(step["operation"], step.get("params", {}))

    def _build_outlier_step(self) -> Optional[Dict[str, Any]]:
        m = self.cmb_outlier.currentText().lower()
        if m.startswith("z"):
            return {"name": "Outliers (z-score)", "operation": "outliers", "params": {"method": "zscore", "threshold": float(self.spn_z.value())}}
        if m.startswith("i") and "isolation" not in m:
            return {"name": "Outliers (IQR)", "operation": "outliers", "params": {"method": "iqr", "multiplier": float(self.spn_iqr.value())}}
        if "isolation" in m:
            return {"name": "Outliers (IsolationForest)", "operation": "outliers", "params": {"method": "isolation_forest", "contamination": float(self.spn_cont.value())}}
        return {"name": "Outliers (Winsorization)", "operation": "outliers", "params": {"method": "winsorize", "tail": float(self.spn_win.value())}}

    # ---- Pipeline actions ----
    def _remove_selected(self) -> None:
        row = self.steps_list.currentRow()
        if row >= 0:
            self.remove_step_requested.emit(row)

    def _validate_pipeline(self) -> None:
        self.validate_steps_requested.emit(self._current_steps_from_list())

    def _preview_pipeline(self) -> None:
        idx = int(self.cmb_preview_stage.currentData())
        self.preview_steps_requested.emit(self._current_steps_from_list(), idx)

    def _save_pipeline(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Preprocessing Pipeline", "", "JSON Files (*.json)")
        if not path:
            return
        steps = self._current_steps_from_list()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(steps, f, indent=2)
        except Exception:
            pass

    def _load_pipeline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Preprocessing Pipeline", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                steps = json.load(f)
            if isinstance(steps, list):
                self.steps_replaced_requested.emit(steps)
        except Exception:
            pass
