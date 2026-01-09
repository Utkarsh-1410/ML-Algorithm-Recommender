"""Tab 5: Intelligent Model Recommender."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QTabWidget,
    QWidget,
)

from ..widgets.radar_chart import RadarChartWidget


class ModelRecommenderTab(QWidget):
    recommend_requested = Signal(dict)  # request payload
    feedback_submitted = Signal(dict)  # {accepted, algorithm_key, ...}
    export_requested = Signal(str, dict)  # fmt, payload

    def __init__(self, parent=None):
        super().__init__(parent)

        self._last_result: Optional[Dict[str, Any]] = None

        self.title = QLabel("Model Recommendation Engine")

        # --- controls row ---
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems(["Best Fit", "Fast & Light", "High Accuracy"])

        self.btn_recommend = QPushButton("Analyze & Recommend")

        self.chk_auto = QCheckBox("Automobile analytics mode")
        self.cmb_auto_preset = QComboBox()
        self.cmb_auto_preset.addItems(
            [
                "(none)",
                "Car Price Prediction",
                "Fuel Efficiency",
                "Maintenance Forecasting",
                "Customer Segmentation",
            ]
        )
        self.cmb_auto_preset.setEnabled(False)

        top = QHBoxLayout()
        top.addWidget(QLabel("Profile:"))
        top.addWidget(self.cmb_profile)
        top.addSpacing(10)
        top.addWidget(self.chk_auto)
        top.addWidget(self.cmb_auto_preset)
        top.addStretch(1)
        top.addWidget(self.btn_recommend)

        # --- requirements ---
        req_group = QGroupBox("Problem Requirements")
        req_form = QFormLayout()

        self.cmb_task = QComboBox()
        self.cmb_task.addItems(["classification", "regression", "clustering", "dimred"])

        self.txt_target = QTextEdit()
        self.txt_target.setFixedHeight(28)
        self.txt_target.setPlaceholderText("Target column (optional; supervised only)")

        self.cmb_interp = QComboBox()
        self.cmb_interp.addItems(["High", "Medium", "Low"])

        self.cmb_train_time = QComboBox()
        self.cmb_train_time.addItems(["Fast", "Moderate", "Unlimited"])

        self.cmb_pred_speed = QComboBox()
        self.cmb_pred_speed.addItems(["Fast", "Moderate", "Unlimited"])

        self.cmb_deploy = QComboBox()
        self.cmb_deploy.addItems(["Small", "Medium", "Unlimited"])

        self.spin_acc_thresh = QDoubleSpinBox()
        self.spin_acc_thresh.setRange(0.0, 1.0)
        self.spin_acc_thresh.setSingleStep(0.05)
        self.spin_acc_thresh.setValue(0.0)

        req_form.addRow("Task type:", self.cmb_task)
        req_form.addRow("Target:", self.txt_target)
        req_form.addRow("Interpretability:", self.cmb_interp)
        req_form.addRow("Training time:", self.cmb_train_time)
        req_form.addRow("Prediction speed:", self.cmb_pred_speed)
        req_form.addRow("Deployment size:", self.cmb_deploy)
        req_form.addRow("Accuracy threshold (0-1):", self.spin_acc_thresh)
        req_group.setLayout(req_form)

        # --- weighting ---
        weight_group = QGroupBox("Weighting")
        wform = QFormLayout()

        self.sld_acc = QSlider(Qt.Horizontal)
        self.sld_acc.setRange(0, 100)
        self.sld_acc.setValue(40)
        self.lbl_acc = QLabel("40")

        self.sld_speed = QSlider(Qt.Horizontal)
        self.sld_speed.setRange(0, 100)
        self.sld_speed.setValue(25)
        self.lbl_speed = QLabel("25")

        self.sld_interp = QSlider(Qt.Horizontal)
        self.sld_interp.setRange(0, 100)
        self.sld_interp.setValue(20)
        self.lbl_interp = QLabel("20")

        self.sld_rob = QSlider(Qt.Horizontal)
        self.sld_rob.setRange(0, 100)
        self.sld_rob.setValue(15)
        self.lbl_rob = QLabel("15")

        wform.addRow("Accuracy:", self._row_slider(self.sld_acc, self.lbl_acc))
        wform.addRow("Speed:", self._row_slider(self.sld_speed, self.lbl_speed))
        wform.addRow("Interpretability:", self._row_slider(self.sld_interp, self.lbl_interp))
        wform.addRow("Robustness:", self._row_slider(self.sld_rob, self.lbl_rob))
        weight_group.setLayout(wform)

        # --- constraints ---
        cons_group = QGroupBox("Constraints")
        cform = QFormLayout()

        self.chk_interpretable = QCheckBox("Must be interpretable")
        self.chk_fast = QCheckBox("Must be fast")
        self.chk_cat = QCheckBox("Must handle categorical")

        self.spin_max_inf = QDoubleSpinBox()
        self.spin_max_inf.setRange(0.0, 10_000.0)
        self.spin_max_inf.setSingleStep(5.0)
        self.spin_max_inf.setValue(0.0)
        self.spin_max_inf.setSuffix(" ms")

        self.spin_max_size = QDoubleSpinBox()
        self.spin_max_size.setRange(0.0, 10_000.0)
        self.spin_max_size.setSingleStep(1.0)
        self.spin_max_size.setValue(0.0)
        self.spin_max_size.setSuffix(" MB")

        cform.addRow(self.chk_interpretable)
        cform.addRow(self.chk_fast)
        cform.addRow(self.chk_cat)
        cform.addRow("Max inference time:", self.spin_max_inf)
        cform.addRow("Max model size:", self.spin_max_size)
        cons_group.setLayout(cform)

        left = QVBoxLayout()
        left.addWidget(req_group)
        left.addWidget(weight_group)
        left.addWidget(cons_group)
        left.addStretch(1)

        # --- results (right) ---
        self.list_top = QListWidget()
        self.list_top.setMinimumWidth(320)

        self.btn_accept = QPushButton("Accept")
        self.btn_reject = QPushButton("Reject")
        self.btn_export_html = QPushButton("Export HTML")
        self.btn_export_pdf = QPushButton("Export PDF")

        actions = QHBoxLayout()
        actions.addWidget(self.btn_accept)
        actions.addWidget(self.btn_reject)
        actions.addStretch(1)
        actions.addWidget(self.btn_export_html)
        actions.addWidget(self.btn_export_pdf)

        self.radar = RadarChartWidget()
        self.tbl_matrix = QTableWidget(0, 6)
        self.tbl_matrix.setHorizontalHeaderLabels(["Model", "Total", "Accuracy", "Speed", "Interpret.", "Robust."])
        self.tbl_matrix.horizontalHeader().setStretchLastSection(True)

        self.txt_why = QTextEdit()
        self.txt_why.setReadOnly(True)

        self.txt_adv = QTextEdit()
        self.txt_adv.setReadOnly(True)
        self.txt_adv.setPlaceholderText("Advanced options will appear here")
        self.txt_adv.setFixedHeight(120)

        self.txt_fe = QTextEdit()
        self.txt_fe.setReadOnly(True)
        self.txt_fe.setPlaceholderText("Feature engineering suggestions will appear here")
        self.txt_fe.setFixedHeight(120)

        right = QVBoxLayout()
        right.addWidget(QLabel("Top Recommendations"))
        right.addWidget(self.list_top, 2)
        right.addLayout(actions)
        right.addWidget(QLabel("Score breakdown"))
        right.addWidget(self.radar, 2)
        right.addWidget(QLabel("Comparison matrix"))
        right.addWidget(self.tbl_matrix, 2)
        right.addWidget(QLabel('"Why this model?"'))
        right.addWidget(self.txt_why, 2)
        right.addWidget(QLabel("Advanced options"))
        right.addWidget(self.txt_adv)
        right.addWidget(QLabel("Feature engineering suggestions"))
        right.addWidget(self.txt_fe)

        # ---- 3-way layout (sub-tabs) to reduce clutter ----
        self.subtabs = QTabWidget()

        tab_req = QWidget()
        tab_req_l = QVBoxLayout()
        tab_req_l.setContentsMargins(10, 10, 10, 10)
        tab_req_l.setSpacing(10)
        tab_req_l.addWidget(req_group, 1)
        tab_req.setLayout(tab_req_l)

        tab_weights = QWidget()
        tab_weights_l = QVBoxLayout()
        tab_weights_l.setContentsMargins(10, 10, 10, 10)
        tab_weights_l.setSpacing(10)
        tab_weights_l.addWidget(weight_group)
        tab_weights_l.addWidget(cons_group)
        tab_weights_l.addStretch(1)
        tab_weights.setLayout(tab_weights_l)

        tab_results = QWidget()
        tab_results_l = QVBoxLayout()
        tab_results_l.setContentsMargins(10, 10, 10, 10)
        tab_results_l.setSpacing(10)
        tab_results_l.addLayout(right, 1)
        tab_results.setLayout(tab_results_l)

        self.subtabs.addTab(tab_req, "Requirements")
        self.subtabs.addTab(tab_weights, "Weights")
        self.subtabs.addTab(tab_results, "Results")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addLayout(top)
        layout.addWidget(self.subtabs, 1)
        self.setLayout(layout)

        # ---- wiring ----
        self.btn_recommend.clicked.connect(self._emit_recommend)
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        self.list_top.currentItemChanged.connect(self._on_selected_model_changed)

        self.btn_accept.clicked.connect(lambda: self._emit_feedback(True))
        self.btn_reject.clicked.connect(lambda: self._emit_feedback(False))
        self.btn_export_html.clicked.connect(lambda: self._emit_export("html"))
        self.btn_export_pdf.clicked.connect(lambda: self._emit_export("pdf"))

        for sld, lbl in [
            (self.sld_acc, self.lbl_acc),
            (self.sld_speed, self.lbl_speed),
            (self.sld_interp, self.lbl_interp),
            (self.sld_rob, self.lbl_rob),
        ]:
            sld.valueChanged.connect(lambda v, l=lbl: l.setText(str(int(v))))

    def set_target(self, target: str) -> None:
        if target:
            self.txt_target.setPlainText(str(target))

    # Controller -> UI
    def set_recommendation_result(self, payload: Dict[str, Any]) -> None:
        self._last_result = dict(payload or {})

        self.list_top.clear()
        self.tbl_matrix.setRowCount(0)
        self.txt_why.clear()
        self.txt_adv.clear()
        self.txt_fe.clear()

        top: List[Dict[str, Any]] = list((payload or {}).get("top", []) or [])
        for rec in top:
            label = f"{rec.get('model_name','')}  ({rec.get('score_total',0)}/100)"
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, rec)
            self.list_top.addItem(it)

        # matrix
        for rec in top:
            r = self.tbl_matrix.rowCount()
            self.tbl_matrix.insertRow(r)
            scores = rec.get("scores", {}) or {}
            self.tbl_matrix.setItem(r, 0, QTableWidgetItem(str(rec.get("model_name", ""))))
            self.tbl_matrix.setItem(r, 1, QTableWidgetItem(str(rec.get("score_total", 0))))
            self.tbl_matrix.setItem(r, 2, QTableWidgetItem(str(scores.get("accuracy", 0))))
            self.tbl_matrix.setItem(r, 3, QTableWidgetItem(str(scores.get("speed", 0))))
            self.tbl_matrix.setItem(r, 4, QTableWidgetItem(str(scores.get("interpretability", 0))))
            self.tbl_matrix.setItem(r, 5, QTableWidgetItem(str(scores.get("robustness", 0))))

        adv = (payload or {}).get("advanced_suggestions", []) or []
        fe = (payload or {}).get("feature_engineering_suggestions", []) or []
        if adv:
            self.txt_adv.setPlainText("\n".join([f"- {a}" for a in adv]))
        if fe:
            self.txt_fe.setPlainText("\n".join([f"- {a}" for a in fe]))

        if self.list_top.count() > 0:
            self.list_top.setCurrentRow(0)

    def set_recommendation_text(self, text: str) -> None:
        # Backwards compatibility fallback
        self.txt_why.setPlainText(text)

    # ---- internals ----
    def _row_slider(self, slider: QSlider, label: QLabel) -> QWidget:
        w = QWidget()
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(label)
        w.setLayout(h)
        return w

    def _on_auto_toggled(self, on: bool) -> None:
        self.cmb_auto_preset.setEnabled(bool(on))

    def _emit_recommend(self) -> None:
        payload = {
            "profile": self.cmb_profile.currentText(),
            "task_type": self.cmb_task.currentText(),
            "target": (self.txt_target.toPlainText() or "").strip() or None,
            "interpretability": self.cmb_interp.currentText(),
            "training_time": self.cmb_train_time.currentText(),
            "prediction_speed": self.cmb_pred_speed.currentText(),
            "deployment_size": self.cmb_deploy.currentText(),
            "required_accuracy": float(self.spin_acc_thresh.value()) if float(self.spin_acc_thresh.value()) > 0 else None,
            "weights": {
                "accuracy": int(self.sld_acc.value()),
                "speed": int(self.sld_speed.value()),
                "interpretability": int(self.sld_interp.value()),
                "robustness": int(self.sld_rob.value()),
            },
            "constraints": {
                "must_be_interpretable": bool(self.chk_interpretable.isChecked()),
                "must_be_fast": bool(self.chk_fast.isChecked()),
                "must_handle_categorical": bool(self.chk_cat.isChecked()),
                "max_inference_ms": float(self.spin_max_inf.value()) if float(self.spin_max_inf.value()) > 0 else None,
                "max_model_size_mb": float(self.spin_max_size.value()) if float(self.spin_max_size.value()) > 0 else None,
            },
            "automobile_mode": bool(self.chk_auto.isChecked()),
            "automobile_preset": self.cmb_auto_preset.currentText() if self.chk_auto.isChecked() else None,
        }
        self.recommend_requested.emit(payload)

    def _on_selected_model_changed(self, current: Optional[QListWidgetItem], _prev: Optional[QListWidgetItem]) -> None:
        if not current:
            return
        rec = current.data(Qt.UserRole) or {}
        scores = rec.get("scores", {}) or {}
        self.radar.set_scores(scores, title=str(rec.get("model_name", "")))

        why = str(rec.get("why", ""))
        strengths = rec.get("strengths", []) or []
        weaknesses = rec.get("weaknesses", []) or []

        lines = []
        if why:
            lines.append(why)
        if strengths:
            lines.append("\nStrengths:\n" + "\n".join([f"- {s}" for s in strengths]))
        if weaknesses:
            lines.append("\nWeaknesses:\n" + "\n".join([f"- {w}" for w in weaknesses]))
        self.txt_why.setPlainText("\n".join(lines).strip())

    def _emit_feedback(self, accepted: bool) -> None:
        item = self.list_top.currentItem()
        if not item:
            return
        rec = item.data(Qt.UserRole) or {}
        payload = {
            "accepted": bool(accepted),
            "recommendation": rec,
            "context": self._last_result or {},
        }
        self.feedback_submitted.emit(payload)

    def _emit_export(self, fmt: str) -> None:
        if not self._last_result:
            return
        self.export_requested.emit(str(fmt), dict(self._last_result))
