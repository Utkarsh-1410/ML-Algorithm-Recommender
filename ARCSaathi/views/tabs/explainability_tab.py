"""Explainability module (Global/Local/Debugging/Drift/Fairness/Auto insights)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSplitter,
)

from ..widgets.mpl_canvas import MatplotlibCanvas
from ..widgets.html_view import HtmlView


class ExplainabilityTab(QWidget):
    # Controller requests
    refresh_requested = Signal()
    set_target_requested = Signal(str, str)  # target, task_type

    run_global_requested = Signal(dict)
    run_local_requested = Signal(dict)
    run_whatif_requested = Signal(dict)
    run_debug_requested = Signal(dict)
    run_drift_requested = Signal(dict)
    run_fairness_requested = Signal(dict)
    run_auto_requested = Signal(dict)

    export_report_requested = Signal(str)  # html/pdf

    # Controller responses
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._payload: Dict[str, Any] = {}
        self._baseline_df: Optional[object] = None

        # ---- Top controls ----
        self.cmb_model = QComboBox()
        self.cmb_task = QComboBox()
        self.cmb_task.addItems(["classification", "regression", "clustering", "dimred"])
        self.cmb_target = QComboBox()

        self.btn_refresh = QPushButton("Refresh")
        self.btn_set_target = QPushButton("Set Target")

        self.btn_run_global = QPushButton("Run Global")
        self.btn_run_local = QPushButton("Run Local")
        self.btn_run_whatif = QPushButton("Run What-If")
        self.btn_run_debug = QPushButton("Run Debug")
        self.btn_run_drift = QPushButton("Run Drift")
        self.btn_run_fairness = QPushButton("Run Fairness")
        self.btn_run_auto = QPushButton("Auto Insights")

        self.btn_export_html = QPushButton("Export HTML")
        self.btn_export_pdf = QPushButton("Export PDF")

        top = QHBoxLayout()
        top.addWidget(QLabel("Model:"))
        top.addWidget(self.cmb_model, 2)
        top.addWidget(QLabel("Task:"))
        top.addWidget(self.cmb_task, 1)
        top.addWidget(QLabel("Target:"))
        top.addWidget(self.cmb_target, 2)
        top.addWidget(self.btn_refresh)
        top.addWidget(self.btn_set_target)
        top.addStretch(1)
        top.addWidget(self.btn_export_html)
        top.addWidget(self.btn_export_pdf)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_run_global)
        actions.addWidget(self.btn_run_local)
        actions.addWidget(self.btn_run_whatif)
        actions.addWidget(self.btn_run_debug)
        actions.addWidget(self.btn_run_drift)
        actions.addWidget(self.btn_run_fairness)
        actions.addWidget(self.btn_run_auto)
        actions.addStretch(1)

        # ---- Inner tabs (3-way) ----
        self.tabs = QTabWidget()

        # Global
        self.lst_features = QListWidget()
        self.lst_features.setSelectionMode(QListWidget.ExtendedSelection)
        self.fig_global = MatplotlibCanvas(width=6.5, height=3.4)
        self.html_global = HtmlView()
        self.txt_global = QTextEdit()
        self.txt_global.setReadOnly(True)

        global_panel = QWidget()
        gl = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Features (for PDP/ALE/Boundary):"))
        left.addWidget(self.lst_features, 1)
        left_box = QGroupBox("Controls")
        left_box.setLayout(left)

        right = QVBoxLayout()
        right.addWidget(QLabel("Global plots (Matplotlib)"))
        right.addWidget(self.fig_global, 2)
        right.addWidget(QLabel("Interactive / HTML (Plotly/SHAP)"))
        right.addWidget(self.html_global, 2)
        right.addWidget(QLabel("Summary"))
        right.addWidget(self.txt_global, 1)

        gl.addWidget(left_box, 1)
        gl.addLayout(right, 3)
        global_panel.setLayout(gl)

        # Local
        self.spn_row = QSpinBox()
        self.spn_row.setMinimum(0)
        self.spn_row.setMaximum(0)
        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(0, 100)
        self.sld_conf.setValue(0)
        self.lbl_conf = QLabel("Min confidence: 0%")

        self.tbl_candidates = QTableWidget(0, 3)
        self.tbl_candidates.setHorizontalHeaderLabels(["Row", "y_pred", "confidence"])
        self.tbl_candidates.setSortingEnabled(True)

        self.fig_local = MatplotlibCanvas(width=6.5, height=3.2)
        self.html_local = HtmlView()
        self.txt_local = QTextEdit()
        self.txt_local.setReadOnly(True)

        local_panel = QWidget()
        ll = QVBoxLayout()
        row_sel = QHBoxLayout()
        row_sel.addWidget(QLabel("Row:"))
        row_sel.addWidget(self.spn_row)
        row_sel.addSpacing(16)
        row_sel.addWidget(self.lbl_conf)
        row_sel.addWidget(self.sld_conf, 2)
        row_sel.addStretch(1)
        ll.addLayout(row_sel)
        ll.addWidget(QLabel("Candidates (filter by confidence; double-click to explain):"))
        ll.addWidget(self.tbl_candidates, 1)
        ll.addWidget(QLabel("Local explanation plot (Matplotlib)"))
        ll.addWidget(self.fig_local, 2)
        ll.addWidget(QLabel("Interactive / HTML (Plotly/SHAP)"))
        ll.addWidget(self.html_local, 1)
        ll.addWidget(QLabel("Why this prediction?"))
        ll.addWidget(self.txt_local, 1)
        local_panel.setLayout(ll)

        # What-if
        self.tbl_whatif = QTableWidget(0, 2)
        self.tbl_whatif.setHorizontalHeaderLabels(["Feature", "Value"])
        self.txt_whatif = QTextEdit()
        self.txt_whatif.setReadOnly(True)
        whatif_panel = QWidget()
        wl = QVBoxLayout()
        wl.addWidget(QLabel("What-If (uses selected row as baseline)."))
        wl.addWidget(self.tbl_whatif, 1)
        wl.addWidget(QLabel("Prediction / sensitivity summary"))
        wl.addWidget(self.txt_whatif, 1)
        whatif_panel.setLayout(wl)

        # Debug
        self.fig_debug = MatplotlibCanvas(width=6.5, height=3.4)
        self.txt_debug = QTextEdit()
        self.txt_debug.setReadOnly(True)
        debug_panel = QWidget()
        dl = QVBoxLayout()
        dl.addWidget(QLabel("Error analysis (click points to explain)"))
        dl.addWidget(self.fig_debug, 2)
        dl.addWidget(QLabel("Summary"))
        dl.addWidget(self.txt_debug, 1)
        debug_panel.setLayout(dl)

        # Drift
        self.txt_drift = QTextEdit()
        self.txt_drift.setReadOnly(True)
        drift_panel = QWidget()
        dr = QVBoxLayout()
        dr.addWidget(QLabel("Data drift detection"))
        dr.addWidget(self.txt_drift, 1)
        drift_panel.setLayout(dr)

        # Fairness
        self.cmb_protected = QComboBox()
        self.cmb_privileged = QComboBox()
        fair_controls = QHBoxLayout()
        fair_controls.addWidget(QLabel("Protected attribute:"))
        fair_controls.addWidget(self.cmb_protected, 2)
        fair_controls.addWidget(QLabel("Privileged group:"))
        fair_controls.addWidget(self.cmb_privileged, 2)
        self.txt_fairness = QTextEdit()
        self.txt_fairness.setReadOnly(True)
        fairness_panel = QWidget()
        fl = QVBoxLayout()
        fl.addLayout(fair_controls)
        fl.addWidget(self.txt_fairness, 1)
        fairness_panel.setLayout(fl)

        # Automobile insights
        self.txt_auto = QTextEdit()
        self.txt_auto.setReadOnly(True)
        auto_panel = QWidget()
        al = QVBoxLayout()
        al.addWidget(QLabel("Automobile analytics insights"))
        al.addWidget(self.txt_auto, 1)
        auto_panel.setLayout(al)

        # Local + What-If combined
        local_whatif = QWidget()
        lw_l = QVBoxLayout()
        lw_l.setContentsMargins(0, 0, 0, 0)
        lw_split = QSplitter(Qt.Vertical)
        lw_split.addWidget(local_panel)
        lw_split.addWidget(whatif_panel)
        lw_split.setStretchFactor(0, 3)
        lw_split.setStretchFactor(1, 1)
        lw_l.addWidget(lw_split, 1)
        local_whatif.setLayout(lw_l)

        # Monitoring combined
        monitoring = QWidget()
        mon_l = QVBoxLayout()
        mon_l.setContentsMargins(0, 0, 0, 0)
        mon_split = QSplitter(Qt.Vertical)
        mon_split.addWidget(debug_panel)
        mon_split.addWidget(drift_panel)
        mon_split.addWidget(fairness_panel)
        mon_split.addWidget(auto_panel)
        mon_split.setStretchFactor(0, 3)
        mon_split.setStretchFactor(1, 1)
        mon_split.setStretchFactor(2, 1)
        mon_split.setStretchFactor(3, 1)
        mon_l.addWidget(mon_split, 1)
        monitoring.setLayout(mon_l)

        self.tabs.addTab(global_panel, "Global")
        self.tabs.addTab(local_whatif, "Local + What-If")
        self.tabs.addTab(monitoring, "Monitoring")

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(actions)
        layout.addWidget(self.tabs, 1)
        self.setLayout(layout)

        # ---- events ----
        self.btn_refresh.clicked.connect(lambda: self.refresh_requested.emit())
        self.btn_set_target.clicked.connect(self._emit_set_target)

        self.btn_run_global.clicked.connect(self._emit_run_global)
        self.btn_run_local.clicked.connect(self._emit_run_local)
        self.btn_run_whatif.clicked.connect(self._emit_run_whatif)
        self.btn_run_debug.clicked.connect(self._emit_run_debug)
        self.btn_run_drift.clicked.connect(self._emit_run_drift)
        self.btn_run_fairness.clicked.connect(self._emit_run_fairness)
        self.btn_run_auto.clicked.connect(self._emit_run_auto)

        self.btn_export_html.clicked.connect(lambda: self.export_report_requested.emit("html"))
        self.btn_export_pdf.clicked.connect(lambda: self.export_report_requested.emit("pdf"))

        self.sld_conf.valueChanged.connect(self._on_conf_changed)
        self.tbl_candidates.cellDoubleClicked.connect(self._on_candidate_double_clicked)

        # Matplotlib click-to-explain (Debug tab)
        try:
            self.fig_debug.canvas.mpl_connect("pick_event", self._on_pick_point)
        except Exception:
            pass

    # ----- payload management -----
    def set_context(self, *, models: List[str], target_candidates: List[str], feature_names: List[str]) -> None:
        self.cmb_model.clear()
        self.cmb_model.addItems(models or [])

        self.cmb_target.clear()
        self.cmb_target.addItems([""] + (target_candidates or []))

        self.lst_features.clear()
        for f in feature_names or []:
            self.lst_features.addItem(QListWidgetItem(str(f)))

        # protected attribute candidates (best-effort: all columns)
        self.cmb_protected.clear()
        self.cmb_protected.addItems([""] + (target_candidates or []))
        self.cmb_privileged.clear()
        self.cmb_privileged.addItems([""])

    def set_candidate_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.tbl_candidates.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.tbl_candidates.setItem(i, 0, QTableWidgetItem(str(r.get("row", ""))))
            self.tbl_candidates.setItem(i, 1, QTableWidgetItem(str(r.get("y_pred", ""))))
            conf = r.get("confidence", "")
            self.tbl_candidates.setItem(i, 2, QTableWidgetItem(str(conf)))
        self.tbl_candidates.resizeColumnsToContents()

        # allow row spinbox range
        try:
            max_row = max(int(r.get("row", 0)) for r in rows) if rows else 0
            self.spn_row.setMaximum(max(0, max_row))
        except Exception:
            pass

    def set_global_output(self, *, text: str = "", html: str = "") -> None:
        if text:
            self.txt_global.setPlainText(text)
        if html:
            self.html_global.set_html(html)

    def set_local_output(self, *, text: str = "", html: str = "") -> None:
        if text:
            self.txt_local.setPlainText(text)
        if html:
            self.html_local.set_html(html)

    def set_debug_output(self, text: str) -> None:
        self.txt_debug.setPlainText(text or "")

    def set_drift_output(self, text: str) -> None:
        self.txt_drift.setPlainText(text or "")

    def set_fairness_output(self, text: str, privileged_groups: Optional[List[str]] = None) -> None:
        self.txt_fairness.setPlainText(text or "")
        if privileged_groups is not None:
            self.cmb_privileged.clear()
            self.cmb_privileged.addItems([""] + [str(x) for x in privileged_groups])

    def set_auto_output(self, text: str) -> None:
        self.txt_auto.setPlainText(text or "")

    def set_whatif_output(self, text: str) -> None:
        self.txt_whatif.setPlainText(text or "")

    # ----- emit helpers -----
    def _emit_set_target(self) -> None:
        self.set_target_requested.emit(self.cmb_target.currentText(), self.cmb_task.currentText())

    def _selected_features(self) -> List[str]:
        out = []
        for it in self.lst_features.selectedItems() or []:
            out.append(str(it.text()))
        return out

    def _emit_run_global(self) -> None:
        self.run_global_requested.emit(
            {
                "model": self.cmb_model.currentText(),
                "task": self.cmb_task.currentText(),
                "target": self.cmb_target.currentText(),
                "features": self._selected_features(),
            }
        )

    def _emit_run_local(self) -> None:
        self.run_local_requested.emit(
            {
                "model": self.cmb_model.currentText(),
                "task": self.cmb_task.currentText(),
                "target": self.cmb_target.currentText(),
                "row": int(self.spn_row.value()),
            }
        )

    def _emit_run_whatif(self) -> None:
        self.run_whatif_requested.emit(
            {
                "model": self.cmb_model.currentText(),
                "task": self.cmb_task.currentText(),
                "target": self.cmb_target.currentText(),
                "row": int(self.spn_row.value()),
            }
        )

    def _emit_run_debug(self) -> None:
        self.run_debug_requested.emit(
            {
                "model": self.cmb_model.currentText(),
                "task": self.cmb_task.currentText(),
                "target": self.cmb_target.currentText(),
            }
        )

    def _emit_run_drift(self) -> None:
        self.run_drift_requested.emit(
            {
                "model": self.cmb_model.currentText(),
                "task": self.cmb_task.currentText(),
                "target": self.cmb_target.currentText(),
            }
        )

    def _emit_run_fairness(self) -> None:
        self.run_fairness_requested.emit(
            {
                "model": self.cmb_model.currentText(),
                "task": self.cmb_task.currentText(),
                "target": self.cmb_target.currentText(),
                "protected": self.cmb_protected.currentText(),
                "privileged": self.cmb_privileged.currentText(),
            }
        )

    def _emit_run_auto(self) -> None:
        self.run_auto_requested.emit(
            {
                "model": self.cmb_model.currentText(),
                "task": self.cmb_task.currentText(),
                "target": self.cmb_target.currentText(),
            }
        )

    def _on_conf_changed(self, v: int) -> None:
        self.lbl_conf.setText(f"Min confidence: {int(v)}%")

    def _on_candidate_double_clicked(self, row: int, _col: int) -> None:
        try:
            it = self.tbl_candidates.item(row, 0)
            if it is not None:
                self.spn_row.setValue(int(it.text()))
                self._emit_run_local()
        except Exception:
            pass

    def _on_pick_point(self, event) -> None:
        # event.ind provides indices in the scatter
        try:
            ind = int(event.ind[0])
            # Controller can map this index if it used the same order
            self.spn_row.setValue(ind)
            self._emit_run_local()
        except Exception:
            pass
