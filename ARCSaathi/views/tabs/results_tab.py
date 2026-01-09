"""Tab 4: Model Comparison Dashboard."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..widgets.mpl_canvas import MatplotlibCanvas
from ..widgets.table_color_delegate import BestWorstColorDelegate


class ResultsComparisonTab(QWidget):
    compare_requested = Signal(str)  # task type
    export_table_requested = Signal(str)  # csv/xlsx
    export_fig_requested = Signal(str)  # png/svg
    export_report_requested = Signal(str)  # html/pdf

    def __init__(self, parent=None):
        super().__init__(parent)

        self._payload: Optional[Dict[str, Any]] = None
        self._primary_metric_key: str = ""

        self.title = QLabel("Model Comparison Dashboard")
        self.cmb_task = QComboBox()
        self.cmb_task.addItems(["classification", "regression", "clustering", "dimred"])
        self.btn_compare = QPushButton("Run Comparison")

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_xlsx = QPushButton("Export Excel")
        self.btn_export_png = QPushButton("Export PNG")
        self.btn_export_svg = QPushButton("Export SVG")
        self.btn_export_html = QPushButton("Export HTML")
        self.btn_export_pdf = QPushButton("Export PDF")

        self.table = QTableWidget(0, 0)
        self.table.setSortingEnabled(True)
        self._color_delegate = BestWorstColorDelegate(self.table)

        top = QHBoxLayout()
        top.addWidget(QLabel("Task:"))
        top.addWidget(self.cmb_task)
        top.addWidget(self.btn_compare)
        top.addStretch(1)

        # --- left model list + table ---
        self.list_models = QListWidget()
        self.list_models.setSelectionMode(QListWidget.ExtendedSelection)

        left_group = QGroupBox("Models")
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.list_models, 1)
        left_layout.addWidget(QLabel("Comparison Table"))
        left_layout.addWidget(self.table, 2)
        left_group.setLayout(left_layout)

        # --- plots (kept as a smaller 3-tab set) ---
        self.tabs = QTabWidget()

        # Visuals
        self.fig_main = MatplotlibCanvas(width=6.0, height=3.2)
        self.fig_secondary = MatplotlibCanvas(width=6.0, height=3.2)
        self.fig_tertiary = MatplotlibCanvas(width=6.0, height=3.2)
        self.fig_parallel = MatplotlibCanvas(width=6.0, height=3.6)
        self.fig_feat = MatplotlibCanvas(width=6.0, height=3.6)

        vis = QWidget()
        vis_l = QVBoxLayout()
        vis_l.addWidget(QLabel("Primary visualization"))
        vis_l.addWidget(self.fig_main, 1)
        vis_l.addWidget(QLabel("Secondary visualization"))
        vis_l.addWidget(self.fig_secondary, 1)
        vis_l.addWidget(QLabel("Tertiary visualization"))
        vis_l.addWidget(self.fig_tertiary, 1)
        vis.setLayout(vis_l)

        # Stats + metrics panel (merged)
        self.txt_stats = QLabel("Run comparison to see statistical tests.")
        self.txt_stats.setWordWrap(True)
        metrics = QWidget()
        metrics_l = QVBoxLayout()
        metrics_l.addWidget(QLabel("Parallel coordinates (metrics across models)"))
        metrics_l.addWidget(self.fig_parallel, 2)
        metrics_l.addWidget(QLabel("Feature importance comparison"))
        metrics_l.addWidget(self.fig_feat, 2)
        metrics_l.addWidget(QLabel("Statistics"))
        metrics_l.addWidget(self.txt_stats)
        metrics.setLayout(metrics_l)

        # Insights panel
        self.txt_insights = QLabel("Select a model to see insights.")
        self.txt_insights.setWordWrap(True)
        ins = QWidget()
        ins_l = QVBoxLayout()
        ins_l.addWidget(self.txt_insights)
        ins.setLayout(ins_l)

        self.tabs.addTab(vis, "Visuals")
        self.tabs.addTab(metrics, "Metrics")
        self.tabs.addTab(ins, "Insights")

        # ---- 3-way layout (sub-tabs) to reduce clutter ----
        self.subtabs = QTabWidget()

        tab_compare = QWidget()
        tab_compare_l = QVBoxLayout()
        tab_compare_l.setContentsMargins(10, 10, 10, 10)
        tab_compare_l.setSpacing(10)
        tab_compare_l.addWidget(self.title)
        tab_compare_l.addLayout(top)
        tab_compare_l.addWidget(left_group, 1)
        tab_compare.setLayout(tab_compare_l)

        tab_visuals = QWidget()
        tab_visuals_l = QVBoxLayout()
        tab_visuals_l.setContentsMargins(10, 10, 10, 10)
        tab_visuals_l.setSpacing(10)
        tab_visuals_l.addWidget(QLabel("Select model(s) in the Compare tab, then view charts here."))
        tab_visuals_l.addWidget(self.tabs, 1)
        tab_visuals.setLayout(tab_visuals_l)

        tab_export = QWidget()
        tab_export_l = QVBoxLayout()
        tab_export_l.setContentsMargins(10, 10, 10, 10)
        tab_export_l.setSpacing(10)
        exp_row1 = QHBoxLayout()
        exp_row1.addWidget(self.btn_export_csv)
        exp_row1.addWidget(self.btn_export_xlsx)
        exp_row1.addStretch(1)
        exp_row2 = QHBoxLayout()
        exp_row2.addWidget(self.btn_export_png)
        exp_row2.addWidget(self.btn_export_svg)
        exp_row2.addStretch(1)
        exp_row3 = QHBoxLayout()
        exp_row3.addWidget(self.btn_export_html)
        exp_row3.addWidget(self.btn_export_pdf)
        exp_row3.addStretch(1)
        tab_export_l.addWidget(QLabel("Exports"))
        tab_export_l.addLayout(exp_row1)
        tab_export_l.addLayout(exp_row2)
        tab_export_l.addLayout(exp_row3)
        tab_export_l.addStretch(1)
        tab_export.setLayout(tab_export_l)

        self.subtabs.addTab(tab_compare, "Compare")
        self.subtabs.addTab(tab_visuals, "Visuals")
        self.subtabs.addTab(tab_export, "Export")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.subtabs, 1)
        self.setLayout(layout)

        self.btn_compare.clicked.connect(lambda: self.compare_requested.emit(self.cmb_task.currentText()))
        self.btn_export_csv.clicked.connect(lambda: self.export_table_requested.emit("csv"))
        self.btn_export_xlsx.clicked.connect(lambda: self.export_table_requested.emit("xlsx"))
        self.btn_export_png.clicked.connect(lambda: self.export_fig_requested.emit("png"))
        self.btn_export_svg.clicked.connect(lambda: self.export_fig_requested.emit("svg"))
        self.btn_export_html.clicked.connect(lambda: self.export_report_requested.emit("html"))
        self.btn_export_pdf.clicked.connect(lambda: self.export_report_requested.emit("pdf"))

        self.list_models.currentItemChanged.connect(self._on_selected_model)

    def set_dashboard_payload(self, payload: Dict[str, Any]) -> None:
        """Controller supplies: {rows, task_type, primary_metric_key, best_model, stats, plots, insights}."""
        self._payload = dict(payload or {})

        rows: List[Dict[str, Any]] = list(self._payload.get("rows", []) or [])
        self._primary_metric_key = str(self._payload.get("primary_metric_key", ""))

        self.list_models.clear()
        for r in rows:
            name = str(r.get("Model Name", r.get("Model", "")))
            it = QListWidgetItem(name)
            it.setData(Qt.UserRole, r)
            self.list_models.addItem(it)

        # table
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        headers = list(rows[0].keys())
        self.table.setColumnCount(len(headers))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(headers)

        best_idx: Optional[int] = self._payload.get("best_row")
        best_rows = set([int(best_idx)]) if best_idx is not None else set()
        worst_rows: set[int] = set()

        higher_is_better = bool(self._payload.get("primary_higher_is_better", True))

        primary_col = headers.index(self._primary_metric_key) if self._primary_metric_key in headers else -1

        # find worst by primary (direction-aware)
        if primary_col >= 0:
            vals = []
            for i, row in enumerate(rows):
                try:
                    vals.append((i, float(row.get(self._primary_metric_key))))
                except Exception:
                    pass
            if vals:
                worst_rows = set([min(vals, key=lambda x: x[1])[0]]) if higher_is_better else set([max(vals, key=lambda x: x[1])[0]])

        for r_i, row in enumerate(rows):
            for c_i, h in enumerate(headers):
                val = row.get(h, "")
                if isinstance(val, float):
                    text = f"{val:.4f}"
                else:
                    text = str(val)
                self.table.setItem(r_i, c_i, QTableWidgetItem(text))

        if primary_col >= 0:
            self._color_delegate.set_best_worst(metric_col=primary_col, best_rows=best_rows, worst_rows=worst_rows)
            self.table.setItemDelegateForColumn(primary_col, self._color_delegate)

        self.table.resizeColumnsToContents()

        # stats summary
        stats_txt = str(self._payload.get("stats_summary", ""))
        self.txt_stats.setText(stats_txt or "No statistical tests available.")

        # draw parallel coordinates
        self._draw_parallel()

        # auto select best
        if best_idx is not None and 0 <= int(best_idx) < self.list_models.count():
            self.list_models.setCurrentRow(int(best_idx))

    def _on_selected_model(self, item: Optional[QListWidgetItem], _prev: Optional[QListWidgetItem]) -> None:
        if not item or not self._payload:
            return
        row = item.data(Qt.UserRole) or {}
        name = str(row.get("Model Name", row.get("Model", "")))
        self._draw_visuals(name)
        self._draw_feature_importance(name)
        self._set_insights(name)

    def _draw_parallel(self) -> None:
        payload = self._payload or {}
        rows = list(payload.get("rows", []) or [])
        metrics = list(payload.get("parallel_metrics", []) or [])
        if not rows or not metrics:
            self.fig_parallel.clear()
            return

        fig = self.fig_parallel.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Normalize each metric to 0..1 for plot
        data = []
        names = []
        for r in rows:
            names.append(str(r.get("Model Name", r.get("Model", ""))))
            vals = []
            for m in metrics:
                try:
                    vals.append(float(r.get(m)))
                except Exception:
                    vals.append(float("nan"))
            data.append(vals)

        import numpy as np

        arr = np.asarray(data, dtype=float)
        # per-column min/max
        mins = np.nanmin(arr, axis=0)
        maxs = np.nanmax(arr, axis=0)
        denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
        norm = (arr - mins) / denom

        xs = np.arange(len(metrics))
        for i in range(norm.shape[0]):
            ax.plot(xs, norm[i, :], alpha=0.6, linewidth=1)

        ax.set_xticks(xs)
        ax.set_xticklabels(metrics, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Parallel Coordinates (normalized)")
        ax.grid(True, alpha=0.25)
        self.fig_parallel.canvas.draw_idle()

    def _draw_visuals(self, model_name: str) -> None:
        payload = self._payload or {}
        task = str(payload.get("task_type", ""))
        plots = (payload.get("plots", {}) or {})
        p = plots.get(model_name) or {}

        self.fig_main.figure.clear()
        self.fig_secondary.figure.clear()
        self.fig_tertiary.figure.clear()

        ax1 = self.fig_main.figure.add_subplot(111)
        ax2 = self.fig_secondary.figure.add_subplot(111)

        need_3d = bool(task in ("clustering", "dimred") and (p.get("proj3", None) is not None))
        ax3 = self.fig_tertiary.figure.add_subplot(111, projection="3d") if need_3d else self.fig_tertiary.figure.add_subplot(111)

        import numpy as np

        if task == "regression":
            y_true = np.asarray(p.get("y_true", []) or [])
            y_pred = np.asarray(p.get("y_pred", []) or [])
            if y_true.size and y_pred.size:
                resid = y_true - y_pred
                ax1.scatter(y_pred, resid, s=10, alpha=0.6)
                ax1.axhline(0, color="black", linewidth=1, alpha=0.4)
                ax1.set_title("Residuals vs Predicted")
                ax1.set_xlabel("Predicted")
                ax1.set_ylabel("Residual")

                ax2.scatter(y_true, y_pred, s=10, alpha=0.6)
                lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
                hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
                ax2.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.4)
                ax2.set_title("Predicted vs Actual")
                ax2.set_xlabel("Actual")
                ax2.set_ylabel("Predicted")

                ax3.hist(resid, bins=30, alpha=0.7)
                ax3.set_title("Error distribution (residuals)")
                ax3.set_xlabel("Residual")
                ax3.set_ylabel("Count")

        elif task == "classification":
            cm = p.get("confusion_matrix")
            if cm is not None:
                cm = np.asarray(cm)
                ax1.imshow(cm, cmap="Blues")
                ax1.set_title("Confusion Matrix")
                ax1.set_xlabel("Pred")
                ax1.set_ylabel("True")

            roc = p.get("roc") or {}
            fpr = np.asarray(roc.get("fpr", []) or [])
            tpr = np.asarray(roc.get("tpr", []) or [])
            if fpr.size and tpr.size:
                ax2.plot(fpr, tpr, linewidth=2)
                ax2.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4)
                ax2.set_title("ROC Curve")
                ax2.set_xlabel("FPR")
                ax2.set_ylabel("TPR")

            pr = p.get("pr") or {}
            prec = np.asarray(pr.get("precision", []) or [])
            rec = np.asarray(pr.get("recall", []) or [])
            if prec.size and rec.size:
                ax3.plot(rec, prec, linewidth=2)
                ax3.set_title("Precision-Recall Curve")
                ax3.set_xlabel("Recall")
                ax3.set_ylabel("Precision")

        elif task == "clustering":
            pts = np.asarray(p.get("proj", []) or [])
            labels = np.asarray(p.get("labels", []) or [])
            if pts.size and labels.size and pts.shape[1] >= 2:
                ax1.scatter(pts[:, 0], pts[:, 1], c=labels, s=10, alpha=0.7)
                ax1.set_title("Cluster visualization (PCA 2D)")
                ax1.set_xlabel("PC1")
                ax1.set_ylabel("PC2")

            dendro = p.get("dendrogram")
            if dendro is not None:
                # dendro contains {icoord, dcoord, color_list}
                try:
                    for xs, ys, col in zip(dendro.get("icoord", []), dendro.get("dcoord", []), dendro.get("color_list", [])):
                        ax2.plot(xs, ys, color=col)
                    ax2.set_title("Dendrogram")
                except Exception:
                    pass

            pts3 = np.asarray(p.get("proj3", []) or [])
            if pts3.size and pts3.shape[1] >= 3:
                ax3.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], c=labels, s=8, alpha=0.7)
                ax3.set_title("Cluster visualization (PCA 3D)")

        elif task == "dimred":
            pts = np.asarray(p.get("proj", []) or [])
            if pts.size and pts.shape[1] >= 2:
                ax1.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7)
                ax1.set_title("2D Projection")
                ax1.set_xlabel("Comp1")
                ax1.set_ylabel("Comp2")
            imp = p.get("component_importance")
            if imp is not None:
                vals = np.asarray(imp)
                ax2.bar(range(len(vals)), vals)
                ax2.set_title("Component importance")
                ax2.set_xlabel("Component")

            pts3 = np.asarray(p.get("proj3", []) or [])
            if pts3.size and pts3.shape[1] >= 3:
                ax3.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], s=8, alpha=0.7)
                ax3.set_title("3D Projection")

        self.fig_main.canvas.draw_idle()
        self.fig_secondary.canvas.draw_idle()
        self.fig_tertiary.canvas.draw_idle()

    def get_payload(self) -> Dict[str, Any]:
        return dict(self._payload or {})

    def export_figs(self, fmt: str) -> List[str]:
        fmt = (fmt or "png").lower()
        if fmt not in ("png", "svg"):
            return []

        out_dir = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not out_dir:
            return []

        model = None
        if self.list_models.currentItem() is not None:
            model = str(self.list_models.currentItem().text())
        safe_model = (model or "dashboard").replace(" ", "_")

        paths: List[str] = []
        try:
            p1 = f"{out_dir}/comparison_primary_{safe_model}.{fmt}"
            self.fig_main.figure.savefig(p1, bbox_inches="tight")
            paths.append(p1)
        except Exception:
            pass
        try:
            p2 = f"{out_dir}/comparison_secondary_{safe_model}.{fmt}"
            self.fig_secondary.figure.savefig(p2, bbox_inches="tight")
            paths.append(p2)
        except Exception:
            pass
        try:
            p3 = f"{out_dir}/comparison_tertiary_{safe_model}.{fmt}"
            self.fig_tertiary.figure.savefig(p3, bbox_inches="tight")
            paths.append(p3)
        except Exception:
            pass
        try:
            p4 = f"{out_dir}/comparison_parallel.{fmt}"
            self.fig_parallel.figure.savefig(p4, bbox_inches="tight")
            paths.append(p4)
        except Exception:
            pass
        try:
            p5 = f"{out_dir}/comparison_feature_importance_{safe_model}.{fmt}"
            self.fig_feat.figure.savefig(p5, bbox_inches="tight")
            paths.append(p5)
        except Exception:
            pass

        return paths

    def _draw_feature_importance(self, model_name: str) -> None:
        payload = self._payload or {}
        fi = (payload.get("feature_importance", {}) or {}).get(model_name)
        fig = self.fig_feat.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if not fi:
            ax.set_title("No feature importance available")
            self.fig_feat.canvas.draw_idle()
            return

        items = sorted(list(fi.items()), key=lambda x: abs(float(x[1])), reverse=True)[:15]
        names = [i[0] for i in items][::-1]
        vals = [float(i[1]) for i in items][::-1]
        ax.barh(range(len(names)), vals)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_title(f"Top features: {model_name}")
        ax.grid(True, axis="x", alpha=0.25)
        self.fig_feat.canvas.draw_idle()

    def _set_insights(self, model_name: str) -> None:
        payload = self._payload or {}
        insights = (payload.get("insights", {}) or {}).get(model_name) or ""
        self.txt_insights.setText(str(insights) or "No insights available")

    # Backwards compatibility
    def set_comparison(self, rows: List[Dict[str, Any]]) -> None:
        self.set_dashboard_payload({"rows": rows, "task_type": "", "primary_metric_key": ""})
