"""Evaluation & Comparison Module controller.

Covers Tab 4 (Results & Comparison).
"""

from __future__ import annotations

import pickle
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from PySide6.QtCore import Signal

from .base_controller import BaseController
from ..models import DataModel, MLModel, EvaluationModel
from ..state import AppState
from ..utils import get_logger


class EvaluationComparisonController(BaseController):
    comparison_ready = Signal(object)  # list[dict]
    error_occurred = Signal(str)

    def __init__(self, state: AppState, data_model: DataModel, ml_model: MLModel, evaluation_model: EvaluationModel):
        super().__init__()
        self._log = get_logger("controllers.evaluation")

        self.state = state
        self.data_model = data_model
        self.ml_model = ml_model
        self.evaluation_model = evaluation_model

        self.evaluation_model.error_occurred.connect(self.error_occurred)
        self.ml_model.error_occurred.connect(self.error_occurred)

        self.target_column: Optional[str] = None
        self.problem_type: str = "classification"  # classification/regression/clustering/dimred

        self._last_split: Optional[Tuple[Any, Any, Any, Any]] = None
        self._last_dataset_sig: Optional[str] = None

    def set_target(self, column: str, problem_type: str = "classification") -> None:
        self.target_column = column
        self.problem_type = problem_type

    def _dataset_signature(self, df) -> str:
        # Cheap signature for caching: shape + column names + missing%
        try:
            import hashlib
            import json

            payload = {
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "cols": [str(c) for c in df.columns.tolist()],
                "missing": float(df.isna().mean().mean()),
            }
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            return hashlib.md5(raw).hexdigest()
        except Exception:
            return ""

    def evaluate_all(self) -> bool:
        df = self.data_model.get_data()
        if df is None or df.empty:
            self.error_occurred.emit("No dataset available for evaluation")
            return False

        task = (self.problem_type or "classification").lower()
        if task in ("classification", "regression"):
            if not self.target_column or self.target_column not in df.columns:
                self.error_occurred.emit("Target column not set (or missing)")
                return False

        try:
            dataset_sig = self._dataset_signature(df)
            self._last_dataset_sig = dataset_sig

            X_train = X_test = y_train = y_test = None
            if task in ("classification", "regression"):
                X = df.drop(columns=[self.target_column])
                y = df[self.target_column]

                from sklearn.model_selection import train_test_split

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    shuffle=True,
                )
                self._last_split = (X_train, X_test, y_train, y_test)
            else:
                # Unsupervised tasks use all features
                X = df
                self._last_split = None

            # Ensure models are trained; if not, train quickly.
            for name in self.ml_model.list_models():
                wrapper = self.ml_model.get_model(name)
                if wrapper is None:
                    continue
                # Skip models that don't match task_type (best-effort)
                try:
                    if getattr(wrapper, "task_type", None) and str(wrapper.task_type).lower() != task:
                        continue
                except Exception:
                    pass

                # Train if needed
                if not getattr(wrapper, "trained", False):
                    if task in ("classification", "regression"):
                        self.ml_model.train_model(name, X_train, y_train)
                    else:
                        self.ml_model.fit_model(name, X)

                # operational metrics
                training_time_s = getattr(wrapper, "training_time", None)
                model_size_mb = None
                try:
                    b = pickle.dumps(wrapper)
                    model_size_mb = float(len(b) / (1024 * 1024))
                except Exception:
                    pass

                # Evaluate by task
                result = None
                if task == "regression":
                    t0 = time.perf_counter()
                    y_pred = self.ml_model.predict(name, X_test)
                    t1 = time.perf_counter()
                    if y_pred is None:
                        continue
                    inference_ms = float((t1 - t0) * 1000.0)
                    result = self.evaluation_model.evaluate_regression(
                        name,
                        np.asarray(y_test),
                        np.asarray(y_pred),
                        n_features=int(X_test.shape[1]) if hasattr(X_test, "shape") else None,
                        training_time_s=training_time_s,
                        inference_time_ms=inference_ms,
                        model_size_mb=model_size_mb,
                        dataset_sig=dataset_sig,
                    )

                elif task == "classification":
                    t0 = time.perf_counter()
                    y_pred = self.ml_model.predict(name, X_test)
                    t1 = time.perf_counter()
                    if y_pred is None:
                        continue
                    inference_ms = float((t1 - t0) * 1000.0)
                    y_proba = self.ml_model.predict_proba(name, X_test)
                    result = self.evaluation_model.evaluate_classification(
                        name,
                        np.asarray(y_test),
                        np.asarray(y_pred),
                        None if y_proba is None else np.asarray(y_proba),
                        training_time_s=training_time_s,
                        inference_time_ms=inference_ms,
                        model_size_mb=model_size_mb,
                        dataset_sig=dataset_sig,
                    )

                elif task == "clustering":
                    # cluster labels
                    labels = None
                    try:
                        est = wrapper.model
                        if hasattr(est, "predict"):
                            labels = est.predict(X)
                        elif hasattr(est, "fit_predict"):
                            labels = est.fit_predict(X)
                    except Exception:
                        labels = None
                    if labels is None:
                        continue
                    result = self.evaluation_model.evaluate_clustering(
                        name,
                        np.asarray(X),
                        np.asarray(labels),
                        training_time_s=training_time_s,
                        inference_time_ms=None,
                        model_size_mb=model_size_mb,
                        dataset_sig=dataset_sig,
                    )

                elif task == "dimred":
                    est = wrapper.model
                    Xr = None
                    evr = None
                    rec_err = None
                    try:
                        if hasattr(est, "transform"):
                            Xr = est.transform(X)
                        elif hasattr(est, "fit_transform"):
                            Xr = est.fit_transform(X)
                    except Exception:
                        Xr = None
                    if Xr is None:
                        continue

                    try:
                        evr = getattr(est, "explained_variance_ratio_", None)
                    except Exception:
                        evr = None
                    try:
                        if hasattr(est, "inverse_transform"):
                            Xhat = est.inverse_transform(Xr)
                            rec_err = float(np.mean((np.asarray(X) - np.asarray(Xhat)) ** 2))
                    except Exception:
                        rec_err = None

                    result = self.evaluation_model.evaluate_dimred(
                        name,
                        np.asarray(X),
                        np.asarray(Xr),
                        explained_variance_ratio=None if evr is None else np.asarray(evr),
                        reconstruction_error=rec_err,
                        training_time_s=training_time_s,
                        inference_time_ms=None,
                        model_size_mb=model_size_mb,
                        dataset_sig=dataset_sig,
                    )

                if result is not None:
                    self.state.upsert_evaluation_result(name, result.to_dict())

            self.state.save()
            return True
        except Exception as exc:
            self.error_occurred.emit(f"Evaluation failed: {exc}")
            return False

    def compare(self) -> Optional[list[Dict[str, Any]]]:
        df = self.evaluation_model.compare_models()
        if df is None:
            return None
        raw_rows = df.to_dict(orient="records")
        rows: list[Dict[str, Any]] = [{str(k): v for k, v in r.items()} for r in raw_rows]
        self.comparison_ready.emit(rows)
        self._log.info("Comparison generated for %d models", len(rows))
        return rows

    def build_dashboard_payload(self) -> Dict[str, Any]:
        """Return a rich payload for the Tab 4 dashboard."""
        task = (self.problem_type or "classification").lower()
        rows_raw = self.compare() or []

        # Decide primary/secondary metrics
        primary_key = ""
        secondary_key = ""
        higher_is_better = True
        if task == "classification":
            primary_key = "f1_score"
            secondary_key = "roc_auc"
            higher_is_better = True
        elif task == "regression":
            primary_key = "rmse"
            secondary_key = "mae"
            higher_is_better = False
        elif task == "clustering":
            primary_key = "silhouette"
            secondary_key = "davies_bouldin"
            higher_is_better = True
        elif task == "dimred":
            primary_key = "explained_variance_ratio_sum"
            secondary_key = "reconstruction_error"
            higher_is_better = True

        # Build required table columns + keep computed metrics
        rows: list[Dict[str, Any]] = []
        for r in rows_raw:
            model = str(r.get("Model", ""))
            row = {
                "Model Name": model,
                "Training Time": r.get("Training Time (s)", None),
                "Inference Time": r.get("Inference Time (ms)", None),
                "Primary Metric": r.get(primary_key, None),
                "Secondary Metric": r.get(secondary_key, None),
                "Model Size": r.get("Model Size (MB)", None),
            }

            # Attach full metrics (flattened)
            for k, v in r.items():
                if k in row:
                    continue
                row[str(k)] = v
            rows.append(row)

        # Find best row by primary
        best_row = None
        best_val = None
        for i, r in enumerate(rows):
            try:
                v = float(r.get("Primary Metric"))
            except Exception:
                continue
            if best_val is None:
                best_val = v
                best_row = i
            else:
                if higher_is_better and v > best_val:
                    best_val, best_row = v, i
                if (not higher_is_better) and v < best_val:
                    best_val, best_row = v, i

        if best_row is not None:
            for i, r in enumerate(rows):
                r["Best Model"] = "BEST" if i == best_row else ""

        dataset_sig = self._last_dataset_sig or ""

        # Statistical testing vs best (paired t-test on per-sample losses)
        stats_lines = []
        stats: Dict[str, Any] = {}
        if best_row is not None and dataset_sig:
            best_name = str(rows[best_row].get("Model Name"))
            pvals = []
            tmp = []
            for r in rows:
                name = str(r.get("Model Name"))
                if not name or name == best_name:
                    continue
                res = self.evaluation_model.paired_ttest(dataset_sig=dataset_sig, task_type=task, a=name, b=best_name)
                if not res or "p" not in res:
                    continue
                pvals.append(float(res["p"]))
                tmp.append((name, res))

            m = max(1, len(pvals))
            for name, res in tmp:
                p = float(res.get("p", 1.0))
                p_adj = min(1.0, p * m)  # Bonferroni
                stats[name] = {**res, "p_adj": p_adj}

            # attach significance column
            for r in rows:
                name = str(r.get("Model Name"))
                if name == str(rows[best_row].get("Model Name")):
                    r["Sig vs Best"] = ""
                else:
                    s = stats.get(name)
                    if not s:
                        r["Sig vs Best"] = ""
                    else:
                        r["Sig vs Best"] = "*" if float(s.get("p_adj", 1.0)) < 0.05 else ""

            stats_lines.append(f"Best model: {best_name} (Primary Metric: {primary_key})")
            stats_lines.append(f"Paired t-tests vs best; Bonferroni-adjusted (m={m}).")
            sig_count = sum(1 for v in stats.values() if float(v.get("p_adj", 1.0)) < 0.05)
            stats_lines.append(f"Significant differences detected: {sig_count}")

        # Build plot artifacts per model
        plots: Dict[str, Any] = {}
        feature_importance: Dict[str, Dict[str, float]] = {}
        insights: Dict[str, str] = {}

        # Grab split if supervised
        split = self._last_split
        X_train = X_test = y_train = y_test = None
        if split is not None:
            X_train, X_test, y_train, y_test = split

        # Snapshot results to avoid "dictionary changed size during iteration" if evaluations
        # are still being recorded while building the dashboard payload.
        results_snapshot = dict(self.evaluation_model.results or {})
        for name, result in results_snapshot.items():
            if str(getattr(result, "task_type", "")).lower() != task:
                continue

            p: Dict[str, Any] = {}
            try:
                if task in ("regression", "classification"):
                    if result.y_true is not None:
                        p["y_true"] = np.asarray(result.y_true).tolist()
                    if result.y_pred is not None:
                        p["y_pred"] = np.asarray(result.y_pred).tolist()
            except Exception:
                pass

            # classification curves
            if task == "classification":
                try:
                    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

                    if result.y_true is not None and result.y_pred is not None:
                        cm = confusion_matrix(result.y_true, result.y_pred)
                        p["confusion_matrix"] = cm.tolist()

                    if result.y_true is not None and result.y_proba is not None:
                        y_proba = np.asarray(result.y_proba)
                        # binary only for curves
                        if y_proba.ndim == 1:
                            proba_1 = y_proba
                        elif y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                            proba_1 = y_proba[:, 1]
                        else:
                            proba_1 = None

                        if proba_1 is not None:
                            fpr, tpr, _ = roc_curve(result.y_true, proba_1)
                            p["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                            prec, rec, _ = precision_recall_curve(result.y_true, proba_1)
                            p["pr"] = {"precision": prec.tolist(), "recall": rec.tolist()}
                except Exception:
                    pass

            # clustering projections + dendrogram
            if task == "clustering":
                try:
                    labels = np.asarray(result.y_pred) if result.y_pred is not None else None
                    if labels is not None:
                        p["labels"] = labels.tolist()

                    df = self.data_model.get_data()
                    if df is not None and not df.empty:
                        X = df.select_dtypes(include=["number"]).to_numpy()
                        from sklearn.decomposition import PCA

                        if X.shape[1] >= 2:
                            pts2 = PCA(n_components=2, random_state=42).fit_transform(X)
                            p["proj"] = pts2.tolist()
                        if X.shape[1] >= 3:
                            pts3 = PCA(n_components=3, random_state=42).fit_transform(X)
                            p["proj3"] = pts3.tolist()

                        # dendrogram on a small sample
                        try:
                            from scipy.cluster.hierarchy import linkage, dendrogram

                            n = min(200, X.shape[0])
                            Xs = X[:n]
                            Z = linkage(Xs, method="ward")
                            d = dendrogram(Z, no_plot=True)
                            p["dendrogram"] = {
                                "icoord": d.get("icoord", []),
                                "dcoord": d.get("dcoord", []),
                                "color_list": d.get("color_list", []),
                            }
                        except Exception:
                            pass
                except Exception:
                    pass

            # dimred projections
            if task == "dimred":
                try:
                    Xr = np.asarray(result.y_pred) if result.y_pred is not None else None
                    if Xr is not None and Xr.ndim == 2:
                        if Xr.shape[1] >= 2:
                            p["proj"] = Xr[:, :2].tolist()
                        if Xr.shape[1] >= 3:
                            p["proj3"] = Xr[:, :3].tolist()

                    # component importance (PCA-like)
                    wrapper = self.ml_model.get_model(name)
                    if wrapper is not None:
                        evr = getattr(wrapper.model, "explained_variance_ratio_", None)
                        if evr is not None:
                            p["component_importance"] = np.asarray(evr).tolist()
                except Exception:
                    pass

            plots[name] = p

            # Feature importance
            try:
                wrapper = self.ml_model.get_model(name)
                if wrapper is not None:
                    est = wrapper.model
                    fnames = list(getattr(wrapper, "feature_names", []) or [])
                    imp: Dict[str, float] = {}
                    if hasattr(est, "feature_importances_"):
                        vals = np.asarray(getattr(est, "feature_importances_"))
                        for i, v in enumerate(vals.tolist()):
                            key = fnames[i] if i < len(fnames) else f"f{i}"
                            imp[str(key)] = float(v)
                    elif hasattr(est, "coef_"):
                        coef = np.asarray(getattr(est, "coef_"))
                        coef = coef.ravel()
                        for i, v in enumerate(coef.tolist()):
                            key = fnames[i] if i < len(fnames) else f"f{i}"
                            imp[str(key)] = float(v)
                    if imp:
                        feature_importance[name] = imp
            except Exception:
                pass

            # Insights
            insights[name] = self._build_model_insight(name=name, task=task, primary_key=primary_key)

        payload = {
            "task_type": task,
            "rows": rows,
            "primary_metric_key": "Primary Metric",
            "primary_metric_name": primary_key,
            "primary_higher_is_better": higher_is_better,
            "best_row": best_row,
            "dataset_sig": dataset_sig,
            "parallel_metrics": self._parallel_metrics(task, rows),
            "stats": stats,
            "stats_summary": "\n".join(stats_lines),
            "plots": plots,
            "feature_importance": feature_importance,
            "insights": insights,
        }
        return payload

    def _parallel_metrics(self, task: str, rows: list[Dict[str, Any]]) -> list[str]:
        # pick a small set of numeric metrics for the parallel plot
        if task == "classification":
            return ["Primary Metric", "Secondary Metric", "accuracy", "precision", "recall", "f1_score"]
        if task == "regression":
            return ["Primary Metric", "Secondary Metric", "r2", "adj_r2", "mape", "msle"]
        if task == "clustering":
            return ["Primary Metric", "Secondary Metric", "silhouette", "davies_bouldin", "calinski_harabasz"]
        if task == "dimred":
            return ["Primary Metric", "Secondary Metric", "explained_variance_ratio_sum", "reconstruction_error"]
        return ["Primary Metric", "Secondary Metric"]

    def _build_model_insight(self, *, name: str, task: str, primary_key: str) -> str:
        r = self.evaluation_model.get_result(name)
        if not r:
            return ""
        lines = []
        lines.append(f"Model: {name}")
        lines.append(f"Task: {task}")
        if r.training_time_s is not None:
            lines.append(f"Training time: {r.training_time_s:.3f}s")
        if r.inference_time_ms is not None:
            lines.append(f"Inference time: {r.inference_time_ms:.3f}ms")
        if r.model_size_mb is not None:
            lines.append(f"Model size: {r.model_size_mb:.3f}MB")

        # performance summary
        pm = r.metrics.get(primary_key)
        if pm is not None:
            lines.append(f"Primary metric ({primary_key}): {pm}")

        # overfit/underfit: compare train vs test on primary metric when we have split
        if task in ("classification", "regression") and self._last_split is not None:
            try:
                X_train, X_test, y_train, y_test = self._last_split
                yhat_train = self.ml_model.predict(name, X_train)
                if yhat_train is not None:
                    if task == "regression":
                        tr = self.evaluation_model.evaluate_regression(
                            f"{name}__train_tmp",
                            np.asarray(y_train),
                            np.asarray(yhat_train),
                            n_features=int(X_train.shape[1]),
                        )
                        if tr and "rmse" in tr.metrics and "rmse" in r.metrics:
                            gap = float(tr.metrics["rmse"]) - float(r.metrics["rmse"])
                            lines.append(f"Overfit check (train_rmse - test_rmse): {gap:.4f}")
                    else:
                        # use accuracy gap
                        from sklearn.metrics import accuracy_score

                        train_acc = float(accuracy_score(y_train, yhat_train))
                        test_acc = float(r.metrics.get("accuracy", 0.0))
                        lines.append(f"Overfit check (train_acc - test_acc): {(train_acc - test_acc):.4f}")
            except Exception:
                pass

        lines.append("Limitations: Metrics are computed on a single holdout split by default.")
        lines.append("Tip: Use tuning/ensembles if the top models are close.")
        return "\n".join(lines)

    def export_table(self, payload: Dict[str, Any], fmt: str) -> Optional[str]:
        try:
            from pathlib import Path

            root = Path.home() / ".arcsaathi" / "reports"
            root.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time())
            rows = list((payload or {}).get("rows", []) or [])
            if not rows:
                return None
            import pandas as pd

            df = pd.DataFrame(rows)
            if (fmt or "").lower() == "csv":
                path = str(root / f"model_comparison_{stamp}.csv")
                df.to_csv(path, index=False)
                return path
            if (fmt or "").lower() in ("xlsx", "excel"):
                path = str(root / f"model_comparison_{stamp}.xlsx")
                df.to_excel(path, index=False)
                return path
            return None
        except Exception:
            return None

    def export_report(self, payload: Dict[str, Any], fmt: str) -> Optional[str]:
        fmt = (fmt or "html").lower()
        try:
            from pathlib import Path

            root = Path.home() / ".arcsaathi" / "reports"
            root.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time())
            if fmt == "html":
                path = str(root / f"model_comparison_{stamp}.html")
                Path(path).write_text(self._render_html(payload), encoding="utf-8")
                return path
            if fmt == "pdf":
                path = str(root / f"model_comparison_{stamp}.pdf")
                self._render_pdf(payload, path)
                return path
            return None
        except Exception:
            return None

    def _render_html(self, payload: Dict[str, Any]) -> str:
        rows = list((payload or {}).get("rows", []) or [])
        if not rows:
            return "<html><body><h1>No results</h1></body></html>"
        headers = list(rows[0].keys())
        def esc(x: Any) -> str:
            s = str(x)
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        th = "".join([f"<th>{esc(h)}</th>" for h in headers])
        tr = "".join([
            "<tr>" + "".join([f"<td>{esc(r.get(h,''))}</td>" for h in headers]) + "</tr>" for r in rows
        ])
        return f"""<!doctype html>
<html><head><meta charset='utf-8'/><title>ARCSaathi Model Comparison</title>
<style>body{{font-family:Arial;margin:24px}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:6px}} th{{background:#f5f5f5}}</style>
</head><body>
<h1>Model Comparison</h1>
<pre>{esc(payload.get('stats_summary',''))}</pre>
<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>
</body></html>"""

    def _render_pdf(self, payload: Dict[str, Any], out_path: str) -> None:
        # ReportLab-based minimal PDF
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(out_path, pagesize=A4)
        w, h = A4
        y = h - 2 * cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, y, "ARCSaathi Model Comparison")
        y -= 0.7 * cm
        c.setFont("Helvetica", 10)
        stats = str((payload or {}).get("stats_summary", ""))
        for line in stats.splitlines()[:8]:
            c.drawString(2 * cm, y, line)
            y -= 0.5 * cm

        rows = list((payload or {}).get("rows", []) or [])
        if rows:
            y -= 0.3 * cm
            c.setFont("Helvetica-Bold", 11)
            c.drawString(2 * cm, y, "Top rows (first 10)")
            y -= 0.6 * cm
            c.setFont("Helvetica", 8)
            headers = list(rows[0].keys())[:8]
            c.drawString(2 * cm, y, " | ".join(headers))
            y -= 0.5 * cm
            for r in rows[:10]:
                c.drawString(2 * cm, y, " | ".join([str(r.get(h, ""))[:16] for h in headers]))
                y -= 0.45 * cm
                if y < 2 * cm:
                    c.showPage()
                    y = h - 2 * cm
                    c.setFont("Helvetica", 8)
        c.save()
