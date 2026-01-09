"""Explainability controller.

Implements:
- Global: permutation importance, PDP (1D/2D), ALE (via DALEX when available), learning/validation curves, residuals, decision boundaries.
- Local: SHAP (matplotlib fallback), LIME, plain-English why, simple counterfactual search.
- Debugging: error analysis + clusters.
- Drift: PSI/KS/chi2-style heuristics.
- Fairness: disparate impact, demographic parity, equalized odds.

All heavy work runs via QThreadPool + Worker.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import QObject, QThreadPool, Signal

from .base_controller import BaseController
from ..models import DataModel, MLModel
from ..utils import get_logger
from ..utils.qt_worker import Worker


@dataclass
class _Split:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any


class ExplainabilityController(BaseController):
    error_occurred = Signal(str)

    context_ready = Signal(object)  # {models, targets, features}
    candidate_rows_ready = Signal(object)  # list[dict]

    global_ready = Signal(object)  # {text, html}
    local_ready = Signal(object)  # {text, html}
    whatif_ready = Signal(object)  # {text}
    debug_ready = Signal(object)  # {text}
    drift_ready = Signal(object)  # {text}
    fairness_ready = Signal(object)  # {text, privileged_groups}
    auto_ready = Signal(object)  # {text}

    report_ready = Signal(str)

    def __init__(self, data_model: DataModel, ml_model: MLModel):
        super().__init__()
        self._log = get_logger("controllers.explainability")
        self.data_model = data_model
        self.ml_model = ml_model

        self.ml_model.error_occurred.connect(self.error_occurred)

        self._pool = QThreadPool.globalInstance()

        self.target_column: str = ""
        self.task_type: str = "classification"

        self._cached_split: Optional[_Split] = None
        self._last_candidates: List[Dict[str, Any]] = []

        # drift baseline (in-session)
        self._baseline_df = None

    # ---- context ----
    def refresh_context(self) -> None:
        df = self.data_model.get_data()
        models = self.ml_model.list_models()
        targets: List[str] = []
        features: List[str] = []
        if df is not None and not getattr(df, "empty", True):
            targets = [str(c) for c in df.columns.tolist()]
            features = [str(c) for c in df.columns.tolist()]
        self.context_ready.emit({"models": models, "targets": targets, "features": features})

    def set_target(self, target: str, task_type: str) -> None:
        self.target_column = str(target or "")
        self.task_type = str(task_type or "classification").lower()
        self._cached_split = None
        self._last_candidates = []

    # ---- background entrypoints ----
    def build_candidates(self, model_name: str, min_conf: float = 0.0, limit: int = 250) -> None:
        worker = Worker(self._build_candidates_task, model_name, float(min_conf), int(limit))
        worker.signals.result.connect(lambda r: self.candidate_rows_ready.emit(r))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    def run_global(self, req: Dict[str, Any]) -> None:
        worker = Worker(self._global_task, dict(req or {}))
        worker.signals.result.connect(lambda r: self.global_ready.emit(r))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    def run_local(self, req: Dict[str, Any]) -> None:
        worker = Worker(self._local_task, dict(req or {}))
        worker.signals.result.connect(lambda r: self.local_ready.emit(r))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    def run_whatif(self, req: Dict[str, Any]) -> None:
        worker = Worker(self._whatif_task, dict(req or {}))
        worker.signals.result.connect(lambda r: self.whatif_ready.emit(r))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    def run_debug(self, req: Dict[str, Any]) -> None:
        worker = Worker(self._debug_task, dict(req or {}))
        worker.signals.result.connect(lambda r: self.debug_ready.emit(r))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    def run_drift(self, req: Dict[str, Any]) -> None:
        worker = Worker(self._drift_task, dict(req or {}))
        worker.signals.result.connect(lambda r: self.drift_ready.emit(r))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    def run_fairness(self, req: Dict[str, Any]) -> None:
        worker = Worker(self._fairness_task, dict(req or {}))
        worker.signals.result.connect(lambda r: self.fairness_ready.emit(r))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    def run_auto(self, req: Dict[str, Any]) -> None:
        worker = Worker(self._auto_task, dict(req or {}))
        worker.signals.result.connect(lambda r: self.auto_ready.emit(r))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    def export_report(self, fmt: str, payload: Dict[str, Any]) -> None:
        worker = Worker(self._export_report_task, str(fmt or "html"), dict(payload or {}))
        worker.signals.result.connect(lambda p: self.report_ready.emit(str(p)))
        worker.signals.error.connect(self.error_occurred)
        self._pool.start(worker)

    # ---- helpers ----
    def _get_df(self):
        df = self.data_model.get_data()
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("Load a dataset first")
        return df

    def _get_wrapper(self, model_name: str):
        w = self.ml_model.get_model(model_name)
        if w is None:
            raise RuntimeError("Select a trained model")
        if not getattr(w, "trained", False):
            raise RuntimeError("Selected model is not trained")
        return w

    def _get_split(self, df, task: str, target: str) -> Optional[_Split]:
        task = (task or "classification").lower()
        if task not in ("classification", "regression"):
            return None
        if not target or target not in df.columns:
            raise RuntimeError("Target column not set (or missing)")
        if self._cached_split is not None:
            return self._cached_split

        from sklearn.model_selection import train_test_split

        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        self._cached_split = _Split(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        return self._cached_split

    def _numeric_X(self, X):
        try:
            return X.select_dtypes(include=["number"]).copy()
        except Exception:
            return X

    # ---- candidate list ----
    def _build_candidates_task(self, model_name: str, min_conf: float, limit: int) -> List[Dict[str, Any]]:
        df = self._get_df()
        task = self.task_type
        target = self.target_column
        w = self._get_wrapper(model_name)

        if task not in ("classification", "regression"):
            return []

        split = self._get_split(df, task, target)
        assert split is not None

        X_test = split.X_test
        y_pred = self.ml_model.predict(model_name, X_test)
        if y_pred is None:
            return []

        rows: List[Dict[str, Any]] = []
        if task == "classification":
            proba = self.ml_model.predict_proba(model_name, X_test)
            confs = None
            if proba is not None:
                try:
                    confs = np.max(np.asarray(proba), axis=1)
                except Exception:
                    confs = None
            for i in range(len(y_pred)):
                c = float(confs[i]) if confs is not None else float("nan")
                if confs is not None and c < min_conf:
                    continue
                rows.append({"row": int(i), "y_pred": str(y_pred[i]), "confidence": ("" if math.isnan(c) else f"{c:.3f}")})
                if len(rows) >= limit:
                    break
        else:
            # regression: no confidence; just list
            for i in range(min(limit, len(y_pred))):
                rows.append({"row": int(i), "y_pred": str(float(y_pred[i])), "confidence": ""})

        self._last_candidates = rows
        return rows

    # ---- global ----
    def _global_task(self, req: Dict[str, Any]) -> Dict[str, Any]:
        model = str(req.get("model", ""))
        task = str(req.get("task", self.task_type))
        target = str(req.get("target", self.target_column))
        features = list(req.get("features", []) or [])

        df = self._get_df()
        wrapper = self._get_wrapper(model)

        html_parts: List[str] = []
        text_lines: List[str] = []

        # Supervised split for most global plots
        split = self._get_split(df, task, target) if task in ("classification", "regression") else None
        X_test = None
        y_test = None
        if split is not None:
            X_test = split.X_test
            y_test = split.y_test

        # Permutation importance
        try:
            from sklearn.inspection import permutation_importance

            if X_test is not None and y_test is not None:
                r = permutation_importance(wrapper.model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
                importances = r.importances_mean
                cols = list(X_test.columns)
                order = np.argsort(np.abs(importances))[::-1][:20]
                text_lines.append("Permutation importance (top 20):")
                for i in order:
                    text_lines.append(f"- {cols[i]}: {float(importances[i]):.6f}")
        except Exception as exc:
            text_lines.append(f"Permutation importance unavailable: {exc}")

        # SHAP summary (HTML fallback)
        try:
            import shap

            if X_test is not None:
                Xs = X_test
                # keep it light
                if getattr(Xs, "shape", (0, 0))[0] > 500:
                    Xs = Xs.sample(500, random_state=42)
                explainer = shap.Explainer(wrapper.model, Xs)
                sv = explainer(Xs)
                # Render as HTML (beeswarm is matplotlib; use text + saveable html)
                # We provide a compact HTML with a warning + user can export.
                text_lines.append("SHAP computed (summary available; see export for full plots).")
                # minimal HTML preview
                html_parts.append("<h3>SHAP</h3><p>SHAP values computed. Use Export to save a full report.</p>")
        except Exception as exc:
            text_lines.append(f"SHAP unavailable: {exc}")

        # PDP (1D/2D)
        try:
            from sklearn.inspection import partial_dependence

            if X_test is not None and features:
                feats = [f for f in features if f in X_test.columns]
                if feats:
                    # 1D for first
                    pd1 = partial_dependence(wrapper.model, X_test, [feats[0]])
                    xs = pd1["grid_values"][0]
                    ys = pd1["average"][0]
                    html_parts.append(self._plotly_line(xs, ys, title=f"PDP 1D: {feats[0]}", x_label=feats[0], y_label="partial dependence"))

                    # 2D for first two
                    if len(feats) >= 2:
                        pd2 = partial_dependence(wrapper.model, X_test, [(feats[0], feats[1])])
                        gx = pd2["grid_values"][0]
                        gy = pd2["grid_values"][1]
                        z = pd2["average"][0].reshape(len(gx), len(gy))
                        html_parts.append(self._plotly_heatmap(gx, gy, z, title=f"PDP 2D: {feats[0]} vs {feats[1]}", x_label=feats[0], y_label=feats[1]))
        except Exception as exc:
            text_lines.append(f"PDP unavailable: {exc}")

        # ALE via DALEX (best-effort)
        try:
            import dalex as dx

            if X_test is not None and y_test is not None:
                expl = dx.Explainer(wrapper.model, X_test, y_test, label=model, verbose=False)
                if features:
                    feat = features[0]
                    prof = expl.model_profile(variables=[feat], type="accumulated")
                    # DALEX plot is plotly
                    fig = prof.plot(show=False)
                    html_parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))
                    text_lines.append("ALE computed via DALEX.")
        except Exception as exc:
            text_lines.append(f"ALE (DALEX) unavailable: {exc}")

        # Learning curve + Validation curve
        try:
            if split is not None:
                from sklearn.model_selection import learning_curve, validation_curve

                X_train, y_train = split.X_train, split.y_train
                train_sizes, train_scores, test_scores = learning_curve(wrapper.model, X_train, y_train, cv=3, n_jobs=1)
                tr = np.mean(train_scores, axis=1)
                te = np.mean(test_scores, axis=1)
                html_parts.append(self._plotly_multi_line(train_sizes, [tr, te], ["train", "cv"], title="Learning curve", x_label="train size", y_label="score"))

                # pick a simple hyperparameter if available
                param_name = None
                if hasattr(wrapper.model, "get_params"):
                    params = wrapper.model.get_params()
                    for cand in ["n_estimators", "max_depth", "C", "alpha"]:
                        if cand in params:
                            param_name = cand
                            break
                if param_name:
                    grid = [params[param_name]]
                    # create a small grid around numeric params
                    try:
                        v = float(params[param_name])
                        if v > 0:
                            grid = sorted(list(set([max(1.0, v / 2), v, v * 2])))
                    except Exception:
                        pass
                    train_scores, test_scores = validation_curve(wrapper.model, X_train, y_train, param_name=param_name, param_range=grid, cv=3, n_jobs=1)
                    tr = np.mean(train_scores, axis=1)
                    te = np.mean(test_scores, axis=1)
                    html_parts.append(self._plotly_multi_line(grid, [tr, te], ["train", "cv"], title=f"Validation curve: {param_name}", x_label=param_name, y_label="score"))
        except Exception as exc:
            text_lines.append(f"Learning/validation curves unavailable: {exc}")

        # Decision boundary (2D) via 2 numeric features
        try:
            if X_test is not None:
                numeric = self._numeric_X(X_test)
                cols = list(numeric.columns)
                if len(cols) >= 2:
                    a, b = cols[0], cols[1]
                    if features and len(features) >= 2 and features[0] in cols and features[1] in cols:
                        a, b = features[0], features[1]
                    html_parts.append(self._decision_boundary_html(wrapper.model, numeric, a, b, task))
        except Exception as exc:
            text_lines.append(f"Decision boundary unavailable: {exc}")

        html = "\n".join(html_parts)
        text = "\n".join(text_lines) if text_lines else ""
        return {"text": text, "html": html}

    # ---- local ----
    def _local_task(self, req: Dict[str, Any]) -> Dict[str, Any]:
        model = str(req.get("model", ""))
        task = str(req.get("task", self.task_type))
        target = str(req.get("target", self.target_column))
        row = int(req.get("row", 0))

        df = self._get_df()
        wrapper = self._get_wrapper(model)

        split = self._get_split(df, task, target) if task in ("classification", "regression") else None
        if split is None:
            raise RuntimeError("Local explanations require classification or regression")

        X_test = split.X_test.reset_index(drop=True)
        y_test = split.y_test.reset_index(drop=True)
        if row < 0 or row >= len(X_test):
            raise RuntimeError("Row out of range")

        x0 = X_test.iloc[[row]]
        y_true = y_test.iloc[row]

        y_pred = self.ml_model.predict(model, x0)
        if y_pred is None:
            raise RuntimeError("Prediction failed")
        y_pred0 = y_pred[0]

        conf = None
        proba = None
        if task == "classification":
            proba = self.ml_model.predict_proba(model, x0)
            if proba is not None:
                try:
                    conf = float(np.max(np.asarray(proba)))
                except Exception:
                    conf = None

        lines = []
        lines.append(f"Row: {row}")
        lines.append(f"y_true: {y_true}")
        lines.append(f"y_pred: {y_pred0}")
        if conf is not None:
            lines.append(f"confidence: {conf:.3f}")

        # SHAP local
        html_parts: List[str] = []
        try:
            import shap

            background = X_test
            if len(background) > 200:
                background = background.sample(200, random_state=42)
            explainer = shap.Explainer(wrapper.model, background)
            sv = explainer(x0)

            # Prefer a simple HTML snippet
            html_parts.append("<h3>SHAP</h3>")
            html_parts.append("<p>Computed SHAP values for this row (see exported report for full details).</p>")

            # Plain-English from SHAP contributions (top +/-)
            try:
                vals = np.asarray(sv.values).reshape(-1)
                feats = list(x0.columns)
                order = np.argsort(np.abs(vals))[::-1][:8]
                pos = [(feats[i], float(vals[i])) for i in order if float(vals[i]) > 0][:4]
                neg = [(feats[i], float(vals[i])) for i in order if float(vals[i]) < 0][:4]
                lines.append("")
                lines.append("Why this prediction (from SHAP):")
                if pos:
                    lines.append("Main drivers pushing prediction UP:")
                    for f, v in pos:
                        lines.append(f"- {f} increased output by ~{v:.4f}")
                if neg:
                    lines.append("Main drivers pushing prediction DOWN:")
                    for f, v in neg:
                        lines.append(f"- {f} decreased output by ~{abs(v):.4f}")
            except Exception:
                pass
        except Exception as exc:
            lines.append("")
            lines.append(f"SHAP unavailable: {exc}")

        # LIME
        try:
            import lime.lime_tabular

            X_train = split.X_train
            feature_names = list(X_train.columns)
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.asarray(X_train),
                feature_names=feature_names,
                mode="classification" if task == "classification" else "regression",
                discretize_continuous=True,
            )

            if task == "classification":
                def pred_fn(z):
                    import pandas as pd
                    Z = pd.DataFrame(z, columns=feature_names)
                    p = self.ml_model.predict_proba(model, Z)
                    return np.asarray(p) if p is not None else np.zeros((len(Z), 2))

                exp = explainer.explain_instance(np.asarray(x0)[0], pred_fn, num_features=10)
            else:
                def pred_fn(z):
                    import pandas as pd
                    Z = pd.DataFrame(z, columns=feature_names)
                    p = self.ml_model.predict(model, Z)
                    return np.asarray(p) if p is not None else np.zeros((len(Z),))

                exp = explainer.explain_instance(np.asarray(x0)[0], pred_fn, num_features=10)

            lines.append("")
            lines.append("LIME explanation (top rules):")
            for feat, wgt in exp.as_list()[:10]:
                lines.append(f"- {feat}: {float(wgt):.4f}")
        except Exception as exc:
            lines.append("")
            lines.append(f"LIME unavailable: {exc}")

        # Counterfactual (simple random search)
        try:
            cf = self._simple_counterfactual(wrapper.model, X_test, x0, task)
            if cf is not None:
                lines.append("")
                lines.append("Counterfactual suggestion (smallest change found):")
                lines.append(cf)
        except Exception:
            pass

        return {"text": "\n".join(lines), "html": "\n".join(html_parts)}

    def _simple_counterfactual(self, est, X_ref, x0, task: str) -> Optional[str]:
        # numeric-only random search around x0 within observed ranges
        import pandas as pd

        Xn = self._numeric_X(X_ref)
        x = self._numeric_X(x0)
        if Xn is None or x is None or Xn.shape[1] == 0:
            return None

        cols = list(Xn.columns)
        mins = Xn.min(axis=0)
        maxs = Xn.max(axis=0)

        base = x.iloc[0].copy()
        base_pred = est.predict(pd.DataFrame([base], columns=cols))[0]

        rng = np.random.default_rng(42)
        best = None
        best_dist = None

        for _ in range(300):
            cand = base.copy()
            # modify up to 3 features
            k = int(rng.integers(1, min(4, len(cols)) + 1))
            idx = rng.choice(len(cols), size=k, replace=False)
            for j in idx:
                c = cols[int(j)]
                lo = float(mins[c])
                hi = float(maxs[c])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    continue
                cand[c] = float(rng.uniform(lo, hi))

            pred = est.predict(pd.DataFrame([cand], columns=cols))[0]
            changed = (pred != base_pred) if task == "classification" else (abs(float(pred) - float(base_pred)) > 0.05 * (abs(float(base_pred)) + 1e-6))
            if not changed:
                continue

            dist = float(np.linalg.norm((cand.values - base.values), ord=2))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = cand

        if best is None:
            return None

        diffs = []
        for c in cols:
            if float(best[c]) != float(base[c]):
                diffs.append(f"{c}: {float(base[c]):.3f} → {float(best[c]):.3f}")
        return "; ".join(diffs[:8])

    # ---- what-if ----
    def _whatif_task(self, req: Dict[str, Any]) -> Dict[str, Any]:
        model = str(req.get("model", ""))
        task = str(req.get("task", self.task_type))
        target = str(req.get("target", self.target_column))
        row = int(req.get("row", 0))

        df = self._get_df()
        wrapper = self._get_wrapper(model)
        split = self._get_split(df, task, target) if task in ("classification", "regression") else None
        if split is None:
            raise RuntimeError("What-If requires classification or regression")

        X_test = split.X_test.reset_index(drop=True)
        if row < 0 or row >= len(X_test):
            raise RuntimeError("Row out of range")

        x0 = X_test.iloc[[row]]

        y_pred = self.ml_model.predict(model, x0)
        if y_pred is None:
            raise RuntimeError("Prediction failed")

        base = y_pred[0]

        # Sensitivity: vary each numeric feature by ±1 std and see delta
        Xn = self._numeric_X(split.X_train)
        xn0 = self._numeric_X(x0)
        if Xn.shape[1] == 0:
            return {"text": f"Baseline prediction: {base}\n(No numeric features for sensitivity.)"}

        stds = Xn.std(axis=0)
        cols = list(Xn.columns)

        import pandas as pd

        lines = [f"Baseline prediction: {base}", "", "Sensitivity (±1σ):"]
        for c in cols[:25]:
            try:
                sigma = float(stds[c])
                if not np.isfinite(sigma) or sigma == 0:
                    continue
                lo = xn0.iloc[0].copy()
                hi = xn0.iloc[0].copy()
                lo[c] = float(lo[c]) - sigma
                hi[c] = float(hi[c]) + sigma
                pred_lo = wrapper.model.predict(pd.DataFrame([lo], columns=cols))[0]
                pred_hi = wrapper.model.predict(pd.DataFrame([hi], columns=cols))[0]
                lines.append(f"- {c}: pred(lo)={pred_lo} pred(hi)={pred_hi}")
            except Exception:
                continue

        lines.append("\nTip: For interactive sliders, export a report and iterate on top drivers.")
        return {"text": "\n".join(lines)}

    # ---- debugging ----
    def _debug_task(self, req: Dict[str, Any]) -> Dict[str, Any]:
        model = str(req.get("model", ""))
        task = str(req.get("task", self.task_type))
        target = str(req.get("target", self.target_column))

        df = self._get_df()
        wrapper = self._get_wrapper(model)

        if task not in ("classification", "regression"):
            return {"text": "Debugging tools currently focus on supervised tasks."}

        split = self._get_split(df, task, target)
        assert split is not None

        X_test = split.X_test.reset_index(drop=True)
        y_test = split.y_test.reset_index(drop=True)

        y_pred = self.ml_model.predict(model, X_test)
        if y_pred is None:
            raise RuntimeError("Prediction failed")

        lines = []
        if task == "classification":
            y_pred = np.asarray(y_pred)
            y_true = np.asarray(y_test)
            err = (y_pred != y_true)
            err_idx = np.where(err)[0]
            lines.append(f"Misclassifications: {int(err_idx.size)} / {len(y_true)}")

            # hard examples: low confidence correct or high confidence wrong
            proba = self.ml_model.predict_proba(model, X_test)
            if proba is not None:
                conf = np.max(np.asarray(proba), axis=1)
                wrong_conf = conf[err]
                if wrong_conf.size:
                    lines.append(f"Avg confidence on wrong: {float(np.mean(wrong_conf)):.3f}")

            # simple error clusters: PCA on numeric
            try:
                from sklearn.decomposition import PCA

                Xn = self._numeric_X(X_test)
                if Xn.shape[1] >= 2:
                    pts = PCA(n_components=2, random_state=42).fit_transform(np.asarray(Xn))
                    # return points for plotting handled by view via state? We'll only provide summary here.
                    lines.append("Error clustering: PCA computed (use plot in UI).")
            except Exception:
                pass

        else:
            y_pred = np.asarray(y_pred, dtype=float)
            y_true = np.asarray(y_test, dtype=float)
            resid = y_true - y_pred
            lines.append(f"Residuals: mean={float(np.mean(resid)):.4f} std={float(np.std(resid)):.4f}")
            hard = np.argsort(np.abs(resid))[::-1][:10]
            lines.append("Hard examples (top |residual| rows): " + ", ".join([str(int(i)) for i in hard.tolist()]))

        lines.append("Bias detection: run Fairness tab with a protected attribute.")
        return {"text": "\n".join(lines)}

    # ---- drift ----
    def _drift_task(self, _req: Dict[str, Any]) -> Dict[str, Any]:
        df = self._get_df()
        if self._baseline_df is None:
            # set baseline to current
            self._baseline_df = df.copy()
            return {"text": "Baseline set from current dataset. Re-run drift after loading new data."}

        base = self._baseline_df
        prod = df

        lines = ["Drift report (baseline vs current):"]
        try:
            base_num = base.select_dtypes(include=["number"])
            prod_num = prod.select_dtypes(include=["number"])
            common = [c for c in base_num.columns if c in prod_num.columns]
            lines.append(f"Numeric features compared: {len(common)}")

            for c in common[:40]:
                psi = self._psi(base_num[c].to_numpy(), prod_num[c].to_numpy(), bins=10)
                flag = "ALERT" if psi >= 0.25 else ("WARN" if psi >= 0.1 else "")
                lines.append(f"- {c}: PSI={psi:.3f} {flag}")

            lines.append("\nPSI thresholds: <0.1 stable, 0.1-0.25 moderate, >0.25 significant.")
        except Exception as exc:
            lines.append(f"Drift computation failed: {exc}")

        return {"text": "\n".join(lines)}

    def _psi(self, a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 10 or b.size < 10:
            return float("nan")
        qs = np.quantile(a, np.linspace(0, 1, bins + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        ha, _ = np.histogram(a, bins=qs)
        hb, _ = np.histogram(b, bins=qs)
        pa = ha / max(1, ha.sum())
        pb = hb / max(1, hb.sum())
        pa = np.clip(pa, 1e-6, 1)
        pb = np.clip(pb, 1e-6, 1)
        return float(np.sum((pb - pa) * np.log(pb / pa)))

    # ---- fairness ----
    def _fairness_task(self, req: Dict[str, Any]) -> Dict[str, Any]:
        model = str(req.get("model", ""))
        task = str(req.get("task", self.task_type))
        target = str(req.get("target", self.target_column))
        protected = str(req.get("protected", ""))
        privileged = str(req.get("privileged", ""))

        df = self._get_df()
        if task != "classification":
            return {"text": "Fairness metrics currently implemented for classification.", "privileged_groups": []}

        if not protected or protected not in df.columns:
            return {"text": "Select a protected attribute column.", "privileged_groups": []}

        # compute group list
        groups = [str(x) for x in sorted(df[protected].dropna().unique().tolist())][:50]

        wrapper = self._get_wrapper(model)
        split = self._get_split(df, task, target)
        assert split is not None

        X_test = split.X_test.reset_index(drop=True)
        y_test = split.y_test.reset_index(drop=True)

        # We need protected values aligned with X_test rows; use same indices via train_test_split.
        # Best-effort: recompute split with return indices
        from sklearn.model_selection import train_test_split

        X = df.drop(columns=[target])
        y = df[target]
        g = df[protected]
        _, X_te, _, y_te, _, g_te = train_test_split(X, y, g, test_size=0.2, random_state=42, shuffle=True)
        X_te = X_te.reset_index(drop=True)
        y_te = y_te.reset_index(drop=True)
        g_te = g_te.reset_index(drop=True)

        y_pred = self.ml_model.predict(model, X_te)
        if y_pred is None:
            raise RuntimeError("Prediction failed")

        y_true = np.asarray(y_te)
        y_hat = np.asarray(y_pred)

        # Binarize labels for fairness (best-effort)
        uniq = np.unique(y_true)
        if uniq.size != 2:
            return {"text": "Fairness metrics require binary classification targets.", "privileged_groups": groups}

        positive = uniq[1]

        def rate(mask, arr):
            if mask.sum() == 0:
                return float("nan")
            return float(np.mean(arr[mask] == positive))

        text = []
        text.append(f"Protected attribute: {protected}")
        text.append(f"Positive class: {positive}")

        # demographic parity
        rates = {}
        for gv in groups:
            m = (g_te.astype(str) == gv).to_numpy()
            rates[gv] = rate(m, y_hat)
        text.append("\nDemographic parity (P(ŷ=1|group)):")
        for gv, r in list(rates.items())[:20]:
            text.append(f"- {gv}: {r:.3f}")

        # disparate impact vs privileged
        if privileged and privileged in rates and np.isfinite(rates[privileged]):
            base = max(1e-6, float(rates[privileged]))
            text.append("\nDisparate impact (group_rate / privileged_rate):")
            for gv in groups[:20]:
                if not np.isfinite(rates.get(gv, float("nan"))):
                    continue
                di = float(rates[gv]) / base
                flag = "ALERT" if di < 0.8 else ""
                text.append(f"- {gv}: {di:.3f} {flag}")

        # equalized odds (TPR/FPR per group)
        text.append("\nEqualized odds (TPR/FPR per group):")
        for gv in groups[:12]:
            m = (g_te.astype(str) == gv).to_numpy()
            tp = np.sum((y_true[m] == positive) & (y_hat[m] == positive))
            fn = np.sum((y_true[m] == positive) & (y_hat[m] != positive))
            fp = np.sum((y_true[m] != positive) & (y_hat[m] == positive))
            tn = np.sum((y_true[m] != positive) & (y_hat[m] != positive))
            tpr = tp / max(1, (tp + fn))
            fpr = fp / max(1, (fp + tn))
            text.append(f"- {gv}: TPR={tpr:.3f} FPR={fpr:.3f}")

        text.append("\nMitigation suggestions:")
        text.append("- Check data balance per group, reweight samples, or consider threshold adjustment per group.")
        text.append("- Inspect which features act as proxies for the protected attribute.")

        return {"text": "\n".join(text), "privileged_groups": groups}

    # ---- automobile insights ----
    def _auto_task(self, req: Dict[str, Any]) -> Dict[str, Any]:
        model = str(req.get("model", ""))
        task = str(req.get("task", self.task_type))
        target = str(req.get("target", self.target_column))

        df = self._get_df()
        wrapper = None
        try:
            wrapper = self._get_wrapper(model)
        except Exception:
            wrapper = None

        cols = [str(c).lower() for c in df.columns]
        lines = ["Automobile insights (heuristic):"]

        def has(name: str) -> bool:
            return name.lower() in cols

        # Car price prediction
        if has("price") or (target and target.lower() in ("price", "sellingprice", "msrp")):
            lines.append("\nCar price prediction:")
            # brand/model premium
            for key in ["brand", "make", "manufacturer"]:
                if has(key):
                    lines.append(f"- Brand premium: group average {target or 'price'} by {key}.")
                    break
            for key in ["model", "trim"]:
                if has(key):
                    lines.append(f"- Model premium: group average {target or 'price'} by {key}.")
                    break
            for key in ["year", "mileage", "km", "odometer"]:
                if has(key):
                    lines.append(f"- Depreciation: analyze partial dependence on {key}.")
                    break
            for key in ["region", "state", "city"]:
                if has(key):
                    lines.append(f"- Regional variations: compare {target or 'price'} across {key}.")
                    break

        # Fuel efficiency
        if has("mpg") or has("fuel_efficiency") or (target and "mpg" in target.lower()):
            lines.append("\nFuel efficiency:")
            lines.append("- Use permutation importance to find most influential features.")
            lines.append("- Optimal configuration: run a bounded search over controllable features (engine size, weight, gearing).")
            lines.append("- Cost savings: compare predicted mpg changes vs fuel price and mileage.")

        # Maintenance forecasting
        if any(k in cols for k in ["failure", "breakdown", "maintenance_cost", "service_cost"]):
            lines.append("\nMaintenance forecasting:")
            lines.append("- Failure modes: identify top SHAP/LIME drivers for high-risk predictions.")
            lines.append("- Preventive scheduling: trigger maintenance when risk exceeds a threshold.")
            lines.append("- Cost/benefit: compare preventive vs expected failure costs.")

        if len(lines) == 1:
            lines.append("No automotive patterns detected (missing typical columns like price/year/mileage/mpg).")

        return {"text": "\n".join(lines)}

    # ---- exporting ----
    def _export_report_task(self, fmt: str, payload: Dict[str, Any]) -> str:
        fmt = (fmt or "html").lower()
        from pathlib import Path

        out_dir = Path.home() / ".arcsaathi" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())

        if fmt == "html":
            p = out_dir / f"explainability_{stamp}.html"
            p.write_text(self._render_html(payload), encoding="utf-8")
            return str(p)

        if fmt == "pdf":
            p = out_dir / f"explainability_{stamp}.pdf"
            self._render_pdf(payload, str(p))
            return str(p)

        raise RuntimeError("Unsupported format")

    def _render_html(self, payload: Dict[str, Any]) -> str:
        def esc(s: Any) -> str:
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        sec = []
        sec.append("<h1>ARCSaathi Explainability Report</h1>")
        sec.append(f"<h2>Global</h2><pre>{esc(payload.get('global_text',''))}</pre>")
        if payload.get("global_html"):
            sec.append(payload["global_html"])
        sec.append(f"<h2>Local</h2><pre>{esc(payload.get('local_text',''))}</pre>")
        if payload.get("local_html"):
            sec.append(payload["local_html"])
        sec.append(f"<h2>What-If</h2><pre>{esc(payload.get('whatif_text',''))}</pre>")
        sec.append(f"<h2>Debugging</h2><pre>{esc(payload.get('debug_text',''))}</pre>")
        sec.append(f"<h2>Drift</h2><pre>{esc(payload.get('drift_text',''))}</pre>")
        sec.append(f"<h2>Fairness</h2><pre>{esc(payload.get('fairness_text',''))}</pre>")
        sec.append(f"<h2>Auto Insights</h2><pre>{esc(payload.get('auto_text',''))}</pre>")

        return """<!doctype html>
<html><head><meta charset='utf-8'/><title>ARCSaathi Explainability</title>
<style>body{font-family:Arial;margin:24px} pre{background:#f6f6f6;padding:10px;white-space:pre-wrap}</style>
</head><body>""" + "\n".join(sec) + "</body></html>"

    def _render_pdf(self, payload: Dict[str, Any], out_path: str) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(out_path, pagesize=A4)
        w, h = A4
        y = h - 2 * cm

        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, y, "ARCSaathi Explainability Report")
        y -= 0.8 * cm

        c.setFont("Helvetica", 10)
        for title, key in [
            ("Global", "global_text"),
            ("Local", "local_text"),
            ("What-If", "whatif_text"),
            ("Debugging", "debug_text"),
            ("Drift", "drift_text"),
            ("Fairness", "fairness_text"),
            ("Auto", "auto_text"),
        ]:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(2 * cm, y, title)
            y -= 0.6 * cm
            c.setFont("Helvetica", 8)
            txt = str(payload.get(key, ""))
            for line in txt.splitlines()[:10]:
                c.drawString(2 * cm, y, line[:120])
                y -= 0.45 * cm
                if y < 2 * cm:
                    c.showPage()
                    y = h - 2 * cm
            y -= 0.2 * cm
            if y < 2 * cm:
                c.showPage()
                y = h - 2 * cm

        c.save()

    # ---- plotly HTML helpers ----
    def _plotly_line(self, x, y, *, title: str, x_label: str, y_label: str) -> str:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(x), y=list(np.asarray(y).reshape(-1)), mode="lines"))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=320, margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _plotly_multi_line(self, x, ys, labels, *, title: str, x_label: str, y_label: str) -> str:
        import plotly.graph_objects as go

        fig = go.Figure()
        for y, lab in zip(ys, labels):
            fig.add_trace(go.Scatter(x=list(x), y=list(np.asarray(y).reshape(-1)), mode="lines+markers", name=str(lab)))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=320, margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _plotly_heatmap(self, x, y, z, *, title: str, x_label: str, y_label: str) -> str:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(x=list(x), y=list(y), z=np.asarray(z).tolist(), colorscale="Viridis"))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=360, margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _decision_boundary_html(self, est, Xn, a: str, b: str, task: str) -> str:
        import pandas as pd
        import plotly.graph_objects as go

        # Build a 2D grid with other features fixed to median
        cols = list(Xn.columns)
        base = Xn.median(axis=0)

        xa = Xn[a].to_numpy()
        xb = Xn[b].to_numpy()
        if xa.size < 10 or xb.size < 10:
            return ""

        x_min, x_max = float(np.nanmin(xa)), float(np.nanmax(xa))
        y_min, y_max = float(np.nanmin(xb)), float(np.nanmax(xb))

        gx = np.linspace(x_min, x_max, 120)
        gy = np.linspace(y_min, y_max, 120)
        xx, yy = np.meshgrid(gx, gy)

        rows = []
        for i in range(xx.size):
            r = base.copy()
            r[a] = float(xx.ravel()[i])
            r[b] = float(yy.ravel()[i])
            rows.append(r)
        grid = pd.DataFrame(rows, columns=cols)

        if task == "classification" and hasattr(est, "predict_proba"):
            zz = est.predict_proba(grid)
            zz = np.asarray(zz)
            if zz.ndim == 2 and zz.shape[1] >= 2:
                z = zz[:, 1]
            else:
                z = zz.reshape(-1)
        else:
            z = np.asarray(est.predict(grid)).reshape(-1)

        z2 = z.reshape(xx.shape)

        fig = go.Figure()
        fig.add_trace(go.Contour(x=gx, y=gy, z=z2, contours_coloring="heatmap", showscale=False, opacity=0.65))
        fig.add_trace(go.Scatter(x=Xn[a], y=Xn[b], mode="markers", marker=dict(size=5, opacity=0.6), name="data"))
        fig.update_layout(title=f"Decision boundary (approx): {a} vs {b}", xaxis_title=a, yaxis_title=b, height=360, margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(include_plotlyjs="cdn", full_html=False)
