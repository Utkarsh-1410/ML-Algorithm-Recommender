"""Intelligent Model Recommender controller.

Covers Tab 5 (Model Recommender).

Uses:
- pandas-based dataset meta-feature analysis
- rule-based scoring engine
- SQLite persistence for learning user preferences
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Signal

from .base_controller import BaseController
from ..models import DataModel, MLModel, ModelStore
from ..models.model_recommendation_engine import (
    HardConstraints,
    ModelRecommendationEngine,
    ProblemRequirements,
    ScoreWeights,
)
from ..state import AppState
from ..utils import get_logger


class ModelRecommenderController(BaseController):
    recommendation_ready = Signal(object)  # structured payload
    error_occurred = Signal(str)

    def __init__(self, state: AppState, data_model: DataModel, ml_model: MLModel):
        super().__init__()
        self._log = get_logger("controllers.recommender")

        self.state = state
        self.data_model = data_model
        self.ml_model = ml_model
        self.store = ModelStore()
        self.engine = ModelRecommendationEngine()

        self._user_id = "default"

    def recommend(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Compute recommendations and emit a structured payload for the UI."""
        df = self.data_model.get_data()
        if df is None or df.empty:
            self.error_occurred.emit("Load a dataset first (Tab 1)")
            return None

        try:
            profile = str((request or {}).get("profile") or "Best Fit")
            task_type = str((request or {}).get("task_type") or "classification").lower()
            target = (request or {}).get("target")

            req = ProblemRequirements(
                task_type=task_type,
                target=str(target) if target else None,
                interpretability=str((request or {}).get("interpretability") or "Medium"),
                training_time=str((request or {}).get("training_time") or "Moderate"),
                prediction_speed=str((request or {}).get("prediction_speed") or "Moderate"),
                deployment_size=str((request or {}).get("deployment_size") or "Medium"),
                required_accuracy=(request or {}).get("required_accuracy"),
            )

            w = (request or {}).get("weights") or {}
            weights = ScoreWeights(
                accuracy=int(w.get("accuracy", 40)),
                speed=int(w.get("speed", 25)),
                interpretability=int(w.get("interpretability", 20)),
                robustness=int(w.get("robustness", 15)),
            )

            c = (request or {}).get("constraints") or {}
            constraints = HardConstraints(
                must_be_interpretable=bool(c.get("must_be_interpretable", False)),
                must_be_fast=bool(c.get("must_be_fast", False)),
                must_handle_categorical=bool(c.get("must_handle_categorical", False)),
                max_inference_ms=c.get("max_inference_ms"),
                max_model_size_mb=c.get("max_model_size_mb"),
            )

            automobile_mode = bool((request or {}).get("automobile_mode", False))
            automobile_preset = (request or {}).get("automobile_preset")

            meta = self.engine.extract_meta_features(df, task_type=task_type, target=req.target)
            registry = self.ml_model.list_registry()
            bias = self.store.get_recommender_bias(user_id=self._user_id, task_type=task_type)

            result = self.engine.recommend(
                registry=registry,
                meta=meta,
                requirements=req,
                profile=profile,
                weights=weights,
                constraints=constraints,
                feedback_bias=bias,
                automobile_mode=automobile_mode,
                automobile_preset=str(automobile_preset) if automobile_preset else None,
            )

            payload = {
                "profile": result.profile,
                "meta": {
                    "n_samples": result.meta.n_samples,
                    "n_features": result.meta.n_features,
                    "n_numeric": result.meta.n_numeric,
                    "n_categorical": result.meta.n_categorical,
                    "missing_pct": result.meta.missing_pct,
                    "class_imbalance": result.meta.class_imbalance,
                    "linearity_score": result.meta.linearity_score,
                    "noise_level": result.meta.noise_level,
                    "corr_mean_abs": result.meta.corr_mean_abs,
                    "corr_high_pair_ratio": result.meta.corr_high_pair_ratio,
                    "dataset_sig": result.meta.signature(),
                },
                "requirements": {
                    "task_type": req.task_type,
                    "target": req.target,
                    "interpretability": req.interpretability,
                    "training_time": req.training_time,
                    "prediction_speed": req.prediction_speed,
                    "deployment_size": req.deployment_size,
                    "required_accuracy": req.required_accuracy,
                },
                "weights": {
                    "accuracy": result.weights.accuracy,
                    "speed": result.weights.speed,
                    "interpretability": result.weights.interpretability,
                    "robustness": result.weights.robustness,
                },
                "constraints": {
                    "must_be_interpretable": constraints.must_be_interpretable,
                    "must_be_fast": constraints.must_be_fast,
                    "must_handle_categorical": constraints.must_handle_categorical,
                    "max_inference_ms": constraints.max_inference_ms,
                    "max_model_size_mb": constraints.max_model_size_mb,
                },
                "top": [
                    {
                        "algorithm_key": r.algorithm_key,
                        "model_name": r.model_name,
                        "family": r.family,
                        "score_total": r.score_total,
                        "scores": r.scores,
                        "why": r.why,
                        "strengths": r.strengths,
                        "weaknesses": r.weaknesses,
                        "suggested_params": r.suggested_params,
                    }
                    for r in result.top
                ],
                "advanced_suggestions": list(result.advanced_suggestions),
                "feature_engineering_suggestions": list(result.feature_engineering_suggestions),
                "generated_at": float(time.time()),
            }

            # persist latest UI prefs (weights + constraints)
            self.store.save_recommender_prefs(
                user_id=self._user_id,
                prefs={
                    "profile": profile,
                    "weights": payload["weights"],
                    "constraints": payload["constraints"],
                    "requirements": payload["requirements"],
                    "automobile": {"mode": automobile_mode, "preset": automobile_preset},
                },
            )

            self.recommendation_ready.emit(payload)
            self._log.info("Recommendations ready: profile=%s task=%s", profile, task_type)
            return payload
        except Exception as exc:
            self.error_occurred.emit(f"Recommendation failed: {exc}")
            return None

    def record_feedback(self, payload: Dict[str, Any]) -> None:
        try:
            accepted = bool((payload or {}).get("accepted", False))
            rec = (payload or {}).get("recommendation") or {}
            ctx = (payload or {}).get("context") or {}

            meta = (ctx or {}).get("meta") or {}
            dataset_sig = str(meta.get("dataset_sig") or "")
            req = (ctx or {}).get("requirements") or {}

            self.store.save_recommender_feedback(
                user_id=self._user_id,
                dataset_sig=dataset_sig,
                task_type=str(req.get("task_type") or ""),
                profile=str((ctx or {}).get("profile") or ""),
                algorithm_key=str(rec.get("algorithm_key") or ""),
                model_name=str(rec.get("model_name") or ""),
                score_total=int(rec.get("score_total") or 0),
                accepted=accepted,
                payload={"rec": rec, "context": ctx},
            )
        except Exception:
            pass

    def export_report(self, fmt: str, payload: Dict[str, Any], out_path: Optional[str] = None) -> Optional[str]:
        fmt = (fmt or "html").lower().strip()
        try:
            root = Path.home() / ".arcsaathi" / "reports"
            root.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time())
            if not out_path:
                out_path = str(root / f"recommendations_{stamp}.{fmt}")

            if fmt == "html":
                html = self._render_html(payload)
                Path(out_path).write_text(html, encoding="utf-8")
                return out_path

            if fmt == "pdf":
                self._render_pdf(payload, out_path)
                return out_path

            self.error_occurred.emit(f"Unsupported export format: {fmt}")
            return None
        except Exception as exc:
            self.error_occurred.emit(f"Export failed: {exc}")
            return None

    def _render_html(self, payload: Dict[str, Any]) -> str:
        top = (payload or {}).get("top", []) or []
        meta = (payload or {}).get("meta", {}) or {}
        req = (payload or {}).get("requirements", {}) or {}
        adv = (payload or {}).get("advanced_suggestions", []) or []
        fe = (payload or {}).get("feature_engineering_suggestions", []) or []

        rows = "".join(
            [
                "<tr>"
                f"<td>{r.get('model_name','')}</td>"
                f"<td>{r.get('algorithm_key','')}</td>"
                f"<td>{r.get('score_total',0)}</td>"
                f"<td>{(r.get('scores',{}) or {}).get('accuracy',0)}</td>"
                f"<td>{(r.get('scores',{}) or {}).get('speed',0)}</td>"
                f"<td>{(r.get('scores',{}) or {}).get('interpretability',0)}</td>"
                f"<td>{(r.get('scores',{}) or {}).get('robustness',0)}</td>"
                f"<td>{r.get('why','')}</td>"
                "</tr>"
                for r in top
            ]
        )

        def li(items):
            return "".join([f"<li>{str(x)}</li>" for x in (items or [])])

        return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>ARCSaathi Recommendations</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    .muted {{ color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Model Recommendations</h1>
  <p class='muted'>Generated by ARCSaathi on {time.ctime(payload.get('generated_at', time.time()))}</p>

  <h2>Dataset characteristics</h2>
  <ul>
    <li>n_samples: {meta.get('n_samples')}</li>
    <li>n_features: {meta.get('n_features')}</li>
    <li>numeric/categorical: {meta.get('n_numeric')}/{meta.get('n_categorical')}</li>
    <li>missing_pct: {meta.get('missing_pct')}</li>
    <li>class_imbalance: {meta.get('class_imbalance')}</li>
    <li>linearity_score: {meta.get('linearity_score')}</li>
    <li>noise_level: {meta.get('noise_level')}</li>
  </ul>

  <h2>Requirements</h2>
  <ul>
    <li>task_type: {req.get('task_type')}</li>
    <li>target: {req.get('target')}</li>
    <li>interpretability: {req.get('interpretability')}</li>
    <li>training_time: {req.get('training_time')}</li>
    <li>prediction_speed: {req.get('prediction_speed')}</li>
    <li>deployment_size: {req.get('deployment_size')}</li>
  </ul>

  <h2>Top 5 recommendations</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th><th>Key</th><th>Total</th><th>Acc</th><th>Speed</th><th>Interp</th><th>Robust</th><th>Why</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <h2>Advanced options</h2>
  <ul>{li(adv)}</ul>

  <h2>Feature engineering suggestions</h2>
  <ul>{li(fe)}</ul>
</body>
</html>"""

    def _render_pdf(self, payload: Dict[str, Any], out_path: str) -> None:
        # Minimal PDF export using matplotlib (no extra dependencies).
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        top = (payload or {}).get("top", []) or []
        meta = (payload or {}).get("meta", {}) or {}
        req = (payload or {}).get("requirements", {}) or {}

        with PdfPages(out_path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.text(0.08, 0.95, "ARCSaathi Model Recommendations", fontsize=16, weight="bold")
            fig.text(0.08, 0.92, f"Generated: {time.ctime(payload.get('generated_at', time.time()))}")

            y = 0.88
            fig.text(0.08, y, "Dataset", weight="bold")
            y -= 0.02
            for k in ["n_samples", "n_features", "n_numeric", "n_categorical", "missing_pct", "class_imbalance", "linearity_score", "noise_level"]:
                fig.text(0.10, y, f"{k}: {meta.get(k)}")
                y -= 0.018

            y -= 0.01
            fig.text(0.08, y, "Requirements", weight="bold")
            y -= 0.02
            for k in ["task_type", "target", "interpretability", "training_time", "prediction_speed", "deployment_size"]:
                fig.text(0.10, y, f"{k}: {req.get(k)}")
                y -= 0.018

            y -= 0.01
            fig.text(0.08, y, "Top recommendations", weight="bold")
            y -= 0.025
            for i, r in enumerate(top[:5], start=1):
                scores = r.get("scores", {}) or {}
                line = (
                    f"{i}. {r.get('model_name','')} ({r.get('score_total',0)}/100) "
                    f"[Acc {scores.get('accuracy',0)}, Speed {scores.get('speed',0)}, "
                    f"Interp {scores.get('interpretability',0)}, Robust {scores.get('robustness',0)}]"
                )
                fig.text(0.10, y, line)
                y -= 0.02
                why = str(r.get("why", ""))
                if why:
                    fig.text(0.12, y, f"Why: {why}")
                    y -= 0.02

            pdf.savefig(fig)
            plt.close(fig)
