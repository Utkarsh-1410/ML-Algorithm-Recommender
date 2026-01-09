"""Task detection for ARCSaathi.

Infers the ML task type (regression/classification/clustering/time series),
suggests a target (when applicable), and proposes evaluation metrics.

Designed to be fast and deterministic; safe to run on the UI thread for
moderate datasets, but can also be used from a background thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TaskDetectionResult:
    task_type: str  # regression | classification | clustering | time_series
    target: Optional[str]
    classification_type: Optional[str] = None  # binary | multiclass
    metrics: List[str] | None = None
    reasoning: List[str] | None = None
    details: Dict[str, Any] | None = None


class TaskDetectionModel:
    TASK_REGRESSION = "regression"
    TASK_CLASSIFICATION = "classification"
    TASK_CLUSTERING = "clustering"
    TASK_TIME_SERIES = "time_series"

    def detect(self, df: pd.DataFrame, *, target: Optional[str] = None) -> TaskDetectionResult:
        df = df if df is not None else pd.DataFrame()
        if df.empty:
            return TaskDetectionResult(
                task_type=self.TASK_CLUSTERING,
                target=None,
                metrics=[],
                reasoning=["Dataset is empty; defaulting to Clustering."],
                details={},
            )

        suggested_target = target if target and target in df.columns else self._suggest_target(df)

        # If we can't confidently select a target, default to clustering.
        if suggested_target is None or suggested_target not in df.columns:
            return TaskDetectionResult(
                task_type=self.TASK_CLUSTERING,
                target=None,
                metrics=[],
                reasoning=["No target selected; treating this as an unsupervised Clustering task."],
                details={},
            )

        y = df[suggested_target]
        rows = len(df)
        nunique = int(y.nunique(dropna=True))

        is_time_series = self._looks_like_time_series(df, y)
        if is_time_series:
            reasoning = [
                f"Detected as Time Series because dataset contains a datetime-like column and target '{suggested_target}' is numeric.",
            ]
            return TaskDetectionResult(
                task_type=self.TASK_TIME_SERIES,
                target=str(suggested_target),
                metrics=["RMSE", "MAE", "R²"],
                reasoning=reasoning,
                details={"nunique_target": nunique},
            )

        if pd.api.types.is_numeric_dtype(y):
            # If low unique count relative to dataset size, treat as classification.
            # Typical: labels 0/1, 1..K, etc.
            unique_ratio = float(nunique / max(1, rows))
            if nunique <= 20 or unique_ratio <= 0.05:
                cls_type = "binary" if nunique == 2 else "multiclass"
                reasoning = [
                    f"Detected as {'Binary' if cls_type == 'binary' else 'Multi-class'} Classification because target '{suggested_target}' has only {nunique} unique values.",
                ]
                return TaskDetectionResult(
                    task_type=self.TASK_CLASSIFICATION,
                    target=str(suggested_target),
                    classification_type=cls_type,
                    metrics=["Accuracy", "F1", "AUC"],
                    reasoning=reasoning,
                    details={"nunique_target": nunique},
                )

            min_v, max_v = self._numeric_range(y)
            reasoning = [
                f"Detected as Regression because target variable '{suggested_target}' is continuous numeric with range {min_v}–{max_v}.",
            ]
            return TaskDetectionResult(
                task_type=self.TASK_REGRESSION,
                target=str(suggested_target),
                metrics=["RMSE", "MAE", "R²"],
                reasoning=reasoning,
                details={"nunique_target": nunique, "min": min_v, "max": max_v},
            )

        # Non-numeric => classification.
        cls_type = "binary" if nunique == 2 else "multiclass"
        sample_values = self._sample_unique_values(y)
        reasoning = [
            f"Detected as {'Binary' if cls_type == 'binary' else 'Multi-class'} Classification because target '{suggested_target}' is non-numeric with {nunique} unique values (e.g., {sample_values}).",
        ]
        return TaskDetectionResult(
            task_type=self.TASK_CLASSIFICATION,
            target=str(suggested_target),
            classification_type=cls_type,
            metrics=["Accuracy", "F1", "AUC"],
            reasoning=reasoning,
            details={"nunique_target": nunique},
        )

    def _suggest_target(self, df: pd.DataFrame) -> Optional[str]:
        # Prefer last column if it is not constant and not an obvious ID.
        for col in reversed(df.columns.tolist()):
            s = df[col]
            if s.nunique(dropna=False) <= 1:
                continue
            if self._looks_like_id(s, rows=len(df)):
                continue
            return str(col)
        return None

    def _looks_like_id(self, s: pd.Series, *, rows: int) -> bool:
        # High uniqueness + integer-ish => probably an ID.
        try:
            if rows > 0 and int(s.nunique(dropna=True)) == rows:
                if pd.api.types.is_integer_dtype(s):
                    return True
                if pd.api.types.is_string_dtype(s):
                    # Many string IDs look like short tokens.
                    return True
        except Exception:
            pass
        return False

    def _looks_like_time_series(self, df: pd.DataFrame, y: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(y):
            return False

        # Heuristic: presence of a datetime dtype column, or a name containing date/time that parses well.
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                return True
            name = str(col).lower()
            if "date" in name or "time" in name:
                parsed = pd.to_datetime(s, errors="coerce")
                if float(parsed.notna().mean()) >= 0.8:
                    return True
        return False

    def _numeric_range(self, s: pd.Series) -> tuple[str, str]:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return ("-", "-")
        mn = float(np.nanmin(vals.to_numpy()))
        mx = float(np.nanmax(vals.to_numpy()))
        # Keep it human-friendly.
        if abs(mn) >= 1000 or abs(mx) >= 1000:
            return (f"{mn:,.0f}", f"{mx:,.0f}")
        return (f"{mn:.4g}", f"{mx:.4g}")

    def _sample_unique_values(self, s: pd.Series, k: int = 3) -> str:
        try:
            vals = [str(v) for v in s.dropna().unique().tolist()[:k]]
            if not vals:
                return "-"
            return ", ".join(vals)
        except Exception:
            return "-"
