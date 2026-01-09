"""Data Processing Module controller.

Covers Tab 1 (Data Loading & Profiling) and Tab 2 (Preprocessing Pipeline).
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

import pandas as pd

from PySide6.QtCore import Signal

from .base_controller import BaseController
from ..models import DataModel, PreprocessingModel
from ..state import AppState
from ..utils import get_logger


class DataProcessingController(BaseController):
    dataset_loaded = Signal()
    pipeline_updated = Signal()
    error_occurred = Signal(str)

    def __init__(self, state: AppState, data_model: DataModel, preprocessing_model: PreprocessingModel):
        super().__init__()
        self._log = get_logger("controllers.data_processing")

        self.state = state
        self.data_model = data_model
        self.preprocessing_model = preprocessing_model

        self.data_model.error_occurred.connect(self.error_occurred)
        self.preprocessing_model.error_occurred.connect(self.error_occurred)

    def load_dataset(self, file_path: str) -> bool:
        ok = self.data_model.load_data(file_path)
        if not ok:
            return False

        data = self.data_model.get_data()
        if data is None:
            self.error_occurred.emit("Loaded dataset is empty")
            return False

        self.preprocessing_model.set_data(data)
        meta = self.data_model.get_metadata()
        self.state.set_dataset(file_path, meta)
        self.state.save()
        self.dataset_loaded.emit()
        self._log.info("Dataset loaded: %s", file_path)
        return True

    def load_dataframe(self, df: pd.DataFrame, file_path: str) -> bool:
        """Accept an already-loaded DataFrame (from background thread) and update models/state."""
        try:
            self.data_model.set_data(df, file_path=file_path)
            self.preprocessing_model.set_data(df)
            meta = self.data_model.get_metadata()
            self.state.set_dataset(file_path, meta)
            self.state.save()
            self.dataset_loaded.emit()
            self._log.info("Dataset injected: %s", file_path)
            return True
        except Exception as exc:
            self.error_occurred.emit(f"Failed to set dataset: {exc}")
            return False

    def add_preprocessing_step(self, operation: str, params: Dict[str, Any]) -> bool:
        name = operation.replace("_", " ").title()
        ok = self.preprocessing_model.add_step(name=name, operation=operation, params=params)
        if ok:
            self.state.set_preprocessing_steps(self.preprocessing_model.get_steps())
            self.state.save()
            self.pipeline_updated.emit()
        return ok

    def remove_preprocessing_step(self, index: int) -> bool:
        ok = self.preprocessing_model.remove_step(index)
        if ok:
            self.state.set_preprocessing_steps(self.preprocessing_model.get_steps())
            self.state.save()
            self.pipeline_updated.emit()
        return ok

    def apply_pipeline(self) -> Optional[pd.DataFrame]:
        processed = self.preprocessing_model.apply_pipeline()
        if processed is None:
            return None

        self.data_model.update_data(processed)
        self.state.set_preprocessing_steps(self.preprocessing_model.get_steps())
        self.state.save()
        self.pipeline_updated.emit()
        self._log.info("Preprocessing pipeline applied (%d steps)", len(self.preprocessing_model.steps))
        return processed

    def replace_preprocessing_steps(self, steps: List[Dict[str, Any]]) -> None:
        self.preprocessing_model.set_steps(steps)
        self.state.set_preprocessing_steps(self.preprocessing_model.get_steps())
        self.state.save()
        self.pipeline_updated.emit()

    def clear_preprocessing_steps(self) -> None:
        self.preprocessing_model.clear_pipeline()
        self.state.set_preprocessing_steps([])
        self.state.save()
        self.pipeline_updated.emit()

    def validate_steps(self, steps: List[Dict[str, Any]]) -> tuple[bool, str]:
        df = self.data_model.get_data()
        if df is None or df.empty:
            return (False, "No dataset loaded")

        for i, s in enumerate(steps or []):
            op = str(s.get("operation") or "")
            params = dict(s.get("params") or {})

            if op in {"feature_selection"}:
                method = str(params.get("method", ""))
                if method in {"select_k_best", "rfe"}:
                    target = params.get("target")
                    if not target or str(target) not in df.columns:
                        return (False, f"Step {i + 1} requires a valid target column")

        return (True, "")

    def preview_steps(self, steps: List[Dict[str, Any]], upto_index: int = -1, n: int = 12) -> tuple[list[str], list[list[str]], str]:
        df = self.data_model.get_data()
        if df is None or df.empty:
            return ([], [], "No dataset")
        try:
            out = self.preprocessing_model.apply_steps_preview(df, steps, upto_index=upto_index)
            sample = out.head(n)
            headers = [str(c) for c in sample.columns]
            rows: list[list[str]] = []
            for _, r in sample.iterrows():
                rows.append([str(v) for v in r.values.tolist()])
            summary = f"Rows: {len(out)} | Cols: {out.shape[1]}"
            return (headers, rows, summary)
        except Exception as exc:
            self.error_occurred.emit(f"Preview failed: {exc}")
            return ([], [], "Preview failed")

    def get_missing_pct(self) -> Dict[str, float]:
        df = self.data_model.get_data()
        if df is None or df.empty:
            return {}
        pct = (df.isna().mean() * 100.0).to_dict()
        return {str(k): float(v) for k, v in pct.items()}

    def smart_preprocess(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a simple, explainable default pipeline based on data + task."""
        df = self.data_model.get_data()
        if df is None or df.empty:
            return []

        task_type = str(task.get("task_type") or "classification")
        target = str(task.get("target") or "").strip() or None

        steps: List[Dict[str, Any]] = []

        # Missing values
        if df.isna().any().any():
            steps.append({
                "name": "Impute missing",
                "operation": "impute",
                "params": {"columns": [], "strategy": "mean"},
                "tooltip": "Fills missing numeric values with the mean (auto-applies to columns with missing values).",
            })

        # Encoding
        cat_cols = [c for c in df.columns if (df[c].dtype == object or 'category' in str(df[c].dtype)) and c != target]
        if cat_cols:
            steps.append({
                "name": "Encode categoricals",
                "operation": "encode",
                "params": {"columns": [], "method": "onehot", "max_categories": 50, "handle_unknown": "ignore", "target": target},
                "tooltip": "One-hot encodes categorical features. High-cardinality columns are capped by max_categories.",
            })

        # Scaling
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
        if num_cols:
            steps.append({
                "name": "Scale numeric",
                "operation": "scale",
                "params": {"columns": [], "method": "standard", "scope": "exclude_binary", "target": target},
                "tooltip": "Standardizes numeric features (z-score). Excludes binary columns by default.",
            })

        # Outliers (gentle)
        if len(num_cols) >= 1:
            steps.append({
                "name": "Handle outliers",
                "operation": "outliers",
                "params": {"method": "iqr", "multiplier": 1.5, "target": target},
                "tooltip": "Clips extreme numeric values using the IQR rule (keeps row count stable).",
            })

        # Feature selection for wide data
        if len(num_cols) >= 20:
            steps.append({
                "name": "Correlation filter",
                "operation": "feature_selection",
                "params": {"method": "correlation", "threshold": 0.95},
                "tooltip": "Drops highly correlated numeric features to reduce redundancy.",
            })

        # Task-specific: supervised selection
        if target and target in df.columns and task_type in {"classification", "regression"} and len(num_cols) >= 30:
            steps.append({
                "name": "SelectKBest",
                "operation": "feature_selection",
                "params": {
                    "method": "select_k_best",
                    "k": 30,
                    "score_func": "f_regression" if task_type == "regression" else "f_classif",
                    "target": target,
                    "task_type": task_type,
                },
                "tooltip": "Keeps the top-k most informative numeric features relative to the target.",
            })

        return steps

    def get_dataset_preview(self, n: int = 8) -> tuple[list[str], list[list[str]]]:
        df = self.data_model.get_sample(n)
        if df is None:
            return ([], [])
        headers = [str(c) for c in df.columns]
        values: List[List[str]] = []
        for _, row in df.iterrows():
            values.append([str(v) for v in row.values.tolist()])
        return headers, values

    def get_dataset_summary(self) -> Dict[str, Any]:
        info = self.data_model.get_data_info()
        if not info:
            return {"rows": 0, "columns": 0, "memory_mb": 0.0}

        shape = info.get("shape", (0, 0))
        return {
            "rows": int(shape[0]),
            "columns": int(shape[1]),
            "memory_mb": float(info.get("memory_usage_mb", 0.0)),
        }
