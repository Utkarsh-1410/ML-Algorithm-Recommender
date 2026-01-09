"""Data profiling + schema validation + auto-inference.

Designed to be called from background threads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfileCacheKey:
    source_id: str


@dataclass
class ProfileResult:
    overview: Dict[str, Any]
    columns: List[Dict[str, Any]]
    warnings: List[str]
    schema: Dict[str, Any]
    inference: Dict[str, Any]


class ProfilingModel:
    """Computes profiling results and caches them by source_id."""

    def __init__(self):
        self._cache: Dict[str, ProfileResult] = {}

    def get_cached(self, key: ProfileCacheKey) -> Optional[ProfileResult]:
        return self._cache.get(key.source_id)

    def set_cached(self, key: ProfileCacheKey, value: ProfileResult) -> None:
        self._cache[key.source_id] = value

    def profile(self, df: pd.DataFrame, *, target: Optional[str] = None, cache_key: Optional[ProfileCacheKey] = None) -> ProfileResult:
        if cache_key:
            cached = self.get_cached(cache_key)
            if cached is not None:
                return cached

        overview = self._overview(df)
        columns = self._column_stats(df)
        schema = self._schema_validation(df)
        warnings = self._quality_warnings(df, columns)
        inference = self._auto_inference(df, columns, target=target)

        result = ProfileResult(
            overview=overview,
            columns=columns,
            warnings=warnings,
            schema=schema,
            inference=inference,
        )

        if cache_key:
            self.set_cached(cache_key, result)

        return result

    def suggest_target(self, df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None
        # Prefer last column unless it's an obvious ID/constant
        last = df.columns[-1]
        if self._looks_like_id(df[last]) or df[last].nunique(dropna=False) <= 1:
            # fallback: find best candidate
            for c in reversed(df.columns.tolist()):
                if df[c].nunique(dropna=False) <= 1:
                    continue
                if self._looks_like_id(df[c]):
                    continue
                return str(c)
        return str(last)

    def infer_problem_type(self, df: pd.DataFrame, target: Optional[str]) -> str:
        if not target or target not in df.columns:
            return "clustering"
        y = df[target]
        if pd.api.types.is_numeric_dtype(y):
            # If few unique values, treat as classification.
            nunique = int(y.nunique(dropna=True))
            if nunique <= 20:
                return "classification"
            return "regression"
        return "classification"

    # ---- internals ----
    def _overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        rows, cols = df.shape
        missing = int(df.isna().sum().sum())
        total = int(rows * cols) if rows and cols else 0
        missing_pct = float((missing / total) * 100) if total else 0.0
        duplicates = int(df.duplicated().sum()) if rows else 0
        mem_bytes = int(df.memory_usage(deep=True).sum())
        mem_mb = float(mem_bytes / (1024 ** 2))

        # Quality score heuristic
        score = 100.0
        score -= min(60.0, missing_pct * 1.5)
        if rows:
            score -= min(25.0, (duplicates / rows) * 100)
        score = max(0.0, min(100.0, score))

        return {
            "rows": rows,
            "columns": cols,
            "missing_count": missing,
            "missing_pct": missing_pct,
            "duplicates": duplicates,
            "memory_mb": mem_mb,
            "quality_score": score,
        }

    def _column_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        stats: List[Dict[str, Any]] = []
        rows = len(df)

        for col in df.columns:
            s = df[col]
            missing = int(s.isna().sum())
            missing_pct = float((missing / rows) * 100) if rows else 0.0
            nunique = int(s.nunique(dropna=True))

            entry: Dict[str, Any] = {
                "name": str(col),
                "dtype": str(s.dtype),
                "unique": nunique,
                "missing_pct": missing_pct,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }

            if pd.api.types.is_numeric_dtype(s):
                cleaned = pd.to_numeric(s, errors="coerce")
                entry["mean"] = float(np.nanmean(cleaned.to_numpy())) if rows else None
                entry["median"] = float(np.nanmedian(cleaned.to_numpy())) if rows else None
                entry["min"] = float(np.nanmin(cleaned.to_numpy())) if rows else None
                entry["max"] = float(np.nanmax(cleaned.to_numpy())) if rows else None

            stats.append(entry)

        return stats

    def _quality_warnings(self, df: pd.DataFrame, columns: List[Dict[str, Any]]) -> List[str]:
        warnings: List[str] = []

        # High missing columns
        high_missing = [c["name"] for c in columns if c["missing_pct"] is not None and float(c["missing_pct"]) > 20.0]
        if high_missing:
            warnings.append(f"High missing values (>20%) in: {', '.join(high_missing[:8])}{'…' if len(high_missing) > 8 else ''}")

        # Skewness for numeric columns
        for c in columns:
            name = c["name"]
            s = df[name]
            if pd.api.types.is_numeric_dtype(s):
                vals = pd.to_numeric(s, errors="coerce").dropna()
                if len(vals) >= 30:
                    try:
                        skew = float(vals.skew())
                        if abs(skew) > 1.0:
                            warnings.append(f"Skewed distribution detected in '{name}' (skew={skew:.2f}).")
                            break
                    except Exception:
                        pass

        # Categorical encoding suggestion
        cat_cols = [c["name"] for c in columns if "object" in c["dtype"] or "category" in c["dtype"]]
        if cat_cols:
            warnings.append(f"Categorical columns present ({len(cat_cols)}). Consider encoding (label/one-hot).")

        # Outlier hint (simple IQR check)
        for c in columns:
            name = c["name"]
            s = df[name]
            if pd.api.types.is_numeric_dtype(s):
                vals = pd.to_numeric(s, errors="coerce").dropna()
                if len(vals) >= 50:
                    q1 = float(vals.quantile(0.25))
                    q3 = float(vals.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers = ((vals < (q1 - 1.5 * iqr)) | (vals > (q3 + 1.5 * iqr))).sum()
                        if outliers / len(vals) > 0.05:
                            warnings.append(f"Outliers detected in '{name}' (>{outliers} points).")
                            break

        return warnings

    def _schema_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        schema: Dict[str, Any] = {
            "potential_id_columns": [],
            "constant_columns": [],
            "inconsistent_type_columns": [],
            "datetime_like_columns": [],
        }

        rows = len(df)
        for col in df.columns:
            s = df[col]

            # constant
            if s.nunique(dropna=False) <= 1:
                schema["constant_columns"].append(str(col))

            # potential id
            if rows and s.nunique(dropna=True) == rows and self._looks_like_id(s):
                schema["potential_id_columns"].append(str(col))

            # inconsistent types (object columns with multiple python types)
            if s.dtype == object:
                types = set(type(v) for v in s.dropna().head(200).tolist())
                if len(types) > 1:
                    schema["inconsistent_type_columns"].append(str(col))

            # datetime-like
            if "date" in str(col).lower() or "time" in str(col).lower():
                parsed = pd.to_datetime(s, errors="coerce")
                ok_rate = float(parsed.notna().mean()) if rows else 0.0
                if ok_rate >= 0.8:
                    schema["datetime_like_columns"].append(str(col))

        return schema

    def _auto_inference(self, df: pd.DataFrame, columns: List[Dict[str, Any]], target: Optional[str]) -> Dict[str, Any]:
        suggested_target = target or self.suggest_target(df)
        problem = self.infer_problem_type(df, suggested_target)

        recommended_steps: List[Dict[str, Any]] = []

        # Recommend missing value handling
        high_missing_cols = [c["name"] for c in columns if float(c["missing_pct"]) > 0.0]
        if high_missing_cols:
            recommended_steps.append({"operation": "impute", "params": {"columns": [], "strategy": "mean"}})

        # Recommend encoding
        cat_cols = [c["name"] for c in columns if "object" in c["dtype"] or "category" in c["dtype"]]
        if cat_cols:
            recommended_steps.append({"operation": "encode", "params": {"columns": [], "method": "onehot"}})

        # Recommend scaling
        num_cols = [c["name"] for c in columns if c["mean"] is not None]
        if num_cols:
            recommended_steps.append({"operation": "scale", "params": {"columns": [], "method": "standard"}})

        return {
            "problem_type": problem,
            "suggested_target": suggested_target,
            "recommended_preprocessing": recommended_steps,
        }

    def _looks_like_id(self, s: pd.Series) -> bool:
        # Heuristic: mostly integer-like or short strings, very high uniqueness
        try:
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s):
                return True
        except Exception:
            pass
        return False
