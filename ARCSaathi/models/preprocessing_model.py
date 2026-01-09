"""
Preprocessing Model for ARCSaathi application.
Handles data preprocessing operations and pipeline management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from PySide6.QtCore import QObject, Signal
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression, chi2
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OrdinalEncoder


@dataclass
class PreprocessingStep:
    """Represents a single preprocessing step."""

    name: str
    operation: str
    params: Dict[str, Any]
    transformer: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "operation": self.operation, "params": self.params}


class PreprocessingModel(QObject):
    """
    Manages data preprocessing operations and pipeline.
    """
    
    # Signals
    step_added = Signal(str)  # step name
    step_removed = Signal(str)
    pipeline_applied = Signal(object)  # Processed DataFrame
    error_occurred = Signal(str)
    
    def __init__(self):
        """Initialize the preprocessing model."""
        super().__init__()
        
        self.steps: List[PreprocessingStep] = []
        self.original_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None

    def set_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Replace pipeline steps from a list of dicts."""
        self.steps = [PreprocessingStep(name=(s.get("name") or s.get("operation") or "Step"), operation=s.get("operation", ""), params=dict(s.get("params") or {})) for s in (steps or [])]

    def apply_steps_preview(self, df: pd.DataFrame, steps: List[Dict[str, Any]], *, upto_index: int = -1) -> pd.DataFrame:
        """Apply steps to a copy of df up to index for preview (does not mutate self)."""
        result = df.copy()
        lim = len(steps) if upto_index < 0 else min(len(steps), upto_index + 1)
        for i in range(lim):
            s = steps[i]
            step = PreprocessingStep(name=(s.get("name") or s.get("operation") or "Step"), operation=s.get("operation", ""), params=dict(s.get("params") or {}))
            result = self._apply_step(result, step)
        return result
    
    def set_data(self, data: pd.DataFrame):
        """
        Set the data to be preprocessed.
        
        Args:
            data: Input DataFrame
        """
        self.original_data = data.copy()
        self.processed_data = data.copy()
    
    def add_step(self, name: str, operation: str, params: Dict[str, Any]) -> bool:
        """
        Add a preprocessing step to the pipeline.
        
        Args:
            name: Name of the step
            operation: Type of operation
            params: Parameters for the operation
            
        Returns:
            True if successful
        """
        try:
            step = PreprocessingStep(name, operation, params)
            self.steps.append(step)
            self.step_added.emit(name)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Error adding step: {str(e)}")
            return False
    
    def remove_step(self, index: int) -> bool:
        """
        Remove a preprocessing step.
        
        Args:
            index: Index of the step to remove
            
        Returns:
            True if successful
        """
        try:
            if 0 <= index < len(self.steps):
                step_name = self.steps[index].name
                self.steps.pop(index)
                self.step_removed.emit(step_name)
                return True
            return False
        except Exception as e:
            self.error_occurred.emit(f"Error removing step: {str(e)}")
            return False
    
    def apply_pipeline(self) -> Optional[pd.DataFrame]:
        """
        Apply all preprocessing steps in sequence.
        
        Returns:
            Processed DataFrame or None if error
        """
        if self.original_data is None:
            self.error_occurred.emit("No data set for preprocessing")
            return None
        
        try:
            self.processed_data = self.original_data.copy()
            
            for step in self.steps:
                try:
                    self.processed_data = self._apply_step(self.processed_data, step)
                except Exception as e:
                    # Provide clearer context about the failing step
                    op = getattr(step, 'operation', '<op>')
                    name = getattr(step, 'name', '<step>')
                    raise RuntimeError(f"Step '{name}' ({op}) failed: {e}") from e
            
            self.pipeline_applied.emit(self.processed_data)
            return self.processed_data
            
        except Exception as e:
            self.error_occurred.emit(f"Error applying pipeline: {str(e)}")
            return None
    
    def _apply_step(self, data: pd.DataFrame, step: PreprocessingStep) -> pd.DataFrame:
        """
        Apply a single preprocessing step.
        
        Args:
            data: Input DataFrame
            step: PreprocessingStep to apply
            
        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        
        if step.operation == 'drop_missing':
            columns = step.params.get('columns', [])
            if columns:
                result = result.dropna(subset=columns)
            else:
                result = result.dropna()
        
        elif step.operation == 'impute':
            columns = step.params.get('columns', [])
            strategy = step.params.get('strategy', 'mean')
            fill_value = step.params.get('fill_value', None)

            if not columns:
                # Default: all columns with missing
                columns = [c for c in result.columns if result[c].isna().any()]

            if columns:
                # Strategy mapping
                if strategy == 'mode':
                    strategy = 'most_frequent'
                if strategy == 'constant':
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                else:
                    imputer = SimpleImputer(strategy=strategy)
                result[columns] = imputer.fit_transform(result[columns])
                step.transformer = imputer

        elif step.operation == 'interpolate':
            columns = step.params.get('columns', [])
            if not columns:
                columns = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
            if columns:
                result[columns] = result[columns].interpolate(limit_direction='both')
        
        elif step.operation == 'scale':
            columns = step.params.get('columns', [])
            method = step.params.get('method', 'standard')
            scope = step.params.get('scope', 'numeric')
            target = step.params.get('target', None)

            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'maxabs':
                scaler = MaxAbsScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            if not columns:
                numeric_cols = [
                    c
                    for c in result.columns
                    if pd.api.types.is_numeric_dtype(result[c]) and not pd.api.types.is_bool_dtype(result[c])
                ]
                if target in numeric_cols:
                    numeric_cols.remove(target)
                if scope == 'all':
                    columns = [c for c in result.columns if c != target]
                elif scope == 'exclude_binary':
                    def _is_binary(s: pd.Series) -> bool:
                        try:
                            u = int(s.dropna().nunique())
                            return u == 2
                        except Exception:
                            return False
                    columns = [c for c in numeric_cols if not _is_binary(result[c])]
                else:
                    columns = numeric_cols

            if columns:
                # Ensure numeric
                cols = [
                    c
                    for c in columns
                    if c in result.columns and pd.api.types.is_numeric_dtype(result[c]) and not pd.api.types.is_bool_dtype(result[c])
                ]
                if cols:
                    result[cols] = scaler.fit_transform(result[cols])
                    step.transformer = scaler
        
        elif step.operation == 'encode':
            columns = step.params.get('columns', [])
            method = step.params.get('method', 'label')
            max_categories = int(step.params.get('max_categories', 0) or 0)
            target = step.params.get('target', None)
            handle_unknown = step.params.get('handle_unknown', 'ignore')

            if not columns:
                columns = [c for c in result.columns if (result[c].dtype == object or 'category' in str(result[c].dtype))]
                if target in columns:
                    columns.remove(target)

            for col in columns:
                if col not in result.columns:
                    continue
                s = result[col].astype(str)

                if max_categories and s.nunique(dropna=True) > max_categories:
                    top = s.value_counts().head(max_categories).index.tolist()
                    s = s.where(s.isin(top), other='__OTHER__')
                    result[col] = s

                if method == 'label':
                    encoder = LabelEncoder()
                    result[col] = encoder.fit_transform(result[col].astype(str))
                    step.transformer = encoder
                elif method == 'ordinal':
                    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    result[[col]] = enc.fit_transform(result[[col]].astype(str))
                    step.transformer = enc
                elif method == 'frequency':
                    freq = result[col].astype(str).value_counts(dropna=False)
                    result[col] = result[col].astype(str).map(freq).fillna(0).astype(float)
                elif method == 'target':
                    if target and target in result.columns:
                        y = pd.to_numeric(result[target], errors='coerce')
                        means = pd.DataFrame({'x': result[col].astype(str), 'y': y}).groupby('x')['y'].mean()
                        global_mean = float(y.mean()) if y.notna().any() else 0.0
                        mapped = result[col].astype(str).map(means)
                        if handle_unknown == 'error' and mapped.isna().any():
                            raise ValueError(f"Unknown categories found in '{col}'")
                        result[col] = mapped.fillna(global_mean).astype(float)
                    else:
                        # Fallback to frequency if target not available
                        freq = result[col].astype(str).value_counts(dropna=False)
                        result[col] = result[col].astype(str).map(freq).fillna(0).astype(float)
                else:
                    # onehot default
                    dummies = pd.get_dummies(result[col].astype(str), prefix=col)
                    result = pd.concat([result.drop(columns=[col]), dummies], axis=1)
        
        elif step.operation == 'drop_columns':
            columns = step.params.get('columns', [])
            result = result.drop(columns=columns, errors='ignore')
        
        elif step.operation == 'rename_columns':
            mapping = step.params.get('mapping', {})
            result = result.rename(columns=mapping)

        elif step.operation == 'feature_selection':
            method = step.params.get('method', 'correlation')
            target = step.params.get('target', None)
            task_type = step.params.get('task_type', 'classification')

            X = result
            y = None
            if target and target in result.columns:
                y = result[target]
                X = result.drop(columns=[target])

            # Only numeric features for these selectors
            num_cols = [
                c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
            ]
            Xn = X[num_cols].copy() if num_cols else pd.DataFrame(index=X.index)

            if method == 'correlation':
                thr = float(step.params.get('threshold', 0.95))
                if not Xn.empty:
                    corr = Xn.corr().abs()
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    to_drop = [c for c in upper.columns if any(upper[c] > thr)]
                    result = result.drop(columns=to_drop, errors='ignore')

            elif method == 'variance':
                thr = float(step.params.get('threshold', 0.0))
                if not Xn.empty:
                    vt = VarianceThreshold(threshold=thr)
                    kept = vt.fit_transform(Xn.fillna(0.0))
                    keep_mask = vt.get_support()
                    keep_cols = [c for c, keep in zip(Xn.columns.tolist(), keep_mask) if keep]
                    drop_cols = [c for c in Xn.columns.tolist() if c not in keep_cols]
                    result = result.drop(columns=drop_cols, errors='ignore')
                    step.transformer = vt

            elif method == 'select_k_best':
                if y is None:
                    raise ValueError("SelectKBest requires a target column")
                k = int(step.params.get('k', 20))
                score_name = str(step.params.get('score_func', 'f_classif'))
                score_func = {
                    'f_classif': f_classif,
                    'f_regression': f_regression,
                    'mutual_info_classif': mutual_info_classif,
                    'mutual_info_regression': mutual_info_regression,
                    'chi2': chi2,
                }.get(score_name, f_classif)

                selector = SelectKBest(score_func=score_func, k=min(k, max(1, Xn.shape[1])))
                selector.fit(Xn.fillna(0.0), y)
                keep_mask = selector.get_support()
                keep_cols = [c for c, keep in zip(Xn.columns.tolist(), keep_mask) if keep]
                drop_cols = [c for c in Xn.columns.tolist() if c not in keep_cols]
                result = result.drop(columns=drop_cols, errors='ignore')
                step.transformer = selector

            elif method == 'rfe':
                if y is None:
                    raise ValueError("RFE requires a target column")
                k = int(step.params.get('k', 10))
                if task_type == 'regression':
                    est = LinearRegression()
                else:
                    est = LogisticRegression(max_iter=500)
                selector = RFE(estimator=est, n_features_to_select=min(k, max(1, Xn.shape[1])))
                selector.fit(Xn.fillna(0.0), y)
                keep_mask = selector.get_support()
                keep_cols = [c for c, keep in zip(Xn.columns.tolist(), keep_mask) if keep]
                drop_cols = [c for c in Xn.columns.tolist() if c not in keep_cols]
                result = result.drop(columns=drop_cols, errors='ignore')
                step.transformer = selector

        elif step.operation == 'outliers':
            method = step.params.get('method', 'iqr')
            target = step.params.get('target', None)
            cols = [
                c
                for c in result.columns
                if pd.api.types.is_numeric_dtype(result[c]) and not pd.api.types.is_bool_dtype(result[c]) and c != target
            ]
            if not cols:
                return result
            X = result[cols].apply(pd.to_numeric, errors='coerce')

            if method == 'zscore':
                thr = float(step.params.get('threshold', 3.0))
                mu = X.mean()
                sigma = X.std(ddof=0).replace(0, np.nan)
                z = (X - mu) / sigma
                clipped = X.copy()
                for c in cols:
                    if sigma[c] is None or np.isnan(float(sigma[c])):
                        continue
                    lo = float(mu[c] - thr * sigma[c])
                    hi = float(mu[c] + thr * sigma[c])
                    clipped[c] = clipped[c].clip(lo, hi)
                result[cols] = clipped

            elif method == 'iqr':
                mult = float(step.params.get('multiplier', 1.5))
                q1 = X.quantile(0.25)
                q3 = X.quantile(0.75)
                iqr = (q3 - q1).replace(0, np.nan)
                clipped = X.copy()
                for c in cols:
                    if np.isnan(float(iqr[c])):
                        continue
                    lo = float(q1[c] - mult * iqr[c])
                    hi = float(q3[c] + mult * iqr[c])
                    clipped[c] = clipped[c].clip(lo, hi)
                result[cols] = clipped

            elif method == 'isolation_forest':
                cont = float(step.params.get('contamination', 0.05))
                iso = IsolationForest(contamination=cont, random_state=42)
                mask = iso.fit_predict(X.fillna(0.0))
                # drop outlier rows
                result = result.loc[mask == 1].reset_index(drop=True)
                step.transformer = iso

            elif method == 'winsorize':
                tail = float(step.params.get('tail', 0.01))
                lo = X.quantile(tail)
                hi = X.quantile(1.0 - tail)
                clipped = X.copy()
                for c in cols:
                    clipped[c] = clipped[c].clip(float(lo[c]), float(hi[c]))
                result[cols] = clipped
        
        return result
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """
        Get all preprocessing steps.
        
        Returns:
            List of step dictionaries
        """
        return [step.to_dict() for step in self.steps]
    
    def clear_pipeline(self):
        """Clear all preprocessing steps."""
        self.steps.clear()
        self.processed_data = self.original_data.copy() if self.original_data is not None else None
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Get the processed data.
        
        Returns:
            Processed DataFrame or None
        """
        return self.processed_data
    
    def handle_missing_values(self, columns: List[str], strategy: str = 'mean') -> bool:
        """
        Convenience method to handle missing values.
        
        Args:
            columns: Columns to process
            strategy: Imputation strategy (mean, median, mode, drop)
            
        Returns:
            True if successful
        """
        if strategy == 'drop':
            return self.add_step(
                f"Drop Missing ({', '.join(columns)})",
                'drop_missing',
                {'columns': columns}
            )
        else:
            return self.add_step(
                f"Impute ({strategy}) - {', '.join(columns)}",
                'impute',
                {'columns': columns, 'strategy': strategy}
            )
    
    def scale_features(self, columns: List[str], method: str = 'standard') -> bool:
        """
        Convenience method to scale features.
        
        Args:
            columns: Columns to scale
            method: Scaling method (standard, minmax)
            
        Returns:
            True if successful
        """
        return self.add_step(
            f"Scale ({method}) - {', '.join(columns)}",
            'scale',
            {'columns': columns, 'method': method}
        )
    
    def encode_categorical(self, columns: List[str], method: str = 'label') -> bool:
        """
        Convenience method to encode categorical variables.
        
        Args:
            columns: Columns to encode
            method: Encoding method (label, onehot)
            
        Returns:
            True if successful
        """
        return self.add_step(
            f"Encode ({method}) - {', '.join(columns)}",
            'encode',
            {'columns': columns, 'method': method}
        )
