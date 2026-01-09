"""
Evaluation Model for ARCSaathi application.
Handles model evaluation and performance metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from PySide6.QtCore import QObject, Signal
from datetime import datetime


@dataclass
class EvaluationResult:
    model_name: str
    task_type: str
    metrics: Dict[str, float]
    timestamp: datetime

    # artifacts (optional)
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    # operational metrics
    training_time_s: Optional[float] = None
    inference_time_ms: Optional[float] = None
    model_size_mb: Optional[float] = None

    # model insights
    feature_importance: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "training_time_s": self.training_time_s,
            "inference_time_ms": self.inference_time_ms,
            "model_size_mb": self.model_size_mb,
        }


class EvaluationModel(QObject):
    """
    Manages model evaluation, metrics calculation, and comparison.
    """
    
    # Signals
    evaluation_complete = Signal(str)  # model name
    comparison_ready = Signal(object)  # comparison DataFrame
    error_occurred = Signal(str)
    
    def __init__(self):
        """Initialize the evaluation model."""
        super().__init__()
        
        self.results: Dict[str, EvaluationResult] = {}

        # Cache for statistical testing / CI: (dataset_sig, task_type) -> {model_name: per_sample_loss}
        self._loss_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    
    def evaluate_classification(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        *,
        training_time_s: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        model_size_mb: Optional[float] = None,
        dataset_sig: Optional[str] = None,
    ) -> Optional[EvaluationResult]:
        """
        Evaluate a classification model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            EvaluationResult or None
        """
        try:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
                log_loss,
                cohen_kappa_score,
                matthews_corrcoef,
            )
            
            metrics: Dict[str, float] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
                "matthews_corr": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) <= 2 else float("nan"),
            }
            
            # Add ROC AUC + log loss if probabilities are provided
            if y_pred_proba is not None:
                try:
                    # For binary classification
                    if y_pred_proba.ndim == 1 or (y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2):
                        proba_1 = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
                        metrics["roc_auc"] = float(roc_auc_score(y_true, proba_1))
                    else:
                        # For multi-class
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")
                        )
                except Exception:
                    pass  # Skip ROC AUC if calculation fails

                try:
                    metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                except Exception:
                    pass

            # per-sample loss for paired tests
            if dataset_sig:
                key = (str(dataset_sig), "classification")
                per_sample = (y_true != y_pred).astype(float)
                self._loss_cache.setdefault(key, {})[model_name] = per_sample
            
            result = EvaluationResult(
                model_name=model_name,
                task_type="classification",
                metrics=metrics,
                timestamp=datetime.now(),
                y_true=np.asarray(y_true),
                y_pred=np.asarray(y_pred),
                y_proba=None if y_pred_proba is None else np.asarray(y_pred_proba),
                residuals=None,
                training_time_s=training_time_s,
                inference_time_ms=inference_time_ms,
                model_size_mb=model_size_mb,
            )
            self.results[model_name] = result
            
            self.evaluation_complete.emit(model_name)
            return result
            
        except Exception as e:
            self.error_occurred.emit(f"Error evaluating classification model: {str(e)}")
            return None
    
    def evaluate_regression(self, model_name: str, y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           *,
                           n_features: Optional[int] = None,
                           training_time_s: Optional[float] = None,
                           inference_time_ms: Optional[float] = None,
                           model_size_mb: Optional[float] = None,
                           dataset_sig: Optional[str] = None,
                           ) -> Optional[EvaluationResult]:
        """
        Evaluate a regression model.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            EvaluationResult or None
        """
        try:
            from sklearn.metrics import (
                mean_squared_error,
                mean_absolute_error,
                r2_score,
                mean_absolute_percentage_error,
                mean_squared_log_error,
                explained_variance_score,
            )
            
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))

            metrics: Dict[str, float] = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "explained_variance": float(explained_variance_score(y_true, y_pred)),
            }

            # Adjusted R^2 if feature count known
            try:
                p = int(n_features) if n_features is not None else None
                n = int(len(y_true))
                if p is not None and n > p + 1:
                    metrics["adj_r2"] = float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))
            except Exception:
                pass
            
            # Add MAPE if no zero values
            if not np.any(y_true == 0):
                try:
                    metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
                except Exception:
                    pass

            # MSLE requires non-negative values
            try:
                if np.all(y_true >= 0) and np.all(y_pred >= 0):
                    metrics["msle"] = float(mean_squared_log_error(y_true, y_pred))
            except Exception:
                pass

            if dataset_sig:
                key = (str(dataset_sig), "regression")
                per_sample = (y_true - y_pred) ** 2
                self._loss_cache.setdefault(key, {})[model_name] = per_sample
            
            residuals = y_true - y_pred
            result = EvaluationResult(
                model_name=model_name,
                task_type="regression",
                metrics=metrics,
                timestamp=datetime.now(),
                y_true=y_true,
                y_pred=y_pred,
                y_proba=None,
                residuals=residuals,
                training_time_s=training_time_s,
                inference_time_ms=inference_time_ms,
                model_size_mb=model_size_mb,
            )
            self.results[model_name] = result
            
            self.evaluation_complete.emit(model_name)
            return result
            
        except Exception as e:
            self.error_occurred.emit(f"Error evaluating regression model: {str(e)}")
            return None
    
    def get_result(self, model_name: str) -> Optional[EvaluationResult]:
        """
        Get evaluation result for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            EvaluationResult or None
        """
        return self.results.get(model_name)
    
    def get_metrics(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get metrics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics or None
        """
        result = self.results.get(model_name)
        return result.metrics if result else None
    
    def compare_models(self, model_names: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Compare multiple models.
        
        Args:
            model_names: List of model names to compare (None = all models)
            
        Returns:
            DataFrame with comparison or None
        """
        try:
            if model_names is None:
                model_names = list(self.results.keys())
            
            if not model_names:
                self.error_occurred.emit("No models to compare")
                return None
            
            # Collect metrics from all models
            comparison_data = []
            
            for name in model_names:
                if name in self.results:
                    result = self.results[name]
                    row = {
                        "Model": name,
                        "Task": result.task_type,
                        "Training Time (s)": result.training_time_s,
                        "Inference Time (ms)": result.inference_time_ms,
                        "Model Size (MB)": result.model_size_mb,
                    }
                    row.update(result.metrics)
                    comparison_data.append(row)
            
            if not comparison_data:
                return None
            
            df = pd.DataFrame(comparison_data)
            self.comparison_ready.emit(df)
            
            return df
            
        except Exception as e:
            self.error_occurred.emit(f"Error comparing models: {str(e)}")
            return None
    
    def get_best_model(self, metric: str = 'accuracy', maximize: bool = True) -> Optional[str]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            maximize: Whether higher is better (True) or lower is better (False)
            
        Returns:
            Name of the best model or None
        """
        if not self.results:
            return None
        
        try:
            best_model = None
            best_value = float('-inf') if maximize else float('inf')
            
            for name, result in self.results.items():
                if metric in result.metrics:
                    value = result.metrics[metric]
                    
                    if maximize:
                        if value > best_value:
                            best_value = value
                            best_model = name
                    else:
                        if value < best_value:
                            best_value = value
                            best_model = name
            
            return best_model
            
        except Exception as e:
            self.error_occurred.emit(f"Error finding best model: {str(e)}")
            return None
    
    def get_confusion_matrix(self, model_name: str, y_true: np.ndarray) -> Optional[np.ndarray]:
        """
        Get confusion matrix for a classification model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            
        Returns:
            Confusion matrix or None
        """
        result = self.results.get(model_name)
        
        if result is None or result.y_pred is None:
            self.error_occurred.emit(f"No predictions found for model '{model_name}'")
            return None
        
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, result.y_pred)
            return cm
            
        except Exception as e:
            self.error_occurred.emit(f"Error computing confusion matrix: {str(e)}")
            return None
    
    def export_results(self, file_path: str) -> bool:
        """
        Export all evaluation results to a file.
        
        Args:
            file_path: Path to save results
            
        Returns:
            True if successful
        """
        try:
            comparison_df = self.compare_models()
            
            if comparison_df is not None:
                comparison_df.to_csv(file_path, index=False)
                return True
            
            return False
            
        except Exception as e:
            self.error_occurred.emit(f"Error exporting results: {str(e)}")
            return False

    # ---- Additional evaluation types ----

    def evaluate_clustering(
        self,
        model_name: str,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        training_time_s: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        model_size_mb: Optional[float] = None,
        dataset_sig: Optional[str] = None,
    ) -> Optional[EvaluationResult]:
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

            X = np.asarray(X)
            labels = np.asarray(labels)

            metrics: Dict[str, float] = {}
            # silhouette requires >=2 clusters and no singletons edge cases
            try:
                if len(np.unique(labels)) >= 2:
                    metrics["silhouette"] = float(silhouette_score(X, labels))
            except Exception:
                pass

            try:
                if len(np.unique(labels)) >= 2:
                    metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
                    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            except Exception:
                pass

            if dataset_sig:
                key = (str(dataset_sig), "clustering")
                # lower DB is better; use DB if available else 1-silhouette
                if "davies_bouldin" in metrics:
                    per_sample = np.full(shape=(X.shape[0],), fill_value=float(metrics["davies_bouldin"]))
                else:
                    per_sample = np.full(shape=(X.shape[0],), fill_value=float(1.0 - metrics.get("silhouette", 0.0)))
                self._loss_cache.setdefault(key, {})[model_name] = per_sample

            result = EvaluationResult(
                model_name=model_name,
                task_type="clustering",
                metrics=metrics,
                timestamp=datetime.now(),
                y_true=None,
                y_pred=np.asarray(labels),
                y_proba=None,
                residuals=None,
                training_time_s=training_time_s,
                inference_time_ms=inference_time_ms,
                model_size_mb=model_size_mb,
            )
            self.results[model_name] = result
            self.evaluation_complete.emit(model_name)
            return result
        except Exception as e:
            self.error_occurred.emit(f"Error evaluating clustering model: {str(e)}")
            return None

    def evaluate_dimred(
        self,
        model_name: str,
        X: np.ndarray,
        X_reduced: np.ndarray,
        *,
        explained_variance_ratio: Optional[np.ndarray] = None,
        reconstruction_error: Optional[float] = None,
        training_time_s: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        model_size_mb: Optional[float] = None,
        dataset_sig: Optional[str] = None,
    ) -> Optional[EvaluationResult]:
        try:
            metrics: Dict[str, float] = {}
            if explained_variance_ratio is not None:
                try:
                    metrics["explained_variance_ratio_sum"] = float(np.sum(explained_variance_ratio))
                except Exception:
                    pass
            if reconstruction_error is not None:
                metrics["reconstruction_error"] = float(reconstruction_error)

            if dataset_sig:
                key = (str(dataset_sig), "dimred")
                per_sample = np.full(shape=(np.asarray(X).shape[0],), fill_value=float(metrics.get("reconstruction_error", 0.0)))
                self._loss_cache.setdefault(key, {})[model_name] = per_sample

            result = EvaluationResult(
                model_name=model_name,
                task_type="dimred",
                metrics=metrics,
                timestamp=datetime.now(),
                y_true=None,
                y_pred=np.asarray(X_reduced),
                y_proba=None,
                residuals=None,
                training_time_s=training_time_s,
                inference_time_ms=inference_time_ms,
                model_size_mb=model_size_mb,
            )
            self.results[model_name] = result
            self.evaluation_complete.emit(model_name)
            return result
        except Exception as e:
            self.error_occurred.emit(f"Error evaluating dimensionality reduction model: {str(e)}")
            return None

    # ---- Statistical helpers ----

    def paired_ttest(self, *, dataset_sig: str, task_type: str, a: str, b: str) -> Optional[Dict[str, float]]:
        """Paired t-test over cached per-sample losses for models a vs b."""
        try:
            loss = self._loss_cache.get((str(dataset_sig), str(task_type)), {})
            xa = loss.get(a)
            xb = loss.get(b)
            if xa is None or xb is None:
                return None
            xa = np.asarray(xa, dtype=float)
            xb = np.asarray(xb, dtype=float)
            n = int(min(len(xa), len(xb)))
            if n < 5:
                return None
            xa = xa[:n]
            xb = xb[:n]
            diff = xa - xb

            # Prefer SciPy if available
            try:
                from scipy.stats import ttest_rel

                t, p = ttest_rel(xa, xb, nan_policy="omit")
                return {"t": float(t), "p": float(p), "mean_diff": float(np.nanmean(diff)), "n": float(n)}
            except Exception:
                # Fallback: normal approx
                m = float(np.nanmean(diff))
                s = float(np.nanstd(diff, ddof=1))
                if s <= 0:
                    return None
                t = m / (s / np.sqrt(n))
                # two-sided approx using erfc
                import math

                p = float(math.erfc(abs(t) / np.sqrt(2.0)))
                return {"t": float(t), "p": float(p), "mean_diff": float(m), "n": float(n)}
        except Exception:
            return None

    def bootstrap_ci(
        self,
        values: np.ndarray,
        *,
        alpha: float = 0.05,
        n_boot: int = 500,
        random_state: int = 42,
    ) -> Optional[Dict[str, float]]:
        try:
            v = np.asarray(values, dtype=float)
            v = v[np.isfinite(v)]
            if v.size < 5:
                return None
            rng = np.random.default_rng(int(random_state))
            means = []
            n = int(v.size)
            for _ in range(int(n_boot)):
                idx = rng.integers(0, n, size=n)
                means.append(float(np.mean(v[idx])))
            lo = float(np.quantile(means, alpha / 2))
            hi = float(np.quantile(means, 1 - alpha / 2))
            return {"lo": lo, "hi": hi, "mean": float(np.mean(v))}
        except Exception:
            return None
    
    def clear_results(self):
        """Clear all evaluation results."""
        self.results.clear()
    
    def list_models(self) -> List[str]:
        """
        Get list of evaluated models.
        
        Returns:
            List of model names
        """
        return list(self.results.keys())
