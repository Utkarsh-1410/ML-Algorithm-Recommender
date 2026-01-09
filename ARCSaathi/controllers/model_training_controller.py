"""Model Training Module controller.

Covers Tab 3 (Model Training & Tuning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple

import numpy as np

import pandas as pd

from PySide6.QtCore import Signal, QThreadPool

from .base_controller import BaseController
from ..models import DataModel, MLModel, ModelStore
from ..state import AppState
from ..utils import get_logger
from ..utils.qt_worker import Worker


class ModelTrainingController(BaseController):
    models_changed = Signal()
    model_trained = Signal(str)
    error_occurred = Signal(str)

    # Batch job signals
    job_enqueued = Signal(str, str)  # run_id, model_name
    job_started = Signal(str, str)  # run_id, model_name
    job_progress = Signal(str, int)
    job_finished = Signal(str, str, dict)  # run_id, status, payload
    queue_changed = Signal()

    def __init__(self, state: AppState, data_model: DataModel, ml_model: MLModel):
        super().__init__()
        self._log = get_logger("controllers.model_training")

        self.state = state
        self.data_model = data_model
        self.ml_model = ml_model

        self.store = ModelStore()

        self._pool = QThreadPool.globalInstance()
        self._max_parallel = max(1, min(4, self._pool.maxThreadCount()))
        self._paused = False
        self._cancel_all = False

        self._queue: List[TrainingJobRequest] = []
        self._running: Dict[str, Worker] = {}

        self.ml_model.error_occurred.connect(self.error_occurred)
        self.ml_model.model_trained.connect(self._on_model_trained)

        self.target_column: Optional[str] = None

    def list_registry(self) -> Dict[str, Any]:
        return self.ml_model.list_registry()

    def list_models(self) -> list[str]:
        return self.ml_model.list_models()

    def list_recent_runs(self, limit: int = 200) -> list:
        return self.store.list_runs(limit=limit)

    def _on_model_trained(self, name: str) -> None:
        # Keep state in sync for any training pathway that emits ml_model.model_trained
        try:
            info = self.ml_model.get_model_info(name) or {}
            self.state.upsert_trained_model(name, info)
            self.state.save()
            self.models_changed.emit()
        except Exception:
            pass

        self.model_trained.emit(name)
        self._log.info("Model trained: %s", name)

    def create_model(self, name: str, model_type: str, params: Dict[str, Any]) -> bool:
        ok = self.ml_model.create_model(name=name, model_type=model_type, params=params)
        if ok:
            info = self.ml_model.get_model_info(name) or {}
            self.state.upsert_trained_model(name, info)
            self.state.save()
            self.models_changed.emit()
            self._log.info("Model created: %s (%s)", name, model_type)
        return ok

    def set_target(self, column: str) -> None:
        self.target_column = column

    def train(self, name: str) -> bool:
        df = self.data_model.get_data()
        if df is None or df.empty:
            self.error_occurred.emit("No dataset available for training")
            return False

        if not self.target_column or self.target_column not in df.columns:
            self.error_occurred.emit("Target column not set (or missing)")
            return False

        try:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            # Coerce features into model-friendly numeric dtypes
            X = self._coerce_features(X)
            from sklearn.model_selection import train_test_split

            X_train, _, y_train, _ = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )

            ok = self.ml_model.train_model(name, X_train, y_train)
            if ok:
                info = self.ml_model.get_model_info(name) or {}
                self.state.upsert_trained_model(name, info)
                self.state.save()
            return ok
        except Exception as exc:
            self.error_occurred.emit(f"Training failed: {exc}")
            return False

    def _coerce_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime/bool/object columns to numeric-friendly types for estimators.

        - datetime → float seconds since epoch (NaT → NaN)
        - bool → uint8 (0/1)
        - category → codes (with -1 as NaN)
        - object → try numeric, else string factorization codes
        """
        Xc = X.copy()
        for col in Xc.columns:
            s = Xc[col]
            try:
                if pd.api.types.is_datetime64_any_dtype(s):
                    sd = pd.to_datetime(s, errors="coerce")
                    mask = sd.isna()
                    # nanoseconds to seconds as float
                    vals = sd.view("int64").astype("float64") / 1e9
                    vals[mask] = np.nan
                    Xc[col] = vals
                    continue
                if pd.api.types.is_bool_dtype(s):
                    Xc[col] = s.astype("uint8")
                    continue
                if pd.api.types.is_categorical_dtype(s):
                    codes = s.cat.codes.astype("float64")
                    Xc[col] = codes.replace(-1, np.nan)
                    continue
                if s.dtype == object:
                    # try numeric first
                    sn = pd.to_numeric(s, errors="coerce")
                    if sn.notna().any():
                        Xc[col] = sn
                    else:
                        codes, _ = pd.factorize(s.astype(str), sort=True)
                        Xc[col] = codes.astype("float64")
                    continue
            except Exception:
                # best-effort: leave as-is
                pass
        return Xc

    def set_parallelism(self, max_parallel: int) -> None:
        self._max_parallel = max(1, int(max_parallel))
        self._pump_queue()

    def pause_queue(self) -> None:
        self._paused = True
        self.queue_changed.emit()

    def resume_queue(self) -> None:
        self._paused = False
        self._pump_queue()

    def cancel_all(self) -> None:
        # Note: cannot safely kill running sklearn fits; we stop dispatching new jobs.
        self._cancel_all = True
        self._queue.clear()
        self.queue_changed.emit()

    def enqueue_jobs(self, jobs: List[Dict[str, Any]], config: Dict[str, Any]) -> List[str]:
        """Enqueue jobs for execution.

        jobs: [{algorithm_key, params, label?}]
        config: {task_type, target, test_size, random_state, tuning_mode, cv_folds, n_iter}
        """
        run_ids: List[str] = []
        for j in jobs or []:
            algorithm_key = str(j.get("algorithm_key") or "").strip()
            if not algorithm_key:
                continue
            run_id = self.store.new_run_id()
            run_ids.append(run_id)
            req = TrainingJobRequest(
                run_id=run_id,
                algorithm_key=algorithm_key,
                params=dict(j.get("params") or {}),
                label=str(j.get("label") or ""),
                config=dict(config or {}),
            )
            self._queue.append(req)
            self.job_enqueued.emit(run_id, self._registry_name(algorithm_key))
            self.store.save_run(
                run_id=run_id,
                task_type=str(req.config.get("task_type") or ""),
                algorithm_key=algorithm_key,
                model_name=self._registry_name(algorithm_key),
                status="queued",
                metrics={},
                params=req.params,
                artifact_path=None,
            )

        self.queue_changed.emit()
        self._pump_queue()
        return run_ids

    def _registry_name(self, algorithm_key: str) -> str:
        reg = self.ml_model.list_registry()
        card = reg.get(algorithm_key, {}).get("card")
        return getattr(card, "name", algorithm_key)

    def _pump_queue(self) -> None:
        if self._paused or self._cancel_all:
            return

        while self._queue and len(self._running) < self._max_parallel:
            req = self._queue.pop(0)
            self._start_job(req)
            self.queue_changed.emit()

    def _start_job(self, req: "TrainingJobRequest") -> None:
        if self._cancel_all:
            return

        def _progress(pct: int) -> None:
            self.job_progress.emit(req.run_id, int(pct))

        worker = Worker(self._run_job, req, on_progress=_progress)

        worker.signals.result.connect(lambda payload, rid=req.run_id: self._on_job_result(rid, payload))
        worker.signals.error.connect(lambda msg, rid=req.run_id: self._on_job_error(rid, msg))
        worker.signals.finished.connect(lambda rid=req.run_id: self._on_job_finished(rid))

        self._running[req.run_id] = worker
        self.job_started.emit(req.run_id, self._registry_name(req.algorithm_key))
        self.store.save_run(
            run_id=req.run_id,
            task_type=str(req.config.get("task_type") or ""),
            algorithm_key=req.algorithm_key,
            model_name=self._registry_name(req.algorithm_key),
            status="running",
            metrics={},
            params=req.params,
            artifact_path=None,
        )
        self._pool.start(worker)

    def _on_job_result(self, run_id: str, payload: dict) -> None:
        status = str(payload.get("status") or "finished")
        self.job_finished.emit(run_id, status, payload)

    def _on_job_error(self, run_id: str, msg: str) -> None:
        payload = {"status": "failed", "error": msg}
        self.job_finished.emit(run_id, "failed", payload)
        self.store.save_run(
            run_id=run_id,
            task_type="",
            algorithm_key="",
            model_name="",
            status="failed",
            metrics={"error": msg},
            params={},
            artifact_path=None,
        )

    def _on_job_finished(self, run_id: str) -> None:
        self._running.pop(run_id, None)
        self._pump_queue()

    def _run_job(self, req: "TrainingJobRequest", _progress: Optional[Callable[[int], None]] = None) -> dict:
        """Execute one training job and persist results."""
        if self._cancel_all:
            return {"status": "canceled"}

        df = self.data_model.get_data()
        if df is None or df.empty:
            raise RuntimeError("No dataset available")

        task_type = str(req.config.get("task_type") or "")
        target = str(req.config.get("target") or self.target_column or "").strip() or None
        test_size = float(req.config.get("test_size") or 0.2)
        random_state = int(req.config.get("random_state") or 42)
        tuning_mode = str(req.config.get("tuning_mode") or "none")
        cv_folds = int(req.config.get("cv_folds") or 3)
        n_iter = int(req.config.get("n_iter") or 20)

        # Build X/y depending on task
        if task_type in ("classification", "regression"):
            if not target or target not in df.columns:
                raise RuntimeError("Target column not set (or missing)")
            X = df.drop(columns=[target])
            y = df[target]
        else:
            X = df.copy()
            y = None

        # Coerce features regardless of task
        X = self._coerce_features(X)

        # Create model
        model_name = f"{req.algorithm_key}_{req.run_id}"
        if not self.ml_model.create_model_from_registry(model_name, req.algorithm_key, req.params):
            raise RuntimeError("Failed to create model")

        if _progress:
            _progress(5)

        # Train + evaluate
        metrics: Dict[str, Any] = {}
        artifact_path: Optional[str] = None

        if task_type in ("classification", "regression"):
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )

            if tuning_mode in ("grid", "random", "optuna", "hyperopt"):
                best_params, best_score = self._tune_supervised(
                    req=req,
                    X_train=X_train,
                    y_train=y_train,
                    task_type=task_type,
                    mode=tuning_mode,
                    cv_folds=cv_folds,
                    n_iter=n_iter,
                    _progress=_progress,
                )
                req.params.update(best_params)
                # re-create model with best params
                self.ml_model.create_model_from_registry(model_name, req.algorithm_key, req.params)
                metrics["tuning_best_cv_score"] = best_score

            if _progress:
                _progress(35)

            ok = self.ml_model.fit_model(model_name, X_train, y_train)
            if not ok:
                raise RuntimeError("Fit failed")

            if _progress:
                _progress(70)

            preds = self.ml_model.predict(model_name, X_test)
            if preds is None:
                raise RuntimeError("Prediction failed")

            metrics.update(self._compute_supervised_metrics(task_type, y_test, preds))
        elif task_type == "clustering":
            est = self.ml_model.get_model(model_name)
            if est is None:
                raise RuntimeError("Model not found")

            if hasattr(est.model, "fit_predict"):
                labels = est.model.fit_predict(X)
            else:
                est.model.fit(X)
                labels = getattr(est.model, "labels_", None)

            if _progress:
                _progress(80)

            metrics.update(self._compute_clustering_metrics(X, labels))
        elif task_type == "dimred":
            est = self.ml_model.get_model(model_name)
            if est is None:
                raise RuntimeError("Model not found")
            if hasattr(est.model, "fit_transform"):
                emb = est.model.fit_transform(X)
                metrics["embedding_shape"] = list(getattr(emb, "shape", ()))
            else:
                est.model.fit(X)
                metrics["status"] = "fit"

            if hasattr(est.model, "explained_variance_ratio_"):
                try:
                    evr = est.model.explained_variance_ratio_
                    metrics["explained_variance_sum"] = float(sum(evr))
                except Exception:
                    pass

        # Persist artifact (best-effort)
        try:
            from joblib import dump as joblib_dump  # type: ignore

            fpath = self.store.artifact_file(req.run_id)
            wrapper = self.ml_model.get_model(model_name)
            joblib_dump(wrapper, str(fpath))
            artifact_path = str(fpath)
        except Exception:
            try:
                fpath = self.store.artifact_file(req.run_id)
                wrapper = self.ml_model.get_model(model_name)
                with open(str(fpath), "wb") as f:
                    pickle.dump(wrapper, f)
                artifact_path = str(fpath)
            except Exception:
                artifact_path = None

        self.store.save_run(
            run_id=req.run_id,
            task_type=task_type,
            algorithm_key=req.algorithm_key,
            model_name=self._registry_name(req.algorithm_key),
            status="finished",
            metrics=metrics,
            params=req.params,
            artifact_path=artifact_path,
        )

        # Also update JSON state with the latest run under this algorithm key
        self.state.upsert_trained_model(
            req.algorithm_key,
            {
                "run_id": req.run_id,
                "task_type": task_type,
                "model_name": self._registry_name(req.algorithm_key),
                "metrics": metrics,
                "params": req.params,
                "artifact_path": artifact_path,
            },
        )
        self.state.save()

        return {
            "status": "finished",
            "run_id": req.run_id,
            "algorithm_key": req.algorithm_key,
            "model_name": self._registry_name(req.algorithm_key),
            "task_type": task_type,
            "metrics": metrics,
            "params": req.params,
            "artifact_path": artifact_path,
        }

    def _compute_supervised_metrics(self, task_type: str, y_true, y_pred) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

        if task_type == "regression":
            # Compatibility: some sklearn versions don't support mean_squared_error(..., squared=False)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            return {"rmse": rmse, "mae": mae, "r2": r2}

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="weighted"))
        return {"accuracy": acc, "f1_weighted": f1}

    def _compute_clustering_metrics(self, X: pd.DataFrame, labels) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if labels is None:
            return {"status": "fit", "note": "No labels produced"}
        try:
            import numpy as np

            unique = sorted(set(int(x) for x in labels if x is not None))
            out["n_clusters"] = int(len([u for u in unique if u != -1]))
            out["has_noise"] = bool(-1 in unique)
            out["label_counts"] = {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        except Exception:
            pass

        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score

            # silhouette requires > 1 cluster and no all-same labels
            if len(set(labels)) > 1:
                out["silhouette"] = float(silhouette_score(X, labels))
            if len(set(labels)) > 1:
                out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            pass

        return out

    def _tune_supervised(
        self,
        *,
        req: "TrainingJobRequest",
        X_train: pd.DataFrame,
        y_train,
        task_type: str,
        mode: str,
        cv_folds: int,
        n_iter: int,
        _progress: Optional[Callable[[int], None]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Very small, safe tuner using sklearn Grid/Random search.

        We generate a compact search space from known registry params.
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

        reg = self.ml_model.list_registry()
        card = reg.get(req.algorithm_key, {}).get("card")
        param_space: Dict[str, Any] = {}

        # Build a small search space around defaults
        for p in getattr(card, "params", []) or []:
            if p.kind == "int" and p.min_value is not None and p.max_value is not None:
                lo = int(p.min_value)
                hi = int(p.max_value)
                default = int(p.default) if p.default is not None else lo
                candidates = sorted(set([default, max(lo, default // 2), min(hi, default * 2)]))
                param_space[p.name] = candidates
            elif p.kind == "float" and p.min_value is not None and p.max_value is not None:
                lo = float(p.min_value)
                hi = float(p.max_value)
                default = float(p.default) if p.default is not None else lo
                candidates = sorted(set([default, max(lo, default / 2), min(hi, default * 2)]))
                param_space[p.name] = candidates
            elif p.kind == "choice" and p.choices:
                param_space[p.name] = list(p.choices)

        if not param_space:
            return ({}, 0.0)

        # Create estimator
        model_name = f"tune_{req.algorithm_key}_{req.run_id}"
        self.ml_model.create_model_from_registry(model_name, req.algorithm_key, dict(req.params or {}))
        wrapper = self.ml_model.get_model(model_name)
        if wrapper is None:
            raise RuntimeError("Failed to create tuner model")

        if _progress:
            _progress(15)

        scoring = "r2" if task_type == "regression" else "accuracy"

        if mode == "grid":
            search = GridSearchCV(wrapper.model, param_space, cv=cv_folds)
            search.fit(X_train, y_train)
            best_params = dict(search.best_params_ or {})
            best_score = float(getattr(search, "best_score_", 0.0) or 0.0)
            if _progress:
                _progress(30)
            return (best_params, best_score)

        if mode == "random":
            search = RandomizedSearchCV(wrapper.model, param_space, n_iter=min(n_iter, 25), cv=cv_folds, random_state=42)
            search.fit(X_train, y_train)
            best_params = dict(search.best_params_ or {})
            best_score = float(getattr(search, "best_score_", 0.0) or 0.0)
            if _progress:
                _progress(30)
            return (best_params, best_score)

        if mode == "optuna":
            try:
                import optuna
            except Exception as exc:
                raise RuntimeError(f"Optuna not installed: {exc}")

            # Convert compact param_space into optuna suggestions
            def objective(trial: "optuna.Trial") -> float:
                params: Dict[str, Any] = {}
                for k, values in param_space.items():
                    if not isinstance(values, list) or not values:
                        continue
                    # If all numeric -> suggest within bounds; else categorical
                    all_int = all(isinstance(v, int) for v in values)
                    all_num = all(isinstance(v, (int, float)) for v in values)
                    if all_int:
                        params[k] = trial.suggest_int(k, int(min(values)), int(max(values)))
                    elif all_num:
                        params[k] = trial.suggest_float(k, float(min(values)), float(max(values)))
                    else:
                        params[k] = trial.suggest_categorical(k, values)

                est = self.ml_model._registry[req.algorithm_key].factory({**dict(req.params or {}), **params})
                scores = cross_val_score(est, X_train, y_train, cv=cv_folds, scoring=scoring)
                return float(scores.mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=min(n_iter, 50))
            if _progress:
                _progress(30)
            return (dict(study.best_params or {}), float(study.best_value or 0.0))

        if mode == "hyperopt":
            try:
                from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
            except Exception as exc:
                raise RuntimeError(f"Hyperopt not installed: {exc}")

            space: Dict[str, Any] = {}
            for k, values in param_space.items():
                if not isinstance(values, list) or not values:
                    continue
                all_int = all(isinstance(v, int) for v in values)
                all_num = all(isinstance(v, (int, float)) for v in values)
                if all_int:
                    space[k] = hp.quniform(k, int(min(values)), int(max(values)), 1)
                elif all_num:
                    space[k] = hp.uniform(k, float(min(values)), float(max(values)))
                else:
                    space[k] = hp.choice(k, values)

            def objective(params: Dict[str, Any]) -> Dict[str, Any]:
                # hyperopt may return floats for quniform
                clean: Dict[str, Any] = {}
                for kk, vv in params.items():
                    if isinstance(space.get(kk), str):
                        clean[kk] = vv
                    else:
                        clean[kk] = int(vv) if isinstance(vv, float) and float(vv).is_integer() else vv

                est = self.ml_model._registry[req.algorithm_key].factory({**dict(req.params or {}), **clean})
                scores = cross_val_score(est, X_train, y_train, cv=cv_folds, scoring=scoring)
                score = float(scores.mean())
                return {"loss": -score, "status": STATUS_OK}

            trials = Trials()
            best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=min(n_iter, 50), trials=trials, rstate=None)
            # Map best back to usable params
            best_params: Dict[str, Any] = {}
            for k, values in param_space.items():
                if not isinstance(values, list) or not values:
                    continue
                if all(isinstance(v, (str, bool)) for v in values):
                    # hp.choice returns index in some cases; best could be index
                    try:
                        idx = int(best.get(k))
                        best_params[k] = values[idx]
                    except Exception:
                        best_params[k] = best.get(k)
                else:
                    v = best.get(k)
                    if isinstance(v, float) and float(v).is_integer():
                        best_params[k] = int(v)
                    else:
                        best_params[k] = v

            # best_score from trials
            try:
                best_score = float(-min([t["result"]["loss"] for t in trials.trials]))
            except Exception:
                best_score = 0.0
            if _progress:
                _progress(30)
            return (best_params, best_score)

        raise RuntimeError(f"Unknown tuning mode: {mode}")


@dataclass
class TrainingJobRequest:
    run_id: str
    algorithm_key: str
    params: Dict[str, Any]
    label: str
    config: Dict[str, Any]
