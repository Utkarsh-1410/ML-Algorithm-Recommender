"""Main controller: coordinates models, views, workflow modules, state, and settings."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QTableWidgetItem

from ..config import SettingsManager
from ..models import DataModel, PreprocessingModel, MLModel, EvaluationModel, TaskDetectionModel
from ..state import AppState
from ..utils import get_logger, show_error, show_info
from ..resources.themes import stylesheet_for
from ..views.main_window import MainWindow

from .data_processing_controller import DataProcessingController
from .model_training_controller import ModelTrainingController
from .evaluation_controller import EvaluationComparisonController
from .recommender_controller import ModelRecommenderController
from .explainability_controller import ExplainabilityController


class MainController(QObject):
    """Wires UI events to workflow controllers and manages application lifetime."""

    def __init__(
        self,
        window: MainWindow,
        settings: SettingsManager,
        state: AppState,
        data_model: DataModel,
        preprocessing_model: PreprocessingModel,
        ml_model: MLModel,
        evaluation_model: EvaluationModel,
    ):
        super().__init__()

        self._log = get_logger("controllers.main")

        self.window = window
        self.settings = settings
        self.state = state

        # Workflow controllers
        self.data_processing = DataProcessingController(state, data_model, preprocessing_model)
        self.training = ModelTrainingController(state, data_model, ml_model)
        self.evaluation = EvaluationComparisonController(state, data_model, ml_model, evaluation_model)
        self.recommender = ModelRecommenderController(state, data_model, ml_model)
        self.explainability = ExplainabilityController(data_model, ml_model)
        self.task_detection = TaskDetectionModel()

        # One-time wiring for recommender output
        self.recommender.recommendation_ready.connect(self.window.tab_recommender.set_recommendation_result)

        # Window-level actions
        self.window.theme_toggle_requested.connect(self._toggle_theme)
        self.window.export_requested.connect(self._export_project)
        self.window.help_requested.connect(self._help)

        # Bubble up errors to UI
        for c in [self.data_processing, self.training, self.evaluation, self.recommender, self.explainability, self.state]:
            if hasattr(c, "error_occurred"):
                c.error_occurred.connect(lambda msg, parent=window: show_error(parent, "ARCSaathi Error", msg))

        self._wire_tabs()
        self._restore_state_to_ui()

        # Initial status
        self.window.set_app_status("Ready")

        # Prime explainability context
        try:
            self.explainability.refresh_context()
        except Exception:
            pass

    def _on_export_explainability_report(self, fmt: str) -> None:
        try:
            t = self.window.tab_explainability
            payload = {
                "global_text": t.txt_global.toPlainText(),
                "global_html": "",
                "local_text": t.txt_local.toPlainText(),
                "local_html": "",
                "whatif_text": t.txt_whatif.toPlainText(),
                "debug_text": t.txt_debug.toPlainText(),
                "drift_text": t.txt_drift.toPlainText(),
                "fairness_text": t.txt_fairness.toPlainText(),
                "auto_text": t.txt_auto.toPlainText(),
            }
        except Exception:
            payload = {}

        self.explainability.export_report(fmt, payload)

    def _wire_tabs(self) -> None:
        # Tab 1
        self.window.tab_data.dataframe_loaded.connect(self._on_loaded_dataframe)
        self.window.tab_data.target_changed.connect(self._on_target_changed)

        # Tab 2
        self.window.tab_preprocess.add_step_requested.connect(self._on_add_step)
        self.window.tab_preprocess.remove_step_requested.connect(self._on_remove_step)
        self.window.tab_preprocess.apply_pipeline_requested.connect(self._on_apply_pipeline)
        self.window.tab_preprocess.detect_task_requested.connect(self._on_detect_task)
        self.window.tab_preprocess.task_config_changed.connect(self._on_task_config_changed)
        self.window.tab_preprocess.validate_steps_requested.connect(self._on_validate_steps)
        self.window.tab_preprocess.preview_steps_requested.connect(self._on_preview_steps)
        self.window.tab_preprocess.steps_replaced_requested.connect(self._on_steps_replaced)
        self.window.tab_preprocess.clear_steps_requested.connect(self._on_clear_steps)
        self.window.tab_preprocess.smart_preprocess_requested.connect(self._on_smart_preprocess)

        # Tab 3
        self.window.tab_training.create_model_requested.connect(self._on_create_model)
        self.window.tab_training.train_model_requested.connect(self._on_train_model)

        # New Tab 3 batch training signals
        if hasattr(self.window.tab_training, "enqueue_jobs_requested"):
            self.window.tab_training.enqueue_jobs_requested.connect(self._on_enqueue_training_jobs)
        if hasattr(self.window.tab_training, "pause_queue_requested"):
            self.window.tab_training.pause_queue_requested.connect(lambda: self.training.pause_queue())
        if hasattr(self.window.tab_training, "resume_queue_requested"):
            self.window.tab_training.resume_queue_requested.connect(lambda: self.training.resume_queue())
        if hasattr(self.window.tab_training, "cancel_all_requested"):
            self.window.tab_training.cancel_all_requested.connect(lambda: self.training.cancel_all())
        if hasattr(self.window.tab_training, "parallelism_changed"):
            self.window.tab_training.parallelism_changed.connect(lambda n: self.training.set_parallelism(int(n)))
        if hasattr(self.window.tab_training, "refresh_requested"):
            self.window.tab_training.refresh_requested.connect(self._refresh_training_views)

        # Controller -> Tab 3 updates
        self.training.job_enqueued.connect(lambda run_id, model_name: self._on_job_enqueued(run_id, model_name))
        self.training.job_started.connect(lambda run_id, model_name: self._on_job_started(run_id, model_name))
        self.training.job_progress.connect(lambda run_id, pct: self.window.tab_training.upsert_queue_status(run_id, "running", int(pct)))
        self.training.job_finished.connect(self._on_training_job_finished)

        # Tab 4
        self.window.tab_results.compare_requested.connect(self._on_compare_models)
        if hasattr(self.window.tab_results, "export_table_requested"):
            self.window.tab_results.export_table_requested.connect(self._on_export_results_table)
        if hasattr(self.window.tab_results, "export_fig_requested"):
            self.window.tab_results.export_fig_requested.connect(self._on_export_results_fig)
        if hasattr(self.window.tab_results, "export_report_requested"):
            self.window.tab_results.export_report_requested.connect(self._on_export_results_report)

        # Tab 5
        self.window.tab_recommender.recommend_requested.connect(self._on_recommend)
        if hasattr(self.window.tab_recommender, "feedback_submitted"):
            self.window.tab_recommender.feedback_submitted.connect(lambda p: self.recommender.record_feedback(p))
        if hasattr(self.window.tab_recommender, "export_requested"):
            self.window.tab_recommender.export_requested.connect(self._on_export_recommendations)

        # Tab 6 (Explainability)
        if hasattr(self.window, "tab_explainability"):
            t = self.window.tab_explainability
            t.refresh_requested.connect(lambda: self.explainability.refresh_context())
            t.set_target_requested.connect(lambda target, task: self.explainability.set_target(target, task))

            t.run_global_requested.connect(lambda req: self.explainability.run_global(req))
            t.run_local_requested.connect(lambda req: self.explainability.run_local(req))
            t.run_whatif_requested.connect(lambda req: self.explainability.run_whatif(req))
            t.run_debug_requested.connect(lambda req: self.explainability.run_debug(req))
            t.run_drift_requested.connect(lambda req: self.explainability.run_drift(req))
            t.run_fairness_requested.connect(lambda req: self.explainability.run_fairness(req))
            t.run_auto_requested.connect(lambda req: self.explainability.run_auto(req))

            t.export_report_requested.connect(self._on_export_explainability_report)

            # Controller -> UI
            self.explainability.context_ready.connect(
                lambda ctx: t.set_context(models=ctx.get("models", []), target_candidates=ctx.get("targets", []), feature_names=ctx.get("features", []))
            )
            self.explainability.candidate_rows_ready.connect(lambda rows: t.set_candidate_rows(list(rows or [])))
            self.explainability.global_ready.connect(lambda out: t.set_global_output(text=out.get("text", ""), html=out.get("html", "")))
            self.explainability.local_ready.connect(lambda out: t.set_local_output(text=out.get("text", ""), html=out.get("html", "")))
            self.explainability.whatif_ready.connect(lambda out: t.set_whatif_output(out.get("text", "")))
            self.explainability.debug_ready.connect(lambda out: t.set_debug_output(out.get("text", "")))
            self.explainability.drift_ready.connect(lambda out: t.set_drift_output(out.get("text", "")))
            self.explainability.fairness_ready.connect(lambda out: t.set_fairness_output(out.get("text", ""), out.get("privileged_groups")))
            self.explainability.auto_ready.connect(lambda out: t.set_auto_output(out.get("text", "")))
            self.explainability.report_ready.connect(lambda path: show_info(self.window, "Export", f"Saved explainability report: {path}"))

    def _restore_state_to_ui(self) -> None:
        # restore preprocessing steps list
        if self.state.snapshot.preprocessing_steps:
            self.window.tab_preprocess.set_steps(self.state.snapshot.preprocessing_steps)

        # restore model list
        model_names = list(self.state.snapshot.trained_models.keys())
        if model_names:
            self.window.tab_training.set_models(model_names)

        # initialize registry + recent runs
        try:
            self.window.tab_training.set_registry(self.training.list_registry())
            self.window.tab_training.set_recent_runs(self.training.list_recent_runs(limit=100))
        except Exception:
            pass

    def _on_load_dataset(self, path: str) -> None:
        self.window.set_app_status("Processing")
        try:
            if self.data_processing.load_dataset(path):
                summary = self.data_processing.get_dataset_summary()
                self.window.tab_data.set_metadata(summary["rows"], summary["columns"], summary["memory_mb"])
                self.window.set_dataset_info(summary["rows"], summary["columns"])

                headers, values = self.data_processing.get_dataset_preview()
                self.window.tab_data.set_preview(headers, values)

                # update steps list from model
                self.window.tab_preprocess.set_steps(self.data_processing.preprocessing_model.get_steps())

                # Workflow progression: 1-3 complete on successful load + heuristic task detection
                self.window.set_step_completed(1, True)
                self.window.set_step_completed(2, True)
                self.window.set_step_completed(3, True)
                self.window.set_app_status("Ready")
                return

            self.window.set_app_status("Error")
        except Exception as exc:
            try:
                self._log.exception("Dataset load failed")
            except Exception:
                pass
            show_error(self.window, "ARCSaathi Error", f"Dataset load failed: {exc}")
            self.window.set_app_status("Error")

    def _on_loaded_dataframe(self, df, path: str) -> None:
        # Called when Tab 1 loads data in background.
        self.window.set_app_status("Processing")
        try:
            if self.data_processing.load_dataframe(df, path):
                summary = self.data_processing.get_dataset_summary()
                self.window.tab_data.set_metadata(summary["rows"], summary["columns"], summary["memory_mb"])
                self.window.set_dataset_info(summary["rows"], summary["columns"])

                headers, values = self.data_processing.get_dataset_preview()
                self.window.tab_data.set_preview(headers, values)

                # Feed dataset columns + missing% into preprocessing tab
                try:
                    missing = self.data_processing.get_missing_pct()
                    self.window.tab_preprocess.set_dataset_columns([str(c) for c in df.columns.tolist()], missing_pct=missing)
                except Exception:
                    pass

                self.window.set_step_completed(1, True)
                self.window.set_step_completed(2, True)
                self.window.set_step_completed(3, True)
                self.window.set_app_status("Ready")
                return

            self.window.set_app_status("Error")
        except Exception as exc:
            try:
                self._log.exception("Dataset inject handler failed")
            except Exception:
                pass
            show_error(self.window, "ARCSaathi Error", f"Dataset load failed: {exc}")
            self.window.set_app_status("Error")

    def _on_detect_task(self) -> None:
        df = self.data_processing.data_model.get_data()
        if df is None or df.empty:
            show_error(self.window, "ARCSaathi Error", "Load a dataset first")
            return

        res = self.task_detection.detect(df)
        self.window.tab_preprocess.set_task_detection(
            task_type=res.task_type,
            target=res.target,
            metrics=res.metrics or [],
            reasoning=res.reasoning or [],
            classification_type=res.classification_type,
        )

        # Propagate target downstream if applicable
        if res.target:
            self.training.set_target(res.target)
            try:
                self.window.tab_training.set_target(res.target)
            except Exception:
                pass
            self.evaluation.set_target(res.target, "classification" if res.task_type == "classification" else "regression")

    def _on_task_config_changed(self, task) -> None:
        # Best-effort propagation to training/evaluation.
        try:
            ttype = getattr(task, "task_type", "classification")
            target = getattr(task, "target", "")
            if target:
                self.training.set_target(target)
                self.evaluation.set_target(target, "classification" if ttype == "classification" else "regression")
                try:
                    self.window.tab_recommender.set_target(target)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_validate_steps(self, steps) -> None:
        ok, msg = self.data_processing.validate_steps(list(steps or []))
        self.window.tab_preprocess.set_validation_result(ok, msg)

    def _on_preview_steps(self, steps, upto_index: int) -> None:
        headers, rows, summary = self.data_processing.preview_steps(list(steps or []), upto_index=int(upto_index), n=12)
        self.window.tab_preprocess.set_preview(headers, rows, summary)

    def _on_steps_replaced(self, steps) -> None:
        self.data_processing.replace_preprocessing_steps(list(steps or []))
        self.window.tab_preprocess.set_steps(self.data_processing.preprocessing_model.get_steps())

    def _on_clear_steps(self) -> None:
        self.data_processing.clear_preprocessing_steps()
        self.window.tab_preprocess.set_steps([])

    def _on_smart_preprocess(self, task) -> None:
        task_dict = {
            "task_type": getattr(task, "task_type", "classification"),
            "target": getattr(task, "target", ""),
        }
        steps = self.data_processing.smart_preprocess(task_dict)
        self.data_processing.replace_preprocessing_steps(steps)
        self.window.tab_preprocess.set_steps(self.data_processing.preprocessing_model.get_steps())

    def _on_target_changed(self, target: str) -> None:
        # Store target for training/evaluation modules.
        if not target:
            return
        self.training.set_target(target)
        try:
            self.window.tab_training.set_target(target)
        except Exception:
            pass
        # default to classification; auto-inference can refine later
        self.evaluation.set_target(target, "classification")

    def _on_enqueue_training_jobs(self, jobs: list, config: dict) -> None:
        # Ensure Tab's parallelism is reflected
        try:
            self.training.enqueue_jobs(list(jobs or []), dict(config or {}))
            self.window.tab_training.set_recent_runs(self.training.list_recent_runs(limit=100))
        except Exception as exc:
            show_error(self.window, "ARCSaathi Error", f"Failed to enqueue jobs: {exc}")

    def _on_training_job_finished(self, run_id: str, status: str, payload: dict) -> None:
        try:
            self.window.tab_training.upsert_queue_status(run_id, status, 100 if status == "finished" else None)
            self.window.tab_training.set_recent_runs(self.training.list_recent_runs(limit=100))
            if status == "finished":
                self.window.set_step_completed(5, True)
        except Exception:
            pass

    def _refresh_training_views(self) -> None:
        try:
            self.window.tab_training.set_registry(self.training.list_registry())
            self.window.tab_training.set_recent_runs(self.training.list_recent_runs(limit=100))
        except Exception:
            pass

    def _on_job_enqueued(self, run_id: str, model_name: str) -> None:
        try:
            self.window.tab_training.upsert_queue_status(run_id, "queued", 0)
            # best-effort set model name
            for r in range(self.window.tab_training.tbl_queue.rowCount()):
                it = self.window.tab_training.tbl_queue.item(r, 0)
                if it and it.text() == str(run_id):
                    self.window.tab_training.tbl_queue.setItem(r, 1, QTableWidgetItem(str(model_name)))
                    break
        except Exception:
            pass

    def _on_job_started(self, run_id: str, model_name: str) -> None:
        try:
            self.window.tab_training.upsert_queue_status(run_id, "running", 0)
            for r in range(self.window.tab_training.tbl_queue.rowCount()):
                it = self.window.tab_training.tbl_queue.item(r, 0)
                if it and it.text() == str(run_id):
                    self.window.tab_training.tbl_queue.setItem(r, 1, QTableWidgetItem(str(model_name)))
                    break
        except Exception:
            pass

    def _on_add_step(self, operation: str, params: dict) -> None:
        if self.data_processing.add_preprocessing_step(operation, params):
            self.window.tab_preprocess.set_steps(self.data_processing.preprocessing_model.get_steps())

    def _on_remove_step(self, index: int) -> None:
        if self.data_processing.remove_preprocessing_step(index):
            self.window.tab_preprocess.set_steps(self.data_processing.preprocessing_model.get_steps())

    def _on_apply_pipeline(self) -> None:
        self.window.set_app_status("Processing")
        processed = self.data_processing.apply_pipeline()
        if processed is not None:
            summary = self.data_processing.get_dataset_summary()
            self.window.tab_data.set_metadata(summary["rows"], summary["columns"], summary["memory_mb"])
            self.window.set_dataset_info(summary["rows"], summary["columns"])
            headers, values = self.data_processing.get_dataset_preview()
            self.window.tab_data.set_preview(headers, values)

            # Step 4 complete
            self.window.set_step_completed(4, True)
            self.window.set_app_status("Ready")
            return

        self.window.set_app_status("Error")

    def _on_create_model(self, name: str, model_type: str, params: dict) -> None:
        if self.training.create_model(name, model_type, params):
            self.window.tab_training.set_models(self.training.list_models())

    def _on_train_model(self, name: str) -> None:
        # Minimal default: infer target from last column (user can improve later)
        df = self.data_processing.data_model.get_data()
        if df is None or df.empty:
            show_error(self.window, "ARCSaathi Error", "Load a dataset first")
            return
        self.training.set_target(df.columns[-1])
        self.evaluation.set_target(df.columns[-1], "classification")

        if self.training.train(name):
            self.window.tab_training.set_models(self.training.list_models())
            self.window.set_step_completed(5, True)
            self.window.set_app_status("Ready")
            return

        self.window.set_app_status("Error")

    def _on_compare_models(self, task_type: str = "classification") -> None:
        # Ensure evaluation exists
        df = self.data_processing.data_model.get_data()
        if df is None or df.empty:
            show_error(self.window, "ARCSaathi Error", "Load a dataset first")
            return

        task = str(task_type or "classification").lower()
        target = getattr(self.training, "target_column", None) or ""
        if task in ("classification", "regression"):
            if not target:
                target = str(df.columns[-1])
            self.evaluation.set_target(target, task)
        else:
            # Unsupervised tasks ignore target
            self.evaluation.set_target("", task)

        self.window.set_app_status("Processing")
        if self.evaluation.evaluate_all():
            payload = self.evaluation.build_dashboard_payload()
            self.window.tab_results.set_dashboard_payload(payload)
            self.window.set_step_completed(6, True)
            self.window.set_app_status("Ready")
            return

        self.window.set_app_status("Error")

    def _on_export_results_table(self, fmt: str) -> None:
        payload = {}
        try:
            payload = self.window.tab_results.get_payload()
        except Exception:
            payload = {}
        path = self.evaluation.export_table(payload, fmt)
        if path:
            show_info(self.window, "Export", f"Saved table: {path}")

    def _on_export_results_fig(self, fmt: str) -> None:
        try:
            paths = self.window.tab_results.export_figs(fmt)
            if paths:
                show_info(self.window, "Export", f"Saved figures: {paths[0]}")
        except Exception as exc:
            show_error(self.window, "Export", f"Failed to export figures: {exc}")

    def _on_export_results_report(self, fmt: str) -> None:
        payload = {}
        try:
            payload = self.window.tab_results.get_payload()
        except Exception:
            payload = {}
        path = self.evaluation.export_report(payload, fmt)
        if path:
            show_info(self.window, "Export", f"Saved report: {path}")

    def _on_recommend(self, request: Optional[dict] = None) -> None:
        self.recommender.recommend(request or {"task_type": "classification", "profile": "Best Fit"})

    def _on_export_recommendations(self, fmt: str, payload: dict) -> None:
        path = self.recommender.export_report(fmt, payload)
        if path:
            show_info(self.window, "Export", f"Saved recommendations report: {path}")

    def _toggle_theme(self) -> None:
        current = self.settings.get("ui.theme", "light")
        new_theme = "dark" if str(current).lower() == "light" else "light"
        self.settings.set("ui.theme", new_theme)
        self.settings.save_settings()
        # Apply immediately
        try:
            from PySide6.QtWidgets import QApplication

            QApplication.instance().setStyleSheet(stylesheet_for(new_theme))
        except Exception:
            pass

        show_info(self.window, "Theme", f"Switched to {new_theme} theme")

    def _export_project(self) -> None:
        # Minimal export hook: persist state and acknowledge.
        self.state.save()
        show_info(self.window, "Export Project", "Project state saved. Add exporters in controllers as needed.")

    def _help(self, context: str) -> None:
        show_info(self.window, "Help/Docs", f"Help requested for: {context}")
