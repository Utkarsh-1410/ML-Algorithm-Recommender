"""Tab 1: Dataset Upload + Data Profiling dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from PySide6.QtCore import Qt, Signal, QThreadPool
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressDialog,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
)

from ..widgets.mpl_canvas import MatplotlibCanvas

from ...models import ProfilingModel
from ...models.profiling_model import ProfileCacheKey, ProfileResult
from ...utils import Worker
from ...utils import get_logger


SUPPORTED_EXT = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


class _DropArea(QFrame):
    files_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropArea")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)

        self.lbl = QLabel("Drop CSV/Excel/JSON/Parquet here")
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setObjectName("DropAreaLabel")

        lay = QVBoxLayout()
        lay.setContentsMargins(10, 18, 10, 18)
        lay.addWidget(self.lbl)
        self.setLayout(lay)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            return
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.files_dropped.emit(path)


class DataLoadingProfilingTab(QWidget):
    """Dataset upload and profiling UI.

    Emits:
    - load_path_requested(path): request to load a dataset (controller decides how).
    - dataframe_loaded(df, path): background-loaded DataFrame ready for controller injection.
    - target_changed(col): user-selected target column.
    """

    load_path_requested = Signal(str)
    dataframe_loaded = Signal(object, str)
    target_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = get_logger("views.data_loading")
        self._thread_pool = QThreadPool.globalInstance()
        # Keep QRunnable workers alive until they finish (prevents silent no-op on some PySide setups).
        self._active_workers: List[Worker] = []
        self._profiling = ProfilingModel()
        # Always enable plots.
        self._plots_enabled = True

        self._current_df: Optional[pd.DataFrame] = None
        self._current_path: Optional[str] = None
        self._profile_cache_key: Optional[ProfileCacheKey] = None

        # ---- Dataset upload container ----
        self.drop_area = _DropArea()
        self.btn_browse = QPushButton("Browse")
        self.lbl_path = QLabel("No dataset loaded")
        self.lbl_path.setObjectName("DatasetPath")

        upload_row = QHBoxLayout()
        upload_row.addWidget(self.drop_area, 1)
        upload_row.addWidget(self.btn_browse)

        meta_group = QGroupBox("Dataset Metadata")
        meta_form = QFormLayout()
        self.lbl_file_size = QLabel("-")
        self.lbl_rows = QLabel("-")
        self.lbl_cols = QLabel("-")
        self.lbl_dtypes = QLabel("-")
        meta_form.addRow("File size:", self.lbl_file_size)
        meta_form.addRow("Rows:", self.lbl_rows)
        meta_form.addRow("Columns:", self.lbl_cols)
        meta_form.addRow("Data types:", self.lbl_dtypes)
        meta_group.setLayout(meta_form)

        self.cmb_target = QComboBox()
        self.cmb_target.setObjectName("TargetSelector")
        self.cmb_target.currentTextChanged.connect(lambda t: self.target_changed.emit(t) if t else None)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target column:"))
        target_row.addWidget(self.cmb_target, 1)

        self.table_preview = QTableWidget(0, 0)
        self.table_preview.setObjectName("PreviewTable")
        self.table_preview.setSortingEnabled(True)

        upload_layout = QVBoxLayout()
        upload_layout.addWidget(QLabel("Dataset Upload"))
        upload_layout.addLayout(upload_row)
        upload_layout.addWidget(self.lbl_path)
        upload_layout.addLayout(target_row)
        upload_layout.addWidget(meta_group)
        upload_layout.addWidget(QLabel("Preview (first 50 rows)"))
        upload_layout.addWidget(self.table_preview, 1)

        upload_container = QWidget()
        upload_container.setLayout(upload_layout)

        # ---- Profiling dashboard ----
        self.cards_group = QGroupBox("Overview")
        cards = QHBoxLayout()
        self.card_quality = self._make_card("Quality Score", "-")
        self.card_missing = self._make_card("Missing", "-")
        self.card_dupes = self._make_card("Duplicates", "-")
        self.card_mem = self._make_card("Memory", "-")
        for w in (self.card_quality, self.card_missing, self.card_dupes, self.card_mem):
            cards.addWidget(w)
        self.cards_group.setLayout(cards)

        self.group_columns = QGroupBox("Column-wise Analysis")
        col_top = QHBoxLayout()
        self.cmb_dtype_filter = QComboBox()
        self.cmb_dtype_filter.addItems(["All", "Numeric", "Categorical", "Datetime", "Other"])
        self.cmb_dtype_filter.currentTextChanged.connect(self._render_column_table)
        col_top.addWidget(QLabel("Filter:"))
        col_top.addWidget(self.cmb_dtype_filter)
        col_top.addStretch(1)

        self.table_columns = QTableWidget(0, 8)
        self.table_columns.setSortingEnabled(True)
        self.table_columns.setHorizontalHeaderLabels([
            "Name",
            "Type",
            "Unique",
            "Missing%",
            "Mean",
            "Median",
            "Min",
            "Max",
        ])
        self.table_columns.itemSelectionChanged.connect(self._on_column_selected)

        col_layout = QVBoxLayout()
        col_layout.addLayout(col_top)
        col_layout.addWidget(self.table_columns, 1)
        self.group_columns.setLayout(col_layout)

        self.group_details = QGroupBox("Selected Column Details")
        self.txt_details = QTextEdit()
        self.txt_details.setReadOnly(True)
        det_layout = QVBoxLayout()
        det_layout.addWidget(self.txt_details)
        self.group_details.setLayout(det_layout)

        # Visualizations
        self.group_viz = QGroupBox("Visualizations")
        viz_controls = QHBoxLayout()
        self.cmb_plot_column = QComboBox()
        self.cmb_plot_column.currentTextChanged.connect(self._refresh_plots)
        viz_controls.addWidget(QLabel("Column:"))
        viz_controls.addWidget(self.cmb_plot_column, 1)

        self._plot = MatplotlibCanvas(width=8.0, height=5.0, dpi=110)
        self.fig = self._plot.figure
        self.canvas = self._plot.canvas
        viz_widget: QWidget = self._plot

        viz_layout = QVBoxLayout()
        viz_layout.addLayout(viz_controls)
        viz_layout.addWidget(viz_widget, 1)
        self.group_viz.setLayout(viz_layout)

        # Warnings
        self.group_warnings = QGroupBox("Quality Warnings")
        self.txt_warnings = QTextEdit()
        self.txt_warnings.setReadOnly(True)
        wlay = QVBoxLayout()
        wlay.addWidget(self.txt_warnings)
        self.group_warnings.setLayout(wlay)

        # Schema + inference
        self.group_schema = QGroupBox("Schema Validation & Auto-inference")
        self.txt_schema = QTextEdit()
        self.txt_schema.setReadOnly(True)
        slay = QVBoxLayout()
        slay.addWidget(self.txt_schema)
        self.group_schema.setLayout(slay)

        profiling_layout = QVBoxLayout()
        profiling_layout.addWidget(self.cards_group)
        profiling_layout.addWidget(self.group_columns, 2)
        profiling_layout.addWidget(self.group_details, 1)
        profiling_layout.addWidget(self.group_viz, 2)
        profiling_layout.addWidget(self.group_warnings, 1)
        profiling_layout.addWidget(self.group_schema, 1)

        profiling_container = QWidget()
        profiling_container.setLayout(profiling_layout)

        # ---- 3-way layout (sub-tabs) to reduce clutter ----
        self.subtabs = QTabWidget()

        tab_upload = QWidget()
        tab_upload_l = QVBoxLayout()
        tab_upload_l.setContentsMargins(10, 10, 10, 10)
        tab_upload_l.addWidget(upload_container, 1)
        tab_upload.setLayout(tab_upload_l)

        tab_profile = QWidget()
        tab_profile_l = QVBoxLayout()
        tab_profile_l.setContentsMargins(10, 10, 10, 10)

        # Nested sub-tabs inside Profiling for better visibility of each section.
        profiling_tabs = QTabWidget()

        tab_overview = QWidget()
        tab_overview_l = QVBoxLayout()
        tab_overview_l.setContentsMargins(0, 0, 0, 0)
        tab_overview_l.addWidget(self.cards_group)
        tab_overview.setLayout(tab_overview_l)

        tab_columns = QWidget()
        tab_columns_l = QVBoxLayout()
        tab_columns_l.setContentsMargins(0, 0, 0, 0)
        col_split = QSplitter()
        col_split.setOrientation(Qt.Vertical)
        col_split.addWidget(self.group_columns)
        col_split.addWidget(self.group_details)
        col_split.setStretchFactor(0, 3)
        col_split.setStretchFactor(1, 2)
        tab_columns_l.addWidget(col_split, 1)
        tab_columns.setLayout(tab_columns_l)

        tab_viz = QWidget()
        tab_viz_l = QVBoxLayout()
        tab_viz_l.setContentsMargins(0, 0, 0, 0)
        tab_viz_l.addWidget(self.group_viz, 1)
        tab_viz.setLayout(tab_viz_l)

        profiling_tabs.addTab(tab_overview, "Overview")
        profiling_tabs.addTab(tab_columns, "Columns")
        profiling_tabs.addTab(tab_viz, "Visualizations")

        tab_profile_l.addWidget(profiling_tabs, 1)
        tab_profile.setLayout(tab_profile_l)

        tab_quality = QWidget()
        tab_quality_l = QVBoxLayout()
        tab_quality_l.setContentsMargins(10, 10, 10, 10)
        tab_quality_l.addWidget(self.group_warnings, 2)
        tab_quality_l.addWidget(self.group_schema, 1)
        tab_quality.setLayout(tab_quality_l)

        self.subtabs.addTab(tab_upload, "Upload")
        self.subtabs.addTab(tab_profile, "Profiling")
        self.subtabs.addTab(tab_quality, "Quality")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.subtabs, 1)
        self.setLayout(layout)

        # Wiring
        self.btn_browse.clicked.connect(self._choose_file)
        self.drop_area.files_dropped.connect(self._handle_path)

        # Disabled until a dataset is loaded
        self._set_profiling_enabled(False)

    # ---- Public API (used by controllers) ----
    def set_metadata(self, rows: int, cols: int, memory_mb: float) -> None:
        """Update dataset metadata section (best-effort).

        Controllers call this after loading/applying preprocessing.
        """
        try:
            self.lbl_rows.setText(str(int(rows)))
        except Exception:
            pass
        try:
            self.lbl_cols.setText(str(int(cols)))
        except Exception:
            pass
        try:
            self._set_card_value(self.card_mem, f"{float(memory_mb):.2f} MB")
        except Exception:
            pass

        # If we have an in-memory DF, refresh dtype preview.
        df = self._current_df
        if df is not None:
            try:
                dtypes = ", ".join([f"{c}:{str(df[c].dtype)}" for c in df.columns[:6]])
                if len(df.columns) > 6:
                    dtypes += ", …"
                self.lbl_dtypes.setText(dtypes)
            except Exception:
                pass

    def set_preview(self, headers: List[str], values: List[List[str]]) -> None:
        """Set the preview table with preformatted values from controller."""
        try:
            headers = [str(h) for h in (headers or [])]
            values = values or []

            self.table_preview.clear()
            self.table_preview.setRowCount(len(values))
            self.table_preview.setColumnCount(len(headers))
            self.table_preview.setHorizontalHeaderLabels(headers)

            for r, row in enumerate(values):
                for c, v in enumerate(list(row or [])[: len(headers)]):
                    self.table_preview.setItem(r, c, QTableWidgetItem(str(v)))

            self.table_preview.resizeColumnsToContents()
        except Exception:
            # Preview is non-critical; ignore to keep UI usable.
            return

    # ---- UI helpers ----
    def _make_card(self, title: str, value: str) -> QWidget:
        w = QFrame()
        w.setObjectName("OverviewCard")
        v = QVBoxLayout()
        v.setContentsMargins(10, 10, 10, 10)
        lbl_t = QLabel(title)
        lbl_t.setObjectName("CardTitle")
        lbl_v = QLabel(value)
        lbl_v.setObjectName("CardValue")
        v.addWidget(lbl_t)
        v.addWidget(lbl_v)
        w.setLayout(v)
        w._value_label = lbl_v  # type: ignore[attr-defined]
        return w

    def _set_card_value(self, card: QWidget, text: str) -> None:
        lbl = getattr(card, "_value_label", None)
        if lbl:
            lbl.setText(text)

    def _set_profiling_enabled(self, enabled: bool) -> None:
        for g in (self.cards_group, self.group_columns, self.group_details, self.group_viz, self.group_warnings, self.group_schema):
            g.setEnabled(enabled)

    # ---- file selection / loading ----
    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select dataset",
            "",
            "Data Files (*.csv *.xlsx *.xls *.json *.parquet);;All Files (*.*)",
        )
        if path:
            self._handle_path(path)

    def _handle_path(self, path: str) -> None:
        path = str(path)
        self.lbl_path.setText(path)
        try:
            self._log.info("Dataset selected: %s", path)
        except Exception:
            pass
        ok, msg = self._validate_file(path)
        if not ok:
            self.lbl_path.setText(f"Invalid file: {msg}")
            try:
                self._log.warning("Invalid dataset path: %s (%s)", path, msg)
            except Exception:
                pass
            return

        # Load in background to keep UI responsive
        self._load_in_background(path)

    def _validate_file(self, path: str) -> Tuple[bool, str]:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return False, "File not found"
        if p.suffix.lower() not in SUPPORTED_EXT:
            return False, f"Unsupported extension {p.suffix}"
        try:
            if p.stat().st_size <= 0:
                return False, "File is empty"
        except Exception:
            return False, "Cannot stat file"
        return True, "OK"

    def _load_in_background(self, path: str) -> None:
        dlg = QProgressDialog("Loading dataset…", "Cancel", 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()

        self.lbl_path.setText(f"Loading: {path}")

        def _load_file(file_path: str) -> Tuple[pd.DataFrame, str, int]:
            ext = Path(file_path).suffix.lower()
            try:
                if ext == ".csv":
                    df = pd.read_csv(file_path)
                elif ext in (".xlsx", ".xls"):
                    # Pandas requires an engine backend for Excel.
                    if ext == ".xlsx":
                        try:
                            import openpyxl  # type: ignore  # noqa: F401
                        except Exception as exc:
                            raise ImportError(
                                "Reading .xlsx requires 'openpyxl'. Install it with: pip install openpyxl"
                            ) from exc
                        df = pd.read_excel(file_path, engine="openpyxl")
                    else:
                        # .xls typically needs xlrd (older Excel format)
                        try:
                            import xlrd  # type: ignore  # noqa: F401
                        except Exception as exc:
                            raise ImportError(
                                "Reading .xls requires 'xlrd'. Install it with: pip install xlrd"
                            ) from exc
                        df = pd.read_excel(file_path, engine="xlrd")
                elif ext == ".json":
                    df = pd.read_json(file_path)
                elif ext == ".parquet":
                    try:
                        df = pd.read_parquet(file_path)
                    except Exception as exc:
                        msg = str(exc)
                        if "pyarrow" in msg.lower() or "fastparquet" in msg.lower() or "engine" in msg.lower():
                            raise ImportError(
                                "Reading .parquet requires 'pyarrow' (recommended) or 'fastparquet'. "
                                "Install one with: pip install pyarrow"
                            ) from exc
                        raise
                else:
                    raise ValueError(f"Unsupported extension: {ext}")
            except Exception as exc:
                # Raise a concise error so Worker emits a readable message.
                raise RuntimeError(str(exc)) from exc

            size = Path(file_path).stat().st_size
            return df, file_path, size

        worker = Worker(_load_file, path)
        worker.signals.result.connect(lambda res: self._on_loaded(res, dlg))
        worker.signals.error.connect(lambda err: self._on_load_error(err, dlg))
        worker.signals.finished.connect(dlg.close)

        # Retain the worker until it finishes so signals reliably fire.
        self._active_workers.append(worker)
        worker.signals.finished.connect(lambda w=worker: self._active_workers.remove(w) if w in self._active_workers else None)

        self._thread_pool.start(worker)

    def _on_load_error(self, err: str, dlg: QProgressDialog) -> None:
        self.lbl_path.setText(f"Load error: {err}")
        try:
            self._log.error("Dataset load failed: %s", err)
        except Exception:
            pass
        try:
            QMessageBox.critical(self, "ARCSaathi Error", f"Failed to load dataset.\n\n{err}")
        except Exception:
            pass
        dlg.close()

    def _on_loaded(self, res: Tuple[pd.DataFrame, str, int], dlg: QProgressDialog) -> None:
        df, path, size = res
        try:
            self._log.info("Dataset loaded in background: %s (%d rows, %d cols)", path, int(df.shape[0]), int(df.shape[1]))
        except Exception:
            pass
        self._current_df = df
        self._current_path = path

        # Cache key for profiling
        try:
            st = Path(path).stat()
            self._profile_cache_key = ProfileCacheKey(source_id=f"{path}|{st.st_mtime_ns}|{st.st_size}")
        except Exception:
            self._profile_cache_key = ProfileCacheKey(source_id=str(path))

        self._update_metadata(size, df)
        self._update_target_selector(df)
        self._set_preview(df)

        # Emit to controller so app state/models are updated
        self.dataframe_loaded.emit(df, path)

        # Start profiling
        self._run_profiling_background()

        dlg.close()

    def _update_metadata(self, file_size: int, df: pd.DataFrame) -> None:
        self.lbl_file_size.setText(f"{file_size / (1024 ** 2):.2f} MB")
        self.lbl_rows.setText(str(len(df)))
        self.lbl_cols.setText(str(len(df.columns)))
        # display compact dtype list
        dtypes = ", ".join([f"{c}:{str(df[c].dtype)}" for c in df.columns[:6]])
        if len(df.columns) > 6:
            dtypes += ", …"
        self.lbl_dtypes.setText(dtypes)

    def _update_target_selector(self, df: pd.DataFrame) -> None:
        self.cmb_target.blockSignals(True)
        self.cmb_target.clear()
        self.cmb_target.addItems([str(c) for c in df.columns])
        suggested = self._profiling.suggest_target(df)
        if suggested and suggested in df.columns:
            self.cmb_target.setCurrentText(str(suggested))
        else:
            self.cmb_target.setCurrentIndex(max(0, len(df.columns) - 1))
        self.cmb_target.blockSignals(False)
        # emit initial
        if self.cmb_target.currentText():
            self.target_changed.emit(self.cmb_target.currentText())

    def _set_preview(self, df: pd.DataFrame) -> None:
        sample = df.head(50)
        headers = [str(c) for c in sample.columns]
        self.table_preview.clear()
        self.table_preview.setRowCount(len(sample))
        self.table_preview.setColumnCount(len(headers))
        self.table_preview.setHorizontalHeaderLabels(headers)

        for r in range(len(sample)):
            row = sample.iloc[r]
            for c, h in enumerate(headers):
                val = row.iloc[c]
                self.table_preview.setItem(r, c, QTableWidgetItem(str(val)))

        self.table_preview.resizeColumnsToContents()

    # ---- profiling ----
    def _run_profiling_background(self) -> None:
        if self._current_df is None:
            return

        self._set_profiling_enabled(False)
        # Profiling can take a while on large datasets. Keep it non-blocking so the
        # user can continue using the app (other tabs / actions).
        dlg = QProgressDialog("Profiling dataset… (you can continue using the app)", "Hide", 0, 0, self)
        dlg.setWindowModality(Qt.NonModal)
        dlg.setMinimumDuration(0)
        dlg.canceled.connect(dlg.close)
        dlg.show()

        df = self._current_df
        target = self.cmb_target.currentText() or None
        cache_key = self._profile_cache_key

        def _profile_task() -> ProfileResult:
            # Use a sampled DF for very large datasets to keep profiling responsive.
            # The full dataset is still used elsewhere in the pipeline.
            try:
                rows = int(len(df))
            except Exception:
                rows = 0

            profile_df = df
            if rows > 100_000:
                try:
                    profile_df = df.sample(n=min(rows, 50_000), random_state=42)
                except Exception:
                    profile_df = df

            return self._profiling.profile(profile_df, target=target, cache_key=cache_key)

        worker = Worker(_profile_task)
        worker.signals.result.connect(lambda prof: self._apply_profile(prof, dlg))
        worker.signals.error.connect(lambda err: self._on_profile_error(err, dlg))
        worker.signals.finished.connect(dlg.close)

        # Retain the worker until it finishes so signals reliably fire.
        self._active_workers.append(worker)
        worker.signals.finished.connect(lambda w=worker: self._active_workers.remove(w) if w in self._active_workers else None)
        self._thread_pool.start(worker)

    def _on_profile_error(self, err: str, dlg: QProgressDialog) -> None:
        self.txt_warnings.setPlainText(f"Profiling error: {err}")
        self._set_profiling_enabled(True)
        dlg.close()

    def _apply_profile(self, prof: ProfileResult, dlg: QProgressDialog) -> None:
        try:
            ov = prof.overview
            self._set_card_value(self.card_quality, f"{ov['quality_score']:.0f}/100")
            self._set_card_value(self.card_missing, f"{ov['missing_count']} ({ov['missing_pct']:.1f}%)")
            self._set_card_value(self.card_dupes, str(ov['duplicates']))
            self._set_card_value(self.card_mem, f"{ov['memory_mb']:.2f} MB")

            # Color-code quality card
            score = float(ov.get("quality_score", 0.0))
            self.card_quality.setProperty("quality", "good" if score >= 80 else "warn" if score >= 50 else "bad")
            self.card_quality.style().unpolish(self.card_quality)
            self.card_quality.style().polish(self.card_quality)

            self._profile_columns = prof.columns
            self._render_column_table()

            # Plot column selector
            self.cmb_plot_column.blockSignals(True)
            self.cmb_plot_column.clear()
            if self._current_df is not None:
                self.cmb_plot_column.addItems([str(c) for c in self._current_df.columns])
            self.cmb_plot_column.blockSignals(False)
            if self.cmb_plot_column.count() > 0:
                self.cmb_plot_column.setCurrentIndex(0)
            try:
                self._refresh_plots()
            except Exception as exc:
                self.txt_warnings.setPlainText(f"Plot error: {exc}")

            self.txt_warnings.setPlainText("\n".join(prof.warnings) if prof.warnings else "No warnings detected.")

            schema = prof.schema
            inference = prof.inference
            schema_text = [
                f"Potential ID columns: {', '.join(schema.get('potential_id_columns', [])) or '-'}",
                f"Constant columns: {', '.join(schema.get('constant_columns', [])) or '-'}",
                f"Inconsistent types: {', '.join(schema.get('inconsistent_type_columns', [])) or '-'}",
                f"Datetime-like: {', '.join(schema.get('datetime_like_columns', [])) or '-'}",
                "",
                f"Problem type: {inference.get('problem_type', '-')}",
                f"Suggested target: {inference.get('suggested_target', '-')}",
                "Recommended preprocessing (high-level):",
            ]
            for step in inference.get("recommended_preprocessing", []):
                schema_text.append(f"- {step.get('operation')} {step.get('params', {})}")
            self.txt_schema.setPlainText("\n".join(schema_text))
        except Exception as exc:
            # Never allow profiling UI update to crash the app.
            self.txt_warnings.setPlainText(f"Profiling display error: {exc}")
        finally:
            self._set_profiling_enabled(True)
            dlg.close()

    def _render_column_table(self) -> None:
        cols = getattr(self, "_profile_columns", [])
        df = self._current_df
        if df is None:
            return

        mode = self.cmb_dtype_filter.currentText()
        filtered: List[Dict[str, Any]] = []
        for c in cols:
            dtype = str(c.get("dtype", ""))
            is_num = "int" in dtype or "float" in dtype
            is_cat = "object" in dtype or "category" in dtype
            is_dt = "datetime" in dtype

            if mode == "Numeric" and not is_num:
                continue
            if mode == "Categorical" and not is_cat:
                continue
            if mode == "Datetime" and not is_dt:
                continue
            if mode == "Other" and (is_num or is_cat or is_dt):
                continue

            filtered.append(c)

        self.table_columns.setRowCount(len(filtered))
        for r, c in enumerate(filtered):
            vals = [
                c.get("name", ""),
                c.get("dtype", ""),
                c.get("unique", ""),
                f"{float(c.get('missing_pct', 0.0)):.2f}",
                self._fmt_num(c.get("mean")),
                self._fmt_num(c.get("median")),
                self._fmt_num(c.get("min")),
                self._fmt_num(c.get("max")),
            ]
            for col_idx, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                self.table_columns.setItem(r, col_idx, item)

        self.table_columns.resizeColumnsToContents()

    def _fmt_num(self, v: Any) -> str:
        if v is None:
            return "-"
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    def _on_column_selected(self) -> None:
        df = self._current_df
        if df is None:
            return
        items = self.table_columns.selectedItems()
        if not items:
            return
        name = items[0].text()
        if name not in df.columns:
            return

        s = df[name]
        lines = [
            f"Column: {name}",
            f"Dtype: {s.dtype}",
            f"Unique: {s.nunique(dropna=True)}",
            f"Missing: {int(s.isna().sum())} ({float(s.isna().mean()) * 100:.2f}%)",
        ]
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce")
            lines += [
                f"Mean: {vals.mean():.4f}",
                f"Median: {vals.median():.4f}",
                f"Min: {vals.min():.4f}",
                f"Max: {vals.max():.4f}",
                f"Std: {vals.std():.4f}",
            ]
        else:
            top = s.astype(str).value_counts(dropna=True).head(10)
            lines.append("Top values:")
            for k, v in top.items():
                lines.append(f"- {k}: {v}")

        self.txt_details.setPlainText("\n".join(lines))

    def _refresh_plots(self) -> None:
        if not getattr(self, "_plots_enabled", False):
            return

        df = self._current_df
        if df is None or df.empty:
            return

        col = self.cmb_plot_column.currentText()
        if not col or col not in df.columns:
            return

        s = df[col]
        sample = df.sample(min(len(df), 5000), random_state=42) if len(df) > 5000 else df

        fig = getattr(self, "fig", None)
        canvas = getattr(self, "canvas", None)
        if fig is None or canvas is None:
            return

        fig.clear()
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        # Histogram or bar chart
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(sample[col], errors="coerce").dropna()
            ax1.hist(vals, bins=30, color="#1d4ed8")
            ax1.set_title(f"Histogram: {col}")
        else:
            vc = sample[col].astype(str).value_counts().head(15)
            ax1.bar(vc.index.tolist(), vc.values.tolist(), color="#1d4ed8")
            ax1.set_title(f"Top categories: {col}")
            ax1.tick_params(axis='x', labelrotation=45)

        # Correlation heatmap
        num = sample.select_dtypes(include=["number"]).dropna(axis=1, how="all")
        if num.shape[1] >= 2:
            corr = num.corr(numeric_only=True)
            im = ax2.imshow(corr.to_numpy(), aspect="auto")
            ax2.set_title("Correlation")
            ax2.set_xticks(range(len(corr.columns)))
            ax2.set_yticks(range(len(corr.columns)))
            ax2.set_xticklabels([str(c) for c in corr.columns], rotation=90, fontsize=7)
            ax2.set_yticklabels([str(c) for c in corr.columns], fontsize=7)
        else:
            ax2.text(0.5, 0.5, "Correlation needs ≥2 numeric columns", ha="center", va="center")
            ax2.set_axis_off()

        # Missing values matrix (sample)
        miss = sample.isna().astype(int)
        ax3.imshow(miss.to_numpy(), aspect="auto", cmap="gray_r")
        ax3.set_title("Missing values matrix")
        ax3.set_xlabel("Columns")
        ax3.set_ylabel("Rows")

        # Basic overview plot: missing% by column
        miss_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
        ax4.bar(miss_pct.index.astype(str), miss_pct.values, color="#1d4ed8")
        ax4.set_title("Top missing% columns")
        ax4.tick_params(axis='x', labelrotation=45)

        try:
            # tight_layout can occasionally fail with "Singular matrix" depending on
            # renderer state / extreme axis sizes; treat it as non-fatal.
            fig.tight_layout()
        except Exception:
            pass
        try:
            canvas.draw_idle()
        except Exception:
            pass
