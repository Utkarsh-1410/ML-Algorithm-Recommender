"""Simple QRunnable-based worker for background tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .logger import get_logger


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    """Run a callable in a background thread and emit signals."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_progress: Optional[Callable[[int], None]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._on_progress = on_progress

    @Slot()
    def run(self) -> None:
        try:
            if self._on_progress is not None:
                self.kwargs["_progress"] = self._on_progress
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as exc:
            # Preserve traceback for diagnosis (common source of "nothing works").
            tb = traceback.format_exc()
            try:
                get_logger("utils.worker").exception("Worker task failed: %s", exc)
            except Exception:
                pass
            self.signals.error.emit(f"{exc}\n\n{tb}")
        finally:
            self.signals.finished.emit()
