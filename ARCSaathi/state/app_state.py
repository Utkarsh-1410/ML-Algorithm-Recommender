"""Persistent application state manager."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal


@dataclass
class AppStateSnapshot:
    current_dataset_path: Optional[str] = None
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)

    preprocessing_steps: List[Dict[str, Any]] = field(default_factory=list)

    trained_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    evaluation_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    user_preferences: Dict[str, Any] = field(default_factory=dict)


class AppState(QObject):
    """Tracks and persists cross-tab application state."""

    state_changed = Signal(str)  # section name
    error_occurred = Signal(str)

    def __init__(self, state_path: Optional[str] = None):
        super().__init__()
        self._state_path = Path(state_path) if state_path else self._default_state_path()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        self.snapshot = AppStateSnapshot()
        self.load()

    def _default_state_path(self) -> Path:
        return Path.home() / ".arcsaathi" / "state.json"

    def load(self) -> bool:
        if not self._state_path.exists():
            return True
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.snapshot = AppStateSnapshot(**data)
            self.state_changed.emit("*")
            return True
        except Exception as exc:
            self.error_occurred.emit(f"Failed to load state: {exc}")
            return False

    def save(self) -> bool:
        try:
            payload = _to_jsonable(asdict(self.snapshot))
            self._state_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return True
        except Exception as exc:
            self.error_occurred.emit(f"Failed to save state: {exc}")
            return False

    def set_dataset(self, dataset_path: str, metadata: Dict[str, Any]) -> None:
        self.snapshot.current_dataset_path = dataset_path
        self.snapshot.dataset_metadata = dict(metadata or {})
        self.state_changed.emit("dataset")

    def set_preprocessing_steps(self, steps: List[Dict[str, Any]]) -> None:
        self.snapshot.preprocessing_steps = list(steps or [])
        self.state_changed.emit("preprocessing")

    def upsert_trained_model(self, name: str, model_info: Dict[str, Any]) -> None:
        self.snapshot.trained_models[str(name)] = dict(model_info or {})
        self.state_changed.emit("models")

    def upsert_evaluation_result(self, model_name: str, result: Dict[str, Any]) -> None:
        self.snapshot.evaluation_results[str(model_name)] = dict(result or {})
        self.state_changed.emit("evaluation")

    def set_user_preference(self, key: str, value: Any) -> None:
        self.snapshot.user_preferences[str(key)] = value
        self.state_changed.emit("preferences")


def _to_jsonable(obj: Any) -> Any:
    """Convert common numpy/pandas objects into JSON-serializable primitives.

    State may contain values like numpy scalars, pandas dtypes (e.g. Int64Dtype),
    timestamps, etc. This keeps persistence robust.
    """

    # Fast-path primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Mapping
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            # JSON object keys must be strings
            try:
                key = str(k)
            except Exception:
                key = "<key>"
            out[key] = _to_jsonable(v)
        return out

    # Sequences
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # pathlib
    if isinstance(obj, Path):
        return str(obj)

    # Numpy / pandas (optional)
    try:
        import numpy as _np  # type: ignore

        if isinstance(obj, _np.generic):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        if isinstance(obj, _np.ndarray):
            try:
                return obj.tolist()
            except Exception:
                return str(obj)
    except Exception:
        pass

    try:
        import pandas as _pd  # type: ignore

        # pandas scalar NA
        try:
            if obj is _pd.NA:  # type: ignore[attr-defined]
                return None
        except Exception:
            pass

        # Timestamps / timedeltas
        if isinstance(obj, getattr(_pd, "Timestamp", ())):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
        if isinstance(obj, getattr(_pd, "Timedelta", ())):
            return str(obj)

        # Dtype objects (e.g., Int64Dtype)
        if isinstance(obj, getattr(_pd, "api", object).types.pandas_dtype("int64").__class__):
            return str(obj)
        # Fallback: many pandas dtypes implement __str__ nicely
        mod = getattr(obj, "__module__", "")
        if mod.startswith("pandas"):
            return str(obj)
    except Exception:
        pass

    # Last-resort: string representation
    return str(obj)
