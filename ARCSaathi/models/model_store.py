"""SQLite-backed store for trained models and results."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_root() -> Path:
    root = Path.home() / ".arcsaathi"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class TrainingRun:
    run_id: str
    created_at: float
    task_type: str
    algorithm_key: str
    model_name: str
    status: str
    metrics_json: str
    params_json: str
    artifact_path: Optional[str] = None


class ModelStore:
    def __init__(self, db_path: Optional[str] = None):
        root = _default_root()
        self.db_path = Path(db_path) if db_path else (root / "models.db")
        self.artifacts_dir = root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at REAL,
                    task_type TEXT,
                    algorithm_key TEXT,
                    model_name TEXT,
                    status TEXT,
                    metrics_json TEXT,
                    params_json TEXT,
                    artifact_path TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommender_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL,
                    user_id TEXT,
                    dataset_sig TEXT,
                    task_type TEXT,
                    profile TEXT,
                    algorithm_key TEXT,
                    model_name TEXT,
                    score_total INTEGER,
                    accepted INTEGER,
                    payload_json TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommender_prefs (
                    user_id TEXT PRIMARY KEY,
                    updated_at REAL,
                    prefs_json TEXT
                )
                """
            )
            conn.commit()

    def new_run_id(self) -> str:
        return f"run_{int(time.time() * 1000)}"

    def save_run(
        self,
        *,
        run_id: str,
        task_type: str,
        algorithm_key: str,
        model_name: str,
        status: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        artifact_path: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> None:
        created_at = float(created_at if created_at is not None else time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO training_runs
                (run_id, created_at, task_type, algorithm_key, model_name, status, metrics_json, params_json, artifact_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    task_type,
                    algorithm_key,
                    model_name,
                    status,
                    json.dumps(metrics or {}),
                    json.dumps(params or {}),
                    artifact_path,
                ),
            )
            conn.commit()

    def list_runs(self, limit: int = 200) -> List[TrainingRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, created_at, task_type, algorithm_key, model_name, status, metrics_json, params_json, artifact_path FROM training_runs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()

        out: List[TrainingRun] = []
        for r in rows:
            out.append(
                TrainingRun(
                    run_id=str(r[0]),
                    created_at=float(r[1]),
                    task_type=str(r[2]),
                    algorithm_key=str(r[3]),
                    model_name=str(r[4]),
                    status=str(r[5]),
                    metrics_json=str(r[6] or "{}"),
                    params_json=str(r[7] or "{}"),
                    artifact_path=str(r[8]) if r[8] else None,
                )
            )
        return out

    def artifact_file(self, run_id: str) -> Path:
        return self.artifacts_dir / f"{run_id}.pkl"

    # ---- Recommender persistence ----

    def save_recommender_feedback(
        self,
        *,
        user_id: str,
        dataset_sig: str,
        task_type: str,
        profile: str,
        algorithm_key: str,
        model_name: str,
        score_total: int,
        accepted: bool,
        payload: Dict[str, Any],
        created_at: Optional[float] = None,
    ) -> None:
        created_at = float(created_at if created_at is not None else time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO recommender_feedback
                (created_at, user_id, dataset_sig, task_type, profile, algorithm_key, model_name, score_total, accepted, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    str(user_id or "default"),
                    str(dataset_sig or ""),
                    str(task_type or ""),
                    str(profile or ""),
                    str(algorithm_key or ""),
                    str(model_name or ""),
                    int(score_total),
                    1 if accepted else 0,
                    json.dumps(payload or {}),
                ),
            )
            conn.commit()

    def get_recommender_bias(self, *, user_id: str, task_type: str, limit: int = 500) -> Dict[str, float]:
        """Return per-algorithm_key bias (small score bump) from accept/reject history."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT algorithm_key, SUM(CASE WHEN accepted=1 THEN 1 ELSE 0 END) AS a,
                       SUM(CASE WHEN accepted=0 THEN 1 ELSE 0 END) AS r
                FROM recommender_feedback
                WHERE user_id=? AND task_type=?
                GROUP BY algorithm_key
                ORDER BY (a+r) DESC
                LIMIT ?
                """,
                (str(user_id or "default"), str(task_type or ""), int(limit)),
            ).fetchall()

        bias: Dict[str, float] = {}
        for alg, a, r in rows or []:
            a = int(a or 0)
            r = int(r or 0)
            if a + r <= 0:
                continue
            # scale: -6..+6 roughly
            b = (a - r) / max(1.0, (a + r))
            bias[str(alg)] = float(max(-6.0, min(6.0, b * 6.0)))
        return bias

    def load_recommender_prefs(self, *, user_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT prefs_json FROM recommender_prefs WHERE user_id=?",
                (str(user_id or "default"),),
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0] or "{}")
        except Exception:
            return {}

    def save_recommender_prefs(self, *, user_id: str, prefs: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recommender_prefs (user_id, updated_at, prefs_json)
                VALUES (?, ?, ?)
                """,
                (str(user_id or "default"), float(time.time()), json.dumps(prefs or {})),
            )
            conn.commit()
