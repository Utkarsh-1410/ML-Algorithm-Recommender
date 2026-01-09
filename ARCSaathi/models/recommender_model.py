"""Model Recommender Engine (Models layer).

This module provides recommendation logic independent of the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .evaluation_model import EvaluationModel


@dataclass(frozen=True)
class Recommendation:
    model_name: str
    metric: str
    maximize: bool


class RecommenderModel:
    """Produces model recommendations from evaluation results."""

    def recommend(self, evaluation_model: EvaluationModel, problem_type: str = "classification") -> Optional[Recommendation]:
        if not evaluation_model.results:
            return None

        metric, maximize = self._default_metric(problem_type)
        best = evaluation_model.get_best_model(metric=metric, maximize=maximize)
        if not best:
            return None

        return Recommendation(model_name=best, metric=metric, maximize=maximize)

    def _default_metric(self, problem_type: str) -> Tuple[str, bool]:
        if (problem_type or "").lower() == "regression":
            return ("rmse", False)
        return ("f1_score", True)
