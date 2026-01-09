"""Workflow module base definitions.

This file provides a light-weight modular separation aligned with the 4 workflow sections:
- Data Processing Module
- Model Training Module
- Evaluation & Comparison Module
- Model Recommender Engine
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowModuleNames:
    data_processing: str = "data_processing"
    model_training: str = "model_training"
    evaluation_comparison: str = "evaluation_comparison"
    model_recommender: str = "model_recommender"
