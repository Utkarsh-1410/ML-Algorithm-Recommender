"""Report controller for PDF generation.

Coordinates data collection from various sources and generates
professional PDF reports.
"""

from __future__ import annotations

from io import BytesIO
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from ARCSaathi.models.data_model import DataModel
from ARCSaathi.models.pdf_generator import PDFReportGenerator, ReportConfig


class ReportController(QObject):
    """Coordinates PDF report generation."""

    pdf_generated = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, data_model: DataModel, logger: Optional[Logger] = None):
        super().__init__()
        self.data_model = data_model
        self.logger = logger

    def generate_report(
        self,
        output_path: str | Path,
        *,
        dataset_info: Dict[str, Any],
        task_detection: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        trained_models: Optional[Dict[str, Any]] = None,
        visualizations: Optional[Dict[str, BytesIO]] = None,
    ) -> bool:
        """
        Generate a comprehensive PDF report.

        Args:
            output_path: Path to save the PDF.
            dataset_info: Dataset metadata.
            task_detection: Task detection results.
            recommendations: Algorithm recommendations.
            trained_models: Trained model results (optional).
            visualizations: Visualization images as BytesIO (optional).

        Returns:
            True if successful, False otherwise.
        """
        try:
            config = ReportConfig(
                title="ML Algorithm Recommender — Comprehensive Analysis Report",
                subtitle="Automated Algorithm Selection & Performance Comparison",
                author="ARCSaathi",
                include_visualizations=visualizations is not None and len(visualizations) > 0,
            )

            generator = PDFReportGenerator(config)

            success = generator.generate(
                output_path,
                dataset_info=dataset_info,
                task_detection=task_detection,
                recommendations=recommendations,
                trained_models=trained_models,
                visualizations=visualizations,
            )

            if success:
                self.pdf_generated.emit(str(output_path))
                if self.logger:
                    self.logger.info(f"PDF report generated: {output_path}")
                return True
            else:
                msg = "PDF generation failed (unknown error)"
                self.error_occurred.emit(msg)
                if self.logger:
                    self.logger.error(msg)
                return False

        except Exception as e:
            msg = f"PDF generation error: {str(e)}"
            self.error_occurred.emit(msg)
            if self.logger:
                self.logger.exception(msg)
            return False

    def sanitize_recommendations_for_pdf(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare recommendations data for PDF inclusion.

        Ensures all fields are serializable and contain useful text.
        """
        sanitized = []

        for rec in recommendations[:5]:
            sanitized.append(
                {
                    "name": str(rec.get("name", "Unknown")),
                    "score": float(rec.get("score", 0.0)),
                    "reasoning": [
                        str(r) for r in (rec.get("reasoning") or [])[:3]
                    ],
                    "pros": str(rec.get("pros", "")),
                    "cons": str(rec.get("cons", "")),
                }
            )

        return sanitized

    def sanitize_models_for_pdf(
        self, trained_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare trained model results for PDF inclusion.
        """
        sanitized = {}

        for name, info in trained_models.items():
            metrics = {}
            if "metrics" in info and isinstance(info["metrics"], dict):
                for metric_name, metric_value in info["metrics"].items():
                    try:
                        metrics[str(metric_name)] = float(metric_value)
                    except (ValueError, TypeError):
                        metrics[str(metric_name)] = None

            sanitized[str(name)] = {
                "trained": bool(info.get("trained", False)),
                "metrics": metrics,
                "error": str(info.get("error", "")),
            }

        return sanitized
