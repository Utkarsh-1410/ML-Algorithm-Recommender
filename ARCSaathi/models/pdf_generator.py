"""Professional PDF report generator for ML recommendations.

Generates comprehensive, industry-grade PDF reports using ReportLab,
including dataset analysis, algorithm recommendations, performance
comparisons, visualizations, and detailed reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    KeepTogether,
)
from reportlab.platypus import SimpleDocTemplate as PDF


matplotlib.use("Agg")


@dataclass
class ReportConfig:
    title: str = "ML Algorithm Recommender Report"
    subtitle: str = "Automated Algorithm Selection & Analysis"
    author: str = "ARCSaathi"
    include_visualizations: bool = True
    page_size: Tuple[float, float] = A4
    margin: float = 0.5 * inch


class PDFReportGenerator:
    """Generate professional PDF reports for ML algorithm recommendations."""

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.styles = self._build_styles()

    def _build_styles(self) -> Dict[str, ParagraphStyle]:
        base_styles = getSampleStyleSheet()
        custom = {}

        custom["title"] = ParagraphStyle(
            "title",
            parent=base_styles["Heading1"],
            fontSize=28,
            textColor=colors.HexColor("#1F77B4"),
            spaceAfter=12,
            alignment=1,  # center
        )

        custom["subtitle"] = ParagraphStyle(
            "subtitle",
            parent=base_styles["Heading2"],
            fontSize=16,
            textColor=colors.HexColor("#555555"),
            spaceAfter=24,
            alignment=1,
        )

        custom["heading2"] = ParagraphStyle(
            "heading2",
            parent=base_styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#1F77B4"),
            spaceAfter=10,
            spaceBefore=10,
        )

        custom["heading3"] = ParagraphStyle(
            "heading3",
            parent=base_styles["Heading3"],
            fontSize=12,
            textColor=colors.HexColor("#2C3E50"),
            spaceAfter=8,
        )

        custom["body"] = ParagraphStyle(
            "body",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=14,
        )

        custom["highlight"] = ParagraphStyle(
            "highlight",
            parent=base_styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#E74C3C"),
            leading=14,
            spaceAfter=6,
        )

        return custom

    def generate(
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
        Generate a complete PDF report.

        Args:
            output_path: Path to save the PDF.
            dataset_info: Dataset metadata (rows, columns, memory, etc.).
            task_detection: Task detection results (task_type, target, reasoning, etc.).
            recommendations: List of algorithm recommendations with scores/reasoning.
            trained_models: Dictionary of trained model results (optional).
            visualizations: Dictionary of visualization images as BytesIO (optional).

        Returns:
            True if success, False if error.
        """
        try:
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=self.config.page_size,
                rightMargin=self.config.margin,
                leftMargin=self.config.margin,
                topMargin=self.config.margin,
                bottomMargin=self.config.margin,
            )

            story = []

            story.append(self._build_title_page())
            story.append(PageBreak())

            story.extend(self._build_execution_summary(task_detection))
            story.append(Spacer(1, 0.3 * inch))

            story.extend(self._build_dataset_analysis(dataset_info))
            story.append(Spacer(1, 0.2 * inch))

            story.extend(self._build_problem_statement(task_detection))
            story.append(PageBreak())

            story.extend(
                self._build_recommendations_section(recommendations, dataset_info)
            )
            story.append(PageBreak())

            if trained_models and len(trained_models) > 0:
                story.extend(
                    self._build_model_comparison(trained_models, task_detection)
                )
                story.append(PageBreak())

            if visualizations:
                story.extend(self._build_visualizations_section(visualizations))
                story.append(PageBreak())

            story.extend(self._build_conclusion(recommendations, trained_models))

            doc.build(story)
            return True

        except Exception as e:
            print(f"PDF generation error: {e}")
            return False

    def _build_title_page(self) -> Paragraph:
        """Build title page section."""
        elements = []
        elements.append(Spacer(1, 1.5 * inch))
        elements.append(Paragraph(self.config.title, self.styles["title"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(self.config.subtitle, self.styles["subtitle"]))
        elements.append(Spacer(1, 0.5 * inch))

        now = datetime.now()
        elements.append(
            Paragraph(
                f"<b>Generated:</b> {now.strftime('%B %d, %Y at %H:%M:%S')}",
                self.styles["body"],
            )
        )
        elements.append(
            Paragraph(
                f"<b>Tool:</b> {self.config.author}",
                self.styles["body"],
            )
        )

        return elements

    def _build_execution_summary(self, task: Dict[str, Any]) -> List[Any]:
        """Build executive summary section."""
        elements = []
        elements.append(Paragraph("Executive Summary", self.styles["heading2"]))

        task_type = task.get("task_type", "Unknown").title()
        target = task.get("target", "(None)")

        summary_text = (
            f"This report presents an automated analysis of your dataset and "
            f"provides recommendations for the most suitable machine learning algorithms. "
            f"The analysis detected a <b>{task_type}</b> task with target variable: "
            f"<b>{target}</b>."
        )
        elements.append(Paragraph(summary_text, self.styles["body"]))
        elements.append(Spacer(1, 0.15 * inch))

        reasoning = task.get("reasoning", [])
        if reasoning:
            elements.append(Paragraph("<b>Task Detection Reasoning:</b>", self.styles["heading3"]))
            for i, reason in enumerate(reasoning[:3], 1):
                elements.append(
                    Paragraph(
                        f"<bullet>•</bullet> {reason}",
                        self.styles["body"],
                    )
                )

        return elements

    def _build_dataset_analysis(self, info: Dict[str, Any]) -> List[Any]:
        """Build dataset analysis section."""
        elements = []
        elements.append(Paragraph("Dataset Overview", self.styles["heading2"]))

        rows = info.get("rows", 0)
        columns = info.get("columns", 0)
        memory_mb = info.get("memory_mb", 0.0)

        size_category = self._categorize_size(rows)

        summary_data = [
            ["Metric", "Value"],
            ["Total Rows", f"{rows:,}"],
            ["Total Columns", f"{columns}"],
            ["Memory Usage", f"{memory_mb:.2f} MB"],
            ["Dataset Size Category", f"{size_category}"],
        ]

        table = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F77B4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                ]
            )
        )
        elements.append(table)

        return elements

    def _build_problem_statement(self, task: Dict[str, Any]) -> List[Any]:
        """Build problem statement and detection explanation."""
        elements = []
        elements.append(Paragraph("Problem Statement & Detection", self.styles["heading2"]))

        task_type = task.get("task_type", "Unknown").title()
        target = task.get("target")
        metrics = task.get("metrics", [])
        details = task.get("details", {})

        elements.append(
            Paragraph(
                f"<b>Detected Task Type:</b> {task_type}",
                self.styles["body"],
            )
        )
        elements.append(Spacer(1, 0.1 * inch))

        if target:
            elements.append(
                Paragraph(
                    f"<b>Target Variable:</b> {target}",
                    self.styles["body"],
                )
            )
        elements.append(Spacer(1, 0.1 * inch))

        if metrics:
            elements.append(
                Paragraph(
                    f"<b>Evaluation Metrics:</b> {', '.join(metrics)}",
                    self.styles["body"],
                )
            )

        reasoning = task.get("reasoning", [])
        if reasoning:
            elements.append(Spacer(1, 0.15 * inch))
            elements.append(Paragraph("<b>Detection Rationale:</b>", self.styles["heading3"]))
            for reason in reasoning:
                elements.append(
                    Paragraph(
                        f"<bullet>•</bullet> {reason}",
                        self.styles["body"],
                    )
                )

        return elements

    def _build_recommendations_section(
        self, recommendations: List[Dict[str, Any]], dataset_info: Dict[str, Any]
    ) -> List[Any]:
        """Build algorithm recommendations section."""
        elements = []
        elements.append(Paragraph("Recommended Algorithms", self.styles["heading2"]))

        elements.append(
            Paragraph(
                "Based on comprehensive dataset analysis, the following algorithms are recommended "
                "in order of suitability:",
                self.styles["body"],
            )
        )
        elements.append(Spacer(1, 0.15 * inch))

        for i, rec in enumerate(recommendations[:5], 1):
            name = rec.get("name", f"Algorithm {i}")
            score = rec.get("score", 0.0)
            reasoning = rec.get("reasoning", [])
            pros = rec.get("pros", "")
            cons = rec.get("cons", "")

            elements.append(
                Paragraph(
                    f"<b>{i}. {name}</b> (Score: {score:.2f}/100)",
                    self.styles["heading3"],
                )
            )

            if reasoning:
                elements.append(
                    Paragraph("<i>Why:</i>", self.styles["body"])
                )
                for r in reasoning[:2]:
                    elements.append(
                        Paragraph(
                            f"<bullet>•</bullet> {r}",
                            self.styles["body"],
                        )
                    )

            if pros:
                elements.append(
                    Paragraph(
                        f"<font color='green'><b>Strengths:</b></font> {pros}",
                        self.styles["body"],
                    )
                )

            if cons:
                elements.append(
                    Paragraph(
                        f"<font color='red'><b>Limitations:</b></font> {cons}",
                        self.styles["body"],
                    )
                )

            elements.append(Spacer(1, 0.1 * inch))

        return elements

    def _build_model_comparison(
        self, trained_models: Dict[str, Any], task: Dict[str, Any]
    ) -> List[Any]:
        """Build model comparison section."""
        elements = []
        elements.append(Paragraph("Model Training Results", self.styles["heading2"]))

        if not trained_models:
            elements.append(
                Paragraph("No trained models available for comparison.", self.styles["body"])
            )
            return elements

        comparisons = []
        comparisons.append(["Algorithm", "Status", "Primary Metric", "Value"])

        for algo_name, model_info in trained_models.items():
            status = "✓ Trained" if model_info.get("trained") else "✗ Error"
            metrics = model_info.get("metrics", {})
            primary_metric = list(metrics.keys())[0] if metrics else "N/A"
            primary_value = f"{list(metrics.values())[0]:.4f}" if metrics else "N/A"

            comparisons.append([algo_name, status, primary_metric, primary_value])

        table = Table(comparisons, colWidths=[2.0 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27AE60")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ECF0F1")]),
                ]
            )
        )
        elements.append(table)

        return elements

    def _build_visualizations_section(
        self, visualizations: Dict[str, BytesIO]
    ) -> List[Any]:
        """Build visualizations section."""
        elements = []
        elements.append(Paragraph("Data & Performance Visualizations", self.styles["heading2"]))

        for name, img_bytes in visualizations.items():
            if img_bytes:
                try:
                    img_bytes.seek(0)
                    img = Image(img_bytes, width=5.5 * inch, height=4.0 * inch)
                    elements.append(Paragraph(f"<i>{name}</i>", self.styles["heading3"]))
                    elements.append(img)
                    elements.append(Spacer(1, 0.2 * inch))
                except Exception as e:
                    elements.append(
                        Paragraph(
                            f"[Visualization '{name}' could not be rendered]",
                            self.styles["body"],
                        )
                    )

        return elements

    def _build_conclusion(
        self,
        recommendations: List[Dict[str, Any]],
        trained_models: Optional[Dict[str, Any]],
    ) -> List[Any]:
        """Build conclusion and final recommendation."""
        elements = []
        elements.append(Paragraph("Conclusion & Recommendations", self.styles["heading2"]))

        if recommendations:
            top_rec = recommendations[0]
            name = top_rec.get("name", "Top Algorithm")
            score = top_rec.get("score", 0.0)

            elements.append(
                Paragraph(
                    f"<b>Recommended Best Model:</b> {name} "
                    f"(Baseline Score: {score:.2f}/100)",
                    self.styles["heading3"],
                )
            )
            elements.append(Spacer(1, 0.1 * inch))

            conclusion_text = (
                f"Based on comprehensive feature analysis, dataset characteristics, "
                f"and empirical testing, <b>{name}</b> emerged as the most suitable algorithm "
                f"for this task. This recommendation considers dataset size, feature complexity, "
                f"task type, and expected performance metrics."
            )
            elements.append(Paragraph(conclusion_text, self.styles["body"]))
            elements.append(Spacer(1, 0.15 * inch))

        elements.append(Paragraph("<b>Next Steps:</b>", self.styles["heading3"]))
        next_steps = [
            "Fine-tune the recommended model's hyperparameters",
            "Perform cross-validation on the full dataset",
            "Conduct feature engineering to improve performance",
            "Compare against multiple baselines in production",
            "Monitor model performance on new, unseen data",
        ]
        for step in next_steps:
            elements.append(
                Paragraph(
                    f"<bullet>•</bullet> {step}",
                    self.styles["body"],
                )
            )

        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(self._build_footer(), self.styles["body"]))

        return elements

    def _build_footer(self) -> str:
        """Build footer text."""
        return (
            f"<i>This report was automatically generated by {self.config.author}. "
            f"Recommendations are based on statistical analysis and should be validated "
            f"against domain knowledge and business requirements.</i>"
        )

    @staticmethod
    def _categorize_size(rows: int) -> str:
        if rows < 1000:
            return "Small (<1K rows)"
        elif rows < 100000:
            return "Medium (1K–100K rows)"
        else:
            return "Large (>100K rows)"
