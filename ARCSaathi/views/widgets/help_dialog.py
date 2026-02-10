"""Help & Documentation Dialog Widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTabWidget,
    QTextEdit,
    QWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QHBoxLayout,
)
from PySide6.QtGui import QFont


class HelpDialog(QDialog):
    """Standalone help documentation dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ARCSaathi - Help & Documentation")
        self.setMinimumSize(900, 700)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the help dialog UI."""
        layout = QVBoxLayout(self)

        # Tabbed help content
        tabs = QTabWidget()
        tabs.addTab(self._build_quickstart_tab(), "🚀 Quick Start")
        tabs.addTab(self._build_features_tab(), "⚙️ Features")
        tabs.addTab(self._build_algorithms_tab(), "🤖 Algorithms")
        tabs.addTab(self._build_predictive_maintenance_tab(), "🔧 Predictive Maintenance")
        tabs.addTab(self._build_tips_tab(), "💡 Tips & Tricks")
        tabs.addTab(self._build_troubleshooting_tab(), "🔍 Troubleshooting")
        tabs.addTab(self._build_api_tab(), "🔌 API Reference")
        tabs.addTab(self._build_about_tab(), "ℹ️ About")

        layout.addWidget(tabs)

    def _build_quickstart_tab(self) -> QWidget:
        """Quick start guide."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        content = QTextEdit()
        content.setReadOnly(True)
        content.setMarkdown("""
# Quick Start Guide

## 1️⃣ Load Your Dataset
- Click **"📁 Load Dataset"** in the Data Loading tab
- Select a CSV or Excel file
- The system will automatically analyze your dataset

## 2️⃣ Review Dataset Analysis
- View dataset summary (rows, columns, memory usage)
- Check data profiling results (missing values, distributions, correlations)
- Examine feature types and statistics

## 3️⃣ Configure Preprocessing (Optional)
- Go to **Preprocessing** tab
- Add preprocessing steps (imputation, encoding, scaling)
- Validate steps with preview functionality
- Click **"Apply Pipeline"** when ready

## 4️⃣ Get Algorithm Recommendations
- Navigate to **Model Recommender** tab
- Review auto-detected problem type
- See ranked algorithm recommendations with scores
- Read detailed reasoning for each algorithm

## 5️⃣ Train & Compare Models
- Go to **Model Training** tab
- Select algorithms to train
- Monitor training progress
- Compare model performance in **Results** dashboard

## 6️⃣ Export Results
- Generate **PDF Report** with complete analysis
- Export comparison tables (CSV/Excel)
- Save visualizations (PNG/SVG)

## 7️⃣ (Optional) Monitor Predictive Maintenance
- Use **Predictive Maintenance** tab for automotive fleet monitoring
- Upload training data and configure real-time API connections
- Track component health and RUL predictions
        """)

        layout.addWidget(content)
        return widget

    def _build_features_tab(self) -> QWidget:
        """Core features documentation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        content = QTextEdit()
        content.setReadOnly(True)
        content.setMarkdown("""
# ARCSaathi Features

## 📊 Data Loading & Analysis
- **Multi-format Support**: CSV, Excel, Parquet, JSON
- **Automatic Profiling**: Size, types, quality metrics
- **Missing Value Detection**: Percentage and patterns
- **Correlation Analysis**: Feature relationships
- **Distribution Visualization**: Histograms, density plots

## 🤖 Intelligent Recommendations
- **Multi-factor Scoring**: Dataset size, feature types, complexity
- **18 Optimized Algorithms**: Regressors, classifiers, clusterers, dimensionality reduction
- **Detailed Reasoning**: Why each algorithm is recommended
- **Confidence Metrics**: Algorithm suitability scores
- **Ranked Top 5**: Most suitable algorithms first

## 🔄 Smart Preprocessing
- **Automated Pipeline**: Suggested transformations
- **Missing Value Handling**: Mean/median/forward fill
- **Categorical Encoding**: One-hot, target, label encoding
- **Feature Scaling**: Standard, MinMax, Robust scaling
- **Step-by-step Validation**: Preview before applying

## 🏋️ Model Training
- **Batch Training**: Train multiple models simultaneously
- **Hyperparameter Support**: Configure algorithm parameters
- **Cross-Validation**: K-fold validation (default: 5-fold)
- **Train/Test Split**: 80/20 by default, customizable
- **Progress Tracking**: Real-time training status

## 📈 Evaluation & Comparison
- **Classification Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Regression Metrics**: RMSE, MAE, R² Score, MAPE
- **Clustering Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz
- **Comparison Dashboard**: Side-by-side model performance
- **Visualization Charts**: Performance comparisons

## 📄 Professional Reports
- **PDF Generation**: Industry-grade report format
- **Executive Summary**: Key findings and recommendations
- **Dataset Analysis**: Statistics and characteristics
- **Algorithm Recommendations**: Top 5 with reasoning
- **Performance Comparison**: Metrics and visualizations
- **Conclusions**: Best model justification

## 🔍 Explainability
- **SHAP Values**: Feature importance and impact
- **LIME**: Local model explanations
- **Feature Importance**: Tree-based algorithm insights
- **Prediction Explanation**: Why model made specific prediction
- **Drift Detection**: Monitor model performance over time

## 🔧 Predictive Maintenance (Automotive)
- **Training Data Management**: Upload and manage datasets
- **Real-time API Integration**: Connect sensor streams
- **RUL Prediction**: Remaining Useful Life calculation
- **Component Health**: Visual health bars and alerts
- **Sensor Monitoring**: 15+ automotive sensor types
        """)

        layout.addWidget(content)
        return widget

    def _build_algorithms_tab(self) -> QWidget:
        """Algorithm descriptions."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Algorithm list
        list_widget = QListWidget()
        algorithms = {
            "Regression": [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
            ],
            "Classification": [
                "Logistic Regression",
                "Random Forest Classifier",
                "SVM (SVC)",
                "KNN Classifier",
                "Naive Bayes",
            ],
            "Clustering": [
                "K-Means",
                "DBSCAN",
                "Agglomerative Clustering",
                "Gaussian Mixture Model",
                "OPTICS",
            ],
            "Dimensionality Reduction": [
                "PCA",
                "t-SNE",
                "UMAP",
            ],
        }

        for category, algos in algorithms.items():
            item = QListWidgetItem(category)
            item.setFont(QFont("Arial", 10, QFont.Bold))
            list_widget.addItem(item)
            for algo in algos:
                sub_item = QListWidgetItem(f"  • {algo}")
                list_widget.addItem(sub_item)

        list_widget.itemSelectionChanged.connect(
            lambda: self._update_algorithm_description(list_widget)
        )

        # Algorithm description
        self.algo_description = QTextEdit()
        self.algo_description.setReadOnly(True)
        self.algo_description.setMarkdown(self._get_algorithm_info("Linear Regression"))

        layout.addWidget(list_widget, 1)
        layout.addWidget(self.algo_description, 2)

        return widget

    def _update_algorithm_description(self, list_widget: QListWidget) -> None:
        """Update algorithm description based on selection."""
        item = list_widget.currentItem()
        if item and not item.text().endswith(":"):
            algo_name = item.text().strip("• ")
            if algo_name:
                self.algo_description.setMarkdown(self._get_algorithm_info(algo_name))

    def _get_algorithm_info(self, algo_name: str) -> str:
        """Get detailed algorithm information."""
        info = {
            "Linear Regression": """
## Linear Regression

**What it does**: Fits a linear relationship between features and target.

**Best for**:
- Continuous numerical targets
- Linear relationships
- Interpretability is important

**Pros**:
- ✅ Very fast training
- ✅ Highly interpretable
- ✅ Good baseline model
- ✅ Low computational cost

**Cons**:
- ❌ Assumes linear relationship
- ❌ Sensitive to outliers
- ❌ Poor with non-linear patterns

**Time Complexity**: O(n·p²)
**Memory**: Low
""",
            "Ridge Regression": """
## Ridge Regression

**What it does**: Linear regression with L2 regularization to prevent overfitting.

**Best for**:
- High-dimensional data with multicollinearity
- Preventing overfitting
- Numerical predictions

**Pros**:
- ✅ More stable than Linear Regression
- ✅ Handles multicollinearity well
- ✅ Fast training
- ✅ Reduces overfitting

**Cons**:
- ❌ Still linear model
- ❌ Requires alpha tuning
- ❌ Less interpretable than Linear Regression

**Time Complexity**: O(n·p²)
**Memory**: Low
""",
            "Random Forest Regressor": """
## Random Forest Regressor

**What it does**: Ensemble of decision trees for robust predictions.

**Best for**:
- Non-linear patterns
- Mixed feature types
- Strong general-purpose regression
- Handling noisy data

**Pros**:
- ✅ Excellent performance
- ✅ Handles non-linearities
- ✅ Feature importance ranking
- ✅ Robust to outliers

**Cons**:
- ❌ Black-box model
- ❌ Slower prediction time
- ❌ Can overfit with many trees

**Time Complexity**: O(T·n·p·log n)
**Memory**: Medium
""",
            "Logistic Regression": """
## Logistic Regression

**What it does**: Linear classifier with probabilistic outputs.

**Best for**:
- Binary classification baseline
- Interpretable models
- Well-balanced datasets

**Pros**:
- ✅ Very fast
- ✅ Interpretable coefficients
- ✅ Probabilistic scores
- ✅ Good baseline

**Cons**:
- ❌ Linear decision boundary
- ❌ Assumes independence
- ❌ Poor with imbalanced data

**Time Complexity**: O(n·p)
**Memory**: Low
""",
            "Random Forest Classifier": """
## Random Forest Classifier

**What it does**: Ensemble of decision trees for classification.

**Best for**:
- Multi-class classification
- Non-linear patterns
- Mixed feature types
- Imbalanced datasets

**Pros**:
- ✅ Strong baseline algorithm
- ✅ Feature importance
- ✅ Handles imbalance well
- ✅ Robust predictions

**Cons**:
- ❌ Hard to interpret
- ❌ Slower than tree alone
- ❌ Memory intensive

**Time Complexity**: O(T·n·p·log n)
**Memory**: Medium
""",
            "K-Means": """
## K-Means Clustering

**What it does**: Partition data into K spherical clusters.

**Best for**:
- Unsupervised learning
- Spherical cluster detection
- Known number of clusters
- Large datasets

**Pros**:
- ✅ Very fast
- ✅ Simple to understand
- ✅ Scalable to large data
- ✅ Low memory usage

**Cons**:
- ❌ Need to specify K
- ❌ Sensitive to outliers
- ❌ Sensitive to initialization
- ❌ Only spherical clusters

**Time Complexity**: O(n·k·i)
**Memory**: Low
""",
            "PCA": """
## Principal Component Analysis

**What it does**: Linear dimensionality reduction preserving variance.

**Best for**:
- Dimensionality reduction
- Visualization (2D/3D)
- Removing multicollinearity
- Feature extraction

**Pros**:
- ✅ Fast computation
- ✅ Interpretable components
- ✅ Optimal variance preservation
- ✅ Visualization ready

**Cons**:
- ❌ Only linear transformations
- ❌ Loses non-linear structure
- ❌ Loses interpretability

**Time Complexity**: O(n·p·min(n,p))
**Memory**: Medium
""",
        }

        return info.get(
            algo_name,
            f"""
# {algo_name}

*Detailed information coming soon.*

For more details, visit the [scikit-learn documentation](https://scikit-learn.org/).
""",
        )

    def _build_predictive_maintenance_tab(self) -> QWidget:
        """Predictive maintenance guide."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        content = QTextEdit()
        content.setReadOnly(True)
        content.setMarkdown("""
# Predictive Maintenance for Automotive

## 🚗 Overview

The Predictive Maintenance module monitors automotive fleet health by analyzing
sensor data to predict component failures before they occur.

## 📊 Sensor Types Supported

- **Temperature Sensors**: Cylinder head, exhaust gas, bearing, oil
- **Vibration Sensors**: Engine, crankshaft, knock sensor
- **Pressure Sensors**: Oil, coolant, injector
- **Contamination & Quality**: Ferrous debris, soot in oil
- **Gas & Flow**: Mass air flow, oxygen sensor, EGR flow

## 🔧 How to Use

**Step 1**: Upload training data
**Step 2**: Train model
**Step 3**: Configure API connection
**Step 4**: Monitor components
**Step 5**: Review sensor details

## 🎯 Key Metrics

- **RUL**: Remaining Useful Life
- **Health Score**: 0-100 component health
- **Trend**: Improvement or degradation
- **Alert Level**: Critical/Warning/Normal
        """)

        layout.addWidget(content)
        return widget

    def _build_tips_tab(self) -> QWidget:
        """Tips and best practices."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        content = QTextEdit()
        content.setReadOnly(True)
        content.setMarkdown("""
# Tips & Best Practices

## 📊 Data Preparation
- Use at least 100 rows for reliable models
- Balance data for classification
- Clean data and handle missing values
- Ensure consistent data types

## 🤖 Algorithm Selection
- **Regression**: Linear/Ridge for linear, Random Forest for non-linear
- **Classification**: Random Forest for imbalanced, Logistic for baseline
- **Clustering**: K-Means for speed, DBSCAN for unknown cluster count

## 🏋️ Training Tips
- Start with default parameters
- Use cross-validation for reliable estimates
- Monitor overfitting: watch validation vs training metrics

## 🔧 Preprocessing
- **Scale features**: Always for SVM, KNN, Linear Models
- **Encoding**: One-Hot for trees, Label for ordinal data
- **Missing Values**: Mean imputation for MCAR, drop if <5%

## 📈 Evaluation
- **Regression**: RMSE (same units as target), R² (0-1, higher better)
- **Classification**: F1 for imbalanced, Accuracy for balanced
        """)

        layout.addWidget(content)
        return widget

    def _build_troubleshooting_tab(self) -> QWidget:
        """Troubleshooting guide."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        content = QTextEdit()
        content.setReadOnly(True)
        content.setMarkdown("""
# Troubleshooting Guide

## 🔴 Common Issues

**"File not found" error**
- Ensure file exists and path is correct
- Check file permissions
- Use supported formats (CSV, Excel, Parquet)

**"No recommendations generated"**
- Dataset might be too small (<50 rows)
- Check if target variable is selected
- Verify problem type is auto-detected

**"Model training fails"**
- Check dataset has both X and y
- Verify target variable is meaningful
- Check for NaN or infinite values
- Try preprocessing first

**"Low model performance"**
- Dataset size too small
- Features poorly correlated with target
- Try feature engineering
- Collect more diverse data

**"API connection fails"**
- Verify endpoint URL is correct
- Check internet connection
- Confirm API is running
- Check firewall/proxy settings

## 💾 System Requirements

**Minimum**:
- 4GB RAM
- 2GB free disk space
- Intel i5 / Ryzen 5 or equivalent
- Python 3.9+

**Recommended**:
- 8GB+ RAM
- 5GB free disk space
- Intel i7 / Ryzen 7 or equivalent
- SSD for faster I/O
        """)

        layout.addWidget(content)
        return widget

    def _build_api_tab(self) -> QWidget:
        """API reference documentation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        content = QTextEdit()
        content.setReadOnly(True)
        content.setMarkdown("""
# API Reference

## 🔌 Sample API Server

Starting the API:
```bash
python sample_api.py
```

Server runs on `http://localhost:5000`

## Endpoint: GET /api/sensor-data

**URL**: `http://localhost:5000/api/sensor-data`

**Response**: JSON with sensor readings, timestamp, and status

**Example cURL**:
```bash
curl http://localhost:5000/api/sensor-data \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Python Integration

```python
from ARCSaathi.models import DataModel, MLModel

# Load data
data = DataModel()
data.load_from_file("data.csv")

# Train model
ml = MLModel()
ml.train_model("reg_random_forest", X_train, y_train)
```

## Authentication

### Bearer Token
```bash
curl http://api.example.com/data \\
  -H "Authorization: Bearer YOUR_TOKEN"
```

### API Key
```bash
curl http://api.example.com/data \\
  -H "X-API-Key: YOUR_KEY"
```
        """)

        layout.addWidget(content)
        return widget

    def _build_about_tab(self) -> QWidget:
        """About and version information."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        content = QTextEdit()
        content.setReadOnly(True)
        content.setMarkdown("""
# About ARCSaathi

## 🎯 Mission

ARCSaathi is an intelligent ML algorithm recommender system designed to make
machine learning accessible to everyone.

## 🚗 Automotive Focus

Specialized support for automotive predictive maintenance, enabling fleet
operators to monitor component health and predict failures.

## 🏆 Key Capabilities

- **Automatic Problem Detection**: Classification, Regression, Clustering
- **Intelligent Recommendations**: 18 optimized algorithms with scoring
- **Smart Preprocessing**: Automated data cleaning and feature engineering
- **Model Training**: Multi-algorithm training with hyperparameter support
- **Professional Reports**: PDF generation with detailed analysis
- **Real-time Monitoring**: Predictive maintenance for automotive fleets
- **Explainability**: SHAP, LIME, drift detection for transparency

## 📚 Components

**Frontend**: PySide6, Matplotlib, Plotly
**Backend**: scikit-learn, pandas, numpy, ReportLab
**Integration**: Flask, SHAP, LIME

## 🏆 Project Statistics

- **Algorithms**: 18 (5 regression, 5 classification, 5 clustering, 3 dimred)
- **Sensor Types**: 15+ automotive sensors
- **UI Tabs**: 8 main tabs
- **Metrics**: 50+ across all task types
- **Lines of Code**: 10,000+

## 📄 License

Open-source project available under appropriate license.

---

**Version**: 1.0.0
**Status**: Production Ready
**Last Updated**: February 2026
        """)

        layout.addWidget(content)
        return widget
