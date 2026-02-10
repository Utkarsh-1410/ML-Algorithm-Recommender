"""Help & Documentation Tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QScrollArea,
    QGroupBox,
    QPushButton,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
)
from PySide6.QtGui import QFont


class HelpDocumentationTab(QWidget):
    """Comprehensive help and documentation interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the complete help UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("📚 Help & Documentation")
        header_font = header.font()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

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

### Temperature Sensors
- **Cylinder Head Temperature**: Engine combustion chamber heat
- **Exhaust Gas Temperature**: Post-combustion gas heat
- **Bearing Temperature**: Bearing surface heat
- **Oil Temperature**: Lubricant thermal state

### Vibration Sensors
- **Engine Vibration**: Overall engine vibration levels
- **Crankshaft Vibration**: Crankshaft oscillation
- **Knock Sensor**: Pre-ignition/knocking detection

### Pressure Sensors
- **Oil Pressure**: Engine lubrication system pressure
- **Coolant Pressure**: Cooling system pressure
- **Injector Pressure**: Fuel injection pressure

### Contamination & Quality
- **Ferrous Debris**: Metal particle concentration
- **Soot in Oil**: Carbon accumulation in lubricant

### Gas & Flow
- **Mass Air Flow**: Intake airflow measurement
- **Oxygen Sensor**: Exhaust oxygen content
- **EGR Flow**: Exhaust gas recirculation flow

## 🔧 How to Use

### Step 1: Upload Training Data
1. Go to **Predictive Maintenance** tab
2. Click **"📚 Training Data"** sub-tab
3. Upload CSV files with historical sensor data
4. Files should contain sensor readings and failure labels

### Step 2: Train Model
1. Click **"Train Model"** button
2. System will analyze training data
3. Model learns failure patterns
4. Receive training metrics and performance scores

### Step 3: Configure API Connection
1. Go to **"🔌 API Config"** sub-tab
2. Enter API endpoint URL
3. Set refresh interval (1-300 seconds)
4. Configure authentication if needed
5. Click **"Test Connection"** to verify

### Step 4: Monitor Components
1. View **"📊 Component Health"** sub-tab
2. See RUL (Remaining Useful Life) bars
3. Check current health status
4. Get alerts for critical components

### Step 5: Review Sensor Details
1. Go to **"📈 Sensor Details"** sub-tab
2. View real-time sensor readings
3. See min/max/average values
4. Track trends over time

## 🎯 Key Metrics

- **RUL (Remaining Useful Life)**: Days/hours until predicted failure
- **Health Score**: 0-100 overall component health
- **Trend**: Improvement or degradation direction
- **Alert Level**: Critical/Warning/Normal status
- **Last Update**: Timestamp of last data point

## 💡 Best Practices

1. **Regular Training**: Retrain model monthly with new data
2. **Data Quality**: Ensure sensor data is clean and accurate
3. **API Monitoring**: Check connection status regularly
4. **Threshold Adjustment**: Set appropriate alert thresholds
5. **Maintenance Scheduling**: Use RUL for preventive maintenance

## 🔍 Troubleshooting

- **High False Positives**: Lower alert thresholds
- **Missed Failures**: Increase training data volume
- **API Connection Issues**: Check network and endpoint URL
- **Slow Updates**: Increase refresh interval

## 📚 Technical Details

- Models: Random Forest with historical sensor data
- Update Frequency: Real-time (configurable 1-300s)
- Accuracy: ±15% RUL prediction (with adequate training data)
- Supported Formats: CSV, JSON from REST API
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

### Dataset Quality
- **Size Matters**: Use at least 100 rows for reliable models
- **Balance Data**: For classification, keep class ratio 1:3 or better
- **Clean Data**: Remove obvious errors and inconsistencies
- **Consistent Types**: Ensure columns have consistent data types
- **Handle Missing**: Fill or remove missing values appropriately

### Feature Engineering
- **Target Selection**: Choose a meaningful target variable
- **Remove IDs**: Drop ID columns auto-detected as identifiers
- **Scale Features**: Use provided scaling for algorithms like SVM, KNN
- **Create Features**: Add domain-specific features when possible
- **Correlations**: Check for highly correlated features

## 🤖 Algorithm Selection

### For Regression
- **Linear data**: Linear/Ridge Regression
- **Non-linear**: Random Forest, Gradient Boosting
- **Sparse data**: Ridge (L2) or Lasso (L1)
- **Unknown pattern**: Start with Random Forest

### For Classification
- **Imbalanced data**: Random Forest, Gradient Boosting
- **Binary**: Logistic Regression (baseline) + Forest
- **Multi-class**: Random Forest, Gradient Boosting
- **Interpretability needed**: Logistic Regression

### For Clustering
- **Known clusters**: K-Means (fast)
- **Unknown pattern**: DBSCAN (automatic K)
- **Hierarchical**: Agglomerative Clustering
- **Visualization**: PCA or t-SNE first

## 🏋️ Training Tips

### Hyperparameter Tuning
- **Start Simple**: Use default parameters first
- **Grid Search**: Systematically test parameter combinations
- **Cross-Validation**: Use 5-10 fold for reliable estimates
- **Monitor Overfitting**: Check validation vs training metrics

### Preventing Overfitting
- **Larger Datasets**: Collect more data if possible
- **Simpler Models**: Use fewer features or shallower trees
- **Regularization**: Use Ridge/Lasso instead of standard regression
- **Early Stopping**: Stop training when validation metrics plateau

## 📈 Evaluation

### Understanding Metrics

**Regression**:
- **RMSE**: Lower is better (same units as target)
- **MAE**: Mean absolute error (easier to interpret)
- **R²**: Proportion of variance explained (0-1, higher better)

**Classification**:
- **Accuracy**: Overall correctness (watch class imbalance)
- **Precision**: Correct positive predictions
- **Recall**: Fraction of actual positives found
- **F1**: Balance between precision and recall

## 🔧 Preprocessing

### When to Scale
- **SVM, KNN, Linear Models**: Always scale
- **Tree-based Models**: No need to scale
- **Neural Networks**: Always scale

### Encoding Strategies
- **One-Hot**: Use for tree models and many algorithms
- **Label**: Use for ordinal categorical data
- **Target**: Use when relationship exists with target

### Missing Values
- **Mean Imputation**: For MCAR (Missing Completely At Random)
- **Forward Fill**: For time-series data
- **Drop**: If < 5% of rows with missing values
- **Keep Missing Indicator**: Create binary "is missing" feature

## 💾 Exporting Results

### Report Generation
- **PDF**: Professional report with all analysis
- **CSV**: Detailed metrics for manual analysis
- **Excel**: Formatted tables with multiple sheets
- **PNG/SVG**: High-quality visualizations for presentations

### Sharing
- Use PDF for stakeholders (professional look)
- Use CSV for further analysis in other tools
- Use PNG for presentations and reports

## 🔍 Common Issues

**Low Performance?**
1. Check data quality
2. Try different algorithms
3. Feature engineer
4. Collect more data
5. Adjust preprocessing

**Overfitting?**
1. Simplify model
2. Reduce features
3. Add regularization
4. Increase training data

**Underfitting?**
1. Use more complex model
2. Add features
3. Reduce regularization
4. More training iterations
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

## 🔴 Common Issues & Solutions

### Data Loading Issues

**"File not found" error**
- Ensure file exists and path is correct
- Check file permissions
- Use supported formats (CSV, Excel, Parquet)

**"Encoding error" reading file**
- Try UTF-8 encoding (default)
- Check for special characters
- Use Excel or CSV converter

**"Too many columns" or memory warning**
- File is very large (>100MB)
- Consider sampling data or filtering columns
- Use Parquet format for efficiency

### Analysis Issues

**"No recommendations generated"**
- Dataset might be too small (<50 rows)
- Check if target variable is selected
- Verify problem type is auto-detected

**"Preprocessing step fails"**
- Incompatible data type for operation
- Check feature selection in step
- Review data preview for issues

### Training Issues

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

**"Training very slow"**
- Large dataset (>1M rows)
- Complex features or models
- Consider sampling or simplification
- Check system resources

### Predictive Maintenance Issues

**"API connection fails"**
- Verify endpoint URL is correct
- Check internet connection
- Confirm API is running
- Review firewall/proxy settings
- Check API authentication

**"No sensor data received"**
- Verify API response format (should be JSON)
- Check field names in returned data
- Review API logs for errors
- Test endpoint manually with curl

**"RUL predictions missing"**
- Ensure model is trained
- Verify sensor data has required fields
- Check model hasn't degraded
- Review data anomalies

## 🛠️ System Requirements

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

## 📋 Check List Before Reporting Issues

Before seeking help:
1. ✅ Updated to latest version
2. ✅ Sufficient system resources available
3. ✅ Data is valid and properly formatted
4. ✅ No background processes consuming resources
5. ✅ All dependencies installed (pip install -r requirements.txt)
6. ✅ Tried with different dataset
7. ✅ Checked logs for error messages

## 📞 Getting Help

### Documentation
- Review feature descriptions in **Features** tab
- Check algorithm details in **Algorithms** tab
- See best practices in **Tips & Tricks** tab

### Testing
1. Use sample dataset to verify functionality
2. Try simplified preprocessing pipeline
3. Test with smaller dataset first

### Debugging
1. Run `python main.py --diagnose`
2. Check error messages in console
3. Review application logs
4. Try reset to default settings

## 💾 Data Recovery

**Recovering from crash**:
- Re-upload dataset
- Use saved model/report from previous session
- Check temporary files in `.cache/` folder

**Backup best practices**:
- Save trained models regularly
- Export results as CSV/PDF
- Keep original dataset backup
- Version control configuration changes
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

The included `sample_api.py` provides a test endpoint for predictive maintenance.

### Starting the API

```bash
python sample_api.py
```

Server runs on `http://localhost:5000`

### Endpoint: GET /api/sensor-data

**URL**:
```
http://localhost:5000/api/sensor-data
```

**Response Format**:
```json
{
  "timestamp": "2026-02-10T10:30:45.123Z",
  "sensors": {
    "ferrous_debris": 15.3,
    "soot_in_oil": 32.5,
    "cylinder_head_temp": 95.2,
    "exhaust_gas_temp": 450.1,
    "bearing_temp": 62.0,
    "engine_vibration": 2.5,
    "knock_sensor": 28.0,
    "crankshaft_vibration": 550.2,
    "oil_temperature": 75.5,
    "injector_pressure": 10.5,
    "oil_pressure": 0.25,
    "coolant_pressure": 0.10,
    "mass_air_flow": 4.2,
    "oxygen_sensor": 0.98,
    "egr_flow": 8.5
  },
  "status": "normal",
  "rul_days": 45
}
```

**Example cURL**:
```bash
curl http://localhost:5000/api/sensor-data \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 🔑 Configuration in App

### API Settings
- **Endpoint URL**: REST API URL providing sensor data
- **API Key**: Bearer token for authentication (if required)
- **Refresh Interval**: Update frequency in seconds (1-300)
- **Custom Headers**: Additional HTTP headers as JSON

### Example Configuration
```json
{
  "endpoint": "http://localhost:5000/api/sensor-data",
  "api_key": "your-secret-key",
  "refresh_interval": 5,
  "headers": {
    "X-Custom-Header": "value"
  }
}
```

## 📦 Python Integration

### Using ARCSaathi as Library

```python
from ARCSaathi.models import (
    DataModel,
    MLModel,
    TaskDetectionModel
)
from ARCSaathi.controllers import (
    DataProcessingController,
    ModelTrainingController
)

# Load data
data_model = DataModel()
data_model.load_from_file("data.csv")

# Detect task
task_detector = TaskDetectionModel()
task = task_detector.detect(data_model.get_data())
print(f"Task: {task.task_type}")

# Recommend algorithms
from ARCSaathi.models import ModelRecommendationEngine
recommender = ModelRecommendationEngine()
recommendations = recommender.recommend_for_dataset(
    data_model.get_data(),
    task_type=task.task_type,
    target=task.target
)

# Print top 3 recommendations
for rec in recommendations[:3]:
    print(f"{rec.card.name}: {rec.score:.2f}")
```

## 🎯 REST API Design

When building custom APIs for integration:

### Best Practices
- Use JSON format for responses
- Include timestamp for data lineage
- Provide consistent field names
- Return meaningful error messages
- Support optional authentication
- Document all endpoints

### Sensor Data Format
```json
{
  "timestamp": "ISO-8601 format",
  "sensors": {
    "sensor_name": value,
    "another_sensor": value
  },
  "metadata": {
    "vehicle_id": "identifier",
    "source": "fleet_id"
  }
}
```

## 🔐 Authentication

### Bearer Token
```bash
curl http://api.example.com/data \\
  -H "Authorization: Bearer YOUR_TOKEN"
```

### API Key (Header)
```bash
curl http://api.example.com/data \\
  -H "X-API-Key: YOUR_KEY"
```

### No Authentication
```bash
curl http://localhost:5000/api/sensor-data
```

## ⚙️ Model Training API

```python
from ARCSaathi.models import MLModel

ml = MLModel()
success = ml.train_model(
    "reg_random_forest",
    X_train,
    y_train
)

if success:
    metrics = ml.get_model_info("reg_random_forest")
    print(f"RMSE: {metrics['metrics'].get('rmse', 'N/A')}")
```

## 📊 Report Generation API

```python
from ARCSaathi.models import PDFReportGenerator, ReportConfig

config = ReportConfig(
    title="My Analysis Report",
    author="Data Scientist"
)

generator = PDFReportGenerator(config)
generator.generate(
    "report.pdf",
    dataset_info={...},
    task_detection={...},
    recommendations=[...],
    trained_models={...}
)
```

## 🚀 Advanced Features

### Custom Algorithms
Extend the algorithm registry by modifying `model_registry.py`.

### Preprocessing Pipeline
Create custom preprocessing steps using scikit-learn transformers.

### Custom Metrics
Implement custom evaluation metrics inheriting from appropriate base classes.

## 📝 Code Examples

See the `sample_api.py` and example notebooks for complete integration examples.
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
machine learning accessible to everyone. It automatically analyzes datasets,
recommends suitable algorithms, and helps users train and compare models.

## 🚗 Automotive Focus

The system includes specialized support for automotive predictive maintenance,
enabling fleet operators to monitor component health and predict failures before
they occur.

## 🏆 Key Capabilities

- **Automatic Problem Detection**: Classification, Regression, Clustering, Time Series
- **Intelligent Recommendations**: 18 optimized algorithms with scoring
- **Smart Preprocessing**: Automated data cleaning and feature engineering
- **Model Training**: Multi-algorithm training with hyperparameter support
- **Comprehensive Evaluation**: Industry-standard metrics and visualizations
- **Professional Reports**: PDF generation with detailed analysis
- **Real-time Monitoring**: Predictive maintenance for automotive fleets
- **Explainability**: SHAP, LIME, drift detection for model transparency

## 📚 Components

### Frontend
- **PySide6**: Modern GUI framework
- **Matplotlib**: Data visualization
- **Plotly**: Interactive charts

### Backend
- **scikit-learn**: Core ML algorithms
- **pandas**: Data processing
- **numpy**: Numerical computing
- **ReportLab**: PDF generation

### Additional Integrations
- **Flask**: REST API support
- **SHAP**: Model explainability
- **LIME**: Local explanations
- **Optuna**: Hyperparameter tuning (optional)

## 👥 Development Team

Created and maintained by data scientists and ML engineers passionate about
democratizing machine learning.

## 📖 Documentation

- **Quick Start**: 7-step guide to get started
- **Features**: Comprehensive feature documentation
- **Algorithms**: Detailed descriptions of all 18 algorithms
- **Tips**: Best practices and optimization strategies
- **Troubleshooting**: Common issues and solutions
- **API Reference**: Integration and extension guide

## 🔄 Version History

### v1.0.0 (Current)
- Initial release with 18 core algorithms
- Comprehensive GUI with 8+ tabs
- PDF report generation
- Predictive maintenance module
- Real-time API integration
- Multi-format data loading
- Full explainability suite

## 📄 License

Open-source project available under appropriate license.
See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional algorithms
- More data format support
- Improved visualizations
- Performance optimizations
- Documentation improvements

## 📞 Support

For issues, feature requests, or questions:
- Check troubleshooting guide
- Review documentation
- Check code examples in repository

## 🙏 Acknowledgments

- scikit-learn team for ML algorithms
- pandas and numpy teams for data tools
- PySide6 for UI framework
- All open-source contributors

---

## 📊 Project Statistics

- **Algorithms Supported**: 18 (5 regression, 5 classification, 5 clustering, 3 dimred)
- **Sensor Types**: 15+ automotive sensors
- **UI Tabs**: 8 main tabs with numerous subtabs
- **Metrics Computed**: 50+ across all task types
- **Report Sections**: 8+ professional report sections
- **Lines of Code**: 10,000+

## 🌟 Why ARCSaathi?

**"ARCSaathi"** means "ARC's companion" - a helpful assistant for machine learning workflows.

### Philosophy
- **Accessibility**: Make ML easy for everyone
- **Intelligence**: Automate tedious analysis tasks
- **Transparency**: Explain recommendations clearly
- **Reliability**: Production-grade quality
- **Extensibility**: Easy to customize and extend

---

**Happy analyzing! 🚀**
        """)

        layout.addWidget(content)
        return widget
