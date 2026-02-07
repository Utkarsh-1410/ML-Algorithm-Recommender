# ML Algorithm Recommender (ARCSaathi) — Project Document

## 1) Problem Statement
Choosing the right machine learning approach (task type, preprocessing steps, and model family) for a new dataset is time-consuming and error-prone, especially for beginners.

**Goal:** Build an interactive desktop application that helps users:
- Load a dataset (CSV/Excel/Parquet/etc.)
- Understand dataset characteristics via profiling and visualizations
- Apply preprocessing steps safely (handling missing values, encoding, scaling, feature engineering)
- Train and compare multiple ML models for the detected/selected task
- View evaluation metrics and results dashboards
- Get recommendations and guidance on suitable algorithms

**Primary outcomes:**
- Faster and more reliable model selection
- Reduced friction in end-to-end ML workflows
- Clear, visual feedback to support learning and decision-making

## 2) Methodology
### 2.1 Architecture
The application is organized using a simple MVC-style separation:
- **Views (UI):** PySide6-based multi-tab interface for each workflow stage.
- **Controllers:** Orchestrate actions from the UI, run jobs, and coordinate models/state.
- **Models:** Handle dataset management, preprocessing logic, training, and evaluation.
- **State:** Central `AppState` tracks user selections, dataset, pipeline steps, and results.

### 2.2 Workflow Pipeline
1. **Data Loading & Validation**
   - Load dataset through UI
   - Provide clear feedback and error messages if file engines/dependencies are missing
   - Profile dataset: column types, missingness, duplicates, basic stats

2. **Preprocessing**
   - Step-based preprocessing pipeline (e.g., missing value handling, encoding/scaling)
   - Guards for problematic dtypes (e.g., boolean columns excluded from numeric transforms)
   - Step-level error reporting to show where/why a transform failed

3. **Model Training**
   - Feature sanitation before fitting:
     - Datetime → numeric representation
     - Bool → 0/1
     - Object/categorical → encoded features
   - Supports training multiple candidate models

4. **Evaluation & Comparison**
   - Metrics computed and aggregated into a results dashboard
   - Stable iteration over results to avoid mutation-related runtime crashes

5. **UX Improvements**
   - Tabs subdivided into nested tabs where needed for space and clarity
   - Plot panels are enabled by default to maintain visibility
   - Background task handling improved to avoid silent failures

### 2.3 Optional Algorithms & Dependencies
Some algorithms depend on optional libraries (e.g., xgboost, lightgbm, catboost, hdbscan, umap-learn).

To keep startup fast and stable, optional dependency availability checks are implemented as **non-import checks** (using `find_spec`) rather than importing heavy modules during initialization.

## 3) Status
### 3.1 Completed
- UI refactor: major workflow tabs subdivided into nested sub-tabs for better space utilization.
- Data loading stability improvements (clearer progress + engine checks for Excel/Parquet).
- Preprocessing dtype safety (avoids boolean arithmetic issues).
- Training feature sanitation for mixed dtypes (datetime/bool/object).
- Evaluation/dashboard stability fix (avoid dict mutation during iteration).
- Cleaned and improved the application entrypoint (`main.py`) to launch the ARCSaathi GUI.
- Startup performance improvement by preventing heavy optional dependency imports during registry checks.

### 3.2 Current State
- The project runs via the root entrypoint:
  - `python main.py`
- Core dependencies are checkable via:
  - `python main.py --diagnose`

### 3.3 Pending / Next Steps
- Finalize any remaining legacy-file cleanup (if applicable) and ensure docs reflect the current ARCSaathi entrypoint.
- Add optional “extras” install instructions (e.g., `pip install .[extras]` or a dedicated requirements file) for algorithms like UMAP/HDBSCAN.
- Add a small smoke-test (optional) that imports the app and builds the UI without launching it.
