# Predictive Maintenance Integration - Enhanced Version

This document describes the enhanced Predictive Maintenance tab integrated into the ML Algorithm Recommender (ARCSaathi) application.

## 🚀 New Features

### 1. 📚 Training Data Management
- **Multi-file Upload**: Add multiple CSV training files to improve model accuracy
- **Training Files Log**: Visual list of all uploaded training files with file info
- **Remove Button (✕)**: Click to remove any file from the training set
- **Continuous Training**: Model combines all uploaded files for better predictions

### 2. 🔌 API Integration for Real-Time Data
- **Flask API Support**: Connect to any REST API endpoint for sensor data
- **Configurable Settings**:
  - API Endpoint URL
  - API Key (Bearer token)
  - Refresh Interval (1-300 seconds)
  - Custom Headers (JSON format)
- **Connection Controls**: Start/Stop/Test API connection
- **Activity Log**: Real-time log of API operations

### 3. 📊 Dynamic Real-Time Interface
- **Auto-Refresh**: UI updates every second with latest predictions
- **Live Status**: Shows last update time, data source, and model status
- **Real-Time Alerts**: Instant critical/warning alerts based on RUL

---

## 📱 Tab Structure

The Predictive Maintenance tab now contains 5 sub-tabs:

| Tab | Description |
|-----|-------------|
| 📚 Training Data | Upload training files, view log, train model |
| 🔌 API Config | Configure API endpoint for real-time data |
| 📊 Component Health | Visual RUL bars for each sensor category |
| 📈 Sensor Details | Metric cards showing current sensor values |
| 🔍 System Overview | Alerts, model metrics, and status |

---

## 🔧 How to Use

### Training a Model

1. Go to **"📚 Training Data"** tab
2. Click **"➕ Add CSV File"** to add training data files
3. View added files in the log (shows filename and row count)
4. Remove unwanted files by clicking the **"✕"** button
5. Click **"🚀 Train Model"** to start training
6. Monitor progress in the training log

### Connecting to API

1. Go to **"🔌 API Config"** tab
2. Enter your API endpoint URL (e.g., `http://localhost:5000/api/sensor-data`)
3. (Optional) Add API key and custom headers
4. Set refresh interval (how often to fetch data)
5. Click **"🧪 Test Connection"** to verify
6. Click **"▶️ Start Fetching"** to begin real-time updates

### Viewing Predictions

- **Component Health**: See RUL percentage for each sensor category
- **Sensor Details**: View individual sensor values with status
- **System Overview**: Monitor alerts and model performance metrics

---

## 📡 API Data Format

The API should return JSON data in this format:

```json
[
  {
    "injector_pressure": 10.5,
    "oil_pressure": 0.25,
    "coolant_pressure": 0.12,
    "oil_temperature": 85.2,
    "ferrous_debris": 15.3,
    "soot_in_oil": 45.6,
    "cylinder_head_temp": 95.0,
    "exhaust_gas_temp": 550.0,
    "bearing_temp": 65.5,
    "engine_vibration": 2.5,
    "knock_sensor": 25.0,
    "crankshaft_vibration": 450.0,
    "mass_air_flow": 5.5,
    "oxygen_sensor": 0.98,
    "egr_flow": 10.0
  }
]
```

### Required Sensor Fields

| Category | Sensors |
|----------|---------|
| Wear & Degradation | `ferrous_debris`, `soot_in_oil` |
| Temperature & Thermal | `cylinder_head_temp`, `exhaust_gas_temp`, `bearing_temp` |
| Vibration & Mechanical | `engine_vibration`, `knock_sensor`, `crankshaft_vibration` |
| Fluid & Pressure | `oil_temperature`, `injector_pressure`, `oil_pressure`, `coolant_pressure` |
| Air & Combustion | `mass_air_flow`, `oxygen_sensor`, `egr_flow` |

---

## 🧪 Testing with Sample API

A sample Flask API is provided for testing:

```bash
# Install dependencies
pip install flask flask-cors

# Run the sample API
cd ML-Algorithm-Recommender
python sample_api.py
```

Then configure in the app:
- **API URL**: `http://localhost:5000/api/sensor-data`
- **Refresh Interval**: 5 seconds

---

## 📁 File Structure

```
ML-Algorithm-Recommender/
├── ARCSaathi/
│   ├── views/tabs/
│   │   └── predictive_maintenance_tab.py  # Main tab implementation
│   ├── predictive_maintenance_model/      # Trained model files
│   │   ├── model.joblib
│   │   ├── scaler.joblib
│   │   └── feature_list.joblib
│   └── predictive_maintenance_data/       # Sample data
│       └── fluid_sensor_data.csv
├── sample_api.py                          # Sample Flask API
└── PREDICTIVE_MAINTENANCE_INTEGRATION.md  # This documentation
```

---

## 🔄 Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Upload CSV(s)  │────▶│  Train Model    │────▶│  Model Saved    │
│  for Training   │     │  (Background)   │     │  (.joblib)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Configure API  │────▶│  Fetch Data     │────▶│  Make Predictions│
│  Endpoint       │     │  (Real-time)    │     │  Update UI       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 📋 Requirements

All dependencies are included in `ARCSaathi/requirements.txt`:
- PySide6 (GUI)
- pandas, numpy (Data processing)
- scikit-learn (Preprocessing)
- xgboost (Model)
- joblib (Model serialization)
- requests (API calls)

For the sample API:
- flask
- flask-cors
