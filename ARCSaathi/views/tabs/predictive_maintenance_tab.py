"""Tab 7: Predictive Maintenance - Engine Health Monitoring Dashboard.

Features:
- Automotive-style digital gauge visualization
- Real-time data streaming from API
- Continuous training with file management
- Zero-latency UI updates
"""

from __future__ import annotations

import os
import json
import math
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal, QThread, QObject, QTimer, QPropertyAnimation, Property, QEasingCurve, QRectF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QStackedWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QRadialGradient, QPainterPath, QConicalGradient

warnings.filterwarnings('ignore')

# ============================================================================
# Sensor Categories and Thresholds
# ============================================================================
SENSOR_CATEGORIES = {
    "Wear & Degradation": {
        "ferrous_debris": {"min": 0, "max": 100, "unit": "μm", "warning": (20, 99), "critical": 100},
        "soot_in_oil": {"min": 0, "max": 140, "unit": "mg/L", "warning": 10000, "critical": 20000}
    },
    "Temperature & Thermal": {
        "cylinder_head_temp": {"min": 70, "max": 130, "unit": "°C", "warning": (110, 129), "critical": 130},
        "exhaust_gas_temp": {"min": 250, "max": 850, "unit": "°C", "warning": (650, 849), "critical": 850},
        "bearing_temp": {"min": 30, "max": 110, "unit": "°C", "warning": (85, 109), "critical": 110}
    },
    "Vibration & Mechanical": {
        "engine_vibration": {"min": 0, "max": 10, "unit": "g", "warning": (5, 9), "critical": 10},
        "knock_sensor": {"min": 0, "max": 100, "unit": "%", "warning": 70, "critical": 85},
        "crankshaft_vibration": {"min": 200, "max": 1000, "unit": "Ω", "warning": [(200, 250), (900, 1000)], "critical": [(0, 200), (1000, float('inf'))]}
    },
    "Fluid & Pressure": {
        "oil_temperature": {"min": 0, "max": 120, "unit": "°C", "warning": 15000, "critical": 15000},
        "injector_pressure": {"min": 1, "max": 20, "unit": "MPa", "warning": 10, "critical": 20},
        "oil_pressure": {"min": 0, "max": 0.5, "unit": "MPa", "warning": (0.2, 0.29), "critical": 0.3},
        "coolant_pressure": {"min": 0, "max": 0.15, "unit": "MPa", "warning": (0.10, 0.15), "critical": [(0, 0.09), (0.25, float('inf'))]}
    },
    "Air & Combustion": {
        "mass_air_flow": {"min": 0.2, "max": 10, "unit": "m/s", "warning": 3.3, "critical": 4.4},
        "oxygen_sensor": {"min": 0, "max": 1.1, "unit": "λ", "warning": (0.90, 0.95), "critical": 0.89},
        "egr_flow": {"min": 0, "max": 20, "unit": "%", "warning": (15, 19), "critical": 20}
    }
}

INPUT_FEATURES = [
    'injector_pressure', 'oil_pressure', 'coolant_pressure', 'oil_temperature',
    'ferrous_debris', 'soot_in_oil', 'cylinder_head_temp', 'exhaust_gas_temp',
    'bearing_temp', 'engine_vibration', 'knock_sensor', 'crankshaft_vibration',
    'mass_air_flow', 'oxygen_sensor', 'egr_flow'
]

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if all(col in df.columns for col in ['cylinder_head_temp', 'exhaust_gas_temp', 'bearing_temp']):
        df['temp_diff_head_exhaust'] = df['exhaust_gas_temp'] - df['cylinder_head_temp']
        df['temp_diff_head_bearing'] = df['cylinder_head_temp'] - df['bearing_temp']
    if all(col in df.columns for col in ['oil_pressure', 'coolant_pressure']):
        df['oil_coolant_pressure_ratio'] = df['oil_pressure'] / (df['coolant_pressure'] + 1e-6)
    if all(col in df.columns for col in ['engine_vibration', 'crankshaft_vibration']):
        df['vibration_composite'] = (df['engine_vibration'] + df['crankshaft_vibration']) / 2
    if all(col in df.columns for col in ['oil_temperature', 'ferrous_debris', 'soot_in_oil']):
        df['wear_composite'] = (100 - df['oil_temperature']) * 0.5 + df['ferrous_debris'] * 0.3 + df['soot_in_oil'] * 0.2
    return df

# ============================================================================
# Digital Gauge Widget (Automotive Style)
# ============================================================================
class DigitalGauge(QWidget):
    """Automotive-style digital gauge with animated needle and LED segments."""
    
    def __init__(self, title: str = "", min_val: float = 0, max_val: float = 100, unit: str = "%", parent=None):
        super().__init__(parent)
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self._value = min_val
        self._target_value = min_val
        self._health = 100.0
        self.setMinimumSize(180, 180)
        self.setMaximumSize(220, 220)
        
        # Animation
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animate)
        self._animation_timer.start(16)  # ~60 FPS
    
    def _animate(self):
        diff = self._target_value - self._value
        if abs(diff) > 0.1:
            self._value += diff * 0.15
            self.update()
    
    def set_value(self, value: float, health: float = 100.0):
        self._target_value = max(self.min_val, min(self.max_val, value))
        self._health = max(0, min(100, health))
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 15
        
        # Background circle with gradient
        bg_gradient = QRadialGradient(cx, cy, radius)
        bg_gradient.setColorAt(0, QColor(40, 40, 50))
        bg_gradient.setColorAt(0.7, QColor(25, 25, 35))
        bg_gradient.setColorAt(1, QColor(15, 15, 25))
        painter.setBrush(bg_gradient)
        painter.setPen(QPen(QColor(60, 60, 80), 3))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        
        # LED arc segments
        arc_radius = radius - 8
        start_angle = 225
        span_angle = -270
        
        # Draw colored arc based on health
        for i in range(27):
            segment_start = start_angle - (i * 10)
            segment_value = (i / 27.0) * 100
            
            if segment_value <= self._health:
                if self._health > 60:
                    color = QColor(0, 255, 100, 200)
                elif self._health > 30:
                    color = QColor(255, 200, 0, 200)
                else:
                    color = QColor(255, 50, 50, 200)
            else:
                color = QColor(50, 50, 60, 100)
            
            painter.setPen(QPen(color, 6, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(cx - arc_radius, cy - arc_radius, arc_radius * 2, arc_radius * 2, 
                          int(segment_start * 16), int(-8 * 16))
        
        # Value text (large digital display)
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Consolas", 20, QFont.Bold)
        painter.setFont(font)
        value_text = f"{self._value:.1f}"
        painter.drawText(QRectF(cx - 50, cy - 15, 100, 30), Qt.AlignCenter, value_text)
        
        # Unit text
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor(150, 150, 170))
        painter.drawText(QRectF(cx - 30, cy + 15, 60, 20), Qt.AlignCenter, self.unit)
        
        # Title
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor(100, 150, 255))
        painter.drawText(QRectF(0, h - 25, w, 20), Qt.AlignCenter, self.title)
        
        # Health indicator LED
        led_color = QColor(0, 255, 100) if self._health > 60 else QColor(255, 200, 0) if self._health > 30 else QColor(255, 50, 50)
        painter.setBrush(led_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx - 5, cy + 35, 10, 10)

# ============================================================================
# Horizontal Bar Gauge (LED Style)
# ============================================================================
class LEDBarGauge(QWidget):
    """LED-style horizontal bar gauge."""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self._value = 0.0
        self._target = 0.0
        self.setMinimumHeight(50)
        self.setMaximumHeight(60)
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(16)
    
    def _animate(self):
        diff = self._target - self._value
        if abs(diff) > 0.5:
            self._value += diff * 0.2
            self.update()
    
    def set_value(self, value: float):
        self._target = max(0, min(100, value))
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        bar_height = 20
        bar_y = (h - bar_height) // 2 + 5
        margin = 10
        bar_width = w - 2 * margin
        
        # Title
        painter.setPen(QColor(180, 180, 200))
        font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font)
        painter.drawText(margin, bar_y - 5, self.title)
        
        # Value percentage
        painter.drawText(w - 50, bar_y - 5, f"{self._value:.0f}%")
        
        # Background bar
        painter.setBrush(QColor(30, 30, 40))
        painter.setPen(QPen(QColor(60, 60, 80), 1))
        painter.drawRoundedRect(margin, bar_y, bar_width, bar_height, 5, 5)
        
        # LED segments
        num_segments = 25
        segment_width = (bar_width - 10) / num_segments
        filled = int((self._value / 100) * num_segments)
        
        for i in range(num_segments):
            x = margin + 5 + i * segment_width
            if i < filled:
                if self._value > 60:
                    color = QColor(0, 255, 100)
                elif self._value > 30:
                    color = QColor(255, 200, 0)
                else:
                    color = QColor(255, 50, 50)
            else:
                color = QColor(40, 40, 50)
            
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(int(x), bar_y + 3, int(segment_width - 2), bar_height - 6, 2, 2)

# ============================================================================
# Status LED Widget
# ============================================================================
class StatusLED(QWidget):
    """Glowing LED status indicator."""
    
    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        self.label = label
        self._status = "normal"  # normal, warning, critical
        self._blink = False
        self._blink_state = True
        self.setFixedSize(100, 30)
        
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)
    
    def _toggle_blink(self):
        self._blink_state = not self._blink_state
        self.update()
    
    def set_status(self, status: str):
        self._status = status
        if status == "critical":
            self._blink_timer.start(500)
        else:
            self._blink_timer.stop()
            self._blink_state = True
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # LED colors
        colors = {
            "normal": QColor(0, 255, 100),
            "warning": QColor(255, 200, 0),
            "critical": QColor(255, 50, 50)
        }
        
        color = colors.get(self._status, colors["normal"])
        if self._status == "critical" and not self._blink_state:
            color = QColor(100, 30, 30)
        
        # Draw LED with glow
        led_x, led_y = 5, 8
        led_size = 14
        
        # Glow effect
        glow = QRadialGradient(led_x + led_size//2, led_y + led_size//2, led_size)
        glow.setColorAt(0, color)
        glow.setColorAt(0.5, QColor(color.red(), color.green(), color.blue(), 100))
        glow.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
        painter.setBrush(glow)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(led_x - 3, led_y - 3, led_size + 6, led_size + 6)
        
        # LED core
        painter.setBrush(color)
        painter.drawEllipse(led_x, led_y, led_size, led_size)
        
        # Label
        painter.setPen(QColor(200, 200, 220))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(led_x + led_size + 8, led_y + 11, self.label)

# ============================================================================
# Digital Readout Widget
# ============================================================================
class DigitalReadout(QFrame):
    """Digital LCD-style readout display."""
    
    def __init__(self, title: str = "", unit: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self._value = 0.0
        self._health = 100.0
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame { background-color: #1a1a2e; border: 2px solid #3a3a5a; border-radius: 8px; }
        """)
        self.setMinimumSize(140, 80)
        self.setMaximumSize(180, 100)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(2)
        
        # Title
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: #6080ff; font-size: 10px; font-weight: bold; border: none;")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        
        # Value display
        self.lbl_value = QLabel("--")
        self.lbl_value.setStyleSheet("color: #00ff88; font-size: 22px; font-weight: bold; font-family: 'Consolas'; border: none;")
        self.lbl_value.setAlignment(Qt.AlignCenter)
        
        # Unit and status
        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet("color: #888; font-size: 9px; border: none;")
        self.lbl_unit.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_value)
        layout.addWidget(self.lbl_unit)
    
    def set_value(self, value: float, health: float = 100.0):
        self._value = value
        self._health = health
        self.lbl_value.setText(f"{value:.2f}")
        
        if health > 60:
            self.lbl_value.setStyleSheet("color: #00ff88; font-size: 22px; font-weight: bold; font-family: 'Consolas'; border: none;")
        elif health > 30:
            self.lbl_value.setStyleSheet("color: #ffcc00; font-size: 22px; font-weight: bold; font-family: 'Consolas'; border: none;")
        else:
            self.lbl_value.setStyleSheet("color: #ff4444; font-size: 22px; font-weight: bold; font-family: 'Consolas'; border: none;")

# ============================================================================
# API Data Fetcher
# ============================================================================
class APIDataFetcher(QObject):
    data_received = Signal(pd.DataFrame)
    error_occurred = Signal(str)
    status_update = Signal(str)
    
    def __init__(self, api_url: str, api_key: str = "", headers: dict = None):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.headers = headers or {}
        self._running = True
    
    def fetch(self):
        if not self._running or not self.api_url:
            return
        try:
            headers = self.headers.copy()
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            headers.setdefault('Content-Type', 'application/json')
            
            response = requests.get(self.api_url, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # Debug: log raw response structure
            if isinstance(data, list):
                if len(data) > 0:
                    self.status_update.emit(f"API returned list with {len(data)} items, first item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'not dict'}")
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    # Single sensor reading as dict
                    self.status_update.emit(f"API returned dict with keys: {list(data.keys())}")
                    df = pd.DataFrame([data])
            else:
                self.error_occurred.emit(f"Unexpected data type: {type(data)}")
                return
            
            # Log DataFrame info
            self.status_update.emit(f"DataFrame: {len(df)} rows, {len(df.columns)} cols: {list(df.columns)[:5]}...")
            
            self.data_received.emit(df)
        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            self.error_occurred.emit(f"JSON parse error: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
    
    def stop(self):
        self._running = False


# ============================================================================
# Model Training Worker
# ============================================================================
class ModelTrainingWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    log = Signal(str)
    
    def __init__(self, data_files: List[str]):
        super().__init__()
        self.data_files = data_files
    
    def run(self):
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from xgboost import XGBRegressor
            
            self.log.emit(f"Loading {len(self.data_files)} file(s)...")
            self.progress.emit(10)
            
            dfs = [pd.read_csv(fp) for fp in self.data_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            self.log.emit(f"Dataset: {len(combined_df)} rows")
            
            if 'RUL' not in combined_df.columns:
                self.error.emit("Missing 'RUL' column")
                return
            
            self.progress.emit(30)
            combined_df = create_derived_features(combined_df)
            
            all_features = [f for f in INPUT_FEATURES if f in combined_df.columns]
            all_features += [c for c in combined_df.columns if c not in INPUT_FEATURES + ['RUL']]
            
            X, y = combined_df[all_features], combined_df['RUL']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            split = int(0.8 * len(X_scaled))
            X_train, X_test = X_scaled[:split], X_scaled[split:]
            y_train, y_test = y[:split], y[split:]
            
            self.progress.emit(50)
            self.log.emit("Training model...")
            
            model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            self.progress.emit(85)
            
            metrics = {
                'train': {'r2': r2_score(y_train, model.predict(X_train))},
                'test': {'r2': r2_score(y_test, model.predict(X_test))}
            }
            
            import joblib
            model_dir = Path(__file__).parent.parent.parent / 'predictive_maintenance_model'
            model_dir.mkdir(exist_ok=True)
            joblib.dump(model, model_dir / 'model.joblib', compress=3)
            joblib.dump(scaler, model_dir / 'scaler.joblib', compress=3)
            joblib.dump(all_features, model_dir / 'feature_list.joblib', compress=3)
            
            self.progress.emit(100)
            self.log.emit(f"Complete! R²: {metrics['test']['r2']:.4f}")
            self.finished.emit({'metrics': metrics, 'samples': len(combined_df)})
        except Exception as e:
            self.error.emit(str(e))

# ============================================================================
# Training File Item
# ============================================================================
class TrainingFileItem(QFrame):
    remove_requested = Signal(str)
    
    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setStyleSheet("QFrame { background-color: #2a2a3a; border-radius: 4px; margin: 2px; }")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        lbl = QLabel(Path(file_path).name)
        lbl.setStyleSheet("color: #fff; font-size: 11px;")
        lbl.setToolTip(file_path)
        
        btn = QPushButton("✕")
        btn.setFixedSize(20, 20)
        btn.setStyleSheet("QPushButton { background-color: #ff4444; color: white; border-radius: 10px; font-weight: bold; } QPushButton:hover { background-color: #ff6666; }")
        btn.clicked.connect(lambda: self.remove_requested.emit(self.file_path))
        
        layout.addWidget(lbl, 1)
        layout.addWidget(btn)

# ============================================================================
# Main Predictive Maintenance Tab
# ============================================================================
class PredictiveMaintenanceTab(QWidget):
    model_trained = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.scaler = None
        self.feature_list = None
        self.training_files: List[str] = []
        self.training_thread: Optional[QThread] = None
        self.api_timer: Optional[QTimer] = None
        self.api_fetcher: Optional[APIDataFetcher] = None
        
        self._setup_ui()
        self._load_model()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("ENGINE HEALTH MONITORING SYSTEM")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #E5E7EB; padding: 12px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a1a2e, stop:1 #16213e);")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #3a3a5a; background: #1a1a2e; }
            QTabBar::tab { background: #2a2a3a; color: #888; padding: 10px 20px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #1a1a2e; color: #00ddff; border-bottom: 2px solid #00ddff; }
        """)
        
        self.tabs.addTab(self._create_dashboard_tab(), "DASHBOARD")
        self.tabs.addTab(self._create_sensors_tab(), "SENSORS")
        self.tabs.addTab(self._create_training_tab(), "TRAINING")
        self.tabs.addTab(self._create_api_tab(), "API")
        
        main_layout.addWidget(self.tabs, 1)
    
    def _create_dashboard_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("background-color: #0f0f1a;")
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Main RUL gauge row
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(20)
        
        self.main_gauge = DigitalGauge("SYSTEM RUL", 0, 100, "%")
        self.main_gauge.setMinimumSize(200, 200)
        
        # Category health bars
        bars_layout = QVBoxLayout()
        self.category_bars: Dict[str, LEDBarGauge] = {}
        for category in SENSOR_CATEGORIES.keys():
            bar = LEDBarGauge(category)
            self.category_bars[category] = bar
            bars_layout.addWidget(bar)
        
        gauge_row.addWidget(self.main_gauge)
        gauge_row.addLayout(bars_layout, 1)
        layout.addLayout(gauge_row)
        
        # Status row
        status_row = QHBoxLayout()
        self.status_leds: Dict[str, StatusLED] = {}
        for name in ["ENGINE", "THERMAL", "PRESSURE", "VIBRATION", "AIR"]:
            led = StatusLED(name)
            self.status_leds[name] = led
            status_row.addWidget(led)
        status_row.addStretch()
        
        # Time and connection status
        self.lbl_time = QLabel("--:--:--")
        self.lbl_time.setStyleSheet("color: #00ff88; font-size: 14px; font-family: Consolas;")
        self.lbl_connection = QLabel("⚪ OFFLINE")
        self.lbl_connection.setStyleSheet("color: #888; font-size: 11px;")
        status_row.addWidget(self.lbl_time)
        status_row.addWidget(self.lbl_connection)
        
        layout.addLayout(status_row)
        
        # Alert display
        self.alert_display = QLabel(" ALL SYSTEMS NOMINAL")
        self.alert_display.setStyleSheet("color: #00ff88; font-size: 16px; font-weight: bold; padding: 15px; background-color: #1a2a1a; border-radius: 8px; border: 1px solid #2a4a2a;")
        self.alert_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.alert_display)
        
        # Update timer
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_time)
        self.ui_timer.start(100)
        
        return widget
    
    def _create_sensors_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("background-color: #0f0f1a;")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #0f0f1a; }")
        
        content = QWidget()
        content.setStyleSheet("background-color: #0f0f1a;")
        layout = QVBoxLayout(content)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Initialize sensor_readouts FIRST
        self.sensor_readouts: Dict[str, Dict[str, DigitalReadout]] = {}
        
        for category, sensors in SENSOR_CATEGORIES.items():
            # Create group box with visible styling
            group = QGroupBox(f" {category}")
            group.setStyleSheet("""
                QGroupBox { 
                    font-size: 14px; 
                    font-weight: bold; 
                    color: #00ddff; 
                    border: 2px solid #3a3a5a; 
                    border-radius: 10px; 
                    margin-top: 15px; 
                    padding: 20px 10px 10px 10px;
                    background-color: #1a1a2e; 
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 5px 10px;
                    background-color: #1a1a2e;
                    border-radius: 5px;
                }
            """)
            
            grid = QGridLayout(group)
            grid.setSpacing(15)
            grid.setContentsMargins(10, 25, 10, 10)
            
            self.sensor_readouts[category] = {}
            col_count = min(4, len(sensors))  # Max 4 columns
            
            for idx, (sensor, specs) in enumerate(sensors.items()):
                readout = DigitalReadout(sensor.replace('_', ' ').upper(), specs['unit'])
                readout.setMinimumSize(150, 90)
                self.sensor_readouts[category][sensor] = readout
                grid.addWidget(readout, idx // col_count, idx % col_count)
            
            layout.addWidget(group)
        
        layout.addStretch()
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(scroll)
        return widget
    
    def _create_training_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("background-color: #0f0f1a;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Buttons
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("➕ ADD FILE")
        self.btn_add.setStyleSheet("QPushButton { background-color: #1976D2; color: white; padding: 10px 20px; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #2196F3; }")
        self.btn_add.clicked.connect(self._add_file)
        
        self.btn_train = QPushButton("TRAIN MODEL")
        self.btn_train.setEnabled(False)
        self.btn_train.setStyleSheet("QPushButton { background-color: #388E3C; color: white; padding: 10px 20px; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #4CAF50; } QPushButton:disabled { background-color: #555; }")
        self.btn_train.clicked.connect(self._train)
        
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_train)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        # Files list
        self.files_container = QWidget()
        self.files_layout = QVBoxLayout(self.files_container)
        self.files_layout.setContentsMargins(0, 0, 0, 0)
        self.files_layout.addStretch()
        
        files_scroll = QScrollArea()
        files_scroll.setWidgetResizable(True)
        files_scroll.setWidget(self.files_container)
        files_scroll.setMaximumHeight(200)
        files_scroll.setStyleSheet("QScrollArea { border: 1px solid #3a3a5a; border-radius: 6px; background-color: #1a1a2e; }")
        layout.addWidget(files_scroll)
        
        # Progress
        self.train_progress = QProgressBar()
        self.train_progress.setStyleSheet("QProgressBar { background-color: #2a2a3a; border-radius: 4px; height: 20px; } QProgressBar::chunk { background-color: #00ff88; border-radius: 4px; }")
        layout.addWidget(self.train_progress)
        
        # Log
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("QTextEdit { background-color: #0a0a15; color: #00ff88; font-family: Consolas; font-size: 11px; border: 1px solid #3a3a5a; border-radius: 6px; }")
        layout.addWidget(self.train_log, 1)
        
        return widget
    
    def _create_api_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("background-color: #0f0f1a;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Config
        form = QFormLayout()
        form.setSpacing(15)
        
        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText("")
        self.txt_url.setStyleSheet("QLineEdit { background-color: #2a2a3a; color: #fff; border: 1px solid #4a4a6a; border-radius: 4px; padding: 8px; }")
        
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 60)
        self.spin_interval.setValue(3)
        self.spin_interval.setSuffix(" sec")
        self.spin_interval.setStyleSheet("QSpinBox { background-color: #2a2a3a; color: #fff; border: 1px solid #4a4a6a; border-radius: 4px; padding: 8px; }")
        
        lbl1 = QLabel("API URL:")
        lbl1.setStyleSheet("color: #aaa;")
        lbl2 = QLabel("Interval:")
        lbl2.setStyleSheet("color: #aaa;")
        
        form.addRow(lbl1, self.txt_url)
        form.addRow(lbl2, self.spin_interval)
        layout.addLayout(form)
        
        # Controls
        ctrl_row = QHBoxLayout()
        self.btn_start = QPushButton("START")
        self.btn_start.setStyleSheet("QPushButton { background-color: #388E3C; color: white; padding: 10px 20px; border-radius: 6px; font-weight: bold; }")
        self.btn_start.clicked.connect(self._start_api)
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton { background-color: #d32f2f; color: white; padding: 10px 20px; border-radius: 6px; font-weight: bold; } QPushButton:disabled { background-color: #555; }")
        self.btn_stop.clicked.connect(self._stop_api)
        
        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)
        
        # Log
        self.api_log = QTextEdit()
        self.api_log.setReadOnly(True)
        self.api_log.setStyleSheet("QTextEdit { background-color: #0a0a15; color: #00ddff; font-family: Consolas; font-size: 11px; border: 1px solid #3a3a5a; border-radius: 6px; }")
        layout.addWidget(self.api_log, 1)
        
        return widget
    
    def _load_model(self):
        try:
            import joblib
            model_dir = Path(__file__).parent.parent.parent / 'predictive_maintenance_model'
            if (model_dir / 'model.joblib').exists():
                self.model = joblib.load(model_dir / 'model.joblib')
                self.scaler = joblib.load(model_dir / 'scaler.joblib')
                self.feature_list = joblib.load(model_dir / 'feature_list.joblib')
                self._log_train("Model loaded successfully")
        except Exception as e:
            self._log_train(f"Model load error: {e}")
    
    def _update_time(self):
        self.lbl_time.setText(datetime.now().strftime("%H:%M:%S"))
    
    def _add_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select CSV", "", "CSV (*.csv)")
        for fp in files:
            if fp not in self.training_files:
                self.training_files.append(fp)
                item = TrainingFileItem(fp)
                item.remove_requested.connect(self._remove_file)
                self.files_layout.insertWidget(self.files_layout.count() - 1, item)
        self.btn_train.setEnabled(len(self.training_files) > 0)
    
    def _remove_file(self, fp: str):
        if fp in self.training_files:
            self.training_files.remove(fp)
        for i in range(self.files_layout.count()):
            w = self.files_layout.itemAt(i).widget()
            if isinstance(w, TrainingFileItem) and w.file_path == fp:
                w.deleteLater()
                break
        self.btn_train.setEnabled(len(self.training_files) > 0)
    
    def _train(self):
        if not self.training_files:
            return
        self.btn_train.setEnabled(False)
        self.train_progress.setValue(0)
        self.train_log.clear()
        
        self.training_thread = QThread()
        self.worker = ModelTrainingWorker(self.training_files.copy())
        self.worker.moveToThread(self.training_thread)
        
        self.training_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.train_progress.setValue)
        self.worker.log.connect(self._log_train)
        self.worker.finished.connect(self._on_train_done)
        self.worker.error.connect(self._on_train_error)
        self.worker.finished.connect(self.training_thread.quit)
        self.worker.error.connect(self.training_thread.quit)
        
        self.training_thread.start()
    
    def _on_train_done(self, results):
        self.btn_train.setEnabled(True)
        self._load_model()
        QMessageBox.information(self, "Success", f"Model trained! R²: {results['metrics']['test']['r2']:.4f}")
    
    def _on_train_error(self, err):
        self.btn_train.setEnabled(True)
        self._log_train(f"ERROR: {err}")
        QMessageBox.critical(self, "Error", err)
    
    def _log_train(self, msg):
        self.train_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    def _start_api(self):
        url = self.txt_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Enter API URL")
            return
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_connection.setText("🟢 ONLINE")
        self.lbl_connection.setStyleSheet("color: #00ff88;")
        
        self.api_fetcher = APIDataFetcher(url)
        self.api_fetcher.data_received.connect(self._on_data)
        self.api_fetcher.error_occurred.connect(self._on_api_error)
        self.api_fetcher.status_update.connect(self._log_api)  # Connect status updates
        
        self.api_timer = QTimer(self)
        self.api_timer.timeout.connect(self.api_fetcher.fetch)
        self.api_timer.start(self.spin_interval.value() * 1000)
        self.api_fetcher.fetch()
        self._log_api(f"Connected to {url}")
    
    def _stop_api(self):
        if self.api_timer:
            self.api_timer.stop()
        if self.api_fetcher:
            self.api_fetcher.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_connection.setText("⚪ OFFLINE")
        self.lbl_connection.setStyleSheet("color: #888;")
        self._log_api("Disconnected")
    
    def _on_data(self, df: pd.DataFrame):
        self._log_api(f"Received {len(df)} records")
        self._update_dashboard(df)
    
    def _on_api_error(self, err):
        self._log_api(f"ERROR: {err}")
        self.lbl_connection.setText("🔴 ERROR")
        self.lbl_connection.setStyleSheet("color: #ff4444;")
    
    def _log_api(self, msg):
        self.api_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    def _calc_health(self, sensor: str, value: float, specs: dict) -> float:
        """Calculate health percentage based on sensor value and thresholds."""
        min_v = specs.get('min', 0)
        max_v = specs.get('max', 100)
        warning = specs.get('warning')
        critical = specs.get('critical')
        
        # Check critical thresholds first
        if critical is not None:
            if isinstance(critical, (int, float)):
                if value >= critical:
                    return max(5, 25 - (value - critical) * 5)
            elif isinstance(critical, list):
                for rng in critical:
                    if isinstance(rng, tuple) and len(rng) == 2:
                        if rng[0] <= value <= rng[1]:
                            return 15
        
        # Check warning thresholds
        if warning is not None:
            if isinstance(warning, tuple) and len(warning) == 2:
                if warning[0] <= value <= warning[1]:
                    return 45
            elif isinstance(warning, (int, float)):
                if value >= warning:
                    return 40
        
        # Normal range - calculate based on position in range
        rng = max_v - min_v if max_v != min_v else 1
        mid = (min_v + max_v) / 2
        
        # Closer to middle = better health
        deviation = abs(value - mid) / (rng / 2) if rng > 0 else 0
        health = 100 - (deviation * 35)
        return max(20, min(100, health))
    
    def _update_dashboard(self, df: pd.DataFrame):
        """Update all dashboard widgets with real-time data."""
        if df is None or df.empty:
            self._log_api("No data to display")
            return
        
        try:
            # Get the latest row
            row = df.iloc[-1]
            available_cols = list(row.index)
            
            # Log available columns for debugging
            expected_sensors = []
            for cat, sensors in SENSOR_CATEGORIES.items():
                expected_sensors.extend(sensors.keys())
            
            matching = [s for s in expected_sensors if s in available_cols]
            missing = [s for s in expected_sensors if s not in available_cols]
            
            if len(matching) == 0:
                self._log_api(f"NO matching sensors! Available: {available_cols[:5]}...")
                self._log_api(f"Expected sensors like: {expected_sensors[:3]}...")
                return
            
            self._log_api(f"Found {len(matching)}/{len(expected_sensors)} sensors")
            
            # ================================================================
            # MODEL-BASED RUL PREDICTION (if model is available)
            # ================================================================
            model_rul = None
            if self.model is not None and self.scaler is not None and self.feature_list is not None:
                try:
                    # Prepare features for model prediction
                    input_df = pd.DataFrame([row])
                    input_df = create_derived_features(input_df)
                    
                    # Get only the features the model was trained on
                    available_features = [f for f in self.feature_list if f in input_df.columns]
                    
                    if len(available_features) >= len(self.feature_list) * 0.7:  # At least 70% features
                        # Fill missing features with median values (0)
                        for f in self.feature_list:
                            if f not in input_df.columns:
                                input_df[f] = 0
                        
                        # Extract features in correct order
                        X = input_df[self.feature_list].values
                        
                        # Scale features
                        X_scaled = self.scaler.transform(X)
                        
                        # Predict RUL
                        model_rul = float(self.model.predict(X_scaled)[0])
                        model_rul = max(0, min(100, model_rul))  # Clamp to 0-100
                        self._log_api(f"🤖 Model RUL prediction: {model_rul:.1f}%")
                    else:
                        self._log_api(f"Only {len(available_features)}/{len(self.feature_list)} features available for model")
                except Exception as e:
                    self._log_api(f"Model prediction error: {str(e)}")
            
            # ================================================================
            # HEURISTIC-BASED HEALTH CALCULATION (always computed)
            # ================================================================
            category_health = {}
            sensors_updated = 0
            
            for category, sensors in SENSOR_CATEGORIES.items():
                healths = []
                
                for sensor_name, specs in sensors.items():
                    if sensor_name in row.index:
                        try:
                            val = float(row[sensor_name])
                            health = self._calc_health(sensor_name, val, specs)
                            healths.append(health)
                            
                            # Update sensor readout widget
                            if hasattr(self, 'sensor_readouts'):
                                if category in self.sensor_readouts:
                                    if sensor_name in self.sensor_readouts[category]:
                                        widget = self.sensor_readouts[category][sensor_name]
                                        widget.set_value(val, health)
                                        widget.update()
                                        widget.repaint()
                                        sensors_updated += 1
                        except (ValueError, TypeError):
                            pass
                
                cat_health = np.mean(healths) if healths else 50.0
                category_health[category] = cat_health
                
                if hasattr(self, 'category_bars') and category in self.category_bars:
                    self.category_bars[category].set_value(cat_health)
                    self.category_bars[category].update()
                    self.category_bars[category].repaint()
            
            # ================================================================
            # SYSTEM RUL: Average of all 5 category component values
            # ================================================================
            # Calculate the average of the 5 category bars
            system_rul = np.mean(list(category_health.values())) if category_health else 50.0
            
            # Log model prediction separately if available (for reference only)
            if model_rul is not None:
                self._log_api(f"Category Avg: {system_rul:.1f}% | Model: {model_rul:.1f}%")
                rul_source = "CATEGORY AVERAGE"
            else:
                rul_source = "CATEGORY AVERAGE"
            
            # Update main gauge with category average RUL
            if hasattr(self, 'main_gauge'):
                self.main_gauge.set_value(system_rul, system_rul)
                self.main_gauge.update()
                self.main_gauge.repaint()
            
            # Update status LEDs
            led_map = {
                "ENGINE": "Wear & Degradation", 
                "THERMAL": "Temperature & Thermal", 
                "PRESSURE": "Fluid & Pressure", 
                "VIBRATION": "Vibration & Mechanical", 
                "AIR": "Air & Combustion"
            }
            
            if hasattr(self, 'status_leds'):
                for led_name, cat in led_map.items():
                    if led_name in self.status_leds and cat in category_health:
                        h = category_health[cat]
                        if h < 30:
                            status = "critical"
                        elif h < 60:
                            status = "warning"
                        else:
                            status = "normal"
                        self.status_leds[led_name].set_status(status)
            
            # Update alert display with RUL source info
            if hasattr(self, 'alert_display'):
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                # Show breakdown of categories
                cat_summary = " | ".join([f"{cat.split('&')[0].strip()[:4]}: {val:.0f}%" 
                                          for cat, val in category_health.items()])
                
                if system_rul < 30:
                    self.alert_display.setText(f"[{timestamp}] CRITICAL RUL: {system_rul:.1f}% | {cat_summary}")
                    self.alert_display.setStyleSheet("color: #ff4444; font-size: 14px; font-weight: bold; padding: 15px; background-color: #2a1a1a; border-radius: 8px; border: 1px solid #4a2a2a;")
                elif system_rul < 60:
                    self.alert_display.setText(f"[{timestamp}] WARNING RUL: {system_rul:.1f}% | {cat_summary}")
                    self.alert_display.setStyleSheet("color: #ffcc00; font-size: 14px; font-weight: bold; padding: 15px; background-color: #2a2a1a; border-radius: 8px; border: 1px solid #4a4a2a;")
                else:
                    self.alert_display.setText(f"[{timestamp}] NOMINAL RUL: {system_rul:.1f}% | {cat_summary}")
                    self.alert_display.setStyleSheet("color: #00ff88; font-size: 14px; font-weight: bold; padding: 15px; background-color: #1a2a1a; border-radius: 8px; border: 1px solid #2a4a2a;")
            
            self._log_api(f"Updated {sensors_updated} sensors | System RUL: {system_rul:.1f}% ({rul_source})")
            
        except Exception as e:
            self._log_api(f"Update error: {str(e)}")



