"""
Real-time Predictive Maintenance API Server
Generates dynamic sensor data with realistic values and patterns.

Usage:
    python sample_api.py

API Endpoint:
    http://localhost:5000/api/sensor-data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import math
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Sensor configuration with exact ranges
SENSOR_CONFIG = {
    "ferrous_debris": {"min": 0, "max": 100, "normal_range": (5, 20), "trend": "stable"},
    "soot_in_oil": {"min": 0, "max": 140, "normal_range": (10, 50), "trend": "increasing"},
    "cylinder_head_temp": {"min": 70, "max": 130, "normal_range": (85, 105), "trend": "oscillating"},
    "exhaust_gas_temp": {"min": 250, "max": 850, "normal_range": (400, 600), "trend": "oscillating"},
    "bearing_temp": {"min": 30, "max": 110, "normal_range": (50, 75), "trend": "stable"},
    "engine_vibration": {"min": 0, "max": 10, "normal_range": (1, 4), "trend": "oscillating"},
    "knock_sensor": {"min": 0, "max": 100, "normal_range": (15, 45), "trend": "stable"},
    "crankshaft_vibration": {"min": 200, "max": 1000, "normal_range": (400, 700), "trend": "oscillating"},
    "oil_temperature": {"min": 0, "max": 120, "normal_range": (60, 95), "trend": "stable"},
    "injector_pressure": {"min": 1, "max": 20, "normal_range": (8, 15), "trend": "stable"},
    "oil_pressure": {"min": 0, "max": 0.5, "normal_range": (0.15, 0.35), "trend": "stable"},
    "coolant_pressure": {"min": 0, "max": 0.15, "normal_range": (0.08, 0.12), "trend": "stable"},
    "mass_air_flow": {"min": 0.2, "max": 10, "normal_range": (2, 6), "trend": "oscillating"},
    "oxygen_sensor": {"min": 0, "max": 1.1, "normal_range": (0.96, 1.02), "trend": "oscillating"},
    "egr_flow": {"min": 0, "max": 20, "normal_range": (5, 12), "trend": "stable"}
}

# State tracking for realistic data evolution
class SensorState:
    def __init__(self):
        self.start_time = time.time()
        self.last_values = {}
        self.trends = {}
        
        # Initialize with mid-range values
        for sensor, config in SENSOR_CONFIG.items():
            norm_min, norm_max = config['normal_range']
            self.last_values[sensor] = (norm_min + norm_max) / 2
            self.trends[sensor] = random.uniform(-0.5, 0.5)
    
    def generate_value(self, sensor: str) -> float:
        """Generate a realistic sensor value with smooth transitions."""
        config = SENSOR_CONFIG[sensor]
        min_val = config['min']
        max_val = config['max']
        norm_min, norm_max = config['normal_range']
        trend_type = config['trend']
        
        # Get last value or initialize
        last_value = self.last_values.get(sensor, (norm_min + norm_max) / 2)
        
        # Time-based oscillation factor
        elapsed = time.time() - self.start_time
        
        if trend_type == "oscillating":
            # Sinusoidal pattern with random frequency
            freq = random.uniform(0.1, 0.3)
            oscillation = math.sin(elapsed * freq) * (norm_max - norm_min) * 0.3
            target = (norm_min + norm_max) / 2 + oscillation
        elif trend_type == "increasing":
            # Slow upward drift
            drift = (elapsed / 100) % (norm_max - norm_min)
            target = norm_min + drift
        else:  # stable
            # Stay near normal range with small variations
            target = random.uniform(norm_min, norm_max)
        
        # Add random noise
        noise = random.gauss(0, (norm_max - norm_min) * 0.05)
        target += noise
        
        # Smooth transition from last value (momentum)
        new_value = last_value * 0.7 + target * 0.3
        
        # Occasionally spike (5% chance)
        if random.random() < 0.05:
            spike_direction = random.choice([-1, 1])
            spike_magnitude = random.uniform(0.2, 0.4) * (max_val - min_val)
            new_value += spike_direction * spike_magnitude
        
        # Clamp to valid range
        new_value = max(min_val, min(max_val, new_value))
        
        # Update state
        self.last_values[sensor] = new_value
        
        return new_value

# Global state
sensor_state = SensorState()

@app.route('/api/sensor-data', methods=['GET'])
def get_sensor_data():
    """
    Get current real-time sensor readings.
    Returns a list with a single sensor reading dictionary.
    
    Response format:
    [
        {
            "ferrous_debris": 15.3,
            "soot_in_oil": 45.6,
            ...
            "timestamp": "2026-02-08T11:30:00.000000"
        }
    ]
    """
    # Generate current sensor readings
    data = {}
    
    for sensor in SENSOR_CONFIG.keys():
        value = sensor_state.generate_value(sensor)
        # Round to appropriate precision
        if sensor in ['oil_pressure', 'coolant_pressure', 'oxygen_sensor']:
            data[sensor] = round(value, 4)
        elif sensor in ['mass_air_flow']:
            data[sensor] = round(value, 3)
        else:
            data[sensor] = round(value, 2)
    
    # Add timestamp
    data['timestamp'] = datetime.now().isoformat()
    
    # Return as list (expected format)
    return jsonify([data])

@app.route('/api/sensor-data/batch', methods=['GET'])
def get_sensor_data_batch():
    """
    Get multiple sensor readings (historical simulation).
    
    Query params:
        - count: Number of readings to return (default: 10, max: 100)
    
    Example: GET /api/sensor-data/batch?count=5
    """
    count = int(request.args.get('count', 10))
    count = min(count, 100)  # Limit to 100
    
    data_list = []
    for _ in range(count):
        reading = {}
        for sensor in SENSOR_CONFIG.keys():
            value = sensor_state.generate_value(sensor)
            if sensor in ['oil_pressure', 'coolant_pressure', 'oxygen_sensor']:
                reading[sensor] = round(value, 4)
            elif sensor in ['mass_air_flow']:
                reading[sensor] = round(value, 3)
            else:
                reading[sensor] = round(value, 2)
        reading['timestamp'] = datetime.now().isoformat()
        data_list.append(reading)
        time.sleep(0.01)  # Small delay between readings
    
    return jsonify(data_list)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'sensors_active': len(SENSOR_CONFIG)
    })

@app.route('/api/schema', methods=['GET'])
def get_schema():
    """
    Returns the sensor configuration schema.
    Useful for understanding expected data format and ranges.
    """
    schema = {}
    for sensor, config in SENSOR_CONFIG.items():
        schema[sensor] = {
            'min': config['min'],
            'max': config['max'],
            'normal_range': config['normal_range'],
            'unit': get_unit(sensor)
        }
    
    return jsonify({
        'sensors': schema,
        'total_sensors': len(SENSOR_CONFIG),
        'description': 'Real-time engine health monitoring sensors'
    })

def get_unit(sensor: str) -> str:
    """Get the unit for a sensor."""
    units = {
        'ferrous_debris': 'μm',
        'soot_in_oil': 'mg/L',
        'cylinder_head_temp': '°C',
        'exhaust_gas_temp': '°C',
        'bearing_temp': '°C',
        'engine_vibration': 'g',
        'knock_sensor': '%',
        'crankshaft_vibration': 'Ω',
        'oil_temperature': '°C',
        'injector_pressure': 'MPa',
        'oil_pressure': 'MPa',
        'coolant_pressure': 'MPa',
        'mass_air_flow': 'm/s',
        'oxygen_sensor': 'λ',
        'egr_flow': '%'
    }
    return units.get(sensor, '')

@app.route('/api/reset', methods=['POST'])
def reset_state():
    """Reset sensor state to initial conditions."""
    global sensor_state
    sensor_state = SensorState()
    return jsonify({
        'status': 'success',
        'message': 'Sensor state reset to initial conditions',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🔧 PREDICTIVE MAINTENANCE - REAL-TIME API SERVER")
    print("="*70)
    print("\n📡 Available Endpoints:")
    print("  GET  /api/sensor-data       - Current sensor readings (auto-updating)")
    print("  GET  /api/sensor-data/batch - Multiple readings (historical)")
    print("  GET  /api/health            - API health check")
    print("  GET  /api/schema            - Sensor configuration schema")
    print("  POST /api/reset             - Reset sensor state")
    
    print("\n⚙️  Sensors Active: 15")
    print("  • Wear & Degradation (2)")
    print("  • Temperature & Thermal (3)")
    print("  • Vibration & Mechanical (3)")
    print("  • Fluid & Pressure (4)")
    print("  • Air & Combustion (3)")
    
    print("\n🔗 Configure in ARCSaathi App:")
    print(f"  API URL: http://localhost:5000/api/sensor-data")
    print(f"  Interval: 3 seconds (recommended)")
    
    print("\n" + "="*70)
    print("  Server starting... Press CTRL+C to stop")
    print("="*70 + "\n")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
