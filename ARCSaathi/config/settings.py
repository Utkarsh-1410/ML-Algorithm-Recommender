"""
Settings Manager for ARCSaathi application.
Handles loading, saving, and managing application configuration.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from PySide6.QtCore import QObject, Signal


class SettingsManager(QObject):
    """
    Manages application settings with JSON persistence.
    Provides signal emission for settings changes.
    """
    
    # Signals
    settings_changed = Signal(str, object)  # (key, value)
    theme_changed = Signal(str)  # theme name
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the settings manager.
        
        Args:
            config_path: Path to the settings JSON file. If None, uses default.
        """
        super().__init__()
        
        self.config_path = config_path or self._get_default_config_path()
        self.settings: Dict[str, Any] = {}
        self._load_default_settings()
        self.load_settings()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        config_dir = Path(__file__).parent
        return str(config_dir / "default_settings.json")
    
    def _get_user_config_path(self) -> str:
        """Get the user-specific configuration file path."""
        home = Path.home()
        user_config_dir = home / ".arcsaathi"
        user_config_dir.mkdir(exist_ok=True)
        return str(user_config_dir / "settings.json")
    
    def _load_default_settings(self):
        """Load default settings from the default configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            print(f"Default settings file not found: {self.config_path}")
            self.settings = self._get_fallback_settings()
        except json.JSONDecodeError as e:
            print(f"Error parsing default settings: {e}")
            self.settings = self._get_fallback_settings()
    
    def _get_fallback_settings(self) -> Dict[str, Any]:
        """Return minimal fallback settings if config file is unavailable."""
        return {
            "application": {"name": "ARCSaathi", "version": "1.0.0"},
            "paths": {
                "default_data_dir": "./data",
                "default_model_dir": "./models",
                "default_export_dir": "./exports",
                "logs_dir": "./logs"
            },
            "ui": {
                "theme": "light",
                "window_width": 1400,
                "window_height": 900
            },
            "compute": {
                "max_workers": 4,
                "memory_limit_mb": 4096
            },
            "autosave": {
                "enabled": True,
                "interval_seconds": 300
            }
        }
    
    def load_settings(self):
        """Load user settings, merging with defaults."""
        user_config_path = self._get_user_config_path()
        
        if os.path.exists(user_config_path):
            try:
                with open(user_config_path, 'r') as f:
                    user_settings = json.load(f)
                    # Merge user settings with defaults
                    self._deep_merge(self.settings, user_settings)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading user settings: {e}")
    
    def save_settings(self):
        """Save current settings to user configuration file."""
        user_config_path = self._get_user_config_path()
        
        try:
            with open(user_config_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a setting value using dot notation.
        
        Args:
            key_path: Dot-separated path to the setting (e.g., "ui.theme")
            default: Default value if key not found
            
        Returns:
            The setting value or default
        """
        keys = key_path.split('.')
        value = self.settings
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any, emit_signal: bool = True):
        """
        Set a setting value using dot notation.
        
        Args:
            key_path: Dot-separated path to the setting
            value: Value to set
            emit_signal: Whether to emit settings_changed signal
        """
        keys = key_path.split('.')
        settings = self.settings
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in settings:
                settings[key] = {}
            settings = settings[key]
        
        # Set the value
        settings[keys[-1]] = value
        
        # Emit signals
        if emit_signal:
            self.settings_changed.emit(key_path, value)
            
            # Special handling for theme changes
            if key_path == "ui.theme":
                self.theme_changed.emit(value)
    
    def _deep_merge(self, base: Dict, updates: Dict):
        """
        Deep merge updates into base dictionary.
        
        Args:
            base: Base dictionary to merge into
            updates: Dictionary with updates
        """
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self._load_default_settings()
        self.save_settings()
        self.settings_changed.emit("*", None)
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get a copy of all settings."""
        return self.settings.copy()
