"""
Data Model for ARCSaathi application.
Handles dataset loading, storage, and basic operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from PySide6.QtCore import QObject, Signal


class DataModel(QObject):
    """
    Manages dataset operations including loading, validation, and basic statistics.
    """
    
    # Signals
    data_loaded = Signal(object)  # Emits DataFrame
    data_changed = Signal()
    error_occurred = Signal(str)  # Error message
    
    def __init__(self):
        """Initialize the data model."""
        super().__init__()
        
        self.data: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self._original_data: Optional[pd.DataFrame] = None
    
    def load_data(self, file_path: str) -> bool:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file (CSV, Excel, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_extension = Path(file_path).suffix.lower()

            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                if file_extension == '.xlsx':
                    try:
                        import openpyxl  # type: ignore  # noqa: F401
                    except Exception:
                        self.error_occurred.emit("Reading .xlsx requires 'openpyxl'. Install it with: pip install openpyxl")
                        return False
                    self.data = pd.read_excel(file_path, engine='openpyxl')
                else:
                    try:
                        import xlrd  # type: ignore  # noqa: F401
                    except Exception:
                        self.error_occurred.emit("Reading .xls requires 'xlrd'. Install it with: pip install xlrd")
                        return False
                    self.data = pd.read_excel(file_path, engine='xlrd')
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
            elif file_extension == '.parquet':
                try:
                    self.data = pd.read_parquet(file_path)
                except Exception as exc:
                    msg = str(exc)
                    if 'pyarrow' in msg.lower() or 'fastparquet' in msg.lower() or 'engine' in msg.lower():
                        self.error_occurred.emit(
                            "Reading .parquet requires 'pyarrow' (recommended) or 'fastparquet'. Install one with: pip install pyarrow"
                        )
                        return False
                    raise
            else:
                self.error_occurred.emit(f"Unsupported file format: {file_extension}")
                return False
            
            self.file_path = file_path
            self._original_data = self.data.copy()
            self._update_metadata()
            self.data_loaded.emit(self.data)
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading data: {str(e)}")
            return False

    def set_data(self, df: pd.DataFrame, file_path: Optional[str] = None) -> None:
        """Set dataset directly (useful for background loading)."""
        self.data = df.copy()
        self.file_path = file_path
        self._original_data = self.data.copy()
        self._update_metadata()
        self.data_loaded.emit(self.data)
    
    def _update_metadata(self):
        """Update metadata based on current data."""
        if self.data is not None:
            self.metadata = {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'column_names': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'missing_values': self.data.isnull().sum().to_dict(),
                'file_path': self.file_path
            }
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the current dataset.
        
        Returns:
            DataFrame or None if no data loaded
        """
        return self.data
    
    def get_column_names(self) -> List[str]:
        """
        Get list of column names.
        
        Returns:
            List of column names
        """
        if self.data is not None:
            return list(self.data.columns)
        return []
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        if self.data is None:
            return {}
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / (1024 ** 2),
        }
        
        # Add statistical summary for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_summary'] = self.data[numeric_cols].describe().to_dict()
        
        return info
    
    def get_sample(self, n: int = 5) -> Optional[pd.DataFrame]:
        """
        Get a sample of the data.
        
        Args:
            n: Number of rows to sample
            
        Returns:
            Sample DataFrame or None
        """
        if self.data is not None:
            return self.data.head(n)
        return None
    
    def reset_to_original(self):
        """Reset data to the original loaded state."""
        if self._original_data is not None:
            self.data = self._original_data.copy()
            self._update_metadata()
            self.data_changed.emit()
    
    def update_data(self, new_data: pd.DataFrame):
        """
        Update the current dataset.
        
        Args:
            new_data: New DataFrame to set
        """
        self.data = new_data.copy()
        self._update_metadata()
        self.data_changed.emit()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary containing metadata
        """
        return self.metadata.copy()
    
    def export_data(self, file_path: str, file_format: str = 'csv') -> bool:
        """
        Export current data to a file.
        
        Args:
            file_path: Output file path
            file_format: Format to export (csv, excel, json, parquet)
            
        Returns:
            True if successful, False otherwise
        """
        if self.data is None:
            self.error_occurred.emit("No data to export")
            return False
        
        try:
            if file_format.lower() == 'csv':
                self.data.to_csv(file_path, index=False)
            elif file_format.lower() in ['xlsx', 'excel']:
                self.data.to_excel(file_path, index=False)
            elif file_format.lower() == 'json':
                self.data.to_json(file_path, orient='records', indent=2)
            elif file_format.lower() == 'parquet':
                self.data.to_parquet(file_path, index=False)
            else:
                self.error_occurred.emit(f"Unsupported export format: {file_format}")
                return False
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error exporting data: {str(e)}")
            return False
