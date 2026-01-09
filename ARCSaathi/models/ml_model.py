"""ML model manager for ARCSaathi.

Supports:
- Registry-driven estimator creation (preferred)
- Supervised and unsupervised fitting
- Persistence via pickle
"""

import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from .model_registry import build_registry


class MLModelWrapper:
    """Wrapper class for ML models with metadata."""
    
    def __init__(
        self,
        model,
        model_type: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        *,
        algorithm_key: Optional[str] = None,
        task_type: Optional[str] = None,
    ):
        """
        Initialize ML model wrapper.
        
        Args:
            model: The actual sklearn/ML model
            model_type: Type of model (e.g., 'RandomForest', 'LogisticRegression')
            params: Model parameters
            metadata: Additional metadata
        """
        self.model = model
        self.model_type = model_type
        self.params = params
        self.metadata = metadata or {}
        self.algorithm_key = algorithm_key
        self.task_type = task_type
        self.trained = False
        self.training_time = None
        self.feature_names = []
        self.target_name = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        return {
            'algorithm_key': self.algorithm_key,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'params': self.params,
            'metadata': self.metadata,
            'trained': self.trained,
            'training_time': self.training_time,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }


class MLModel(QObject):
    """
    Manages machine learning models for training, prediction, and persistence.
    """
    
    # Signals
    model_trained = Signal(str)  # model name
    prediction_complete = Signal(object)  # predictions array
    error_occurred = Signal(str)
    training_progress = Signal(int)  # progress percentage
    
    def __init__(self):
        """Initialize the ML model manager."""
        super().__init__()
        
        self.models: Dict[str, MLModelWrapper] = {}
        self.active_model: Optional[str] = None

        # Algorithm registry (preferred creation path)
        self._registry = build_registry()

    def list_registry(self) -> Dict[str, Any]:
        """Return registry entries with availability + model cards."""
        out: Dict[str, Any] = {}
        for k, spec in self._registry.items():
            available, reason = spec.is_available()
            out[k] = {
                "available": bool(available),
                "unavailable_reason": reason,
                "card": spec.card,
            }
        return out

    def is_algorithm_available(self, algorithm_key: str) -> tuple[bool, str]:
        spec = self._registry.get(algorithm_key)
        if not spec:
            return (False, f"Unknown algorithm_key: {algorithm_key}")
        return spec.is_available()

    def create_model_from_registry(self, name: str, algorithm_key: str, params: Dict[str, Any]) -> bool:
        """Create a model instance from the registry."""
        try:
            spec = self._registry.get(algorithm_key)
            if not spec:
                self.error_occurred.emit(f"Unknown algorithm: {algorithm_key}")
                return False

            available, reason = spec.is_available()
            if not available:
                self.error_occurred.emit(f"{spec.card.name} unavailable ({reason})")
                return False

            model = spec.factory(dict(params or {}))

            wrapper = MLModelWrapper(
                model=model,
                model_type=spec.card.name,
                params=dict(params or {}),
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'name': name,
                    'family': spec.card.family,
                },
                algorithm_key=spec.card.key,
                task_type=spec.card.task_type,
            )

            self.models[name] = wrapper
            self.active_model = name
            return True
        except Exception as e:
            self.error_occurred.emit(f"Error creating model: {str(e)}")
            return False

    def fit_model(self, name: str, X: pd.DataFrame, y: Optional[pd.Series] = None) -> bool:
        """Fit a model; supports supervised and unsupervised estimators."""
        if name not in self.models:
            self.error_occurred.emit(f"Model '{name}' not found")
            return False

        try:
            wrapper = self.models[name]
            start_time = datetime.now()

            if y is None:
                wrapper.model.fit(X)
            else:
                wrapper.model.fit(X, y)

            end_time = datetime.now()
            wrapper.trained = True
            wrapper.training_time = (end_time - start_time).total_seconds()
            wrapper.feature_names = list(X.columns)
            wrapper.target_name = y.name if (y is not None and hasattr(y, 'name')) else None

            self.model_trained.emit(name)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Error training model: {str(e)}")
            return False
    
    def create_model(self, name: str, model_type: str, params: Dict[str, Any]) -> bool:
        """
        Create a new ML model.
        
        Args:
            name: Name identifier for the model
            model_type: Type of model to create
            params: Model hyperparameters
            
        Returns:
            True if successful
        """
        # Backwards compatible: if model_type is a registry key, use registry.
        if model_type in self._registry:
            return self.create_model_from_registry(name=name, algorithm_key=model_type, params=params)

        try:
            # Import here to avoid circular dependencies
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
            from sklearn.svm import SVC, SVR
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            
            # Map model types to classes
            model_classes = {
                'RandomForestClassifier': RandomForestClassifier,
                'RandomForestRegressor': RandomForestRegressor,
                'LogisticRegression': LogisticRegression,
                'LinearRegression': LinearRegression,
                'Ridge': Ridge,
                'Lasso': Lasso,
                'SVC': SVC,
                'SVR': SVR,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'DecisionTreeRegressor': DecisionTreeRegressor,
                'KNeighborsClassifier': KNeighborsClassifier,
                'KNeighborsRegressor': KNeighborsRegressor,
                'GradientBoostingClassifier': GradientBoostingClassifier,
            }
            
            if model_type not in model_classes:
                self.error_occurred.emit(f"Unknown model type: {model_type}")
                return False
            
            # Create the model instance
            model = model_classes[model_type](**params)
            
            # Wrap the model
            wrapper = MLModelWrapper(
                model=model,
                model_type=model_type,
                params=params,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'name': name
                },
                algorithm_key=None,
                task_type=None,
            )
            
            self.models[name] = wrapper
            self.active_model = name
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error creating model: {str(e)}")
            return False
    
    def train_model(self, name: str, X_train: pd.DataFrame, y_train: pd.Series) -> bool:
        """
        Train a model.
        
        Args:
            name: Name of the model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            True if successful
        """
        return self.fit_model(name=name, X=X_train, y=y_train)
    
    def predict(self, name: str, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make predictions using a trained model.
        
        Args:
            name: Name of the model
            X: Features for prediction
            
        Returns:
            Predictions array or None
        """
        if name not in self.models:
            self.error_occurred.emit(f"Model '{name}' not found")
            return None
        
        wrapper = self.models[name]
        
        if not wrapper.trained:
            self.error_occurred.emit(f"Model '{name}' is not trained")
            return None
        
        try:
            predictions = wrapper.model.predict(X)
            self.prediction_complete.emit(predictions)
            return predictions
            
        except Exception as e:
            self.error_occurred.emit(f"Error making predictions: {str(e)}")
            return None
    
    def predict_proba(self, name: str, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Predict class probabilities (for classifiers).
        
        Args:
            name: Name of the model
            X: Features for prediction
            
        Returns:
            Probability array or None
        """
        if name not in self.models:
            self.error_occurred.emit(f"Model '{name}' not found")
            return None
        
        wrapper = self.models[name]
        
        if not wrapper.trained:
            self.error_occurred.emit(f"Model '{name}' is not trained")
            return None
        
        try:
            if hasattr(wrapper.model, 'predict_proba'):
                probabilities = wrapper.model.predict_proba(X)
                return probabilities
            else:
                self.error_occurred.emit(f"Model '{name}' does not support probability prediction")
                return None
                
        except Exception as e:
            self.error_occurred.emit(f"Error predicting probabilities: {str(e)}")
            return None
    
    def get_model(self, name: str) -> Optional[MLModelWrapper]:
        """
        Get a model wrapper.
        
        Args:
            name: Model name
            
        Returns:
            MLModelWrapper or None
        """
        return self.models.get(name)
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get model information.
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with model info or None
        """
        wrapper = self.models.get(name)
        return wrapper.to_dict() if wrapper else None
    
    def list_models(self) -> List[str]:
        """
        Get list of all model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def save_model(self, name: str, file_path: str) -> bool:
        """
        Save a model to disk.
        
        Args:
            name: Model name
            file_path: Path to save the model
            
        Returns:
            True if successful
        """
        if name not in self.models:
            self.error_occurred.emit(f"Model '{name}' not found")
            return False
        
        try:
            wrapper = self.models[name]
            
            with open(file_path, 'wb') as f:
                pickle.dump(wrapper, f)
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, name: str, file_path: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            name: Name to assign to the loaded model
            file_path: Path to the model file
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, 'rb') as f:
                wrapper = pickle.load(f)
            
            self.models[name] = wrapper
            self.active_model = name
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading model: {str(e)}")
            return False
    
    def delete_model(self, name: str) -> bool:
        """
        Delete a model.
        
        Args:
            name: Model name
            
        Returns:
            True if successful
        """
        if name in self.models:
            del self.models[name]
            
            if self.active_model == name:
                self.active_model = None
            
            return True
        return False
    
    def set_active_model(self, name: str) -> bool:
        """
        Set the active model.
        
        Args:
            name: Model name
            
        Returns:
            True if successful
        """
        if name in self.models:
            self.active_model = name
            return True
        return False
    
    def get_active_model(self) -> Optional[str]:
        """
        Get the name of the active model.
        
        Returns:
            Model name or None
        """
        return self.active_model
