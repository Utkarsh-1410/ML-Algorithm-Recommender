"""Custom exceptions used across ARCSaathi."""


class ARCSaathiError(Exception):
    """Base exception for application errors."""


class ConfigurationError(ARCSaathiError):
    """Raised when configuration cannot be loaded or validated."""


class DataError(ARCSaathiError):
    """Raised for data loading/processing errors."""


class TrainingError(ARCSaathiError):
    """Raised for model training/tuning errors."""


class EvaluationError(ARCSaathiError):
    """Raised for evaluation/comparison errors."""
