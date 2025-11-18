"""Custom exceptions for the Digital Twin Factory system."""


class DigitalTwinError(Exception):
    """Base exception for all Digital Twin Factory errors."""

    pass


class ConfigurationError(DigitalTwinError):
    """Raised when configuration is invalid or missing."""

    pass


class ModelError(DigitalTwinError):
    """Base exception for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""

    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""

    pass


class DataError(DigitalTwinError):
    """Base exception for data-related errors."""

    pass


class DatasetNotFoundError(DataError):
    """Raised when dataset is not found."""

    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""

    pass


class APIError(DigitalTwinError):
    """Base exception for API-related errors."""

    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(APIError):
    """Raised when user lacks permission."""

    pass


class DigitalTwinStateError(DigitalTwinError):
    """Raised when digital twin state is invalid."""

    pass


class SchedulingError(DigitalTwinError):
    """Raised when scheduling optimization fails."""

    pass
