"""Configuration management using Pydantic settings."""

from typing import List, Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    DEVICE_CUDA,
)


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    url: str = Field(
        default="postgresql://admin:changeme@localhost:5432/digital_twin",
        description="Database URL",
    )
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")

    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class RedisSettings(BaseSettings):
    """Redis configuration."""

    url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    password: Optional[str] = Field(default=None, description="Redis password")

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class ModelSettings(BaseSettings):
    """Model configuration."""

    name: str = Field(default="swin_tiny_patch4_window7_224", description="Model name")
    num_classes: int = Field(default=2, description="Number of classes")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    device: str = Field(default=DEVICE_CUDA, description="Device (cuda or cpu)")
    image_size: int = Field(default=224, description="Input image size")

    model_config = SettingsConfigDict(env_prefix="MODEL_")


class TrainingSettings(BaseSettings):
    """Training configuration."""

    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, description="Batch size")
    num_epochs: int = Field(default=DEFAULT_NUM_EPOCHS, description="Number of epochs")
    learning_rate: float = Field(default=DEFAULT_LEARNING_RATE, description="Learning rate")
    weight_decay: float = Field(default=DEFAULT_WEIGHT_DECAY, description="Weight decay")
    num_workers: int = Field(default=4, description="Number of data loader workers")
    gradient_clip: Optional[float] = Field(default=None, description="Gradient clipping value")
    early_stopping_patience: int = Field(default=5, description="Early stopping patience")

    model_config = SettingsConfigDict(env_prefix="TRAINING_")


class MLflowSettings(BaseSettings):
    """MLflow configuration."""

    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI")
    experiment_name: str = Field(
        default="defect-detection", description="MLflow experiment name"
    )

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")


class APISettings(BaseSettings):
    """API configuration."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of workers")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    secret_key: str = Field(
        default="change-me-in-production", description="Secret key for JWT"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration (minutes)"
    )

    model_config = SettingsConfigDict(env_prefix="API_")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="text", description="Log format (text or json)")
    file: Optional[Path] = Field(default=None, description="Log file path")

    model_config = SettingsConfigDict(env_prefix="LOG_")


class Settings(BaseSettings):
    """Main application settings."""

    # Environment
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
