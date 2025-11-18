"""Constants used throughout the Digital Twin Factory system."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ONNX_DIR = MODELS_DIR / "onnx"
TENSORRT_DIR = MODELS_DIR / "tensorrt"
MLFLOW_DIR = MODELS_DIR / "mlflow"

# Config directories
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_CONFIG_DIR = CONFIG_DIR / "models"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Image normalization constants (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default image size
DEFAULT_IMAGE_SIZE = 224

# Model names
SWIN_TINY = "swin_tiny_patch4_window7_224"
SWIN_SMALL = "swin_small_patch4_window7_224"
SWIN_BASE = "swin_base_patch4_window7_224"
VIT_BASE = "vit_base_patch16_224"
VIT_LARGE = "vit_large_patch16_224"

# Defect classes (binary classification)
CLASS_NAMES = ["good", "defect"]
NUM_CLASSES = len(CLASS_NAMES)

# Training constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4

# Device
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"

# API constants
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Digital Twin update frequency (Hz)
DT_UPDATE_FREQUENCY = 1.0

# Scheduling time limit (seconds)
SCHEDULING_TIME_LIMIT = 60
