"""Configuration for plateau license plate detection model (PyTorch)."""

from typing import Dict, Any


class Config:
    """Model and training configuration."""
    
    # Dataset
    DATASET_ID: str = "saisirishan/indian-vehicle-dataset"
    IMG_SHAPE: tuple = (320, 320)
    
    # Model
    BACKBONE: str = "EfficientNetV2B2"  # PyTorch torchvision
    FREEZE_BACKBONE: bool = True
    INPUT_SHAPE: tuple = (320, 320, 3)
    
    # Architecture (FPN + CBAM + Dense head)
    FPN_CHANNELS: int = 256
    DENSE_LAYERS: list = [512, 256, 128, 64]
    DROPOUT_RATES: list = [0.35, 0.25, 0.15, 0.1]
    USE_CBAM: bool = True
    
    # Training
    PHASE1_EPOCHS: int = 25
    PHASE2_EPOCHS: int = 40
    PHASE3_EPOCHS: int = 20
    BATCH_SIZE: int = 16
    PHASE1_LR: float = 2e-4
    PHASE2_LR: float = 5e-5
    PHASE3_LR: float = 1e-5
    LOSS_FN: str = "ciou"  # Complete IoU loss
    OPTIMIZER: str = "adamw"
    WARMUP_EPOCHS: int = 3
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 10
    LR_REDUCE_PATIENCE: int = 3
    LR_REDUCE_FACTOR: float = 0.5
    MIN_LR: float = 1e-6
    
    # Data split
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    RANDOM_SEED: int = 42
    
    # Augmentation
    AUGMENT: bool = True
    MAX_ROTATION_ANGLE: float = 5.0
    
    # Device
    DEVICE: str = "cuda"  # cuda or cpu
    USE_AMP: bool = True  # Automatic Mixed Precision for faster training
    SCALE_RANGE: tuple = (0.85, 1.15)
    CUTOUT_PATCHES: int = 2
    
    # Output
    MODEL_SAVE_PATH: str = "license_plate_detector.keras"
    VISUALIZATION_PATH: str = "predictions.png"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export config as dictionary."""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
