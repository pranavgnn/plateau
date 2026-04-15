"""Configuration for plateau license plate detection model."""

from typing import Dict, Any


class Config:
    """Model and training configuration."""
    
    # Dataset
    DATASET_ID: str = "saisirishan/indian-vehicle-dataset"
    IMG_SHAPE: tuple = (224, 224)
    
    # Model
    BACKBONE: str = "MobileNetV2"  # Pretrained CNN backbone
    FREEZE_BACKBONE: bool = True
    INPUT_SHAPE: tuple = (224, 224, 3)
    
    # Architecture (ANN head)
    DENSE_LAYERS: list = [512, 256, 128]
    DROPOUT_RATES: list = [0.3, 0.3, 0.2]
    
    # Training
    EPOCHS: int = 50
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-4
    LOSS_FN: str = "mse"  # "mse" or "iou"
    OPTIMIZER: str = "adam"
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 10
    LR_REDUCE_PATIENCE: int = 5
    LR_REDUCE_FACTOR: float = 0.5
    MIN_LR: float = 1e-6
    
    # Data split
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    RANDOM_SEED: int = 42
    
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
