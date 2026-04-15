"""Type-safe ANN model for license plate detection."""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class LicensePlateDetectionModel:
    """ANN-based license plate detector (CNN + regression head)."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.input_shape = input_shape
        self.model: Optional[keras.Model] = None
    
    def build(self) -> keras.Model:
        """Build CNN + ANN architecture for bbox regression."""
        
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN backbone (MobileNetV2 for efficiency)
        backbone = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        backbone.trainable = False  # Freeze pretrained weights
        
        x = backbone(inputs)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers (ANN head)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer: 4 values (xmin, ymin, xmax, ymax) in [0, 1]
        outputs = layers.Dense(4, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile(
        self,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        loss: Optional[keras.losses.Loss] = None,
        metrics: Optional[list] = None
    ) -> None:
        """Compile model."""
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        
        if loss is None:
            # Use MSE for bbox regression, but could try IoU loss
            loss = keras.losses.MeanSquaredError()
        
        if metrics is None:
            metrics = [keras.metrics.MeanAbsoluteError()]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("Model compiled.")
    
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 16,
        callbacks: Optional[list] = None
    ) -> keras.callbacks.History:
        """Train model."""
        
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val) if x_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict bboxes for images. Returns array of shape (N, 4)."""
        return self.model.predict(images)
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_summary(self) -> None:
        """Print model summary."""
        self.model.summary()


class IoULoss(keras.losses.Loss):
    """Intersection over Union loss for bbox regression."""
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute IoU loss."""
        
        # y_true, y_pred shape: (batch_size, 4) - [xmin, ymin, xmax, ymax]
        
        # Compute intersection
        inter_xmin = tf.maximum(y_true[:, 0], y_pred[:, 0])
        inter_ymin = tf.maximum(y_true[:, 1], y_pred[:, 1])
        inter_xmax = tf.minimum(y_true[:, 2], y_pred[:, 2])
        inter_ymax = tf.minimum(y_true[:, 3], y_pred[:, 3])
        
        inter_width = tf.maximum(0.0, inter_xmax - inter_xmin)
        inter_height = tf.maximum(0.0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height
        
        # Compute union
        true_width = y_true[:, 2] - y_true[:, 0]
        true_height = y_true[:, 3] - y_true[:, 1]
        true_area = true_width * true_height
        
        pred_width = y_pred[:, 2] - y_pred[:, 0]
        pred_height = y_pred[:, 3] - y_pred[:, 1]
        pred_area = pred_width * pred_height
        
        union_area = true_area + pred_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        
        return 1.0 - tf.reduce_mean(iou)
