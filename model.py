"""Type-safe model for license plate detection with FPN + CBAM + CIoU."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def box_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute IoU for normalized corner-format boxes."""
    inter_xmin = tf.maximum(y_true[:, 0], y_pred[:, 0])
    inter_ymin = tf.maximum(y_true[:, 1], y_pred[:, 1])
    inter_xmax = tf.minimum(y_true[:, 2], y_pred[:, 2])
    inter_ymax = tf.minimum(y_true[:, 3], y_pred[:, 3])

    inter_w = tf.maximum(0.0, inter_xmax - inter_xmin)
    inter_h = tf.maximum(0.0, inter_ymax - inter_ymin)
    intersection = inter_w * inter_h

    true_w = tf.maximum(0.0, y_true[:, 2] - y_true[:, 0])
    true_h = tf.maximum(0.0, y_true[:, 3] - y_true[:, 1])
    pred_w = tf.maximum(0.0, y_pred[:, 2] - y_pred[:, 0])
    pred_h = tf.maximum(0.0, y_pred[:, 3] - y_pred[:, 1])

    union = true_w * true_h + pred_w * pred_h - intersection
    return tf.math.divide_no_nan(intersection, union + 1e-6)


def box_ciou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute Complete IoU (CIoU) - better for bbox regression than IoU/GIoU."""
    # IoU
    inter_xmin = tf.maximum(y_true[:, 0], y_pred[:, 0])
    inter_ymin = tf.maximum(y_true[:, 1], y_pred[:, 1])
    inter_xmax = tf.minimum(y_true[:, 2], y_pred[:, 2])
    inter_ymax = tf.minimum(y_true[:, 3], y_pred[:, 3])

    inter_w = tf.maximum(0.0, inter_xmax - inter_xmin)
    inter_h = tf.maximum(0.0, inter_ymax - inter_ymin)
    intersection = inter_w * inter_h

    true_w = tf.maximum(1e-6, y_true[:, 2] - y_true[:, 0])
    true_h = tf.maximum(1e-6, y_true[:, 3] - y_true[:, 1])
    pred_w = tf.maximum(1e-6, y_pred[:, 2] - y_pred[:, 0])
    pred_h = tf.maximum(1e-6, y_pred[:, 3] - y_pred[:, 1])

    true_area = true_w * true_h
    pred_area = pred_w * pred_h
    union = true_area + pred_area - intersection
    iou = tf.math.divide_no_nan(intersection, union + 1e-6)

    # Enclosing box
    enc_xmin = tf.minimum(y_true[:, 0], y_pred[:, 0])
    enc_ymin = tf.minimum(y_true[:, 1], y_pred[:, 1])
    enc_xmax = tf.maximum(y_true[:, 2], y_pred[:, 2])
    enc_ymax = tf.maximum(y_true[:, 3], y_pred[:, 3])
    enc_w = tf.maximum(0.0, enc_xmax - enc_xmin)
    enc_h = tf.maximum(0.0, enc_ymax - enc_ymin)
    enc_diag = enc_w ** 2 + enc_h ** 2

    # Center distance
    true_cx = (y_true[:, 0] + y_true[:, 2]) / 2.0
    true_cy = (y_true[:, 1] + y_true[:, 3]) / 2.0
    pred_cx = (y_pred[:, 0] + y_pred[:, 2]) / 2.0
    pred_cy = (y_pred[:, 1] + y_pred[:, 3]) / 2.0
    center_dist = (true_cx - pred_cx) ** 2 + (true_cy - pred_cy) ** 2

    # Aspect ratio
    v = (4.0 / (np.pi ** 2)) * (
        tf.atan(true_w / tf.maximum(1e-6, true_h)) - 
        tf.atan(pred_w / tf.maximum(1e-6, pred_h))
    ) ** 2
    alpha = tf.math.divide_no_nan(v, (1.0 - iou + v) + 1e-6)

    ciou = iou - (center_dist / (enc_diag + 1e-6)) - alpha * v
    return tf.clip_by_value(ciou, -1.0, 1.0)


class ChannelAttention(layers.Layer):
    """CBAM Channel Attention Module."""
    
    def __init__(self, reduction: int = 16, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        channels = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        self.dense1 = layers.Dense(channels // self.reduction, activation="relu")
        self.dense2 = layers.Dense(channels)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        avg = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        
        avg = self.dense1(avg)
        avg = self.dense2(avg)
        
        max_pool = self.dense1(max_pool)
        max_pool = self.dense2(max_pool)
        
        channel_out = tf.nn.sigmoid(avg + max_pool)
        return inputs * channel_out


class SpatialAttention(layers.Layer):
    """CBAM Spatial Attention Module."""
    
    def __init__(self, kernel_size: int = 7, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid")
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        avg = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg, max_pool], axis=-1)
        spatial_out = self.conv(concat)
        return inputs * spatial_out


class CBAM(layers.Layer):
    """Convolutional Block Attention Module (Channel + Spatial)."""
    
    def __init__(self, reduction: int = 16, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.channel_att = ChannelAttention(reduction=reduction)
        self.spatial_att = SpatialAttention()
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.channel_att(inputs)
        x = self.spatial_att(x)
        return x


class BoundingBoxMetric(keras.metrics.Metric):
    """Mean CIoU metric for normalized boxes."""

    def __init__(self, name: str = "mean_ciou", **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None) -> None:
        ciou = box_ciou(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(ciou))
        self.count.assign_add(tf.cast(tf.size(ciou), tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)


class BoxRegressionLoss(keras.losses.Loss):
    """Complete IoU (CIoU) loss - better for bbox regression."""

    def __init__(self, name: str = "ciou_loss") -> None:
        super().__init__(name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ciou = box_ciou(y_true, y_pred)
        return 1.0 - ciou


def create_optimizer(learning_rate: float, weight_decay: float = 1e-5) -> keras.optimizers.Optimizer:
    """Create AdamW optimizer when available, fallback to Adam."""
    try:
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=1.0)
    except AttributeError:
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)


class LicensePlateDetectionModel:
    """EfficientNetV2 + FPN + CBAM + Dense head for plate localization."""

    def __init__(self, input_shape: Tuple[int, int, int] = (320, 320, 3)) -> None:
        self.input_shape = input_shape
        self.backbone: Optional[keras.Model] = None
        self.model: Optional[keras.Model] = None

    def _build_fpn(self, backbone_outputs: list[tf.Tensor], fpn_channels: int = 256) -> tf.Tensor:
        """Build simplified FPN by pooling multi-scale features separately."""
        # Process each scale independently
        fpn_features = []
        
        for i, backbone_out in enumerate(backbone_outputs):
            # 1×1 convolution to standardize channels
            x = layers.Conv2D(fpn_channels, 1, padding="same", name=f"fpn_conv_{i}")(backbone_out)
            
            # Refine with 3×3 convolution
            x = layers.Conv2D(fpn_channels, 3, padding="same", activation="relu", name=f"fpn_refine_{i}")(x)
            
            # Global average pooling to get fixed-size feature
            x = layers.GlobalAveragePooling2D(name=f"fpn_pool_{i}")(x)
            fpn_features.append(x)
        
        # Concatenate all scale features
        merged = layers.Concatenate(name="fpn_concat")(fpn_features)
        return merged

    def build(self, fine_tune_at: int = 180) -> keras.Model:
        """Build model with FPN + CBAM + CIoU loss."""
        inputs = layers.Input(shape=self.input_shape)

        # Data augmentation
        x = layers.RandomBrightness(factor=0.2)(inputs)
        x = layers.RandomContrast(factor=0.2)(x)
        x = layers.RandomRotation(factor=0.1)(x)
        x = layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))(x)

        # Backbone: EfficientNetV2B2 (larger than B0)
        backbone = keras.applications.EfficientNetV2B2(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
        )
        backbone.trainable = False
        self.backbone = backbone

        # Get intermediate layers for FPN
        layer_names = ["block2b_add", "block4a_expand_activation", "top_activation"]
        backbone_outputs = [backbone.get_layer(name).output for name in layer_names]
        
        # Create intermediate model
        intermediate_model = keras.Model(inputs=backbone.input, outputs=backbone_outputs)
        intermediate_outputs = intermediate_model(x)

        # FPN
        fpn_features = self._build_fpn(intermediate_outputs, fpn_channels=256)

        # CBAM attention
        x = layers.Dense(512, activation="swish")(fpn_features)
        x = layers.Reshape((1, 1, 512))(x)
        x = CBAM(reduction=16)(x)
        x = layers.Flatten()(x)

        # Dense head
        x = layers.LayerNormalization()(x)
        x = layers.Dense(512, activation="swish")(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.LayerNormalization()(x)
        x = layers.Dense(256, activation="swish")(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation="swish")(x)
        x = layers.Dropout(0.2)(x)

        # Output: corner format [xmin, ymin, xmax, ymax]
        bbox = layers.Dense(4, activation="sigmoid", name="bbox")(x)

        self.model = keras.Model(inputs=inputs, outputs=bbox, name="plate_detector_fpn_cbam")
        self._unfreeze_top_layers(fine_tune_at)
        return self.model

    def _unfreeze_top_layers(self, fine_tune_at: int) -> None:
        if self.backbone is None:
            return

        self.backbone.trainable = True
        for layer in self.backbone.layers[:fine_tune_at]:
            layer.trainable = False

    def set_backbone_trainable(self, trainable: bool, fine_tune_at: int = 180) -> None:
        """Freeze or unfreeze backbone."""
        if self.backbone is None:
            return

        self.backbone.trainable = trainable
        if trainable:
            for layer in self.backbone.layers[:fine_tune_at]:
                layer.trainable = False

    def compile(
        self,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        loss: Optional[keras.losses.Loss] = None,
        metrics: Optional[list[keras.metrics.Metric | str]] = None,
    ) -> None:
        """Compile model."""
        if self.model is None:
            raise ValueError("Build model first.")

        if optimizer is None:
            optimizer = create_optimizer(learning_rate=1e-4)
        if loss is None:
            loss = BoxRegressionLoss()
        if metrics is None:
            metrics = [BoundingBoxMetric(), keras.metrics.MeanAbsoluteError(name="mae")]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 16,
        callbacks: Optional[list[keras.callbacks.Callback]] = None,
    ) -> keras.callbacks.History:
        """Train model."""
        if self.model is None:
            raise ValueError("Build model first.")

        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            ]

        return self.model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            validation_data=(x_val, y_val) if x_val is not None and y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict normalized boxes."""
        if self.model is None:
            raise ValueError("Build or load model first.")
        return self.model.predict(images, verbose=0)

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Build model first.")
        self.model.save(filepath)

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = keras.models.load_model(
            filepath,
            compile=False,
            custom_objects={
                "BoxRegressionLoss": BoxRegressionLoss,
                "BoundingBoxMetric": BoundingBoxMetric,
                "ChannelAttention": ChannelAttention,
                "SpatialAttention": SpatialAttention,
                "CBAM": CBAM,
            },
        )

    def get_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Build or load model first.")
        self.model.summary()
