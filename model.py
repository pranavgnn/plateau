"""Type-safe model for license plate detection."""

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


class BoundingBoxMetric(keras.metrics.Metric):
    """Mean IoU metric for normalized boxes."""

    def __init__(self, name: str = "mean_iou", **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None) -> None:
        iou = box_iou(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.size(iou), tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)


class CenterSizeToCorners(layers.Layer):
    """Convert [cx, cy, w, h] to [xmin, ymin, xmax, ymax]."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        cx = inputs[:, 0]
        cy = inputs[:, 1]
        width = inputs[:, 2] * 0.8 + 0.02
        height = inputs[:, 3] * 0.5 + 0.01

        xmin = tf.clip_by_value(cx - width / 2.0, 0.0, 1.0)
        ymin = tf.clip_by_value(cy - height / 2.0, 0.0, 1.0)
        xmax = tf.clip_by_value(cx + width / 2.0, 0.0, 1.0)
        ymax = tf.clip_by_value(cy + height / 2.0, 0.0, 1.0)
        return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    @staticmethod
    def _box_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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


class BoxRegressionLoss(keras.losses.Loss):
    """Blend of Huber regression + IoU loss."""

    def __init__(self, iou_weight: float = 1.0, huber_delta: float = 0.1, name: str = "box_regression_loss") -> None:
        super().__init__(name=name)
        self.iou_weight = iou_weight
        self.huber = keras.losses.Huber(delta=huber_delta, reduction=keras.losses.Reduction.NONE)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        huber_loss = tf.reduce_mean(self.huber(y_true, y_pred), axis=-1)
        iou_loss = 1.0 - box_iou(y_true, y_pred)
        return huber_loss + self.iou_weight * iou_loss


class LicensePlateDetectionModel:
    """CNN encoder + box regression head for plate localization."""

    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> None:
        self.input_shape = input_shape
        self.backbone: Optional[keras.Model] = None
        self.model: Optional[keras.Model] = None

    def build(self, fine_tune_at: int = 180) -> keras.Model:
        """Build constrained bbox model."""
        inputs = layers.Input(shape=self.input_shape)

        backbone = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
        )
        backbone.trainable = False
        self.backbone = backbone

        x = backbone(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="swish")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(256, activation="swish")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation="swish")(x)
        x = layers.Dropout(0.15)(x)

        raw_box = layers.Dense(4, activation="sigmoid", name="raw_box")(x)
        box = CenterSizeToCorners(name="box")(raw_box)

        self.model = keras.Model(inputs=inputs, outputs=box, name="plate_detector")
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
            optimizer = keras.optimizers.Adam(learning_rate=1e-4)
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
                "CenterSizeToCorners": CenterSizeToCorners,
            },
        )

    def get_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Build or load model first.")
        self.model.summary()
