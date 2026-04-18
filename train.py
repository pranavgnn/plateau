"""Type-safe training script for license plate detection with 3-phase training."""

from typing import Tuple
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from dataset_loader import LicensePlateDataset
from model import (
    LicensePlateDetectionModel,
    BoxRegressionLoss,
    BoundingBoxMetric,
    create_optimizer,
)


def compute_area_weights(bboxes: np.ndarray) -> np.ndarray:
    """Upweight small plates so localization does not overfit to larger boxes."""
    widths = np.maximum(1e-6, bboxes[:, 2] - bboxes[:, 0])
    heights = np.maximum(1e-6, bboxes[:, 3] - bboxes[:, 1])
    areas = widths * heights
    median_area = float(np.median(areas))
    weights = np.sqrt(median_area / areas)
    return np.clip(weights, 0.5, 2.5).astype(np.float32)


def make_area_bins(bboxes: np.ndarray, num_bins: int = 5) -> np.ndarray:
    """Create quantile bins from bbox area for approximate stratification."""
    widths = np.maximum(1e-6, bboxes[:, 2] - bboxes[:, 0])
    heights = np.maximum(1e-6, bboxes[:, 3] - bboxes[:, 1])
    areas = widths * heights
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(areas, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(areas), dtype=np.int32)
    return np.digitize(areas, edges[1:-1], right=False).astype(np.int32)


def visualize_predictions(
    images: np.ndarray,
    true_bboxes: np.ndarray,
    pred_bboxes: np.ndarray,
    num_samples: int = 3
) -> None:
    """Visualize predictions vs ground truth."""
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    
    for i in range(min(num_samples, len(images))):
        img = (images[i] * 255).astype(np.uint8)
        
        # True bbox
        ax_true = axes[i, 0]
        ax_true.imshow(img)
        true_bbox = true_bboxes[i]
        h, w = img.shape[:2]
        rect_true = plt.Rectangle(
            (true_bbox[0] * w, true_bbox[1] * h),
            (true_bbox[2] - true_bbox[0]) * w,
            (true_bbox[3] - true_bbox[1]) * h,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax_true.add_patch(rect_true)
        ax_true.set_title('Ground Truth')
        ax_true.axis('off')
        
        # Predicted bbox
        ax_pred = axes[i, 1]
        ax_pred.imshow(img)
        pred_bbox = pred_bboxes[i]
        rect_pred = plt.Rectangle(
            (pred_bbox[0] * w, pred_bbox[1] * h),
            (pred_bbox[2] - pred_bbox[0]) * w,
            (pred_bbox[3] - pred_bbox[1]) * h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax_pred.add_patch(rect_pred)
        ax_pred.set_title('Prediction')
        ax_pred.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=100)
    print("Saved predictions.png")
    plt.close()


def main() -> None:
    """Main training pipeline with 3-phase strategy."""
    
    print("=== License Plate Detection - Enhanced Training (FPN + CBAM + CIoU) ===\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load dataset
    print("1. Loading dataset...")
    dataset = LicensePlateDataset(dataset_id="saisirishan/indian-vehicle-dataset")
    dataset.load(img_shape=(320, 320), augment=True)
    
    images, bboxes, plate_texts = dataset.get_arrays()
    print(f"Images: {images.shape}, Bboxes: {bboxes.shape}")
    
    # Split data with stratification by box size
    print("\n2. Splitting dataset...")
    area_bins = make_area_bins(bboxes)
    x_train, x_temp, y_train, y_temp = train_test_split(
        images,
        bboxes,
        test_size=0.3,
        random_state=42,
        stratify=area_bins,
    )
    temp_bins = make_area_bins(y_temp)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=temp_bins,
    )
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    train_weights = compute_area_weights(y_train)
    
    # Build model
    print("\n3. Building model with FPN + CBAM...")
    detector = LicensePlateDetectionModel(input_shape=(320, 320, 3))
    detector.build()
    detector.get_summary()
    
    # ===== PHASE 1: Warmup + Head Training (frozen backbone) =====
    print("\n" + "="*60)
    print("PHASE 1: Training Head with Frozen Backbone (Warmup)")
    print("="*60)
    
    detector.set_backbone_trainable(False)
    detector.compile(
        optimizer=create_optimizer(learning_rate=2e-4),
        loss=BoxRegressionLoss(),
        metrics=[BoundingBoxMetric(), keras.metrics.MeanAbsoluteError(name='mae')],
    )

    phase1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mean_ciou",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_ciou",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            f"license_plate_detector_phase1_{timestamp}.keras",
            monitor="val_mean_ciou",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]
    
    print("\nPhase 1: Training head (20 epochs with aggressive augmentation)...")
    detector.train(
        x_train, y_train,
        x_val=x_val, y_val=y_val,
        sample_weight=train_weights,
        epochs=20,
        batch_size=16,
        callbacks=phase1_callbacks,
    )

    # ===== PHASE 2: Fine-tune Top Backbone Layers =====
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning Top Backbone Layers")
    print("="*60)
    
    detector.set_backbone_trainable(True, fine_tune_at=200)
    detector.compile(
        optimizer=create_optimizer(learning_rate=5e-5),
        loss=BoxRegressionLoss(),
        metrics=[BoundingBoxMetric(), keras.metrics.MeanAbsoluteError(name='mae')],
    )

    phase2_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mean_ciou",
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_ciou",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            f"license_plate_detector_phase2_{timestamp}.keras",
            monitor="val_mean_ciou",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nPhase 2: Fine-tuning backbone (30 epochs)...")
    detector.train(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        sample_weight=train_weights,
        epochs=30,
        batch_size=16,
        callbacks=phase2_callbacks,
    )

    # ===== PHASE 3: Full Network Fine-tuning =====
    print("\n" + "="*60)
    print("PHASE 3: Full Network Fine-tuning with Lower LR")
    print("="*60)
    
    detector.set_backbone_trainable(True, fine_tune_at=100)
    detector.compile(
        optimizer=create_optimizer(learning_rate=1e-5),
        loss=BoxRegressionLoss(),
        metrics=[BoundingBoxMetric(), keras.metrics.MeanAbsoluteError(name='mae')],
    )

    phase3_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mean_ciou",
            mode="max",
            patience=12,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_ciou",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=5e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            f"license_plate_detector_phase3_{timestamp}.keras",
            monitor="val_mean_ciou",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nPhase 3: Full network fine-tuning (25 epochs)...")
    detector.train(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        sample_weight=train_weights,
        epochs=25,
        batch_size=16,
        callbacks=phase3_callbacks,
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    test_loss, test_ciou, test_mae = detector.model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test CIoU: {test_ciou:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save final model
    print("\n7. Saving final model...")
    detector.save('license_plate_detector.keras')
    
    # Visualize predictions
    print("\n8. Visualizing predictions...")
    pred_test = detector.predict(x_test)
    visualize_predictions(x_test, y_test, pred_test, num_samples=5)
    
    print("\n" + "="*60)
    print("=== Training Complete ===")
    print("="*60)
    print(f"Model saved as: license_plate_detector.keras")
    print(f"Checkpoint models: license_plate_detector_phase*.keras")
    print("\nRun inference with:")
    print("  from inference import PlateDetector")
    print("  detector = PlateDetector('license_plate_detector.keras')")
    print("  detector.detect_and_draw('image.jpg', 'output.jpg')")



if __name__ == "__main__":
    main()
