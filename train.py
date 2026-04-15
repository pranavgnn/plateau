"""Type-safe training script for license plate detection."""

from typing import Tuple
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dataset_loader import LicensePlateDataset
from model import LicensePlateDetectionModel, BoxRegressionLoss, BoundingBoxMetric


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
    """Main training pipeline."""
    
    print("=== License Plate Detection Training ===\n")
    
    # Load dataset
    print("1. Loading dataset...")
    dataset = LicensePlateDataset(dataset_id="saisirishan/indian-vehicle-dataset")
    dataset.load(img_shape=(224, 224), augment=True)
    
    images, bboxes, plate_texts = dataset.get_arrays()
    print(f"Images: {images.shape}, Bboxes: {bboxes.shape}")
    
    # Split data
    print("\n2. Splitting dataset...")
    x_train, x_temp, y_train, y_temp = train_test_split(
        images, bboxes, test_size=0.3, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    
    # Build model
    print("\n3. Building model...")
    detector = LicensePlateDetectionModel(input_shape=(224, 224, 3))
    detector.build()
    detector.get_summary()
    
    # Phase 1: train head while backbone frozen
    print("\n4. Compiling model...")
    detector.set_backbone_trainable(False)
    detector.compile(
        optimizer=None,
        loss=BoxRegressionLoss(),
        metrics=[BoundingBoxMetric(), keras.metrics.MeanAbsoluteError(name='mae')],
    )
    
    print("\n5. Training head..." )
    detector.train(
        x_train, y_train,
        x_val=x_val, y_val=y_val,
        epochs=15,
        batch_size=16
    )

    # Phase 2: fine-tune top backbone layers with lower LR
    print("\n6. Fine-tuning backbone...")
    detector.set_backbone_trainable(True, fine_tune_at=180)
    detector.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=BoxRegressionLoss(),
        metrics=[BoundingBoxMetric(), keras.metrics.MeanAbsoluteError(name='mae')],
    )
    detector.train(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=20,
        batch_size=16,
    )
    
    # Evaluate
    print("\n7. Evaluating on test set...")
    test_loss, test_iou, test_mae = detector.model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test MAE: {test_mae:.4f}")
    
    # Save model
    print("\n8. Saving model...")
    detector.save('license_plate_detector.keras')
    
    # Visualize predictions
    print("\n9. Visualizing predictions...")
    pred_test = detector.predict(x_test)
    visualize_predictions(x_test, y_test, pred_test, num_samples=3)
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
