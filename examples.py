"""Quick start examples for plateau license plate detection."""

from typing import List
import numpy as np
from tensorflow import keras
from dataset_loader import LicensePlateDataset, BoundingBox
from model import LicensePlateDetectionModel
from inference import PlateDetector


def example_load_dataset() -> None:
    """Example: Load and inspect dataset."""
    print("=== Example 1: Load Dataset ===\n")
    
    dataset = LicensePlateDataset(dataset_id="saisirishan/indian-vehicle-dataset")
    dataset.download()
    
    samples = dataset.load(img_shape=(320, 320))
    print(f"Loaded {len(samples)} samples")
    
    if samples:
        img, bbox, plate_text = samples[0]
        print(f"First sample - Image: {img.shape}, Plate: {plate_text}")
        print(f"Bbox (normalized): {bbox}")


def example_build_model() -> None:
    """Example: Build and inspect model."""
    print("\n=== Example 2: Build Model ===\n")
    
    detector = LicensePlateDetectionModel(input_shape=(320, 320, 3))
    model = detector.build()
    print("Model architecture:")
    detector.get_summary()


def example_quick_train() -> None:
    """Example: Quick training on small subset."""
    print("\n=== Example 3: Quick Train (Small Subset) ===\n")
    
    # Load dataset
    dataset = LicensePlateDataset()
    dataset.download()
    samples = dataset.load(img_shape=(320, 320))
    
    if len(samples) < 100:
        print("Dataset too small for training")
        return
    
    images, bboxes, _ = dataset.get_arrays()
    
    # Use only first 100 samples for quick test
    x_train = images[:80].astype(np.float32)
    y_train = bboxes[:80].astype(np.float32)
    x_val = images[80:100].astype(np.float32)
    y_val = bboxes[80:100].astype(np.float32)
    
    print(f"Train: {x_train.shape}, Val: {x_val.shape}")
    
    # Build & train
    detector = LicensePlateDetectionModel()
    detector.build()
    detector.compile(loss=keras.losses.MeanSquaredError())
    
    history = detector.train(
        x_train, y_train,
        x_val, y_val,
        epochs=5,
        batch_size=16
    )
    
    print("Quick training completed")


def example_inference() -> None:
    """Example: Run inference on test image."""
    print("\n=== Example 4: Inference ===\n")
    
    # This assumes model was already trained and saved
    model_path = "license_plate_detector.h5"
    
    try:
        detector = PlateDetector(model_path)
        print(f"Loaded model from {model_path}")
        
        # Replace with actual test image path
        test_image = "test_plate.png"
        bbox = detector.detect_and_draw(test_image, "detected_plate.png")
        print(f"Detection bbox: {bbox}")
        
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Train model first using train.py")


def example_batch_detection() -> None:
    """Example: Batch detection on multiple images."""
    print("\n=== Example 5: Batch Detection ===\n")
    
    image_paths: List[str] = [
        "image1.png",
        "image2.png",
        "image3.png"
    ]
    
    model_path = "license_plate_detector.h5"
    
    try:
        detector = PlateDetector(model_path)
        
        for image_path in image_paths:
            try:
                img, bbox = detector.detect(image_path)
                print(f"{image_path}: bbox={bbox}")
                img_drawn = detector.draw_bbox(img, bbox)
                
            except Exception as e:
                print(f"Error on {image_path}: {e}")
                
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Train model first.")


def example_custom_bbox():
    """Example: Create custom BoundingBox object."""
    print("\n=== Example 6: Custom BoundingBox ===\n")
    
    # Create normalized bbox (xmin, ymin, xmax, ymax in [0, 1])
    bbox = BoundingBox(xmin=0.3, ymin=0.4, xmax=0.7, ymax=0.8)
    
    print(f"Custom bbox: xmin={bbox.xmin}, ymin={bbox.ymin}, "
          f"xmax={bbox.xmax}, ymax={bbox.ymax}")
    
    # Convert to array
    bbox_array = bbox.to_array()
    print(f"As array: {bbox_array}")


if __name__ == "__main__":
    print("Plateau License Plate Detection - Quick Start Examples\n")
    print("=" * 60)
    
    # Uncomment examples to run:
    
    example_build_model()
    # example_load_dataset()
    # example_quick_train()
    # example_inference()
    # example_batch_detection()
    # example_custom_bbox()
    
    print("\n" + "=" * 60)
    print("For full training, run: python train.py")
    print("For dataset exploration, run: python explore.py explore")
