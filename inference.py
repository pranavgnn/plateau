"""Type-safe inference script for license plate detection."""

from typing import Tuple, List
from pathlib import Path
import cv2
import numpy as np
from model import LicensePlateDetectionModel


class PlateDetector:
    """Inference wrapper for license plate detection."""
    
    def __init__(self, model_path: str):
        self.detector = LicensePlateDetectionModel()
        self.detector.load(model_path)
    
    def detect(
        self,
        image_path: str,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[np.ndarray, List[float]]:
        """Detect plate bbox in image. Returns (image, [xmin, ymin, xmax, ymax])."""
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load {image_path}")
        
        original_img = img.copy()
        h, w = img.shape[:2]
        
        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Predict
        img_batch = np.expand_dims(img_normalized, axis=0)
        bbox = self.detector.predict(img_batch)[0]

        xmin = float(np.clip(min(bbox[0], bbox[2]), 0.0, 1.0))
        ymin = float(np.clip(min(bbox[1], bbox[3]), 0.0, 1.0))
        xmax = float(np.clip(max(bbox[0], bbox[2]), 0.0, 1.0))
        ymax = float(np.clip(max(bbox[1], bbox[3]), 0.0, 1.0))
        
        # Denormalize to original image size
        bbox_original = np.array([
            xmin * w,
            ymin * h,
            xmax * w,
            ymax * h
        ], dtype=int)
        
        return original_img, bbox_original.tolist()
    
    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding box on image."""
        
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        
        # Clip to image bounds
        h, w = image.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w))
        ymax = max(0, min(ymax, h))
        
        result = image.copy()
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, thickness)
        
        return result
    
    def detect_and_draw(
        self,
        image_path: str,
        output_path: str
    ) -> List[float]:
        """Detect plate and save image with bbox drawn."""
        
        img, bbox = self.detect(image_path)
        img_with_bbox = self.draw_bbox(img, bbox)
        
        # Convert RGB back to BGR for OpenCV
        img_with_bbox = cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_with_bbox)
        
        print(f"Saved result to {output_path}")
        print(f"Detected bbox: {bbox}")
        
        return bbox


def batch_detect(
    detector: PlateDetector,
    image_paths: List[str],
    output_dir: str = "detections/"
) -> List[Tuple[str, List[float]]]:
    """Batch detection on multiple images."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for image_path in image_paths:
        try:
            output_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"detected_{output_name}")
            
            bbox = detector.detect_and_draw(image_path, output_path)
            results.append((image_path, bbox))
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results


# Example usage
if __name__ == "__main__":
    # Initialize detector
    model_path = Path("license_plate_detector.keras")
    if not model_path.exists():
        model_path = Path("license_plate_detector.h5")

    detector = PlateDetector(str(model_path))
    
    # Single image detection
    image_path = "path/to/image.png"  # Replace with actual image path
    try:
        bbox = detector.detect_and_draw(image_path, "output.png")
    except Exception as e:
        print(f"Detection failed: {e}")
    
    # Batch detection (optional)
    # image_paths = ["img1.png", "img2.png", "img3.png"]
    # results = batch_detect(detector, image_paths)
