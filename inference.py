"""PyTorch inference script for license plate detection.

Improvements over v1:
  - Test-Time Augmentation (TTA): multi-scale + horizontal flip
  - Averages predictions across augmented views for robustness
  - Updated to 320×320 input resolution
  - GPU acceleration support
  - Auto-detects available checkpoints and resumes from best available
"""

from typing import Tuple, List, Optional
from pathlib import Path
import cv2
import numpy as np
import torch
from model import LicensePlateDetectionModel


def find_best_checkpoint(model_path: Optional[str] = None) -> str:
    """Find the best available checkpoint. Priority: Phase 3 > Phase 2 > Phase 1 > final.
    
    Args:
        model_path: Explicit model path. If provided, use it directly.
        
    Returns:
        Path to the checkpoint to load.
        
    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    if model_path and Path(model_path).exists():
        print(f"✓ Using specified model: {model_path}")
        return model_path
    
    # Search for checkpoints in priority order
    phase3_checkpoint = list(Path(".").glob("license_plate_detector_phase3_*.pt"))
    phase2_checkpoint = list(Path(".").glob("license_plate_detector_phase2_*.pt"))
    phase1_checkpoint = list(Path(".").glob("license_plate_detector_phase1_*.pt"))
    final_checkpoint = Path("license_plate_detector.pt")
    
    if phase3_checkpoint:
        checkpoint = str(max(phase3_checkpoint, key=lambda p: p.stat().st_mtime))
        print(f"✓ Found Phase 3 checkpoint: {checkpoint}")
        return checkpoint
    
    if phase2_checkpoint:
        checkpoint = str(max(phase2_checkpoint, key=lambda p: p.stat().st_mtime))
        print(f"✓ Found Phase 2 checkpoint: {checkpoint} (Phase 3 not yet trained)")
        return checkpoint
    
    if phase1_checkpoint:
        checkpoint = str(max(phase1_checkpoint, key=lambda p: p.stat().st_mtime))
        print(f"✓ Found Phase 1 checkpoint: {checkpoint} (Phase 2-3 not yet trained)")
        return checkpoint
    
    if final_checkpoint.exists():
        print(f"✓ Using final model: {final_checkpoint}")
        return str(final_checkpoint)
    
    raise FileNotFoundError(
        "No checkpoint found! Available options:\n"
        "  1. Run training first: python train.py\n"
        "  2. Specify model path: PlateDetector('model_path.pt')\n"
        "  3. Place checkpoint in current directory"
    )


class PlateDetector:
    """Inference wrapper for license plate detection with TTA."""
    
    def __init__(self, model_path: Optional[str] = None, device: torch.device | None = None):
        # Auto-detect best checkpoint if not specified
        resolved_path = find_best_checkpoint(model_path)
        
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.detector = LicensePlateDetectionModel(device=self.device)
        self.detector.load(resolved_path)
        self.detector.eval()
        self._tta_scales = [0.85, 1.0, 1.15]
        self._use_tta = True
    
    def _preprocess(self, img_bgr: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """BGR image → normalized RGB tensor [1, H, W, 3]."""
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        return torch.tensor(np.expand_dims(img, axis=0), dtype=torch.float32).to(self.device)

    def _predict_single(self, img_batch: torch.Tensor) -> np.ndarray:
        """Run model on a single [1, H, W, 3] batch and return [4] bbox."""
        with torch.no_grad():
            output = self.detector(img_batch)
        return output[0].cpu().numpy()

    def _predict_with_tta(
        self,
        img_bgr: np.ndarray,
        target_size: Tuple[int, int] = (320, 320),
    ) -> np.ndarray:
        """Run test-time augmentation: multi-scale + flip, average results."""
        all_preds: List[np.ndarray] = []

        for scale in self._tta_scales:
            scaled_size = (int(target_size[0] * scale), int(target_size[1] * scale))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_scaled = cv2.resize(img_rgb, scaled_size)
            # Center crop / pad to target_size
            img_final = self._center_crop_or_pad(img_scaled, target_size)
            img_norm = img_final.astype(np.float32) / 255.0
            batch = torch.tensor(np.expand_dims(img_norm, axis=0), dtype=torch.float32).to(self.device)

            # Original orientation
            pred = self._predict_single(batch)
            # Undo scale offset
            pred_adjusted = self._undo_scale_offset(pred, scale, target_size)
            all_preds.append(pred_adjusted)

            # Horizontal flip
            flipped = np.ascontiguousarray(batch.cpu().numpy()[:, :, ::-1, :])
            flipped = torch.tensor(flipped, dtype=torch.float32).to(self.device)
            pred_flip = self._predict_single(flipped)
            # Un-flip x coordinates
            pred_unflip = np.array([
                1.0 - pred_flip[2],
                pred_flip[1],
                1.0 - pred_flip[0],
                pred_flip[3],
            ], dtype=np.float32)
            pred_unflip_adjusted = self._undo_scale_offset(pred_unflip, scale, target_size)
            all_preds.append(pred_unflip_adjusted)

        # Average all predictions
        avg_pred = np.mean(all_preds, axis=0)
        return np.clip(avg_pred, 0.0, 1.0)


    @staticmethod
    def _center_crop_or_pad(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Center-crop or zero-pad to exact target_size (W, H)."""
        tw, th = target_size
        h, w = img.shape[:2]

        if h >= th and w >= tw:
            y_off = (h - th) // 2
            x_off = (w - tw) // 2
            return img[y_off:y_off + th, x_off:x_off + tw]
        else:
            out = np.zeros((th, tw, 3), dtype=img.dtype)
            y_off = max(0, (th - h) // 2)
            x_off = max(0, (tw - w) // 2)
            paste_h = min(h, th)
            paste_w = min(w, tw)
            out[y_off:y_off + paste_h, x_off:x_off + paste_w] = img[:paste_h, :paste_w]
            return out

    @staticmethod
    def _undo_scale_offset(pred: np.ndarray, scale: float, target_size: Tuple[int, int]) -> np.ndarray:
        """Reverse the center-crop/pad offset introduced by scaling."""
        if abs(scale - 1.0) < 1e-6:
            return pred

        tw, th = target_size
        sw, sh = int(tw * scale), int(th * scale)

        if scale >= 1.0:
            x_off = (sw - tw) / 2.0 / sw
            y_off = (sh - th) / 2.0 / sh
            return np.array([
                pred[0] / scale + x_off * scale,
                pred[1] / scale + y_off * scale,
                pred[2] / scale + x_off * scale,
                pred[3] / scale + y_off * scale,
            ], dtype=np.float32).clip(0, 1)
        else:
            x_off = (tw - sw) / 2.0 / tw
            y_off = (th - sh) / 2.0 / th
            return np.array([
                (pred[0] - x_off) * tw / sw,
                (pred[1] - y_off) * th / sh,
                (pred[2] - x_off) * tw / sw,
                (pred[3] - y_off) * th / sh,
            ], dtype=np.float32).clip(0, 1)

    def detect(
        self,
        image_path: str,
        target_size: Tuple[int, int] = (320, 320)
    ) -> Tuple[np.ndarray, List[float]]:
        """Detect plate bbox in image. Returns (image, [xmin, ymin, xmax, ymax])."""
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load {image_path}")
        
        original_img = img.copy()
        h, w = img.shape[:2]
        
        # Predict (with or without TTA)
        if self._use_tta:
            bbox = self._predict_with_tta(img, target_size)
        else:
            img_batch = self._preprocess(img, target_size)
            bbox = self._predict_single(img_batch)

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
    model_path = Path("license_plate_detector.pt")
    if not model_path.exists():
        model_path = Path("license_plate_detector.keras")

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
