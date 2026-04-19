"""Improved PyTorch inference script for license plate detection with confidence filtering."""

from typing import Tuple, List, Optional
from pathlib import Path
import cv2
import numpy as np
import torch

# Try to import improved model, fall back to original
try:
    from model_improved import LicensePlateDetectionModelImproved as PlateDetectionModel
    USE_IMPROVED = True
except ImportError:
    from model import LicensePlateDetectionModel as PlateDetectionModel
    USE_IMPROVED = False


def find_best_checkpoint(model_path: Optional[str] = None) -> str:
    """Find the best available checkpoint."""
    if model_path and Path(model_path).exists():
        print(f"✓ Using specified model: {model_path}")
        return model_path
    
    def is_valid_checkpoint(filepath: str) -> bool:
        try:
            torch.load(filepath, map_location='cpu')
            return True
        except Exception as e:
            print(f"  Invalid checkpoint: {str(e)[:60]}...")
            return False
    
    # Search for checkpoints
    all_checkpoints = [
        ("Phase 3", list(Path(".").glob("license_plate_detector_phase3_*.pt"))),
        ("Phase 2", list(Path(".").glob("license_plate_detector_phase2_*.pt"))),
        ("Phase 1", list(Path(".").glob("license_plate_detector_phase1_*.pt"))),
        ("Final", [Path("license_plate_detector.pt")] if Path("license_plate_detector.pt").exists() else []),
    ]
    
    for phase_name, checkpoints in all_checkpoints:
        if checkpoints:
            sorted_checkpoints = sorted([c for c in checkpoints if c.exists()], 
                                       key=lambda p: p.stat().st_mtime, 
                                       reverse=True)
            
            for checkpoint in sorted_checkpoints:
                checkpoint_str = str(checkpoint)
                print(f"Checking {phase_name} checkpoint: {checkpoint_str}...", end=" ")
                
                if is_valid_checkpoint(checkpoint_str):
                    print(f"✓ Valid!")
                    return checkpoint_str
                else:
                    print(f"⚠ Corrupted, skipping...")
    
    raise FileNotFoundError(
        "No valid checkpoint found! Run training again:\n"
        "  python train.py"
    )


class PlateDetector:
    """Improved inference wrapper with confidence filtering."""
    
    def __init__(self, model_path: Optional[str] = None, device: torch.device | None = None, 
                 confidence_threshold: float = 0.5):
        try:
            resolved_path = find_best_checkpoint(model_path)
        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            raise
        
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.detector = PlateDetectionModel(device=self.device)
        self.confidence_threshold = confidence_threshold
        self.use_improved = USE_IMPROVED
        
        try:
            self.detector.load(resolved_path)
        except RuntimeError as e:
            print(f"\n❌ Checkpoint file is corrupted: {resolved_path}")
            raise
        
        self.detector.eval()
        self._tta_scales = [0.85, 1.0, 1.15]
        self._use_tta = True
    
    def _preprocess(self, img_bgr: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """BGR image → normalized RGB tensor [1, C, H, W] (CHW format)."""
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        # Transpose from HWC to CHW format
        img_chw = np.transpose(img, (2, 0, 1))
        return torch.tensor(np.expand_dims(img_chw, axis=0), dtype=torch.float32).to(self.device)

    def _predict_single(self, img_batch: torch.Tensor) -> np.ndarray:
        """Run model on batch and return predictions."""
        with torch.no_grad():
            output = self.detector(img_batch)
        return output[0].cpu().numpy()

    def _predict_with_tta(
        self,
        img_bgr: np.ndarray,
        target_size: Tuple[int, int] = (320, 320),
    ) -> np.ndarray:
        """Test-time augmentation: multi-scale + flip, average results."""
        all_preds: List[np.ndarray] = []

        for scale in self._tta_scales:
            scaled_size = (int(target_size[0] * scale), int(target_size[1] * scale))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_scaled = cv2.resize(img_rgb, scaled_size)
            img_final = self._center_crop_or_pad(img_scaled, target_size)
            img_norm = img_final.astype(np.float32) / 255.0
            # Transpose from HWC to CHW format
            img_norm_chw = np.transpose(img_norm, (2, 0, 1))
            batch = torch.tensor(np.expand_dims(img_norm_chw, axis=0), dtype=torch.float32).to(self.device)

            # Original orientation
            pred = self._predict_single(batch)
            pred_adjusted = self._undo_scale_offset(pred, scale, target_size)
            all_preds.append(pred_adjusted)

            # Horizontal flip
            flipped = np.ascontiguousarray(batch.cpu().numpy()[:, :, ::-1, :])
            flipped = torch.tensor(flipped, dtype=torch.float32).to(self.device)
            pred_flip = self._predict_single(flipped)
            pred_unflip = np.array([
                1.0 - pred_flip[2],
                pred_flip[1],
                1.0 - pred_flip[0],
                pred_flip[3],
            ] + ([pred_flip[4]] if len(pred_flip) > 4 else []), dtype=np.float32)
            pred_unflip_adjusted = self._undo_scale_offset(pred_unflip, scale, target_size)
            all_preds.append(pred_unflip_adjusted)

        # Average all predictions
        avg_pred = np.mean(all_preds, axis=0)
        
        # Clip to [0, 1]
        if len(avg_pred) > 4:  # Has confidence
            avg_pred[:4] = np.clip(avg_pred[:4], 0.0, 1.0)
            avg_pred[4] = np.clip(avg_pred[4], 0.0, 1.0)
        else:
            avg_pred = np.clip(avg_pred, 0.0, 1.0)
        
        return avg_pred

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

        bbox = pred[:4]
        conf = pred[4:] if len(pred) > 4 else np.array([])

        if scale >= 1.0:
            x_off = (sw - tw) / 2.0 / sw
            y_off = (sh - th) / 2.0 / sh
            bbox_adj = np.array([
                bbox[0] / scale + x_off * scale,
                bbox[1] / scale + y_off * scale,
                bbox[2] / scale + x_off * scale,
                bbox[3] / scale + y_off * scale,
            ], dtype=np.float32).clip(0, 1)
        else:
            x_off = (tw - sw) / 2.0 / tw
            y_off = (th - sh) / 2.0 / th
            bbox_adj = np.array([
                (bbox[0] - x_off) * tw / sw,
                (bbox[1] - y_off) * th / sh,
                (bbox[2] - x_off) * tw / sw,
                (bbox[3] - y_off) * th / sh,
            ], dtype=np.float32).clip(0, 1)
        
        return np.concatenate([bbox_adj, conf]) if len(conf) > 0 else bbox_adj

    def detect(
        self,
        image_path: str,
        target_size: Tuple[int, int] = (320, 320)
    ) -> Tuple[np.ndarray, List[float], float]:
        """Detect plate bbox. Returns (image, [xmin, ymin, xmax, ymax], confidence)."""
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load {image_path}")
        
        original_img = img.copy()
        h, w = img.shape[:2]
        
        # Predict with TTA
        if self._use_tta:
            pred = self._predict_with_tta(img, target_size)
        else:
            img_batch = self._preprocess(img, target_size)
            pred = self._predict_single(img_batch)

        # Extract bbox and confidence
        if len(pred) > 4:  # Improved model with confidence
            bbox = pred[:4]
            confidence = float(pred[4])
        else:  # Old model without confidence
            bbox = pred
            confidence = 1.0  # Assume high confidence for old model

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
        
        return original_img, bbox_original.tolist(), confidence
    
    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: List[float],
        confidence: float = 1.0,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding box and confidence on image."""
        
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        
        h, w = image.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w))
        ymax = max(0, min(ymax, h))
        
        result = image.copy()
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, thickness)
        
        # Draw confidence text
        if confidence < 1.0:
            text = f"Conf: {confidence:.2f}"
            cv2.putText(result, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def detect_and_draw(
        self,
        image_path: str,
        output_path: str
    ) -> Tuple[List[float], float]:
        """Detect plate and save image with bbox drawn."""
        
        img, bbox, confidence = self.detect(image_path)
        
        # Only draw if confidence is above threshold
        if confidence >= self.confidence_threshold:
            img_with_bbox = self.draw_bbox(img, bbox, confidence)
            cv2.imwrite(output_path, img_with_bbox)
            return bbox, confidence
        else:
            # Still save image but with warning
            cv2.imwrite(output_path, img)
            return bbox, confidence


if __name__ == "__main__":
    import os
    
    print(f"Using {'improved' if USE_IMPROVED else 'original'} model\n")
    
    model_path = Path("license_plate_detector.pt")
    if not model_path.exists():
        model_path = Path("license_plate_detector.keras")

    # Lower threshold for better detection (0.3-0.4 recommended)
    detector = PlateDetector(str(model_path), confidence_threshold=0.3)
    
    output_dir = "predictions_output"
    os.makedirs(output_dir, exist_ok=True)
    
    images_dir = "images"
    if not os.path.exists(images_dir):
        print(f"❌ '{images_dir}' folder not found!")
        print(f"📁 Please create a '{images_dir}' folder and add images to it.")
    else:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG', '.JPEG'}
        image_files = [
            f for f in os.listdir(images_dir) 
            if os.path.splitext(f)[1] in image_extensions
        ]
        
        if not image_files:
            print(f"❌ No images found in '{images_dir}' folder!")
        else:
            print(f"✓ Found {len(image_files)} image(s)\n")
            print("Processing...")
            print("="*60)
            
            for image_file in image_files:
                input_path = os.path.join(images_dir, image_file)
                output_filename = f"detected_{os.path.splitext(image_file)[0]}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                try:
                    bbox, confidence = detector.detect_and_draw(input_path, output_path)
                    status = "✓" if confidence >= detector.confidence_threshold else "⚠"
                    print(f"{status} {image_file}")
                    print(f"  → Confidence: {confidence:.3f}")
                    print(f"  → Bbox: {bbox}\n")
                except Exception as e:
                    print(f"✗ {image_file}")
                    print(f"  → Error: {e}\n")
            
            print("="*60)
            print(f"✓ Results saved to '{output_dir}/' folder")
