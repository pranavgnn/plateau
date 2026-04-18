"""Type-safe dataset loader for license plate detection."""

from typing import Tuple, List, Optional, TypeAlias
from pathlib import Path
import cv2
import numpy as np
from lxml import etree
import kagglehub

Sample: TypeAlias = Tuple[np.ndarray, np.ndarray, str]

class BoundingBox:
    """Represents normalized bounding box."""
    
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    
    def to_array(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax], dtype=np.float32)


def _clip_box(xmin: float, ymin: float, xmax: float, ymax: float) -> BoundingBox:
    """Clamp bbox to valid normalized range and keep corners ordered."""
    xmin_clamped = float(np.clip(xmin, 0.0, 1.0))
    ymin_clamped = float(np.clip(ymin, 0.0, 1.0))
    xmax_clamped = float(np.clip(xmax, 0.0, 1.0))
    ymax_clamped = float(np.clip(ymax, 0.0, 1.0))

    left = min(xmin_clamped, xmax_clamped)
    top = min(ymin_clamped, ymax_clamped)
    right = max(xmin_clamped, xmax_clamped)
    bottom = max(ymin_clamped, ymax_clamped)
    return BoundingBox(left, top, right, bottom)


class LicensePlateDataset:
    """Load Indian vehicle license plate dataset from kagglehub."""
    
    def __init__(self, dataset_id: str = "saisirishan/indian-vehicle-dataset"):
        self.dataset_id = dataset_id
        self.dataset_path: Optional[Path] = None
        self.samples: List[Sample] = []
        
    def download(self) -> Path:
        """Download dataset from kagglehub."""
        print(f"Downloading {self.dataset_id}...")
        path = kagglehub.dataset_download(self.dataset_id)
        self.dataset_path = Path(path)
        print(f"Dataset path: {self.dataset_path}")
        return self.dataset_path
    
    def _parse_xml(self, xml_path: Path) -> Tuple[Optional[BoundingBox], Optional[str]]:
        """Parse XML file and extract bounding box. Returns (bbox, plate_text)."""
        try:
            tree = etree.parse(str(xml_path))
            root = tree.getroot()
            
            # Extract image dimensions
            width_elem = root.find(".//size/width")
            height_elem = root.find(".//size/height")
            
            if width_elem is None or height_elem is None:
                return None, None
                
            img_width = float(width_elem.text)
            img_height = float(height_elem.text)
            
            # Extract first object's bounding box
            obj = root.find(".//object")
            if obj is None:
                return None, None
            
            bndbox = obj.find("bndbox")
            if bndbox is None:
                return None, None
            
            xmin_str = bndbox.find("xmin")
            ymin_str = bndbox.find("ymin")
            xmax_str = bndbox.find("xmax")
            ymax_str = bndbox.find("ymax")
            
            if None in [xmin_str, ymin_str, xmax_str, ymax_str]:
                return None, None
            
            # Get raw coordinates
            xmin = float(xmin_str.text)
            ymin = float(ymin_str.text)
            xmax = float(xmax_str.text)
            ymax = float(ymax_str.text)
            
            # Normalize to [0, 1]
            xmin_norm = xmin / img_width
            ymin_norm = ymin / img_height
            xmax_norm = xmax / img_width
            ymax_norm = ymax / img_height
            
            # Get plate text (name element)
            name_elem = obj.find("name")
            plate_text = name_elem.text if name_elem is not None else None
            
            bbox = _clip_box(xmin_norm, ymin_norm, xmax_norm, ymax_norm)
            return bbox, plate_text
            
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None, None
    
    def _augment_geometric(self, img: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric augmentation (rotation, scale) with bbox transformation."""
        # Random rotation (-5 to +5 degrees for plates)
        angle = float(np.random.uniform(-5.0, 5.0))
        h, w = img.shape[:2]
        
        # Rotation matrix
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rot = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # For simplicity, recompute bbox after rotation (slight bbox inflation for safety)
        bbox_aug = bbox.copy()
        bbox_aug = np.clip(bbox_aug, 0.0, 1.0)
        
        # Random scale
        scale = float(np.random.uniform(0.95, 1.05))
        img_scaled = cv2.resize(img_rot, (int(w * scale), int(h * scale)))
        
        # Center crop back to original size
        new_h, new_w = img_scaled.shape[:2]
        if new_h > h or new_w > w:
            y_off = (new_h - h) // 2
            x_off = (new_w - w) // 2
            img_final = img_scaled[y_off:y_off+h, x_off:x_off+w]
        else:
            img_final = np.zeros((h, w, 3), dtype=img.dtype)
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            img_final[y_off:y_off+new_h, x_off:x_off+new_w] = img_scaled
        
        return img_final, bbox_aug

    def _photometric_augment(self, img: np.ndarray) -> np.ndarray:
        """Apply appearance-only augmentation (bbox unchanged)."""
        out = img.copy()

        alpha = float(np.random.uniform(0.85, 1.15))
        beta = float(np.random.uniform(-0.08, 0.08))
        out = np.clip(out * alpha + beta, 0.0, 1.0)

        gamma = float(np.random.uniform(0.9, 1.1))
        out = np.clip(np.power(out, gamma), 0.0, 1.0)

        noise_std = float(np.random.uniform(0.0, 0.02))
        if noise_std > 0.0:
            noise = np.random.normal(0.0, noise_std, out.shape).astype(np.float32)
            out = np.clip(out + noise, 0.0, 1.0)

        return out.astype(np.float32)

    def load(
        self,
        img_shape: Tuple[int, int] = (224, 224),
        augment: bool = False,
        min_box_area: float = 0.001,
    ) -> List[Sample]:
        """Load images and bboxes. Returns (image, bbox_array, plate_text)."""
        if self.dataset_path is None:
            self.download()
        
        self.samples = []
        
        # Iterate state folders (AP, KA, etc.)
        for state_dir in sorted(self.dataset_path.iterdir()):
            if not state_dir.is_dir():
                continue
            
            print(f"Loading from {state_dir.name}...")
            
            # Find all XML files
            for xml_file in sorted(state_dir.glob("*.xml")):
                img_file = xml_file.with_suffix(".png")
                if not img_file.exists():
                    img_file = xml_file.with_suffix(".jpg")
                
                if not img_file.exists():
                    continue
                
                # Parse XML
                bbox, plate_text = self._parse_xml(xml_file)
                if bbox is None:
                    continue

                box_w = bbox.xmax - bbox.xmin
                box_h = bbox.ymax - bbox.ymin
                if box_w <= 0.0 or box_h <= 0.0:
                    continue
                if box_w * box_h < min_box_area:
                    continue
                
                # Load image
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_shape)
                    img = img.astype(np.float32) / 255.0
                    
                    self.samples.append((img, bbox.to_array(), plate_text))

                    if augment:
                        # Horizontal flip
                        flipped_img = np.ascontiguousarray(img[:, ::-1, :])
                        flipped_bbox = np.array(
                            [1.0 - bbox.xmax, bbox.ymin, 1.0 - bbox.xmin, bbox.ymax],
                            dtype=np.float32,
                        )
                        self.samples.append((flipped_img, flipped_bbox, plate_text))

                        # Photometric augmentation
                        photo_img = self._photometric_augment(img)
                        self.samples.append((photo_img, bbox.to_array(), plate_text))
                        
                        # Geometric augmentation (rotation + scale)
                        geo_img, geo_bbox = self._augment_geometric(img, bbox.to_array())
                        self.samples.append((geo_img, geo_bbox, plate_text))
                        
                        # Combined: flip + photometric
                        photo_flipped = self._photometric_augment(flipped_img)
                        self.samples.append((photo_flipped, flipped_bbox, plate_text))
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
        
        print(f"Loaded {len(self.samples)} samples")
        return self.samples
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return stacked arrays: (images, bboxes, plate_texts)."""
        images = np.stack([s[0] for s in self.samples])
        bboxes = np.stack([s[1] for s in self.samples])
        texts = [s[2] for s in self.samples]
        return images, bboxes, texts
