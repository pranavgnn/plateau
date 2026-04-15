# Plateau — License Plate Detection ANN

Type-safe Python implementation of CNN + ANN for detecting Indian vehicle license plates.

## Structure

- **dataset_loader.py** — Download & parse XML annotations, normalize bboxes
- **model.py** — MobileNetV2 backbone + ANN head for bbox regression
- **train.py** — Main training pipeline with train/val/test split
- **inference.py** — Inference wrapper for single/batch predictions
- **config.py** — Hyperparameters
- **requirements.txt** — Dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Format

Indian vehicle dataset from Kaggle. Organized by state folders (AP, KA, etc.):

```
dataset/
├── AP/
│   ├── AN1.png
│   ├── AN1.xml
│   └── ...
├── KA/
│   └── ...
```

XML format (`bndbox` section):
```xml
<annotation>
  <size>
    <width>272</width>
    <height>575</height>
  </size>
  <object>
    <name>AN01P9687</name>
    <bndbox>
      <xmin>94</xmin>
      <ymin>364</ymin>
      <xmax>185</xmax>
      <ymax>385</ymax>
    </bndbox>
  </object>
</annotation>
```

Loader extracts raw coords, normalizes to [0, 1].

## Training

```bash
python train.py
```

Pipeline:
1. Download dataset via kagglehub
2. Parse XML + load images
3. Normalize bboxes to [0, 1]
4. Split 70/15/15 (train/val/test)
5. Build CNN backbone (MobileNetV2, pretrained) + ANN head
6. Train 50 epochs with early stopping
7. Save model + visualize predictions

**Output:**
- `license_plate_detector.h5` — Trained model
- `predictions.png` — Test predictions visualization

## Model Architecture

```
Input (224×224×3)
  ↓
MobileNetV2 (pretrained, frozen)
  ↓
GlobalAveragePooling2D
  ↓
Dense 512 + ReLU + BatchNorm + Dropout(0.3)
  ↓
Dense 256 + ReLU + BatchNorm + Dropout(0.3)
  ↓
Dense 128 + ReLU + BatchNorm + Dropout(0.2)
  ↓
Dense 4 + Sigmoid → [xmin, ymin, xmax, ymax] ∈ [0, 1]
```

**Loss:** MSE (can swap for IoU loss for better bbox accuracy)

## Inference

Single image:
```python
from inference import PlateDetector

detector = PlateDetector("license_plate_detector.h5")
bbox = detector.detect_and_draw("test_image.png", "output.png")
print(bbox)  # [xmin, ymin, xmax, ymax] in original image coords
```

Batch:
```python
image_paths = ["img1.png", "img2.png", "img3.png"]
results = batch_detect(detector, image_paths, output_dir="detections/")
```

## Type Safety

Full type hints throughout:
- `BoundingBox` class for bbox objects
- Function signatures typed (e.g., `def train(...) -> keras.callbacks.History`)
- NumPy arrays typed for clarity
- Optional types for nullable values

## Configuration

Edit `config.py` to tune:
- Model architecture (dense layer sizes, dropout)
- Training (epochs, batch size, learning rate)
- Loss function (MSE vs IoU)
- Data split ratios

## Loss Options

**MSE** (default):
- Simple, fast convergence
- Treats all errors equally

**IoU Loss** (custom implementation):
- Better for bbox regression
- Focuses on overlap accuracy
- Use: `detector.compile(loss=IoULoss())`

## Performance Tips

1. Increase MobileNetV2 trainable layers for more capacity:
   ```python
   backbone.trainable = True
   ```

2. Try IoU loss for better bbox accuracy:
   ```python
   from model import IoULoss
   detector.compile(loss=IoULoss())
   ```

3. Data augmentation (add in dataset_loader):
   - Rotation, brightness, horizontal flip (preserve bbox coords!)

4. Ensemble multiple checkpoints for robustness

## Output Interpretation

Model outputs normalized coords [0, 1]. To get pixel coords:
```python
xmin_px = bbox[0] * image_width
ymin_px = bbox[1] * image_height
xmax_px = bbox[2] * image_width
ymax_px = bbox[3] * image_height
```

`inference.py` handles this automatically in `detect()`.

## Next Steps

1. Extract plate region using predicted bbox
2. OCR on cropped region (e.g., EasyOCR, Tesseract)
3. Validate formatting (Indian plate rules)
4. Deploy as API (FastAPI)
