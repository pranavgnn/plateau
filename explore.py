"""Type-safe utilities for data exploration and validation."""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import LicensePlateDataset


def explore_dataset() -> None:
    """Explore dataset statistics."""
    
    print("=== Plateau Dataset Exploration ===\n")
    
    # Load dataset
    dataset = LicensePlateDataset()
    dataset.download()
    dataset.load(img_shape=(224, 224))
    
    images, bboxes, plate_texts = dataset.get_arrays()
    
    # Statistics
    print(f"Total samples: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Bbox shape: {bboxes.shape}")
    print()
    
    # Bbox statistics
    print("Bounding Box Statistics (normalized [0, 1]):")
    print(f"  xmin: mean={bboxes[:, 0].mean():.3f}, std={bboxes[:, 0].std():.3f}")
    print(f"  ymin: mean={bboxes[:, 1].mean():.3f}, std={bboxes[:, 1].std():.3f}")
    print(f"  xmax: mean={bboxes[:, 2].mean():.3f}, std={bboxes[:, 2].std():.3f}")
    print(f"  ymax: mean={bboxes[:, 3].mean():.3f}, std={bboxes[:, 3].std():.3f}")
    print()
    
    # Bbox sizes
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    print("Bbox Size Statistics (normalized):")
    print(f"  Width: mean={widths.mean():.3f}, std={widths.std():.3f}, "
          f"min={widths.min():.3f}, max={widths.max():.3f}")
    print(f"  Height: mean={heights.mean():.3f}, std={heights.std():.3f}, "
          f"min={heights.min():.3f}, max={heights.max():.3f}")
    print()
    
    # Plate text distribution
    plate_counts: dict = {}
    for plate in plate_texts:
        plate_counts[plate] = plate_counts.get(plate, 0) + 1
    
    print(f"Unique plates: {len(plate_counts)}")
    print("Top 10 plates:")
    for plate, count in sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {plate}: {count}")
    print()


def visualize_dataset_samples(num_samples: int = 9) -> None:
    """Visualize dataset samples with bboxes."""
    
    dataset = LicensePlateDataset()
    dataset.download()
    samples = dataset.load(img_shape=(224, 224))
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (img, bbox, plate_text) in enumerate(samples[:num_samples]):
        ax = axes[idx]
        ax.imshow((img * 255).astype(np.uint8))
        
        h, w = img.shape[:2]
        rect = plt.Rectangle(
            (bbox[0] * w, bbox[1] * h),
            (bbox[2] - bbox[0]) * w,
            (bbox[3] - bbox[1]) * h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title(f"Plate: {plate_text}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=100)
    print("Saved dataset_samples.png")
    plt.close()


def validate_annotations() -> Tuple[int, int]:
    """Validate XML annotations. Returns (valid_count, invalid_count)."""
    
    dataset = LicensePlateDataset()
    dataset.download()
    
    valid = 0
    invalid = 0
    
    for state_dir in sorted(dataset.dataset_path.iterdir()):
        if not state_dir.is_dir():
            continue
        
        for xml_file in state_dir.glob("*.xml"):
            bbox, plate_text = dataset._parse_xml(xml_file)
            
            if bbox is not None:
                valid += 1
            else:
                invalid += 1
                print(f"Invalid: {xml_file}")
    
    print(f"\nValidation Summary:")
    print(f"  Valid: {valid}")
    print(f"  Invalid: {invalid}")
    print(f"  Total: {valid + invalid}")
    
    return valid, invalid


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "explore":
            explore_dataset()
        elif command == "visualize":
            visualize_dataset_samples(num_samples=9)
        elif command == "validate":
            validate_annotations()
        else:
            print("Usage: python explore.py [explore|visualize|validate]")
    else:
        print("Usage: python explore.py [explore|visualize|validate]")
