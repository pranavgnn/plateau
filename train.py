"""PyTorch training script for license plate detection with 3-phase training."""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from dataset_loader import LicensePlateDataset
from model import (
    LicensePlateDetectionModel,
    BoxRegressionLoss,
    box_ciou,
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



class PlateDataset(Dataset):
    """PyTorch Dataset for license plate detection."""
    
    def __init__(self, images: np.ndarray, bboxes: np.ndarray, weights: np.ndarray | None = None):
        # Transpose from HWC to CHW format
        images_chw = np.transpose(images, (0, 3, 1, 2))
        self.images = torch.tensor(images_chw, dtype=torch.float32)
        self.bboxes = torch.tensor(bboxes, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple:
        if self.weights is not None:
            return self.images[idx], self.bboxes[idx], self.weights[idx]
        return self.images[idx], self.bboxes[idx]


def find_latest_checkpoint(pattern: str) -> Optional[str]:
    """Find the most recent checkpoint matching the pattern."""
    checkpoints = list(Path(".").glob(pattern))
    if not checkpoints:
        return None
    # Return the most recently modified checkpoint
    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


def show_training_menu() -> Tuple[str, list]:
    """Show interactive menu to choose training mode and phases.
    
    Returns:
        Tuple of (mode, phases_to_run) where:
        - mode: 'fresh', 'resume', or 'custom'
        - phases_to_run: list of phases to execute (e.g., [1, 2, 3])
    """
    print("\n" + "="*60)
    print("TRAINING MODE SELECTION")
    print("="*60)
    
    # Check for existing checkpoints
    phase1_exists = find_latest_checkpoint("license_plate_detector_phase1_*.pt") is not None
    phase2_exists = find_latest_checkpoint("license_plate_detector_phase2_*.pt") is not None
    phase3_exists = find_latest_checkpoint("license_plate_detector_phase3_*.pt") is not None
    
    print("\nCheckpoint Status:")
    print(f"  Phase 1: {'✓ Found' if phase1_exists else '✗ Not found'}")
    print(f"  Phase 2: {'✓ Found' if phase2_exists else '✗ Not found'}")
    print(f"  Phase 3: {'✓ Found' if phase3_exists else '✗ Not found'}")
    
    print("\n\nTraining Modes:")
    print("  1. Start Fresh (delete all checkpoints, train from scratch)")
    print("  2. Resume Training (auto-detect checkpoint, continue from there)")
    print("  3. Custom Phase Selection (choose specific phases to run)")
    
    while True:
        choice = input("\nSelect mode (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    if choice == '1':
        # Start fresh - confirm deletion
        confirm = input("\n⚠️  This will DELETE all checkpoints. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            for checkpoint in Path(".").glob("license_plate_detector_phase*.pt"):
                checkpoint.unlink()
                print(f"Deleted: {checkpoint}")
            print("✓ All checkpoints deleted. Starting fresh training...")
            return 'fresh', [1, 2, 3]
        else:
            print("✗ Operation cancelled. Exiting.")
            return None, None
    
    elif choice == '2':
        # Resume - determine starting phase
        if phase3_exists:
            print("\n✓ Training already complete (Phase 3 found).")
            resume_choice = input("Run evaluation only? (y/n): ").strip().lower()
            if resume_choice == 'y':
                return 'resume', [0]  # 0 means evaluation only
            else:
                return 'resume', []  # Empty list means skip to evaluation
        elif phase2_exists:
            print("\n✓ Resuming from Phase 3...")
            return 'resume', [3]
        elif phase1_exists:
            print("\n✓ Resuming from Phase 2...")
            return 'resume', [2, 3]
        else:
            print("\n✓ No checkpoints found. Starting fresh...")
            return 'fresh', [1, 2, 3]
    
    else:  # choice == '3'
        # Custom phase selection
        print("\n\nPhase Details:")
        print("  Phase 1: Warmup - Train head with frozen backbone (20 epochs)")
        print("  Phase 2: Fine-tune top backbone layers (30 epochs)")
        print("  Phase 3: Full network fine-tuning with lower LR (25 epochs)")
        
        phases_to_run = []
        for phase_num in [1, 2, 3]:
            include = input(f"\nInclude Phase {phase_num}? (y/n): ").strip().lower()
            if include == 'y':
                phases_to_run.append(phase_num)
        
        if not phases_to_run:
            print("✗ No phases selected. Exiting.")
            return None, None
        
        # Warn about skipping earlier phases
        if 1 not in phases_to_run and (2 in phases_to_run or 3 in phases_to_run):
            print("\n⚠️  Warning: Skipping Phase 1. Model may not train well.")
        
        return 'custom', phases_to_run


def train_phase(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    phase_name: str,
) -> None:
    """Train model for one phase."""
    print(f"\n{phase_name}: Training for {epochs} epochs...")
    
    best_val_ciou = -float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_ciou = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data[0].to(device)
            bboxes = batch_data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(bboxes, outputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                ciou = torch.mean(box_ciou(bboxes, outputs))
                train_ciou += ciou.item()
            
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_ciou /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_ciou = 0.0
        
        with torch.no_grad():
            for images, bboxes in val_loader:
                images = images.to(device)
                bboxes = bboxes.to(device)
                
                outputs = model(images)
                loss = criterion(bboxes, outputs)
                
                val_loss += loss.item()
                ciou = torch.mean(box_ciou(bboxes, outputs))
                val_ciou += ciou.item()
        
        val_loss /= len(val_loader)
        val_ciou /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train CIoU: {train_ciou:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val CIoU: {val_ciou:.4f}")
        
        # Early stopping
        if val_ciou > best_val_ciou:
            best_val_ciou = val_ciou
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break


def main() -> None:
    """Main training pipeline with 3-phase strategy."""
    
    print("=== License Plate Detection - PyTorch Training (FPN + CBAM + CIoU) ===")
    
    # Show training menu and get user choice
    mode, phases_to_run = show_training_menu()
    if mode is None:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
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
    
    # Create datasets and dataloaders
    train_dataset = PlateDataset(x_train, y_train, train_weights)
    val_dataset = PlateDataset(x_val, y_val)
    test_dataset = PlateDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Build model
    print("\n3. Building model with FPN + CBAM...")
    detector = LicensePlateDetectionModel(input_shape=(320, 320, 3), device=device)
    detector.get_summary()
    
    # Loss and optimizer
    criterion = BoxRegressionLoss()
    
    # Determine which phase to start from based on mode
    phase3_checkpoint = find_latest_checkpoint("license_plate_detector_phase3_*.pt") if mode != 'fresh' else None
    phase2_checkpoint = find_latest_checkpoint("license_plate_detector_phase2_*.pt") if mode != 'fresh' else None
    phase1_checkpoint = find_latest_checkpoint("license_plate_detector_phase1_*.pt") if mode != 'fresh' else None
    
    # Load appropriate checkpoint for resume mode
    if mode == 'resume':
        if phase3_checkpoint:
            print(f"\nLoading Phase 3 checkpoint: {phase3_checkpoint}")
            detector.load(phase3_checkpoint)
        elif phase2_checkpoint:
            print(f"\nLoading Phase 2 checkpoint: {phase2_checkpoint}")
            detector.load(phase2_checkpoint)
        elif phase1_checkpoint:
            print(f"\nLoading Phase 1 checkpoint: {phase1_checkpoint}")
            detector.load(phase1_checkpoint)
    
    # ===== PHASE 1: Warmup + Head Training (frozen backbone) =====
    if 1 in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 1: Training Head with Frozen Backbone (Warmup)")
        print("="*60)
        
        detector.freeze_backbone(freeze=True)
        optimizer = create_optimizer(detector, learning_rate=2e-4)
        train_phase(detector, device, train_loader, val_loader, criterion, optimizer, epochs=20, phase_name="Phase 1")
        detector.save(f"license_plate_detector_phase1_{timestamp}.pt")
        phase1_checkpoint = f"license_plate_detector_phase1_{timestamp}.pt"

    # ===== PHASE 2: Fine-tune Top Backbone Layers =====
    if 2 in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning Top Backbone Layers")
        print("="*60)
        
        detector.freeze_top_layers(fine_tune_at=200)
        optimizer = create_optimizer(detector, learning_rate=5e-5)
        train_phase(detector, device, train_loader, val_loader, criterion, optimizer, epochs=30, phase_name="Phase 2")
        detector.save(f"license_plate_detector_phase2_{timestamp}.pt")
        phase2_checkpoint = f"license_plate_detector_phase2_{timestamp}.pt"

    # ===== PHASE 3: Full Network Fine-tuning =====
    if 3 in phases_to_run:
        print("\n" + "="*60)
        print("PHASE 3: Full Network Fine-tuning with Lower LR")
        print("="*60)
        
        detector.freeze_top_layers(fine_tune_at=100)
        optimizer = create_optimizer(detector, learning_rate=1e-5)
        train_phase(detector, device, train_loader, val_loader, criterion, optimizer, epochs=25, phase_name="Phase 3")
        detector.save(f"license_plate_detector_phase3_{timestamp}.pt")
    
    # Evaluate (skip if no phases were run)
    if not phases_to_run or 0 in phases_to_run or any(p in [1, 2, 3] for p in phases_to_run):
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        detector.eval()
        test_loss = 0.0
        test_ciou = 0.0
        
        with torch.no_grad():
            for images, bboxes in test_loader:
                images = images.to(device)
                bboxes = bboxes.to(device)
                
                outputs = detector(images)
                loss = criterion(bboxes, outputs)
                
                test_loss += loss.item()
                ciou = torch.mean(box_ciou(bboxes, outputs))
                test_ciou += ciou.item()
        
        test_loss /= len(test_loader)
        test_ciou /= len(test_loader)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test CIoU: {test_ciou:.4f}")
        
        # Save final model
        print("\n7. Saving final model...")
        detector.save('license_plate_detector.pt')
        
        # Visualize predictions
        print("\n8. Visualizing predictions...")
        with torch.no_grad():
            # Transpose from HWC to CHW format to match model input expectations
            x_test_chw = np.transpose(x_test, (0, 3, 1, 2))
            pred_test = detector(torch.tensor(x_test_chw, dtype=torch.float32).to(device))
        pred_test = pred_test.cpu().numpy()
        visualize_predictions(x_test, y_test, pred_test, num_samples=5)
    
    print("\n" + "="*60)
    print("=== Training Complete ===")
    print("="*60)
    print(f"Model saved as: license_plate_detector.pt")
    print(f"Checkpoint models: license_plate_detector_phase*.pt")
    print("\nRun inference with:")
    print("  from inference import PlateDetector")
    print("  detector = PlateDetector('license_plate_detector.pt')")
    print("  detector.detect_and_draw('image.jpg', 'output.jpg')")



if __name__ == "__main__":
    main()
