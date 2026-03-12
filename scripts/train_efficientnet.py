"""Training script for EfficientNet-B0 deepfake detection.

This script trains a 6-channel EfficientNet-B0 model:
1. Uses SixChannelPreprocessor to generate 6-channel inputs
2. Trains DeepfakeEfficientNet end-to-end with PyTorch
3. Exports final model to ONNX
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports when running as script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from models.custom_efficientnet import DeepfakeEfficientNet, export_to_onnx
from models.extractors import SixChannelPreprocessor
from models.face_preprocessor import create_face_preprocessor
from models.rgb_augmentation import build_rgb_augmentation
from models.ood_tracker import add_ood_tracking_to_trainer

# Import necessary functions from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# Helper functions
def collect_image_paths_and_labels(data_dir: str):
    """Collect image paths and labels from ImageFolder-style directory."""
    data_path = Path(data_dir)
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}")
    
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    all_paths = []
    all_labels = []
    
    for class_dir in class_dirs:
        class_idx = class_to_idx[class_dir.name]
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                all_paths.append(str(img_path))
                all_labels.append(class_idx)
    
    return all_paths, all_labels, class_to_idx, idx_to_class


def split_dataset(paths, labels, test_size=0.2, val_size=0.2, random_state=42):
    """Split dataset into train/val/test sets."""
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class SixChannelDataset(Dataset):
    """Dataset that converts face images to 6-channel tensors."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        face_preprocessor=None,
        six_channel_processor: SixChannelPreprocessor = None,
        skip_failed: bool = True,
        rgb_augmentation=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.face_preprocessor = face_preprocessor
        self.six_channel_processor = six_channel_processor or SixChannelPreprocessor()
        self.skip_failed = skip_failed
        self.rgb_augmentation = rgb_augmentation  # NEW: RGB augmentation before channel calculation
        
        # Pre-filter valid samples to avoid runtime issues
        self.valid_indices = []
        print("Validating dataset samples...")
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            if self._is_valid_sample(i):
                self.valid_indices.append(i)
        
        print(f"Valid samples: {len(self.valid_indices)} / {len(image_paths)}")
        
    def _is_valid_sample(self, idx: int) -> bool:
        """Check if a sample can be processed without errors."""
        try:
            self._process_sample(idx)
            return True
        except Exception:
            if not self.skip_failed:
                raise
            return False
    
    def _process_sample(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Process a single sample to 6-channel tensor."""
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(path).convert('RGB')
        
        # Extract face if preprocessor is provided
        if self.face_preprocessor is not None:
            face_crop, _, metadata = self.face_preprocessor.preprocess_image(image)
            if face_crop is None:
                raise ValueError(f"No face detected in {path}")
            face_array = np.array(face_crop)
        else:
            # Use full image, resize to target size
            image_resized = image.resize(self.six_channel_processor.output_size)
            face_array = np.array(image_resized)
        
        # CRITICAL: Apply RGB augmentation BEFORE calculating derived channels
        # This preserves the mathematical relationship between RGB and derived channels
        if self.rgb_augmentation is not None:
            face_array = self.rgb_augmentation(face_array)
        
        # Convert to 6-channel tensor (SRM, FFT, Saturation calculated from augmented RGB)
        six_channel_array = self.six_channel_processor.process_face_to_6_channels(face_array)
        six_channel_tensor = torch.from_numpy(six_channel_array).float()
        
        return six_channel_tensor, label
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a preprocessed 6-channel sample."""
        actual_idx = self.valid_indices[idx]
        return self._process_sample(actual_idx)


def train_model(
    model: DeepfakeEfficientNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = 'cuda',
    save_best: bool = True,
    model_save_path: str = 'best_model.pth',
    scheduler_type: str = 'plateau',
    use_wandb: bool = False,
    class_weights: torch.Tensor = None,
    weight_decay: float = 1e-4,
    ood_tracker = None,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0
) -> Dict[str, List]:
    """Train the EfficientNet model with learning rate scheduling.
    
    Args:
        scheduler_type: Type of LR scheduler - 'plateau', 'cosine', or 'none'
        use_wandb: Whether to log metrics to Weights & Biases
        class_weights: Class weights for balanced loss (optional)
        weight_decay: L2 regularization strength (default: 1e-4)
        ood_tracker: Out-of-domain test tracker for monitoring predictions
        label_smoothing: Label smoothing for better calibration (0.0-0.2, default: 0.0)
        mixup_alpha: Mixup alpha parameter (0.0 = off, 0.2 = recommended, default: 0.0)
    """
    
    model = model.to(device)
    
    # Use class weights if provided for balanced training
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print(f"Using weighted loss with class weights: {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    if label_smoothing > 0:
        print(f"Label smoothing: {label_smoothing} (prevents overconfidence)")
    if mixup_alpha > 0:
        print(f"Mixup alpha: {mixup_alpha} (data augmentation)")
    
    # Add weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Optimizer: Adam(lr={lr}, weight_decay={weight_decay})")
    
    # Learning rate scheduler to prevent overfitting
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        scheduler = None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    
    print(f"Training on {device} for {epochs} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"LR Scheduler: {scheduler_type}, Early stopping patience: {early_stop_patience}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply Mixup augmentation if enabled
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                targets_a, targets_b = targets, targets[index]
                
                optimizer.zero_grad()
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_acc = train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss_avg)
            else:  # cosine
                scheduler.step()
        
        # Save best model
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Saved best model with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs (no improvement for {early_stop_patience} epochs)")
            break
        
        # Log metrics
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss_avg,
                'train/accuracy': train_acc,
                'val/loss': val_loss_avg,
                'val/accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'patience': patience_counter
            })
        
        # Log OOD test predictions (every 5 epochs)
        if ood_tracker is not None and (epoch + 1) % 5 == 0:
            ood_tracker.log_predictions(model, device, epoch + 1, use_wandb=use_wandb)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}, Patience: {patience_counter}/{early_stop_patience}")
    
    return history


def evaluate_model(
    model: DeepfakeEfficientNet,
    test_loader: DataLoader,
    device: str = 'cuda',
    class_names: List[str] = None
) -> Dict[str, Any]:
    """Evaluate the trained model."""
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # ROC-AUC for binary classification
    try:
        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
    except Exception:
        auc = None
    
    # Classification report
    if class_names is None:
        class_names = ['Fake', 'Real']  # Default binary labels
    
    # Get unique classes in predictions and targets
    unique_classes = sorted(set(all_targets.tolist() + all_predictions.tolist()))
    
    report = classification_report(
        all_targets, all_predictions, 
        labels=unique_classes,
        target_names=[class_names[i] for i in unique_classes],
        output_dict=True,
        zero_division=0
    )
    
    confusion = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'classification_report': report,
        'confusion_matrix': confusion.tolist(),
        'predictions': all_predictions.tolist(),
        'probabilities': all_probabilities.tolist(),
        'targets': all_targets.tolist()
    }
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 for deepfake detection')
    
    # Data
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset directory (with fake/ and real/ subdirs)')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Limit number of samples (for quick testing, -1 for all)')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                        help='Fraction of dataset to use (0.0-1.0, e.g., 0.1 for 10%%)')
    parser.add_argument('--additional-data', type=str, nargs='*', default=[],
                           help='Additional dataset directories to combine with main dataset (e.g., data/ddata/train)')

    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for L2 regularization')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing (0.0-0.2, prevents overconfidence, default: 0.1)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Mixup alpha (0.0=off, 0.2=recommended, default: 0.2)')
    parser.add_argument('--augmentation', type=str, default='normal', choices=['none', 'normal', 'aggressive'],
                        help='RGB augmentation level (applied before channel calculation)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ImageNet weights')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler (plateau=ReduceLROnPlateau, cosine=CosineAnnealing, none=no scheduler)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    
    # Face detection
    parser.add_argument('--use-face-detection', action='store_true',
                        help='Use face detection preprocessing')
    parser.add_argument('--face-detector', type=str, default='mediapipe',
                        choices=['mediapipe', 'mtcnn', 'retinaface'],
                        help='Face detector to use')
    parser.add_argument('--skip-failed', action='store_true',
                        help='Skip images with failed face detection')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='efficientnet_output',
                        help='Directory to save outputs')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export model to ONNX format')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Weights & Biases
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='deepfake-detection',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='Weights & Biases run name (default: auto-generated)')
    parser.add_argument('--ood-test-dir', type=str, default='image_test',
                        help='Directory with out-of-domain test images for tracking (default: image_test)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Configure CUDA if available
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    paths, labels, class_to_idx, idx_to_class = collect_image_paths_and_labels(args.data)
    
    # Load and combine additional datasets
    if args.additional_data:
        for additional_dir in args.additional_data:
            print(f"Loading additional dataset from: {additional_dir}")
            add_paths, add_labels, add_class_to_idx, add_idx_to_class = collect_image_paths_and_labels(additional_dir)
            
            # Verify class compatibility
            if add_class_to_idx != class_to_idx:
                print(f"Warning: Class indices differ between datasets!")
                print(f"Main dataset classes: {class_to_idx}")
                print(f"Additional dataset classes: {add_class_to_idx}")
                # Remap labels to match main dataset classes
                label_map = {add_class_to_idx[name]: class_to_idx[name] for name in add_class_to_idx if name in class_to_idx}
                add_labels = [label_map[lbl] for lbl in add_labels if lbl in label_map]
                add_paths = [add_paths[i] for i, lbl in enumerate(add_labels) if lbl in label_map]
            
            # Combine datasets
            paths.extend(add_paths)
            labels.extend(add_labels)
            print(f"Added {len(add_paths)} samples from {additional_dir}")
    
    print(f"Total dataset after combining: {len(paths)} samples")
    
    # Sample dataset if requested
    if args.sample_fraction < 1.0:
        from sklearn.model_selection import train_test_split
        paths, _, labels, _ = train_test_split(
            paths, labels, 
            train_size=args.sample_fraction,
            random_state=args.seed,
            stratify=labels
        )
        print(f"Sampled {args.sample_fraction*100:.1f}% of data: {len(paths)} samples")
    
    if args.limit > 0:
        paths = paths[:args.limit]
        labels = labels[:args.limit]
    
    print(f"Dataset: {len(paths)} samples, {len(class_to_idx)} classes")
    print(f"Classes: {class_to_idx}")
    
    # Print class distribution
    from collections import Counter
    class_counts = Counter(labels)
    print("\nClass distribution in full dataset:")
    for class_name, class_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        count = class_counts[class_idx]
        percentage = (count / len(labels)) * 100
        print(f"  {class_name} (class {class_idx}): {count} samples ({percentage:.1f}%)")
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        paths, labels, test_size=0.2, val_size=0.2, random_state=args.seed
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Print split distributions
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    test_counts = Counter(y_test)
    print("\nClass distribution in splits:")
    for class_name, class_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        train_c = train_counts[class_idx]
        val_c = val_counts[class_idx]
        test_c = test_counts[class_idx]
        print(f"  {class_name}: Train={train_c}, Val={val_c}, Test={test_c}")
    
    # Compute class weights for balanced training
    class_weights = torch.zeros(len(class_to_idx))
    for class_idx in range(len(class_to_idx)):
        class_weights[class_idx] = len(y_train) / (len(class_to_idx) * train_counts[class_idx])
    print(f"\nClass weights for balanced loss: {class_weights.tolist()}")
    
    # Print split distributions
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    test_counts = Counter(y_test)
    print("\nClass distribution in splits:")
    for class_name, class_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        train_c = train_counts[class_idx]
        val_c = val_counts[class_idx]
        test_c = test_counts[class_idx]
        print(f"  {class_name}: Train={train_c}, Val={val_c}, Test={test_c}")
    
    # Compute class weights for balanced training
    class_weights = torch.zeros(len(class_to_idx))
    for class_idx in range(len(class_to_idx)):
        class_weights[class_idx] = len(y_train) / (len(class_to_idx) * train_counts[class_idx])
    print(f"\nClass weights for balanced loss: {class_weights.tolist()}")
    
    # Create preprocessors
    six_channel_processor = SixChannelPreprocessor()
    
    # Create face preprocessor FIRST (critical for OOD consistency)
    face_preprocessor = None
    if args.use_face_detection:
        print(f"Creating face preprocessor with detector: {args.face_detector}")
        face_preprocessor = create_face_preprocessor(
            detector_name=args.face_detector,
            output_size=(224, 224),
            normalize_imagenet=False,  # We handle normalization in SixChannelPreprocessor
        )
    
    # Create OOD test tracker with SAME preprocessing pipeline as training
    class_names = [name for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1])]
    ood_tracker = add_ood_tracking_to_trainer(
        args.ood_test_dir,
        six_channel_processor,
        class_names,
        face_preprocessor=face_preprocessor  # CRITICAL: Use same face detection as training
    )
    
    # Create RGB augmentation (applied BEFORE channel calculation)
    if args.augmentation == 'none':
        train_augmentation = None
        print("RGB Augmentation: DISABLED")
    elif args.augmentation == 'aggressive':
        train_augmentation = build_rgb_augmentation(train=True, aggressive=True)
        print("RGB Augmentation: AGGRESSIVE (JPEG 40-90, high jitter)")
    else:  # normal
        train_augmentation = build_rgb_augmentation(train=True, aggressive=False)
        print("RGB Augmentation: NORMAL (JPEG 50-95, moderate jitter)")
    
    val_augmentation = None  # No augmentation for validation/test
    
    # Create datasets with RGB augmentation
    print("\nCreating datasets...")
    train_dataset = SixChannelDataset(
        X_train, y_train, face_preprocessor, six_channel_processor, 
        args.skip_failed, rgb_augmentation=train_augmentation
    )
    val_dataset = SixChannelDataset(
        X_val, y_val, face_preprocessor, six_channel_processor, 
        args.skip_failed, rgb_augmentation=val_augmentation
    )
    test_dataset = SixChannelDataset(
        X_test, y_test, face_preprocessor, six_channel_processor, 
        args.skip_failed, rgb_augmentation=val_augmentation
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print("Creating EfficientNet-B0 model...")
    model = DeepfakeEfficientNet(
        pretrained=args.pretrained,
        num_classes=len(class_to_idx),
        dropout=args.dropout
    )
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Model info: {json.dumps(model_info, indent=2)}")
    
    # Initialize Weights & Biases
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'model': 'EfficientNet-B0',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'dropout': args.dropout,
                'scheduler': args.scheduler,
                'pretrained': args.pretrained,
                'sample_fraction': args.sample_fraction,
                'device': device,
                'seed': args.seed
            }
        )
    
    # Train model
    model_save_path = output_dir / 'best_model.pth'
    try:
        history = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_best=True,
            model_save_path=str(model_save_path),
            scheduler_type=args.scheduler,
            use_wandb=args.wandb,
            class_weights=class_weights,
            weight_decay=args.weight_decay,
            ood_tracker=ood_tracker,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha
        )
    except RuntimeError as e:
        if 'CUDNN' in str(e) or 'CUDA' in str(e):
            print(f"\nCUDA error encountered: {e}")
            print("Falling back to CPU...")
            device = 'cpu'
            model = model.to(device)
            torch.cuda.empty_cache()
            history = train_model(
                model, train_loader, val_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                save_best=True,
                model_save_path=str(model_save_path),
                scheduler_type=args.scheduler,
                use_wandb=args.wandb
            )
        else:
            raise
    
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model = model.to(device)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    results = evaluate_model(model, test_loader, device, class_names)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    if results['auc'] is not None:
        print(f"Test AUC: {results['auc']:.4f}")
    
    # Log test results to wandb
    if args.wandb:
        wandb.log({
            'test/accuracy': results['accuracy'],
            'test/auc': results['auc'] if results['auc'] is not None else 0
        })
        # Save model as wandb artifact
        artifact = wandb.Artifact('efficientnet-model', type='model')
        artifact.add_file(str(model_save_path))
        wandb.log_artifact(artifact)
    
    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Export to ONNX
    if args.export_onnx:
        onnx_path = output_dir / 'deepfake_efficientnet.onnx'
        print(f"Exporting model to ONNX: {onnx_path}")
        export_to_onnx(model, str(onnx_path))
        
        # Check file size
        onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"ONNX model size: {onnx_size_mb:.2f} MB")
        
        if onnx_size_mb <= 100:
            print(f"✅ Model size is within 100MB limit")
        else:
            print(f"⚠️  Model size exceeds 100MB limit")
    
    # Save preprocessing info
    preprocessing_info = {
        "input_size": [224, 224],
        "input_channels": 6,
        "channel_description": {
            "0": "Red (normalized 0-1)",
            "1": "Green (normalized 0-1)",
            "2": "Blue (normalized 0-1)",
            "3": "Saturation from HSV (normalized 0-1)",
            "4": "SRM noise filter output",
            "5": "FFT magnitude (log scale, normalized)"
        },
        "model_type": "EfficientNet-B0",
        "num_classes": len(class_to_idx),
        "class_names": class_names
    }
    
    preprocessing_path = output_dir / 'preprocessing_info.json'
    with open(preprocessing_path, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"Preprocessing info saved to {preprocessing_path}")
    
    print("Training completed successfully!")
    print(f"All outputs saved in: {output_dir}")
    
    # Finish wandb run
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
