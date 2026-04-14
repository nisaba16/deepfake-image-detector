"""
Train ForensicMobileNetV3 in FP32.

Experiments: vary --features to measure the impact of each input modality.

Examples
--------
# Baseline: RGB only (≡ standard MobileNetV3)
python scripts/train_forensic_mobilenet.py --features rgb

# RGB + HSV
python scripts/train_forensic_mobilenet.py --features rgb hsv

# RGB + HSV + FFT (frequency artifacts)
python scripts/train_forensic_mobilenet.py --features rgb hsv fft

# RGB + noise residual + SRM filters
python scripts/train_forensic_mobilenet.py --features rgb noise srm

# All five modalities
python scripts/train_forensic_mobilenet.py --features rgb hsv fft noise srm

# SLURM: see slurm_train_models.sh with MODEL=forensic_mobilenet
"""

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.forensic_mobilenet import AVAILABLE_FEATURES, ForensicMobileNetV3
from utils.data_loader import collect_image_paths_and_labels, stratified_split


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def features_tag(features) -> str:
    """Short string for checkpoint naming, e.g. 'rgb-hsv-fft'."""
    return "-".join(features)


def checkpoint_name(features, save_dir: str) -> str:
    tag  = features_tag(features)
    return os.path.join(save_dir, f"best_forensic_mobilenet_{tag}_fp32.pth")


# ---------------------------------------------------------------------------
# Training / validation one epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, desc: str):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss    = criterion(outputs, lbls)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == lbls).sum().item()
            total      += imgs.size(0)
            n_batches   = total / loader.batch_size
            pbar.set_postfix({
                "loss": f"{total_loss / n_batches:.4f}",
                "acc":  f"{correct / total:.4f}",
            })

    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ForensicMobileNetV3 in FP32")

    # --- Feature selection ---
    parser.add_argument(
        "--features", nargs="+", default=["rgb", "hsv", "fft", "noise"],
        choices=AVAILABLE_FEATURES, metavar="FEATURE",
        help=(
            "Input feature modalities. Any subset of: "
            f"{' '.join(AVAILABLE_FEATURES)}. "
            "Example: --features rgb hsv fft"
        ),
    )
    parser.add_argument(
        "--no_pretrained_rgb", action="store_true",
        help="Skip warm-initialisation of the RGB channels from ImageNet weights",
    )
    parser.add_argument(
        "--no_noise_augment", action="store_true",
        help="Disable training-time noise augmentation (gaussian/jpeg/blur/erase)",
    )
    parser.add_argument("--noise_gaussian_std",  type=float, default=0.02)
    parser.add_argument("--noise_jpeg_min",      type=int,   default=50)
    parser.add_argument("--noise_jpeg_max",      type=int,   default=95)
    parser.add_argument("--noise_p_gaussian",    type=float, default=0.5)
    parser.add_argument("--noise_p_jpeg",        type=float, default=0.5)
    parser.add_argument("--noise_p_blur",        type=float, default=0.3)
    parser.add_argument("--noise_p_erase",       type=float, default=0.3)

    # --- Data ---
    parser.add_argument("--data_dir",   default="data/dataset")
    parser.add_argument("--subsample",  type=int, default=0,
                        help="Use only this many images (0 = full dataset)")

    # --- Training ---
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 weight decay for AdamW (default: 1e-4)")
    parser.add_argument("--scheduler",  action="store_true",
                        help="Enable CosineAnnealingLR scheduler (default: constant LR)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save_dir",   default="checkpoints")
    parser.add_argument("--disable_cudnn", action="store_true")

    # --- Logging ---
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb_project", default="deepfake-detection")
    parser.add_argument("--wandb_name",    default=None)

    args = parser.parse_args()

    # Deduplicate while preserving order
    seen, features = set(), []
    for f in args.features:
        if f not in seen:
            seen.add(f)
            features.append(f)
    args.features = features

    # -----------------------------------------------------------------------
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled.")

    set_seed(args.seed)

    # -----------------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------------
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"forensic_mobilenet_{features_tag(args.features)}"
        wandb.init(
            project = args.wandb_project,
            name    = run_name,
            config  = vars(args),
        )

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print(f"Loading dataset from {args.data_dir} ...")
    paths, labels, class_to_idx, _ = collect_image_paths_and_labels(args.data_dir)
    print(f"  Classes : {class_to_idx}")

    if args.subsample > 0:
        paths  = paths[:args.subsample]
        labels = labels[:args.subsample]

    num_classes = len(class_to_idx)
    train_paths, val_paths, train_labels, val_labels = stratified_split(
        paths, labels, test_size=0.2, seed=args.seed
    )
    print(f"  Train : {len(train_paths)} | Val : {len(val_paths)}")

    # Standard ImageNet transforms — the model handles all feature computation
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),   # standard ImageNet eval: resize shorter side to 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        ImageDataset(train_paths, train_labels, train_tf),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        ImageDataset(val_paths, val_labels, val_tf),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    print(f"\nFeatures : {args.features}")
    noise_kwargs = dict(
        gaussian_std      = args.noise_gaussian_std,
        jpeg_quality_min  = args.noise_jpeg_min,
        jpeg_quality_max  = args.noise_jpeg_max,
        p_gaussian        = args.noise_p_gaussian,
        p_jpeg            = args.noise_p_jpeg,
        p_blur            = args.noise_p_blur,
        p_erase           = args.noise_p_erase,
    )
    model = ForensicMobileNetV3(
        features       = tuple(args.features),
        num_classes    = num_classes,
        pretrained_rgb = not args.no_pretrained_rgb,
        noise_augment  = not args.no_noise_augment,
        noise_kwargs   = noise_kwargs,
    )
    print(model.describe())
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler else None

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = checkpoint_name(args.features, args.save_dir)
    best_acc  = 0.0

    print(f"\nCheckpoint path : {ckpt_path}\n")

    for epoch in range(1, args.epochs + 1):
        desc_train = f"Epoch {epoch}/{args.epochs} [Train]"
        desc_val   = f"Epoch {epoch}/{args.epochs} [Val  ]"

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, desc_train
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, desc_val
        )
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"Train  loss={train_loss:.4f}  acc={train_acc:.4f} | "
            f"Val    loss={val_loss:.4f}  acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved  (val_acc={best_acc:.4f})  →  {ckpt_path}")

        if args.wandb:
            import wandb
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
            wandb.log({
                "epoch":        epoch,
                "train_loss":   train_loss,
                "train_acc":    train_acc,
                "val_loss":     val_loss,
                "val_acc":      val_acc,
                "best_val_acc": best_acc,
                "lr":           current_lr,
            })

    print(f"\nDone. Best val acc: {best_acc:.4f}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
