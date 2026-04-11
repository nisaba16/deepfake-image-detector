"""
Evaluate all PyTorch checkpoints on an external dataset and print a comparison table.

Usage
-----
python scripts/evaluate_all.py --data_dir data/ddata/test --n_samples 1000
python scripts/evaluate_all.py --data_dir data/ddata/test --n_samples 1000 --checkpoints_dir checkpoints
"""

import argparse
import os
import sys
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import collect_image_paths_and_labels
from common.forensic_mobilenet import ForensicMobileNetV3, AVAILABLE_FEATURES


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SimpleDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return TRANSFORM(Image.open(self.paths[idx]).convert("RGB")), self.labels[idx]


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_resnet50(num_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_vit_b_16(num_classes):
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    return m

def build_mobilenet_v3_small(num_classes):
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

def build_forensic_mobilenet(features, num_classes):
    return ForensicMobileNetV3(features=tuple(features), num_classes=num_classes, pretrained_rgb=False)


def _is_qat_state(state: dict) -> bool:
    return any(k.endswith("input_scale") or k.endswith("weight_scale") for k in state)


def load_checkpoint(ckpt_path, num_classes, device):
    """
    Infer model type from filename and load state dict.
    Handles FP32, QAT (Quantized_* layers), and DINOv2 checkpoints.
    Returns model or None on failure.
    """
    from common.utils import replace_with_quantized_modules

    name = os.path.basename(ckpt_path)

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if any(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}

        is_qat = _is_qat_state(state)

        if "forensic_mobilenet" in name:
            parts    = name.replace("best_forensic_mobilenet_", "").replace("_fp32.pth", "").replace("_qat.pth", "")
            features = [f for f in parts.split("-") if f in AVAILABLE_FEATURES] or ["rgb", "hsv", "fft", "noise", "srm"]
            model    = build_forensic_mobilenet(features, num_classes)
        elif "resnet50" in name:
            model = build_resnet50(num_classes)
        elif "vit_b_16" in name:
            model = build_vit_b_16(num_classes)
        elif "mobilenet_v3_small" in name:
            model = build_mobilenet_v3_small(num_classes)
        elif "dinov2" in name:
            base = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
            class _Dino(nn.Module):
                def __init__(self, base, nc):
                    super().__init__()
                    self.base_model = base
                    self.classifier = nn.Linear(768, nc)
                def forward(self, x): return self.classifier(self.base_model(x))
            model = _Dino(base, num_classes)
        else:
            print(f"  [SKIP] Cannot infer model type from: {name}")
            return None

        # QAT checkpoints contain Quantized_* layers — rebuild the model with them
        if is_qat:
            replace_with_quantized_modules(model)

        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        return model

    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for imgs, lbls in tqdm(loader, leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total   += lbls.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())

    acc = correct / total

    # Per-class accuracy
    classes  = sorted(set(all_labels))
    per_class = {}
    for c in classes:
        idx = [i for i, l in enumerate(all_labels) if l == c]
        per_class[c] = sum(all_preds[i] == all_labels[i] for i in idx) / len(idx)

    return acc, per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        default="data/ddata/test")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--n_samples",       type=int, default=1000)
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output",          default=None, help="Optional JSON output path")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # --- Load dataset -------------------------------------------------------
    print(f"\nLoading {args.n_samples} images from {args.data_dir} ...")
    paths, labels, class_to_idx, idx_to_class = collect_image_paths_and_labels(args.data_dir)
    print(f"  Classes : {class_to_idx}  |  Total available : {len(paths)}")

    # Stratified sample
    by_class = {}
    for p, l in zip(paths, labels):
        by_class.setdefault(l, []).append(p)
    per_class_n = args.n_samples // len(by_class)
    sampled_paths, sampled_labels = [], []
    for cls, ps in by_class.items():
        chosen = random.sample(ps, min(per_class_n, len(ps)))
        sampled_paths  += chosen
        sampled_labels += [cls] * len(chosen)

    # Shuffle together
    combined = list(zip(sampled_paths, sampled_labels))
    random.shuffle(combined)
    sampled_paths, sampled_labels = zip(*combined)

    loader = DataLoader(
        SimpleDataset(list(sampled_paths), list(sampled_labels)),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    num_classes = len(class_to_idx)
    print(f"  Evaluating on {len(sampled_paths)} images ({per_class_n} per class)\n")

    # --- Find checkpoints ---------------------------------------------------
    ckpts = sorted(
        f for f in os.listdir(args.checkpoints_dir)
        if f.endswith(".pth") and f.startswith("best_")
    )
    if not ckpts:
        print(f"No checkpoints found in {args.checkpoints_dir}")
        return

    # --- Evaluate each ------------------------------------------------------
    results = []
    for ckpt_name in ckpts:
        ckpt_path = os.path.join(args.checkpoints_dir, ckpt_name)
        print(f"→ {ckpt_name}")
        model = load_checkpoint(ckpt_path, num_classes, device)
        if model is None:
            continue
        acc, per_class = evaluate(model, loader, device)
        fake_acc = per_class.get(class_to_idx.get("fake", 0), float("nan"))
        real_acc = per_class.get(class_to_idx.get("real", 1), float("nan"))
        print(f"  Acc={acc*100:.2f}%  fake={fake_acc*100:.2f}%  real={real_acc*100:.2f}%")
        results.append({
            "checkpoint": ckpt_name,
            "accuracy":   round(acc * 100, 2),
            "fake_acc":   round(fake_acc * 100, 2),
            "real_acc":   round(real_acc * 100, 2),
        })
        del model
        torch.cuda.empty_cache()

    # --- Print table --------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"{'Checkpoint':<50} {'Acc':>6} {'Fake':>6} {'Real':>6}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(f"{r['checkpoint']:<50} {r['accuracy']:>5.2f}% {r['fake_acc']:>5.2f}% {r['real_acc']:>5.2f}%")
    print("=" * 75)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
