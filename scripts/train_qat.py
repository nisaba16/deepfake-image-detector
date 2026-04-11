import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import sys
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.data_loader import collect_image_paths_and_labels, stratified_split
except ImportError:
    print("Could not import data_loader")
    sys.exit(1)

from scripts.train_fp32 import CustomDataset, set_seed
from common.utils import replace_with_quantized_modules
from common.test_functions import model_to_quant

# Custom QAT support using common modules
def get_qat_model(model_name, num_classes, features=("rgb", "hsv", "fft", "noise", "srm")):
    from torchvision.models import resnet50, vit_b_16, mobilenet_v3_small

    if model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "vit_b_16":
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "forensic_mobilenet":
        from common.forensic_mobilenet import ForensicMobileNetV3
        model = ForensicMobileNetV3(
            features=tuple(features),
            num_classes=num_classes,
            pretrained_rgb=False,
            noise_augment=False,  # noise augmentation handled by transforms, not needed here
        )
    else:
        raise ValueError(f"Model not implemented: {model_name}.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train models in QAT")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "mobilenet_v3_small", "vit_b_16", "forensic_mobilenet"], help="Model to train")
    parser.add_argument("--features", nargs="+", default=["rgb", "hsv", "fft", "noise", "srm"],
                        help="Feature modalities for forensic_mobilenet (ignored for other models)")
    parser.add_argument("--data_dir", type=str, default="data/dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for QAT")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (usually smaller for QAT)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consistent split")
    parser.add_argument("--subsample", type=int, default=0, help="Use a small subpart of the dataset. 0 for full dataset.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--pretrained", type=str, help="Path to FP32 pretrained weights (optional but recommended)")
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN (useful for architecture mismatch errors)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="deepfake-detection", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, help="W&B run name")
    
    args = parser.parse_args()
    
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled to avoid architecture mismatch errors.")

    torch.backends.quantized.engine = 'fbgemm'

    set_seed(args.seed)
    
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Note: QAT operations might fall back to CPU)")
    
    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    paths, labels, class_to_idx, idx_to_class = collect_image_paths_and_labels(args.data_dir)
    num_classes = len(class_to_idx)
    
    # 2. Split with fixed seed
    train_paths, val_paths, train_labels, val_labels = stratified_split(paths, labels, test_size=0.2, seed=args.seed)
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    
    target_size = 224
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(target_size * 256 / 224)),  # 256 for 224 target — standard ImageNet eval
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Model
    print(f"Initializing FP32 model {args.model}...")
    model = get_qat_model(args.model, num_classes, features=args.features)
    
    if args.pretrained:
        print(f"Loading pretrained FP32 weights from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        
    print("Replacing standard layers with custom Quantized layers...")
    replace_with_quantized_modules(model)
    model = model.to(device)

    print("Calibrating quantization parameters...")
    # use a subset of train loader (since model_to_quant pops next(iter(...)))
    model = model_to_quant(model, train_loader, act_N_bits=8, weight_N_bits=8, method='sym', device=device)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for imgs, lbls in train_pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == lbls.data).item()
            total_train += imgs.size(0)
            
            train_pbar.set_postfix({"loss": f"{train_loss/(total_train/args.batch_size):.4f}", "acc": f"{correct_train/total_train:.4f}"})
            
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
            
        if epoch == 0:
            print("\n=== Quantized Elements in 8-bits ===")
            count = 0
            for name, m in model.named_modules():
                # We identify quantized modules (e.g. Quantized_Linear, Quantized_Conv2d, etc)
                if 'quantize' in str(type(m)).lower() or 'qlinear' in str(type(m)).lower() or 'qconv' in str(type(m)).lower():
                    print(f"- {name}: {type(m).__name__}")
                    count += 1
            print(f"Total 8-bit quantized layers injected: {count}\n")
        
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for imgs, lbls in val_pbar:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == lbls.data).item()
                total_val += imgs.size(0)
                
                val_pbar.set_postfix({"loss": f"{val_loss/(total_val/args.batch_size):.4f}", "acc": f"{correct_val/total_val:.4f}"})
                
        val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.4f}")
              
        if val_acc > best_acc:
            best_acc = val_acc
            if args.model == "forensic_mobilenet":
                tag = "-".join(args.features)
                model_stem = f"forensic_mobilenet_{tag}"
            else:
                model_stem = args.model
            save_path = os.path.join(args.save_dir, f"best_{model_stem}_qat.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with acc {best_acc:.4f} to {save_path}")

        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "train_acc": train_acc,
                "val_loss": val_loss / len(val_loader),
                "val_acc": val_acc,
                "best_val_acc": best_acc
            })

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
