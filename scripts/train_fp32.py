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

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.data_loader import collect_image_paths_and_labels, stratified_split
except ImportError:
    print("Could not import data_loader")
    sys.exit(1)

class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(model_name, num_classes):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "dinov2_vitb14":
        # Load pre-trained dinov2
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # Add classification head
        class DinoV2Classifier(nn.Module):
            def __init__(self, base_model, num_classes):
                super().__init__()
                self.base_model = base_model
                self.classifier = nn.Linear(768, num_classes) # DINOv2 ViT-B/14 hidden dim is 768
            def forward(self, x):
                x = self.base_model(x)
                return self.classifier(x)
        model = DinoV2Classifier(model, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train models in FP32")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "vit_b_16", "mobilenet_v3_small", "dinov2_vitb14"], help="Model to train")
    parser.add_argument("--data_dir", type=str, default="data/dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consistent split")
    parser.add_argument("--subsample", type=int, default=0, help="Use a small subpart of the dataset. 0 for full dataset.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN (useful for architecture mismatch errors)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="deepfake-detection", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, help="W&B run name")
    
    args = parser.parse_args()
    
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled to avoid architecture mismatch errors.")

    set_seed(args.seed)
    
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    paths, labels, class_to_idx, idx_to_class = collect_image_paths_and_labels(args.data_dir)
    
    if args.subsample > 0:
        # keep subset but stratified
        import numpy as np
        # just sample randomly if needed or take first args.subsample
        paths = paths[:args.subsample]
        labels = labels[:args.subsample]

    num_classes = len(class_to_idx)
    
    # 2. Split with fixed seed
    train_paths, val_paths, train_labels, val_labels = stratified_split(paths, labels, test_size=0.2, seed=args.seed)
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    
    # Transforms
    # DINOv2 requires 224x224 (patch size 14 -> 224 / 14 = 16)
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
    print(f"Initializing {args.model}...")
    model = get_model(args.model, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        # Variables to track loss and accuracy over epoch
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
            save_path = os.path.join(args.save_dir, f"best_{args.model}_fp32.pth")
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
