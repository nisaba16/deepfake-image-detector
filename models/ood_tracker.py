"""Out-of-Domain Test Set Visualization for W&B

This module provides functionality to track model predictions on a fixed set
of out-of-domain images throughout training. This helps visualize:
1. How predictions change over epochs
2. When the model starts generalizing
3. Confidence evolution on specific examples

Usage:
    tracker = OODTestTracker(image_dir='image_test', class_names=['Real', 'Fake'])
    
    # During training (in train_model function)
    if epoch % 5 == 0:  # Every 5 epochs
        tracker.log_predictions(model, device, epoch, use_wandb=True)
"""

from __future__ import annotations

import wandb
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import io


class OODTestTracker:
    """Track predictions on out-of-domain test images during training."""
    
    def __init__(
        self,
        image_dir: str,
        six_channel_processor,
        class_names: List[str] = ['Real', 'Fake'],
        max_images: int = 10,
        face_preprocessor=None
    ):
        """Initialize OOD test tracker.
        
        Args:
            image_dir: Directory containing test images
            six_channel_processor: SixChannelPreprocessor instance
            class_names: List of class names
            max_images: Maximum number of images to track
            face_preprocessor: Optional FacePreprocessor for face detection/cropping
        """
        self.image_dir = Path(image_dir)
        self.six_channel_processor = six_channel_processor
        self.class_names = class_names
        self.max_images = max_images
        self.face_preprocessor = face_preprocessor
        
        # Load test images
        self.test_data = self._load_test_images()
        
        if len(self.test_data) == 0:
            print(f"⚠️  Warning: No images found in {image_dir}")
        else:
            print(f"📊 OOD Test Tracker: Loaded {len(self.test_data)} images from {image_dir}")
    
    def _load_test_images(self) -> List[Dict]:
        """Load test images and prepare them for prediction."""
        if not self.image_dir.exists():
            return []
        
        test_data = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for img_path in sorted(self.image_dir.iterdir())[:self.max_images]:
            if img_path.suffix.lower() not in image_extensions:
                continue
            
            try:
                # Load RGB image
                pil_img = Image.open(img_path).convert('RGB')
                
                # Apply face detection/cropping if preprocessor is provided (CRITICAL for consistency)
                if self.face_preprocessor is not None:
                    face_crop, _, metadata = self.face_preprocessor.preprocess_image(pil_img)
                    if face_crop is None:
                        print(f"  ⚠️  No face detected in {img_path.name}, using full image")
                        img_array = np.array(pil_img.resize(self.six_channel_processor.output_size))
                    else:
                        img_array = np.array(face_crop)
                else:
                    # No face detection - use full image, resize to target size
                    img_array = np.array(pil_img.resize(self.six_channel_processor.output_size))
                
                # Resize for display
                display_img = pil_img.copy()
                display_img.thumbnail((224, 224))
                
                # Convert to 6-channel tensor
                six_channel_array = self.six_channel_processor.process_face_to_6_channels(img_array)
                tensor = torch.from_numpy(six_channel_array).float()
                
                test_data.append({
                    'name': img_path.name,
                    'tensor': tensor,
                    'display_img': display_img,
                    'pil_img': pil_img
                })
            except Exception as e:
                print(f"  ⚠️  Failed to load {img_path.name}: {e}")
        
        return test_data
    
    def predict_batch(
        self,
        model: torch.nn.Module,
        device: str = 'cuda'
    ) -> List[Dict]:
        """Make predictions on all test images.
        
        Returns:
            List of dicts with 'name', 'prediction', 'confidence', 'probabilities'
        """
        if len(self.test_data) == 0:
            return []
        
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in self.test_data:
                # Add batch dimension and move to device
                tensor = data['tensor'].unsqueeze(0).to(device)
                
                # Predict
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                
                predictions.append({
                    'name': data['name'],
                    'prediction': self.class_names[pred_class],
                    'confidence': confidence,
                    'probabilities': {
                        self.class_names[i]: probs[0, i].item() 
                        for i in range(len(self.class_names))
                    },
                    'display_img': data['display_img'],
                    'pil_img': data['pil_img']
                })
        
        return predictions
    
    def log_predictions(
        self,
        model: torch.nn.Module,
        device: str,
        epoch: int,
        use_wandb: bool = True
    ):
        """Log predictions to W&B.
        
        Args:
            model: Trained model
            device: Device to run predictions on
            epoch: Current epoch number
            use_wandb: Whether to log to W&B
        """
        if len(self.test_data) == 0 or not use_wandb:
            return
        
        predictions = self.predict_batch(model, device)
        
        # 1. Log as W&B Images with predictions overlaid
        wandb_images = []
        for pred in predictions:
            # Create caption with prediction
            caption = f"{pred['name']}\n{pred['prediction']} ({pred['confidence']:.1%})"
            
            # Log image with caption
            wandb_images.append(
                wandb.Image(
                    pred['pil_img'],
                    caption=caption
                )
            )
        
        wandb.log({
            "ood_test_predictions": wandb_images,
            "epoch": epoch
        })
        
        # 2. Log as W&B Table for tracking over time
        table_data = []
        # Get class names (works with any class labels)
        class_0 = self.class_names[0]
        class_1 = self.class_names[1]
        
        for pred in predictions:
            row = [
                epoch,
                pred['name'],
                pred['prediction'],
                pred['confidence'],
                pred['probabilities'].get(class_0, 0.0),
                pred['probabilities'].get(class_1, 0.0)
            ]
            table_data.append(row)
        
        table = wandb.Table(
            columns=['Epoch', 'Image', 'Prediction', 'Confidence', f'P({class_0})', f'P({class_1})'],
            data=table_data
        )
        
        wandb.log({f"ood_test_table_epoch_{epoch}": table})
        
        # 3. Log summary statistics
        class_counts = {}
        for class_name in self.class_names:
            class_counts[class_name] = sum(1 for p in predictions if p['prediction'] == class_name)
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        # Build dynamic metrics dict
        metrics = {
            "ood_test/avg_confidence": avg_confidence,
            "ood_test/total_images": len(predictions),
            "epoch": epoch
        }
        for class_name, count in class_counts.items():
            metrics[f"ood_test/{class_name}_count"] = count
        
        wandb.log(metrics)
        
        # Print summary
        print(f"\n📊 OOD Test Predictions (Epoch {epoch}):")
        for pred in predictions:
            print(f"  {pred['name']:30s} → {pred['prediction']:4s} ({pred['confidence']:.1%})")
        
        # Print class counts
        counts_str = ', '.join([f"{count} {name}" for name, count in class_counts.items()])
        print(f"  Summary: {counts_str}, Avg Conf: {avg_confidence:.1%}\n")
        
        # 4. Create and log visualization grid
        fig = self._create_prediction_grid(predictions, epoch)
        wandb.log({
            "ood_test/prediction_grid": wandb.Image(fig),
            "epoch": epoch
        })
        plt.close(fig)
    
    def _create_prediction_grid(
        self,
        predictions: List[Dict],
        epoch: int
    ) -> plt.Figure:
        """Create a grid visualization of predictions.
        
        Args:
            predictions: List of prediction dicts
            epoch: Current epoch number
            
        Returns:
            Matplotlib figure
        """
        n = len(predictions)
        if n == 0:
            return plt.figure()
        
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        fig.suptitle(f'OOD Test Predictions - Epoch {epoch}', fontsize=16, weight='bold')
        
        # Handle different subplot layouts
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, pred in enumerate(predictions):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Show image
            ax.imshow(pred['display_img'])
            ax.axis('off')
            
            # Color based on prediction
            pred_class = pred['prediction']
            confidence = pred['confidence']
            
            # Use color coding: green for high confidence, yellow for medium, red for low
            if confidence >= 0.9:
                color = '#00AA00'  # Dark green - confident
            elif confidence >= 0.7:
                color = '#FFA500'  # Orange - moderate confidence
            else:
                color = '#FF0000'  # Red - low confidence
            
            # Create multi-line title
            title = f"{pred['name']}\n{pred_class} ({confidence:.1%})"
            
            # Add probability breakdown
            prob_strs = [f"{cls}: {prob:.1%}" for cls, prob in pred['probabilities'].items()]
            subtitle = ' | '.join(prob_strs)
            
            ax.set_title(title, fontsize=11, color=color, weight='bold', pad=10)
            ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, 
                   ha='center', fontsize=9, color='gray')
        
        # Hide unused subplots
        for idx in range(len(predictions), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(
        self,
        predictions: List[Dict]
    ) -> plt.Figure:
        """Create a comparison plot of predictions.
        
        Args:
            predictions: List of prediction dicts
            
        Returns:
            Matplotlib figure
        """
        n = len(predictions)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, pred in enumerate(predictions):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Show image
            ax.imshow(pred['display_img'])
            ax.axis('off')
            
            # Add title with prediction
            color = 'green' if pred['prediction'] == 'Real' else 'red'
            title = f"{pred['name']}\n{pred['prediction']} ({pred['confidence']:.1%})"
            ax.set_title(title, fontsize=10, color=color, weight='bold')
        
        # Hide unused subplots
        for idx in range(len(predictions), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig


def add_ood_tracking_to_trainer(
    ood_image_dir: Optional[str],
    six_channel_processor,
    class_names: List[str] = ['Real', 'Fake'],
    face_preprocessor=None
) -> Optional[OODTestTracker]:
    """Factory function to create OOD tracker if directory exists.
    
    Args:
        ood_image_dir: Directory with OOD test images (or None)
        six_channel_processor: SixChannelPreprocessor instance
        class_names: List of class names
        face_preprocessor: Optional FacePreprocessor for face detection (CRITICAL for consistency)
        
    Returns:
        OODTestTracker instance or None
    """
    if ood_image_dir is None:
        return None
    
    ood_path = Path(ood_image_dir)
    if not ood_path.exists():
        print(f"⚠️  OOD test directory not found: {ood_image_dir}")
        return None
    
    return OODTestTracker(ood_image_dir, six_channel_processor, class_names, face_preprocessor=face_preprocessor)
