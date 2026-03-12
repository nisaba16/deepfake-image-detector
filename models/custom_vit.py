"""Custom Vision Transformer for 6-channel deepfake detection.

This module provides a modified ViT-B/16 architecture that:
1. Accepts 6-channel input instead of standard RGB (3 channels)
2. Maintains pre-trained ImageNet weights for the RGB channels
3. Initializes new channels (Saturation, SRM, FFT) by averaging RGB weights
4. Uses smaller patch embedding for better performance
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from typing import Optional, Dict, Any


class DeepfakeViT(nn.Module):
    """Modified ViT-B/16 for 6-channel deepfake detection.
    
    Architecture:
    - Input: 6 channels (RGB + Saturation + SRM + FFT)
    - Backbone: Vision Transformer Base/16 (modified patch embedding)
    - Head: Binary classification (Real vs Fake)
    """
    
    def __init__(
        self, 
        pretrained: bool = True,
        num_classes: int = 2,  # Binary: Real vs Fake
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Load pre-trained ViT-B/16
        self.backbone = vit_b_16(pretrained=pretrained)
        
        # Modify patch embedding layer for 6-channel input
        self._modify_patch_embedding()
        
        # Replace classifier head for binary classification
        # ViT-B has 768 features (hidden dimension)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
        self.input_channels = 6
    
    def _modify_patch_embedding(self):
        """Modify patch embedding conv layer to accept 6-channel input.
        
        Strategy:
        - Original: Conv2d(3, 768, kernel_size=16, stride=16)
        - New: Conv2d(6, 768, kernel_size=16, stride=16)
        - Copy RGB weights, average them for new channels
        """
        # Get the patch embedding convolutional layer
        conv_proj = self.backbone.conv_proj
        
        # Original weights: (out_channels=768, in_channels=3, kernel_h=16, kernel_w=16)
        original_weight = conv_proj.weight.data
        original_bias = conv_proj.bias.data if conv_proj.bias is not None else None
        out_channels, in_channels, kh, kw = original_weight.shape
        
        # Create new conv layer with 6 input channels
        new_conv = nn.Conv2d(
            6, out_channels,
            kernel_size=conv_proj.kernel_size,
            stride=conv_proj.stride,
            padding=conv_proj.padding,
            bias=conv_proj.bias is not None
        )
        
        # Initialize new weights
        with torch.no_grad():
            # Copy RGB weights (channels 0-2)
            new_conv.weight[:, :3, :, :] = original_weight
            
            # For channels 3-5: average of RGB weights
            avg_weight = original_weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:6, :, :] = avg_weight.repeat(1, 3, 1, 1)
            
            # Copy bias if it exists
            if original_bias is not None:
                new_conv.bias.data = original_bias
        
        # Replace the patch embedding layer
        self.backbone.conv_proj = new_conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 6, 224, 224)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Validate input shape
        if x.size(1) != 6:
            raise ValueError(f"Expected 6 input channels, got {x.size(1)}")
        
        return self.backbone(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "DeepfakeViT-B/16",
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def export_to_onnx(
    model: DeepfakeViT,
    save_path: str,
    input_shape: tuple = (1, 6, 224, 224),
    opset_version: int = 13
) -> None:
    """Export model to ONNX format.
    
    Args:
        model: Trained DeepfakeViT model
        save_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version (default 13 for ViT support)
    """
    # Move model to CPU for export
    device = next(model.parameters()).device
    model = model.cpu()
    model.eval()
    
    # Create dummy input on CPU
    dummy_input = torch.randn(*input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Move model back to original device
    model = model.to(device)
