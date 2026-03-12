"""Custom EfficientNet for 6-channel deepfake detection.

This module provides a modified EfficientNet-B0 architecture that:
1. Accepts 6-channel input instead of standard RGB (3 channels)
2. Maintains pre-trained ImageNet weights for the RGB channels
3. Initializes new channels (Saturation, SRM, FFT) by averaging RGB weights
4. Optimized for deployment (<100MB)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from typing import Optional, Dict, Any
import numpy as np


class DeepfakeEfficientNet(nn.Module):
    """Modified EfficientNet-B0 for 6-channel deepfake detection.
    
    Architecture:
    - Input: 6 channels (RGB + Saturation + SRM + FFT)
    - Channel prioritization: Learnable weights for SRM/FFT emphasis
    - Backbone: EfficientNet-B0 (modified first conv layer)
    - Head: Binary classification (Real vs Fake)
    """
    
    def __init__(
        self, 
        pretrained: bool = True,
        num_classes: int = 2,  # Binary: Real vs Fake
        dropout: float = 0.2,
        use_channel_prioritization: bool = True
    ):
        super().__init__()
        
        self.use_channel_prioritization = use_channel_prioritization
        
        # Channel prioritization: Learnable weights to emphasize SRM/FFT channels
        # These channels contain forensic artifacts that are more discriminative
        if use_channel_prioritization:
            self.channel_weights = nn.Parameter(torch.ones(6))
            # Initialize with bias toward forensic channels
            with torch.no_grad():
                self.channel_weights[0:3] = 1.0  # RGB: baseline weight
                self.channel_weights[3] = 1.2    # Saturation: slight boost
                self.channel_weights[4] = 1.5    # SRM: high weight (forensic!)
                self.channel_weights[5] = 1.5    # FFT: high weight (forensic!)
        
        # Load pre-trained EfficientNet-B0
        self.backbone = efficientnet_b0(pretrained=pretrained)
        
        # Modify first convolutional layer for 6-channel input
        self._modify_first_conv_layer()
        
        # Replace classifier head for binary classification
        # EfficientNet-B0 has 1280 features before classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
        self.input_channels = 6
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional channel prioritization.
        
        Args:
            x: Input tensor of shape (batch_size, 6, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        if x.size(1) != 6:
            raise ValueError(f"Expected 6 input channels, got {x.size(1)}")
        
        # Apply channel prioritization to emphasize SRM/FFT
        if self.use_channel_prioritization:
            # Broadcast channel weights: (6,) -> (1, 6, 1, 1)
            weights = self.channel_weights.view(1, 6, 1, 1)
            x = x * weights
        
        return self.backbone(x)
    
    def _modify_first_conv_layer(self):
        """Modify first conv layer to accept 6-channel input.
        
        Strategy:
        - Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        - New: Conv2d(6, 32, kernel_size=3, stride=2, padding=1)
        - Copy RGB weights, average them for new channels
        """
        # Get the first convolutional layer
        first_conv = self.backbone.features[0][0]  # features is Sequential, [0] is ConvNormActivation
        
        # Original weights: (out_channels=32, in_channels=3, kernel_h, kernel_w)
        original_weight = first_conv.weight.data
        out_channels, in_channels, kh, kw = original_weight.shape
        
        # Create new conv layer with 6 input channels
        new_conv = nn.Conv2d(
            6, out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False  # EfficientNet uses batch norm, no bias
        )
        
        # Initialize new weights
        with torch.no_grad():
            # Copy RGB weights (channels 0-2)
            new_conv.weight[:, :3, :, :] = original_weight
            
            # For channels 3-5: average of RGB weights
            avg_weight = original_weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:6, :, :] = avg_weight.repeat(1, 3, 1, 1)
        
        # Replace the first conv layer
        self.backbone.features[0][0] = new_conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 6, 224, 224)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Validate input shape (avoid tensor-to-boolean during tracing)
        assert x.size(1) == 6, f"Expected 6 input channels, got {x.size(1)}"
        
        return self.backbone(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "DeepfakeEfficientNet-B0",
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def export_to_onnx(
    model: DeepfakeEfficientNet,
    save_path: str,
    input_shape: tuple = (1, 6, 224, 224),
    opset_version: int = 12
) -> None:
    """Export model to ONNX format.
    
    Args:
        model: Trained DeepfakeEfficientNet model
        save_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version
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
