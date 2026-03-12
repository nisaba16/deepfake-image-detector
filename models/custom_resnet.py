"""Custom ResNet50 for 6-channel deepfake detection.

This module provides a modified ResNet-50 architecture that:
1. Accepts 6-channel input instead of standard RGB (3 channels)
2. Maintains pre-trained ImageNet weights for the RGB channels
3. Initializes new channels (Saturation, SRM, FFT) by averaging RGB weights
4. Optimized for deployment (~100MB)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import Optional, Dict, Any


class DeepfakeResNet50(nn.Module):
    """Modified ResNet-50 for 6-channel deepfake detection.
    
    Architecture:
    - Input: 6 channels (RGB + Saturation + SRM + FFT)
    - Channel prioritization: Learnable weights for SRM/FFT emphasis
    - Backbone: ResNet-50 (modified first conv layer)
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
        
        # Channel prioritization: Emphasize SRM/FFT forensic channels
        if use_channel_prioritization:
            self.channel_weights = nn.Parameter(torch.ones(6))
            with torch.no_grad():
                self.channel_weights[0:3] = 1.0  # RGB: baseline
                self.channel_weights[3] = 1.2    # Saturation: +20%
                self.channel_weights[4] = 1.5    # SRM: +50% (forensic!)
                self.channel_weights[5] = 1.5    # FFT: +50% (forensic!)
        
        # Load pre-trained ResNet-50
        self.backbone = resnet50(pretrained=pretrained)
        
        # Modify first convolutional layer for 6-channel input
        self._modify_first_conv_layer()
        
        # Replace classifier head for binary classification
        # ResNet-50 has 2048 features before FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
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
        """Forward pass with channel prioritization.
        
        Args:
            x: Input tensor of shape (batch_size, 6, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        if x.size(1) != 6:
            raise ValueError(f"Expected 6 input channels, got {x.size(1)}")
        
        # Apply channel prioritization
        if self.use_channel_prioritization:
            weights = self.channel_weights.view(1, 6, 1, 1)
            x = x * weights
        
        return self.backbone(x)
    
    def _modify_first_conv_layer(self):
        """Modify first conv layer to accept 6-channel input.
        
        Strategy:
        - Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        - New: Conv2d(6, 64, kernel_size=7, stride=2, padding=3)
        - Copy RGB weights, average them for new channels
        """
        # Get the first convolutional layer
        first_conv = self.backbone.conv1
        
        # Original weights: (out_channels=64, in_channels=3, kernel_h, kernel_w)
        original_weight = first_conv.weight.data
        out_channels, in_channels, kh, kw = original_weight.shape
        
        # Create new conv layer with 6 input channels
        new_conv = nn.Conv2d(
            6, out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False  # ResNet uses batch norm, no bias in conv1
        )
        
        # Initialize new weights
        with torch.no_grad():
            # Copy RGB weights (channels 0-2)
            new_conv.weight[:, :3, :, :] = original_weight
            
            # For channels 3-5: average of RGB weights
            avg_weight = original_weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:6, :, :] = avg_weight.repeat(1, 3, 1, 1)
        
        # Replace the first conv layer
        self.backbone.conv1 = new_conv
    
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
            "model_name": "DeepfakeResNet50",
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def freeze_early_layers(self, freeze_until: str = 'layer3'):
        """Freeze early layers to prevent overfitting on small datasets.
        
        Args:
            freeze_until: Freeze all layers before and including this layer.
                         Options: 'conv1', 'layer1', 'layer2', 'layer3'
                         
        Recommended for ResNet-50 on small datasets:
        - freeze_until='layer2': Freeze first ~60% of model (layers 1-2)
        - freeze_until='layer3': Freeze first ~80% of model (layers 1-3)
        """
        layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2']
        
        if freeze_until == 'layer3':
            layers_to_freeze.append('layer3')
        elif freeze_until == 'layer2':
            pass  # Already includes layer2
        elif freeze_until == 'layer1':
            layers_to_freeze = ['conv1', 'bn1', 'layer1']
        elif freeze_until == 'conv1':
            layers_to_freeze = ['conv1', 'bn1']
        else:
            raise ValueError(f"freeze_until must be one of: conv1, layer1, layer2, layer3")
        
        frozen_params = 0
        for name, param in self.backbone.named_parameters():
            # Check if this parameter belongs to a layer we want to freeze
            if any(name.startswith(layer) for layer in layers_to_freeze):
                param.requires_grad = False
                frozen_params += param.numel()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"🔒 Froze {frozen_params:,} parameters ({frozen_params/total_params*100:.1f}% of model)")
        print(f"📊 Trainable: {trainable_params:,} / {total_params:,} parameters")


def export_to_onnx(
    model: DeepfakeResNet50,
    save_path: str,
    input_shape: tuple = (1, 6, 224, 224),
    opset_version: int = 12
) -> None:
    """Export model to ONNX format.
    
    Args:
        model: Trained DeepfakeResNet50 model
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
