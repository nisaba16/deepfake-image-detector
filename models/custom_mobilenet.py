"""Custom MobileNetV3 for 6-channel deepfake detection.

This module provides a modified MobileNetV3-Small architecture that:
1. Accepts 6-channel input instead of standard RGB (3 channels)
2. Maintains pre-trained ImageNet weights for the RGB channels
3. Initializes new channels (Saturation, SRM, FFT) by averaging RGB weights
4. Optimized for client-side deployment via ONNX export (<20MB)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from typing import Optional, Dict, Any
import numpy as np

try:
    from .coordinate_attention import CoordinateAttention
except ImportError:
    from coordinate_attention import CoordinateAttention


class DeepfakeMobileNetV3(nn.Module):
    """Modified MobileNetV3-Small for 6-channel deepfake detection.
    
    Architecture:
    - Input: 6 channels (RGB + Saturation + SRM + FFT)
    - Channel prioritization: Learnable weights for SRM/FFT channels
    - Coordinate Attention: Enhances spatial feature awareness
    - Backbone: MobileNetV3-Small (modified first conv layer)
    - Head: Binary classification (Real vs Fake)
    - Export: ONNX compatible for browser inference
    """
    
    def __init__(
        self, 
        pretrained: bool = True,
        num_classes: int = 2,  # Binary: Real vs Fake
        dropout: float = 0.2,
        use_coordinate_attention: bool = True,
        use_channel_prioritization: bool = True,
        ca_reduction: int = 32
    ):
        super().__init__()
        
        self.use_coordinate_attention = use_coordinate_attention
        self.use_channel_prioritization = use_channel_prioritization
        
        # Channel prioritization layer (applied before backbone)
        # Learnable weights to emphasize SRM/FFT channels which contain deepfake artifacts
        if use_channel_prioritization:
            self.channel_weights = nn.Parameter(torch.ones(6))
            # Initialize with slight bias toward forensic channels (SRM, FFT)
            with torch.no_grad():
                self.channel_weights[0:3] = 1.0  # RGB: normal weight
                self.channel_weights[3] = 1.2    # Saturation: slightly higher
                self.channel_weights[4] = 1.5    # SRM: higher (forensic)
                self.channel_weights[5] = 1.5    # FFT: higher (forensic)
        
        # Load pre-trained MobileNetV3-Small
        self.backbone = mobilenet_v3_small(pretrained=pretrained)
        
        # Modify first convolutional layer for 6-channel input
        self._modify_first_conv_layer()
        
        # Add Coordinate Attention after key bottleneck layers
        if use_coordinate_attention:
            self._add_coordinate_attention(ca_reduction)
        
        # Replace classifier head for binary classification
        # MobileNetV3-Small has 576 features before classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(256, 64),
            nn.Hardswish(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(64, num_classes)
        )
        
        # Store metadata for ONNX export
        self.input_channels = 6
        self.num_classes = num_classes
        
    def _modify_first_conv_layer(self):
        """Modify first conv layer to accept 6 channels instead of 3.
        
        The new channels (Saturation, SRM, FFT) are initialized by averaging
        the pre-trained RGB weights to preserve learned patterns.
        """
        # Get the original first convolution layer
        original_conv = self.backbone.features[0][0]  # First conv in first InvertedResidual
        
        # Create new conv layer with 6 input channels
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights
        with torch.no_grad():
            # Copy original RGB weights (channels 0, 1, 2)
            new_conv.weight.data[:, :3, :, :] = original_conv.weight.data
            
            # Initialize new channels (3, 4, 5) by averaging RGB weights
            rgb_mean = original_conv.weight.data.mean(dim=1, keepdim=True)
            new_conv.weight.data[:, 3:6, :, :] = rgb_mean.repeat(1, 3, 1, 1)
            
            # Copy bias if it exists
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data
        
        # Replace the first conv layer
        self.backbone.features[0][0] = new_conv
    
    def _add_coordinate_attention(self, reduction: int = 32):
        """Add Coordinate Attention modules after key layers.
        
        CoordinateAttention is inserted after layers that extract important features:
        - After layer 3 (early features, ~24 channels)
        - After layer 6 (mid features, ~40 channels)  
        - After layer 9 (high-level features, ~96 channels)
        
        This helps the model focus on spatial regions with deepfake artifacts.
        """
        # MobileNetV3-Small architecture has 12 InvertedResidual blocks
        # We add attention after blocks 3, 6, 9 for multi-scale awareness
        
        attention_layers = [3, 6, 9]
        for layer_idx in attention_layers:
            if layer_idx < len(self.backbone.features):
                # Get output channels of this layer
                block = self.backbone.features[layer_idx]
                # InvertedResidual blocks have a specific output channel count
                # We'll wrap the entire block with attention
                out_channels = block.out_channels if hasattr(block, 'out_channels') else None
                
                if out_channels is None:
                    # Fallback: try to infer from the block's last conv
                    try:
                        for module in reversed(list(block.modules())):
                            if isinstance(module, nn.Conv2d):
                                out_channels = module.out_channels
                                break
                    except:
                        continue
                
                if out_channels:
                    # Create attention-enhanced block
                    original_block = self.backbone.features[layer_idx]
                    self.backbone.features[layer_idx] = nn.Sequential(
                        original_block,
                        CoordinateAttention(out_channels, out_channels, reduction)
                    )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 6, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        if x.size(1) != 6:
            raise ValueError(f"Expected 6 input channels, got {x.size(1)}")
        
        # Apply channel prioritization
        if self.use_channel_prioritization:
            # Apply learnable per-channel weights
            # Shape: (batch, 6, H, W) * (6,) -> (batch, 6, H, W)
            weights = self.channel_weights.view(1, 6, 1, 1)
            x = x * weights
            
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, 6, height, width)
            
        Returns:
            Probabilities tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard predictions.
        
        Args:
            x: Input tensor of shape (batch_size, 6, height, width)
            
        Returns:
            Class predictions of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging/debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "DeepfakeMobileNetV3",
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def export_to_onnx(
    model: DeepfakeMobileNetV3,
    output_path: str,
    input_shape: tuple = (1, 6, 224, 224),
    opset_version: int = 11,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> None:
    """Export the model to ONNX format for browser deployment.
    
    Args:
        model: Trained DeepfakeMobileNetV3 model
        output_path: Path to save the ONNX model (e.g., "model.onnx")
        input_shape: Input tensor shape (batch_size, channels, height, width)
        opset_version: ONNX opset version (11 is widely supported)
        dynamic_axes: Dictionary specifying dynamic axes for variable batch size
    """
    # Move model to CPU for export
    device = next(model.parameters()).device
    model = model.cpu()
    model.eval()
    
    # Create dummy input on CPU
    dummy_input = torch.randn(*input_shape)
    
    # Default dynamic axes for variable batch size
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    # Move model back to original device
    model = model.to(device)
    
    print(f"Model exported to {output_path}")
    
    # Check file size
    import os
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 20:
        print("⚠️  Warning: Model size exceeds 20MB target for client-side deployment")
    else:
        print("✅ Model size is within 20MB limit for client-side deployment")


def create_deepfake_mobilenet(
    pretrained: bool = True,
    num_classes: int = 2,
    dropout: float = 0.2,
    use_coordinate_attention: bool = True,
    use_channel_prioritization: bool = True,
    ca_reduction: int = 32
) -> DeepfakeMobileNetV3:
    """Factory function to create a DeepfakeMobileNetV3 model.
    
    Args:
        pretrained: Whether to use ImageNet pre-trained weights
        num_classes: Number of output classes (2 for binary classification)
        dropout: Dropout rate in classifier head
        use_coordinate_attention: Whether to add CoordinateAttention modules
        use_channel_prioritization: Whether to use learnable channel weights
        ca_reduction: Reduction ratio for CoordinateAttention
        
    Returns:
        Configured DeepfakeMobileNetV3 model
    """
    model = DeepfakeMobileNetV3(
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        use_coordinate_attention=use_coordinate_attention,
        use_channel_prioritization=use_channel_prioritization,
        ca_reduction=ca_reduction
    )
    
    # Print model info
    info = model.get_model_info()
    print(f"Created {info['model_name']}:")
    print(f"  - Input channels: {info['input_channels']}")
    print(f"  - Output classes: {info['num_classes']}")
    print(f"  - Parameters: {info['total_parameters']:,} ({info['model_size_mb']:.1f} MB)")
    
    return model


# Testing and validation functions
def test_model_shapes():
    """Test that the model accepts 6-channel input and produces correct outputs."""
    model = create_deepfake_mobilenet(pretrained=False)  # Faster for testing
    model.eval()
    
    # Test different batch sizes
    test_cases = [
        (1, 6, 224, 224),   # Single image
        (4, 6, 224, 224),   # Small batch
        (16, 6, 224, 224),  # Larger batch
    ]
    
    print("\nTesting model shapes:")
    for shape in test_cases:
        dummy_input = torch.randn(*shape)
        with torch.no_grad():
            output = model(dummy_input)
            probs = model.predict_proba(dummy_input)
            preds = model.predict(dummy_input)
        
        print(f"Input {shape} -> Output {output.shape}, Probs {probs.shape}, Preds {preds.shape}")
        
        # Verify shapes
        assert output.shape == (shape[0], 2), f"Expected output shape ({shape[0]}, 2), got {output.shape}"
        assert probs.shape == (shape[0], 2), f"Expected probs shape ({shape[0]}, 2), got {probs.shape}"
        assert preds.shape == (shape[0],), f"Expected preds shape ({shape[0]},), got {preds.shape}"
    
    print("✅ All shape tests passed!")


if __name__ == "__main__":
    # Run basic tests
    test_model_shapes()
    
    # Demo model creation
    model = create_deepfake_mobilenet(pretrained=True)
    
    # Demo ONNX export (commented out to avoid creating files during testing)
    # export_to_onnx(model, "deepfake_mobilenet.onnx")