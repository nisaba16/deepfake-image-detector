"""Coordinate Attention Module for Enhanced Spatial Awareness.

Coordinate Attention (CA) is more effective than standard attention for deepfake detection
because it captures long-range dependencies with precise positional information.

Key benefits for deepfake detection:
- Captures spatial inconsistencies in manipulated regions
- Lightweight (minimal parameter overhead)
- Works well with mobile architectures

Reference: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Coordinate Attention module for spatial feature enhancement.
    
    This attention mechanism encodes channel relationships along spatial dimensions,
    making it particularly effective for detecting localized artifacts in deepfakes.
    
    Args:
        inp: Number of input channels
        oup: Number of output channels
        reduction: Reduction ratio for intermediate channels (higher = fewer params)
    """
    
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super(CoordinateAttention, self).__init__()
        
        # Separate pooling for height and width dimensions
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Intermediate channel size (minimum 8 to preserve capacity)
        mip = max(8, inp // reduction)
        
        # Shared transformation
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        # Separate output transformations for height and width
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinate attention to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Attention-weighted tensor of same shape as input
        """
        identity = x
        n, c, h, w = x.size()
        
        # Pool information horizontally and vertically
        # x_h: (n, c, h, 1) - preserves height information
        # x_w: (n, c, 1, w) - preserves width information
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # Concatenate along height dimension
        y = torch.cat([x_h, x_w], dim=2)
        
        # Shared transformation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back into height and width components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Generate attention weights for each dimension
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # Apply multiplicative attention
        out = identity * a_w * a_h
        
        return out


def add_coordinate_attention(module: nn.Module, num_channels: int, reduction: int = 32) -> nn.Sequential:
    """Helper function to add CoordinateAttention after any module.
    
    Args:
        module: The module to enhance with attention
        num_channels: Number of output channels from the module
        reduction: Reduction ratio for the attention module
        
    Returns:
        Sequential module with original module + coordinate attention
        
    Example:
        >>> conv_layer = nn.Conv2d(64, 128, 3, padding=1)
        >>> enhanced_layer = add_coordinate_attention(conv_layer, 128)
    """
    return nn.Sequential(
        module,
        CoordinateAttention(num_channels, num_channels, reduction)
    )
