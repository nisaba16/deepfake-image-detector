"""Advanced Loss Functions for Deepfake Detection.

This module provides specialized loss functions that improve generalization
to out-of-distribution (OOD) deepfakes by enforcing better feature separation.

Available losses:
- BinaryArcFace: Forces tight clustering of real faces, better OOD detection
- LabelSmoothing: Prevents overconfidence, improves calibration
- FocalLoss: Handles class imbalance by focusing on hard examples
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryArcFace(nn.Module):
    """Binary ArcFace Loss with Additive Angular Margin.
    
    This loss enforces a margin in the angular space, creating tighter clusters
    for "Real" faces and pushing "Fake" faces far away. This dramatically improves
    detection of novel deepfake types (like Nano Banana) that fall outside the
    tight "Real" cluster.
    
    Key advantages for OOD generalization:
    - Forces model to learn discriminative features, not just separability
    - Creates clear decision boundaries in feature space
    - New deepfake types naturally fall outside the "Real" cluster
    - Reduces false positives on unseen manipulation techniques
    
    Args:
        s: Scale factor (larger = stronger gradients). Default: 64.0
        m: Angular margin in radians (larger = stricter separation). Default: 0.5
        easy_margin: If True, uses easier margin constraint. Default: False
        
    Reference:
        "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
        Adapted for binary deepfake classification.
    """
    
    def __init__(self, s: float = 64.0, m: float = 0.5, easy_margin: bool = False):
        super(BinaryArcFace, self).__init__()
        self.s = s  # Scale factor
        self.m = m  # Margin
        self.easy_margin = easy_margin
        
        # Pre-compute margin values for efficiency
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # Threshold
        self.mm = math.sin(math.pi - m) * m  # Margin offset

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss.
        
        Args:
            logits: Model output logits of shape (batch_size, num_classes)
                   Should be cosine similarities (normalized embeddings)
            labels: Ground truth labels of shape (batch_size,)
                   0 = Real, 1 = Fake
            
        Returns:
            Scalar loss value
        """
        # Normalize logits to get cosine similarity
        cosine = logits
        
        # Compute sin(theta) from cos(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Compute cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            # Easy margin: just use phi when cos(theta) > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard margin with threshold
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding of labels
        one_hot = torch.zeros(cosine.size(), device=logits.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to the ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale by s
        output *= self.s
        
        # Compute cross-entropy loss
        return F.cross_entropy(output, labels)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-Entropy Loss with Label Smoothing.
    
    Label smoothing prevents the model from becoming over-confident by
    distributing some probability mass to incorrect classes. This improves
    calibration and generalization to OOD samples.
    
    Args:
        smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        reduction: Loss reduction method ('mean' or 'sum')
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        assert 0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.
        
        Args:
            pred: Model predictions (logits) of shape (batch_size, num_classes)
            target: Ground truth labels of shape (batch_size,)
            
        Returns:
            Scalar loss value
        """
        num_classes = pred.size(1)
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (num_classes - 1)
        
        # Create smooth labels
        smooth_label = torch.full_like(pred, smooth_value)
        smooth_label.scatter_(1, target.unsqueeze(1), confidence)
        
        # Compute loss
        log_probs = torch.log_softmax(pred, dim=1)
        loss = -torch.sum(smooth_label * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses training on hard examples.
    Useful when there's imbalance between real and fake samples.
    
    Args:
        alpha: Weighting factor for class balance (None for no weighting)
        gamma: Focusing parameter (larger = more focus on hard examples)
        reduction: Loss reduction method
        
    Reference:
        "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha: float = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Scalar loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            # Apply alpha weighting
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combine multiple losses with configurable weights.
    
    Example: Combine ArcFace with label smoothing for best of both worlds.
    
    Args:
        losses: Dictionary mapping loss names to (loss_fn, weight) tuples
    """
    
    def __init__(self, losses: dict):
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleDict({k: v[0] for k, v in losses.items()})
        self.weights = {k: v[1] for k, v in losses.items()}
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted combination of losses."""
        total_loss = 0
        for name, loss_fn in self.losses.items():
            total_loss += self.weights[name] * loss_fn(inputs, targets)
        return total_loss


def get_loss_function(loss_type: str = 'cross_entropy', **kwargs) -> nn.Module:
    """Factory function to get loss function by name.
    
    Args:
        loss_type: Type of loss ('cross_entropy', 'arcface', 'label_smoothing', 'focal')
        **kwargs: Additional arguments passed to loss constructor
        
    Returns:
        Initialized loss function
        
    Example:
        >>> criterion = get_loss_function('arcface', s=64.0, m=0.5)
        >>> criterion = get_loss_function('label_smoothing', smoothing=0.1)
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'arcface':
        return BinaryArcFace(**kwargs)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
