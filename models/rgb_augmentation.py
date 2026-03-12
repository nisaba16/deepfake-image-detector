"""RGB Augmentation for 6-Channel Training

CRITICAL: Apply augmentations to RGB BEFORE calculating derived channels (SRM, FFT, Saturation).
If you augment the 6-channel tensor, you break the mathematical relationships.

Recommended augmentations:
1. JPEG Compression (50-90 quality) - Crucial for real-world robustness
2. Gaussian Blur - Simulates out-of-focus faces
3. Horizontal Flip - Standard geometric augmentation
4. Color Jitter - Slight brightness/contrast changes
5. Random rotation (small angles)

Usage:
    augmenter = RGBAugmentation(train=True)
    rgb_image = augmenter(rgb_image)  # Apply BEFORE channel calculation
    six_channels = preprocessor.process_face_to_6_channels(rgb_image)
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional
import random


class RGBAugmentation:
    """Apply augmentations to RGB images before 6-channel processing.
    
    This is critical because derived channels (SRM, FFT, Saturation) must be
    calculated from the augmented RGB, not augmented themselves.
    """
    
    def __init__(
        self,
        train: bool = True,
        jpeg_prob: float = 0.5,
        jpeg_quality_range: tuple = (50, 95),
        blur_prob: float = 0.3,
        blur_kernel_range: tuple = (3, 7),
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.4,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        saturation_range: float = 0.2,
        rotation_prob: float = 0.3,
        rotation_range: int = 15,
    ):
        """Initialize RGB augmentation.
        
        Args:
            train: If False, no augmentations are applied
            jpeg_prob: Probability of applying JPEG compression
            jpeg_quality_range: Range of JPEG quality (lower = more compression)
            blur_prob: Probability of applying Gaussian blur
            blur_kernel_range: Range of blur kernel sizes (odd numbers)
            hflip_prob: Probability of horizontal flip
            color_jitter_prob: Probability of color jittering
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            saturation_range: Range for saturation adjustment
            rotation_prob: Probability of small rotation
            rotation_range: Max rotation angle in degrees
        """
        self.train = train
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.rotation_prob = rotation_prob
        self.rotation_range = rotation_range
    
    def apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression artifacts.
        
        Critical for real-world robustness since most images are compressed.
        """
        quality = random.randint(*self.jpeg_quality_range)
        
        # Convert numpy to PIL
        pil_img = Image.fromarray(img)
        
        # Compress to JPEG in memory
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Read back
        compressed_img = Image.open(buffer)
        return np.array(compressed_img)
    
    def apply_gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to simulate out-of-focus images."""
        kernel_size = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
        sigma = kernel_size / 6.0
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        return blurred
    
    def apply_horizontal_flip(self, img: np.ndarray) -> np.ndarray:
        """Horizontal flip."""
        return np.fliplr(img).copy()
    
    def apply_color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply random brightness, contrast, and saturation adjustments."""
        img_float = img.astype(np.float32)
        
        # Brightness
        if random.random() < 0.5:
            brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            img_float = np.clip(img_float * brightness_factor, 0, 255)
        
        # Contrast
        if random.random() < 0.5:
            contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            mean = img_float.mean()
            img_float = np.clip((img_float - mean) * contrast_factor + mean, 0, 255)
        
        # Saturation
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img_float.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            saturation_factor = 1.0 + random.uniform(-self.saturation_range, self.saturation_range)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            img_float = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        return img_float.astype(np.uint8)
    
    def apply_rotation(self, img: np.ndarray) -> np.ndarray:
        """Apply small random rotation."""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentations to RGB image.
        
        Args:
            img: RGB image as numpy array (H, W, 3), values 0-255
            
        Returns:
            Augmented RGB image (H, W, 3), values 0-255
        """
        if not self.train:
            return img
        
        # Make a copy to avoid modifying original
        img = img.copy()
        
        # 1. JPEG Compression (most important for real-world robustness)
        if random.random() < self.jpeg_prob:
            img = self.apply_jpeg_compression(img)
        
        # 2. Gaussian Blur
        if random.random() < self.blur_prob:
            img = self.apply_gaussian_blur(img)
        
        # 3. Horizontal Flip
        if random.random() < self.hflip_prob:
            img = self.apply_horizontal_flip(img)
        
        # 4. Color Jitter
        if random.random() < self.color_jitter_prob:
            img = self.apply_color_jitter(img)
        
        # 5. Small Rotation
        if random.random() < self.rotation_prob:
            img = self.apply_rotation(img)
        
        return img


def build_rgb_augmentation(train: bool = True, aggressive: bool = False) -> RGBAugmentation:
    """Factory function to build RGB augmentation.
    
    Args:
        train: Whether to apply augmentations (False for validation/test)
        aggressive: Use more aggressive augmentations
        
    Returns:
        RGBAugmentation instance
    """
    if aggressive:
        return RGBAugmentation(
            train=train,
            jpeg_prob=0.7,
            jpeg_quality_range=(40, 90),
            blur_prob=0.4,
            hflip_prob=0.5,
            color_jitter_prob=0.5,
            brightness_range=0.3,
            contrast_range=0.3,
            saturation_range=0.3,
            rotation_prob=0.4,
            rotation_range=20,
        )
    else:
        return RGBAugmentation(
            train=train,
            jpeg_prob=0.5,
            jpeg_quality_range=(50, 95),
            blur_prob=0.3,
            hflip_prob=0.5,
            color_jitter_prob=0.4,
            brightness_range=0.2,
            contrast_range=0.2,
            saturation_range=0.2,
            rotation_prob=0.3,
            rotation_range=15,
        )
