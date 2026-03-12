from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import math
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage


class BaseImageFeatureExtractor:
    """Base interface for image feature extractors.

    Implementors should return a 1D numpy array (features) from a PIL Image.
    """

    output_dim: Optional[int] = None

    def extract(self, image: Image.Image) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class FrequencyAnalysisExtractor(BaseImageFeatureExtractor):
    """Frequency analysis using 2D FFT radial power spectrum.

    - Converts to grayscale
    - Resizes to square `size`
    - Computes FFT magnitude, then radial average into `num_bins`
    - Appends simple global stats: mean, std, high-frequency energy ratio
    """

    size: int = 224
    num_bins: int = 64
    log_magnitude: bool = True

    def _radial_profile(self, mag: np.ndarray, num_bins: int) -> np.ndarray:
        h, w = mag.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        y, x = np.indices((h, w))
        r = np.hypot(x - cx, y - cy)
        r_max = r.max()
        # Bin edges from 0..r_max
        bins = np.linspace(0, r_max + 1e-6, num_bins + 1)
        which = np.digitize(r.ravel(), bins) - 1
        which = np.clip(which, 0, num_bins - 1)
        sums = np.bincount(which, weights=mag.ravel(), minlength=num_bins)
        counts = np.bincount(which, minlength=num_bins)
        counts[counts == 0] = 1
        radial = sums / counts
        return radial

    def extract(self, image: Image.Image) -> np.ndarray:
        img = image.convert("L").resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # 2D FFT
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        if self.log_magnitude:
            mag = np.log1p(mag)
        radial = self._radial_profile(mag, self.num_bins)
        # Global stats
        mean_val = float(mag.mean())
        std_val = float(mag.std())
        # High frequency: top 25% of radial bins
        k = max(1, self.num_bins // 4)
        high_freq_ratio = float(radial[-k:].sum() / (radial.sum() + 1e-8))
        feats = np.concatenate([
            radial.astype(np.float32),
            np.asarray([mean_val, std_val, high_freq_ratio], dtype=np.float32),
        ])
        self.output_dim = feats.size
        return feats


@dataclass
class SixChannelPreprocessor:
    """Client-side 6-channel preprocessor for MobileNetV3.
    
    Generates a 6-channel tensor from a cropped face image:
    - Ch 0-2: RGB (normalized 0-1)
    - Ch 3: Saturation from HSV (normalized 0-1)
    - Ch 4: SRM Noise Filter (5x5 high-pass kernel)
    - Ch 5: FFT Magnitude (log scale, centered, normalized)
    
    All operations are simple enough to replicate in JavaScript.
    """
    
    output_size: tuple[int, int] = (224, 224)
    
    def __init__(self):
        # SRM High-Pass Filter Kernel (5x5)
        # Simple high-pass kernel for noise residual extraction
        self.srm_kernel = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  2,  2,  2, -1],
            [-1,  2,  8,  2, -1],
            [-1,  2,  2,  2, -1],
            [-1, -1, -1, -1, -1]
        ], dtype=np.float32) / 16.0
    
    def process_face_to_6_channels(self, face_image: np.ndarray) -> np.ndarray:
        """Convert face image (H, W, 3) to 6-channel tensor (6, 224, 224).
        
        Args:
            face_image: RGB face image as numpy array (H, W, 3), values 0-255
            
        Returns:
            6-channel tensor of shape (6, 224, 224)
        """
        # Ensure input is the right size and format
        if face_image.shape[-1] != 3:
            raise ValueError("Input must be RGB image with shape (H, W, 3)")
            
        # Resize to target size if needed
        if face_image.shape[:2] != self.output_size:
            face_image = cv2.resize(face_image, self.output_size)
        
        h, w = self.output_size
        channels = np.zeros((6, h, w), dtype=np.float32)
        
        # Normalize to 0-1 for processing
        face_norm = face_image.astype(np.float32) / 255.0
        
        # Channels 0-2: RGB (normalized 0-1)
        channels[0] = face_norm[:, :, 0]  # R
        channels[1] = face_norm[:, :, 1]  # G
        channels[2] = face_norm[:, :, 2]  # B
        
        # Channel 3: Saturation from HSV
        hsv = cv2.cvtColor((face_norm * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        channels[3] = hsv[:, :, 1].astype(np.float32) / 255.0
        
        # Channel 4: SRM Noise Filter
        # Apply to grayscale version for simplicity
        gray = cv2.cvtColor((face_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        # Apply SRM filter using scipy convolution
        noise_residual = ndimage.convolve(gray_norm, self.srm_kernel, mode='constant')
        # Normalize to 0-1 range (center around 0.5)
        noise_residual = np.clip(noise_residual + 0.5, 0, 1)
        channels[4] = noise_residual
        
        # Channel 5: FFT Magnitude (log scale, centered, normalized)
        # Apply to grayscale for consistency
        f = np.fft.fft2(gray_norm)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        log_mag = np.log1p(mag)  # log(1 + mag) for numerical stability
        # Normalize to 0-1
        log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
        channels[5] = log_mag_norm
        
        return channels
    
    def __call__(self, face_image: np.ndarray) -> np.ndarray:
        """Convenience method for calling process_face_to_6_channels."""
        return self.process_face_to_6_channels(face_image)


# Simplified factory for client-side architecture
def build_six_channel_preprocessor(output_size: tuple[int, int] = (224, 224)) -> SixChannelPreprocessor:
    """Build the 6-channel preprocessor for MobileNetV3 client-side inference."""
    preprocessor = SixChannelPreprocessor()
    preprocessor.output_size = output_size
    return preprocessor


# Legacy factory for frequency extractor (kept for compatibility)
def build_extractor(
    name: str,
    *,
    img_size: int = 224,
    **kwargs
) -> BaseImageFeatureExtractor:
    """Legacy factory - only supports frequency extractor for client-side compatibility."""
    name = name.lower()
    if name == "frequency":
        extractor: BaseImageFeatureExtractor = FrequencyAnalysisExtractor(size=img_size)
    else:
        raise ValueError(f"Heavy extractors removed for client-side architecture. Only 'frequency' supported, got: {name}")
    return extractor
