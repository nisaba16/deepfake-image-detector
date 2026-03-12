"""Face-aware feature extractors for deepfake detection.

Combines:
1. Spatial model (ResNet/ViT) - learns texture & lighting artifacts
2. Frequency model (FFT/Wavelet) - detects periodic GAN patterns
3. Multi-scale fusion - combine low and high-level cues
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .extractors import BaseImageFeatureExtractor, FrequencyAnalysisExtractor, _TorchBackboneExtractor
from .face_preprocessor import FacePreprocessor, create_face_preprocessor


@dataclass
class FaceAwareFrequencyExtractor(BaseImageFeatureExtractor):
    """Frequency analysis optimized for face deepfake detection.
    
    Detects periodic patterns characteristic of GAN-generated faces (StyleGAN, diffusion).
    """

    size: int = 224
    num_bins: int = 64
    log_magnitude: bool = True
    face_preprocessor: Optional[FacePreprocessor] = None

    def __post_init__(self):
        if self.face_preprocessor is None:
            self.face_preprocessor = create_face_preprocessor(
                detector_name="mediapipe",
                output_size=(self.size, self.size),
                normalize_imagenet=False,
            )
        # Use internal frequency extractor
        self._freq_extractor = FrequencyAnalysisExtractor(
            size=self.size,
            num_bins=self.num_bins,
            log_magnitude=self.log_magnitude,
        )

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract frequency features from face region."""
        # Preprocess: detect & align face
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image

        # Try to detect and crop face
        face_crop, _, metadata = self.face_preprocessor.preprocess_image(
            img_np, return_metadata=True
        )

        if face_crop is None:
            # No face detected, use full image (fallback)
            # Denormalize if needed and use full image
            face_crop_pil = image if isinstance(image, Image.Image) else Image.fromarray(img_np)
        else:
            # Denormalize to [0, 255] and convert to PIL
            face_crop_denorm = (face_crop.clip(0, 1) * 255).astype(np.uint8)
            face_crop_pil = Image.fromarray(face_crop_denorm)

        # Extract frequency features
        feats = self._freq_extractor.extract(face_crop_pil)
        
        # Add face detection confidence as metadata
        confidence = 1.0 if face_crop is not None else 0.0
        feats = np.concatenate([feats, np.array([confidence], dtype=np.float32)])
        
        self.output_dim = feats.size
        return feats


@dataclass
class FaceAwareSpatialExtractor(BaseImageFeatureExtractor):
    """Spatial model (ResNet/ViT) optimized for face deepfake detection.
    
    Learns texture & lighting artifacts characteristic of manipulated faces.
    """

    model_name: str = "resnet50"
    pretrained: bool = True
    img_size: int = 224
    device: Optional[str] = None
    face_preprocessor: Optional[FacePreprocessor] = None

    def __post_init__(self):
        if self.face_preprocessor is None:
            self.face_preprocessor = create_face_preprocessor(
                detector_name="mediapipe",
                output_size=(self.img_size, self.img_size),
                normalize_imagenet=True,
            )
        # Create underlying torch extractor
        self._spatial_extractor = _TorchBackboneExtractor(
            model_name=self.model_name,
            pretrained=self.pretrained,
            img_size=self.img_size,
            device=self.device,
        )

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract spatial features from aligned face region."""
        # Preprocess: detect & align face
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image

        face_crop, _, metadata = self.face_preprocessor.preprocess_image(
            img_np, return_metadata=True
        )

        if face_crop is None:
            # No face detected, use full image
            face_crop_pil = image if isinstance(image, Image.Image) else Image.fromarray(img_np)
        else:
            # Denormalize and convert to PIL
            face_crop_denorm = (face_crop.clip(0, 1) * 255).astype(np.uint8)
            face_crop_pil = Image.fromarray(face_crop_denorm)

        # Extract spatial features
        feats = self._spatial_extractor.extract(face_crop_pil)
        
        # Add alignment status as metadata
        alignment_status = float(metadata.get("aligned", False))
        feats = np.concatenate([feats, np.array([alignment_status], dtype=np.float32)])
        
        self.output_dim = feats.size
        return feats


@dataclass
class FaceAwareMultiModalExtractor(BaseImageFeatureExtractor):
    """Multi-modal fusion: spatial + frequency + multi-scale analysis.
    
    Combines:
    - Spatial model (ResNet/ViT) for texture/lighting
    - Frequency model (FFT) for GAN artifacts
    - Multi-scale analysis for robustness
    """

    model_name: str = "resnet50"
    pretrained: bool = True
    img_size: int = 224
    device: Optional[str] = None
    face_preprocessor: Optional[FacePreprocessor] = None
    fusion_method: str = "concat"  # "concat" or "mean"
    use_frequency: bool = True
    use_spatial: bool = True

    def __post_init__(self):
        if self.face_preprocessor is None:
            self.face_preprocessor = create_face_preprocessor(
                detector_name="mediapipe",
                output_size=(self.img_size, self.img_size),
                normalize_imagenet=True,
            )

        self._spatial_extractor = None
        self._freq_extractor = None

        if self.use_spatial:
            self._spatial_extractor = _TorchBackboneExtractor(
                model_name=self.model_name,
                pretrained=self.pretrained,
                img_size=self.img_size,
                device=self.device,
            )

        if self.use_frequency:
            self._freq_extractor = FrequencyAnalysisExtractor(
                size=self.img_size,
                num_bins=64,
                log_magnitude=True,
            )

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract multi-modal features from face."""
        # Preprocess: detect & align face
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image

        face_crop, _, metadata = self.face_preprocessor.preprocess_image(
            img_np, return_metadata=True
        )

        if face_crop is None:
            face_crop_pil = image if isinstance(image, Image.Image) else Image.fromarray(img_np)
            face_crop_normalized = None
        else:
            face_crop_denorm = (face_crop.clip(0, 1) * 255).astype(np.uint8)
            face_crop_pil = Image.fromarray(face_crop_denorm)
            face_crop_normalized = face_crop  # Keep normalized version for frequency

        feats_list = []

        # Spatial features
        if self.use_spatial and self._spatial_extractor is not None:
            spatial_feats = self._spatial_extractor.extract(face_crop_pil)
            feats_list.append(spatial_feats)

        # Frequency features
        if self.use_frequency and self._freq_extractor is not None:
            # Denormalize for frequency analysis
            if face_crop_normalized is not None:
                freq_feats = self._freq_extractor.extract(face_crop_pil)
            else:
                freq_feats = self._freq_extractor.extract(image)
            feats_list.append(freq_feats)

        if not feats_list:
            raise ValueError("No feature extractors enabled")

        # Fusion
        if self.fusion_method == "concat":
            out = np.concatenate(feats_list, axis=0)
        else:  # mean
            # Pad to max length
            max_len = max(f.shape[0] for f in feats_list)
            padded = [np.pad(f, (0, max_len - f.shape[0])) for f in feats_list]
            out = np.mean(np.stack(padded, axis=0), axis=0)

        # Add metadata
        metadata_feats = np.array(
            [
                float(metadata.get("aligned", False)),
                float(metadata.get("n_faces", 0)),
            ],
            dtype=np.float32,
        )
        out = np.concatenate([out, metadata_feats])
        self.output_dim = out.size

        return out


def build_face_aware_extractor(
    name: str,
    *,
    img_size: int = 224,
    pretrained: bool = True,
    model_name: str = "resnet50",
    detector_name: str = "mediapipe",
    fusion_method: str = "concat",
    face_preprocessor: Optional[FacePreprocessor] = None,
) -> BaseImageFeatureExtractor:
    """Factory to create face-aware extractors.
    
    Args:
        name: "face_frequency", "face_spatial", or "face_multimodal"
        img_size: output face size
        pretrained: use pretrained weights
        model_name: backbone model for spatial extractor
        detector_name: face detector ("mediapipe", "mtcnn", "retinaface")
        fusion_method: "concat" or "mean"
        face_preprocessor: optional custom preprocessor
    
    Returns:
        FaceAware*Extractor instance
    """
    if face_preprocessor is None:
        face_preprocessor = create_face_preprocessor(
            detector_name=detector_name,
            output_size=(img_size, img_size),
            normalize_imagenet=True,
        )

    name = name.lower()
    
    if name == "face_frequency":
        return FaceAwareFrequencyExtractor(
            size=img_size,
            face_preprocessor=face_preprocessor,
        )
    elif name == "face_spatial":
        return FaceAwareSpatialExtractor(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            face_preprocessor=face_preprocessor,
        )
    elif name == "face_multimodal":
        return FaceAwareMultiModalExtractor(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            face_preprocessor=face_preprocessor,
            fusion_method=fusion_method,
        )
    else:
        raise ValueError(
            f"Unknown face-aware extractor: {name}. "
            "Choose from: face_frequency, face_spatial, face_multimodal"
        )
