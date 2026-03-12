"""Face-specific preprocessing for deepfake detection.

This module provides face detection, alignment, and normalization utilities
optimized for face deepfake detection tasks (swap, reenactment, diffusion).

Key steps:
1. Face detection (MTCNN, RetinaFace, or Mediapipe)
2. Face alignment (normalize orientation using landmarks)
3. Crop & resize to fixed size (e.g., 224×224)
4. Optional landmark-based masking (eyes, lips, jawline focus)
5. Normalization (ImageNet mean/std)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import cv2
import warnings


class FaceDetector:
    """Base interface for face detection."""

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces and return bounding boxes.
        
        Returns:
            List of (x_min, y_min, x_max, y_max) bounding boxes
        """
        raise NotImplementedError

    def detect_with_landmarks(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """Detect faces with optional landmarks.
        
        Returns:
            (bboxes, landmarks) where landmarks shape is (n_faces, n_points, 2)
        """
        return self.detect(image), None


class MediapipeFaceDetector(FaceDetector):
    """Face detection using MediaPipe (lightweight, no heavy deps)."""

    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection
            self.detector = self.mp_face.FaceDetection(
                model_selection=1,  # 1 = full range, 0 = short-range
                min_detection_confidence=min_detection_confidence,
            )
            self.mp = mp
        except ImportError as e:
            raise RuntimeError(
                "MediaPipe is required for face detection. "
                "Install it with: pip install mediapipe"
            ) from e

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe."""
        # MediaPipe expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
        else:
            rgb = image

        results = self.detector.process(rgb)
        bboxes = []

        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                x_max = int((bbox.xmin + bbox.width) * w)
                y_max = int((bbox.ymin + bbox.height) * h)
                # Clamp
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                bboxes.append((x_min, y_min, x_max, y_max))

        return bboxes


class MTCNNFaceDetector(FaceDetector):
    """Face detection using MTCNN (returns landmarks)."""

    def __init__(self):
        try:
            from facenet_pytorch import MTCNN
            self.detector = MTCNN(keep_all=True)
        except ImportError as e:
            raise RuntimeError(
                "facenet-pytorch is required for MTCNN detection. "
                "Install it with: pip install facenet-pytorch"
            ) from e

    def detect_with_landmarks(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """Detect faces with landmarks using MTCNN."""
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image

        batches_boxes, probs, landmarks = self.detector.detect(
            pil_image, landmarks=True
        )

        if batches_boxes is None or len(batches_boxes) == 0:
            return [], None

        bboxes = [
            (int(x_min), int(y_min), int(x_max), int(y_max))
            for x_min, y_min, x_max, y_max in batches_boxes
        ]
        return bboxes, landmarks

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        bboxes, _ = self.detect_with_landmarks(image)
        return bboxes


class RetinaFaceDetector(FaceDetector):
    """Face detection using RetinaFace (good balance of speed/accuracy)."""

    def __init__(self):
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
        except ImportError as e:
            raise RuntimeError(
                "retinaface is required for RetinaFace detection. "
                "Install it with: pip install retinaface"
            ) from e

    def detect_with_landmarks(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """Detect faces with landmarks using RetinaFace."""
        # RetinaFace returns dict: {"face_1": {"bbox": [...], "landmarks": {...}}, ...}
        resp = self.detector.detect_faces(image)

        bboxes = []
        landmarks_list = []

        for key in resp:
            bbox = resp[key]["bbox"]
            x_min, y_min, w, h = bbox
            bboxes.append((int(x_min), int(y_min), int(x_min + w), int(y_min + h)))

            # Extract landmarks: left_eye, right_eye, nose, mouth_left, mouth_right
            lms = resp[key]["landmarks"]
            lm_points = np.array(
                [lms["left_eye"], lms["right_eye"], lms["nose"], lms["mouth_left"], lms["mouth_right"]],
                dtype=np.float32,
            )
            landmarks_list.append(lm_points)

        landmarks = np.array(landmarks_list) if landmarks_list else None
        return bboxes, landmarks

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        bboxes, _ = self.detect_with_landmarks(image)
        return bboxes


def face_alignment_transform(
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (224, 224),
) -> Tuple[cv2.Mat, Tuple[int, int]]:
    """Compute similarity transform to align face landmarks (eyes horizontal).
    
    Uses left and right eye landmarks to compute alignment.
    
    Args:
        landmarks: shape (5, 2) with [left_eye, right_eye, nose, mouth_l, mouth_r]
        output_size: target size (h, w)
    
    Returns:
        (transform_matrix, new_landmarks_size)
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # Target eyes position in output image
    output_h, output_w = output_size
    left_eye_target = np.array([output_w * 0.3, output_h * 0.4], dtype=np.float32)
    right_eye_target = np.array([output_w * 0.7, output_h * 0.4], dtype=np.float32)

    # Compute similarity transform (rotation + scale + translation)
    src_pts = np.array([left_eye, right_eye], dtype=np.float32)
    dst_pts = np.array([left_eye_target, right_eye_target], dtype=np.float32)

    # Use cv2.estimateAffinePartial2D for similarity transform
    transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if transform_matrix is None:
        # Fallback: identity
        transform_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    return transform_matrix, output_size


def align_and_crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    landmarks: Optional[np.ndarray] = None,
    output_size: Tuple[int, int] = (224, 224),
    margin: float = 0.2,
) -> Tuple[np.ndarray, dict]:
    """Crop and align face region.
    
    Args:
        image: input image (BGR or RGB)
        bbox: (x_min, y_min, x_max, y_max)
        landmarks: optional (5, 2) landmarks for alignment
        output_size: output face size
        margin: expand bbox by this factor
    
    Returns:
        (aligned_face_image, metadata)
    """
    x_min, y_min, x_max, y_max = bbox
    h, w = image.shape[:2]

    # Expand bbox with margin
    face_w = x_max - x_min
    face_h = y_max - y_min
    x_min = max(0, int(x_min - face_w * margin))
    y_min = max(0, int(y_min - face_h * margin))
    x_max = min(w, int(x_max + face_w * margin))
    y_max = min(h, int(y_max + face_h * margin))

    # Crop
    face_crop = image[y_min:y_max, x_min:x_max].copy()

    metadata = {
        "original_bbox": bbox,
        "expanded_bbox": (x_min, y_min, x_max, y_max),
        "aligned": False,
        "face_size": (face_crop.shape[1], face_crop.shape[0]),
    }

    # Align if landmarks provided
    if landmarks is not None:
        # Adjust landmarks to cropped coordinates
        landmarks_cropped = landmarks.copy()
        landmarks_cropped[:, 0] -= x_min
        landmarks_cropped[:, 1] -= y_min

        # Check if landmarks are within crop
        if (
            (landmarks_cropped >= 0).all()
            and (landmarks_cropped[:, 0] < face_crop.shape[1]).all()
            and (landmarks_cropped[:, 1] < face_crop.shape[0]).all()
        ):
            transform_matrix, _ = face_alignment_transform(landmarks_cropped, output_size)
            face_crop = cv2.warpAffine(
                face_crop, transform_matrix, output_size, borderMode=cv2.BORDER_REFLECT
            )
            metadata["aligned"] = True
            metadata["transform_matrix"] = transform_matrix
        else:
            # Landmarks out of crop, skip alignment
            face_crop = cv2.resize(face_crop, output_size, interpolation=cv2.INTER_CUBIC)
    else:
        # No landmarks, just resize
        face_crop = cv2.resize(face_crop, output_size, interpolation=cv2.INTER_CUBIC)

    return face_crop, metadata


@dataclass
class FacePreprocessor:
    """Main face preprocessing pipeline.
    
    Attributes:
        face_detector: FaceDetector instance
        output_size: target face size (h, w)
        return_landmarks: whether to return landmarks for each face
        normalize_imagenet: whether to apply ImageNet normalization
    """

    face_detector: FaceDetector
    output_size: Tuple[int, int] = (224, 224)
    return_landmarks: bool = False
    normalize_imagenet: bool = True
    margin: float = 0.2

    def preprocess_image(
        self, image: Image.Image | np.ndarray, return_metadata: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """Preprocess image: detect faces, align, crop, normalize.
        
        Returns:
            (processed_face_image, landmarks, metadata)
            If no face found, returns (None, None, metadata)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Ensure BGR for OpenCV
        if image_np.shape[2] == 3 and image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        # Detect faces
        if hasattr(self.face_detector, "detect_with_landmarks"):
            bboxes, landmarks = self.face_detector.detect_with_landmarks(image_np)
        else:
            bboxes = self.face_detector.detect(image_np)
            landmarks = None

        if not bboxes:
            return None, None, {"n_faces": 0, "error": "No faces detected"}

        # Process first (largest) face
        if landmarks is not None:
            face_landmarks = landmarks[0]
        else:
            face_landmarks = None

        face_crop, metadata = align_and_crop_face(
            image_np, bboxes[0], face_landmarks, self.output_size, self.margin
        )

        # Normalize to [0, 1]
        face_crop = face_crop.astype(np.float32) / 255.0

        # ImageNet normalization
        if self.normalize_imagenet:
            face_crop = self._normalize_imagenet(face_crop)

        metadata["n_faces"] = len(bboxes)
        metadata["all_bboxes"] = bboxes

        return face_crop, face_landmarks, metadata

    @staticmethod
    def _normalize_imagenet(image: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # Assume image is (h, w, 3) and already in [0, 1]
        return (image - mean) / std

    @staticmethod
    def denormalize_imagenet(image: np.ndarray) -> np.ndarray:
        """Reverse ImageNet normalization."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return image * std + mean


def create_landmark_mask(
    image_shape: Tuple[int, int],
    landmarks: np.ndarray,
    focus_zones: str = "eyes_lips_jaw",
    expansion: int = 20,
) -> np.ndarray:
    """Create mask focusing on high-impact facial zones.
    
    Args:
        image_shape: (h, w)
        landmarks: (5, 2) [left_eye, right_eye, nose, mouth_l, mouth_r]
        focus_zones: "eyes_lips_jaw", "eyes_only", or "all"
        expansion: expand mask region by this pixel radius
    
    Returns:
        Binary mask (h, w) where 1 = focus zone
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if focus_zones == "eyes_only":
        points = landmarks[[0, 1]]  # left, right eye
    elif focus_zones == "eyes_lips_jaw":
        # Eyes, mouth corners, estimate jaw area
        points = landmarks[[0, 1, 3, 4]]  # left_eye, right_eye, mouth_left, mouth_right
    else:  # all
        points = landmarks

    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(mask, (x, y), expansion, 1, -1)

    # Optional: dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion, expansion))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# Preset detectors
def create_face_preprocessor(
    detector_name: str = "mediapipe",
    output_size: Tuple[int, int] = (224, 224),
    normalize_imagenet: bool = True,
    margin: float = 0.2,
) -> FacePreprocessor:
    """Factory function to create a FacePreprocessor with specified detector.
    
    Args:
        detector_name: "mediapipe", "mtcnn", or "retinaface"
        output_size: target face size
        normalize_imagenet: apply ImageNet normalization
        margin: expand bbox by this factor
    
    Returns:
        FacePreprocessor instance
    """
    detector_name = detector_name.lower()
    
    if detector_name == "mediapipe":
        detector = MediapipeFaceDetector()
    elif detector_name == "mtcnn":
        detector = MTCNNFaceDetector()
    elif detector_name == "retinaface":
        detector = RetinaFaceDetector()
    else:
        raise ValueError(
            f"Unknown detector: {detector_name}. "
            "Choose from: mediapipe, mtcnn, retinaface"
        )

    return FacePreprocessor(
        face_detector=detector,
        output_size=output_size,
        normalize_imagenet=normalize_imagenet,
        margin=margin,
    )
