"""
CalibrationDataReader for onnxruntime static quantization.
Feeds batches from the dataset's validation split to the quantizer.
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Iterator

import onnxruntime
from onnxruntime.quantization import CalibrationDataReader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import collect_image_paths_and_labels, stratified_split


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_path: str, size: int = 224) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    # Resize shorter side to 256, then center-crop to size (standard ImageNet eval)
    scale_size = int(size * 256 / 224)
    w, h = img.size
    if w < h:
        img = img.resize((scale_size, int(h * scale_size / w)), Image.BILINEAR)
    else:
        img = img.resize((int(w * scale_size / h), scale_size), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)                            # (3, H, W)
    return arr[np.newaxis]                                  # (1, 3, H, W)


class DeepfakeCalibrationReader(CalibrationDataReader):
    """
    Yields individual (1, 3, 224, 224) float32 arrays from the validation split
    for onnxruntime static calibration.
    """

    def __init__(
        self,
        data_dir: str,
        model_path: str,
        num_samples: int = 256,
        seed: int = 42,
    ):
        paths, labels, _, _ = collect_image_paths_and_labels(data_dir)
        _, val_paths, _, _ = stratified_split(paths, labels, test_size=0.2, seed=seed)

        # Subsample uniformly from the validation set
        step = max(1, len(val_paths) // num_samples)
        self._paths = val_paths[::step][:num_samples]

        session = onnxruntime.InferenceSession(model_path)
        self._input_name = session.get_inputs()[0].name
        del session

        self._iter: Iterator = iter(self._paths)

    def get_next(self):
        try:
            path = next(self._iter)
            return {self._input_name: preprocess(path)}
        except StopIteration:
            return None

    def rewind(self):
        self._iter = iter(self._paths)
