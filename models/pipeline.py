from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict

import numpy as np
from PIL import Image

from .extractors import BaseImageFeatureExtractor
from .classifiers import BaseClassifier


@dataclass
class ImageDetectionPipeline:
    extractor: BaseImageFeatureExtractor
    classifier: BaseClassifier
    skip_failed: bool = False  # Skip images that fail preprocessing

    def transform(self, image_paths: Sequence[str]) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Transform images to features.
        
        Returns:
            (X, metadata) where X is (n_samples, n_features) and metadata is list of dicts
        """
        feats: List[np.ndarray] = []
        metadata: List[Dict] = []
        failed_count = 0
        
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                f = self.extractor.extract(img)
                if f is not None:
                    feats.append(f)
                    metadata.append({"path": p, "status": "success"})
                else:
                    if not self.skip_failed:
                        raise ValueError(f"Extractor returned None for {p}")
                    metadata.append({"path": p, "status": "skipped"})
                    failed_count += 1
            except Exception as e:
                if not self.skip_failed:
                    raise
                metadata.append({"path": p, "status": "failed", "error": str(e)})
                failed_count += 1
        
        if not feats:
            raise RuntimeError("No images processed successfully")
        
        # Align dims
        max_len = max(f.shape[0] for f in feats)
        feats = [np.pad(f, (0, max_len - f.shape[0])) for f in feats]
        X = np.stack(feats, axis=0)
        
        if failed_count > 0:
            print(f"Warning: {failed_count} images failed/skipped during transform")
        
        return X, metadata

    def fit(self, image_paths: Sequence[str], labels: Sequence[int]):
        X, _ = self.transform(image_paths)
        y = np.asarray(labels)
        self.classifier.fit(X, y)
        return self

    def predict(self, image_paths: Sequence[str]) -> np.ndarray:
        X, _ = self.transform(image_paths)
        return self.classifier.predict(X)

    def predict_proba(self, image_paths: Sequence[str]):
        X, _ = self.transform(image_paths)
        return self.classifier.predict_proba(X)

    def score(self, image_paths: Sequence[str], labels: Sequence[int]) -> float:
        y_true = np.asarray(labels)
        y_pred = self.predict(image_paths)
        return float((y_true == y_pred).mean())
