from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import os
import random


def collect_image_paths_and_labels(root: str | os.PathLike) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    """Collect image file paths and integer labels from a folder.

    Expects structure similar to torchvision.datasets.ImageFolder:
    root/
      class_a/
        a1.jpg
      class_b/
        b1.jpg
    """
    root = str(root)
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths: List[str] = []
    labels: List[int] = []

    for c in classes:
        class_dir = p / c
        for fp in class_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in image_exts:
                paths.append(str(fp))
                labels.append(class_to_idx[c])

    if not paths:
        raise RuntimeError(f"No images found under: {root}")

    return paths, labels, class_to_idx, idx_to_class


def stratified_split(
    paths: Sequence[str],
    labels: Sequence[int],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """Stratified train/test split. Uses sklearn if available, else minimal fallback."""
    try:
        from sklearn.model_selection import train_test_split  # type: ignore

        X_train, X_test, y_train, y_test = train_test_split(
            list(paths), list(labels), test_size=test_size, random_state=seed, stratify=list(labels)
        )
        return X_train, X_test, y_train, y_test
    except Exception:
        # Fallback: manual stratified split
        rng = random.Random(seed)
        by_class: Dict[int, List[int]] = defaultdict(list)
        for i, y in enumerate(labels):
            by_class[int(y)].append(i)
        train_idx: List[int] = []
        test_idx: List[int] = []
        for cls, idxs in by_class.items():
            idxs = idxs[:]
            rng.shuffle(idxs)
            k = max(1, int(round(len(idxs) * (1 - test_size))))
            train_idx.extend(idxs[:k])
            test_idx.extend(idxs[k:])
        train_idx.sort()
        test_idx.sort()
        X_train = [paths[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test = [paths[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]
        return X_train, X_test, y_train, y_test
