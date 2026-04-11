"""
Dataset statistics for training and OOD evaluation splits.

Covers:
  - data/dataset/          (training data)
  - data/ddata/train/      (ddata train)
  - data/ddata/test/       (out-of-domain test)

Per split reports:
  - Total images and per-class counts
  - Class balance ratio and imbalance warning
  - Resolution distribution (min / max / mean / p5 / p95) — sampled for speed
  - File format breakdown

Usage:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --data_root data --sample 2000 --plot
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def collect_split(root: Path) -> Dict[str, List[Path]]:
    """Return {class_name: [image_path, ...]} for an ImageFolder-style root."""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    result: Dict[str, List[Path]] = {}
    for cls in classes:
        files = [
            p for p in (root / cls).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        result[cls] = sorted(files)
    return result


def resolution_stats(paths: List[Path], sample: int) -> Dict:
    """Sample up to `sample` images and return resolution statistics."""
    rng = np.random.default_rng(42)
    subset = paths if len(paths) <= sample else [
        paths[i] for i in rng.choice(len(paths), sample, replace=False)
    ]
    widths, heights = [], []
    errors = 0
    for p in tqdm(subset, desc="    reading sizes", leave=False, ncols=80):
        try:
            w, h = Image.open(p).size
            widths.append(w)
            heights.append(h)
        except Exception:
            errors += 1

    if not widths:
        return {}

    def _stats(arr):
        a = np.array(arr)
        return {
            "min": int(a.min()),
            "max": int(a.max()),
            "mean": float(a.mean()),
            "p5":  int(np.percentile(a, 5)),
            "p95": int(np.percentile(a, 95)),
        }

    return {
        "sampled": len(widths),
        "errors": errors,
        "width":  _stats(widths),
        "height": _stats(heights),
        "square_pct": round(100 * sum(w == h for w, h in zip(widths, heights)) / len(widths), 1),
    }


def format_counts(paths: List[Path]) -> Dict[str, int]:
    return dict(Counter(p.suffix.lower() for p in paths))


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

W = 70

def section(title: str):
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def subsection(title: str):
    print(f"\n  {'─' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'─' * (W - 4)}")


def print_split_stats(split_name: str, root: Path, sample: int, plot: bool):
    section(f"{split_name}   [{root}]")

    if not root.exists():
        print(f"  [NOT FOUND] {root}")
        return

    by_class = collect_split(root)
    if not by_class:
        print("  No class subdirectories found.")
        return

    all_paths: List[Path] = []
    for paths in by_class.values():
        all_paths.extend(paths)

    total = len(all_paths)

    # --- Counts ---
    print(f"\n  Total images : {total:,}")
    print(f"  Classes      : {list(by_class.keys())}")
    print()

    counts = {cls: len(p) for cls, p in by_class.items()}
    for cls, n in counts.items():
        bar_len = int(40 * n / total) if total else 0
        bar = "█" * bar_len
        pct = 100 * n / total if total else 0
        print(f"    {cls:<12} {n:>8,}  ({pct:5.1f}%)  {bar}")

    # Balance check
    if len(counts) == 2:
        vals = list(counts.values())
        ratio = max(vals) / min(vals) if min(vals) > 0 else float("inf")
        majority = max(counts, key=counts.get)
        minority = min(counts, key=counts.get)
        balance_label = "balanced" if ratio < 1.1 else (
            "slightly imbalanced" if ratio < 1.5 else "IMBALANCED"
        )
        print(f"\n  Balance      : {ratio:.2f}x  ({majority} / {minority})  → {balance_label}")

    # --- File formats ---
    subsection("File formats")
    fmts = format_counts(all_paths)
    for ext, n in sorted(fmts.items(), key=lambda x: -x[1]):
        print(f"    {ext:<10} {n:>8,}")

    # --- Resolution ---
    subsection(f"Resolution  (sample={min(sample, total):,} / {total:,})")
    res = resolution_stats(all_paths, sample)
    if res:
        print(f"    {'':12} {'width':>8}  {'height':>8}")
        print(f"    {'min':12} {res['width']['min']:>8}  {res['height']['min']:>8}")
        print(f"    {'max':12} {res['width']['max']:>8}  {res['height']['max']:>8}")
        print(f"    {'mean':12} {res['width']['mean']:>8.1f}  {res['height']['mean']:>8.1f}")
        print(f"    {'p5':12} {res['width']['p5']:>8}  {res['height']['p5']:>8}")
        print(f"    {'p95':12} {res['width']['p95']:>8}  {res['height']['p95']:>8}")
        print(f"    Square images : {res['square_pct']}%")
        if res["errors"]:
            print(f"    [WARN] Could not open {res['errors']} images in sample")

    # --- Optional plot ---
    if plot and res:
        _plot_resolution(split_name, all_paths, sample)


def _plot_resolution(split_name: str, paths: List[Path], sample: int):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [INFO] matplotlib not available, skipping plot")
        return

    rng = np.random.default_rng(42)
    subset = paths if len(paths) <= sample else [
        paths[i] for i in rng.choice(len(paths), sample, replace=False)
    ]
    sizes = []
    for p in subset:
        try:
            sizes.append(Image.open(p).size)
        except Exception:
            pass

    if not sizes:
        return

    widths  = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{split_name} — resolution distribution (n={len(sizes):,})")

    axes[0].hist(widths,  bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Width")
    axes[0].set_xlabel("pixels")
    axes[0].set_ylabel("count")

    axes[1].hist(heights, bins=40, color="coral", edgecolor="white")
    axes[1].set_title("Height")
    axes[1].set_xlabel("pixels")

    plt.tight_layout()
    safe = split_name.lower().replace(" ", "_").replace("/", "_")
    out = f"results/resolution_{safe}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Plot saved → {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(splits: List[Tuple[str, Path]]):
    section("SUMMARY")
    header = f"  {'Split':<28} {'Total':>8}  {'fake':>8}  {'real':>8}  {'Balance':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for name, root in splits:
        if not root.exists():
            print(f"  {name:<28} {'NOT FOUND':>8}")
            continue
        by_class = collect_split(root)
        total = sum(len(p) for p in by_class.values())
        fake  = len(by_class.get("fake",  []))
        real  = len(by_class.get("real",  []))
        if fake and real:
            ratio = max(fake, real) / min(fake, real)
            bal = f"{ratio:.2f}x"
        else:
            bal = "n/a"
        print(f"  {name:<28} {total:>8,}  {fake:>8,}  {real:>8,}  {bal:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("--data_root", default="data",
                        help="Root folder containing dataset/ and ddata/")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Max images to sample for resolution stats (default: 1000)")
    parser.add_argument("--plot", action="store_true",
                        help="Save resolution histogram plots to results/")
    args = parser.parse_args()

    root = Path(args.data_root)

    splits = [
        ("Training  (data/dataset)",      root / "dataset"),
        ("ddata train (data/ddata/train)", root / "ddata" / "train"),
        ("OOD test  (data/ddata/test)",    root / "ddata" / "test"),
    ]

    for name, path in splits:
        print_split_stats(name, path, sample=args.sample, plot=args.plot)

    print_summary(splits)
    print()


if __name__ == "__main__":
    main()
