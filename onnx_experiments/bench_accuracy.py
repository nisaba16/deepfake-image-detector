"""
Accuracy benchmark for ONNX models on the validation split of the dataset.

Runs on CPU by default (accuracy does not depend on the execution provider).
Pass --gpu to use CUDAExecutionProvider if you want faster throughput.

Usage:
    python onnx_experiments/bench_accuracy.py \
        --models_dir onnx_experiments/models \
        --data_dir   data/dataset \
        --models     resnet50 mobilenet_v3_small \
        --max_samples 1000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import collect_image_paths_and_labels, stratified_split


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_path: str, size: int = 224) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    scale = int(size * 256 / 224)
    w, h = img.size
    img = img.resize((scale, int(h * scale / w)) if w < h else (int(w * scale / h), scale),
                     Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis]       # (1, 3, H, W)


def accuracy(session: ort.InferenceSession,
             val_paths: list, val_labels: list,
             max_samples: int = None) -> dict:
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    paths  = val_paths[:max_samples]  if max_samples else val_paths
    labels = val_labels[:max_samples] if max_samples else val_labels

    correct = errors = 0
    for i, (path, label) in enumerate(zip(paths, labels)):
        if i > 0 and i % 200 == 0:
            print(f"    [{i}/{len(paths)}] acc so far: "
                  f"{correct / (i - errors) * 100:.1f}%", flush=True)
        try:
            logits = session.run([output_name], {input_name: preprocess(path)})[0][0]
            correct += int(np.argmax(logits) == label)
        except Exception:
            errors += 1

    total = len(paths) - errors
    return {"accuracy": round(correct / total * 100, 3) if total else 0.0,
            "samples": total, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Accuracy benchmark on validation split")
    parser.add_argument("--models_dir",  default="onnx_experiments/models")
    parser.add_argument("--data_dir",    required=True, help="Dataset root (fake/ real/)")
    parser.add_argument("--models",      nargs="+", default=None,
                        help="Filter by model family prefix (e.g. resnet50)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap validation images (default: all)")
    parser.add_argument("--gpu",         action="store_true",
                        help="Use CUDAExecutionProvider (faster throughput, same accuracy)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    # Build validation split once
    print(f"Loading dataset from {args.data_dir} ...")
    paths, labels, class_to_idx, _ = collect_image_paths_and_labels(args.data_dir)
    _, val_paths, _, val_labels = stratified_split(paths, labels, test_size=0.2, seed=args.seed)
    cap = args.max_samples or len(val_paths)
    print(f"  Val set: {len(val_paths)} images | using up to {cap} | classes: {class_to_idx}")

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.gpu else ["CPUExecutionProvider"])

    onnx_files = sorted(Path(args.models_dir).glob("*.onnx"))
    if args.models:
        onnx_files = [p for p in onnx_files
                      if any(p.stem.startswith(m) for m in args.models)]
    if not onnx_files:
        print(f"No .onnx files found in {args.models_dir}")
        return

    results = []
    for path in onnx_files:
        print(f"\n{'='*55}")
        print(f"  {path.stem}")
        print(f"{'='*55}")
        try:
            session = ort.InferenceSession(str(path), providers=providers)
            print(f"  provider : {session.get_providers()[0]}")
            metrics = accuracy(session, val_paths, val_labels, args.max_samples)
            print(f"  accuracy : {metrics['accuracy']:.2f}%  "
                  f"({metrics['samples']} samples, {metrics['errors']} errors)")
            results.append({"model": path.stem, **metrics})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"model": path.stem, "error": str(e)})

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Model':<45} {'Accuracy':>10}  {'Samples':>8}")
    print(f"{'-'*45} {'-'*10}  {'-'*8}")
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<43}  ERROR: {r['error']}")
        else:
            print(f"  {r['model']:<43}  {r['accuracy']:>8.2f}%  {r['samples']:>8}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
