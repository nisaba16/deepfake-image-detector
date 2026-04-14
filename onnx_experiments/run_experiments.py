"""
Measure and compare: accuracy, model size, and latency for all ONNX models.

Experiment matrix per model:
  - FP32 ONNX        (exported from best_*_fp32.pth)
  - INT8 ONNX        (ORT static quantization of FP32 ONNX)
  - QAT ONNX         (exported from best_*_qat.pth, float graph with simulated quant)

Metrics:
  - Accuracy   (Top-1 on validation split)
  - Model size (MB)
  - Latency    (ms/image, median over N warm runs, batch_size=1)

Usage:
    python onnx_experiments/run_experiments.py \
        --models_dir onnx_experiments/models \
        --data_dir   data/dataset \
        --output     onnx_experiments/results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import collect_image_paths_and_labels, stratified_split


# ---------------------------------------------------------------------------
# Preprocessing (matches training transforms)
# ---------------------------------------------------------------------------
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
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)           # (3, H, W)
    return arr[np.newaxis]                 # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def make_session(model_path: str, use_gpu: bool = False,
                 use_trt: bool = False, verbose: bool = False) -> ort.InferenceSession:
    if use_trt and use_gpu:
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": ".trt_cache",
                "trt_fp16_enable": False,   # keep FP32 accuracy for benchmarking
                "trt_int8_enable": False,    # INT8 handled by QDQ nodes in the ONNX graph
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    elif use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    if verbose:
        # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
        # Level 1 shows per-op kernel dispatch, fallback decisions, and provider selection
        opts.log_severity_level = 1
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

    active = session.get_providers()[0]
    print(f"  [ORT] Active provider: {active}")
    print(f"  [ORT] All providers (priority order): {session.get_providers()}")
    if use_gpu and "CPU" in active:
        print(f"  [ORT] WARNING: requested GPU but got CPU — "
              f"check onnxruntime-gpu install and CUDA availability")

    if verbose:
        # Show which nodes got assigned to which provider
        print(f"  [ORT] Node placement summary:")
        for node_meta in session.get_providers():
            print(f"    provider active: {node_meta}")

    return session


def run_accuracy(
    session: ort.InferenceSession,
    val_paths: List[str],
    val_labels: List[int],
    max_samples: Optional[int] = None,
) -> Dict:
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    paths  = val_paths[:max_samples]  if max_samples else val_paths
    labels = val_labels[:max_samples] if max_samples else val_labels

    correct = 0
    errors  = 0
    n_total = len(paths)
    for i, (path, label) in enumerate(zip(paths, labels)):
        if i > 0 and i % 100 == 0:
            print(f"    [{i}/{n_total}] acc so far: "
                  f"{correct/(i - errors)*100:.1f}%", flush=True)
        try:
            inp  = preprocess(path)
            logits = session.run([output_name], {input_name: inp})[0][0]  # (num_classes,)
            pred = int(np.argmax(logits))
            correct += int(pred == label)
        except Exception as e:
            errors += 1

    total = len(paths) - errors
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": round(acc * 100, 3), "samples": total, "errors": errors}


def run_latency(
    session: ort.InferenceSession,
    warmup: int = 20,
    runs: int = 100,
    input_size: tuple = (1, 3, 224, 224),
) -> Dict:
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(*input_size).astype(np.float32)

    # Warm-up
    for _ in range(warmup):
        session.run([], {input_name: dummy})

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run([], {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "latency_mean_ms":   round(float(arr.mean()),  3),
        "latency_median_ms": round(float(np.median(arr)), 3),
        "latency_p95_ms":    round(float(np.percentile(arr, 95)), 3),
        "latency_min_ms":    round(float(arr.min()),   3),
        "latency_max_ms":    round(float(arr.max()),   3),
    }


def model_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / 1024 ** 2, 3)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_all(
    models_dir: str,
    data_dir: Optional[str],
    max_val_samples: Optional[int],
    warmup: int,
    latency_runs: int,
    use_gpu: bool,
    use_trt: bool,
    seed: int,
    model_families: Optional[List[str]] = None,
    variants: Optional[List[str]] = None,
    latency_only: bool = False,
    verbose: bool = False,
) -> List[Dict]:
    # Build validation split once (skipped in latency_only mode)
    if latency_only:
        val_paths, val_labels = [], []
        print("Latency-only mode: skipping dataset load and accuracy evaluation.")
    else:
        print(f"Loading dataset from {data_dir} ...")
        paths, labels, class_to_idx, idx_to_class = collect_image_paths_and_labels(data_dir)
        _, val_paths, _, val_labels = stratified_split(paths, labels, test_size=0.2, seed=seed)
        print(f"  Validation set: {len(val_paths)} images | classes: {class_to_idx}")

    all_onnx = sorted(Path(models_dir).glob("*.onnx"))

    onnx_files = all_onnx
    if model_families:
        # Keep only files whose stem starts with one of the requested family names.
        # e.g. family "resnet50" matches resnet50_fp32.onnx, resnet50_int8.onnx
        onnx_files = [p for p in onnx_files
                      if any(p.stem.startswith(f) for f in model_families)]
        print(f"  Filtering to families: {model_families} "
              f"({len(onnx_files)}/{len(all_onnx)} files)")
    if variants:
        # Keep only files whose stem ends with one of the requested variant suffixes.
        # e.g. variants ["fp32", "int8"] excludes *_qat.onnx
        onnx_files = [p for p in onnx_files
                      if any(p.stem.endswith(f"_{v}") for v in variants)]
        print(f"  Filtering to variants: {variants} ({len(onnx_files)} files)")

    if not onnx_files:
        print(f"No .onnx files found in {models_dir}")
        return []

    results = []
    for onnx_path in onnx_files:
        model_path = str(onnx_path)
        name = onnx_path.stem          # e.g. "resnet50_fp32"
        size = model_size_mb(model_path)

        print(f"\n{'='*60}")
        print(f"  Model : {name}")
        print(f"  File  : {model_path}")
        print(f"  Size  : {size} MB")
        print(f"{'='*60}")

        try:
            session = make_session(model_path, use_gpu=use_gpu, use_trt=use_trt, verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Failed to load session: {e}")
            results.append({"model": name, "size_mb": size, "error": str(e)})
            continue

        if not latency_only:
            print(f"  Running accuracy ({max_val_samples or len(val_paths)} samples) ...")
            acc_metrics = run_accuracy(session, val_paths, val_labels, max_val_samples)
            print(f"  → Accuracy: {acc_metrics['accuracy']:.2f}%  "
                  f"({acc_metrics['samples']} samples)")
        else:
            acc_metrics = {}

        print(f"  Running latency ({warmup} warm-up + {latency_runs} runs) ...")
        lat_metrics = run_latency(session, warmup=warmup, runs=latency_runs)
        print(f"  → Median: {lat_metrics['latency_median_ms']:.2f} ms  "
              f"| Mean: {lat_metrics['latency_mean_ms']:.2f} ms  "
              f"| P95: {lat_metrics['latency_p95_ms']:.2f} ms")

        entry = {"model": name, "size_mb": size}
        entry.update(acc_metrics)
        entry.update(lat_metrics)
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Pretty table
# ---------------------------------------------------------------------------
def print_table(results: List[Dict]):
    cols = [
        ("Model",           "model",                "45s"),
        ("Size (MB)",       "size_mb",              "10.2f"),
        ("Accuracy (%)",    "accuracy",             "13.2f"),
        ("Median (ms)",     "latency_median_ms",    "13.2f"),
        ("Mean (ms)",       "latency_mean_ms",      "11.2f"),
        ("P95 (ms)",        "latency_p95_ms",       "10.2f"),
    ]

    def _col_width(fmt: str) -> int:
        # Format spec is either "Ns" or "N.Mf" — extract the leading integer width.
        return int(fmt.split(".")[0].rstrip("sfgdeE"))

    header = "  ".join(f"{h:<{_col_width(fmt)}}" for h, _, fmt in cols)
    sep    = "  ".join("-" * _col_width(f) for _, _, f in cols)
    print("\n" + "=" * len(sep))
    print("EXPERIMENT RESULTS")
    print("=" * len(sep))
    print(header)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<43}  ERROR: {r['error']}")
            continue
        row = "  ".join(
            (f"{r.get(k, float('nan')):{f}}" if k != "model" else f"{r[k]:<{f[:-1]}}")
            for _, k, f in cols
        )
        print(row)
    print("=" * len(sep))

    # Speedup vs FP32 baseline per model family
    fp32 = {r["model"].replace("_fp32", ""): r for r in results
            if r["model"].endswith("_fp32") and "latency_median_ms" in r}
    print("\nSpeedup over FP32 baseline (latency):")
    for r in results:
        if "latency_median_ms" not in r or r["model"].endswith("_fp32"):
            continue
        family = r["model"].rsplit("_", 1)[0]
        if family in fp32:
            speedup = fp32[family]["latency_median_ms"] / r["latency_median_ms"]
            size_ratio = 1 - r["size_mb"] / fp32[family]["size_mb"]
            acc_drop   = fp32[family].get("accuracy", 0) - r.get("accuracy", 0)
            print(f"  {r['model']:<45} speedup={speedup:.2f}x  "
                  f"size_reduction={size_ratio*100:.1f}%  "
                  f"acc_drop={acc_drop:.2f}pp")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ONNX model experiments: accuracy, size, latency")
    parser.add_argument("--models_dir", default="onnx_experiments/models",
                        help="Directory containing .onnx files")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--latency_only", action="store_true",
                        help="Skip accuracy evaluation; no --data_dir needed")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable ORT verbose logging (log_severity_level=1): "
                             "shows per-op kernel dispatch, provider selection, "
                             "and CPU fallback decisions")
    parser.add_argument("--output", default="onnx_experiments/results.json",
                        help="Save results to JSON")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Cap validation samples (default: all)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Latency warm-up iterations")
    parser.add_argument("--latency_runs", type=int, default=100,
                        help="Latency measurement iterations")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDAExecutionProvider for latency")
    parser.add_argument("--trt", action="store_true",
                        help="Use TensorrtExecutionProvider (requires --gpu and TensorRT install)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict evaluation to these model families "
                             "(e.g. --models resnet50 vit_b_16). "
                             "Default: evaluate all .onnx files in models_dir.")
    parser.add_argument("--variants", nargs="+", default=["fp32", "int8"],
                        help="Which variants to compare by stem suffix "
                             "(e.g. --variants fp32 int8 qat). "
                             "Default: fp32 int8 (excludes qat float-sim graphs).")
    args = parser.parse_args()

    if not args.latency_only and args.data_dir is None:
        parser.error("--data_dir is required unless --latency_only is set")

    results = run_all(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        max_val_samples=args.max_val_samples,
        warmup=args.warmup,
        latency_runs=args.latency_runs,
        use_gpu=args.gpu,
        use_trt=args.trt,
        seed=args.seed,
        model_families=args.models,
        variants=args.variants,
        latency_only=args.latency_only,
        verbose=args.verbose,
    )

    if results:
        print_table(results)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
