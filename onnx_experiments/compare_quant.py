"""
Accuracy comparison: FP32 vs PTQ INT8 (from FP32) vs QAT→INT8 (from QAT pipeline).

For each model:
  fp32        : {models_dir}/{model}_fp32.onnx           (must exist, pre-built)
  ptq_int8    : {models_dir}/{model}_ptq_int8.onnx       (built here via ORT PTQ on fp32)
  qat_int8    : {models_dir}/{model}_int8.onnx           (optional, pre-built from QAT pipeline)

Only measures accuracy — run run_experiments.py for latency benchmarks.

Default quantization scheme: U8S8 (asymmetric activations, signed weights).
U8S8 works better than S8S8 for ReLU / hard-swish activations because the
activation distribution is non-negative — asymmetric quantization uses all
256 INT8 levels efficiently, while symmetric wastes half the range.

Usage:
    python onnx_experiments/compare_quant.py \\
        --models mobilenet_v3_small resnet50 \\
        --models_dir onnx_experiments/models \\
        --data_dir   data/dataset

    # More calibration samples (better accuracy, slower):
    python onnx_experiments/compare_quant.py --num_cal_samples 1024

    # Entropy calibration (good for models with outlier activations like ViT):
    python onnx_experiments/compare_quant.py --calibration_method Entropy
"""

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import (
    CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import collect_image_paths_and_labels, stratified_split
from onnx_experiments.data_reader import DeepfakeCalibrationReader


# ---------------------------------------------------------------------------
# Preprocessing — must match training transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_path: str, size: int = 224) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
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
    return arr.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------
def eval_accuracy(
    onnx_path: str,
    val_paths: List[str],
    val_labels: List[int],
    max_samples: Optional[int] = None,
) -> Dict:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(
        onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    paths  = val_paths[:max_samples]  if max_samples else val_paths
    labels = val_labels[:max_samples] if max_samples else val_labels

    correct = errors = 0
    n = len(paths)
    for i, (path, label) in enumerate(zip(paths, labels)):
        if i > 0 and i % 200 == 0:
            print(f"    [{i}/{n}] acc so far: {correct / (i - errors) * 100:.1f}%",
                  flush=True)
        try:
            logits = session.run([output_name], {input_name: preprocess(path)})[0][0]
            correct += int(np.argmax(logits) == label)
        except Exception:
            errors += 1

    total = len(paths) - errors
    return {
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "samples":  total,
        "errors":   errors,
        "size_mb":  round(os.path.getsize(onnx_path) / 1024 ** 2, 2),
    }


# ---------------------------------------------------------------------------
# PTQ builder
# ---------------------------------------------------------------------------
def _patch_nan_scale():
    """Patch ORT to handle NaN calibration ranges (forensic_mobilenet FFT zeros branch)."""
    try:
        import numpy as _np
        import onnxruntime.quantization.quant_utils as _qu
        _orig = _qu.compute_scale_zp

        def _safe(rmin, rmax, qmin, qmax, symmetric=False, min_real_range=None):
            def fix(v, d):
                if isinstance(v, _np.ndarray):
                    return _np.where(_np.isnan(v), _np.float32(d), v)
                return d if (isinstance(v, float) and _np.isnan(v)) else v
            return _orig(fix(rmin, -1.0), fix(rmax, 1.0), qmin, qmax,
                         symmetric, min_real_range)

        _qu.compute_scale_zp = _safe
        try:
            import onnxruntime.quantization.qdq_quantizer as _qdq
            _qdq.compute_scale_zp = _safe
        except Exception:
            pass
    except Exception as exc:
        warnings.warn(f"Could not patch ORT compute_scale_zp: {exc}")


def build_ptq(
    fp32_path: str,
    output_path: str,
    data_dir: str,
    num_cal_samples: int,
    per_channel: bool,
    symmetric: bool,
    calibration_method: str,
) -> None:
    """Run ORT static PTQ on fp32_path and write INT8 ONNX to output_path."""
    _patch_nan_scale()

    cal_method = {
        "MinMax":     CalibrationMethod.MinMax,
        "Entropy":    CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }[calibration_method]
    act_type   = QuantType.QInt8  if symmetric else QuantType.QUInt8
    scheme_str = "S8S8"           if symmetric else "U8S8"
    print(f"  PTQ: {scheme_str}, per_channel={per_channel}, "
          f"calibration={calibration_method}, cal_samples={num_cal_samples}")

    # Pre-process model (adds shape annotations needed for per-channel quantization)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        prep_path = tmp.name
    model_in = fp32_path
    try:
        quant_pre_process(fp32_path, prep_path)
        model_in = prep_path
        print("  quant_pre_process: OK")
    except Exception as e:
        try:
            quant_pre_process(fp32_path, prep_path, skip_symbolic_shape=True)
            model_in = prep_path
            print("  quant_pre_process: OK (skip_symbolic_shape=True)")
        except Exception as e2:
            print(f"  quant_pre_process failed ({e2}), using raw model (disabling per_channel)")
            per_channel = False

    dr = DeepfakeCalibrationReader(
        data_dir=data_dir, model_path=model_in,
        num_samples=num_cal_samples, split="train",
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN",         category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value",    category=RuntimeWarning)
        quantize_static(
            model_input=model_in,
            model_output=output_path,
            calibration_data_reader=dr,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type=act_type,
            calibrate_method=cal_method,
            extra_options={"ActivationSymmetric": symmetric},
        )

    if model_in != fp32_path and os.path.exists(prep_path):
        os.unlink(prep_path)


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------
def print_table(rows: List[Dict], model_name: str):
    w = [22, 10, 14, 9, 7, 12]
    header = (f"  {'Variant':<{w[0]}}  {'Size (MB)':>{w[1]}}  "
              f"{'Accuracy (%)':>{w[2]}}  {'Samples':>{w[3]}}  {'Errors':>{w[4]}}  "
              f"{'vs FP32':>{w[5]}}")
    sep = "  " + "  ".join("-" * x for x in w)
    width = len(sep)

    print(f"\n{'=' * width}")
    print(f"  {model_name}")
    print(f"{'=' * width}")
    print(header)
    print(sep)
    for r in rows:
        if "error" in r:
            print(f"  {r['variant']:<{w[0]}}  ERROR: {r['error']}")
            continue
        vs = f"{r['acc_drop']:+.2f}pp" if r.get("acc_drop") is not None else "—"
        print(f"  {r['variant']:<{w[0]}}  {r['size_mb']:>{w[1]}.2f}  "
              f"{r['accuracy']:>{w[2]}.2f}  {r['samples']:>{w[3]}}  "
              f"{r['errors']:>{w[4]}}  {vs:>{w[5]}}")
    print(f"{'=' * width}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Accuracy comparison: FP32 vs PTQ INT8 (fp32→) vs QAT→INT8"
    )
    parser.add_argument("--models", nargs="+",
                        default=["resnet50", "vit_b_16",
                                 "mobilenet_v3_small", "forensic_mobilenet"])
    parser.add_argument("--models_dir",      default="onnx_experiments/models")
    parser.add_argument("--data_dir",        default="data/dataset")
    parser.add_argument("--max_val_samples", type=int, default=1000,
                        help="Cap validation images per model (default: 1000)")
    parser.add_argument("--num_cal_samples", type=int, default=512,
                        help="Calibration images for PTQ (more = better, default: 512)")
    parser.add_argument("--per_channel",     action="store_true",  default=True,
                        help="Per-channel weight quantization (default: on)")
    parser.add_argument("--no_per_channel",  dest="per_channel",   action="store_false")
    parser.add_argument("--symmetric",       action="store_true",  default=False,
                        help="S8S8 symmetric activations. Default: U8S8 (asymmetric, "
                             "recommended for ReLU/hard-swish models)")
    parser.add_argument("--calibration_method", default="MinMax",
                        choices=["MinMax", "Entropy", "Percentile"])
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Save results to JSON (optional)")
    args = parser.parse_args()

    # Load validation split once
    print(f"Loading dataset from {args.data_dir} ...")
    paths, labels, class_to_idx, _ = collect_image_paths_and_labels(args.data_dir)
    _, val_paths, _, val_labels = stratified_split(
        paths, labels, test_size=0.2, seed=args.seed
    )
    print(f"  Validation: {len(val_paths)} images | classes: {class_to_idx}")
    print(f"  Evaluating up to {args.max_val_samples} samples per variant\n")

    all_results = {}

    for model in args.models:
        mdir        = Path(args.models_dir)
        fp32_path   = mdir / f"{model}_fp32.onnx"
        ptq_path    = mdir / f"{model}_ptq_int8.onnx"
        qat_int8    = mdir / f"{model}_int8.onnx"

        rows = []
        fp32_acc = None

        # ── FP32 ────────────────────────────────────────────────────────────
        if not fp32_path.exists():
            print(f"[{model}] FP32 ONNX not found: {fp32_path} — skipping\n")
            continue

        print(f"[{model}] Evaluating FP32 ...")
        r = eval_accuracy(str(fp32_path), val_paths, val_labels, args.max_val_samples)
        fp32_acc = r["accuracy"]
        rows.append({"variant": "fp32", "acc_drop": None, **r})
        print(f"  → {r['accuracy']:.2f}%  ({r['samples']} samples)")

        # ── PTQ INT8 (from FP32) ─────────────────────────────────────────────
        print(f"[{model}] Building PTQ INT8 from FP32 ...")
        try:
            build_ptq(
                str(fp32_path), str(ptq_path), args.data_dir,
                args.num_cal_samples, args.per_channel,
                args.symmetric, args.calibration_method,
            )
            print(f"[{model}] Evaluating PTQ INT8 ...")
            r = eval_accuracy(str(ptq_path), val_paths, val_labels, args.max_val_samples)
            r["acc_drop"] = round(r["accuracy"] - fp32_acc, 2)
            rows.append({"variant": "ptq_int8  (fp32→)", **r})
            print(f"  → {r['accuracy']:.2f}%  (drop: {r['acc_drop']:+.2f}pp)")
        except Exception as e:
            print(f"  [ERROR] PTQ failed: {e}")
            rows.append({"variant": "ptq_int8  (fp32→)", "error": str(e)})

        # ── QAT→INT8 (pre-built by QAT pipeline, if present) ─────────────────
        if qat_int8.exists():
            print(f"[{model}] Evaluating QAT→INT8 ...")
            r = eval_accuracy(str(qat_int8), val_paths, val_labels, args.max_val_samples)
            r["acc_drop"] = round(r["accuracy"] - fp32_acc, 2)
            rows.append({"variant": "qat_int8  (qat→)", **r})
            print(f"  → {r['accuracy']:.2f}%  (drop: {r['acc_drop']:+.2f}pp)")
        else:
            print(f"[{model}] QAT INT8 not found ({qat_int8.name}) — skipping")

        print_table(rows, model)
        all_results[model] = rows
        print()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
