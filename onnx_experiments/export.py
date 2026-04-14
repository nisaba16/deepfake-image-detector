"""
Export a model to FP32 ONNX, PTQ INT8 ONNX, and QAT INT8 ONNX in one shot.

  {model}_fp32.onnx       — FP32 baseline
  {model}_ptq_int8.onnx   — ORT static INT8 quantization of the FP32 ONNX
  {model}_qat_int8.onnx   — ORT static INT8 quantization of the QAT checkpoint

Usage:
    python onnx_experiments/export.py \
        --model      resnet50 \
        --fp32_ckpt  checkpoints/best_resnet50_fp32.pth \
        --qat_ckpt   checkpoints/best_resnet50_qat.pth \
        --data_dir   data/dataset \
        --output_dir onnx_experiments/models
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_experiments.export_to_onnx import load_fp32, load_qat, export as onnx_export

# ViT / DINOv2 use dynamic quantization (no calibration data, per-batch scales).
# All CNN models use static QDQ S8S8.
DYNAMIC_QUANT_MODELS = {"vit_b_16", "dinov2_vitb14"}


def _quantize(input_onnx: str, output_onnx: str, data_dir: str,
              dynamic: bool = False, num_cal_samples: int = 256):
    """Call quantize_onnx.py as a subprocess so its NaN-patch + pre-process logic runs cleanly."""
    cmd = [sys.executable, "onnx_experiments/quantize_onnx.py",
           "--input",  input_onnx,
           "--output", output_onnx]
    if dynamic:
        cmd += ["--dynamic"]
    else:
        cmd += ["--data_dir", data_dir,
                "--num_calibration_samples", str(num_cal_samples),
                "--quant_format", "QDQ",
                "--per_channel"]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Export model to FP32, PTQ INT8, and QAT INT8 ONNX")
    parser.add_argument("--model", required=True,
                        choices=["resnet50", "vit_b_16", "mobilenet_v3_small",
                                 "dinov2_vitb14", "forensic_mobilenet"])
    parser.add_argument("--fp32_ckpt", required=True,
                        help="FP32 checkpoint (.pth)")
    parser.add_argument("--qat_ckpt",  default=None,
                        help="QAT checkpoint (.pth). Skipped if not provided or dinov2.")
    parser.add_argument("--data_dir",  default="data/dataset",
                        help="Dataset root used for PTQ and QAT calibration")
    parser.add_argument("--output_dir", default="onnx_experiments/models")
    parser.add_argument("--features", nargs="+",
                        default=["rgb", "hsv", "fft", "noise", "srm"],
                        help="Feature modalities (forensic_mobilenet only)")
    parser.add_argument("--num_cal_samples", type=int, default=256,
                        help="Calibration images for static quantization")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features = tuple(args.features)

    fp32_onnx = str(out / f"{args.model}_fp32.onnx")
    ptq_onnx  = str(out / f"{args.model}_ptq_int8.onnx")
    qat_onnx  = str(out / f"{args.model}_qat_int8.onnx")
    qat_tmp   = str(out / f"{args.model}_qat_tmp.onnx")

    dynamic = args.model in DYNAMIC_QUANT_MODELS
    skip_qat = args.model == "dinov2_vitb14" or args.qat_ckpt is None

    # ── 1. FP32 export ────────────────────────────────────────────────────────
    print(f"\n[1/3] FP32 export → {fp32_onnx}")
    model = load_fp32(args.model, args.fp32_ckpt, features=features)
    onnx_export(model, fp32_onnx, opset=args.opset)

    # ── 2. PTQ INT8: quantize the FP32 ONNX ──────────────────────────────────
    print(f"\n[2/3] PTQ INT8 → {ptq_onnx}")
    _quantize(fp32_onnx, ptq_onnx, args.data_dir,
              dynamic=dynamic, num_cal_samples=args.num_cal_samples)

    # ── 3. QAT INT8: export QAT checkpoint then quantize ─────────────────────
    if skip_qat:
        print(f"\n[3/3] QAT skipped (dinov2 is FP32-only / no --qat_ckpt provided)")
    else:
        print(f"\n[3/3] QAT INT8 → {qat_onnx}")
        model = load_qat(args.model, args.qat_ckpt, features=features)
        onnx_export(model, qat_tmp, opset=args.opset)
        _quantize(qat_tmp, qat_onnx, args.data_dir,
                  dynamic=dynamic, num_cal_samples=args.num_cal_samples)
        Path(qat_tmp).unlink(missing_ok=True)

    print(f"\n── Exported to {args.output_dir}/ ──")
    for p in [fp32_onnx, ptq_onnx] + ([] if skip_qat else [qat_onnx]):
        mb = os.path.getsize(p) / 1024 ** 2
        print(f"  {Path(p).name:<40} {mb:.2f} MB")


if __name__ == "__main__":
    main()
