"""
Apply onnxruntime static INT8 quantization (S8S8 QDQ) to an ONNX model.

Intended use: quantize a QAT-trained ONNX model to produce a true INT8 graph
for comparison against an FP32 baseline.

  resnet50_fp32.onnx            ← FP32 baseline, no quantization
  resnet50_qat.onnx  ──► ORT quantize_static (S8S8 QDQ) ──► resnet50_qat_int8.onnx

S8S8 means: signed INT8 for both weights AND activations (symmetric, zero_point=0).
This is achieved via:
  - weight_type=QInt8, activation_type=QInt8
  - extra_options={"ActivationSymmetric": True}  ← required for zero_point=0 on activations

Usage:
    # QAT model → INT8 (default S8S8 QDQ):
    python onnx_experiments/quantize_onnx.py \
        --input  onnx_experiments/models/resnet50_qat.onnx \
        --output onnx_experiments/models/resnet50_qat_int8.onnx \
        --data_dir data/dataset

    # ViT:
    python onnx_experiments/quantize_onnx.py \
        --input  onnx_experiments/models/vit_b_16_qat.onnx \
        --output onnx_experiments/models/vit_b_16_qat_int8.onnx \
        --data_dir data/dataset --per_channel
"""

import argparse
import os
import sys
import tempfile
import warnings

from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_experiments.data_reader import DeepfakeCalibrationReader


def _patch_ort_nan_scale():
    """
    Monkey-patch ORT's compute_scale_zp to survive all-NaN calibration ranges.

    Root cause: the forensic_mobilenet ONNX export replaces the FFT branch with
    constant zeros (aten::fft_rfft2 has no ONNX symbolic).  Those zeros feed into
    the HSV/SRM/noise preprocessing ops which can produce NaN activations (e.g.
    atan2(0,0), 0/0 in saturation).  ORT collects per-tensor min/max statistics
    via np.nanmin / np.nanmax; when every sample is NaN, both return NaN, and then
    ``assert scale >= 0`` fires because ``NaN >= 0 == False``.

    Fix: replace NaN rmin/rmax with a safe unit range [-1, 1] so quantization of
    the *real* backbone ops is unaffected, and the broken preprocessing tensors get
    a harmless placeholder scale (they carry no useful information anyway — they are
    all-zero or all-NaN in the exported graph).
    """
    try:
        import numpy as _np
        import onnxruntime.quantization.quant_utils as _qu

        _orig = _qu.compute_scale_zp

        def _safe_compute_scale_zp(rmin, rmax, qmin, qmax,
                                   symmetric=False, min_real_range=None):
            def _fix(v, default):
                if isinstance(v, _np.ndarray):
                    nan_mask = _np.isnan(v)
                    if nan_mask.any():
                        v = _np.where(nan_mask, _np.float32(default), v)
                elif isinstance(v, float) and _np.isnan(v):
                    v = default
                return v

            rmin = _fix(rmin, -1.0)
            rmax = _fix(rmax,  1.0)
            return _orig(rmin, rmax, qmin, qmax, symmetric, min_real_range)

        _qu.compute_scale_zp = _safe_compute_scale_zp

        # qdq_quantizer may have already imported the symbol — patch that reference too
        try:
            import onnxruntime.quantization.qdq_quantizer as _qdq
            _qdq.compute_scale_zp = _safe_compute_scale_zp
        except Exception:
            pass

        print("  [patch] ORT compute_scale_zp patched to handle NaN calibration ranges.")
    except Exception as exc:
        warnings.warn(f"Could not patch ORT compute_scale_zp: {exc}", RuntimeWarning)


def main():
    # Must run before any ORT quantization import resolves compute_scale_zp
    _patch_ort_nan_scale()

    parser = argparse.ArgumentParser(description="ORT static INT8 quantization")
    parser.add_argument("--input",  required=True, help="Input FP32 ONNX model")
    parser.add_argument("--output", required=True, help="Output INT8 ONNX model")
    parser.add_argument("--data_dir", default="data/dataset",
                        help="Dataset root for calibration")
    parser.add_argument("--num_calibration_samples", type=int, default=256,
                        help="Number of images for calibration")
    parser.add_argument("--quant_format", default="QDQ",
                        choices=["QDQ", "QOperator"],
                        help="QDQ = QuantizeLinear/DequantizeLinear nodes (better HW support); "
                             "QOperator = fused quantized ops")
    parser.add_argument("--per_channel", action="store_true",
                        help="Per-channel weight quantization (more accurate, slower calibration)")
    parser.add_argument("--calibration_method", default="MinMax",
                        choices=["MinMax", "Entropy", "Percentile"])
    parser.add_argument("--calibration_split", default="train",
                        choices=["train", "val"],
                        help="Which dataset split to use for calibration (default: train)")
    parser.add_argument("--no_symmetric_activation", action="store_true",
                        help="Disable symmetric activation quantization (S8S8 → S8U8). "
                             "Default is S8S8: ActivationSymmetric=True, zero_point=0.")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic quantization instead of static. "
                             "Recommended for transformer models (ViT, DINOv2): "
                             "activation scales are computed per-batch at runtime, "
                             "no calibration data needed, usually more accurate for transformers.")
    args = parser.parse_args()

    # Dynamic quantization path — for transformer models (ViT, DINOv2).
    # ORT computes activation scales per-batch at runtime; no calibration data needed.
    # Weights-only INT8, activations stay FP32 during compute (scale applied dynamically).
    if args.dynamic:
        print(f"Dynamic quantization: {args.input} → {args.output}")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        quantize_dynamic(
            model_input=args.input,
            model_output=args.output,
            weight_type=QuantType.QInt8,
        )
        in_mb  = os.path.getsize(args.input)  / 1024 ** 2
        out_mb = os.path.getsize(args.output) / 1024 ** 2
        print(f"Done. (dynamic INT8 weights-only)")
        print(f"  Input : {args.input}  ({in_mb:.2f} MB)")
        print(f"  INT8  : {args.output} ({out_mb:.2f} MB)  "
              f"[{(1 - out_mb / in_mb) * 100:.1f}% smaller]")
        return

    fmt = QuantFormat.QDQ if args.quant_format == "QDQ" else QuantFormat.QOperator
    cal_method = {
        "MinMax":     CalibrationMethod.MinMax,
        "Entropy":    CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }[args.calibration_method]

    symmetric_act = not args.no_symmetric_activation
    quant_scheme = "S8S8" if symmetric_act else "S8U8"

    # QOperator + QInt8 (S8S8) is broken in onnxruntime: ORT itself warns to use QDQ
    # for this combination. Known issues include NaN calibration ranges (scale issue
    # on ViT-style models) and per-channel bias broadcast errors on CNNs.
    if fmt == QuantFormat.QOperator and symmetric_act:
        print("WARNING: QOperator format is incompatible with S8S8 (QInt8+QInt8+symmetric) "
              "— ORT recommendation. Switching to QDQ.")
        fmt = QuantFormat.QDQ

    # QOperator + per_channel + QInt8 is also broken (scale broadcast mismatch on
    # bias quantization). Fall back to per-tensor as an extra safety net.
    per_channel = args.per_channel
    if per_channel and fmt == QuantFormat.QOperator:
        print("WARNING: per_channel=True is incompatible with QOperator + QInt8 "
              "(onnxruntime bug). Disabling per_channel for this run.")
        per_channel = False

    print(f"Calibrating with {args.num_calibration_samples} samples "
          f"({args.calibration_method}, {args.quant_format}, {quant_scheme}, "
          f"per_channel={per_channel}) ...")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ORT recommends running quant_pre_process before static quantization.
    # It adds shape information required for correct calibration (especially ViT
    # attention layers, which otherwise produce all-NaN activation ranges).
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        preprocessed_path = tmp.name
    try:
        print("Pre-processing model for quantization ...")
        quant_pre_process(args.input, preprocessed_path)
        model_to_quantize = preprocessed_path
    except Exception as e:
        # ViT (and other transformer models) fail symbolic shape inference because
        # their Reshape ops have a dynamic batch dimension that SymbolicShapeInference
        # can't resolve.  Retry with skip_symbolic_shape=True: standard ONNX shape
        # inference still runs and correctly annotates static weight shapes, which is
        # all that per-channel weight quantization needs.
        print(f"  quant_pre_process failed ({type(e).__name__}), "
              f"retrying with skip_symbolic_shape=True ...")
        try:
            quant_pre_process(args.input, preprocessed_path, skip_symbolic_shape=True)
            model_to_quantize = preprocessed_path
            print("  → pre-processing succeeded (symbolic shape inference skipped).")
        except Exception as e2:
            # Full fallback: give up on pre-processing and also disable per-channel,
            # because rank-1 LayerNorm weights would trigger "axis out-of-range" errors.
            print(f"WARNING: quant_pre_process failed entirely ({type(e2).__name__}: {e2}), "
                  f"using original model.")
            model_to_quantize = args.input
            if per_channel:
                print("  → per_channel disabled (requires quant_pre_process on this model).")
                per_channel = False

    dr = DeepfakeCalibrationReader(
        data_dir=args.data_dir,
        model_path=model_to_quantize,
        num_samples=args.num_calibration_samples,
        split=args.calibration_split,
    )

    # S8S8: both weights and activations are signed INT8, zero_point=0 (symmetric).
    # ActivationSymmetric=True is required — without it ORT defaults to asymmetric
    # (non-zero zero_point) even when activation_type=QInt8.
    extra_options = {"ActivationSymmetric": symmetric_act}

    def _run_quantize(op_types=None):
        dr.rewind()
        kwargs = dict(
            model_input=model_to_quantize,
            model_output=args.output,
            calibration_data_reader=dr,
            quant_format=fmt,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=cal_method,
            extra_options=extra_options,
        )
        if op_types is not None:
            kwargs["op_types_to_quantize"] = op_types
        # Suppress warnings that are expected consequences of NaN values in the
        # preprocessing subgraph (FFT replaced with zeros → NaN activations):
        #   - calibrate.py: "All-NaN axis encountered" from np.nanmin/nanmax
        #   - quant_utils.py / base_quantizer.py: "invalid value encountered in cast"
        #     when NaN activation/weight values are clipped and cast to int8
        # These are harmless — the affected nodes are the dummy FFT/preprocessing
        # ops that carry no real information in the exported ONNX graph.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered in cast",
                                    category=RuntimeWarning)
            quantize_static(**kwargs)

    try:
        try:
            _run_quantize()
        except AssertionError as e:
            if "scale issue" not in str(e):
                raise
            # Defence-in-depth: if the NaN patch somehow didn't fire (e.g. a
            # different ORT version with a different import path), fall back to
            # quantizing only backbone ops where calibration ranges are well-defined.
            print("WARNING: scale issue despite NaN patch — retrying with "
                  "op_types_to_quantize=['Conv', 'Gemm', 'MatMul'].")
            _run_quantize(op_types=["Conv", "Gemm", "MatMul"])
    finally:
        if model_to_quantize != args.input and os.path.exists(preprocessed_path):
            os.unlink(preprocessed_path)

    in_mb  = os.path.getsize(args.input)  / 1024 ** 2
    out_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"Done. ({quant_scheme} QDQ)")
    print(f"  Input : {args.input}  ({in_mb:.2f} MB)")
    print(f"  INT8  : {args.output} ({out_mb:.2f} MB)  "
          f"[{(1 - out_mb / in_mb) * 100:.1f}% smaller]")


if __name__ == "__main__":
    main()
