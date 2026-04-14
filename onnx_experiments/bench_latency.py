"""
Latency benchmark for ONNX models using dummy inputs.  No dataset required.

Provider routing (when --gpu --trt):
  *_fp32.onnx       → CUDA EP (float precision, TRT skipped)
  *_ptq_int8.onnx   → TensorRT EP with trt_int8_enable=True
  *_qat_int8.onnx   → TensorRT EP with trt_int8_enable=True

Usage:
    python onnx_experiments/bench_latency.py \
        --models_dir onnx_experiments/models \
        --models     resnet50 mobilenet_v3_small \
        --gpu --trt \
        --runs 100 --warmup 20
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _is_quantized(model_path: str) -> bool:
    return "int8" in Path(model_path).stem


def make_session(model_path: str, use_gpu: bool = False, use_trt: bool = False,
                 verbose: bool = False) -> ort.InferenceSession:
    quantized = _is_quantized(model_path)

    if use_trt and use_gpu and quantized:
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": ".trt_cache",
                "trt_fp16_enable": False,
                "trt_int8_enable": True,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        print(f"  provider : TensorRT INT8")
    elif use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_trt and not quantized:
            print(f"  provider : CUDA (TRT skipped — FP32 model stays in float)")
        else:
            print(f"  provider : CUDA")
    else:
        providers = ["CPUExecutionProvider"]
        print(f"  provider : CPU")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    if use_gpu:
        opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    if verbose:
        opts.log_severity_level = 1

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    print(f"  active   : {session.get_providers()[0]}")
    return session


def bench(session: ort.InferenceSession, warmup: int = 20, runs: int = 100) -> dict:
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)

    for _ in range(warmup):
        session.run([], {input_name: dummy})

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run([], {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "median_ms": round(float(np.median(arr)), 3),
        "mean_ms":   round(float(arr.mean()), 3),
        "p95_ms":    round(float(np.percentile(arr, 95)), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark (dummy inputs, no dataset)")
    parser.add_argument("--models_dir", default="onnx_experiments/models")
    parser.add_argument("--models",  nargs="+", default=None,
                        help="Filter by model family prefix (e.g. resnet50 mobilenet_v3_small)")
    parser.add_argument("--gpu",     action="store_true", help="Use CUDAExecutionProvider")
    parser.add_argument("--trt",     action="store_true",
                        help="Use TensorRT for INT8 models (requires --gpu)")
    parser.add_argument("--runs",    type=int, default=100, help="Measurement iterations")
    parser.add_argument("--warmup",  type=int, default=20,  help="Warm-up iterations (discarded)")
    parser.add_argument("--verbose", action="store_true",   help="ORT INFO-level logs")
    args = parser.parse_args()

    onnx_files = sorted(Path(args.models_dir).glob("*.onnx"))
    if args.models:
        onnx_files = [p for p in onnx_files
                      if any(p.stem.startswith(m) for m in args.models)]
    if not onnx_files:
        print(f"No .onnx files found in {args.models_dir}")
        return

    results = []
    for path in onnx_files:
        size_mb = round(os.path.getsize(str(path)) / 1024 ** 2, 2)
        print(f"\n{'='*55}")
        print(f"  {path.stem}  ({size_mb} MB)")
        print(f"{'='*55}")
        try:
            session = make_session(str(path), use_gpu=args.gpu, use_trt=args.trt,
                                   verbose=args.verbose)
            metrics = bench(session, warmup=args.warmup, runs=args.runs)
            print(f"  median={metrics['median_ms']} ms  "
                  f"mean={metrics['mean_ms']} ms  "
                  f"p95={metrics['p95_ms']} ms")
            results.append({"model": path.stem, "size_mb": size_mb, **metrics})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"model": path.stem, "size_mb": size_mb, "error": str(e)})

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"{'Model':<45} {'Size':>8}  {'Median':>9}  {'Mean':>9}  {'P95':>9}")
    print(f"{'-'*45} {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<43}  ERROR: {r['error']}")
        else:
            print(f"  {r['model']:<43}  {r['size_mb']:>6.2f}MB"
                  f"  {r['median_ms']:>7.2f}ms"
                  f"  {r['mean_ms']:>7.2f}ms"
                  f"  {r['p95_ms']:>7.2f}ms")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
