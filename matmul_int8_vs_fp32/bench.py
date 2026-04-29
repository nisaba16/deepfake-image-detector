import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


# ONNx graph builders 
# Each graph has exactly one compute node — everything else is stripped away
# so we measure the kernel itself, not model loading or pre/post-processing.

def make_fp32_matmul_model(M: int, K: int, N: int) -> bytes:
    """Single MatMul node: Y = A @ B, all float32."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [K, N])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"])
    graph = helper.make_graph([node], "fp32_matmul", [A, B], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def make_int8_matmul_model(M: int, K: int, N: int) -> bytes:
    """Single QLinearMatMul node: y = quantize(dequantize(a) @ dequantize(b)).

    QLinearMatMul is the INT8 equivalent of MatMul in ONNX. It takes int8
    inputs and produces an int8 output; scale and zero-point tensors (stored
    as graph initializers) tell ORT how to interpret the integer values.

    Without VNNI ORT uses a slower scalar fallback.
    """
    a = helper.make_tensor_value_info("a", TensorProto.INT8, [M, K])
    b = helper.make_tensor_value_info("b", TensorProto.INT8, [K, N])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [M, N])

    def sf32(name, val):
        return numpy_helper.from_array(np.array(val, dtype=np.float32), name=name)

    def si8(name, val):
        return numpy_helper.from_array(np.array(val, dtype=np.int8), name=name)

    # Scale = 0.02, zero_point = 0 (symmetric quantization).
    initializers = [
        sf32("a_scale", 0.02), si8("a_zp", 0),
        sf32("b_scale", 0.02), si8("b_zp", 0),
        sf32("y_scale", 0.02), si8("y_zp", 0),
    ]
    node = helper.make_node(
        "QLinearMatMul",
        inputs=["a", "a_scale", "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp"],
        outputs=["y"],
    )
    graph = helper.make_graph([node], "int8_matmul", [a, b], [y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def make_session(model_bytes: bytes, threads: int,
                 profile_prefix: str | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Single-threaded by default so FP32 and INT8 compete on equal footing.
    # Pass --threads N to see how parallelism affects each path.
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    if profile_prefix:
        opts.enable_profiling = True
        opts.profile_file_prefix = profile_prefix
    return ort.InferenceSession(
        model_bytes, sess_options=opts, providers=["CPUExecutionProvider"]
    )


def bench(session: ort.InferenceSession, feeds: dict,
          warmup: int, runs: int) -> dict:
    # Warmup: let ORT finish any lazy compilation and let the CPU
    # ramp up from idle frequency before we start recording times.
    for _ in range(warmup):
        session.run(None, feeds)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1_000)  # ms

    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms":   float(arr.mean()),
        "p95_ms":    float(np.percentile(arr, 95)),
    }


def parse_profile(json_path: str) -> list[dict]:
    with open(json_path) as f:
        events = json.load(f)
    nodes = [e for e in events if e.get("cat") == "Node"]
    totals: dict[str, dict] = {}
    for e in nodes:
        op = e.get("args", {}).get("op_name") or e.get("name", "?")
        provider = e.get("args", {}).get("provider", "?")
        dur = e.get("dur", 0)
        if op not in totals:
            totals[op] = {"op_name": op, "provider": provider, "total_us": 0, "calls": 0}
        totals[op]["total_us"] += dur
        totals[op]["calls"] += 1
    return sorted(totals.values(), key=lambda x: x["total_us"], reverse=True)


def run_profile(label: str, model_bytes: bytes, feeds: dict,
                threads: int, runs: int = 20) -> list[dict]:
    prefix = f"matmul_int8_vs_fp32/profile_{label}"
    sess = make_session(model_bytes, threads, profile_prefix=prefix)
    for _ in range(runs):
        sess.run(None, feeds)
    json_path = sess.end_profiling()
    return parse_profile(json_path), json_path


def print_profile_comparison(fp32_ops: list[dict], int8_ops: list[dict]) -> None:
    print(f"\n{'Op':<30} {'Provider':<30} {'Total (µs)':>12}  {'Calls':>6}  {'Avg (µs)':>10}")
    sep = "─" * 95

    for label, ops in [("── FP32 graph ──", fp32_ops), ("── INT8 graph ──", int8_ops)]:
        print(f"\n{label}")
        print(sep)
        for o in ops:
            avg = o["total_us"] / max(o["calls"], 1)
            print(f"  {o['op_name']:<28} {o['provider']:<30} "
                  f"{o['total_us']:>10}µs  {o['calls']:>6}  {avg:>8.1f}µs")
        print(sep)


def main():
    parser = argparse.ArgumentParser(description="INT8 vs FP32 MatMul latency on CPU")
    parser.add_argument("--sizes",   nargs="+", type=int,
                        default=[64, 128, 256, 512, 1024, 2048],
                        help="Square matrix sizes N (runs N×N @ N×N)")
    parser.add_argument("--warmup",  type=int, default=30)
    parser.add_argument("--runs",    type=int, default=200)
    parser.add_argument("--threads", type=int, default=1,
                        help="intra_op_num_threads for ORT (default 1)")
    parser.add_argument("--profile", action="store_true",
                        help="Run ORT profiling and show per-op breakdown")
    parser.add_argument("--profile-size", type=int, default=512,
                        help="Matrix size to use for profiling (default 512)")
    parser.add_argument("--profile-runs", type=int, default=50,
                        help="Number of runs to collect in the profile (default 50)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    # -------- Latency benchmark 
    header = (f"{'Size':>6}  {'FP32 median':>12}  {'INT8 median':>12}  "
              f"{'FP32 p95':>10}  {'INT8 p95':>10}  {'ratio (FP32/INT8)':>18}")
    sep = "─" * len(header)

    print(f"\nCPU MatMul benchmark — threads={args.threads}, "
          f"warmup={args.warmup}, runs={args.runs}\n")
    print(header)
    print(sep)

    for N in args.sizes:
        fp32_model = make_fp32_matmul_model(N, N, N)
        int8_model = make_int8_matmul_model(N, N, N)

        fp32_sess = make_session(fp32_model, args.threads)
        int8_sess = make_session(int8_model, args.threads)

        fp32_feeds = {
            "A": rng.standard_normal((N, N)).astype(np.float32),
            "B": rng.standard_normal((N, N)).astype(np.float32),
        }
        int8_feeds = {
            "a": rng.integers(-127, 127, (N, N), dtype=np.int8),
            "b": rng.integers(-127, 127, (N, N), dtype=np.int8),
        }

        fp32 = bench(fp32_sess, fp32_feeds, args.warmup, args.runs)
        int8 = bench(int8_sess, int8_feeds, args.warmup, args.runs)

        ratio = fp32["median_ms"] / int8["median_ms"]
        faster = "INT8 faster" if ratio > 1.0 else "FP32 faster"

        print(
            f"{N:>6}  "
            f"{fp32['median_ms']:>10.4f}ms  "
            f"{int8['median_ms']:>10.4f}ms  "
            f"{fp32['p95_ms']:>8.4f}ms  "
            f"{int8['p95_ms']:>8.4f}ms  "
            f"{ratio:>8.2f}x  ({faster})"
        )

    print(sep)
    print("\nratio > 1  → INT8 is faster than FP32")
    print("ratio < 1  → INT8 is SLOWER than FP32  (common on x86 without VNNI)\n")

    # ------  Per-op profiling 
    if args.profile:
        N = args.profile_size
        print(f"\n{'='*70}")
        print(f"  ORT per-op profiling  —  {N}×{N} matrix, {args.profile_runs} runs")
        print(f"{'='*70}")

        fp32_model = make_fp32_matmul_model(N, N, N)
        int8_model = make_int8_matmul_model(N, N, N)

        fp32_feeds = {
            "A": rng.standard_normal((N, N)).astype(np.float32),
            "B": rng.standard_normal((N, N)).astype(np.float32),
        }
        int8_feeds = {
            "a": rng.integers(-127, 127, (N, N), dtype=np.int8),
            "b": rng.integers(-127, 127, (N, N), dtype=np.int8),
        }

        fp32_ops, fp32_json = run_profile("fp32", fp32_model, fp32_feeds,
                                          args.threads, args.profile_runs)
        int8_ops, int8_json = run_profile("int8", int8_model, int8_feeds,
                                          args.threads, args.profile_runs)

        print_profile_comparison(fp32_ops, int8_ops)
        print(f"\nRaw profiles written to:")
        print(f"  FP32 : {fp32_json}")
        print(f"  INT8 : {int8_json}\n")

        int8_op_names = {o["op_name"] for o in int8_ops}
        if "DequantizeLinear" in int8_op_names or "MatMul" in int8_op_names:
            print("  *** ORT decomposed QLinearMatMul into DequantizeLinear + MatMul + QuantizeLinear")
            print("      → running float matmul internally with QDQ overhead on top ***\n")
        elif "QLinearMatMul" in int8_op_names:
            print("  *** ORT kept QLinearMatMul as a single fused op ***\n")


if __name__ == "__main__":
    main()
