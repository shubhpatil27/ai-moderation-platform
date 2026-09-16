import csv
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from src.model.predictor import ModerationModel
from src.model.onnx_predictor import ONNXModerationModel
from src.model.onnx_int8_predictor import INT8ModerationModel


TEXT = (
    "This is a sample comment for "
    "moderation performance testing."
)

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

OUTPUT_FILE = Path(
    "benchmarks/raw/model_runtime.csv"
)


def percentile(values, fraction):

    values = sorted(values)

    index = int(
        fraction * (len(values) - 1)
    )

    return values[index]


def benchmark(name, model):

    print(f"\nBenchmarking {name}...")

    for _ in range(WARMUP_RUNS):
        model.predict(TEXT)

    latencies = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        model.predict(TEXT)

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    result = {
        "runtime": name,
        "average_ms": statistics.mean(latencies),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
    }

    print(f"Average : {result['average_ms']:.2f} ms")
    print(f"p50     : {result['p50_ms']:.2f} ms")
    print(f"p95     : {result['p95_ms']:.2f} ms")
    print(f"p99     : {result['p99_ms']:.2f} ms")

    return result


def save_results(results):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    file_exists = OUTPUT_FILE.exists()

    with OUTPUT_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "timestamp_utc",
            "runtime",
            "warmup_runs",
            "benchmark_runs",
            "average_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        for result in results:

            writer.writerow({
                "timestamp_utc": timestamp,
                "runtime": result["runtime"],
                "warmup_runs": WARMUP_RUNS,
                "benchmark_runs": BENCHMARK_RUNS,
                "average_ms": round(
                    result["average_ms"], 4
                ),
                "p50_ms": round(
                    result["p50_ms"], 4
                ),
                "p95_ms": round(
                    result["p95_ms"], 4
                ),
                "p99_ms": round(
                    result["p99_ms"], 4
                ),
            })


def main():

    print("Loading PyTorch FP32...")
    pytorch_model = ModerationModel()

    print("Loading ONNX FP32...")
    onnx_model = ONNXModerationModel()

    print("Loading ONNX INT8...")
    int8_model = INT8ModerationModel()


    pytorch = benchmark(
        "pytorch_fp32",
        pytorch_model,
    )

    onnx = benchmark(
        "onnx_fp32",
        onnx_model,
    )

    int8 = benchmark(
        "onnx_int8",
        int8_model,
    )


    save_results([
        pytorch,
        onnx,
        int8,
    ])


    print("\n" + "=" * 55)
    print("RUNTIME COMPARISON")
    print("=" * 55)

    print(
        f"PyTorch FP32 : "
        f"{pytorch['average_ms']:.2f} ms"
    )

    print(
        f"ONNX FP32    : "
        f"{onnx['average_ms']:.2f} ms"
    )

    print(
        f"ONNX INT8    : "
        f"{int8['average_ms']:.2f} ms"
    )


    onnx_speedup = (
        pytorch["average_ms"]
        / onnx["average_ms"]
    )

    int8_speedup_vs_pytorch = (
        pytorch["average_ms"]
        / int8["average_ms"]
    )

    int8_speedup_vs_onnx = (
        onnx["average_ms"]
        / int8["average_ms"]
    )


    print(
        f"\nONNX vs PyTorch : "
        f"{onnx_speedup:.2f}x"
    )

    print(
        f"INT8 vs PyTorch : "
        f"{int8_speedup_vs_pytorch:.2f}x"
    )

    print(
        f"INT8 vs ONNX    : "
        f"{int8_speedup_vs_onnx:.2f}x"
    )

    print(
        f"\nResults saved to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()