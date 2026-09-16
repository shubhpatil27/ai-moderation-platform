import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CONFIG_FILE = Path(
    "config/promotion_gate.json"
)

RESULTS_DIR = Path(
    "evaluation/results"
)

REPORT_FILE = Path(
    "evaluation/results/promotion_report.json"
)


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


MODEL_FILES = {
    "onnx_fp32":
        Path("models/onnx/model.onnx"),

    "onnx_int8":
        Path(
            "models/onnx_int8/"
            "model_quantized.onnx"
        ),
}


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def model_size_mb(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    return (
        path.stat().st_size
        / (1024 ** 2)
    )


def pass_fail(value):

    return "PASS" if value else "FAIL"


# ============================================================
# MAIN PROMOTION LOGIC
# ============================================================

def main():

    config = load_json(
        CONFIG_FILE
    )


    baseline_name = config[
        "baseline_runtime"
    ]

    candidate_name = config[
        "candidate_runtime"
    ]


    quality_gates = config[
        "quality_gates"
    ]

    performance_gates = config[
        "performance_gates"
    ]


    # ========================================================
    # LOAD EVALUATION RESULTS
    # ========================================================

    baseline_metrics = load_json(
        RESULTS_DIR
        / f"{baseline_name}_metrics.json"
    )

    candidate_metrics = load_json(
        RESULTS_DIR
        / f"{candidate_name}_metrics.json"
    )


    # ========================================================
    # QUALITY GATE 1
    # MACRO F1
    # ========================================================

    baseline_macro_f1 = (
        baseline_metrics["macro_f1"]
    )

    candidate_macro_f1 = (
        candidate_metrics["macro_f1"]
    )

    macro_f1_drop = (
        baseline_macro_f1
        - candidate_macro_f1
    )

    macro_f1_pass = (
        macro_f1_drop
        <= quality_gates[
            "max_macro_f1_drop"
        ]
    )


    # ========================================================
    # QUALITY GATE 2
    # MACRO RECALL
    # ========================================================

    baseline_macro_recall = (
        baseline_metrics["macro_recall"]
    )

    candidate_macro_recall = (
        candidate_metrics["macro_recall"]
    )

    macro_recall_drop = (
        baseline_macro_recall
        - candidate_macro_recall
    )

    macro_recall_pass = (
        macro_recall_drop
        <= quality_gates[
            "max_macro_recall_drop"
        ]
    )


    # ========================================================
    # QUALITY GATE 3
    # PER-CATEGORY F1
    # ========================================================

    category_results = {}

    category_gate_pass = True


    for label in LABELS:

        baseline_f1 = (
            baseline_metrics[label]["f1"]
        )

        candidate_f1 = (
            candidate_metrics[label]["f1"]
        )

        f1_drop = (
            baseline_f1
            - candidate_f1
        )

        passed = (
            f1_drop
            <= quality_gates[
                "max_category_f1_drop"
            ]
        )


        if not passed:
            category_gate_pass = False


        category_results[label] = {
            "baseline_f1":
                baseline_f1,

            "candidate_f1":
                candidate_f1,

            "f1_drop":
                f1_drop,

            "passed":
                passed,
        }


    # ========================================================
    # PERFORMANCE GATE
    # ========================================================

    baseline_latency = (
        baseline_metrics[
            "average_ms_per_comment"
        ]
    )

    candidate_latency = (
        candidate_metrics[
            "average_ms_per_comment"
        ]
    )


    speedup = (
        baseline_latency
        / candidate_latency
    )


    speedup_pass = (
        speedup
        >= performance_gates[
            "min_speedup"
        ]
    )


    # ========================================================
    # MODEL SIZE GATE
    # ========================================================

    baseline_size = model_size_mb(
        MODEL_FILES[baseline_name]
    )

    candidate_size = model_size_mb(
        MODEL_FILES[candidate_name]
    )


    size_reduction_percent = (
        (
            baseline_size
            - candidate_size
        )
        / baseline_size
    ) * 100


    size_pass = (
        size_reduction_percent
        >= performance_gates[
            "min_size_reduction_percent"
        ]
    )


    # ========================================================
    # OVERALL RESULT
    # ========================================================

    overall_pass = all([
        macro_f1_pass,
        macro_recall_pass,
        category_gate_pass,
        speedup_pass,
        size_pass,
    ])


    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "baseline_runtime":
            baseline_name,

        "candidate_runtime":
            candidate_name,

        "overall_pass":
            overall_pass,


        "quality": {

            "macro_f1": {
                "baseline":
                    baseline_macro_f1,

                "candidate":
                    candidate_macro_f1,

                "drop":
                    macro_f1_drop,

                "maximum_allowed_drop":
                    quality_gates[
                        "max_macro_f1_drop"
                    ],

                "passed":
                    macro_f1_pass,
            },


            "macro_recall": {
                "baseline":
                    baseline_macro_recall,

                "candidate":
                    candidate_macro_recall,

                "drop":
                    macro_recall_drop,

                "maximum_allowed_drop":
                    quality_gates[
                        "max_macro_recall_drop"
                    ],

                "passed":
                    macro_recall_pass,
            },


            "categories":
                category_results,
        },


        "performance": {

            "baseline_ms_per_comment":
                baseline_latency,

            "candidate_ms_per_comment":
                candidate_latency,

            "speedup":
                speedup,

            "minimum_required_speedup":
                performance_gates[
                    "min_speedup"
                ],

            "passed":
                speedup_pass,
        },


        "model_size": {

            "baseline_mb":
                baseline_size,

            "candidate_mb":
                candidate_size,

            "reduction_percent":
                size_reduction_percent,

            "minimum_required_reduction_percent":
                performance_gates[
                    "min_size_reduction_percent"
                ],

            "passed":
                size_pass,
        },
    }


    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )


    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL PROMOTION REPORT"
    )

    print(
        "=" * 70
    )


    print(
        f"\nBaseline : "
        f"{baseline_name}"
    )

    print(
        f"Candidate: "
        f"{candidate_name}"
    )


    print(
        "\nQUALITY GATES"
    )

    print(
        "-" * 70
    )


    print(
        f"Macro F1       "
        f"{pass_fail(macro_f1_pass):5}  "
        f"drop={macro_f1_drop:.4f}"
    )


    print(
        f"Macro Recall   "
        f"{pass_fail(macro_recall_pass):5}  "
        f"drop={macro_recall_drop:.4f}"
    )


    print(
        "\nCategory F1:"
    )


    for label in LABELS:

        result = category_results[
            label
        ]

        print(
            f"  {label:18}"
            f"{pass_fail(result['passed']):5}  "
            f"drop={result['f1_drop']:+.4f}"
        )


    print(
        "\nPERFORMANCE GATE"
    )

    print(
        "-" * 70
    )


    print(
        f"Speedup        "
        f"{pass_fail(speedup_pass):5}  "
        f"{speedup:.2f}x"
    )


    print(
        "\nMODEL SIZE GATE"
    )

    print(
        "-" * 70
    )


    print(
        f"Size reduction "
        f"{pass_fail(size_pass):5}  "
        f"{size_reduction_percent:.2f}%"
    )


    print(
        "\n"
        + "=" * 70
    )


    print(
        "OVERALL: "
        + (
            "PASS"
            if overall_pass
            else "FAIL"
        )
    )


    print(
        "=" * 70
    )


    print(
        f"\nReport saved to: "
        f"{REPORT_FILE}"
    )


    # ========================================================
    # CI/CD EXIT CODE
    # ========================================================

    if overall_pass:
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()