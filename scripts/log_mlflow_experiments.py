import json
from pathlib import Path

import mlflow


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "moderation-runtime-comparison"

TRACKING_URI = "sqlite:///mlflow.db"

RESULTS_DIR = Path(
    "evaluation/results"
)

MODEL_DIR = Path(
    "models"
)

BENCHMARK_FILE = Path(
    "benchmarks/raw/model_runtime.csv"
)

COMPARISON_FILE = (
    RESULTS_DIR / "comparison.csv"
)


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


RUNTIMES = {

    "onnx_fp32": {

        "metrics_file":
            RESULTS_DIR
            / "onnx_fp32_metrics.json",

        "model_file":
            MODEL_DIR
            / "onnx/model.onnx",

        "model_size_mb":
            417.86,
    },


    "onnx_int8": {

        "metrics_file":
            RESULTS_DIR
            / "onnx_int8_metrics.json",

        "model_file":
            MODEL_DIR
            / "onnx_int8/model_quantized.onnx",

        "model_size_mb":
            105.14,
    },
}


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Metrics file missing: {path}"
        )


    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# VALIDATE ARTIFACTS
# ============================================================

def validate_runtime(
    runtime_name,
    config,
):

    metrics_file = config[
        "metrics_file"
    ]

    model_file = config[
        "model_file"
    ]


    if not metrics_file.exists():

        raise FileNotFoundError(
            f"{runtime_name} metrics missing: "
            f"{metrics_file}"
        )


    if not model_file.exists():

        raise FileNotFoundError(
            f"{runtime_name} model missing: "
            f"{model_file}"
        )


# ============================================================
# LOG ONE RUNTIME
# ============================================================

def log_runtime(
    runtime_name,
    config,
):

    validate_runtime(
        runtime_name,
        config,
    )


    metrics = load_metrics(
        config["metrics_file"]
    )


    print(
        f"\nLogging {runtime_name}..."
    )


    with mlflow.start_run(
        run_name=runtime_name
    ) as run:


        # ====================================================
        # PARAMETERS
        # ====================================================

        mlflow.log_param(
            "runtime",
            runtime_name,
        )


        mlflow.log_param(
            "base_model",
            "unitary/toxic-bert",
        )


        mlflow.log_param(
            "device",
            "cpu",
        )


        mlflow.log_param(
            "evaluation_rows",
            metrics["rows"],
        )


        mlflow.log_param(
            "classification_threshold",
            metrics[
                "classification_threshold"
            ],
        )


        if runtime_name == "onnx_fp32":

            mlflow.log_param(
                "precision",
                "fp32",
            )

            mlflow.log_param(
                "quantized",
                False,
            )


        elif runtime_name == "onnx_int8":

            mlflow.log_param(
                "precision",
                "int8",
            )

            mlflow.log_param(
                "quantized",
                True,
            )


        # ====================================================
        # AGGREGATE QUALITY METRICS
        # ====================================================

        mlflow.log_metric(
            "macro_precision",
            metrics["macro_precision"],
        )


        mlflow.log_metric(
            "macro_recall",
            metrics["macro_recall"],
        )


        mlflow.log_metric(
            "macro_f1",
            metrics["macro_f1"],
        )


        # ====================================================
        # PERFORMANCE METRICS
        # ====================================================

        mlflow.log_metric(
            "evaluation_ms_per_comment",
            metrics[
                "average_ms_per_comment"
            ],
        )


        mlflow.log_metric(
            "evaluation_total_seconds",
            metrics[
                "total_seconds"
            ],
        )


        mlflow.log_metric(
            "model_size_mb",
            config[
                "model_size_mb"
            ],
        )


        # ====================================================
        # PER-CATEGORY QUALITY
        # ====================================================

        for label in LABELS:

            category_metrics = (
                metrics[label]
            )


            mlflow.log_metric(
                f"{label}_precision",
                category_metrics[
                    "precision"
                ],
            )


            mlflow.log_metric(
                f"{label}_recall",
                category_metrics[
                    "recall"
                ],
            )


            mlflow.log_metric(
                f"{label}_f1",
                category_metrics[
                    "f1"
                ],
            )


        # ====================================================
        # ARTIFACTS
        # ====================================================

        mlflow.log_artifact(
            str(
                config[
                    "metrics_file"
                ]
            ),
            artifact_path="evaluation",
        )


        if COMPARISON_FILE.exists():

            mlflow.log_artifact(
                str(COMPARISON_FILE),
                artifact_path="evaluation",
            )


        if BENCHMARK_FILE.exists():

            mlflow.log_artifact(
                str(BENCHMARK_FILE),
                artifact_path="benchmarks",
            )


        # Store the actual model artifact.
        mlflow.log_artifact(
            str(
                config[
                    "model_file"
                ]
            ),
            artifact_path="model",
        )


        print(
            f"Run ID: {run.info.run_id}"
        )


        print(
            f"Logged {runtime_name} successfully."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Database-backed MLflow tracking
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        TRACKING_URI
    )


    print(
        f"MLflow tracking URI: "
        f"{TRACKING_URI}"
    )


    # --------------------------------------------------------
    # Create/select experiment
    # --------------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


    print(
        f"Experiment: "
        f"{EXPERIMENT_NAME}"
    )


    # --------------------------------------------------------
    # Log both model variants
    # --------------------------------------------------------

    for runtime_name, config in (
        RUNTIMES.items()
    ):

        log_runtime(
            runtime_name,
            config,
        )


    print(
        "\n"
        + "=" * 60
    )


    print(
        "MLflow experiment logging complete."
    )


    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()