import json
from pathlib import Path

import mlflow
import onnx

from mlflow import MlflowClient


# ============================================================
# CONFIG
# ============================================================

TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_NAME = "moderation-model-registry"

REGISTERED_MODEL_NAME = "toxicity-moderation-model"

MODEL_PATH = Path(
    "models/onnx_int8/model_quantized.onnx"
)

PROMOTION_REPORT = Path(
    "evaluation/results/promotion_report.json"
)

METRICS_FILE = Path(
    "evaluation/results/onnx_int8_metrics.json"
)


# ============================================================
# LOAD JSON
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


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate promotion gate
    # --------------------------------------------------------

    promotion = load_json(
        PROMOTION_REPORT
    )

    if not promotion["overall_pass"]:

        raise RuntimeError(
            "Candidate failed promotion gate. "
            "Registration blocked."
        )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"INT8 model not found: {MODEL_PATH}"
        )


    metrics = load_json(
        METRICS_FILE
    )


    # --------------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_registry_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


    print(
        f"Tracking URI: {TRACKING_URI}"
    )

    print(
        f"Registered model: "
        f"{REGISTERED_MODEL_NAME}"
    )


    # --------------------------------------------------------
    # Load ONNX model
    # --------------------------------------------------------

    print(
        "\nLoading approved INT8 ONNX model..."
    )

    onnx_model = onnx.load(
        str(MODEL_PATH)
    )


    # --------------------------------------------------------
    # Create MLflow run and register model
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="onnx_int8_approved"
    ) as run:

        # Parameters

        mlflow.log_param(
            "runtime",
            "onnx_int8",
        )

        mlflow.log_param(
            "base_model",
            "unitary/toxic-bert",
        )

        mlflow.log_param(
            "precision",
            "int8",
        )

        mlflow.log_param(
            "promotion_gate",
            "PASS",
        )


        # Quality metrics

        mlflow.log_metric(
            "macro_f1",
            metrics["macro_f1"],
        )

        mlflow.log_metric(
            "macro_precision",
            metrics["macro_precision"],
        )

        mlflow.log_metric(
            "macro_recall",
            metrics["macro_recall"],
        )

        mlflow.log_metric(
            "evaluation_ms_per_comment",
            metrics[
                "average_ms_per_comment"
            ],
        )


        # Promotion report artifact

        mlflow.log_artifact(
            str(PROMOTION_REPORT),
            artifact_path="promotion",
        )


        # ----------------------------------------------------
        # Log + register actual MLflow ONNX model
        # ----------------------------------------------------

        model_info = mlflow.onnx.log_model(
            onnx_model=onnx_model,

            name="model",

            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),

            onnx_execution_providers=[
                "CPUExecutionProvider"
            ],
        )


        print(
            f"\nRun ID: {run.info.run_id}"
        )


        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )


        version = (
            model_info.registered_model_version
        )


        if version is None:

            raise RuntimeError(
                "MLflow did not return a "
                "registered model version."
            )


        version = str(version)


    # --------------------------------------------------------
    # Registry metadata + alias
    # --------------------------------------------------------

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )


    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="promotion_gate",
        value="PASSED",
    )


    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="runtime",
        value="onnx_int8",
    )


    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="base_model",
        value="unitary/toxic-bert",
    )


    # Approved but not yet deployed.
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="candidate",
        version=version,
    )


    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    candidate = (
        client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            "candidate",
        )
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "MODEL REGISTRATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version: "
        f"{candidate.version}"
    )

    print(
        "Alias: candidate"
    )

    print(
        "Promotion gate: PASSED"
    )

    print(
        "\nRegistry URI:"
    )

    print(
        f"models:/"
        f"{REGISTERED_MODEL_NAME}"
        f"@candidate"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()