import json
import time
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
)

from src.model.onnx_predictor import (
    ONNXModerationModel,
)

from src.model.onnx_int8_predictor import (
    INT8ModerationModel,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    "evaluation/data/jigsaw_eval_subset.csv"
)

RESULTS_DIR = Path(
    "evaluation/results"
)

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# This is the threshold used to turn model probabilities
# into binary predictions for quality evaluation.
CLASSIFICATION_THRESHOLD = 0.5


# ============================================================
# DATA LOADING + VALIDATION
# ============================================================

def load_and_clean_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{DATA_FILE}"
        )

    print("Loading evaluation data...")

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"Original evaluation rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_columns = [
        "id",
        "comment_text",
        *LABELS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # Remove missing text
    # --------------------------------------------------------

    missing_text_count = (
        df["comment_text"]
        .isna()
        .sum()
    )

    print(
        f"Rows with missing text: "
        f"{missing_text_count:,}"
    )

    df = df.dropna(
        subset=["comment_text"]
    ).copy()


    # --------------------------------------------------------
    # Convert valid comments to strings
    # --------------------------------------------------------

    df["comment_text"] = (
        df["comment_text"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Remove empty text
    # --------------------------------------------------------

    empty_text_count = (
        df["comment_text"]
        .eq("")
        .sum()
    )

    print(
        f"Rows with empty text: "
        f"{empty_text_count:,}"
    )

    df = df[
        df["comment_text"] != ""
    ].copy()


    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    for label in LABELS:

        if df[label].isna().any():

            missing_labels = (
                df[label]
                .isna()
                .sum()
            )

            raise ValueError(
                f"Label '{label}' contains "
                f"{missing_labels} missing values."
            )


        # Convert labels to integers.
        df[label] = (
            df[label]
            .astype(int)
        )


        # Labels must be binary.
        invalid_values = set(
            df[label].unique()
        ) - {0, 1}

        if invalid_values:

            raise ValueError(
                f"Label '{label}' contains "
                f"invalid values: "
                f"{invalid_values}"
            )


    # Reset index so progress counting is clean.
    df = df.reset_index(
        drop=True
    )


    print(
        f"Usable evaluation rows: "
        f"{len(df):,}"
    )


    print(
        "\nPositive examples in evaluation set:"
    )

    for label in LABELS:

        positive_count = int(
            df[label].sum()
        )

        print(
            f"{label:20}"
            f"{positive_count:,}"
        )


    return df


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    runtime_name,
    model,
    df,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Evaluating {runtime_name}"
    )

    print(
        "=" * 70
    )


    predictions = []

    total_rows = len(df)

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for index, row in df.iterrows():

        text = row["comment_text"]

        # Extra safety check.
        if not isinstance(text, str):

            raise TypeError(
                f"Invalid comment_text "
                f"at row {index}. "
                f"Received type: "
                f"{type(text)}"
            )


        scores = model.predict(
            text
        )


        # Make sure model returned every category.
        missing_outputs = [
            label
            for label in LABELS
            if label not in scores
        ]

        if missing_outputs:

            raise ValueError(
                f"{runtime_name} did not return "
                f"these labels: "
                f"{missing_outputs}"
            )


        predictions.append(
            scores
        )


        # Progress display
        completed = index + 1

        if (
            completed % 500 == 0
            or completed == total_rows
        ):

            print(
                f"{completed:,}"
                f"/{total_rows:,}"
            )


    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )


    # --------------------------------------------------------
    # Convert predictions into DataFrame
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        predictions
    )


    # --------------------------------------------------------
    # Calculate quality metrics
    # --------------------------------------------------------

    metrics = {}

    macro_precision_values = []
    macro_recall_values = []
    macro_f1_values = []


    for label in LABELS:

        y_true = (
            df[label]
            .astype(int)
            .to_numpy()
        )


        probabilities = (
            prediction_df[label]
            .astype(float)
            .to_numpy()
        )


        y_pred = (
            probabilities
            >= CLASSIFICATION_THRESHOLD
        ).astype(int)


        (
            precision,
            recall,
            f1,
            _,
        ) = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )


        metrics[label] = {

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),
        }


        macro_precision_values.append(
            precision
        )

        macro_recall_values.append(
            recall
        )

        macro_f1_values.append(
            f1
        )


    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    metrics["macro_precision"] = float(
        sum(macro_precision_values)
        / len(macro_precision_values)
    )

    metrics["macro_recall"] = float(
        sum(macro_recall_values)
        / len(macro_recall_values)
    )

    metrics["macro_f1"] = float(
        sum(macro_f1_values)
        / len(macro_f1_values)
    )


    metrics["rows"] = int(
        total_rows
    )


    metrics["classification_threshold"] = (
        CLASSIFICATION_THRESHOLD
    )


    metrics["total_seconds"] = float(
        elapsed_seconds
    )


    metrics[
        "average_ms_per_comment"
    ] = float(
        (
            elapsed_seconds
            / total_rows
        )
        * 1000
    )


    print(
        f"\n{runtime_name} completed."
    )

    print(
        f"Total time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print(
        f"Average evaluation time: "
        f"{metrics['average_ms_per_comment']:.2f} "
        f"ms/comment"
    )


    return (
        metrics,
        prediction_df,
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    runtime_name,
    metrics,
):

    output_file = (
        RESULTS_DIR
        / f"{runtime_name}_metrics.json"
    )


    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )


    print(
        f"Saved metrics: "
        f"{output_file}"
    )


# ============================================================
# BUILD COMPARISON CSV
# ============================================================

def create_comparison(
    fp32_metrics,
    int8_metrics,
):

    rows = []


    for label in LABELS:

        fp32_precision = (
            fp32_metrics[label][
                "precision"
            ]
        )

        int8_precision = (
            int8_metrics[label][
                "precision"
            ]
        )


        fp32_recall = (
            fp32_metrics[label][
                "recall"
            ]
        )

        int8_recall = (
            int8_metrics[label][
                "recall"
            ]
        )


        fp32_f1 = (
            fp32_metrics[label][
                "f1"
            ]
        )

        int8_f1 = (
            int8_metrics[label][
                "f1"
            ]
        )


        rows.append({

            "category":
                label,

            "fp32_precision":
                fp32_precision,

            "int8_precision":
                int8_precision,

            "precision_change":
                int8_precision
                - fp32_precision,

            "fp32_recall":
                fp32_recall,

            "int8_recall":
                int8_recall,

            "recall_change":
                int8_recall
                - fp32_recall,

            "fp32_f1":
                fp32_f1,

            "int8_f1":
                int8_f1,

            "f1_change":
                int8_f1
                - fp32_f1,
        })


    comparison_df = pd.DataFrame(
        rows
    )


    output_file = (
        RESULTS_DIR
        / "comparison.csv"
    )


    comparison_df.to_csv(
        output_file,
        index=False,
    )


    print(
        f"Saved comparison: "
        f"{output_file}"
    )


    return comparison_df


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

def save_predictions(
    df,
    fp32_predictions,
    int8_predictions,
):

    # Keep original ground-truth information.
    output = df[
        [
            "id",
            "comment_text",
            *LABELS,
        ]
    ].copy()


    # Add FP32 and INT8 probability scores.
    for label in LABELS:

        output[
            f"fp32_{label}"
        ] = fp32_predictions[
            label
        ].to_numpy()


        output[
            f"int8_{label}"
        ] = int8_predictions[
            label
        ].to_numpy()


        # Also save binary predictions.
        output[
            f"fp32_pred_{label}"
        ] = (
            fp32_predictions[label]
            >= CLASSIFICATION_THRESHOLD
        ).astype(int).to_numpy()


        output[
            f"int8_pred_{label}"
        ] = (
            int8_predictions[label]
            >= CLASSIFICATION_THRESHOLD
        ).astype(int).to_numpy()


    output_file = (
        RESULTS_DIR
        / "predictions.csv"
    )


    output.to_csv(
        output_file,
        index=False,
    )


    print(
        f"Saved predictions: "
        f"{output_file}"
    )


# ============================================================
# PRINT QUALITY SUMMARY
# ============================================================

def print_summary(
    fp32_metrics,
    int8_metrics,
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        "MODEL QUALITY COMPARISON"
    )

    print(
        "=" * 78
    )


    print(
        "\n"
        f"{'CATEGORY':20}"
        f"{'FP32 F1':>12}"
        f"{'INT8 F1':>12}"
        f"{'CHANGE':>12}"
    )


    print(
        "-" * 56
    )


    for label in LABELS:

        fp32_f1 = (
            fp32_metrics[label][
                "f1"
            ]
        )

        int8_f1 = (
            int8_metrics[label][
                "f1"
            ]
        )

        difference = (
            int8_f1
            - fp32_f1
        )


        print(
            f"{label:20}"
            f"{fp32_f1:>12.4f}"
            f"{int8_f1:>12.4f}"
            f"{difference:>+12.4f}"
        )


    print(
        "\n"
        + "-" * 56
    )


    print(
        f"{'Macro Precision':20}"
        f"{fp32_metrics['macro_precision']:>12.4f}"
        f"{int8_metrics['macro_precision']:>12.4f}"
    )


    print(
        f"{'Macro Recall':20}"
        f"{fp32_metrics['macro_recall']:>12.4f}"
        f"{int8_metrics['macro_recall']:>12.4f}"
    )


    print(
        f"{'Macro F1':20}"
        f"{fp32_metrics['macro_f1']:>12.4f}"
        f"{int8_metrics['macro_f1']:>12.4f}"
    )


    macro_f1_change = (
        int8_metrics["macro_f1"]
        - fp32_metrics["macro_f1"]
    )


    print(
        f"\nMacro F1 change: "
        f"{macro_f1_change:+.4f}"
    )


    print(
        "\n"
        f"FP32 evaluation speed: "
        f"{fp32_metrics['average_ms_per_comment']:.2f} "
        f"ms/comment"
    )


    print(
        f"INT8 evaluation speed: "
        f"{int8_metrics['average_ms_per_comment']:.2f} "
        f"ms/comment"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # 1. Load and validate evaluation data
    # --------------------------------------------------------

    df = load_and_clean_data()


    if len(df) == 0:

        raise ValueError(
            "No valid evaluation rows remain "
            "after cleaning."
        )


    # --------------------------------------------------------
    # 2. Load models
    # --------------------------------------------------------

    print(
        "\nLoading ONNX FP32 model..."
    )

    fp32_model = (
        ONNXModerationModel()
    )


    print(
        "Loading ONNX INT8 model..."
    )

    int8_model = (
        INT8ModerationModel()
    )


    # --------------------------------------------------------
    # 3. Evaluate FP32
    # --------------------------------------------------------

    (
        fp32_metrics,
        fp32_predictions,
    ) = evaluate_model(
        "onnx_fp32",
        fp32_model,
        df,
    )


    # --------------------------------------------------------
    # 4. Evaluate INT8
    # --------------------------------------------------------

    (
        int8_metrics,
        int8_predictions,
    ) = evaluate_model(
        "onnx_int8",
        int8_model,
        df,
    )


    # --------------------------------------------------------
    # 5. Save metrics
    # --------------------------------------------------------

    save_metrics(
        "onnx_fp32",
        fp32_metrics,
    )

    save_metrics(
        "onnx_int8",
        int8_metrics,
    )


    # --------------------------------------------------------
    # 6. Save comparison
    # --------------------------------------------------------

    create_comparison(
        fp32_metrics,
        int8_metrics,
    )


    # --------------------------------------------------------
    # 7. Save raw predictions
    # --------------------------------------------------------

    save_predictions(
        df,
        fp32_predictions,
        int8_predictions,
    )


    # --------------------------------------------------------
    # 8. Print final summary
    # --------------------------------------------------------

    print_summary(
        fp32_metrics,
        int8_metrics,
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "Evaluation complete."
    )

    print(
        "Results saved to:"
    )

    print(
        "evaluation/results/"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()