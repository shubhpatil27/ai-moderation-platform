from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "evaluation/data/jigsaw_eval.csv"
)

OUTPUT_FILE = Path(
    "evaluation/data/jigsaw_eval_subset.csv"
)


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


TARGET_SIZE = 5000
MAX_POSITIVES_PER_LABEL = 500
RANDOM_SEED = 42


def validate_source(
    df: pd.DataFrame,
):

    required = [
        "id",
        "comment_text",
        *LABELS,
    ]

    missing_columns = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            f"{missing_columns}"
        )


    missing_text = (
        df["comment_text"]
        .isna()
        .sum()
    )

    if missing_text:

        raise ValueError(
            f"Source dataset contains "
            f"{missing_text} missing comments."
        )


    empty_text = (
        df["comment_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_text:

        raise ValueError(
            f"Source dataset contains "
            f"{empty_text} empty comments."
        )


    for label in LABELS:

        invalid = (
            ~df[label].isin(
                [0, 1]
            )
        ).sum()

        if invalid:

            raise ValueError(
                f"{label} contains "
                f"{invalid} invalid labels."
            )


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{INPUT_FILE}"
        )


    print(
        "Loading validated evaluation dataset..."
    )


    df = pd.read_csv(
        INPUT_FILE
    )


    validate_source(
        df
    )


    print(
        f"Available rows: "
        f"{len(df):,}"
    )


    selected_indices = set()


    # --------------------------------------------------------
    # Preserve positive examples for each label.
    # --------------------------------------------------------

    for label in LABELS:

        positives = df[
            df[label] == 1
        ]


        sample_size = min(
            MAX_POSITIVES_PER_LABEL,
            len(positives),
        )


        if sample_size == 0:

            print(
                f"WARNING: "
                f"No positive rows for {label}"
            )

            continue


        sampled = positives.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        )


        selected_indices.update(
            sampled.index.tolist()
        )


    # --------------------------------------------------------
    # Fill remaining capacity randomly.
    # --------------------------------------------------------

    remaining = (
        TARGET_SIZE
        - len(selected_indices)
    )


    if remaining > 0:

        available = df.drop(
            index=list(
                selected_indices
            )
        )


        random_rows = available.sample(
            n=min(
                remaining,
                len(available),
            ),
            random_state=RANDOM_SEED,
        )


        selected_indices.update(
            random_rows.index.tolist()
        )


    subset = df.loc[
        list(selected_indices)
    ].copy()


    subset = subset.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Final subset validation
    # --------------------------------------------------------

    validate_source(
        subset
    )


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    subset.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "EVALUATION SUBSET READY"
    )

    print(
        "=" * 60
    )


    print(
        f"Rows: "
        f"{len(subset):,}"
    )


    print(
        "\nPositive examples:"
    )


    for label in LABELS:

        print(
            f"{label:20}"
            f"{int(subset[label].sum()):,}"
        )


    print(
        f"\nSaved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()