from pathlib import Path

import pandas as pd
from datasets import load_dataset


OUTPUT_DIR = Path("evaluation/data")
OUTPUT_FILE = OUTPUT_DIR / "jigsaw_eval.csv"


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def find_text_column(df: pd.DataFrame) -> str:
    """
    Find the actual text column used by the dataset.
    """

    candidates = [
        "comment_text",
        "text",
        "comment",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find a text column. "
        f"Available columns: {df.columns.tolist()}"
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading Jigsaw dataset...")

    dataset = load_dataset(
        "Heliosoph/Jigsaw-Toxic-Comments"
    )

    print(
        "Available splits:",
        list(dataset.keys()),
    )

    # --------------------------------------------------------
    # Prefer a split that already contains BOTH
    # comment text and labels.
    # --------------------------------------------------------

    usable_df = None

    for split_name in dataset.keys():

        print(
            f"\nInspecting split: {split_name}"
        )

        split_df = (
            dataset[split_name]
            .to_pandas()
        )

        print(
            "Columns:",
            split_df.columns.tolist(),
        )

        try:
            text_column = find_text_column(
                split_df
            )

        except ValueError:
            continue

        has_labels = all(
            label in split_df.columns
            for label in LABELS
        )

        if not has_labels:
            continue

        candidate = split_df.copy()

        if text_column != "comment_text":

            candidate = candidate.rename(
                columns={
                    text_column:
                        "comment_text"
                }
            )

        # ----------------------------------------------------
        # Remove Kaggle rows whose test labels are unavailable.
        #
        # Released test labels may use -1 to mean:
        # "not part of scored evaluation set".
        # ----------------------------------------------------

        for label in LABELS:

            candidate[label] = (
                pd.to_numeric(
                    candidate[label],
                    errors="coerce",
                )
            )

        candidate = candidate.dropna(
            subset=LABELS
        )

        valid_mask = (
            candidate[LABELS]
            .isin([0, 1])
            .all(axis=1)
        )

        candidate = candidate[
            valid_mask
        ].copy()

        # ----------------------------------------------------
        # Validate text BEFORE accepting this split.
        # ----------------------------------------------------

        candidate = candidate.dropna(
            subset=["comment_text"]
        ).copy()

        candidate["comment_text"] = (
            candidate["comment_text"]
            .astype(str)
            .str.strip()
        )

        candidate = candidate[
            candidate["comment_text"] != ""
        ].copy()

        if len(candidate) == 0:
            continue

        usable_df = candidate

        print(
            f"Using split: {split_name}"
        )

        break


    if usable_df is None:

        raise RuntimeError(
            "Could not find a dataset split containing "
            "both valid comment text and binary labels."
        )


    df = usable_df


    # --------------------------------------------------------
    # Ensure ID exists
    # --------------------------------------------------------

    if "id" not in df.columns:

        df = df.reset_index(
            drop=True
        )

        df.insert(
            0,
            "id",
            df.index.astype(str),
        )


    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    df = df[
        [
            "id",
            "comment_text",
            *LABELS,
        ]
    ].copy()


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    missing_text = (
        df["comment_text"]
        .isna()
        .sum()
    )

    if missing_text != 0:

        raise RuntimeError(
            f"Dataset still contains "
            f"{missing_text} missing comments."
        )


    for label in LABELS:

        invalid = (
            ~df[label].isin(
                [0, 1]
            )
        ).sum()

        if invalid:

            raise RuntimeError(
                f"{label} contains "
                f"{invalid} invalid labels."
            )

        df[label] = (
            df[label]
            .astype(int)
        )


    df = df.reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DATASET READY"
    )

    print(
        "=" * 60
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Missing comments: "
        f"{df['comment_text'].isna().sum():,}"
    )

    print(
        "\nPositive labels:"
    )


    for label in LABELS:

        positives = int(
            df[label].sum()
        )

        print(
            f"{label:20}"
            f"{positives:,}"
        )


    print(
        f"\nSaved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()