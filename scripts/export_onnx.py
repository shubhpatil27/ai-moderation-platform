from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


MODEL_NAME = "unitary/toxic-bert"

OUTPUT_DIR = Path("models/onnx")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Exporting {MODEL_NAME} to ONNX...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        export=True,
    )

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print(f"ONNX model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()