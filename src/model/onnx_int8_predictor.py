from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class INT8ModerationModel:
    """
    INT8 quantized ONNX version of Toxic-BERT.
    """

    def __init__(
        self,
        model_path: str = (
            "models/onnx_int8/model_quantized.onnx"
        ),
        tokenizer_name: str = "unitary/toxic-bert",
        max_length: int = 256,
    ):
        self.model_name = (
            "unitary/toxic-bert-onnx-int8"
        )

        self.device = "cpu"
        self.max_length = max_length

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"INT8 model not found: "
                f"{self.model_path}"
            )

        # Same tokenizer as FP32/PyTorch
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

        # Load quantized ONNX model
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=[
                "CPUExecutionProvider"
            ],
        )

        # Find which inputs the ONNX graph expects
        self.input_names = {
            item.name
            for item in self.session.get_inputs()
        }

        # Toxic-BERT output order
        self.labels = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]

    def predict(
        self,
        text: str,
    ) -> Dict[str, float]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "text cannot be empty"
            )

        # Convert text into token arrays
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # Only send inputs expected by ONNX
        ort_inputs = {
            key: value
            for key, value in inputs.items()
            if key in self.input_names
        }

        # INT8 ONNX inference
        outputs = self.session.run(
            None,
            ort_inputs,
        )

        logits = outputs[0][0]

        # Toxic-BERT is multi-label,
        # therefore use sigmoid.
        probabilities = (
            1.0
            / (
                1.0
                + np.exp(-logits)
            )
        )

        return {
            label: round(
                float(probability),
                4,
            )
            for label, probability
            in zip(
                self.labels,
                probabilities,
            )
        }