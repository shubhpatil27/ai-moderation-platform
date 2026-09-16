from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXModerationModel:
    """
    ONNX Runtime version of our moderation model.

    It exposes the same predict(text) interface as
    the existing PyTorch ModerationModel.
    """

    def __init__(
        self,
        model_path: str = "models/onnx/model.onnx",
        tokenizer_name: str = "unitary/toxic-bert",
        max_length: int = 256,
    ):
        self.model_name = "unitary/toxic-bert-onnx"
        self.max_length = max_length

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        # Same tokenizer as the PyTorch model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

        # Create ONNX Runtime inference session
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.device = "cpu"

        # Same label order used by Toxic-BERT
        self.labels = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]

        # Cache the names of inputs expected by ONNX.
        self.input_names = {
            input_info.name
            for input_info in self.session.get_inputs()
        }

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

        # Tokenize directly into NumPy arrays
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # Only send inputs that this ONNX model expects.
        ort_inputs = {
            key: value
            for key, value in inputs.items()
            if key in self.input_names
        }

        # Run inference
        outputs = self.session.run(
            None,
            ort_inputs,
        )

        logits = outputs[0][0]

        # Multi-label classification -> sigmoid
        probabilities = (
            1.0
            / (1.0 + np.exp(-logits))
        )

        return {
            label: round(
                float(probability),
                4,
            )
            for label, probability in zip(
                self.labels,
                probabilities,
            )
        }