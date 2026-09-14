from typing import Dict

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class ModerationModel:

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.max_length = max_length

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.id2label = (
            self.model.config.id2label
        )


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

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )

        probabilities = torch.sigmoid(
            outputs.logits
        )[0]

        return {
            self.id2label[i]:
                round(
                    probabilities[i].item(),
                    4,
                )

            for i
            in range(
                len(probabilities)
            )
        }