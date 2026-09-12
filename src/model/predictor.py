from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModerationModel:
    """
    Loads the toxicity classification model and produces
    a score between 0 and 1 for each moderation category.
    """

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.max_length = max_length

        # Use GPU if one is available, otherwise CPU.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

        # Read labels directly from the model configuration.
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> Dict[str, float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # Move tensors to the same device as the model.
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # We are predicting, not training.
        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits

        # This is a multi-label model, so each label gets
        # an independent sigmoid score.
        probabilities = torch.sigmoid(logits)[0]

        scores = {
            self.id2label[i]: round(probabilities[i].item(), 4)
            for i in range(len(probabilities))
        }

        return scores