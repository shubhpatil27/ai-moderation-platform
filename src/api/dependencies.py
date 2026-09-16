import os
from functools import lru_cache


@lru_cache
def get_moderation_model():
    """
    Load only the inference runtime that is actually configured.

    MODEL_RUNTIME=onnx
        -> ONNX Runtime only
        -> PyTorch is never imported

    MODEL_RUNTIME=pytorch
        -> PyTorch model
    """

    runtime = os.getenv(
        "MODEL_RUNTIME",
        "pytorch",
    ).lower()

    if runtime == "onnx":
        from src.model.onnx_predictor import (
            ONNXModerationModel,
        )

        return ONNXModerationModel()

    if runtime == "pytorch":
        from src.model.predictor import (
            ModerationModel,
        )

        return ModerationModel()

    raise RuntimeError(
        f"Unsupported MODEL_RUNTIME: {runtime}"
    )