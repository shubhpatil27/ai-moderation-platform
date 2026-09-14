from functools import lru_cache

from src.model.predictor import ModerationModel


@lru_cache
def get_moderation_model() -> ModerationModel:
    """
    Load one model per application process
    and reuse it for inference requests.
    """

    return ModerationModel()