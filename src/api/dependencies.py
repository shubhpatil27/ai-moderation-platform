from functools import lru_cache

from src.model.predictor import ModerationModel


@lru_cache
def get_moderation_model() -> ModerationModel:
    """
    Load the real moderation model once and reuse it.
    """
    return ModerationModel()