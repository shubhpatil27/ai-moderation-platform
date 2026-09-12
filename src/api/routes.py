from fastapi import APIRouter, Depends

from src.api.dependencies import get_moderation_model
from src.api.schemas import (
    ModerationRequest,
    ModerationResponse,
)
from src.model.policy import moderate
from src.model.predictor import ModerationModel


router = APIRouter()


# LIVENESS
@router.get("/health")
def health():
    return {
        "status": "healthy"
    }


# READINESS
@router.get("/ready")
def ready(
    model: ModerationModel = Depends(get_moderation_model),
):
    return {
        "status": "ready",
        "model": model.model_name,
        "device": str(model.device),
    }


# ML INFERENCE
@router.post(
    "/moderate",
    response_model=ModerationResponse,
)
def moderate_comment(
    request: ModerationRequest,
    model: ModerationModel = Depends(get_moderation_model),
):
    scores = model.predict(request.text)

    preferences = request.preferences.model_dump()

    result = moderate(
        scores=scores,
        preferences=preferences,
    )

    return ModerationResponse(
        decision=result["decision"],
        triggered_categories=result["triggered_categories"],
        scores=scores,
        model_version=model.model_name,
    )