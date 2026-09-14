from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sqlalchemy.orm import Session

from src.api.dependencies import (
    get_moderation_model,
)

from src.api.schemas import (
    FeedbackCreate,
    FeedbackResponse,
    ModerationRequest,
    ModerationResponse,
    PredictionResponse,
    PreferencesResponse,
    PreferencesUpdate,
    UserCreate,
    UserResponse,
)

from src.database import repository

from src.database.db import get_db

from src.model.policy import moderate

from src.model.predictor import (
    ModerationModel,
)


router = APIRouter()


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# READINESS
# ============================================================

@router.get("/ready")
def ready(
    model: ModerationModel = Depends(
        get_moderation_model
    ),
):

    return {
        "status": "ready",
        "model": model.model_name,
        "device": str(model.device),
    }


# ============================================================
# CREATE USER
# ============================================================

@router.post(
    "/users",
    response_model=UserResponse,
    status_code=201,
)
def create_user(
    request: UserCreate,
    db: Session = Depends(get_db),
):

    existing = (
        repository.get_user_by_username(
            db,
            request.username,
        )
    )

    if existing is not None:

        raise HTTPException(
            status_code=409,
            detail="Username already exists",
        )

    return repository.create_user(
        db,
        request.username,
    )


# ============================================================
# GET PREFERENCES
# ============================================================

@router.get(
    "/users/{user_id}/preferences",
    response_model=PreferencesResponse,
)
def get_preferences(
    user_id: int,
    db: Session = Depends(get_db),
):

    user = repository.get_user(
        db,
        user_id,
    )

    if user is None:

        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    preferences = (
        repository.get_user_preferences(
            db,
            user_id,
        )
    )

    if preferences is None:

        raise HTTPException(
            status_code=404,
            detail="Preferences not found",
        )

    return preferences


# ============================================================
# UPDATE PREFERENCES
# ============================================================

@router.put(
    "/users/{user_id}/preferences",
    response_model=PreferencesResponse,
)
def update_preferences(
    user_id: int,
    request: PreferencesUpdate,
    db: Session = Depends(get_db),
):

    user = repository.get_user(
        db,
        user_id,
    )

    if user is None:

        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    preferences = (
        repository.update_user_preferences(
            db,
            user_id,
            request.model_dump(),
        )
    )

    if preferences is None:

        raise HTTPException(
            status_code=404,
            detail="Preferences not found",
        )

    return preferences


# ============================================================
# MODERATE COMMENT
# ============================================================

@router.post(
    "/moderate",
    response_model=ModerationResponse,
)
def moderate_comment(
    request: ModerationRequest,

    model: ModerationModel = Depends(
        get_moderation_model
    ),

    db: Session = Depends(
        get_db
    ),
):

    user = repository.get_user(
        db,
        request.user_id,
    )

    if user is None:

        raise HTTPException(
            status_code=404,
            detail="User not found",
        )


    stored = (
        repository.get_user_preferences(
            db,
            request.user_id,
        )
    )

    if stored is None:

        raise HTTPException(
            status_code=404,
            detail="Preferences not found",
        )


    preferences = {
        "toxic": stored.toxic,
        "severe_toxic":
            stored.severe_toxic,
        "obscene": stored.obscene,
        "threat": stored.threat,
        "insult": stored.insult,
        "identity_hate":
            stored.identity_hate,
    }


    scores = model.predict(
        request.text
    )


    result = moderate(
        scores=scores,
        preferences=preferences,
    )


    prediction = (
        repository.create_prediction(
            db=db,
            user_id=request.user_id,
            text=request.text,
            decision=result["decision"],
            scores=scores,
            model_version=model.model_name,
        )
    )


    return ModerationResponse(
        prediction_id=prediction.id,

        decision=result["decision"],

        triggered_categories=result[
            "triggered_categories"
        ],

        scores=scores,

        model_version=model.model_name,
    )


# ============================================================
# GET PREDICTION
# ============================================================

@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionResponse,
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
):

    prediction = (
        repository.get_prediction(
            db,
            prediction_id,
        )
    )

    if prediction is None:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return prediction


# ============================================================
# CREATE FEEDBACK
# ============================================================

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
)
def create_feedback(
    request: FeedbackCreate,
    db: Session = Depends(get_db),
):

    prediction = (
        repository.get_prediction(
            db,
            request.prediction_id,
        )
    )

    if prediction is None:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )


    existing = (
        repository
        .get_feedback_for_prediction(
            db,
            request.prediction_id,
        )
    )

    if existing is not None:

        raise HTTPException(
            status_code=409,
            detail=(
                "Feedback already exists "
                "for this prediction"
            ),
        )


    return repository.create_feedback(
        db=db,

        prediction_id=(
            request.prediction_id
        ),

        corrected_decision=(
            request.corrected_decision
        ),

        corrected_category=(
            request.corrected_category
        ),
    )