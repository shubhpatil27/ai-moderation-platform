from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from sqlalchemy.orm import Session

from src.api.dependencies import get_moderation_model
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


router = APIRouter()


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health():
    """
    Liveness check.

    Checks whether the FastAPI application is running.
    Does not load the ML model.
    """

    return {
        "status": "healthy"
    }


# ============================================================
# READINESS
# ============================================================

@router.get("/ready")
def ready(
    model=Depends(get_moderation_model),
):
    """
    Readiness check.

    Loads/checks whichever inference runtime is configured.

    MODEL_RUNTIME=onnx
        -> ONNX Runtime

    MODEL_RUNTIME=pytorch
        -> PyTorch
    """

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
    """
    Create a new user.

    Default moderation preferences are automatically
    created by the repository layer.
    """

    existing_user = (
        repository.get_user_by_username(
            db,
            request.username,
        )
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=409,
            detail="Username already exists",
        )

    user = repository.create_user(
        db,
        request.username,
    )

    return user


# ============================================================
# GET USER PREFERENCES
# ============================================================

@router.get(
    "/users/{user_id}/preferences",
    response_model=PreferencesResponse,
)
def get_preferences(
    user_id: int,
    db: Session = Depends(get_db),
):
    """
    Retrieve the user's stored moderation preferences.
    """

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
# UPDATE USER PREFERENCES
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
    """
    Update a user's moderation preferences.
    """

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
    model=Depends(get_moderation_model),
    db: Session = Depends(get_db),
):
    """
    Moderate a comment using the configured ML runtime
    and the user's stored moderation preferences.

    Flow:

    user_id + text
        ↓
    PostgreSQL preferences
        ↓
    ONNX/PyTorch model
        ↓
    category scores
        ↓
    personalized policy
        ↓
    allow / review / hide
        ↓
    store prediction
    """

    # --------------------------------------------------------
    # 1. Verify user exists
    # --------------------------------------------------------

    user = repository.get_user(
        db,
        request.user_id,
    )

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )


    # --------------------------------------------------------
    # 2. Retrieve stored preferences
    # --------------------------------------------------------

    stored_preferences = (
        repository.get_user_preferences(
            db,
            request.user_id,
        )
    )

    if stored_preferences is None:
        raise HTTPException(
            status_code=404,
            detail="Preferences not found",
        )


    # --------------------------------------------------------
    # 3. Convert preferences into policy dictionary
    # --------------------------------------------------------

    preferences = {
        "toxic":
            stored_preferences.toxic,

        "severe_toxic":
            stored_preferences.severe_toxic,

        "obscene":
            stored_preferences.obscene,

        "threat":
            stored_preferences.threat,

        "insult":
            stored_preferences.insult,

        "identity_hate":
            stored_preferences.identity_hate,
    }


    # --------------------------------------------------------
    # 4. ML inference
    # --------------------------------------------------------

    scores = model.predict(
        request.text
    )


    # --------------------------------------------------------
    # 5. Personalized moderation policy
    # --------------------------------------------------------

    result = moderate(
        scores=scores,
        preferences=preferences,
    )


    # --------------------------------------------------------
    # 6. Store prediction
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 7. Return response
    # --------------------------------------------------------

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
    """
    Retrieve a stored moderation prediction.
    """

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
    """
    Store human feedback about a model prediction.

    A prediction may only have one feedback record.
    """

    # --------------------------------------------------------
    # 1. Verify prediction exists
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 2. Prevent duplicate feedback
    # --------------------------------------------------------

    existing_feedback = (
        repository.get_feedback_for_prediction(
            db,
            request.prediction_id,
        )
    )

    if existing_feedback is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Feedback already exists "
                "for this prediction"
            ),
        )


    # --------------------------------------------------------
    # 3. Store feedback
    # --------------------------------------------------------

    feedback = repository.create_feedback(
        db=db,

        prediction_id=request.prediction_id,

        corrected_decision=(
            request.corrected_decision
        ),

        corrected_category=(
            request.corrected_category
        ),
    )

    return feedback