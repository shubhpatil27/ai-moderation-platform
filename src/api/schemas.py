from typing import Dict, List, Literal

from pydantic import BaseModel, Field


ModerationDecision = Literal[
    "allow",
    "review",
    "hide",
]

ModerationCategory = Literal[
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


# ============================================================
# USERS
# ============================================================

class UserCreate(BaseModel):
    username: str = Field(
        ...,
        min_length=3,
        max_length=100,
    )


class UserResponse(BaseModel):
    id: int
    username: str

    model_config = {
        "from_attributes": True
    }


# ============================================================
# PREFERENCES
# ============================================================

class PreferencesResponse(BaseModel):
    user_id: int

    toxic: bool
    severe_toxic: bool
    obscene: bool
    threat: bool
    insult: bool
    identity_hate: bool

    model_config = {
        "from_attributes": True
    }


class PreferencesUpdate(BaseModel):
    toxic: bool
    severe_toxic: bool
    obscene: bool
    threat: bool
    insult: bool
    identity_hate: bool


# ============================================================
# MODERATION
# ============================================================

class ModerationRequest(BaseModel):
    user_id: int = Field(
        ...,
        gt=0,
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class ModerationResponse(BaseModel):
    prediction_id: int

    decision: ModerationDecision

    triggered_categories: List[str]

    scores: Dict[str, float]

    model_version: str


# ============================================================
# PREDICTIONS
# ============================================================

class PredictionResponse(BaseModel):
    id: int
    user_id: int
    text: str

    decision: ModerationDecision

    toxic_score: float
    severe_toxic_score: float
    obscene_score: float
    threat_score: float
    insult_score: float
    identity_hate_score: float

    model_version: str

    model_config = {
        "from_attributes": True
    }


# ============================================================
# FEEDBACK
# ============================================================

class FeedbackCreate(BaseModel):
    prediction_id: int = Field(
        ...,
        gt=0,
    )

    corrected_decision: ModerationDecision

    corrected_category: (
        ModerationCategory | None
    ) = None


class FeedbackResponse(BaseModel):
    id: int
    prediction_id: int

    corrected_decision: ModerationDecision

    corrected_category: str | None

    model_config = {
        "from_attributes": True
    }