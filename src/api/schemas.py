from typing import Dict, List

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    toxic: bool = True
    severe_toxic: bool = True
    obscene: bool = True
    threat: bool = True
    insult: bool = True
    identity_hate: bool = True


class ModerationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    preferences: UserPreferences = Field(
        default_factory=UserPreferences
    )


class ModerationResponse(BaseModel):
    decision: str
    triggered_categories: List[str]
    scores: Dict[str, float]
    model_version: str