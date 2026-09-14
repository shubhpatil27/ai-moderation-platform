from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import (
    Feedback,
    ModerationPreference,
    Prediction,
    User,
)


# ============================================================
# USERS
# ============================================================

def get_user(
    db: Session,
    user_id: int,
) -> User | None:
    return db.get(User, user_id)


def get_user_by_username(
    db: Session,
    username: str,
) -> User | None:

    statement = select(User).where(
        User.username == username
    )

    return db.scalar(statement)


def create_user(
    db: Session,
    username: str,
) -> User:

    user = User(
        username=username
    )

    db.add(user)
    db.flush()

    preferences = ModerationPreference(
        user_id=user.id,
        toxic=True,
        severe_toxic=True,
        obscene=True,
        threat=True,
        insult=True,
        identity_hate=True,
    )

    db.add(preferences)

    db.commit()
    db.refresh(user)

    return user


# ============================================================
# PREFERENCES
# ============================================================

def get_user_preferences(
    db: Session,
    user_id: int,
) -> ModerationPreference | None:

    statement = select(
        ModerationPreference
    ).where(
        ModerationPreference.user_id == user_id
    )

    return db.scalar(statement)


def update_user_preferences(
    db: Session,
    user_id: int,
    data: dict,
) -> ModerationPreference | None:

    preferences = get_user_preferences(
        db,
        user_id,
    )

    if preferences is None:
        return None

    for field, value in data.items():
        setattr(
            preferences,
            field,
            value,
        )

    db.commit()
    db.refresh(preferences)

    return preferences


# ============================================================
# PREDICTIONS
# ============================================================

def create_prediction(
    db: Session,
    user_id: int,
    text: str,
    decision: str,
    scores: dict[str, float],
    model_version: str,
) -> Prediction:

    prediction = Prediction(
        user_id=user_id,
        text=text,
        decision=decision,

        toxic_score=scores.get(
            "toxic",
            0.0,
        ),

        severe_toxic_score=scores.get(
            "severe_toxic",
            0.0,
        ),

        obscene_score=scores.get(
            "obscene",
            0.0,
        ),

        threat_score=scores.get(
            "threat",
            0.0,
        ),

        insult_score=scores.get(
            "insult",
            0.0,
        ),

        identity_hate_score=scores.get(
            "identity_hate",
            0.0,
        ),

        model_version=model_version,
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return prediction


def get_prediction(
    db: Session,
    prediction_id: int,
) -> Prediction | None:

    return db.get(
        Prediction,
        prediction_id,
    )


# ============================================================
# FEEDBACK
# ============================================================

def get_feedback_for_prediction(
    db: Session,
    prediction_id: int,
) -> Feedback | None:

    statement = select(Feedback).where(
        Feedback.prediction_id == prediction_id
    )

    return db.scalar(statement)


def create_feedback(
    db: Session,
    prediction_id: int,
    corrected_decision: str,
    corrected_category: str | None,
) -> Feedback:

    feedback = Feedback(
        prediction_id=prediction_id,
        corrected_decision=corrected_decision,
        corrected_category=corrected_category,
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return feedback