from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.db import Base


def utc_now():
    return datetime.now(timezone.utc)


# ============================================================
# USER
# ============================================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    preferences = relationship(
        "ModerationPreference",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    predictions = relationship(
        "Prediction",
        back_populates="user",
        cascade="all, delete-orphan",
    )


# ============================================================
# MODERATION PREFERENCES
# ============================================================

class ModerationPreference(Base):
    __tablename__ = "moderation_preferences"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        unique=True,
        nullable=False,
    )

    toxic: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    severe_toxic: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    obscene: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    threat: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    insult: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    identity_hate: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="preferences",
    )


# ============================================================
# PREDICTION
# ============================================================

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    decision: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    toxic_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    severe_toxic_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    obscene_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    threat_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    insult_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    identity_hate_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="predictions",
    )

    feedback = relationship(
        "Feedback",
        back_populates="prediction",
        uselist=False,
        cascade="all, delete-orphan",
    )


# ============================================================
# FEEDBACK
# ============================================================

class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    prediction_id: Mapped[int] = mapped_column(
        ForeignKey(
            "predictions.id",
            ondelete="CASCADE",
        ),
        unique=True,
        nullable=False,
    )

    corrected_decision: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    corrected_category: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    prediction = relationship(
        "Prediction",
        back_populates="feedback",
    )