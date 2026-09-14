import pytest

from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_moderation_model
from src.api.main import app

from src.database.db import (
    Base,
    get_db,
)

# Import models so SQLAlchemy knows all tables.
from src.database import models

from src.database.models import (
    ModerationPreference,
    User,
)


# ============================================================
# FAKE ML MODEL
# ============================================================

class FakeModerationModel:
    """
    Lightweight predictable model for API tests.

    We do NOT want to load Toxic-BERT during normal API tests.
    """

    model_name = "fake-test-model"
    device = "cpu"

    def predict(self, text: str):

        if "idiot" in text.lower():

            return {
                "toxic": 0.95,
                "severe_toxic": 0.10,
                "obscene": 0.20,
                "threat": 0.05,
                "insult": 0.96,
                "identity_hate": 0.01,
            }

        return {
            "toxic": 0.02,
            "severe_toxic": 0.01,
            "obscene": 0.01,
            "threat": 0.01,
            "insult": 0.02,
            "identity_hate": 0.01,
        }


def override_moderation_model():
    return FakeModerationModel()


# ============================================================
# TEST DATABASE
# ============================================================

TEST_DATABASE_URL = "sqlite://"


test_engine = create_engine(
    TEST_DATABASE_URL,

    connect_args={
        "check_same_thread": False,
    },

    poolclass=StaticPool,
)


TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    expire_on_commit=False,
)


def override_get_db():

    db = TestingSessionLocal()

    try:
        yield db

    finally:
        db.close()


# ============================================================
# DEPENDENCY OVERRIDES
# ============================================================

app.dependency_overrides[
    get_moderation_model
] = override_moderation_model


app.dependency_overrides[
    get_db
] = override_get_db


client = TestClient(app)


# ============================================================
# DATABASE SETUP / CLEANUP
# ============================================================

@pytest.fixture(autouse=True)
def setup_database():

    # Create clean tables before every test.
    Base.metadata.create_all(
        bind=test_engine
    )

    db = TestingSessionLocal()

    # Every moderation test can safely use user_id = 1.
    user = User(
        username="test_user"
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
    db.close()

    yield

    # Completely clean test DB after test.
    Base.metadata.drop_all(
        bind=test_engine
    )


# ============================================================
# BASIC API TESTS
# ============================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message":
            "AI Moderation Platform is running"
    }


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


def test_ready():

    response = client.get(
        "/ready"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"

    assert (
        data["model"]
        == "fake-test-model"
    )

    assert data["device"] == "cpu"


# ============================================================
# MODERATION TESTS
# ============================================================

def test_safe_comment_allowed():

    response = client.post(
        "/moderate",

        json={
            "user_id": 1,
            "text":
                "Thanks for helping me!",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["decision"]
        == "allow"
    )

    assert (
        data["model_version"]
        == "fake-test-model"
    )

    assert (
        data["prediction_id"]
        > 0
    )


def test_insult_hidden():

    response = client.post(
        "/moderate",

        json={
            "user_id": 1,
            "text":
                "You are an idiot.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["decision"]
        == "hide"
    )

    assert (
        "insult"
        in data[
            "triggered_categories"
        ]
    )


# ============================================================
# VALIDATION TEST
# ============================================================

def test_empty_comment_rejected():

    response = client.post(
        "/moderate",

        json={
            "user_id": 1,
            "text": "",
        },
    )

    assert (
        response.status_code
        == 422
    )


# ============================================================
# UNKNOWN USER
# ============================================================

def test_unknown_user_rejected():

    response = client.post(
        "/moderate",

        json={
            "user_id": 999,
            "text": "Hello!",
        },
    )

    assert (
        response.status_code
        == 404
    )


# ============================================================
# USER CREATION
# ============================================================

def test_create_user():

    response = client.post(
        "/users",

        json={
            "username": "alice"
        },
    )

    assert (
        response.status_code
        == 201
    )

    data = response.json()

    assert (
        data["username"]
        == "alice"
    )


# ============================================================
# PREFERENCES
# ============================================================

def test_get_preferences():

    response = client.get(
        "/users/1/preferences"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["insult"]
        is True
    )


def test_update_preferences():

    response = client.put(
        "/users/1/preferences",

        json={
            "toxic": True,
            "severe_toxic": True,
            "obscene": False,
            "threat": True,
            "insult": False,
            "identity_hate": True,
        },
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["obscene"]
        is False
    )

    assert (
        data["insult"]
        is False
    )