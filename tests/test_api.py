from fastapi.testclient import TestClient

from src.api.dependencies import get_moderation_model
from src.api.main import app


class FakeModerationModel:
    """
    Small fake model used only for API tests.

    No PyTorch.
    No Hugging Face.
    No 438 MB model.
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


app.dependency_overrides[get_moderation_model] = (
    override_moderation_model
)


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }

def test_ready():
    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["model"] == "fake-test-model"
    assert data["device"] == "cpu"

def test_safe_comment_allowed():
    response = client.post(
        "/moderate",
        json={
            "text": "Thanks for helping me!"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["decision"] == "allow"


def test_insult_hidden():
    response = client.post(
        "/moderate",
        json={
            "text": "You are an idiot."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["decision"] == "hide"

    assert "insult" in data["triggered_categories"]


def test_empty_comment_rejected():
    response = client.post(
        "/moderate",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422