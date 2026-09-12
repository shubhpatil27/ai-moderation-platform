from src.model.policy import moderate


def test_hides_enabled_high_score_category():

    scores = {
        "insult": 0.95,
        "threat": 0.10,
    }

    preferences = {
        "insult": True,
        "threat": True,
    }

    result = moderate(scores, preferences)

    assert result["decision"] == "hide"
    assert "insult" in result["triggered_categories"]


def test_ignores_disabled_category():

    scores = {
        "insult": 0.99,
    }

    preferences = {
        "insult": False,
    }

    result = moderate(scores, preferences)

    assert result["decision"] == "allow"


def test_allows_safe_content():

    scores = {
        "insult": 0.05,
        "threat": 0.01,
        "toxic": 0.04,
    }

    preferences = {
        "insult": True,
        "threat": True,
        "toxic": True,
    }

    result = moderate(scores, preferences)

    assert result["decision"] == "allow"