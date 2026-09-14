from typing import Dict, List


DEFAULT_THRESHOLDS = {
    "toxic": 0.80,
    "severe_toxic": 0.70,
    "obscene": 0.80,
    "threat": 0.65,
    "insult": 0.75,
    "identity_hate": 0.70,
}


def moderate(
    scores: Dict[str, float],
    preferences: Dict[str, bool],
    thresholds: Dict[str, float] | None = None,
) -> Dict:

    thresholds = (
        thresholds
        or DEFAULT_THRESHOLDS
    )

    review_categories: List[str] = []
    hide_categories: List[str] = []

    for category, score in scores.items():

        enabled = preferences.get(
            category,
            False,
        )

        if not enabled:
            continue

        threshold = thresholds.get(
            category,
            0.80,
        )

        if score >= threshold:

            hide_categories.append(
                category
            )

        elif score >= threshold * 0.70:

            review_categories.append(
                category
            )


    if hide_categories:

        return {
            "decision": "hide",
            "triggered_categories":
                hide_categories,
        }


    if review_categories:

        return {
            "decision": "review",
            "triggered_categories":
                review_categories,
        }


    return {
        "decision": "allow",
        "triggered_categories": [],
    }