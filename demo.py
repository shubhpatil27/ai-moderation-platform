from src.model.predictor import ModerationModel
from src.model.policy import moderate


model = ModerationModel()


preferences = {
    "toxic": True,
    "severe_toxic": True,
    "obscene": False,
    "threat": True,
    "insult": True,
    "identity_hate": True,
}


while True:

    text = input("\nEnter a comment (or 'quit'): ")

    if text.lower() == "quit":
        break

    try:
        scores = model.predict(text)

        result = moderate(
            scores=scores,
            preferences=preferences,
        )

        print("\n--- MODEL SCORES ---")

        for category, score in scores.items():
            print(f"{category:15} {score:.4f}")

        print("\n--- MODERATION DECISION ---")

        print(f"Decision: {result['decision'].upper()}")

        print(
            "Triggered:",
            result["triggered_categories"],
        )

    except (ValueError, TypeError) as error:
        print(f"Error: {error}")