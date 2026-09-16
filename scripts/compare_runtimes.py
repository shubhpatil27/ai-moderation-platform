from src.model.predictor import ModerationModel
from src.model.onnx_predictor import ONNXModerationModel


COMMENTS = [
    "Thanks for helping me!",
    "Great stream today!",
    "You are an idiot.",
    "That was a stupid thing to say.",
    "You're insane at this game!",
]


print("Loading PyTorch model...")
pytorch_model = ModerationModel()

print("Loading ONNX model...")
onnx_model = ONNXModerationModel()


for comment in COMMENTS:

    pytorch_scores = pytorch_model.predict(
        comment
    )

    onnx_scores = onnx_model.predict(
        comment
    )

    print("\n" + "=" * 70)

    print(f"COMMENT: {comment}")

    print("\nCATEGORY             PYTORCH      ONNX      DIFFERENCE")

    for category in pytorch_scores:

        pytorch_score = pytorch_scores[
            category
        ]

        onnx_score = onnx_scores[
            category
        ]

        difference = abs(
            pytorch_score - onnx_score
        )

        print(
            f"{category:20}"
            f"{pytorch_score:<13.4f}"
            f"{onnx_score:<10.4f}"
            f"{difference:.6f}"
        )