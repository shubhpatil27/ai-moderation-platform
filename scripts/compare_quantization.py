from src.model.onnx_predictor import ONNXModerationModel
from src.model.onnx_int8_predictor import INT8ModerationModel


COMMENTS = [
    "Thanks for helping me!",
    "Great stream today!",
    "You are an idiot.",
    "That was a stupid thing to say.",
    "You're insane at this game!",
]


print("Loading ONNX FP32 model...")
fp32_model = ONNXModerationModel()

print("Loading ONNX INT8 model...")
int8_model = INT8ModerationModel()


for comment in COMMENTS:

    fp32_scores = fp32_model.predict(comment)
    int8_scores = int8_model.predict(comment)

    print("\n" + "=" * 75)
    print(f"COMMENT: {comment}")

    print(
        "\nCATEGORY             "
        "FP32        INT8        DIFFERENCE"
    )

    for category in fp32_scores:

        fp32_score = fp32_scores[category]
        int8_score = int8_scores[category]

        difference = abs(
            fp32_score - int8_score
        )

        print(
            f"{category:20}"
            f"{fp32_score:<12.4f}"
            f"{int8_score:<12.4f}"
            f"{difference:.6f}"
        )
        