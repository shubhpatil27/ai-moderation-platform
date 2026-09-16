from pathlib import Path

from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
)


INPUT_MODEL = Path(
    "models/onnx/model.onnx"
)

OUTPUT_DIR = Path(
    "models/onnx_int8"
)

OUTPUT_MODEL = (
    OUTPUT_DIR / "model_quantized.onnx"
)


def main():

    if not INPUT_MODEL.exists():
        raise FileNotFoundError(
            f"FP32 model not found: {INPUT_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Quantizing ONNX model...")
    print(f"Input : {INPUT_MODEL}")
    print(f"Output: {OUTPUT_MODEL}")

    quantize_dynamic(
        model_input=str(INPUT_MODEL),
        model_output=str(OUTPUT_MODEL),
        weight_type=QuantType.QInt8,
    )

    fp32_size = (
        INPUT_MODEL.stat().st_size
        / (1024 ** 2)
    )

    int8_size = (
        OUTPUT_MODEL.stat().st_size
        / (1024 ** 2)
    )

    reduction = (
        (fp32_size - int8_size)
        / fp32_size
    ) * 100

    print("\nQuantization complete.")
    print(
        f"FP32 size : {fp32_size:.2f} MB"
    )
    print(
        f"INT8 size : {int8_size:.2f} MB"
    )
    print(
        f"Reduction : {reduction:.2f}%"
    )


if __name__ == "__main__":
    main()