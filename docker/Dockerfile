FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src

# Model artifacts are mounted at runtime.
# They are not committed to Git or baked into this image.
RUN mkdir -p /app/models/onnx /app/models/onnx_int8

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]