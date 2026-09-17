from fastapi import FastAPI

from src.api.routes import router
from src.monitoring.metrics import (
    metrics_response,
    prometheus_middleware,
)


app = FastAPI(
    title="AI Moderation Platform",
    description=(
        "Personalized AI content moderation "
        "with ONNX Runtime and MLOps."
    ),
    version="1.0.0",
)


# ============================================================
# PROMETHEUS MIDDLEWARE
# ============================================================

app.middleware("http")(
    prometheus_middleware
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AI Moderation Platform is running"
    }


# ============================================================
# APPLICATION ROUTES
# ============================================================

app.include_router(
    router
)


# ============================================================
# PROMETHEUS METRICS
# ============================================================

@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics():
    return metrics_response()