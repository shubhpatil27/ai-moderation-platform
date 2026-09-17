import time

from fastapi import Request
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response


# ============================================================
# HTTP METRICS
# ============================================================

HTTP_REQUESTS_TOTAL = Counter(
    "moderation_http_requests_total",
    "Total number of HTTP requests processed by the API.",
    [
        "method",
        "endpoint",
        "status_code",
    ],
)


HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "moderation_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    [
        "method",
        "endpoint",
    ],
)


# ============================================================
# MODEL METRICS
# ============================================================

MODEL_INFERENCE_TOTAL = Counter(
    "moderation_model_inference_total",
    "Total number of model inference operations.",
    [
        "runtime",
    ],
)


MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    "moderation_model_inference_duration_seconds",
    "Model inference duration in seconds.",
    [
        "runtime",
    ],
)


# ============================================================
# MODERATION DECISION METRICS
# ============================================================

MODERATION_DECISIONS_TOTAL = Counter(
    "moderation_decisions_total",
    "Total number of moderation decisions.",
    [
        "decision",
    ],
)


MODERATION_CATEGORY_FLAGS_TOTAL = Counter(
    "moderation_category_flags_total",
    "Number of times a toxicity category crossed its policy threshold.",
    [
        "category",
    ],
)


# ============================================================
# FEEDBACK METRICS
# ============================================================

MODERATION_FEEDBACK_TOTAL = Counter(
    "moderation_feedback_total",
    "Total number of human feedback records submitted.",
)


# ============================================================
# HTTP MIDDLEWARE
# ============================================================

async def prometheus_middleware(
    request: Request,
    call_next,
):
    """
    Measure request count and latency for every API request.

    Route templates are used when available so IDs do not create
    unbounded Prometheus label cardinality.
    """

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        duration = (
            time.perf_counter()
            - start_time
        )

        endpoint = _get_endpoint(
            request
        )

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code="500",
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        raise


    duration = (
        time.perf_counter()
        - start_time
    )

    endpoint = _get_endpoint(
        request
    )


    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=str(
            response.status_code
        ),
    ).inc()


    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(duration)


    return response


def _get_endpoint(
    request: Request,
) -> str:
    """
    Prefer FastAPI's route template over the raw path.

    Example:

    /users/123/preferences

    becomes:

    /users/{user_id}/preferences

    This prevents a separate Prometheus time series for every user ID.
    """

    route = request.scope.get(
        "route"
    )

    if route is not None:

        path = getattr(
            route,
            "path",
            None,
        )

        if path:
            return path


    return request.url.path


# ============================================================
# METRICS ENDPOINT
# ============================================================

def metrics_response() -> Response:
    """
    Return all registered Prometheus metrics.
    """

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )