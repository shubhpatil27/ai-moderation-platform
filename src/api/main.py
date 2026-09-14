from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="AI Moderation Platform",
    description=(
        "Personalized real-time "
        "AI content moderation service."
    ),
    version="1.0.0",
)


app.include_router(router)


@app.get("/")
def root():

    return {
        "message":
            "AI Moderation Platform is running"
    }