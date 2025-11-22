# en api/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .config import API_TITLE, API_VERSION
from .routes import api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
    )

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    app.include_router(api_router)
    return app


app = create_app()
