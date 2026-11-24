import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.db import init_db
from app.llm import init_llm_cache
from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    app.state.settings = settings
    init_db(settings.db_path)
    init_llm_cache(settings.cache_dir)
    logger.info(
        "Started backend with %d API keys configured, model=%s, simulation=%s",
        len(settings.llm_api_keys),
        settings.llm_model,
        settings.enable_failure_simulation,
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="LLM Chatbot with Key Failover and Response Caching",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    init_db(settings.db_path)
    init_llm_cache(settings.cache_dir)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Keep FastAPI's standard 422 JSON shape while ensuring non-serializable ctx objects don't break
        clean_errors = []
        for err in exc.errors():
            item = {
                "type": err.get("type", "validation_error"),
                "loc": err.get("loc", []),
                "msg": err.get("msg", "Invalid value"),
            }
            if "input" in err and isinstance(
                err["input"], (str, int, float, bool, list, dict, type(None))
            ):
                item["input"] = err["input"]
            clean_errors.append(item)
        return JSONResponse(
            status_code=422,
            content={"detail": clean_errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled server exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error"},
        )

    app.include_router(router)
    return app


try:
    app = create_app()
except RuntimeError:
    # Allow importing main module during tests before env vars are monkeypatched if needed
    app = None  # type: ignore[assignment]
