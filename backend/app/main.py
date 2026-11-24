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
    try:
        init_db(settings.db_path)
        init_llm_cache(settings.cache_dir)
    except PermissionError:
        # Fallback to /tmp if /data is read-only (e.g. serverless environments)
        settings.db_path = "/tmp/app.db"
        settings.cache_dir = "/tmp/cache"
        init_db(settings.db_path)
        init_llm_cache(settings.cache_dir)

    allow_all_origins = "*" in settings.allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all_origins else settings.allowed_origins,
        allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+" if not allow_all_origins else None,
        allow_credentials=not allow_all_origins,
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
except RuntimeError as exc:
    logger.error("App startup failed: %s", exc)
    # Provide a diagnostic app instead of None so ASGI server does not crash with 500 FUNCTION_INVOCATION_FAILED
    app = FastAPI(title="Configuration Error")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def config_error_handler(path: str) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "LLM_API_KEYS not configured in environment variables"},
        )
