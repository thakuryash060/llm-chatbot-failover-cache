import asyncio
import time
from typing import Any
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
import litellm

from app.db import (
    check_db_health,
    get_history,
    get_session_messages,
    get_session_prompt,
    save_turn,
)
from app.llm import (
    AllProvidersUnavailableError,
    complete_with_failover,
    parse_simulate_header,
)
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ResponseMetadata

router = APIRouter()

_session_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()


async def get_session_lock(session_id: str) -> asyncio.Lock:
    async with _locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _session_locks[session_id] = lock
        return lock


@router.get("/")
async def root_info() -> Any:
    return {
        "service": "LLM Chatbot Backend",
        "status": "ok",
        "endpoints": ["/health", "/chat", "/sessions/{session_id}/messages"],
    }


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> Any:
    settings = request.app.state.settings
    try:
        db_ok = await run_in_threadpool(check_db_health, settings.db_path)
    except Exception:
        db_ok = False

    if not db_ok:
        raise HTTPException(status_code=503, detail="database unreachable")

    return {
        "status": "ok",
        "keys_configured": len(settings.llm_api_keys),
        "cache_enabled": litellm.cache is not None,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    x_simulate_failure: str | None = Header(default=None, alias="X-Simulate-Failure"),
) -> Any:
    t0 = time.perf_counter()
    settings = request.app.state.settings

    simulate = parse_simulate_header(
        x_simulate_failure,
        settings.enable_failure_simulation,
    )

    session_lock = await get_session_lock(payload.session_id)
    async with session_lock:
        stored_prompt = await run_in_threadpool(
            get_session_prompt, settings.db_path, payload.session_id
        )
        effective_system_prompt = (
            payload.system_prompt
            if payload.system_prompt is not None
            else (stored_prompt or settings.default_system_prompt)
        )

        history = await run_in_threadpool(
            get_history, settings.db_path, payload.session_id
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": effective_system_prompt},
            *history,
            {"role": "user", "content": payload.message},
        ]

        try:
            reply_text, partial_meta = await run_in_threadpool(
                complete_with_failover,
                messages,
                settings.llm_api_keys,
                settings.llm_model,
                simulate,
            )
        except AllProvidersUnavailableError:
            raise HTTPException(status_code=503, detail="all providers unavailable")

        latency_ms = max(0, int((time.perf_counter() - t0) * 1000))
        full_metadata = {
            **partial_meta,
            "latency_ms": latency_ms,
        }

        await run_in_threadpool(
            save_turn,
            settings.db_path,
            payload.session_id,
            payload.message,
            reply_text,
            full_metadata,
            payload.system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=reply_text,
            metadata=ResponseMetadata(**full_metadata),
        )


@router.get("/sessions/{session_id}/messages")
async def session_messages_endpoint(session_id: str, request: Request) -> Any:
    settings = request.app.state.settings
    messages = await run_in_threadpool(
        get_session_messages, settings.db_path, session_id
    )
    if not messages:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session_id,
        "messages": messages,
    }
