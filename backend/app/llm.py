import logging
from pathlib import Path
from typing import Any
import diskcache
import litellm
from litellm.caching.base_cache import BaseCache
from litellm.caching.caching import Cache

logger = logging.getLogger("chatbot")


class PersistentDiskCacheBackend(BaseCache):
    """DiskCache-backed BaseCache for LiteLLM so cached responses survive process restarts."""

    def __init__(self, disk_cache_dir: str, default_ttl: int = 86400) -> None:
        super().__init__(default_ttl=default_ttl)
        self.disk_cache_dir = disk_cache_dir
        self._store = diskcache.Cache(disk_cache_dir)

    def set_cache(self, key: Any, value: Any, **kwargs: Any) -> None:
        ttl = self.get_ttl(**kwargs)
        self._store.set(str(key), value, expire=ttl)

    async def async_set_cache(self, key: Any, value: Any, **kwargs: Any) -> None:
        self.set_cache(key, value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list: Any, **kwargs: Any) -> None:
        for k, v in cache_list:
            self.set_cache(k, v, **kwargs)

    def get_cache(self, key: Any, **kwargs: Any) -> Any:
        return self._store.get(str(key), default=None)

    async def async_get_cache(self, key: Any, **kwargs: Any) -> Any:
        return self.get_cache(key, **kwargs)

    async def disconnect(self) -> None:
        self._store.close()


class AllProvidersUnavailableError(Exception):
    """Raised when all configured API keys fail."""


RETRYABLE_LITELLM_ERRORS = tuple(
    exc
    for exc in (
        getattr(litellm, "RateLimitError", None),
        getattr(litellm, "ServiceUnavailableError", None),
        getattr(litellm, "InternalServerError", None),
        getattr(litellm, "APIConnectionError", None),
        getattr(litellm, "Timeout", None),
        getattr(litellm, "AuthenticationError", None),
        getattr(litellm, "PermissionDeniedError", None),
        getattr(litellm, "BadGatewayError", None),
    )
    if exc is not None
)


def init_llm_cache(cache_dir: str) -> None:
    """Initialize LiteLLM persistent disk cache."""
    litellm.drop_params = True
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    try:
        # In LiteLLM >= 1.110, _should_defer_streaming_cache_hit_callbacks imports anthropic response_cache
        # which pulls in proxy transcribe_passthrough_logging_handler (requiring soundfile).
        # Since we only make non-streaming completion calls (stream=False), patch it to return False safely.
        import litellm.caching.caching_handler as _ch
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        _ch._should_defer_streaming_cache_hit_callbacks = (  # type: ignore[attr-defined]
            lambda *, cached_result: isinstance(cached_result, CustomStreamWrapper)
        )
    except Exception:
        pass

    try:
        litellm.cache = Cache(type="disk", disk_cache_dir=cache_dir)  # type: ignore[arg-type]
    except ValueError:
        backend = PersistentDiskCacheBackend(disk_cache_dir=cache_dir)
        litellm.cache = Cache(_backend=backend, max_messages=None)


def parse_simulate_header(
    header_value: str | None,
    enable_failure_simulation: bool,
) -> dict[str, int]:
    """
    Parse X-Simulate-Failure header into a dict like {'key_1': 429, 'key_2': 500} or {'*': 429}.
    Returns empty dict if simulation is disabled or header is empty/absent.
    """
    if not enable_failure_simulation or not header_value:
        return {}

    result: dict[str, int] = {}
    parts = [p.strip() for p in header_value.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            status_code = int(v)
            result[k] = status_code
        except ValueError:
            continue
    return result


def make_litellm_error(status: int, model: str) -> Exception:
    """Instantiate a real LiteLLM exception for the given simulated HTTP status code."""
    provider = model.split("/")[0] if "/" in model else "custom"
    if status == 429:
        return litellm.RateLimitError(
            message="Simulated 429 Rate Limit / Quota Exhausted",
            llm_provider=provider,
            model=model,
        )
    elif status == 401:
        return litellm.AuthenticationError(
            message="Simulated 401 Unauthorized",
            llm_provider=provider,
            model=model,
        )
    elif status == 403:
        return litellm.PermissionDeniedError(
            message="Simulated 403 Forbidden",
            llm_provider=provider,
            model=model,
        )
    elif status == 503:
        return litellm.ServiceUnavailableError(
            message="Simulated 503 Service Unavailable",
            llm_provider=provider,
            model=model,
        )
    elif status == 504:
        return litellm.Timeout(
            message="Simulated 504 Gateway Timeout",
            model=model,
            llm_provider=provider,
        )
    else:
        return litellm.InternalServerError(
            message=f"Simulated {status} Internal Server Error",
            llm_provider=provider,
            model=model,
        )


def is_retryable_exception(exc: Exception) -> bool:
    """Check whether an exception should trigger failover to the next key."""
    if isinstance(exc, RETRYABLE_LITELLM_ERRORS):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (
        status_code in (401, 403, 408, 429) or 500 <= status_code <= 599
    ):
        return True
    msg = str(exc).lower()
    retry_keywords = (
        "quota",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "resource_exhausted",
        "resource exhausted",
        "too many requests",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "timeout",
        "timed out",
        "connection error",
        "bad gateway",
        "empty reply",
    )
    if any(kw in msg for kw in retry_keywords):
        return True
    return False


def _extract_reply_text(resp: Any) -> str:
    try:
        content = resp.choices[0].message.content
        if content is None:
            return ""
        return str(content).strip()
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def _extract_usage(resp: Any) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
    else:
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    return in_tok, out_tok


def _is_cache_hit(resp: Any) -> bool:
    hidden = getattr(resp, "_hidden_params", None)
    if isinstance(hidden, dict) and (
        hidden.get("cache_hit") is True or bool(hidden.get("cache_key"))
    ):
        return True
    return False


def complete_with_failover(
    messages: list[dict[str, str]],
    keys: list[str],
    model: str,
    simulate: dict[str, int],
) -> tuple[str, dict[str, Any]]:
    """
    Execute LLM completion across configured keys with failover and disk cache support.
    Returns (reply_text, partial_metadata_dict).
    Raises AllProvidersUnavailableError if all keys fail.
    """
    use_cache = len(simulate) == 0 and litellm.cache is not None
    cache_lookup_params = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if use_cache and litellm.cache is not None:
        try:
            cache_key = litellm.cache.get_cache_key(**cache_lookup_params)
            cached_entry = litellm.cache.cache.get_cache(cache_key)
            if isinstance(cached_entry, dict) and "reply" in cached_entry:
                return str(cached_entry["reply"]), {
                    "model": model,
                    "resolution_source": "LLM_CACHE",
                    "key_id": None,
                    "fallback_attempts": 0,
                    "input_tokens": int(cached_entry.get("input_tokens", 0)),
                    "output_tokens": int(cached_entry.get("output_tokens", 0)),
                }
        except Exception:
            pass

    attempts = 0

    for i, key in enumerate(keys, start=1):
        kid = f"key_{i}"
        try:
            if "*" in simulate:
                raise make_litellm_error(simulate["*"], model)
            if kid in simulate:
                raise make_litellm_error(simulate[kid], model)

            resp = litellm.completion(
                model=model,
                messages=messages,
                api_key=key,
                caching=use_cache,
                temperature=0,
                timeout=30,
            )

            reply_text = _extract_reply_text(resp)
            if not reply_text:
                provider = model.split("/")[0] if "/" in model else "custom"
                raise litellm.InternalServerError(
                    message="Model returned an empty reply",
                    llm_provider=provider,
                    model=model,
                )

            input_tokens, output_tokens = _extract_usage(resp)
            cache_hit = use_cache and _is_cache_hit(resp)

            if use_cache and not cache_hit and litellm.cache is not None:
                try:
                    cache_key = litellm.cache.get_cache_key(**cache_lookup_params)
                    litellm.cache.cache.set_cache(
                        cache_key,
                        {
                            "reply": reply_text,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                    )
                except Exception:
                    pass

            if cache_hit:
                meta = {
                    "model": model,
                    "resolution_source": "LLM_CACHE",
                    "key_id": None,
                    "fallback_attempts": 0,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            else:
                meta = {
                    "model": model,
                    "resolution_source": "LIVE_INFERENCE",
                    "key_id": kid,
                    "fallback_attempts": attempts,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            return reply_text, meta

        except Exception as exc:
            if isinstance(exc, getattr(litellm, "NotFoundError", ())):
                logger.error(
                    "Model/endpoint NotFoundError on key_%d (check LLM_MODEL=%s): %s",
                    i,
                    model,
                    str(exc)[:300],
                )
            else:
                logger.warning(
                    "key_%d failed: %s: %s",
                    i,
                    type(exc).__name__,
                    str(exc)[:300],
                )
            attempts += 1
            continue

    raise AllProvidersUnavailableError("all providers unavailable")
