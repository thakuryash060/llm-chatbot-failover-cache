import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
import litellm

from app.llm import init_llm_cache
from app.main import create_app


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeCompletionResponse:
    def __init__(
        self,
        content: str = "Paris is the capital of France.",
        prompt_tokens: int = 18,
        completion_tokens: int = 9,
        cache_hit: bool = False,
    ) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)
        self._hidden_params = {"cache_hit": cache_hit}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_file = tmp_path / "test_app.db"
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("LLM_API_KEYS", "test_key_1,test_key_2,test_key_3")
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")

    app = create_app()
    with TestClient(app) as tc:
        yield tc


def test_health_shape(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "status": "ok",
        "keys_configured": 3,
        "cache_enabled": True,
    }


def test_health_503_when_db_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "missing_dir" / "uninitialized.db"
    monkeypatch.setenv("LLM_API_KEYS", "k1,k2")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    app = create_app()
    with TestClient(app) as tc:
        # Delete the db file after startup init to simulate DB becoming unreachable
        db_file.unlink()
        resp = tc.get("/health")
        assert resp.status_code == 503
        assert "detail" in resp.json()


def test_validation_422_cases(client: TestClient) -> None:
    # Empty message
    r1 = client.post("/chat", json={"session_id": "s-1", "message": ""})
    assert r1.status_code == 422

    # Whitespace-only message
    r2 = client.post("/chat", json={"session_id": "s-1", "message": "   \n\t  "})
    assert r2.status_code == 422

    # 4001 chars message
    r3 = client.post("/chat", json={"session_id": "s-1", "message": "a" * 4001})
    assert r3.status_code == 422

    # Missing session_id
    r4 = client.post("/chat", json={"message": "hello"})
    assert r4.status_code == 422

    # Empty session_id
    r5 = client.post("/chat", json={"session_id": "   ", "message": "hello"})
    assert r5.status_code == 422

    # Malformed JSON
    r6 = client.post(
        "/chat",
        content="{bad json",
        headers={"Content-Type": "application/json"},
    )
    assert r6.status_code == 422


def test_happy_path_all_fields_and_types(client: TestClient) -> None:
    with patch("app.llm.litellm.completion") as mock_comp:
        mock_comp.return_value = FakeCompletionResponse(
            content="Paris.", prompt_tokens=15, completion_tokens=4, cache_hit=False
        )
        resp = client.post(
            "/chat",
            json={"session_id": "s-1", "message": "  What is the capital of France?  "},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "s-1"
        assert body["reply"] == "Paris."

        meta = body["metadata"]
        assert meta["model"] == "gemini/gemini-2.5-flash"
        assert meta["resolution_source"] == "LIVE_INFERENCE"
        assert meta["key_id"] == "key_1"
        assert meta["fallback_attempts"] == 0
        assert isinstance(meta["latency_ms"], int)
        assert meta["latency_ms"] >= 0
        assert meta["input_tokens"] == 15
        assert meta["output_tokens"] == 4

        # Verify stripped message was sent to model
        called_messages = mock_comp.call_args.kwargs["messages"]
        assert called_messages[-1] == {
            "role": "user",
            "content": "What is the capital of France?",
        }


def test_multi_turn_and_session_isolation(client: TestClient) -> None:
    captured_calls: list[list[dict[str, str]]] = []

    def side_effect(**kwargs: Any) -> FakeCompletionResponse:
        msgs = kwargs["messages"]
        captured_calls.append(list(msgs))
        turn_num = len(captured_calls)
        return FakeCompletionResponse(
            content=f"Reply {turn_num}",
            prompt_tokens=10 * turn_num,
            completion_tokens=5,
        )

    with patch("app.llm.litellm.completion", side_effect=side_effect):
        # Turn 1 in s-alpha
        r1 = client.post(
            "/chat",
            json={"session_id": "s-alpha", "message": "First question"},
        )
        assert r1.status_code == 200

        # Turn 2 in s-alpha
        r2 = client.post(
            "/chat",
            json={"session_id": "s-alpha", "message": "Follow up question"},
        )
        assert r2.status_code == 200

        # Turn 1 in s-beta (must be isolated from s-alpha)
        r3 = client.post(
            "/chat",
            json={"session_id": "s-beta", "message": "Separate session question"},
        )
        assert r3.status_code == 200

    assert len(captured_calls) == 3

    # First call: system + user1
    assert [m["role"] for m in captured_calls[0]] == ["system", "user"]
    assert captured_calls[0][1]["content"] == "First question"

    # Second call in s-alpha: system + user1 + assistant1 + user2
    assert [m["role"] for m in captured_calls[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert captured_calls[1][1]["content"] == "First question"
    assert captured_calls[1][2]["content"] == "Reply 1"
    assert captured_calls[1][3]["content"] == "Follow up question"

    # First call in s-beta: only system + user
    assert [m["role"] for m in captured_calls[2]] == ["system", "user"]
    assert captured_calls[2][1]["content"] == "Separate session question"


def test_history_endpoint_order_no_system_prompt_and_404(client: TestClient) -> None:
    with patch("app.llm.litellm.completion") as mock_comp:
        mock_comp.return_value = FakeCompletionResponse(
            content="Hello back!", prompt_tokens=12, completion_tokens=6
        )
        client.post(
            "/chat",
            json={
                "session_id": "s-hist",
                "message": "Hi there",
                "system_prompt": "Custom prompt",
            },
        )

    hist_resp = client.get("/sessions/s-hist/messages")
    assert hist_resp.status_code == 200
    data = hist_resp.json()
    messages = data["messages"] if isinstance(data, dict) and "messages" in data else data
    assert len(messages) == 2
    assert all(m["role"] != "system" for m in messages)

    assert messages[0] == {"role": "user", "content": "Hi there"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hello back!"
    assert "metadata" in messages[1]
    assert messages[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert messages[1]["metadata"]["input_tokens"] == 12
    assert messages[1]["metadata"]["output_tokens"] == 6

    # Unknown session -> 404
    not_found = client.get("/sessions/unknown-session-id/messages")
    assert not_found.status_code == 404
    assert not_found.json() == {"detail": "session not found"}


def test_simulated_failover_key1_and_key2(client: TestClient) -> None:
    with patch("app.llm.litellm.completion") as mock_comp:
        mock_comp.return_value = FakeCompletionResponse(content="From backup key")

        # key_1=429 -> key_2 with fallback_attempts=1
        r1 = client.post(
            "/chat",
            json={"session_id": "s-fail-1", "message": "Test failover 1"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert r1.status_code == 200
        meta1 = r1.json()["metadata"]
        assert meta1["key_id"] == "key_2"
        assert meta1["fallback_attempts"] == 1
        assert meta1["resolution_source"] == "LIVE_INFERENCE"

        # key_1=429,key_2=500 -> key_3 with fallback_attempts=2
        r2 = client.post(
            "/chat",
            json={"session_id": "s-fail-2", "message": "Test failover 2"},
            headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
        )
        assert r2.status_code == 200
        meta2 = r2.json()["metadata"]
        assert meta2["key_id"] == "key_3"
        assert meta2["fallback_attempts"] == 2


def test_all_fail_gives_503_and_saves_nothing(client: TestClient) -> None:
    with patch("app.llm.litellm.completion") as mock_comp:
        mock_comp.return_value = FakeCompletionResponse(content="Should not reach")
        resp = client.post(
            "/chat",
            json={"session_id": "s-all-fail", "message": "Hello"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert resp.status_code == 503
        assert resp.json() == {"detail": "all providers unavailable"}
        assert mock_comp.call_count == 0

    # Verify nothing was saved in DB
    hist = client.get("/sessions/s-all-fail/messages")
    assert hist.status_code == 404


def test_simulation_header_ignored_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "sim_disabled.db"
    cache_dir = tmp_path / "cache_disabled"
    monkeypatch.setenv("LLM_API_KEYS", "k1,k2")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "false")

    app = create_app()
    with TestClient(app) as tc:
        with patch("app.llm.litellm.completion") as mock_comp:
            mock_comp.return_value = FakeCompletionResponse(content="Normal response")
            resp = tc.post(
                "/chat",
                json={"session_id": "s-sim-off", "message": "Hello"},
                headers={"X-Simulate-Failure": "*=429"},
            )
            assert resp.status_code == 200
            assert resp.json()["metadata"]["key_id"] == "key_1"
            assert resp.json()["metadata"]["fallback_attempts"] == 0


def test_cache_hit_and_miss(client: TestClient) -> None:
    seen_payloads: dict[str, FakeCompletionResponse] = {}

    def mock_completion_with_cache(**kwargs: Any) -> FakeCompletionResponse:
        key = json.dumps(kwargs["messages"], sort_keys=True)
        use_cache = kwargs.get("caching", False)
        if use_cache and key in seen_payloads:
            orig = seen_payloads[key]
            return FakeCompletionResponse(
                content=orig.choices[0].message.content,
                prompt_tokens=orig.usage.prompt_tokens,
                completion_tokens=orig.usage.completion_tokens,
                cache_hit=True,
            )
        resp = FakeCompletionResponse(
            content=f"Answer for {kwargs['messages'][-1]['content']}",
            prompt_tokens=22,
            completion_tokens=11,
            cache_hit=False,
        )
        if use_cache:
            seen_payloads[key] = resp
        return resp

    with patch("app.llm.litellm.completion", side_effect=mock_completion_with_cache):
        # First call in s-1 -> LIVE_INFERENCE
        r1 = client.post(
            "/chat",
            json={"session_id": "s-1", "message": "What is the capital of France?"},
        )
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert b1["metadata"]["key_id"] == "key_1"

        # Same message in new session s-2 -> LLM_CACHE, identical reply, key_id=None
        r2 = client.post(
            "/chat",
            json={"session_id": "s-2", "message": "What is the capital of France?"},
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["reply"] == b1["reply"]
        assert b2["metadata"]["resolution_source"] == "LLM_CACHE"
        assert b2["metadata"]["key_id"] is None
        assert b2["metadata"]["fallback_attempts"] == 0
        assert b2["metadata"]["input_tokens"] == 22
        assert b2["metadata"]["output_tokens"] == 11

        # Different message in s-3 -> LIVE_INFERENCE
        r3 = client.post(
            "/chat",
            json={"session_id": "s-3", "message": "What is the capital of Spain?"},
        )
        assert r3.status_code == 200
        b3 = r3.json()
        assert b3["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert b3["metadata"]["key_id"] == "key_1"


def test_real_error_failover(client: TestClient) -> None:
    def fail_on_first_key(**kwargs: Any) -> FakeCompletionResponse:
        if kwargs.get("api_key") == "test_key_1":
            raise litellm.RateLimitError(
                message="Quota exceeded on key 1",
                llm_provider="gemini",
                model="gemini/gemini-2.5-flash",
            )
        return FakeCompletionResponse(content="Recovered via key 2")

    with patch("app.llm.litellm.completion", side_effect=fail_on_first_key):
        resp = client.post(
            "/chat",
            json={"session_id": "s-real-err", "message": "Test real rate limit"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["reply"] == "Recovered via key 2"
        assert body["metadata"]["key_id"] == "key_2"
        assert body["metadata"]["fallback_attempts"] == 1


def test_litellm_cache_key_excludes_api_key_and_persists_to_disk(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "disk_cache_verify"
    init_llm_cache(str(cache_dir))
    assert litellm.cache is not None

    messages = [
        {"role": "system", "content": "You are a helpful, concise AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    k1 = litellm.cache.get_cache_key(
        model="gemini/gemini-2.5-flash",
        messages=messages,
        api_key="secret_key_1",
        temperature=0,
    )
    k2 = litellm.cache.get_cache_key(
        model="gemini/gemini-2.5-flash",
        messages=messages,
        api_key="secret_key_2",
        temperature=0,
    )
    assert k1 == k2

    # Verify disk persistence across re-initialization
    litellm.cache.cache.set_cache(k1, {"test": "persisted_value"})
    init_llm_cache(str(cache_dir))
    assert litellm.cache.cache.get_cache(k2) == {"test": "persisted_value"}

