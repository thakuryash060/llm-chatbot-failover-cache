# LLM Chatbot with Key Failover and Response Caching

Full-stack AI chatbot application built with a **React + TypeScript (Vite)** frontend, **FastAPI** backend, **LiteLLM** gateway with persistent **DiskCache**, and **SQLite** session storage.

---

## 1. Live URLs

- **Frontend**: https://llm-chatbot-failover-cache-29ak.vercel.app
- **Backend**: https://llm-chatbot-failover-cache.vercel.app
- **Health Check (`/health`)**: https://llm-chatbot-failover-cache.vercel.app/health

---

## 2. Run locally

### With Docker

```bash
cp .env.example .env
# Edit .env and add your real API keys to LLM_API_KEYS=key1,key2
make up
```

- Frontend is served at `http://localhost:3000`
- Backend API is served at `http://localhost:8000`
- View logs: `make logs`
- Stop containers: `make down`

### Without Docker

**Backend:**

```bash
cd backend
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Set environment variables (or create a .env file in project root / backend)
export LLM_API_KEYS="your_key_1,your_key_2"
export LLM_MODEL="gemini/gemini-3.5-flash-lite"
export DATABASE_URL="sqlite:///./data/app.db"
export CACHE_DIR="./data/cache"
export ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
export ENABLE_FAILURE_SIMULATION="true"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**

```bash
cd frontend
npm install
export VITE_API_URL="http://localhost:8000"
npm run dev
```

---

## 3. Design decisions

### Key fallback

- **Custom Failover Loop**: Rather than using LiteLLM's `Router` (which abstracts away which specific key answered and how many fallback attempts occurred), `complete_with_failover` in `backend/app/llm.py` iterates directly over the configured `LLM_API_KEYS` pool (`key_1`, `key_2`, `key_3`, ...). Raw API keys are never logged or returned in API responses.
- **Triggering Errors**: Failover triggers automatically on `RateLimitError` (429), quota-exhausted errors (429 or messages matching `"quota"` / `"resource_exhausted"`), `ServiceUnavailableError` / `InternalServerError` / `BadGatewayError` / `APIConnectionError` / `Timeout` (5xx), `AuthenticationError` (401), `PermissionDeniedError` (403), and empty model replies.
- **Failure Simulation**: When `ENABLE_FAILURE_SIMULATION=true`, the `X-Simulate-Failure` header is parsed into a target dictionary (e.g. `{"key_1": 429, "key_2": 500}` or `{"*": 429}`). Inside the failover loop, real LiteLLM exception classes (`litellm.RateLimitError`, `litellm.InternalServerError`, etc.) are raised so the exact same `except` branch handles both real and simulated provider failures.
- **Cache Bypass on Simulation**: Whenever a valid `X-Simulate-Failure` directive is active, the backend passes `caching=False` to `litellm.completion`. This ensures that sending `X-Simulate-Failure: key_1=429` on an already-cached prompt genuinely demonstrates failover to `key_2` instead of short-circuiting from the disk cache.

### Caching

- **LiteLLM Disk Cache**: Initialized at startup via `litellm.cache = Cache(type="disk", disk_cache_dir=CACHE_DIR)` backed by `diskcache` on the persistent `/data/cache` volume so cached responses survive container restarts.
- **Cache Key & Why `session_id` is Excluded**: The LLM payload is built strictly from `[{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": stripped_message}]` with `temperature=0`. Neither `session_id`, timestamps, current dates, nor any per-session random identifiers are injected into the messages list or system prompt (injecting a live date/time into the system prompt would invalidate cache keys across day/minute boundaries), and LiteLLM's cache key does not include `api_key`. Consequently, asking the same initial question in a brand-new session hits the cache deterministically.
- **What a Cache Hit Returns**: Cache hits are detected via `resp._hidden_params.get("cache_hit") == True`. On a hit, the API returns `resolution_source: "LLM_CACHE"`, `key_id: null`, `fallback_attempts: 0`, the cached `input_tokens` and `output_tokens` for that request, and a significantly lower `latency_ms`.

### SQLite session storage

- **Schema**: Two tables created on startup (`CREATE TABLE IF NOT EXISTS`):
  - `sessions (session_id TEXT PRIMARY KEY, system_prompt TEXT NULL, created_at TEXT NOT NULL)`
  - `messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NULL, created_at TEXT NOT NULL)` with composite index `idx_messages_session_id_id ON messages (session_id, id)`.
- **WAL & Concurrency**: Every connection enables `PRAGMA journal_mode=WAL;` for concurrent read/write performance. In addition, an in-memory per-session `asyncio.Lock` serializes turns within the same session so rapid requests cannot interleave conversation history.
- **Persistent Volume**: Mounted at `/data` (`DATABASE_URL=sqlite:////data/app.db`) in Docker and production deployments so conversation history and disk cache persist across restarts.
- **Atomic Transactions & Failed Requests Not Stored**: `save_turn` writes the session upsert, user message row, and assistant message row (with JSON-serialized `metadata` and UTC ISO-8601 `created_at`) in a single SQLite transaction **only after** a successful LLM completion. Any request failing validation (422) or exhausting all keys (503) leaves zero rows in the database.

---

## 4. Verification

```bash
B=https://llm-chatbot-failover-cache.vercel.app
curl $B/health

curl -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-1","message":"What is the capital of France?"}'

# same message, new session -> LLM_CACHE
curl -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-2","message":"What is the capital of France?"}'

# failover
curl -X POST $B/chat -H 'Content-Type: application/json' \
  -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"s-3","message":"Name a prime number above 50"}'

# all fail -> 503
curl -i -X POST $B/chat -H 'Content-Type: application/json' \
  -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"s-4","message":"hello"}'

curl $B/sessions/s-1/messages
curl -i $B/sessions/unknown/messages # 404
curl -i -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-5","message":"   "}' # 422
```

---

## 5. Setup Used

- **Editor / IDE**: Antigravity IDE / VS Code
- **AI Tools**: Google Antigravity Agent
- **Stack**: Python 3.12+, FastAPI, Pydantic v2, LiteLLM, DiskCache, SQLite3 (WAL), React 18, TypeScript, Vite, Nginx, Docker Compose

---

## 6. Bonus features

- **Failure-Simulation Dropdown**: Built-in UI selector in the frontend toolbar (`None`, `key_1=429`, `key_1=500`, `key_1=429,key_2=500`, `*=429`) that sets the `X-Simulate-Failure` header for live testing and demos.
- **Key & Fallback Metadata Chip**: Displays `key_id` and `fallback_attempts` alongside `meta-source`, `meta-model`, `meta-latency`, `meta-input-tokens`, and `meta-output-tokens` on every assistant message.
- **Per-Session System Prompt**: Optional `system-prompt-input` field in the UI that persists per session in the SQLite `sessions` table once a turn succeeds.
- **Backend Automated Test Suite**: Comprehensive `pytest` suite (`backend/tests/test_app.py`) mocking `litellm.completion` and covering `/health`, 422 validation edge cases, multi-turn history & session isolation, simulated and real error failover, 503 rollback guarantees, and cache hits/misses.
