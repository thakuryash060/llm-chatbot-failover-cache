import logging
import os
from pathlib import Path
from dotenv import load_dotenv

_ROOT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ROOT_ENV)
load_dotenv()

logger = logging.getLogger("chatbot")


def parse_sqlite_path(db_url: str) -> str:
    """Parse sqlite:////data/app.db or sqlite:///./data/app.db into a filesystem path."""
    url = db_url.strip()
    if url.startswith("sqlite:////"):
        # 4 slashes: absolute path on POSIX (e.g. /data/app.db) or Windows drive if applicable
        raw_path = url[len("sqlite:///"):]
        return raw_path
    elif url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    return url


class Settings:
    def __init__(self) -> None:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            load_dotenv(_ROOT_ENV, override=True)
        raw_keys = os.getenv("LLM_API_KEYS", "")
        self.llm_api_keys: list[str] = [
            k.strip().strip('"').strip("'").strip()
            for k in raw_keys.split(",")
            if k.strip().strip('"').strip("'").strip()
        ]
        if not self.llm_api_keys:
            raise RuntimeError(
                "LLM_API_KEYS environment variable is required and must contain at least one API key."
            )
        if len(self.llm_api_keys) < 2:
            logger.warning(
                "Only %d API key configured in LLM_API_KEYS; at least 2 keys are recommended for failover.",
                len(self.llm_api_keys),
            )

        self.llm_model: str = os.getenv(
            "LLM_MODEL", "gemini/gemini-3.5-flash-lite"
        ).strip()
        default_db_url = (
            "sqlite:///./data/app.db" if os.name == "nt" else "sqlite:////data/app.db"
        )
        raw_db_url = os.getenv("DATABASE_URL", default_db_url).strip()
        if os.name == "nt" and raw_db_url == "sqlite:////data/app.db":
            raw_db_url = "sqlite:///./data/app.db"
        self.database_url: str = raw_db_url
        self.db_path: str = parse_sqlite_path(self.database_url)

        raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
        self.allowed_origins: list[str] = [
            o.strip() for o in raw_origins.split(",") if o.strip()
        ]

        sim_val = os.getenv("ENABLE_FAILURE_SIMULATION", "true").strip().lower()
        self.enable_failure_simulation: bool = sim_val in ("1", "true", "yes", "on")

        self.default_system_prompt: str = os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            "You are a helpful, concise AI assistant.",
        ).strip()

        default_cache_dir = str(Path(self.db_path).parent / "cache")
        raw_cache_dir = os.getenv("CACHE_DIR", default_cache_dir).strip()
        if os.name == "nt" and raw_cache_dir == "/data/cache":
            raw_cache_dir = "./data/cache"
        self.cache_dir: str = raw_cache_dir


def get_settings() -> Settings:
    return Settings()
