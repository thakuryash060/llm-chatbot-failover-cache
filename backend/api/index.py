import os
from pathlib import Path

# On Vercel's serverless filesystem, only /tmp is writable
if os.getenv("VERCEL"):
    os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/app.db")
    os.environ.setdefault("CACHE_DIR", "/tmp/cache")
    os.environ.setdefault("ALLOWED_ORIGINS", "*")
    Path("/tmp/cache").mkdir(parents=True, exist_ok=True)

from app.main import app  # noqa: E402, F401
