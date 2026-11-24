import os
import sys
from pathlib import Path

# Ensure backend root is on sys.path for Vercel serverless runtime
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# On Vercel or any serverless runtime where /data is not writable, use /tmp
if os.getenv("VERCEL") or not os.access("/", os.W_OK):
    os.environ["DATABASE_URL"] = "sqlite:////tmp/app.db"
    os.environ["CACHE_DIR"] = "/tmp/cache"
    os.environ.setdefault("ALLOWED_ORIGINS", "*")
    Path("/tmp/cache").mkdir(parents=True, exist_ok=True)

from app.main import app  # noqa: E402, F401
