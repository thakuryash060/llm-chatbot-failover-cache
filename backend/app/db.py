import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    parent = Path(db_path).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)

    with get_connection(db_path) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    system_prompt TEXT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                ON messages (session_id, id);
                """
            )


def check_db_health(db_path: str) -> bool:
    if not Path(db_path).exists():
        return False
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT 1;").fetchone()
        return row is not None and row[0] == 1


def get_session_prompt(db_path: str, session_id: str) -> str | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?;",
            (session_id,),
        ).fetchone()
        if row is not None and row["system_prompt"] is not None:
            return str(row["system_prompt"])
        return None


def get_history(db_path: str, session_id: str) -> list[dict[str, str]]:
    """Return [{role, content}, ...] for the LLM messages payload, oldest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC;
            """,
            (session_id,),
        ).fetchall()
        return [{"role": str(r["role"]), "content": str(r["content"])} for r in rows]


def get_session_messages(db_path: str, session_id: str) -> list[dict[str, Any]]:
    """Return full history for GET /sessions/{id}/messages, oldest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, metadata, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC;
            """,
            (session_id,),
        ).fetchall()

        result: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = {
                "role": str(r["role"]),
                "content": str(r["content"]),
            }
            if r["role"] == "assistant" and r["metadata"]:
                try:
                    item["metadata"] = json.loads(r["metadata"])
                except (ValueError, TypeError):
                    item["metadata"] = None
            result.append(item)
        return result


def save_turn(
    db_path: str,
    session_id: str,
    user_message: str,
    assistant_reply: str,
    metadata: dict[str, Any],
    system_prompt: str | None = None,
) -> None:
    """Save user and assistant rows (and upsert session if needed) in ONE transaction."""
    now = utc_now_iso()
    metadata_json = json.dumps(metadata)

    with get_connection(db_path) as conn:
        with conn:
            if system_prompt is not None:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, system_prompt, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        system_prompt = excluded.system_prompt;
                    """,
                    (session_id, system_prompt, now),
                )
            else:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO sessions (session_id, system_prompt, created_at)
                    VALUES (?, NULL, ?);
                    """,
                    (session_id, now),
                )

            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'user', ?, NULL, ?);
                """,
                (session_id, user_message, now),
            )
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'assistant', ?, ?, ?);
                """,
                (session_id, assistant_reply, metadata_json, now),
            )
