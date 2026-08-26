"""NPC memory + DM tables: MySQL in production, SQLite fallback for local tests.

Production / Compose: set MEMORY_MYSQL_HOST (and related keys) in game_api.env or
container env. When MEMORY_MYSQL_HOST is unset, uses MEMORY_DB SQLite path so
pytest and quick local runs do not require MySQL.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_SQLITE_PATH = BASE_DIR / "npc_memory.db"

# Optional PyMySQL — only required when MySQL backend is selected.
try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:  # pragma: no cover - exercised when MySQL env is set without dep
    pymysql = None  # type: ignore[assignment]
    DictCursor = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class MemoryDbConfig:
    """Resolved backend for NPC memory / DM persistence."""

    backend: str  # "mysql" | "sqlite"
    sqlite_path: Path | None = None
    mysql_host: str = ""
    mysql_port: int = 3306
    mysql_database: str = ""
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_charset: str = "utf8mb4"

    @property
    def is_mysql(self) -> bool:
        return self.backend == "mysql"

    @property
    def is_sqlite(self) -> bool:
        return self.backend == "sqlite"


def _env_get(environ: dict[str, str] | os._Environ, key: str) -> str:
    return (environ.get(key) or "").strip().strip('"').strip("'")


def resolve_sqlite_path(env: dict[str, str] | None = None) -> Path:
    """SQLite file path when MySQL is not configured."""
    environ = os.environ if env is None else env
    configured = _env_get(environ, "MEMORY_DB")
    if configured:
        return Path(configured)
    return _DEFAULT_SQLITE_PATH


def resolve_memory_db_config(env: dict[str, str] | None = None) -> MemoryDbConfig:
    """Pick MySQL when MEMORY_MYSQL_HOST is set; otherwise SQLite."""
    environ = os.environ if env is None else env
    host = _env_get(environ, "MEMORY_MYSQL_HOST")
    if host:
        port_raw = _env_get(environ, "MEMORY_MYSQL_PORT") or "3306"
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(f"MEMORY_MYSQL_PORT must be an integer, got {port_raw!r}") from exc
        database = _env_get(environ, "MEMORY_MYSQL_DATABASE") or "npc_middleware"
        user = _env_get(environ, "MEMORY_MYSQL_USER")
        password = _env_get(environ, "MEMORY_MYSQL_PASSWORD")
        if not user:
            raise ValueError("MEMORY_MYSQL_USER is required when MEMORY_MYSQL_HOST is set")
        charset = _env_get(environ, "MEMORY_MYSQL_CHARSET") or "utf8mb4"
        return MemoryDbConfig(
            backend="mysql",
            mysql_host=host,
            mysql_port=port,
            mysql_database=database,
            mysql_user=user,
            mysql_password=password,
            mysql_charset=charset,
        )
    return MemoryDbConfig(backend="sqlite", sqlite_path=resolve_sqlite_path(environ))


# Module-level config (refreshed after game_api.env load).
CONFIG = resolve_memory_db_config()

# Back-compat alias used by older tests / call sites that patch a Path.
MEMORY_DB = CONFIG.sqlite_path or _DEFAULT_SQLITE_PATH


def refresh_memory_db_config() -> MemoryDbConfig:
    """Re-read env and update module-level CONFIG / MEMORY_DB."""
    global CONFIG, MEMORY_DB
    CONFIG = resolve_memory_db_config()
    MEMORY_DB = CONFIG.sqlite_path or _DEFAULT_SQLITE_PATH
    return CONFIG


# Older name used by server.py / tests.
def refresh_memory_db() -> Path:
    """Refresh config; return SQLite path (MySQL returns default placeholder path)."""
    refresh_memory_db_config()
    return MEMORY_DB


def resolve_memory_db(env: dict[str, str] | None = None) -> Path:
    """Back-compat: SQLite path resolution only."""
    return resolve_sqlite_path(env)


def _qmark_to_percent(sql: str) -> str:
    """Translate sqlite ``?`` placeholders to PyMySQL ``%s``."""
    return sql.replace("?", "%s")


class _CursorResult:
    """Thin wrapper so callers can use ``cur.lastrowid`` / ``.fetchall()``."""

    def __init__(self, cursor: Any):
        self._cursor = cursor
        self.lastrowid = getattr(cursor, "lastrowid", None)
        self.rowcount = getattr(cursor, "rowcount", -1)

    def fetchall(self) -> list[Any]:
        return list(self._cursor.fetchall())

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def __iter__(self):
        return iter(self.fetchall())


class DbConnection:
    """sqlite3-like connection: ``execute(sql, params)``, commit, context manager."""

    def __init__(self, config: MemoryDbConfig, raw: Any):
        self.config = config
        self._raw = raw

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> _CursorResult:
        params = params or ()
        if self.config.is_mysql:
            cursor = self._raw.cursor()
            cursor.execute(_qmark_to_percent(sql), tuple(params))
            return _CursorResult(cursor)
        cursor = self._raw.execute(sql, params)
        return _CursorResult(cursor)

    def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> _CursorResult:
        if self.config.is_mysql:
            cursor = self._raw.cursor()
            cursor.executemany(_qmark_to_percent(sql), list(seq_of_params))
            return _CursorResult(cursor)
        cursor = self._raw.executemany(sql, seq_of_params)
        return _CursorResult(cursor)

    def commit(self) -> None:
        self._raw.commit()

    def close(self) -> None:
        self._raw.close()

    def __enter__(self) -> DbConnection:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self._raw.commit()
            else:
                try:
                    self._raw.rollback()
                except Exception:
                    pass
        finally:
            self._raw.close()


def connect(config: MemoryDbConfig | None = None) -> DbConnection:
    """Open a connection for the active (or provided) memory DB config."""
    cfg = config or CONFIG
    if cfg.is_mysql:
        if pymysql is None:
            raise RuntimeError(
                "PyMySQL is required for MEMORY_MYSQL_HOST. "
                "Install with: pip install PyMySQL"
            )
        raw = pymysql.connect(
            host=cfg.mysql_host,
            port=cfg.mysql_port,
            user=cfg.mysql_user,
            password=cfg.mysql_password,
            database=cfg.mysql_database,
            charset=cfg.mysql_charset,
            autocommit=False,
            cursorclass=DictCursor,
        )
        return DbConnection(cfg, raw)

    path = cfg.sqlite_path or _DEFAULT_SQLITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(str(path))
    raw.row_factory = sqlite3.Row
    return DbConnection(cfg, raw)


def get_db() -> DbConnection:
    """Open a connection using the current module CONFIG."""
    return connect(CONFIG)


def table_columns(conn: DbConnection, table: str, allowed: frozenset[str]) -> set[str]:
    if table not in allowed:
        raise ValueError(f"Refusing unsafe SQL table name: {table}")
    if conn.config.is_mysql:
        rows = conn.execute(
            """
            SELECT COLUMN_NAME AS name
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = ?
            """,
            (table,),
        ).fetchall()
        return {str(row["name"]) for row in rows}
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # sqlite3.Row: index 1 is name
    return {row[1] for row in rows}


def ensure_campaign_id_column(
    conn: DbConnection,
    table: str,
    allowed: frozenset[str],
    *,
    backfill: str,
) -> None:
    """Add campaign_id if missing and backfill legacy empty rows once."""
    if table not in allowed:
        raise ValueError(f"Refusing unsafe SQL table name: {table}")
    columns = table_columns(conn, table, allowed)
    if "campaign_id" not in columns:
        if conn.config.is_mysql:
            conn.execute(
                f"ALTER TABLE {table} "
                f"ADD COLUMN campaign_id VARCHAR(128) NOT NULL DEFAULT ''"
            )
        else:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN campaign_id TEXT NOT NULL DEFAULT ''"
            )
        if backfill:
            conn.execute(
                f"UPDATE {table} SET campaign_id = ? "
                f"WHERE campaign_id = '' OR campaign_id IS NULL",
                (backfill,),
            )
    index_name = f"idx_{table}_campaign_npc_player"
    if conn.config.is_mysql:
        # MySQL 8.0+ supports IF NOT EXISTS on indexes.
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table} (campaign_id, npc_id, player_id)"
        )
    else:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table} (campaign_id, npc_id, player_id)"
        )


def init_npc_memory_tables(conn: DbConnection) -> None:
    """Create conversation_turns + npc_memories if missing."""
    if conn.config.is_mysql:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                campaign_id VARCHAR(128) NOT NULL DEFAULT '',
                npc_id VARCHAR(128) NOT NULL,
                player_id VARCHAR(128) NOT NULL,
                role VARCHAR(32) NOT NULL,
                content TEXT NOT NULL,
                created_at VARCHAR(64) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS npc_memories (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                campaign_id VARCHAR(128) NOT NULL DEFAULT '',
                npc_id VARCHAR(128) NOT NULL,
                player_id VARCHAR(128) NOT NULL,
                memory TEXT NOT NULL,
                created_at VARCHAR(64) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS npc_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                memory TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def init_dm_schema_tables(conn: DbConnection) -> None:
    """Create DM channel tables if missing."""
    if conn.config.is_mysql:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_messages (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                campaign_id VARCHAR(128) NOT NULL DEFAULT '',
                npc_id VARCHAR(128) NOT NULL,
                player_id VARCHAR(128) NOT NULL,
                content TEXT NOT NULL,
                detected_verdict VARCHAR(32) NULL,
                detected_claim TEXT NULL,
                detection_confidence DOUBLE NULL,
                created_at VARCHAR(64) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_pending_checks (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                campaign_id VARCHAR(128) NOT NULL DEFAULT '',
                npc_id VARCHAR(128) NOT NULL,
                player_id VARCHAR(128) NOT NULL,
                claim_summary TEXT NOT NULL,
                player_text TEXT NULL,
                npc_text TEXT NULL,
                source VARCHAR(64) NOT NULL DEFAULT 'npc',
                status VARCHAR(32) NOT NULL DEFAULT 'pending',
                ruling_id BIGINT NULL,
                created_at VARCHAR(64) NOT NULL,
                resolved_at VARCHAR(64) NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_rulings (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                campaign_id VARCHAR(128) NOT NULL DEFAULT '',
                npc_id VARCHAR(128) NOT NULL,
                player_id VARCHAR(128) NOT NULL,
                claim_summary TEXT NOT NULL,
                verdict VARCHAR(32) NOT NULL,
                ruling_text TEXT NOT NULL,
                source VARCHAR(64) NOT NULL,
                pending_check_id BIGINT NULL,
                created_at VARCHAR(64) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                content TEXT NOT NULL,
                detected_verdict TEXT,
                detected_claim TEXT,
                detection_confidence REAL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_pending_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                claim_summary TEXT NOT NULL,
                player_text TEXT,
                npc_text TEXT,
                source TEXT NOT NULL DEFAULT 'npc',
                status TEXT NOT NULL DEFAULT 'pending',
                ruling_id INTEGER,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_rulings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                claim_summary TEXT NOT NULL,
                verdict TEXT NOT NULL,
                ruling_text TEXT NOT NULL,
                source TEXT NOT NULL,
                pending_check_id INTEGER,
                created_at TEXT NOT NULL
            )
        """)
