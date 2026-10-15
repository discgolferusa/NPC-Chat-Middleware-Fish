"""NPC memory + DM tables: MySQL in production, SQLite fallback for local tests.

Production / Compose: set MEMORY_MYSQL_HOST (and related keys) in game_api.env or
container env. When MEMORY_MYSQL_HOST is unset, uses MEMORY_DB SQLite path so
pytest and quick local runs do not require MySQL.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

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
    close_connection_pools()
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


# PyMySQL reports a server-closed or timed-out socket as these codes.
# 2006 server has gone away, 2013 lost connection during query,
# 2014 commands out of sync, 2055 lost connection at handshake.
_MYSQL_DISCONNECT_CODES = frozenset({2006, 2013, 2014, 2055})


def _is_mysql_disconnect(exc: BaseException) -> bool:
    if pymysql is None:
        return False
    errors = getattr(pymysql, "err", None)
    if errors is None or not isinstance(
        exc, (errors.OperationalError, errors.InterfaceError)
    ):
        return False
    code = exc.args[0] if exc.args else None
    if code in _MYSQL_DISCONNECT_CODES:
        return True
    return isinstance(exc, errors.InterfaceError)


class _CursorResult:
    """Thin wrapper so callers can use ``cur.lastrowid`` / ``.fetchall()``."""

    def __init__(self, cursor: Any, on_close=None):
        self._cursor = cursor
        self._on_close = on_close
        self._closed = False
        self.lastrowid = getattr(cursor, "lastrowid", None)
        self.rowcount = getattr(cursor, "rowcount", -1)

    def fetchall(self) -> list[Any]:
        try:
            return list(self._cursor.fetchall())
        finally:
            self.close()

    def fetchone(self) -> Any:
        try:
            return self._cursor.fetchone()
        finally:
            self.close()

    def __iter__(self):
        return iter(self.fetchall())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._cursor, "close", None)
            if callable(close):
                close()
        finally:
            if self._on_close is not None:
                on_close = self._on_close
                self._on_close = None
                on_close(self)


@dataclass
class _PoolEntry:
    raw: Any
    created_at: float
    returned_at: float


class _MySqlConnectionPool:
    """Small bounded, thread-safe pool for PyMySQL connections."""

    def __init__(
        self,
        config: MemoryDbConfig,
        *,
        max_size: int,
        acquire_timeout: float,
        recycle_seconds: float,
        ping_interval_seconds: float,
        min_idle: int = 1,
        maintenance_interval_seconds: float = 15.0,
        maintenance_wait_fn: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        self.config = config
        self.max_size = max(1, int(max_size))
        self.acquire_timeout = max(0.01, float(acquire_timeout))
        self.recycle_seconds = max(0.0, float(recycle_seconds))
        self.ping_interval_seconds = max(0.0, float(ping_interval_seconds))
        self.min_idle = min(self.max_size, max(0, int(min_idle)))
        self.maintenance_interval_seconds = max(
            0.0, float(maintenance_interval_seconds)
        )
        self._condition = threading.Condition()
        self._idle: list[_PoolEntry] = []
        self._total = 0
        self._closed = False
        self._started = False
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._maintenance_thread: threading.Thread | None = None
        self._maintenance_wait_fn = maintenance_wait_fn

    def _new_raw_connection(self) -> Any:
        cfg = self.config
        return pymysql.connect(
            host=cfg.mysql_host,
            port=cfg.mysql_port,
            user=cfg.mysql_user,
            password=cfg.mysql_password,
            database=cfg.mysql_database,
            charset=cfg.mysql_charset,
            autocommit=False,
            cursorclass=DictCursor,
            connect_timeout=_positive_int_env("MEMORY_MYSQL_CONNECT_TIMEOUT_SECONDS", 10),
            read_timeout=_positive_int_env("MEMORY_MYSQL_READ_TIMEOUT_SECONDS", 30),
            write_timeout=_positive_int_env("MEMORY_MYSQL_WRITE_TIMEOUT_SECONDS", 30),
        )

    @staticmethod
    def _log_discard(reason: str) -> None:
        print(
            f"[MEMORY-DB] pooled MySQL connection dropped reason={reason}",
            flush=True,
        )

    def _discard(
        self,
        entry: _PoolEntry,
        *,
        reason: str,
        reason_already_logged: bool = False,
    ) -> None:
        if not reason_already_logged:
            self._log_discard(reason)
        try:
            entry.raw.close()
        except Exception:
            pass
        with self._condition:
            self._total = max(0, self._total - 1)
            self._condition.notify()

    def _entry_is_usable(
        self,
        entry: _PoolEntry,
        now: float,
        *,
        checkout: bool = False,
    ) -> tuple[bool, str | None, bool]:
        if self.recycle_seconds and now - entry.created_at >= self.recycle_seconds:
            return False, "recycle-age", False
        if not bool(getattr(entry.raw, "open", True)):
            return False, "socket-closed", False
        idle_seconds = now - entry.returned_at
        ping = getattr(entry.raw, "ping", None)
        # Checkout always pings. PyMySQL keeps .open true after the server or a
        # proxy has already closed the socket, so an idle-time threshold hands
        # that connection to the next DM check or NPC turn and the query dies
        # with "Lost connection to MySQL server during query".
        should_ping = callable(ping) and self.ping_interval_seconds and (
            checkout or idle_seconds >= self.ping_interval_seconds
        )
        if should_ping:
            try:
                # Reconnect is deliberately disabled so the pool retains an exact
                # count and can discard/recreate failed connections itself.
                ping(reconnect=False)
            except Exception:
                # Log here, where the actual liveness failure is observed. The
                # caller passes reason_already_logged to _discard so one failed
                # ping produces exactly one non-secret line.
                self._log_discard("ping-failed")
                return False, "ping-failed", True
        return True, None, False

    def _open_reserved_entry(self) -> _PoolEntry:
        """Open one connection after the caller has reserved a _total slot."""
        try:
            connect_started = time.monotonic()
            raw = self._new_raw_connection()
            now = time.monotonic()
            print(
                "[MEMORY-DB] pooled MySQL connection opened "
                f"elapsed={now - connect_started:.3f}s",
                flush=True,
            )
            return _PoolEntry(raw=raw, created_at=now, returned_at=now)
        except Exception:
            with self._condition:
                self._total = max(0, self._total - 1)
                self._condition.notify()
            raise

    def _top_up_min_idle(self, *, stage: str) -> None:
        """Best-effort creation of idle entries, bounded by min_idle/max_size."""
        while True:
            with self._condition:
                if self._closed:
                    return
                if len(self._idle) >= self.min_idle or self._total >= self.max_size:
                    return
                self._total += 1

            try:
                entry = self._open_reserved_entry()
            except Exception as ex:
                # Pool warming is an optimization. A transient database outage
                # must not make merely starting the maintenance worker fatal.
                print(
                    f"[MEMORY-DB] MySQL pool {stage} failed "
                    f"error={type(ex).__name__}",
                    flush=True,
                )
                return
            self.release(entry)

    def _run_maintenance_cycle(self) -> None:
        """Validate idle entries, recycle failures, then restore MIN_IDLE."""
        with self._condition:
            entries_to_check = len(self._idle)

        for _ in range(entries_to_check):
            with self._condition:
                if self._closed or not self._idle:
                    break
                # Oldest first. Returning a validated entry appends it at the end,
                # so this cycle cannot repeatedly select the same connection.
                entry = self._idle.pop(0)

            usable, reason, reason_logged = self._entry_is_usable(
                entry, time.monotonic()
            )
            if usable:
                self.release(entry)
            else:
                self._discard(
                    entry,
                    reason=reason or "socket-closed",
                    reason_already_logged=reason_logged,
                )

        self._top_up_min_idle(stage="maintenance")

    def _wait_for_maintenance(self) -> bool:
        if self._maintenance_wait_fn is not None:
            return bool(
                self._maintenance_wait_fn(
                    self._stop_event,
                    self.maintenance_interval_seconds,
                )
            )
        return self._stop_event.wait(self.maintenance_interval_seconds)

    def _maintenance_loop(self) -> None:
        while not self._wait_for_maintenance():
            if self._stop_event.is_set():
                break
            try:
                self._run_maintenance_cycle()
            except Exception as ex:
                # Keep a single unexpected maintenance error from permanently
                # disabling future refresh cycles.
                print(
                    "[MEMORY-DB] MySQL pool maintenance cycle failed "
                    f"error={type(ex).__name__}",
                    flush=True,
                )

    def start(self) -> None:
        """Non-fatally prewarm MIN_IDLE and start the daemon maintainer once."""
        with self._start_lock:
            if self._started or self._closed:
                return
            self._started = True

            self._top_up_min_idle(stage="prewarm")
            if self.maintenance_interval_seconds <= 0 or self._closed:
                return

            thread = threading.Thread(
                target=self._maintenance_loop,
                name="mysql-pool-maintenance",
                daemon=True,
            )
            self._maintenance_thread = thread
            thread.start()

    def acquire(self) -> _PoolEntry:
        deadline = time.monotonic() + self.acquire_timeout
        while True:
            create_new = False
            with self._condition:
                if self._closed:
                    raise RuntimeError("MySQL connection pool is closed")
                if self._idle:
                    entry = self._idle.pop()
                elif self._total < self.max_size:
                    # Reserve the slot before connecting outside the lock so two
                    # WSGI threads cannot both exceed max_size.
                    self._total += 1
                    create_new = True
                    entry = None
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Timed out waiting for a MySQL connection from the "
                            f"pool (size={self.max_size}, timeout={self.acquire_timeout:.2f}s)"
                        )
                    self._condition.wait(timeout=remaining)
                    continue

            if create_new:
                return self._open_reserved_entry()

            assert entry is not None
            now = time.monotonic()
            usable, reason, reason_logged = self._entry_is_usable(
                entry, now, checkout=True
            )
            if usable:
                return entry
            self._discard(
                entry,
                reason=reason or "socket-closed",
                reason_already_logged=reason_logged,
            )

    def release(self, entry: _PoolEntry, *, invalid: bool = False) -> None:
        if invalid or not bool(getattr(entry.raw, "open", True)):
            self._discard(
                entry,
                reason=(
                    "socket-closed"
                    if not bool(getattr(entry.raw, "open", True))
                    else "transaction-reset-failed"
                ),
            )
            return

        entry.returned_at = time.monotonic()
        with self._condition:
            if self._closed:
                should_discard = True
            else:
                self._idle.append(entry)
                self._condition.notify()
                should_discard = False
        if should_discard:
            self._discard(entry, reason="pool-closed")

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._closed = True
            idle = list(self._idle)
            self._idle.clear()
            self._condition.notify_all()
        for entry in idle:
            self._discard(entry, reason="pool-closed")

        thread = self._maintenance_thread
        if thread is not None and thread is not threading.current_thread():
            # _stop_event wakes the normal interval wait immediately. If a custom
            # wait is injected for tests it must likewise observe stop_event.
            thread.join()

    def status(self) -> dict[str, int]:
        with self._condition:
            idle = len(self._idle)
            total = self._total
            return {
                "max_size": self.max_size,
                "total": total,
                "idle": idle,
                "in_use": max(0, total - idle),
            }


_MYSQL_POOLS_LOCK = threading.Lock()
_MYSQL_POOLS: dict[MemoryDbConfig, _MySqlConnectionPool] = {}


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.environ.get(name) or str(default)).strip()))
    except ValueError:
        return default


def _nonnegative_int_env(name: str, default: int) -> int:
    try:
        return max(0, int((os.environ.get(name) or str(default)).strip()))
    except ValueError:
        return default


def _nonnegative_float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float((os.environ.get(name) or str(default)).strip()))
    except ValueError:
        return default


def _pool_for_config(config: MemoryDbConfig) -> _MySqlConnectionPool:
    with _MYSQL_POOLS_LOCK:
        pool = _MYSQL_POOLS.get(config)
        if pool is None:
            pool = _MySqlConnectionPool(
                config,
                max_size=_positive_int_env("MEMORY_MYSQL_POOL_SIZE", 10),
                acquire_timeout=_nonnegative_float_env(
                    "MEMORY_MYSQL_POOL_TIMEOUT_SECONDS", 10.0
                ),
                recycle_seconds=_nonnegative_float_env(
                    "MEMORY_MYSQL_POOL_RECYCLE_SECONDS", 1800.0
                ),
                ping_interval_seconds=_nonnegative_float_env(
                    "MEMORY_MYSQL_POOL_PING_INTERVAL_SECONDS", 30.0
                ),
                min_idle=_nonnegative_int_env("MEMORY_MYSQL_POOL_MIN_IDLE", 1),
                maintenance_interval_seconds=_nonnegative_float_env(
                    "MEMORY_MYSQL_POOL_MAINTENANCE_INTERVAL_SECONDS", 15.0
                ),
            )
            _MYSQL_POOLS[config] = pool
            print(
                "[MEMORY-DB] MySQL connection pool ready "
                f"size={pool.max_size} timeout={pool.acquire_timeout:.1f}s "
                f"recycle={pool.recycle_seconds:.1f}s "
                f"min_idle={pool.min_idle} "
                f"maintenance={pool.maintenance_interval_seconds:.1f}s",
                flush=True,
            )
            pool.start()
        return pool


def close_connection_pools() -> None:
    """Close idle pooled connections; checked-out connections close on return."""
    with _MYSQL_POOLS_LOCK:
        pools = list(_MYSQL_POOLS.values())
        _MYSQL_POOLS.clear()
    for pool in pools:
        pool.close()


def connection_pool_status(config: MemoryDbConfig | None = None) -> dict[str, int]:
    """Return non-secret pool counters for diagnostics and tests."""
    cfg = config or CONFIG
    if not cfg.is_mysql:
        return {"max_size": 0, "total": 0, "idle": 0, "in_use": 0}
    return _pool_for_config(cfg).status()


class DbConnection:
    """sqlite3-like connection: ``execute(sql, params)``, commit, context manager."""

    def __init__(
        self,
        config: MemoryDbConfig,
        raw: Any,
        *,
        pool: _MySqlConnectionPool | None = None,
        pool_entry: _PoolEntry | None = None,
    ):
        self.config = config
        self._raw = raw
        self._pool = pool
        self._pool_entry = pool_entry
        self._closed = False
        self._results: set[_CursorResult] = set()
        self._statements_done = False
        self._disconnect_retried = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Database connection is already closed")

    def _forget_result(self, result: _CursorResult) -> None:
        self._results.discard(result)

    def _close_results(self) -> None:
        for result in list(self._results):
            try:
                result.close()
            except Exception:
                pass
        self._results.clear()

    def _finish(self, *, invalid: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_results()
        if self._pool is not None and self._pool_entry is not None:
            self._pool.release(self._pool_entry, invalid=invalid)
        else:
            self._raw.close()

    def _replace_after_disconnect(self, exc: BaseException) -> None:
        """Drop the dead socket and check out a new one for a single retry."""
        code = exc.args[0] if getattr(exc, "args", None) else ""
        print(
            "[MEMORY-DB] retrying MySQL statement after disconnect "
            f"code={code}",
            flush=True,
        )
        self._disconnect_retried = True
        self._close_results()
        entry = self._pool_entry
        self._pool_entry = None
        if self._pool is not None and entry is not None:
            self._pool.release(entry, invalid=True)
        if self._pool is None:
            self._raw = None
            raise RuntimeError("MySQL connection cannot be retried") from exc
        try:
            fresh = self._pool.acquire()
        except Exception:
            self._raw = None
            raise
        self._pool_entry = fresh
        self._raw = fresh.raw

    def _run_mysql(self, sql: str, params: Sequence[Any], *, many: bool) -> _CursorResult:
        bound = list(params) if many else tuple(params)
        try:
            cursor = self._raw.cursor()
            if many:
                cursor.executemany(_qmark_to_percent(sql), bound)
            else:
                cursor.execute(_qmark_to_percent(sql), bound)
        except Exception as exc:
            if not _is_mysql_disconnect(exc) or self._statements_done or self._disconnect_retried:
                raise
            self._replace_after_disconnect(exc)
            cursor = self._raw.cursor()
            if many:
                cursor.executemany(_qmark_to_percent(sql), bound)
            else:
                cursor.execute(_qmark_to_percent(sql), bound)
        self._statements_done = True
        result = _CursorResult(cursor, self._forget_result)
        self._results.add(result)
        return result

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> _CursorResult:
        self._ensure_open()
        params = params or ()
        if self.config.is_mysql:
            return self._run_mysql(sql, params, many=False)
        cursor = self._raw.execute(sql, params)
        result = _CursorResult(cursor, self._forget_result)
        self._results.add(result)
        return result

    def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> _CursorResult:
        self._ensure_open()
        if self.config.is_mysql:
            return self._run_mysql(sql, seq_of_params, many=True)
        cursor = self._raw.executemany(sql, seq_of_params)
        result = _CursorResult(cursor, self._forget_result)
        self._results.add(result)
        return result

    def commit(self) -> None:
        self._ensure_open()
        self._raw.commit()

    def close(self) -> None:
        if self._closed:
            return
        invalid = False
        if self._pool is not None:
            try:
                # Explicit close without a context-manager exit must not leak an
                # unfinished transaction into the next request.
                self._raw.rollback()
            except Exception:
                invalid = True
        self._finish(invalid=invalid)

    def __enter__(self) -> DbConnection:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # close() may already have returned this raw connection to the pool. Never
        # touch it again: another WSGI request may have checked it out meanwhile.
        if self._closed or self._raw is None:
            self._closed = True
            return
        invalid = False
        try:
            if exc_type is None:
                self._raw.commit()
            else:
                try:
                    self._raw.rollback()
                except Exception:
                    invalid = True
        except Exception:
            invalid = True
            raise
        finally:
            self._finish(invalid=invalid)


def connect(config: MemoryDbConfig | None = None) -> DbConnection:
    """Open a connection for the active (or provided) memory DB config."""
    cfg = config or CONFIG
    if cfg.is_mysql:
        if pymysql is None:
            raise RuntimeError(
                "PyMySQL is required for MEMORY_MYSQL_HOST. "
                "Install with: pip install PyMySQL"
            )
        pool = _pool_for_config(cfg)
        entry = pool.acquire()
        return DbConnection(cfg, entry.raw, pool=pool, pool_entry=entry)

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


def index_exists(
    conn: DbConnection,
    table: str,
    index_name: str,
    allowed: frozenset[str],
) -> bool:
    """True if ``index_name`` already exists on ``table``."""
    if table not in allowed:
        raise ValueError(f"Refusing unsafe SQL table name: {table}")
    if conn.config.is_mysql:
        # Oracle MySQL has no CREATE INDEX IF NOT EXISTS (MariaDB does).
        rows = conn.execute(
            """
            SELECT 1 AS present
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = ?
              AND INDEX_NAME = ?
            LIMIT 1
            """,
            (table, index_name),
        ).fetchall()
        return bool(rows)
    rows = conn.execute(
        """
        SELECT 1 AS present
        FROM sqlite_master
        WHERE type = 'index'
          AND name = ?
          AND tbl_name = ?
        LIMIT 1
        """,
        (index_name, table),
    ).fetchall()
    return bool(rows)


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
    if not index_exists(conn, table, index_name, allowed):
        conn.execute(
            f"CREATE INDEX {index_name} ON {table} (campaign_id, npc_id, player_id)"
        )


def ensure_nullable_column(
    conn: DbConnection,
    table: str,
    column: str,
    allowed: frozenset[str],
    *,
    mysql_type: str,
    sqlite_type: str = "TEXT",
) -> None:
    """Add a nullable column when missing. Existing rows stay NULL (legacy)."""
    if table not in allowed:
        raise ValueError(f"Refusing unsafe SQL table name: {table}")
    if not column.replace("_", "").isalnum():
        raise ValueError(f"Refusing unsafe SQL column name: {column}")
    columns = table_columns(conn, table, allowed)
    if column in columns:
        return
    if conn.config.is_mysql:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {mysql_type} NULL"
        )
    else:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {sqlite_type}"
        )


_ENCOUNTER_TABLES = frozenset({
    "conversation_turns",
    "npc_memories",
    "dm_rulings",
})


def _existing_tables(conn: DbConnection) -> set[str]:
    if conn.config.is_mysql:
        rows = conn.execute(
            """
            SELECT TABLE_NAME AS name
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
            """
        ).fetchall()
        return {str(row["name"]) for row in rows}
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row[0] for row in rows}


def ensure_encounter_columns(conn: DbConnection) -> None:
    """Additive encounter columns. NULL means a legacy player-scoped row."""
    present = _existing_tables(conn)
    text_cols = {
        "conversation_turns": (
            ("visibility", "VARCHAR(32)"),
            ("audience_player_ids", "TEXT"),
            ("scene_ref", "TEXT"),
            ("event_kind", "VARCHAR(32)"),
        ),
        "npc_memories": (
            ("visibility", "VARCHAR(32)"),
        ),
        "dm_rulings": (
            ("consequence_text", "TEXT"),
            ("visibility", "VARCHAR(32)"),
            ("audience_player_ids", "TEXT"),
        ),
    }
    for table, columns in text_cols.items():
        if table not in present:
            continue
        for column, mysql_type in columns:
            ensure_nullable_column(
                conn,
                table,
                column,
                _ENCOUNTER_TABLES,
                mysql_type=mysql_type,
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
    ensure_encounter_columns(conn)


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
    ensure_encounter_columns(conn)
