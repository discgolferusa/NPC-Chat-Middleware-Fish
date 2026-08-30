"""Unit and concurrency tests for the bounded MySQL connection pool."""

from __future__ import annotations

import threading
import time

import memory_db
import pytest


class _FakeCursor:
    def __init__(self) -> None:
        self.lastrowid = 1
        self.rowcount = 1
        self.closed = False
        self.sql = ""
        self.params = ()

    def execute(self, sql, params=()):
        self.sql = sql
        self.params = params

    def executemany(self, sql, params):
        self.sql = sql
        self.params = list(params)

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    def close(self):
        self.closed = True


class _FakeRawConnection:
    def __init__(self) -> None:
        self.open = True
        self.fail_ping = False
        self.commits = 0
        self.rollbacks = 0
        self.pings = 0
        self.cursors: list[_FakeCursor] = []

    def cursor(self):
        cursor = _FakeCursor()
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        if not self.open:
            raise RuntimeError("connection closed")
        self.commits += 1

    def rollback(self):
        if not self.open:
            raise RuntimeError("connection closed")
        self.rollbacks += 1

    def ping(self, reconnect=True):
        self.pings += 1
        if not self.open or self.fail_ping:
            raise RuntimeError("connection closed")

    def close(self):
        self.open = False


@pytest.fixture(autouse=True)
def _clean_pool(monkeypatch):
    memory_db.close_connection_pools()
    monkeypatch.setenv("MEMORY_MYSQL_POOL_SIZE", "2")
    monkeypatch.setenv("MEMORY_MYSQL_POOL_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("MEMORY_MYSQL_POOL_RECYCLE_SECONDS", "300")
    monkeypatch.setenv("MEMORY_MYSQL_POOL_PING_INTERVAL_SECONDS", "300")
    monkeypatch.setenv("MEMORY_MYSQL_POOL_MIN_IDLE", "0")
    monkeypatch.setenv("MEMORY_MYSQL_POOL_MAINTENANCE_INTERVAL_SECONDS", "0")
    yield
    memory_db.close_connection_pools()


@pytest.fixture
def mysql_config():
    return memory_db.MemoryDbConfig(
        backend="mysql",
        mysql_host="db.example",
        mysql_port=3306,
        mysql_database="npc_middleware",
        mysql_user="middleware",
        mysql_password="secret",
    )


@pytest.fixture
def fake_connect(monkeypatch):
    created: list[_FakeRawConnection] = []

    def create(**kwargs):
        raw = _FakeRawConnection()
        created.append(raw)
        return raw

    monkeypatch.setattr(memory_db.pymysql, "connect", create)
    return created


def test_pool_reuses_mysql_connection(mysql_config, fake_connect):
    with memory_db.connect(mysql_config) as first:
        first.execute("SELECT ?", (1,)).fetchall()

    with memory_db.connect(mysql_config) as second:
        second.execute("SELECT ?", (2,)).fetchall()

    assert len(fake_connect) == 1
    assert fake_connect[0].commits == 2
    assert memory_db.connection_pool_status(mysql_config) == {
        "max_size": 2,
        "total": 1,
        "idle": 1,
        "in_use": 0,
    }


def test_pool_rolls_back_exception_and_reuses_connection(mysql_config, fake_connect):
    with pytest.raises(ValueError, match="test failure"):
        with memory_db.connect(mysql_config):
            raise ValueError("test failure")

    with memory_db.connect(mysql_config):
        pass

    assert len(fake_connect) == 1
    assert fake_connect[0].rollbacks == 1
    assert fake_connect[0].commits == 1


def test_unfetched_cursor_is_closed_before_pool_return(mysql_config, fake_connect):
    with memory_db.connect(mysql_config) as conn:
        conn.execute("UPDATE example SET value = ?", (1,))

    assert fake_connect[0].cursors[0].closed is True


def test_explicit_close_rolls_back_before_pool_return(mysql_config, fake_connect):
    conn = memory_db.connect(mysql_config)
    conn.close()

    with memory_db.connect(mysql_config):
        pass

    assert len(fake_connect) == 1
    assert fake_connect[0].rollbacks == 1


@pytest.mark.parametrize("exit_with_error", [False, True])
def test_exit_after_explicit_close_never_touches_reborrowed_connection(
    monkeypatch,
    mysql_config,
    fake_connect,
    exit_with_error,
):
    """A closed wrapper must not commit/rollback its raw connection again."""
    monkeypatch.setenv("MEMORY_MYSQL_POOL_SIZE", "1")
    first = memory_db.connect(mysql_config)
    first.__enter__()
    first.close()

    second = memory_db.connect(mysql_config)
    raw = fake_connect[0]
    commits_before = raw.commits
    rollbacks_before = raw.rollbacks

    if exit_with_error:
        first.__exit__(ValueError, ValueError("old request"), None)
    else:
        first.__exit__(None, None, None)

    assert raw.commits == commits_before
    assert raw.rollbacks == rollbacks_before
    second.__exit__(None, None, None)
    assert raw.commits == commits_before + 1


def test_pool_blocks_at_capacity_then_reuses_released_connection(
    monkeypatch,
    mysql_config,
    fake_connect,
):
    monkeypatch.setenv("MEMORY_MYSQL_POOL_SIZE", "1")
    first = memory_db.connect(mysql_config)
    acquired = threading.Event()
    release_second = threading.Event()
    errors: list[BaseException] = []

    def checkout_second() -> None:
        try:
            with memory_db.connect(mysql_config):
                acquired.set()
                release_second.wait(timeout=1)
        except BaseException as ex:  # pragma: no cover - assertion reports it
            errors.append(ex)

    thread = threading.Thread(target=checkout_second)
    thread.start()
    assert not acquired.wait(timeout=0.05)

    first.close()
    assert acquired.wait(timeout=1)
    release_second.set()
    thread.join(timeout=1)

    assert errors == []
    assert not thread.is_alive()
    assert len(fake_connect) == 1


def test_pool_checkout_times_out_at_capacity(
    monkeypatch,
    mysql_config,
    fake_connect,
):
    monkeypatch.setenv("MEMORY_MYSQL_POOL_SIZE", "1")
    monkeypatch.setenv("MEMORY_MYSQL_POOL_TIMEOUT_SECONDS", "0.05")
    first = memory_db.connect(mysql_config)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="pool"):
        memory_db.connect(mysql_config)
    elapsed = time.monotonic() - started

    first.close()
    assert elapsed >= 0.04
    assert len(fake_connect) == 1


def test_closed_connection_is_discarded_instead_of_reused(mysql_config, fake_connect):
    with pytest.raises(RuntimeError, match="connection closed"):
        with memory_db.connect(mysql_config) as conn:
            conn._raw.open = False

    with memory_db.connect(mysql_config):
        pass

    assert len(fake_connect) == 2


def test_close_connection_pools_closes_idle_connections(mysql_config, fake_connect):
    with memory_db.connect(mysql_config):
        pass

    raw = fake_connect[0]
    assert raw.open is True
    memory_db.close_connection_pools()
    assert raw.open is False


def test_pool_prewarm_opens_configured_min_idle(
    monkeypatch,
    mysql_config,
    fake_connect,
):
    monkeypatch.setenv("MEMORY_MYSQL_POOL_SIZE", "3")
    monkeypatch.setenv("MEMORY_MYSQL_POOL_MIN_IDLE", "2")

    pool = memory_db._pool_for_config(mysql_config)

    assert len(fake_connect) == 2
    assert pool.status() == {
        "max_size": 3,
        "total": 2,
        "idle": 2,
        "in_use": 0,
    }


def test_pool_prewarm_failure_is_nonfatal(
    monkeypatch,
    mysql_config,
    capsys,
):
    monkeypatch.setenv("MEMORY_MYSQL_POOL_MIN_IDLE", "1")

    def fail_connect(**kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(memory_db.pymysql, "connect", fail_connect)

    pool = memory_db._pool_for_config(mysql_config)

    assert pool.status()["total"] == 0
    assert "pool prewarm failed error=OSError" in capsys.readouterr().out


def test_maintenance_replaces_ping_failure_before_acquire(
    mysql_config,
    fake_connect,
    capsys,
):
    pool = memory_db._MySqlConnectionPool(
        mysql_config,
        max_size=2,
        acquire_timeout=1,
        recycle_seconds=300,
        ping_interval_seconds=1,
        min_idle=1,
        maintenance_interval_seconds=0,
    )
    pool.start()
    assert len(fake_connect) == 1

    original = fake_connect[0]
    original.fail_ping = True
    with pool._condition:
        pool._idle[0].returned_at -= 5

    pool._run_maintenance_cycle()

    assert original.open is False
    assert len(fake_connect) == 2
    assert pool.status()["idle"] == 1
    assert "reason=ping-failed" in capsys.readouterr().out

    entry = pool.acquire()
    assert entry.raw is fake_connect[1]
    assert len(fake_connect) == 2
    pool.release(entry)
    pool.close()


def test_maintenance_thread_stops_on_pool_close(mysql_config, fake_connect):
    waiter_entered = threading.Event()

    def wait_until_stopped(stop_event, interval):
        waiter_entered.set()
        return stop_event.wait()

    pool = memory_db._MySqlConnectionPool(
        mysql_config,
        max_size=2,
        acquire_timeout=1,
        recycle_seconds=300,
        ping_interval_seconds=30,
        min_idle=0,
        maintenance_interval_seconds=15,
        maintenance_wait_fn=wait_until_stopped,
    )
    pool.start()
    thread = pool._maintenance_thread
    assert thread is not None
    assert waiter_entered.wait(timeout=1)

    pool.close()

    assert not thread.is_alive()


def test_steady_state_acquire_opens_no_new_connection(
    monkeypatch,
    mysql_config,
    fake_connect,
):
    monkeypatch.setenv("MEMORY_MYSQL_POOL_MIN_IDLE", "1")
    pool = memory_db._pool_for_config(mysql_config)
    assert len(fake_connect) == 1

    entry = pool.acquire()

    assert entry.raw is fake_connect[0]
    assert len(fake_connect) == 1
    pool.release(entry)
