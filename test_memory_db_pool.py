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
        if not self.open:
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


def test_exit_after_explicit_close_does_not_touch_pooled_connection(
    mysql_config, fake_connect
):
    with memory_db.connect(mysql_config) as conn:
        conn.close()

    raw = fake_connect[0]
    assert (raw.commits, raw.rollbacks) == (0, 1)

    # The connection is back in the pool and usable by the next caller.
    with memory_db.connect(mysql_config) as second:
        second.execute("SELECT ?", (1,)).fetchall()

    assert len(fake_connect) == 1
    assert (raw.commits, raw.rollbacks) == (1, 1)


def test_exit_after_explicit_close_with_exception_does_not_roll_back(
    mysql_config, fake_connect
):
    with pytest.raises(ValueError, match="boom"):
        with memory_db.connect(mysql_config) as conn:
            conn.close()
            raise ValueError("boom")

    raw = fake_connect[0]
    assert (raw.commits, raw.rollbacks) == (0, 1)
    assert memory_db.connection_pool_status(mysql_config)["idle"] == 1


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

