"""Request timing helpers for latency analytics headers."""

from __future__ import annotations

import time
from typing import Any

from flask import g


def begin_request_timing() -> None:
    g._timing_steps = []
    g._timing_start = time.perf_counter()


def log_timing(request_id: str, label: str, start_time: float) -> float:
    elapsed = time.perf_counter() - start_time
    print(f"[TIMING] {request_id} {label}: {elapsed:.3f}s", flush=True)
    steps = getattr(g, "_timing_steps", None)
    if isinstance(steps, list):
        safe_label = str(label or "").replace("|", "/").replace("=", ":")
        steps.append(f"{safe_label}={int(round(elapsed * 1000))}")
    return time.perf_counter()


def attach_timing_headers(response: Any, request_id: str | None = None) -> Any:
    """Attach X-Timing-* headers for editor latency analytics."""
    start = getattr(g, "_timing_start", None)
    if start is not None:
        total_ms = int(round((time.perf_counter() - start) * 1000))
        response.headers["X-Timing-Total-Ms"] = str(max(0, total_ms))
    steps = getattr(g, "_timing_steps", None)
    if isinstance(steps, list) and steps:
        compact = "|".join(steps[:24])
        if len(compact) > 1800:
            compact = compact[:1800]
        response.headers["X-Timing-Steps"] = compact
    if request_id:
        response.headers.setdefault("X-Turn-Request-Id", str(request_id))
    return response
