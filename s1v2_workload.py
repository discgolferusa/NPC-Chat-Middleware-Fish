"""Workload generators for S1 v2 (arrivals, mixed lengths, capacity math)."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Iterable


MIN_SAMPLES_FOR_TAIL = 20


@dataclass(frozen=True)
class ArrivalEvent:
    response_index: int
    arrive_at_s: float
    dialogue_class: str
    text: str
    voice_model: str
    speaker_id: int | None


def choose_mixed_class(rng: random.Random, mix: dict[str, float]) -> str:
    items = [(k, float(v)) for k, v in mix.items() if float(v) > 0]
    if not items:
        return "medium"
    total = sum(w for _, w in items)
    x = rng.random() * total
    acc = 0.0
    for name, w in items:
        acc += w
        if x <= acc:
            return name
    return items[-1][0]


def build_poisson_arrivals(
    *,
    n: int,
    rate_per_s: float,
    seed: int,
    dialogue_classes: list[str],
    texts: dict[str, str],
    mix: dict[str, float] | None,
    voice_model: str,
    speaker_ids: list[int | None] | None = None,
    pattern: str = "poisson",
) -> list[ArrivalEvent]:
    """
    Build reproducible arrival schedules.

    pattern:
      - poisson: exponential inter-arrival at rate_per_s
      - sync_burst: all arrive at t=0
      - sustained: fixed interval 1/rate
    """
    rng = random.Random(seed)
    speakers = speaker_ids or [None]
    events: list[ArrivalEvent] = []
    t = 0.0
    for i in range(n):
        if pattern == "sync_burst":
            arrive = 0.0
        elif pattern == "sustained":
            arrive = (i / rate_per_s) if rate_per_s > 0 else float(i)
        else:
            # poisson
            if i == 0:
                arrive = 0.0
            else:
                gap = rng.expovariate(rate_per_s) if rate_per_s > 0 else 1.0
                t += gap
                arrive = t
        if mix:
            cls = choose_mixed_class(rng, mix)
        else:
            cls = dialogue_classes[i % len(dialogue_classes)]
        text = texts[cls]
        sid = speakers[i % len(speakers)]
        events.append(
            ArrivalEvent(
                response_index=i,
                arrive_at_s=arrive,
                dialogue_class=cls,
                text=text,
                voice_model=voice_model,
                speaker_id=sid,
            )
        )
    return events


def summarize_latencies(values: Iterable[float]) -> dict[str, float | None]:
    data = sorted(float(v) for v in values)
    if not data:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "min": None, "max": None}

    def pct(p: float) -> float:
        k = int(round((len(data) - 1) * (p / 100.0)))
        return data[max(0, min(len(data) - 1, k))]

    return {
        "count": len(data),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "min": data[0],
        "max": data[-1],
    }


def small_n_warning(n: int, *, min_n: int = MIN_SAMPLES_FOR_TAIL) -> str | None:
    if n < min_n:
        return (
            f"sample_size={n} below recommended {min_n} for reliable p95/p99; "
            "treat tail latencies as exploratory only"
        )
    return None


def build_capacity_summary(
    *,
    measured: dict[str, Any],
    assumptions: dict[str, Any],
) -> dict[str, Any]:
    """
    Separate measured facts from extrapolated planning estimates.

    Does not modify any financial workbook.
    """
    responses_per_s = measured.get("sustainable_responses_per_s")
    audio_s_per_s = measured.get("sustainable_audio_seconds_per_s")
    burst = measured.get("burst_concurrency_before_fail")
    ttfa_ok_rate = measured.get("ttfa_target_pass_rate")
    stall_rate = measured.get("playback_stall_rate")

    players = float(assumptions.get("players_per_table") or 6)
    interactions_per_session = float(assumptions.get("npc_interactions_per_session") or 60)
    session_hours = float(assumptions.get("session_hours") or 2)
    peak_mult = float(assumptions.get("peak_traffic_multiplier") or 2.0)
    dialogue_intensity = float(assumptions.get("dialogue_intensive_multiplier") or 1.0)

    # Mean interaction rate per table (responses/s) during a session.
    base_table_rps = (interactions_per_session / (session_hours * 3600.0)) * dialogue_intensity
    peak_table_rps = base_table_rps * peak_mult

    estimated_tables = None
    if isinstance(responses_per_s, (int, float)) and peak_table_rps > 0:
        estimated_tables = float(responses_per_s) / peak_table_rps

    warnings: list[str] = []
    n = int(measured.get("sample_size") or 0)
    warn = small_n_warning(n)
    if warn:
        warnings.append(warn)

    return {
        "measured": {
            "sustainable_responses_per_s": responses_per_s,
            "sustainable_audio_seconds_per_s": audio_s_per_s,
            "burst_concurrency_before_fail": burst,
            "ttfa_p50_ms": measured.get("ttfa_p50_ms"),
            "ttfa_p95_ms": measured.get("ttfa_p95_ms"),
            "complete_p95_ms": measured.get("complete_p95_ms"),
            "ttfa_target_pass_rate": ttfa_ok_rate,
            "playback_stall_rate": stall_rate,
            "sample_size": n,
            "note": (
                "Eight Piper instances are not eight tables; capacity is derived "
                "from measured throughput and QoS targets."
            ),
        },
        "assumed": {
            "players_per_table": players,
            "npc_interactions_per_session": interactions_per_session,
            "session_hours": session_hours,
            "peak_traffic_multiplier": peak_mult,
            "dialogue_intensive_multiplier": dialogue_intensity,
            "base_table_responses_per_s": base_table_rps,
            "peak_table_responses_per_s": peak_table_rps,
            "ttfa_p95_target_ms": assumptions.get("ttfa_p95_target_ms"),
            "max_stall_rate": assumptions.get("max_stall_rate"),
            "model_usage_distribution": assumptions.get("model_usage_distribution"),
        },
        "extrapolated": {
            "estimated_active_tables_per_piper_worker": estimated_tables,
            "method": (
                "sustainable_responses_per_s / (interactions_per_session / "
                "session_seconds * dialogue_intensive * peak_multiplier)"
            ),
        },
        "warnings": warnings,
    }


def variability(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "stdev": None, "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "stdev": 0.0, "n": 1}
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values),
        "n": len(values),
    }


def estimate_duration_seconds(
    *,
    n_responses: int,
    concurrency: int,
    avg_complete_s: float = 8.0,
    arrival_span_s: float = 0.0,
) -> float:
    waves = math.ceil(n_responses / max(1, concurrency))
    return float(waves * avg_complete_s + arrival_span_s + 60.0)
