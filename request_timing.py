"""Request timing helpers for latency analytics headers."""

from __future__ import annotations

import json
import time
from typing import Any

from flask import g

HeaderTotalMs = "X-Timing-Total-Ms"
HeaderSteps = "X-Timing-Steps"
HeaderPhases = "X-Timing-Phases"
HeaderFirstAudioMs = "X-First-Audio-Ms"
HeaderLlmModelId = "X-Llm-Model-Id"
HeaderLlmSelectedModelId = "X-Llm-Selected-Model-Id"
HeaderLlmProvider = "X-Llm-Provider"

VOICE_PRE_LLM_LABELS = frozenset(
    {
        "save uploaded audio",
        "speech-to-text",
        "load npc/location/memory/recent turns",
        "build prompt/messages",
        "load npc context",
    }
)
VOICE_LLM_LABELS = frozenset(
    {
        "llm npc response",
        "llm text npc response",
    }
)
VOICE_POST_LLM_LABELS = frozenset(
    {
        "post-llm database phase",
        "split tts segments",
        "save conversation turns",
        "schedule async memory extraction",
        "sync memory extraction/save",
    }
)
VOICE_TTS_LABELS = frozenset(
    {
        "piper segmented tts",
        "piper tts",
        "synthesize speech",
    }
)


def begin_request_timing() -> None:
    g._timing_steps = []
    g._timing_start = time.perf_counter()
    g._timing_first_audio_ms = None
    g._timing_llm_model_id = None
    g._timing_llm_selected_model_id = None
    g._timing_llm_provider = None


def note_first_audio_ms() -> None:
    """
    Mark the moment the first playable clip exists.

    Replies are synthesized clip by clip and published as they finish, so the
    turn total says nothing about how long the table waited to hear anything.
    First one wins: later clips are the tail, not the wait.
    """
    try:
        from flask import has_request_context

        if not has_request_context():
            return
    except Exception:
        return

    start = getattr(g, "_timing_start", None)
    if start is None or getattr(g, "_timing_first_audio_ms", None) is not None:
        return
    g._timing_first_audio_ms = max(0, int(round((time.perf_counter() - start) * 1000)))


def note_request_llm_model(
    *,
    active_model_id: str | None,
    selected_model_id: str | None = None,
    provider: str | None = None,
    purpose: str | None = None,
) -> None:
    """Bind the model used for this HTTP request (first chat write wins).

    Process-wide LAST_ACTIVE_* is unsafe under concurrency; timing headers must
    read request-local state so another turn or async memory extract cannot
    steal attribution.
    """
    try:
        from flask import has_request_context

        if not has_request_context():
            return
    except Exception:
        return

    purpose_norm = str(purpose or "").strip().lower() or None
    if purpose_norm and purpose_norm not in ("chat", "default"):
        return

    active = str(active_model_id or "").strip() or None
    selected = str(selected_model_id or "").strip() or None
    provider_name = str(provider or "").strip() or None
    if not active and not selected:
        return

    # First chat LLM in this request owns turn attribution (NPC reply, not memory).
    already_attributed = bool(getattr(g, "_timing_llm_model_id", None))
    if active and not already_attributed:
        g._timing_llm_model_id = active
        # Provider travels with the model that won attribution, so a later
        # fallback hop cannot relabel the host that actually served the turn.
        g._timing_llm_provider = provider_name
    if selected and not getattr(g, "_timing_llm_selected_model_id", None):
        g._timing_llm_selected_model_id = selected


def _parse_step_entry(step: str) -> tuple[str, int]:
    if "=" not in step:
        return "", 0
    label, ms_raw = step.rsplit("=", 1)
    try:
        return label.strip(), max(0, int(ms_raw))
    except (TypeError, ValueError):
        return label.strip(), 0


def build_voice_timing_phases(steps: list[str]) -> dict[str, Any]:
    """Roll middleware step timings into voice-turn phases with cumulative totals."""
    pre_llm = llm = post_llm = tts = other = 0
    for step in steps:
        label, ms = _parse_step_entry(step)
        if not label:
            continue
        if label in VOICE_PRE_LLM_LABELS:
            pre_llm += ms
        elif label in VOICE_LLM_LABELS:
            llm += ms
        elif label in VOICE_POST_LLM_LABELS:
            post_llm += ms
        elif label in VOICE_TTS_LABELS:
            tts += ms
        else:
            other += ms

    cum_pre = pre_llm
    cum_llm = cum_pre + llm
    cum_post = cum_llm + post_llm
    cum_tts = cum_post + tts
    middleware_total = cum_tts + other

    return {
        "pre_llm_ms": pre_llm,
        "llm_ms": llm,
        "post_llm_ms": post_llm,
        "tts_ms": tts,
        "other_ms": other,
        # Everything the labelled steps account for. The buckets only see work
        # wrapped in log_timing, so comparing this against the wall clock is the
        # only way to see time no step is measuring.
        "tracked_ms": middleware_total,
        "middleware_total_ms": middleware_total,
        "cumulative_ms": {
            "pre_llm": cum_pre,
            "llm": cum_llm,
            "post_llm": cum_post,
            "tts": cum_tts,
            "middleware": middleware_total,
        },
    }


def log_timing(request_id: str, label: str, start_time: float) -> float:
    elapsed = time.perf_counter() - start_time
    print(f"[TIMING] {request_id} {label}: {elapsed:.3f}s", flush=True)
    steps = getattr(g, "_timing_steps", None)
    if isinstance(steps, list):
        safe_label = str(label or "").replace("|", "/").replace("=", ":")
        steps.append(f"{safe_label}={int(round(elapsed * 1000))}")
    return time.perf_counter()


def _request_llm_model_headers() -> tuple[str | None, str | None, str | None]:
    """Return (active_model_id, selected_model_id, provider) from this request only."""
    try:
        active = str(getattr(g, "_timing_llm_model_id", None) or "").strip() or None
        selected = str(getattr(g, "_timing_llm_selected_model_id", None) or "").strip() or None
        provider = str(getattr(g, "_timing_llm_provider", None) or "").strip() or None
        return active, selected, provider
    except Exception:
        return None, None, None


def attach_timing_headers(
    response: Any,
    request_id: str | None = None,
    *,
    include_voice_phases: bool = False,
) -> Any:
    """Attach X-Timing-* headers for editor latency analytics."""
    start = getattr(g, "_timing_start", None)
    middleware_total_ms: int | None = None
    if start is not None:
        middleware_total_ms = int(round((time.perf_counter() - start) * 1000))
        response.headers[HeaderTotalMs] = str(max(0, middleware_total_ms))

    steps = getattr(g, "_timing_steps", None)
    active_model_id, selected_model_id, provider = _request_llm_model_headers()
    if active_model_id:
        response.headers[HeaderLlmModelId] = active_model_id
    if selected_model_id:
        response.headers[HeaderLlmSelectedModelId] = selected_model_id
    if provider:
        response.headers[HeaderLlmProvider] = provider

    if isinstance(steps, list) and steps:
        compact = "|".join(steps[:24])
        if len(compact) > 1800:
            compact = compact[:1800]
        response.headers[HeaderSteps] = compact

        if include_voice_phases:
            phases = build_voice_timing_phases(steps)
            first_audio_ms = getattr(g, "_timing_first_audio_ms", None)
            if first_audio_ms is not None:
                phases["first_audio_ms"] = first_audio_ms
                phases["cumulative_ms"]["first_audio"] = first_audio_ms
                response.headers[HeaderFirstAudioMs] = str(first_audio_ms)
            if middleware_total_ms is not None:
                phases["middleware_total_ms"] = middleware_total_ms
                phases["cumulative_ms"]["middleware"] = middleware_total_ms
                # Wall clock minus what the steps measured. A large value means
                # the turn is spending its time somewhere no log_timing call
                # wraps, which the phase breakdown alone cannot show.
                phases["unaccounted_ms"] = max(
                    0, middleware_total_ms - int(phases.get("tracked_ms") or 0)
                )
            if active_model_id:
                phases["model_id"] = active_model_id
            if selected_model_id:
                phases["selected_model_id"] = selected_model_id
            if provider:
                phases["provider"] = provider
            phases_json = json.dumps(phases, separators=(",", ":"))
            if len(phases_json) <= 1800:
                response.headers[HeaderPhases] = phases_json
            if request_id:
                print(f"[TIMING] {request_id} phases: {phases_json}", flush=True)

    if request_id:
        response.headers.setdefault("X-Turn-Request-Id", str(request_id))
    return response
