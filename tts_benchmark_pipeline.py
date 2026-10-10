"""
Production-equivalent TTS pipeline entry for S1 v2 benchmarks.

Calls the same segmentation + synthesize_segmented_speech_clips path as
/npc-voice-turn with stream_clips semantics. No OpenRouter, Discord, or Flask
request context required.
"""

from __future__ import annotations

import secrets
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class TimingEvent:
    event: str
    t: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    request_id: str
    ok: bool
    error: str | None
    server_ttfa_ms: float | None
    complete_ms: float | None
    segmentation_ms: float | None
    first_acquire_wait_ms: float | None
    first_synth_ms: float | None
    first_postprocess_ms: float | None
    segment_count: int
    segments: list[dict[str, Any]]
    clip_paths: list[str]
    clip_durations_s: list[float]
    total_inference_ms: float | None
    first_chunk_acquire_wait_ms: float | None
    followup_acquire_wait_ms_sum: float | None
    events: list[TimingEvent]
    pending_seen: bool
    pool_metrics_end: dict[str, Any] | None = None
    playback: dict[str, Any] | None = None


def _first_postprocess_ms(events: list[TimingEvent]) -> float | None:
    for ev in events:
        if ev.event == "segment_postprocess_complete" and int(ev.payload.get("index", -1)) == 0:
            v = ev.payload.get("postprocess_ms")
            return float(v) if isinstance(v, (int, float)) else None
    return None


def wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            rate = float(wav_file.getframerate() or 0)
            frames = int(wav_file.getnframes() or 0)
            if rate <= 0:
                return None
            return frames / rate
    except Exception:
        return None


def simulate_sequential_playback(
    *,
    ready_times_s: list[float],
    durations_s: list[float],
    origin_s: float,
) -> dict[str, Any]:
    """
    Simulated clip-by-clip playback (SignalR-style), not browser QoE.

    Clip 0 starts when ready. Clip i+1 starts at max(ready, previous play end).
    Stall when the next clip is not ready when the previous clip finishes.
    """
    if not ready_times_s or not durations_s or len(ready_times_s) != len(durations_s):
        return {
            "stall_count": 0,
            "total_stall_ms": 0.0,
            "largest_gap_ms": 0.0,
            "play_timeline": [],
            "label": "simulated_sequential_playback_not_browser_qoe",
        }

    stalls = 0
    total_stall = 0.0
    largest_gap = 0.0
    timeline: list[dict[str, Any]] = []
    play_cursor = ready_times_s[0]

    for i, (ready, dur) in enumerate(zip(ready_times_s, durations_s)):
        if i == 0:
            start = ready
        else:
            if ready > play_cursor:
                gap = ready - play_cursor
                stalls += 1
                total_stall += gap
                largest_gap = max(largest_gap, gap)
                start = ready
            else:
                start = play_cursor
        end = start + max(0.0, float(dur))
        timeline.append(
            {
                "index": i,
                "ready_rel_ms": (ready - origin_s) * 1000.0,
                "play_start_rel_ms": (start - origin_s) * 1000.0,
                "play_end_rel_ms": (end - origin_s) * 1000.0,
                "duration_ms": float(dur) * 1000.0,
                "stalled_before": i > 0 and ready > play_cursor,
            }
        )
        play_cursor = end

    return {
        "stall_count": stalls,
        "total_stall_ms": total_stall * 1000.0,
        "largest_gap_ms": largest_gap * 1000.0,
        "play_timeline": timeline,
        "label": "simulated_sequential_playback_not_browser_qoe",
    }


def run_production_tts_pipeline(
    npc_text: str,
    *,
    npc_voice_model: str | Path,
    npc_tts_options: dict[str, Any] | None = None,
    return_after_first_clip: bool = True,
    output_dir: Path | None = None,
    request_id: str | None = None,
    wait_for_remaining_timeout_s: float = 600.0,
    simulate_playback: bool = True,
    on_clip_ready: Callable[..., Any] | None = None,
) -> PipelineResult:
    """
    Run production segmented TTS with stream_clips-equivalent early return.

    Imports server lazily so unit tests can patch before call.
    """
    import server as server_mod
    import piper_pool

    rid = request_id or secrets.token_hex(16)
    events: list[TimingEvent] = []
    events_lock = threading.Lock()
    done = threading.Event()
    stream_error: list[BaseException | None] = [None]
    final_result: list[dict[str, Any] | None] = [None]

    def on_timing_event(event: str, **payload: Any) -> None:
        t = float(payload.pop("t", time.perf_counter()))
        with events_lock:
            events.append(TimingEvent(event=event, t=t, payload=dict(payload)))

    def _on_stream_complete(result: dict[str, Any], error: BaseException | None) -> None:
        final_result[0] = result
        stream_error[0] = error
        done.set()

    prev_output: Path | None = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prev_output = Path(server_mod.OUTPUT_DIR)
        server_mod.OUTPUT_DIR = output_dir

    t0 = time.perf_counter()
    ok = True
    error_s: str | None = None
    pending_seen = False
    synth_result: dict[str, Any] | None = None

    try:
        synth_result = server_mod.synthesize_segmented_speech_clips(
            npc_text,
            str(npc_voice_model),
            rid,
            npc_tts_options=npc_tts_options,
            on_clip_ready=on_clip_ready,
            return_after_first_clip=return_after_first_clip,
            on_stream_complete=_on_stream_complete if return_after_first_clip else None,
            on_timing_event=on_timing_event,
        )
        pending_seen = bool(synth_result.get("pending"))
        if pending_seen:
            if not done.wait(timeout=wait_for_remaining_timeout_s):
                ok = False
                error_s = "timeout_waiting_for_remaining_clips"
            elif stream_error[0] is not None:
                ok = False
                error_s = repr(stream_error[0])
            synth_result = final_result[0] or synth_result
        else:
            # Sync path already emitted remaining_audio_complete.
            done.set()
    except Exception as ex:
        ok = False
        error_s = repr(ex)
        synth_result = synth_result or {}
    finally:
        if prev_output is not None:
            server_mod.OUTPUT_DIR = prev_output

    t_end = time.perf_counter()

    def _first_t(name: str) -> float | None:
        with events_lock:
            for ev in events:
                if ev.event == name:
                    return ev.t
        return None

    def _payload(name: str, key: str) -> Any:
        with events_lock:
            for ev in events:
                if ev.event == name and key in ev.payload:
                    return ev.payload[key]
        return None

    start = _first_t("tts_request_start") or t0
    seg_s = _first_t("segmentation_start")
    seg_e = _first_t("segmentation_complete")
    first_ready = _first_t("first_audio_ready")
    remaining_done = _first_t("remaining_audio_complete") or t_end

    server_ttfa_ms = (
        (first_ready - start) * 1000.0 if first_ready is not None else None
    )
    complete_ms = (remaining_done - start) * 1000.0
    segmentation_ms = (
        (seg_e - seg_s) * 1000.0 if seg_s is not None and seg_e is not None else None
    )

    acquire_waits: list[tuple[int, float]] = []
    synth_ms_list: list[float] = []
    with events_lock:
        for ev in events:
            if ev.event == "segment_pool_metrics":
                idx = int(ev.payload.get("index", -1))
                aw = ev.payload.get("acquire_wait_ms")
                sm = ev.payload.get("synth_ms")
                if isinstance(aw, (int, float)):
                    acquire_waits.append((idx, float(aw)))
                if isinstance(sm, (int, float)):
                    synth_ms_list.append(float(sm))

    first_acq = next((v for i, v in acquire_waits if i == 0), None)
    follow_sum = sum(v for i, v in acquire_waits if i > 0) or None
    if follow_sum == 0 and not any(i > 0 for i, _ in acquire_waits):
        follow_sum = None

    segments_meta = _payload("segmentation_complete", "segments") or []
    clip_paths = [str(p) for p in (synth_result or {}).get("clip_paths") or []]
    durations: list[float] = []
    for p in clip_paths:
        d = wav_duration_seconds(Path(p))
        durations.append(float(d) if d is not None else 0.0)

    playback = None
    if simulate_playback and first_ready is not None and clip_paths:
        ready_times: list[float] = []
        with events_lock:
            ready_by_index: dict[int, float] = {}
            for ev in events:
                if ev.event == "segment_ready":
                    ready_by_index[int(ev.payload.get("index", -1))] = ev.t
        for i in range(len(clip_paths)):
            ready_times.append(ready_by_index.get(i, first_ready if i == 0 else remaining_done))
        playback = simulate_sequential_playback(
            ready_times_s=ready_times,
            durations_s=durations,
            origin_s=start,
        )

    return PipelineResult(
        request_id=rid,
        ok=ok and error_s is None,
        error=error_s,
        server_ttfa_ms=server_ttfa_ms,
        complete_ms=complete_ms,
        segmentation_ms=segmentation_ms,
        first_acquire_wait_ms=first_acq,
        first_synth_ms=(
            float(_payload("first_inference_complete", "synth_ms"))
            if _payload("first_inference_complete", "synth_ms") is not None
            else None
        ),
        first_postprocess_ms=_first_postprocess_ms(events),
        segment_count=len(segments_meta) if segments_meta else len(clip_paths),
        segments=list(segments_meta) if isinstance(segments_meta, list) else [],
        clip_paths=clip_paths,
        clip_durations_s=durations,
        total_inference_ms=sum(synth_ms_list) if synth_ms_list else None,
        first_chunk_acquire_wait_ms=first_acq,
        followup_acquire_wait_ms_sum=follow_sum,
        events=list(events),
        pending_seen=pending_seen,
        pool_metrics_end=piper_pool.pool_runtime_metrics(),
        playback=playback,
    )
