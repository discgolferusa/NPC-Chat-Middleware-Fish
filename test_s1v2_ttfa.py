"""Unit + functional tests for S1 v2 TTFA pipeline helpers."""

from __future__ import annotations

import threading
import time
from pathlib import Path
import s1v2_workload
from tts_benchmark_pipeline import (
    TimingEvent,
    simulate_sequential_playback,
    wav_duration_seconds,
)


def test_simulate_playback_no_stall_when_ready_early():
    origin = 100.0
    # Clip1 ready before clip0 finishes
    result = simulate_sequential_playback(
        ready_times_s=[origin, origin + 0.5],
        durations_s=[1.0, 1.0],
        origin_s=origin,
    )
    assert result["stall_count"] == 0
    assert result["total_stall_ms"] == 0.0
    assert result["label"].startswith("simulated_sequential")


def test_simulate_playback_detects_stall():
    origin = 0.0
    result = simulate_sequential_playback(
        ready_times_s=[0.0, 2.5],
        durations_s=[1.0, 1.0],
        origin_s=origin,
    )
    assert result["stall_count"] == 1
    assert abs(result["total_stall_ms"] - 1500.0) < 1e-6
    assert abs(result["largest_gap_ms"] - 1500.0) < 1e-6


def test_timing_boundaries_ordering():
    events = [
        TimingEvent("tts_request_start", 1.0),
        TimingEvent("first_audio_ready", 1.5, {"index": 0}),
        TimingEvent("remaining_audio_complete", 3.0),
    ]
    start = events[0].t
    first = events[1].t
    done = events[2].t
    assert start <= first <= done


def test_segmentation_equivalence_streaming(monkeypatch):
    from tts_processing import split_text_for_streaming

    text = (
        "Traveler, the market closes at dusk. If you seek the blacksmith, "
        "follow the lane past the well, then turn left at the crooked oak."
    )
    # Import server split after ensuring light path
    import server

    segs = server.split_into_ordered_tts_segments(text)
    assert segs
    assert all(s["type"] == "npc" for s in segs)
    # Same streaming chunks as production helper
    chunks = split_text_for_streaming(text)
    assert [s["text"] for s in segs] == chunks
    # First chunk respects first-stream budget
    assert len(segs[0]["text"]) <= 140 or len(chunks) == 1


def test_small_n_warning():
    assert s1v2_workload.small_n_warning(3) is not None
    assert s1v2_workload.small_n_warning(50) is None


def test_capacity_summary_separates_measured_and_assumed():
    summary = s1v2_workload.build_capacity_summary(
        measured={
            "sustainable_responses_per_s": 1.0,
            "sustainable_audio_seconds_per_s": 2.0,
            "burst_concurrency_before_fail": 4,
            "ttfa_p50_ms": 800,
            "ttfa_p95_ms": 1200,
            "complete_p95_ms": 4000,
            "playback_stall_rate": 0.0,
            "sample_size": 8,
        },
        assumptions={
            "players_per_table": 6,
            "npc_interactions_per_session": 60,
            "session_hours": 2,
            "peak_traffic_multiplier": 2.0,
            "dialogue_intensive_multiplier": 1.0,
            "ttfa_p95_target_ms": 2500,
            "max_stall_rate": 0.05,
        },
    )
    assert "measured" in summary and "assumed" in summary and "extrapolated" in summary
    assert summary["warnings"]  # small n
    assert summary["extrapolated"]["estimated_active_tables_per_piper_worker"] is not None


def test_poisson_arrivals_reproducible():
    texts = {"short": "a", "medium": "b", "long": "c"}
    a = s1v2_workload.build_poisson_arrivals(
        n=10,
        rate_per_s=0.5,
        seed=123,
        dialogue_classes=["short", "medium", "long"],
        texts=texts,
        mix={"short": 0.4, "medium": 0.5, "long": 0.1},
        voice_model="v.onnx",
        pattern="poisson",
    )
    b = s1v2_workload.build_poisson_arrivals(
        n=10,
        rate_per_s=0.5,
        seed=123,
        dialogue_classes=["short", "medium", "long"],
        texts=texts,
        mix={"short": 0.4, "medium": 0.5, "long": 0.1},
        voice_model="v.onnx",
        pattern="poisson",
    )
    assert [e.arrive_at_s for e in a] == [e.arrive_at_s for e in b]
    assert [e.dialogue_class for e in a] == [e.dialogue_class for e in b]


def test_pipeline_stream_early_return_sequential(monkeypatch, tmp_path):
    """Functional: return_after_first_clip schedules remaining; no parallel within response."""
    import tts_benchmark_pipeline as pipeline

    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def fake_synth_clips(
        npc_text,
        npc_voice_model,
        request_id,
        npc_length_scale=1.0,
        npc_tts_options=None,
        on_clip_ready=None,
        on_narration_ready=None,
        *,
        return_after_first_clip=False,
        on_stream_complete=None,
        on_timing_event=None,
    ):
        segments = [{"type": "npc", "text": "one."}, {"type": "npc", "text": "two."}]
        if callable(on_timing_event):
            on_timing_event("tts_request_start", t=time.perf_counter())
            on_timing_event(
                "segmentation_complete",
                t=time.perf_counter(),
                segment_count=2,
                segments=segments,
            )

        paths = []
        state = {"clips": []}

        def synth_one(index: int):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            try:
                if callable(on_timing_event) and index == 0:
                    on_timing_event("first_audio_ready", t=time.perf_counter(), index=0)
                out = tmp_path / f"{request_id}_{index}.wav"
                # minimal wav header-ish empty: use write of silence via wave
                import wave

                with wave.open(str(out), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(22050)
                    w.writeframes(b"\x00\x00" * 2205)
                paths.append(out)
                clip = {"type": "speech", "index": index, "url": "/", "text": segments[index]["text"]}
                state["clips"].append(clip)
                if callable(on_timing_event):
                    on_timing_event(
                        "segment_ready",
                        t=time.perf_counter(),
                        index=index,
                        path=str(out),
                    )
                if callable(on_clip_ready):
                    on_clip_ready(clip, list(state["clips"]))
            finally:
                with lock:
                    active["n"] -= 1

        synth_one(0)

        if return_after_first_clip:

            def rem():
                synth_one(1)
                if callable(on_timing_event):
                    on_timing_event("remaining_audio_complete", t=time.perf_counter())
                if callable(on_stream_complete):
                    on_stream_complete(
                        {
                            "segments": segments,
                            "clip_paths": paths,
                            "clips": state["clips"],
                            "pending": False,
                        },
                        None,
                    )

            threading.Thread(target=rem, daemon=True).start()
            return {
                "segments": segments,
                "clip_paths": paths[:1],
                "clips": state["clips"],
                "pending": True,
            }
        synth_one(1)
        return {
            "segments": segments,
            "clip_paths": paths,
            "clips": state["clips"],
            "pending": False,
        }

    import server

    monkeypatch.setattr(server, "synthesize_segmented_speech_clips", fake_synth_clips)
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)

    result = pipeline.run_production_tts_pipeline(
        "ignored",
        npc_voice_model=tmp_path / "v.onnx",
        return_after_first_clip=True,
        output_dir=tmp_path / "out",
        simulate_playback=True,
    )
    assert result.ok
    assert result.pending_seen
    assert result.server_ttfa_ms is not None
    assert result.complete_ms is not None
    assert result.server_ttfa_ms <= result.complete_ms
    assert active["max"] == 1  # no within-response parallel synth


def test_wav_duration(tmp_path):
    import wave

    path = tmp_path / "t.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 22050)
    d = wav_duration_seconds(path)
    assert d is not None
    assert abs(d - 1.0) < 0.01
