#!/usr/bin/env python3
"""
S1 v2: production-equivalent segmented Piper TTS / server-side TTFA harness.

In-process only. No OpenRouter, Discord, or customer sessions.
Uses synthesize_segmented_speech_clips with stream_clips semantics.

Examples:

  .\\.venv\\Scripts\\python.exe scripts\\run_s1v2_ttfa.py ^
    --voice C:\\AI\\Voices\\en_US-libritts-high.onnx ^
    --platform-root C:\\Users\\jayhi\\source\\repos\\NPC-Platform ^
    --scenario ab --json-out reports\\s1v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import piper_pool  # noqa: E402
from s1v2_workload import (  # noqa: E402
    ArrivalEvent,
    build_capacity_summary,
    build_poisson_arrivals,
    estimate_duration_seconds,
    small_n_warning,
    summarize_latencies,
    variability,
)
from tts_benchmark_pipeline import PipelineResult, run_production_tts_pipeline  # noqa: E402


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _cpu_seconds() -> float | None:
    try:
        import psutil

        t = psutil.Process(os.getpid()).cpu_times()
        return float(t.user + t.system)
    except Exception:
        return None


def _host_cpu_percent() -> float | None:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def _load_platform(platform_root: Path) -> None:
    if str(platform_root) not in sys.path:
        sys.path.insert(0, str(platform_root))


def _assert_safety(platform_root: Path, *, max_concurrency: int, est_duration: float) -> None:
    _load_platform(platform_root)
    from perf.common.config import load_allowlist, load_harness_config
    from perf.common.safety import assert_suite_safe, build_policy

    cfg = dict(load_harness_config(platform_root / "perf" / "config" / "default.yaml"))
    cfg["max_duration_seconds"] = max(
        int(cfg.get("max_duration_seconds") or 600), int(est_duration) + 120
    )
    cfg["max_concurrency"] = max(int(cfg.get("max_concurrency") or 10), max_concurrency)
    cfg["allow_openrouter"] = False
    cfg["allow_discord_gateway"] = False
    policy = build_policy(cfg, load_allowlist(cfg))
    assert_suite_safe(
        policy,
        base_url=str(cfg.get("middleware_base_url") or "http://127.0.0.1:5000"),
        concurrency=max_concurrency,
        duration_seconds=est_duration,
        uses_openrouter=False,
        uses_discord_gateway=False,
    )


def _load_fixtures(platform_root: Path) -> dict[str, dict[str, Any]]:
    _load_platform(platform_root)
    from perf.common.fixtures import load_dialogue_fixture

    names = ("short", "medium", "long")
    return {
        n: load_dialogue_fixture(n, fixtures_dir=platform_root / "perf" / "fixtures")
        for n in names
    }


def _load_assumptions(platform_root: Path) -> dict[str, Any]:
    path = platform_root / "perf" / "config" / "capacity_assumptions.yaml"
    if not path.is_file():
        return {
            "players_per_table": 6,
            "npc_interactions_per_session": 60,
            "session_hours": 2,
            "peak_traffic_multiplier": 2.0,
            "dialogue_intensive_multiplier": 1.0,
            "ttfa_p95_target_ms": 2500,
            "max_stall_rate": 0.05,
        }
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return dict(yaml.safe_load(fh) or {})


def _rebuild_pool(models: list[Path], *, slots: int, intra: int | None) -> dict[str, Any]:
    if intra is None:
        pass
    elif intra <= 0:
        os.environ["PIPER_ONNX_INTRA_OP_THREADS"] = "0"
    else:
        os.environ["PIPER_ONNX_INTRA_OP_THREADS"] = str(intra)
    os.environ["PIPER_POOL_SLOTS"] = str(slots)
    piper_pool.clear_pool_for_tests()
    t0 = time.perf_counter()
    status = piper_pool.warm_voice_models(
        models,
        request_id="s1v2-warm",
        wait_for_ready=True,
        timeout_seconds=600.0,
    )
    return {
        "intra_op_threads": piper_pool.onnx_intra_op_threads() if hasattr(piper_pool, "onnx_intra_op_threads") else intra,
        "warm_seconds": time.perf_counter() - t0,
        "pool_status": status or piper_pool.pool_status(),
        "peak_rss_after_warm": _rss_bytes(),
    }


def _result_to_row(r: PipelineResult, *, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "request_id": r.request_id,
        "ok": r.ok,
        "error": r.error,
        "server_ttfa_ms": r.server_ttfa_ms,
        "complete_ms": r.complete_ms,
        "segmentation_ms": r.segmentation_ms,
        "first_acquire_wait_ms": r.first_acquire_wait_ms,
        "first_synth_ms": r.first_synth_ms,
        "first_postprocess_ms": r.first_postprocess_ms,
        "segment_count": r.segment_count,
        "total_inference_ms": r.total_inference_ms,
        "first_chunk_acquire_wait_ms": r.first_chunk_acquire_wait_ms,
        "followup_acquire_wait_ms_sum": r.followup_acquire_wait_ms_sum,
        "pending_seen": r.pending_seen,
        "audio_seconds": sum(r.clip_durations_s),
        "stall_count": (r.playback or {}).get("stall_count"),
        "total_stall_ms": (r.playback or {}).get("total_stall_ms"),
        "largest_gap_ms": (r.playback or {}).get("largest_gap_ms"),
        "metric_label": "server_ttfa_ms_not_browser_playback_start",
    }


def _run_one(
    *,
    text: str,
    voice: Path,
    speaker_id: int | None,
    out_dir: Path,
    simulate_playback: bool,
) -> PipelineResult:
    from tts_processing import resolve_tts_options

    raw: dict[str, Any] = {}
    if speaker_id is not None:
        raw["tts_speaker_id"] = speaker_id
    opts = resolve_tts_options(raw)
    return run_production_tts_pipeline(
        text,
        npc_voice_model=voice,
        npc_tts_options=opts,
        return_after_first_clip=True,
        output_dir=out_dir,
        simulate_playback=simulate_playback,
    )


def _aggregate_responses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    ttfa = [float(r["server_ttfa_ms"]) for r in ok if r.get("server_ttfa_ms") is not None]
    complete = [float(r["complete_ms"]) for r in ok if r.get("complete_ms") is not None]
    audio = [float(r["audio_seconds"]) for r in ok if r.get("audio_seconds") is not None]
    stalls = [r for r in ok if int(r.get("stall_count") or 0) > 0]
    wall_span = None
    if rows:
        # approximate from complete max when simultaneous
        if complete:
            wall_span = max(complete) / 1000.0
    n_ok = len(ok)
    rps = (n_ok / wall_span) if wall_span and wall_span > 0 else None
    audio_rate = (sum(audio) / wall_span) if wall_span and wall_span > 0 and audio else None
    warn = small_n_warning(n_ok)
    return {
        "requests": len(rows),
        "successes": n_ok,
        "failures": len(rows) - n_ok,
        "server_ttfa_ms": summarize_latencies(ttfa),
        "complete_ms": summarize_latencies(complete),
        "throughput_responses_per_s": rps,
        "audio_seconds_per_s": audio_rate,
        "playback_stall_rate": (len(stalls) / n_ok) if n_ok else None,
        "sample_size_warning": warn,
        "tail_statistically_exploratory": bool(warn),
    }


def run_burst_cell(
    *,
    text: str,
    dialogue_class: str,
    voice: Path,
    concurrency: int,
    iterations: int,
    out_root: Path,
    simulate_playback: bool,
    speaker_id: int | None = None,
) -> dict[str, Any]:
    """Scenario A/B: launch `iterations` responses with up to `concurrency` in flight."""
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    peak_box = [_rss_bytes() or 0]
    cpu0 = _cpu_seconds()
    host0 = _host_cpu_percent()
    t0 = time.perf_counter()

    def one_wrap(i: int) -> None:
        out_dir = out_root / f"c{concurrency}-{dialogue_class}-{i}"
        r = _run_one(
            text=text,
            voice=voice,
            speaker_id=speaker_id,
            out_dir=out_dir,
            simulate_playback=simulate_playback,
        )
        row = _result_to_row(
            r,
            meta={
                "dialogue_class": dialogue_class,
                "concurrency": concurrency,
                "iteration": i,
                "voice_model": str(voice),
                "speaker_id": speaker_id,
            },
        )
        with lock:
            rows.append(row)
            rss = _rss_bytes()
            if rss is not None:
                peak_box[0] = max(peak_box[0], rss)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = [pool.submit(one_wrap, i) for i in range(iterations)]
        for fut in as_completed(futs):
            fut.result()

    wall = time.perf_counter() - t0
    cpu1 = _cpu_seconds()
    cpu_delta = (cpu1 - cpu0) if cpu0 is not None and cpu1 is not None else None
    ok_n = sum(1 for r in rows if r.get("ok"))
    agg = _aggregate_responses(rows)
    # Fix throughput using true wall clock of the cell
    agg["throughput_responses_per_s"] = (ok_n / wall) if wall > 0 else None
    audio_sum = sum(float(r.get("audio_seconds") or 0) for r in rows if r.get("ok"))
    agg["audio_seconds_per_s"] = (audio_sum / wall) if wall > 0 else None
    agg["cell_wall_seconds"] = wall
    agg["cpu_seconds_delta"] = cpu_delta
    agg["cpu_seconds_per_response_approx"] = (
        (cpu_delta / ok_n) if cpu_delta is not None and ok_n > 0 else "unavailable"
    )
    agg["peak_rss_bytes"] = peak_box[0] or "unavailable"
    agg["host_cpu_percent_sample"] = _host_cpu_percent() if host0 is not None else "unavailable"
    agg["pool_status"] = piper_pool.pool_status()
    agg["pool_runtime"] = piper_pool.pool_runtime_metrics()
    return {
        "dialogue_class": dialogue_class,
        "concurrency": concurrency,
        "iterations": iterations,
        "voice_model": str(voice),
        **agg,
        "responses": rows,
    }


def run_arrival_scenario(
    *,
    events: list[ArrivalEvent],
    out_root: Path,
    simulate_playback: bool,
    max_in_flight: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    peak_box = [_rss_bytes() or 0]
    cpu0 = _cpu_seconds()
    start = time.perf_counter()
    sem = threading.Semaphore(max(1, max_in_flight))

    def worker(ev: ArrivalEvent) -> None:
        delay = ev.arrive_at_s - (time.perf_counter() - start)
        if delay > 0:
            time.sleep(delay)
        with sem:
            out_dir = out_root / f"arr-{ev.response_index}"
            r = _run_one(
                text=ev.text,
                voice=Path(ev.voice_model),
                speaker_id=ev.speaker_id,
                out_dir=out_dir,
                simulate_playback=simulate_playback,
            )
            row = _result_to_row(
                r,
                meta={
                    "dialogue_class": ev.dialogue_class,
                    "arrival_s": ev.arrive_at_s,
                    "response_index": ev.response_index,
                    "voice_model": ev.voice_model,
                    "speaker_id": ev.speaker_id,
                    "concurrency": max_in_flight,
                },
            )
            with lock:
                rows.append(row)
                rss = _rss_bytes()
                if rss is not None:
                    peak_box[0] = max(peak_box[0], rss)

    with ThreadPoolExecutor(max_workers=max(1, max_in_flight)) as pool:
        futs = [pool.submit(worker, ev) for ev in events]
        for fut in as_completed(futs):
            fut.result()

    wall = time.perf_counter() - start
    cpu1 = _cpu_seconds()
    cpu_delta = (cpu1 - cpu0) if cpu0 is not None and cpu1 is not None else None
    ok_n = sum(1 for r in rows if r.get("ok"))
    agg = _aggregate_responses(rows)
    agg["throughput_responses_per_s"] = (ok_n / wall) if wall > 0 else None
    audio_sum = sum(float(r.get("audio_seconds") or 0) for r in rows if r.get("ok"))
    agg["audio_seconds_per_s"] = (audio_sum / wall) if wall > 0 else None
    agg["cell_wall_seconds"] = wall
    agg["cpu_seconds_delta"] = cpu_delta
    agg["cpu_seconds_per_response_approx"] = (
        (cpu_delta / ok_n) if cpu_delta is not None and ok_n > 0 else "unavailable"
    )
    agg["peak_rss_bytes"] = peak_box[0] or "unavailable"
    agg["arrival_timestamps"] = [
        {"response_index": e.response_index, "arrive_at_s": e.arrive_at_s, "dialogue_class": e.dialogue_class}
        for e in events
    ]
    return {"max_in_flight": max_in_flight, **agg, "responses": rows}


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# S1 v2 — Production-equivalent Piper TTS / server-side TTFA",
        "",
        f"- Suite: `{report.get('suite')}`",
        f"- Capacity kind: `{report.get('capacity_kind')}`",
        f"- Primary metric: **server_ttfa_ms** (not browser playback-start)",
        f"- Voice: `{report.get('voice_model')}`",
        f"- ORT intra-op: `{report.get('ort_intra_op_threads')}`",
        "",
        "## Scenario A/B cells",
        "",
        "| dialogue | c | n | TTFA p50 | TTFA p95 | complete p95 | rps | stall rate | warn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in report.get("cells") or []:
        ttfa = cell.get("server_ttfa_ms") or {}
        comp = cell.get("complete_ms") or {}
        lines.append(
            "| {dlg} | {c} | {n} | {p50} | {p95} | {cp95} | {rps} | {stall} | {warn} |".format(
                dlg=cell.get("dialogue_class"),
                c=cell.get("concurrency"),
                n=cell.get("successes"),
                p50=_fmt(ttfa.get("p50")),
                p95=_fmt(ttfa.get("p95")),
                cp95=_fmt(comp.get("p95")),
                rps=_fmt(cell.get("throughput_responses_per_s")),
                stall=_fmt(cell.get("playback_stall_rate"), 3),
                warn="yes" if cell.get("sample_size_warning") else "",
            )
        )
    if report.get("ort_comparison"):
        lines.extend(["", "## ORT comparison (medium / representative)", ""])
        lines.append(json.dumps(report["ort_comparison"], indent=2))
    if report.get("capacity_summary"):
        lines.extend(["", "## Capacity summary", ""])
        lines.append(json.dumps(report["capacity_summary"], indent=2))
    if report.get("pool_scheduler_notes"):
        lines.extend(["", "## Pool / scheduler observations", ""])
        for note in report["pool_scheduler_notes"]:
            lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- S1 v1 (preview full-utterance) is unchanged and separate.",
            "- Within one response, clips synthesize sequentially; concurrency is across responses.",
            "- Playback stalls are simulated sequential timelines from actual WAV durations.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def run_s1v2(args: argparse.Namespace) -> dict[str, Any]:
    platform_root = Path(args.platform_root)
    voice = Path(args.voice)
    if not voice.is_file():
        raise FileNotFoundError(voice)
    secondary = Path(args.secondary_voice) if args.secondary_voice else None
    fixtures = _load_fixtures(platform_root)
    texts = {k: v["text"] for k, v in fixtures.items()}
    assumptions = _load_assumptions(platform_root)

    ladder = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    ort_grid = [int(x) for x in str(args.ort_threads).split(",") if x.strip()] if args.ort_threads else [None]
    mix = {"short": 0.4, "medium": 0.5, "long": 0.1}

    max_c = max(ladder) if ladder else 1
    est = estimate_duration_seconds(
        n_responses=max(int(args.iterations) * len(classes) * len(ladder) * max(1, len(ort_grid)), 8),
        concurrency=max_c,
    )
    if args.scenario in ("c", "all", "arrival"):
        est += 300
    _assert_safety(platform_root, max_concurrency=max_c, est_duration=est)

    out_root = Path(tempfile.mkdtemp(prefix="s1v2-"))
    cells: list[dict[str, Any]] = []
    arrival_cells: list[dict[str, Any]] = []
    ort_comparison: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    warms: list[dict[str, Any]] = []

    models = [voice]
    if secondary and secondary.is_file() and args.model_mode in ("both", "secondary"):
        models.append(secondary)

    speaker_ids = [None]
    if args.speaker_ids:
        speaker_ids = [int(x) for x in str(args.speaker_ids).split(",") if x.strip()]

    for intra in ort_grid:
        warm = _rebuild_pool(models if args.model_mode != "secondary" else [secondary or voice], slots=int(args.slots), intra=intra)
        warms.append(warm)
        # Warmup synths excluded from measurements
        for _ in range(max(0, int(args.warmup))):
            _run_one(
                text=texts["short"],
                voice=voice if args.model_mode != "secondary" else (secondary or voice),
                speaker_id=speaker_ids[0],
                out_dir=out_root / "warmup",
                simulate_playback=False,
            )

        active_voice = voice
        if args.model_mode == "secondary" and secondary:
            active_voice = secondary

        if args.scenario in ("a", "b", "ab", "all"):
            for dialogue in classes:
                for concurrency in ladder if args.scenario != "a" else [1]:
                    cell = run_burst_cell(
                        text=texts[dialogue],
                        dialogue_class=dialogue,
                        voice=active_voice,
                        concurrency=concurrency,
                        iterations=int(args.iterations),
                        out_root=out_root / f"ort{intra}",
                        simulate_playback=bool(args.simulate_playback),
                        speaker_id=speaker_ids[0],
                    )
                    cell["ort_intra_op_threads"] = warm.get("intra_op_threads")
                    cell["model_mode"] = args.model_mode
                    cells.append(cell)
                    all_responses.extend(cell.get("responses") or [])
                    print(
                        f"[S1V2] ort={intra} {dialogue} c={concurrency} "
                        f"ttfa_p95={(cell.get('server_ttfa_ms') or {}).get('p95')} "
                        f"complete_p95={(cell.get('complete_ms') or {}).get('p95')}",
                        flush=True,
                    )

        if args.scenario in ("c", "d", "all", "arrival"):
            patterns = [
                ("low", "poisson", 0.05),
                ("moderate", "poisson", 0.2),
                ("intensive", "poisson", 0.5),
                ("sync_burst", "sync_burst", 1.0),
                ("sustained", "sustained", 0.25),
            ]
            for name, pattern, rate in patterns:
                use_mix = mix if args.scenario in ("d", "all") or args.mixed else None
                events = build_poisson_arrivals(
                    n=int(args.arrival_n),
                    rate_per_s=rate,
                    seed=int(args.seed) + hash(name) % 10000,
                    dialogue_classes=classes,
                    texts=texts,
                    mix=use_mix,
                    voice_model=str(active_voice),
                    speaker_ids=speaker_ids,
                    pattern=pattern,
                )
                # Persist arrivals
                arr_path = out_root / f"arrivals-{name}-ort{intra}.json"
                arr_path.write_text(
                    json.dumps([asdict(e) for e in events], indent=2),
                    encoding="utf-8",
                )
                cell = run_arrival_scenario(
                    events=events,
                    out_root=out_root / f"arr-{name}-ort{intra}",
                    simulate_playback=bool(args.simulate_playback),
                    max_in_flight=max_c,
                )
                cell["arrival_pattern"] = name
                cell["ort_intra_op_threads"] = warm.get("intra_op_threads")
                cell["arrivals_file"] = str(arr_path)
                arrival_cells.append(cell)
                all_responses.extend(cell.get("responses") or [])

        if args.scenario in ("ort", "all") or (len(ort_grid) > 1 and args.scenario in ("ab", "b")):
            # Representative medium @ c=4 for ORT comparison table
            medium_cells = [
                c
                for c in cells
                if c.get("dialogue_class") == "medium"
                and c.get("concurrency") == min(4, max(ladder))
                and c.get("ort_intra_op_threads") == warm.get("intra_op_threads")
            ]
            for c in medium_cells:
                ort_comparison.append(
                    {
                        "ort_intra_op_threads": c.get("ort_intra_op_threads"),
                        "server_ttfa_ms": c.get("server_ttfa_ms"),
                        "complete_ms": c.get("complete_ms"),
                        "throughput_responses_per_s": c.get("throughput_responses_per_s"),
                        "cpu_seconds_per_response_approx": c.get("cpu_seconds_per_response_approx"),
                        "peak_rss_bytes": c.get("peak_rss_bytes"),
                    }
                )

        if args.model_mode == "both" and secondary and secondary.is_file():
            # Alternate models under shared pool
            for dialogue in ("medium",):
                for concurrency in ([4] if 4 in ladder else ladder[:1]):
                    # half iterations each model via arrival mix
                    events = []
                    for i in range(int(args.iterations)):
                        v = voice if i % 2 == 0 else secondary
                        events.append(
                            ArrivalEvent(
                                response_index=i,
                                arrive_at_s=0.0,
                                dialogue_class=dialogue,
                                text=texts[dialogue],
                                voice_model=str(v),
                                speaker_id=speaker_ids[i % len(speaker_ids)],
                            )
                        )
                    cell = run_arrival_scenario(
                        events=events,
                        out_root=out_root / f"both-{dialogue}-c{concurrency}-ort{intra}",
                        simulate_playback=bool(args.simulate_playback),
                        max_in_flight=concurrency,
                    )
                    cell["model_mode"] = "both"
                    cell["dialogue_class"] = dialogue
                    cell["concurrency"] = concurrency
                    cell["ort_intra_op_threads"] = warm.get("intra_op_threads")
                    # Split by model
                    by_model: dict[str, list] = {}
                    for r in cell.get("responses") or []:
                        by_model.setdefault(str(r.get("voice_model")), []).append(r)
                    cell["by_model"] = {
                        k: _aggregate_responses(v) for k, v in by_model.items()
                    }
                    cells.append(cell)
                    all_responses.extend(cell.get("responses") or [])

    # Capacity from best moderate / c=4 medium cell or arrival moderate
    capacity_source = None
    for c in cells:
        if c.get("dialogue_class") == "medium" and c.get("concurrency") == min(4, max(ladder) if ladder else 4):
            capacity_source = c
            break
    if capacity_source is None and arrival_cells:
        capacity_source = next(
            (c for c in arrival_cells if c.get("arrival_pattern") == "moderate"),
            arrival_cells[0],
        )

    ttfa_target = float(assumptions.get("ttfa_p95_target_ms") or 2500)
    max_stall = float(assumptions.get("max_stall_rate") or 0.05)
    burst_ok = None
    for concurrency in sorted(set(ladder)):
        class_ok = []
        for name in classes:
            matches = [
                c
                for c in cells
                if c.get("concurrency") == concurrency and c.get("dialogue_class") == name
            ]
            if not matches:
                class_ok.append(False)
                continue
            ttfa_p95 = (matches[0].get("server_ttfa_ms") or {}).get("p95")
            stall = matches[0].get("playback_stall_rate")
            ok = (
                isinstance(ttfa_p95, (int, float))
                and ttfa_p95 <= ttfa_target
                and (stall is None or float(stall) <= max_stall)
                and int(matches[0].get("failures") or 0) == 0
            )
            class_ok.append(ok)
        if class_ok and all(class_ok):
            burst_ok = concurrency
        else:
            break

    measured = {
        "sustainable_responses_per_s": (capacity_source or {}).get("throughput_responses_per_s"),
        "sustainable_audio_seconds_per_s": (capacity_source or {}).get("audio_seconds_per_s"),
        "burst_concurrency_before_fail": burst_ok,
        "ttfa_p50_ms": ((capacity_source or {}).get("server_ttfa_ms") or {}).get("p50"),
        "ttfa_p95_ms": ((capacity_source or {}).get("server_ttfa_ms") or {}).get("p95"),
        "complete_p95_ms": ((capacity_source or {}).get("complete_ms") or {}).get("p95"),
        "ttfa_target_pass_rate": None,
        "playback_stall_rate": (capacity_source or {}).get("playback_stall_rate"),
        "sample_size": (capacity_source or {}).get("successes") or 0,
    }
    if capacity_source:
        ok_rows = [r for r in capacity_source.get("responses") or [] if r.get("ok")]
        if ok_rows:
            passed = sum(
                1
                for r in ok_rows
                if isinstance(r.get("server_ttfa_ms"), (int, float))
                and float(r["server_ttfa_ms"]) <= ttfa_target
            )
            measured["ttfa_target_pass_rate"] = passed / len(ok_rows)

    capacity_summary = build_capacity_summary(measured=measured, assumptions=assumptions)

    # Pool scheduler notes (observe only)
    follow_delay_note = (
        "Production does not prioritize first-chunk acquires over follow-up segments. "
        "Under concurrency, follow-up clips from early responses share the same pool and "
        "can increase first-chunk acquire wait for other responses."
    )
    high_follow = [
        c
        for c in cells
        if isinstance(c.get("concurrency"), int)
        and c["concurrency"] >= 4
        and (c.get("server_ttfa_ms") or {}).get("p95") is not None
    ]
    pool_notes = [
        follow_delay_note,
        "Recommendation (not implemented): consider reserved slots or first-chunk priority "
        "if first-chunk acquire waits dominate TTFA under mixed follow-up load.",
        f"Observed burst concurrency meeting TTFA/stall targets: {burst_ok}",
        f"High-concurrency cells examined: {len(high_follow)}",
    ]

    report = {
        "suite": "s1v2",
        "capacity_kind": "production_tts_pipeline_capacity",
        "not_production_browser_playback": True,
        "primary_metric": "server_ttfa_ms",
        "primary_metric_definition": (
            "first_audio_ready - tts_request_start after clip-0 Piper+ffmpeg; "
            "not player-perceived browser playback start"
        ),
        "voice_model": str(voice),
        "secondary_voice": str(secondary) if secondary else None,
        "model_mode": args.model_mode,
        "ort_intra_op_threads": ort_grid,
        "slots": int(args.slots),
        "seed": int(args.seed),
        "fixtures": {
            k: {
                "word_count": v["word_count"],
                "char_count": v["char_count"],
                "text_sha256": v["text_sha256"],
            }
            for k, v in fixtures.items()
        },
        "safety": {
            "allow_openrouter": False,
            "allow_discord_gateway": False,
            "calls_openrouter": False,
            "calls_discord": False,
            "in_process_only": True,
        },
        "warms": warms,
        "cells": [{k: v for k, v in c.items() if k != "responses"} for c in cells],
        "arrival_cells": [{k: v for k, v in c.items() if k != "responses"} for c in arrival_cells],
        "responses": all_responses,
        "ort_comparison": ort_comparison,
        "capacity_summary": capacity_summary,
        "burst_concurrency_capacity": burst_ok,
        "sustained_table_capacity": capacity_summary.get("extrapolated", {}).get(
            "estimated_active_tables_per_piper_worker"
        ),
        "pool_scheduler_notes": pool_notes,
        "scratch_dir": str(out_root),
        "notes": [
            "S1 v1 preview full-utterance benchmark remains separate and unchanged.",
            "CPU per response is aggregate process CPU delta / completions (approximate).",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voice", type=Path, required=True, help="Primary .onnx (e.g. LibriTTS High)")
    p.add_argument("--secondary-voice", type=Path, default=None, help="Clean 100 or second model")
    p.add_argument(
        "--platform-root",
        type=Path,
        default=Path(os.environ.get("NPC_PLATFORM_ROOT") or r"C:\Users\jayhi\source\repos\NPC-Platform"),
    )
    p.add_argument("--scenario", default="ab", help="a|b|ab|c|d|ort|arrival|all")
    p.add_argument("--concurrency", default="1,2,4,6,8,10")
    p.add_argument("--classes", default="short,medium,long")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--slots", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arrival-n", type=int, default=24)
    p.add_argument("--mixed", action="store_true", help="Use 40/50/10 mix in arrival scenarios")
    p.add_argument("--ort-threads", default="", help="Comma list e.g. 0,2,4 (empty=current env)")
    p.add_argument("--model-mode", default="primary", choices=["primary", "secondary", "both"])
    p.add_argument("--speaker-ids", default="", help="Comma speaker ids for LibriTTS variation")
    p.add_argument("--simulate-playback", action="store_true", default=True)
    p.add_argument("--no-simulate-playback", action="store_false", dest="simulate_playback")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--reports-dir", type=Path, default=None, help="Platform-style reports dir")
    args = p.parse_args(argv)

    try:
        report = run_s1v2(args)
    except Exception as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1

    text = json.dumps(report, indent=2, default=str)
    md = _markdown_summary(report)

    if args.reports_dir:
        _load_platform(Path(args.platform_root))
        from perf.common.config import load_harness_config
        from perf.report.writers import report_dir, write_csv_rows, write_json

        cfg = load_harness_config()
        out = report_dir(
            Path(args.reports_dir),
            str(cfg.get("environment_id") or "local"),
            "s1v2",
        )
        write_json(out / "report.json", report)
        write_json(out / "capacity_summary.json", report.get("capacity_summary") or {})
        write_csv_rows(out / "responses.csv", report.get("responses") or [])
        write_csv_rows(out / "cells.csv", report.get("cells") or [])
        (out / "summary.md").write_text(md, encoding="utf-8")
        print(f"Wrote {out}", flush=True)
        report["report_dir"] = str(out)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
        args.json_out.with_suffix(".md").write_text(md, encoding="utf-8")
        print(f"Wrote {args.json_out}", flush=True)

    print(md)
    fails = sum(1 for r in report.get("responses") or [] if not r.get("ok"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
