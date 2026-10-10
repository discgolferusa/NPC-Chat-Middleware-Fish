#!/usr/bin/env python3
"""
Controlled ONNX Runtime intra-op threading benchmark for in-process Piper.

Compares PIPER_ONNX_INTRA_OP_THREADS in {0,1,2,4} across short/medium/long
dialogue fixtures and concurrency ladder 1/2/4/6/8 (LibriTTS High).

Does NOT call OpenRouter or Discord. Does NOT change pool allocation or
synthesis logic beyond setting the opt-in env var and rebuilding sessions.

Each thread configuration:
  1. Sets PIPER_ONNX_INTRA_OP_THREADS
  2. clear_pool_for_tests() so prior InferenceSessions are dropped
  3. Warms all replicas (new ORT sessions with the thread setting)
  4. Measures after warm-up

Usage (middleware repo root, Voices available):

  .\\.venv\\Scripts\\python.exe scripts\\benchmark_piper_ort_threads.py ^
    --model C:\\AI\\Voices\\en_US-libritts-high.onnx ^
    --json-out reports\\ort-threads.json

Optional:
  --repeats 3
  --slots 8
  --platform-root C:\\Users\\jayhi\\source\\repos\\NPC-Platform
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import piper_pool  # noqa: E402
from piper import PiperVoice  # noqa: E402

THREAD_GRID = (0, 1, 2, 4)
CONCURRENCY_LADDER = (1, 2, 4, 6, 8)
DIALOGUE_CLASSES = ("short", "medium", "long")


@dataclass
class SynthSample:
    request_id: str
    wall_ms: float
    inference_ms: float | None
    acquire_wait_ms: float | None
    error: str | None = None


_TLS = threading.local()
_ORIG_PHONEME_IDS_TO_AUDIO = PiperVoice.phoneme_ids_to_audio


def _percentile(values: list[float], p: float) -> float | None:
    data = sorted(values)
    if not data:
        return None
    if p <= 0:
        return data[0]
    if p >= 100:
        return data[-1]
    k = int(round((len(data) - 1) * (p / 100.0)))
    return data[max(0, min(len(data) - 1, k))]


def summarize_latencies(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_repeat_variability(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/stdev across repeat cells for wall p50/p95 and throughput."""

    def _series(getter) -> list[float]:
        out: list[float] = []
        for r in repeats:
            v = getter(r)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    p50s = _series(lambda r: (r.get("wall_ms") or {}).get("p50"))
    p95s = _series(lambda r: (r.get("wall_ms") or {}).get("p95"))
    tps = _series(lambda r: r.get("throughput_rps"))

    def _mean_stdev(xs: list[float]) -> tuple[float | None, float | None]:
        if not xs:
            return None, None
        mean = statistics.fmean(xs)
        stdev = statistics.stdev(xs) if len(xs) >= 2 else 0.0
        return mean, stdev

    p50_m, p50_s = _mean_stdev(p50s)
    p95_m, p95_s = _mean_stdev(p95s)
    tp_m, tp_s = _mean_stdev(tps)
    return {
        "n_repeats": len(repeats),
        "wall_p50_mean": p50_m,
        "wall_p50_stdev": p50_s,
        "wall_p95_mean": p95_m,
        "wall_p95_stdev": p95_s,
        "throughput_rps_mean": tp_m,
        "throughput_rps_stdev": tp_s,
    }


def pick_best_intra_op(
    cells: list[dict[str, Any]],
    *,
    dialogue_class: str,
    concurrency: int,
) -> dict[str, Any]:
    """Lowest wall p95, then p50, then highest throughput."""
    candidates = [
        c
        for c in cells
        if c.get("dialogue_class") == dialogue_class
        and int(c.get("concurrency") or -1) == concurrency
        and isinstance(c.get("wall_ms"), dict)
        and c["wall_ms"].get("p95") is not None
    ]
    if not candidates:
        return {
            "dialogue_class": dialogue_class,
            "concurrency": concurrency,
            "intra_op_threads": None,
            "criterion": "lowest_wall_p95_then_p50",
            "note": "no candidates",
        }

    def key(c: dict[str, Any]) -> tuple:
        wall = c["wall_ms"]
        tp = c.get("throughput_rps")
        return (
            float(wall["p95"]),
            float(wall.get("p50") or wall["p95"]),
            -float(tp) if isinstance(tp, (int, float)) else 0.0,
        )

    best = min(candidates, key=key)
    return {
        "dialogue_class": dialogue_class,
        "concurrency": concurrency,
        "intra_op_threads": best.get("intra_op_threads"),
        "wall_p50_ms": best["wall_ms"].get("p50"),
        "wall_p95_ms": best["wall_ms"].get("p95"),
        "throughput_rps": best.get("throughput_rps"),
        "criterion": "lowest_wall_p95_then_p50",
    }


def _install_inference_timer() -> None:
    def phoneme_ids_to_audio(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        sample: SynthSample | None = getattr(_TLS, "sample", None)
        t0 = time.perf_counter()
        try:
            return _ORIG_PHONEME_IDS_TO_AUDIO(self, *args, **kwargs)
        finally:
            if sample is not None:
                elapsed = (time.perf_counter() - t0) * 1000.0
                prev = sample.inference_ms or 0.0
                sample.inference_ms = prev + elapsed

    PiperVoice.phoneme_ids_to_audio = phoneme_ids_to_audio  # type: ignore[method-assign]


def _restore_inference_timer() -> None:
    PiperVoice.phoneme_ids_to_audio = _ORIG_PHONEME_IDS_TO_AUDIO  # type: ignore[method-assign]


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


def _load_fixtures(platform_root: Path) -> dict[str, dict[str, Any]]:
    fixtures_dir = platform_root / "perf" / "fixtures"
    if not fixtures_dir.is_dir():
        raise FileNotFoundError(f"fixtures dir not found: {fixtures_dir}")
    sys.path.insert(0, str(platform_root))
    from perf.common.fixtures import load_dialogue_fixture

    return {name: load_dialogue_fixture(name, fixtures_dir=fixtures_dir) for name in DIALOGUE_CLASSES}


def _assert_safety(platform_root: Path, *, max_concurrency: int, est_duration: float) -> None:
    sys.path.insert(0, str(platform_root))
    from perf.common.config import load_allowlist, load_harness_config
    from perf.common.safety import SafetyError, assert_suite_safe, build_policy

    cfg = load_harness_config(platform_root / "perf" / "config" / "default.yaml")
    # This suite rebuilds pools and may exceed the S0/S1 600s default.
    cfg = dict(cfg)
    cfg["max_duration_seconds"] = max(int(cfg.get("max_duration_seconds") or 600), int(math.ceil(est_duration)) + 60)
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


def _synth_one(
    *,
    request_id: str,
    model_path: Path,
    text: str,
    out_dir: Path,
) -> SynthSample:
    sample = SynthSample(
        request_id=request_id,
        wall_ms=0.0,
        inference_ms=None,
        acquire_wait_ms=None,
    )
    _TLS.sample = sample
    out_path = out_dir / f"{request_id}.wav"
    t0 = time.perf_counter()
    try:
        piper_pool.synthesize_wav_file(
            model_path,
            out_path,
            text,
            length_scale=1.0,
            noise_scale=0.667,
            noise_w=0.8,
            sentence_silence=0.0,
            speaker_id=0,
        )
        wait = getattr(piper_pool._tls, "last_acquire_wait_ms", None)
        sample.acquire_wait_ms = float(wait) if wait is not None else None
    except Exception as ex:
        sample.error = repr(ex)
    finally:
        sample.wall_ms = (time.perf_counter() - t0) * 1000.0
        _TLS.sample = None
    return sample


def _run_wave(
    *,
    model_path: Path,
    text: str,
    concurrency: int,
    out_dir: Path,
    wave_id: str,
) -> dict[str, Any]:
    peak_rss = _rss_bytes() or 0
    cpu0 = _cpu_seconds()
    wall0 = time.perf_counter()
    samples: list[SynthSample] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(
                _synth_one,
                request_id=f"{wave_id}-r{i}",
                model_path=model_path,
                text=text,
                out_dir=out_dir,
            )
            for i in range(concurrency)
        ]
        for fut in as_completed(futs):
            s = fut.result()
            samples.append(s)
            rss = _rss_bytes()
            if rss is not None:
                peak_rss = max(peak_rss, rss)

    wall_s = time.perf_counter() - wall0
    cpu1 = _cpu_seconds()
    ok = [s for s in samples if s.error is None]
    walls = [s.wall_ms for s in samples]
    inf = [float(s.inference_ms) for s in ok if s.inference_ms is not None]
    cpu_delta = (cpu1 - cpu0) if cpu0 is not None and cpu1 is not None else None
    cpu_per = (cpu_delta / len(ok)) if cpu_delta is not None and ok else None

    return {
        "requests": len(samples),
        "successes": len(ok),
        "failures": len(samples) - len(ok),
        "wall_ms": summarize_latencies(walls),
        "inference_ms": summarize_latencies(inf) if inf else None,
        "throughput_rps": (len(ok) / wall_s) if wall_s > 0 else None,
        "wave_wall_seconds": wall_s,
        "cpu_seconds_delta": cpu_delta,
        "cpu_seconds_per_synthesis": cpu_per,
        "peak_rss_bytes": peak_rss if peak_rss else None,
        "errors": [s.error for s in samples if s.error],
        "samples": [asdict(s) for s in samples],
    }


def _rebuild_pool(model_path: Path, *, slots: int, intra: int) -> dict[str, Any]:
    """Drop all sessions and warm new ones under the given ORT thread setting."""
    if intra <= 0:
        os.environ.pop("PIPER_ONNX_INTRA_OP_THREADS", None)
        # Explicit 0 preserves automatic behavior and documents intent in status.
        os.environ["PIPER_ONNX_INTRA_OP_THREADS"] = "0"
    else:
        os.environ["PIPER_ONNX_INTRA_OP_THREADS"] = str(intra)
    os.environ["PIPER_POOL_SLOTS"] = str(slots)
    piper_pool.clear_pool_for_tests()
    print(
        f"[ORT-BENCH] rebuild pool intra_op={intra} slots={slots} model={model_path.name}",
        flush=True,
    )
    t0 = time.perf_counter()
    status = piper_pool.warm_voice_models(
        [model_path],
        request_id=f"ort-bench-warm-{intra}",
        wait_for_ready=True,
        timeout_seconds=600.0,
    )
    warm_s = time.perf_counter() - t0
    # Discard one cold synth per replica path (pool already loaded; one wave warms kernels).
    with tempfile.TemporaryDirectory(prefix="ort-bench-warm-") as td:
        warm_dir = Path(td)
        for i in range(min(slots, 4)):
            _synth_one(
                request_id=f"warm-{intra}-{i}",
                model_path=model_path,
                text="Warmup.",
                out_dir=warm_dir,
            )
    return {
        "intra_op_threads": intra,
        "effective_onnx_intra_op_threads": piper_pool.onnx_intra_op_threads(),
        "warm_seconds": warm_s,
        "pool_status": status or piper_pool.pool_status(),
        "peak_rss_bytes_after_warm": _rss_bytes(),
    }


def build_comparison(aggregated_cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Best configs for concurrency 1 / 4 / 8 across dialogue classes."""
    targets = (1, 4, 8)
    by_class: dict[str, Any] = {}
    for dialogue in DIALOGUE_CLASSES:
        by_class[dialogue] = {
            str(c): pick_best_intra_op(
                aggregated_cells, dialogue_class=dialogue, concurrency=c
            )
            for c in targets
        }
    # Overall recommendation: majority vote on medium (most representative),
    # broken by concurrency scenario.
    medium = by_class["medium"]
    return {
        "by_dialogue_class": by_class,
        "recommendations": {
            "single_user_latency_concurrency_1": medium["1"],
            "four_concurrent_requests": medium["4"],
            "eight_concurrent_requests": medium["8"],
        },
        "notes": [
            "Primary ranking uses medium fixture (36 words) wall p95 then p50.",
            "Per-class winners are listed under by_dialogue_class.",
            "intra_op_threads=0 is stock Piper/ORT automatic behavior.",
        ],
    }


def run_benchmark(
    *,
    model_path: Path,
    platform_root: Path,
    slots: int,
    repeats: int,
    thread_grid: tuple[int, ...] = THREAD_GRID,
    concurrency_ladder: tuple[int, ...] = CONCURRENCY_LADDER,
    json_out: Path | None = None,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not (Path(str(model_path) + ".json")).is_file():
        raise FileNotFoundError(f"{model_path}.json")

    fixtures = _load_fixtures(platform_root)
    max_c = max(concurrency_ladder)
    # Rough upper bound: thread configs × classes × ladder × repeats × ~15s/wave + warms
    est = (
        len(thread_grid) * 90
        + len(thread_grid) * len(DIALOGUE_CLASSES) * len(concurrency_ladder) * repeats * 15
    )
    _assert_safety(platform_root, max_concurrency=max_c, est_duration=float(est))

    out_root = Path(tempfile.mkdtemp(prefix="ort-bench-wav-"))
    _install_inference_timer()
    aggregated: list[dict[str, Any]] = []
    raw_repeats: list[dict[str, Any]] = []
    warms: list[dict[str, Any]] = []

    try:
        for intra in thread_grid:
            warm_info = _rebuild_pool(model_path, slots=slots, intra=intra)
            warms.append(warm_info)

            for dialogue in DIALOGUE_CLASSES:
                text = fixtures[dialogue]["text"]
                for concurrency in concurrency_ladder:
                    repeat_cells: list[dict[str, Any]] = []
                    for rep in range(repeats):
                        wave = _run_wave(
                            model_path=model_path.resolve(),
                            text=text,
                            concurrency=concurrency,
                            out_dir=out_root,
                            wave_id=f"t{intra}-{dialogue}-c{concurrency}-rep{rep}",
                        )
                        cell = {
                            "intra_op_threads": intra,
                            "dialogue_class": dialogue,
                            "concurrency": concurrency,
                            "repeat": rep,
                            "fixture_word_count": fixtures[dialogue]["word_count"],
                            "fixture_text_sha256": fixtures[dialogue]["text_sha256"],
                            **wave,
                        }
                        repeat_cells.append(cell)
                        raw_repeats.append(cell)
                        print(
                            f"[ORT-BENCH] intra={intra} {dialogue} c={concurrency} "
                            f"rep={rep} p50={wave['wall_ms'].get('p50')} "
                            f"p95={wave['wall_ms'].get('p95')} "
                            f"rps={wave.get('throughput_rps')}",
                            flush=True,
                        )

                    variability = summarize_repeat_variability(repeat_cells)
                    # Aggregate: use mean latencies across repeats for comparison cells
                    wall_p50s = [
                        c["wall_ms"]["p50"]
                        for c in repeat_cells
                        if c["wall_ms"].get("p50") is not None
                    ]
                    wall_p95s = [
                        c["wall_ms"]["p95"]
                        for c in repeat_cells
                        if c["wall_ms"].get("p95") is not None
                    ]
                    inf_p50s = [
                        c["inference_ms"]["p50"]
                        for c in repeat_cells
                        if isinstance(c.get("inference_ms"), dict)
                        and c["inference_ms"].get("p50") is not None
                    ]
                    inf_p95s = [
                        c["inference_ms"]["p95"]
                        for c in repeat_cells
                        if isinstance(c.get("inference_ms"), dict)
                        and c["inference_ms"].get("p95") is not None
                    ]
                    tps = [
                        c["throughput_rps"]
                        for c in repeat_cells
                        if isinstance(c.get("throughput_rps"), (int, float))
                    ]
                    cpu_per = [
                        c["cpu_seconds_per_synthesis"]
                        for c in repeat_cells
                        if isinstance(c.get("cpu_seconds_per_synthesis"), (int, float))
                    ]
                    peaks = [
                        c["peak_rss_bytes"]
                        for c in repeat_cells
                        if isinstance(c.get("peak_rss_bytes"), int)
                    ]
                    aggregated.append(
                        {
                            "intra_op_threads": intra,
                            "dialogue_class": dialogue,
                            "concurrency": concurrency,
                            "fixture_word_count": fixtures[dialogue]["word_count"],
                            "fixture_text_sha256": fixtures[dialogue]["text_sha256"],
                            "wall_ms": {
                                "p50": statistics.fmean(wall_p50s) if wall_p50s else None,
                                "p95": statistics.fmean(wall_p95s) if wall_p95s else None,
                            },
                            "inference_ms": {
                                "p50": statistics.fmean(inf_p50s) if inf_p50s else None,
                                "p95": statistics.fmean(inf_p95s) if inf_p95s else None,
                            },
                            "throughput_rps": statistics.fmean(tps) if tps else None,
                            "cpu_seconds_per_synthesis": (
                                statistics.fmean(cpu_per) if cpu_per else None
                            ),
                            "peak_rss_bytes": max(peaks) if peaks else None,
                            "variability": variability,
                            "successes_total": sum(int(c["successes"]) for c in repeat_cells),
                            "failures_total": sum(int(c["failures"]) for c in repeat_cells),
                        }
                    )
    finally:
        _restore_inference_timer()
        piper_pool.clear_pool_for_tests()

    comparison = build_comparison(aggregated)
    report = {
        "suite": "piper_ort_intra_op_threads",
        "model": str(model_path.resolve()),
        "slots": slots,
        "thread_grid": list(thread_grid),
        "concurrency_ladder": list(concurrency_ladder),
        "repeats": repeats,
        "fixtures": {
            k: {
                "word_count": v["word_count"],
                "char_count": v["char_count"],
                "text_sha256": v["text_sha256"],
                "fixture_relative": v["fixture_relative"],
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
        "session_construction_note": (
            "PiperVoice.load() does not expose SessionOptions. "
            "PIPER_ONNX_INTRA_OP_THREADS<=0 uses stock load(); "
            ">0 builds InferenceSession with sess_options.intra_op_num_threads "
            "in piper_pool._create_piper_voice (no site-packages edits). "
            "Each thread config clear_pool_for_tests() + rewarm so sessions are new."
        ),
        "warms": warms,
        "cells": aggregated,
        "raw_repeat_cells": raw_repeats,
        "comparison": comparison,
        "wav_scratch_dir": str(out_root),
    }

    text = json.dumps(report, indent=2)
    print("\n=== COMPARISON ===", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text, encoding="utf-8")
        print(f"Wrote {json_out}", flush=True)
        md_path = json_out.with_suffix(".md")
        md_path.write_text(_markdown_report(report), encoding="utf-8")
        print(f"Wrote {md_path}", flush=True)
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Piper ONNX Runtime intra-op threading comparison",
        "",
        f"- Model: `{report.get('model')}`",
        f"- Slots: {report.get('slots')}",
        f"- Thread grid: {report.get('thread_grid')}",
        f"- Concurrency: {report.get('concurrency_ladder')}",
        f"- Repeats: {report.get('repeats')}",
        "",
        "## Recommendations (medium fixture)",
        "",
    ]
    rec = (report.get("comparison") or {}).get("recommendations") or {}
    for key, label in (
        ("single_user_latency_concurrency_1", "Single-user latency (c=1)"),
        ("four_concurrent_requests", "Four concurrent (c=4)"),
        ("eight_concurrent_requests", "Eight concurrent (c=8)"),
    ):
        row = rec.get(key) or {}
        lines.append(
            f"- **{label}**: `PIPER_ONNX_INTRA_OP_THREADS={row.get('intra_op_threads')}` "
            f"(wall p50={row.get('wall_p50_ms')}, p95={row.get('wall_p95_ms')}, "
            f"rps={row.get('throughput_rps')})"
        )
    lines.extend(["", "## Aggregated cells", ""])
    lines.append(
        "| intra | dialogue | c | wall p50 | wall p95 | infer p50 | "
        "infer p95 | rps | cpu-s/synth | peak RSS MiB | p50 σ |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in report.get("cells") or []:
        var = c.get("variability") or {}
        peak = c.get("peak_rss_bytes")
        peak_mib = (peak / (1024 * 1024)) if isinstance(peak, int) else None
        inf = c.get("inference_ms") or {}
        wall = c.get("wall_ms") or {}
        lines.append(
            "| {intra} | {dlg} | {conc} | {wp50} | {wp95} | {ip50} | {ip95} | "
            "{rps} | {cpu} | {rss} | {sig} |".format(
                intra=c.get("intra_op_threads"),
                dlg=c.get("dialogue_class"),
                conc=c.get("concurrency"),
                wp50=_fmt(wall.get("p50")),
                wp95=_fmt(wall.get("p95")),
                ip50=_fmt(inf.get("p50")),
                ip95=_fmt(inf.get("p95")),
                rps=_fmt(c.get("throughput_rps")),
                cpu=_fmt(c.get("cpu_seconds_per_synthesis"), digits=3),
                rss=_fmt(peak_mib, digits=1),
                sig=_fmt(var.get("wall_p50_stdev")),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            report.get("session_construction_note") or "",
            "",
            "Default production remains `PIPER_ONNX_INTRA_OP_THREADS` unset/0 "
            "(stock PiperVoice.load automatic ORT threading).",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("PIPER_DIAG_MODEL") or os.environ.get("PERF_VOICE_PATH") or ""),
        help="Path to en_US-libritts-high.onnx (or PIPER_DIAG_MODEL / PERF_VOICE_PATH)",
    )
    parser.add_argument(
        "--platform-root",
        type=Path,
        default=Path(os.environ.get("NPC_PLATFORM_ROOT") or r"C:\Users\jayhi\source\repos\NPC-Platform"),
        help="NPC-Platform root (fixtures + safety config)",
    )
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--threads",
        type=str,
        default="0,1,2,4",
        help="Comma-separated intra-op thread counts (default 0,1,2,4)",
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,2,4,6,8",
        help="Comma-separated concurrency ladder (default 1,2,4,6,8)",
    )
    args = parser.parse_args(argv)

    if not args.model or not args.model.is_file():
        print(
            "ERROR: provide --model path\\to\\en_US-libritts-high.onnx "
            "or set PIPER_DIAG_MODEL / PERF_VOICE_PATH",
            file=sys.stderr,
        )
        return 2

    thread_grid = tuple(int(x.strip()) for x in args.threads.split(",") if x.strip())
    concurrency_ladder = tuple(
        int(x.strip()) for x in args.concurrency.split(",") if x.strip()
    )
    if not thread_grid or not concurrency_ladder:
        print("ERROR: empty --threads or --concurrency", file=sys.stderr)
        return 2

    try:
        report = run_benchmark(
            model_path=args.model,
            platform_root=args.platform_root,
            slots=int(args.slots),
            repeats=max(1, int(args.repeats)),
            thread_grid=thread_grid,
            concurrency_ladder=concurrency_ladder,
            json_out=args.json_out,
        )
    except Exception as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1

    fails = sum(int(c.get("failures_total") or 0) for c in report.get("cells") or [])
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
