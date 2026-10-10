#!/usr/bin/env python3
"""
Diagnostic only: measure whether Piper pool inference windows overlap.

Does NOT change pool locking, allocation, or production code paths.
Monkeypatches PiperVoice.phonemize / phoneme_ids_to_audio for timing only.

Usage (from middleware repo root, with voices available):

  set PIPER_POOL_SLOTS=8
  set PIPER_DIAG_MODEL=C:\\path\\to\\en_US-libritts-high.onnx
  python scripts/diagnose_piper_concurrency.py --concurrency 4

Optional:
  --text "..."   synthesis text (default: medium-length sentence)
  --json-out path\\to\\report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import piper_pool  # noqa: E402
from piper import PiperVoice  # noqa: E402


@dataclass
class RequestTrace:
    request_id: str
    thread_id: int
    process_id: int
    piper_instance_id: str | None = None
    acquire_start: float | None = None
    acquire_end: float | None = None
    phonemize_start: float | None = None
    phonemize_end: float | None = None
    inference_start: float | None = None
    inference_end: float | None = None
    release_end: float | None = None
    error: str | None = None


_TLS = threading.local()
_ORIG_PHONEMIZE = PiperVoice.phonemize
_ORIG_PHONEME_IDS_TO_AUDIO = PiperVoice.phoneme_ids_to_audio
_TRACES_LOCK = threading.Lock()
_TRACES: list[RequestTrace] = []


def _install_timing_patches() -> None:
    def phonemize(self, text: str):  # type: ignore[no-untyped-def]
        tr: RequestTrace | None = getattr(_TLS, "trace", None)
        if tr is not None:
            tr.phonemize_start = time.perf_counter()
        try:
            return _ORIG_PHONEMIZE(self, text)
        finally:
            if tr is not None:
                tr.phonemize_end = time.perf_counter()

    def phoneme_ids_to_audio(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        tr: RequestTrace | None = getattr(_TLS, "trace", None)
        if tr is not None:
            tr.inference_start = time.perf_counter()
            # Prefer ONNX session identity as stable instance id.
            tr.piper_instance_id = f"session:{id(self.session)}"
        try:
            return _ORIG_PHONEME_IDS_TO_AUDIO(self, *args, **kwargs)
        finally:
            if tr is not None:
                tr.inference_end = time.perf_counter()

    PiperVoice.phonemize = phonemize  # type: ignore[method-assign]
    PiperVoice.phoneme_ids_to_audio = phoneme_ids_to_audio  # type: ignore[method-assign]


def _restore_patches() -> None:
    PiperVoice.phonemize = _ORIG_PHONEMIZE  # type: ignore[method-assign]
    PiperVoice.phoneme_ids_to_audio = _ORIG_PHONEME_IDS_TO_AUDIO  # type: ignore[method-assign]


def _worker(request_id: str, model_path: Path, text: str, out_dir: Path) -> RequestTrace:
    tr = RequestTrace(
        request_id=request_id,
        thread_id=threading.get_ident(),
        process_id=os.getpid(),
    )
    _TLS.trace = tr
    out_path = out_dir / f"{request_id}.wav"
    try:
        tr.acquire_start = time.perf_counter()
        # Mirror production path: acquire happens inside synthesize_wav_file.
        # Split timing around pool acquire by wrapping the public entry.
        instance = piper_pool._acquire_instance(model_path)
        tr.acquire_end = time.perf_counter()
        tr.piper_instance_id = f"session:{id(instance.voice.session)}"
        try:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(
                speaker_id=0,
                length_scale=1.0,
                noise_scale=0.667,
                noise_w_scale=0.8,
                normalize_audio=True,
            )
            import wave

            with wave.open(str(out_path), "wb") as wav_file:
                wav_params_set = False
                for chunk_index, audio_chunk in enumerate(
                    instance.voice.synthesize(text, syn_config)
                ):
                    if not wav_params_set:
                        wav_file.setframerate(audio_chunk.sample_rate)
                        wav_file.setsampwidth(audio_chunk.sample_width)
                        wav_file.setnchannels(audio_chunk.sample_channels)
                        wav_params_set = True
                    wav_file.writeframes(audio_chunk.audio_int16_bytes)
        finally:
            piper_pool._release_instance(instance)
            tr.release_end = time.perf_counter()
    except Exception as ex:
        tr.error = repr(ex)
        if tr.acquire_end is None:
            tr.acquire_end = time.perf_counter()
    finally:
        with _TRACES_LOCK:
            _TRACES.append(tr)
        _TLS.trace = None
    return tr


def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def _analyze(traces: list[RequestTrace]) -> dict[str, Any]:
    ok = [t for t in traces if t.error is None and t.inference_start and t.inference_end]
    pairs: list[dict[str, Any]] = []
    overlap_count = 0
    for i, a in enumerate(ok):
        for b in ok[i + 1 :]:
            assert a.inference_start is not None and a.inference_end is not None
            assert b.inference_start is not None and b.inference_end is not None
            ov = _intervals_overlap(
                a.inference_start, a.inference_end, b.inference_start, b.inference_end
            )
            if ov:
                overlap_count += 1
            pairs.append(
                {
                    "a": a.request_id,
                    "b": b.request_id,
                    "inference_overlap": ov,
                    "same_instance": a.piper_instance_id == b.piper_instance_id,
                }
            )
    phon_ok = [
        t for t in traces if t.error is None and t.phonemize_start and t.phonemize_end
    ]
    phon_overlap = 0
    for i, a in enumerate(phon_ok):
        for b in phon_ok[i + 1 :]:
            assert a.phonemize_start is not None and a.phonemize_end is not None
            assert b.phonemize_start is not None and b.phonemize_end is not None
            if _intervals_overlap(
                a.phonemize_start, a.phonemize_end, b.phonemize_start, b.phonemize_end
            ):
                phon_overlap += 1

    distinct_instances = sorted({t.piper_instance_id for t in ok if t.piper_instance_id})
    return {
        "requests": len(traces),
        "successful": len(ok),
        "distinct_piper_instances": distinct_instances,
        "distinct_instance_count": len(distinct_instances),
        "inference_overlap_pair_count": overlap_count,
        "phonemize_overlap_pair_count": phon_overlap,
        "pair_details": pairs,
        "verdict": (
            "INFERENCE_WINDOWS_OVERLAP"
            if overlap_count > 0
            else "NO_INFERENCE_OVERLAP_OBSERVED"
        ),
        "phonemize_verdict": (
            "PHONEMIZE_WINDOWS_OVERLAP"
            if phon_overlap > 0
            else "NO_PHONEMIZE_OVERLAP_OBSERVED"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("PIPER_DIAG_MODEL") or ""),
        help="Path to .onnx (or set PIPER_DIAG_MODEL)",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--text",
        default=(
            "Traveler, the market closes at dusk. If you seek the blacksmith, "
            "follow the lane past the well, then turn left at the crooked oak."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--slots", type=int, default=8)
    args = parser.parse_args(argv)

    if not args.model or not args.model.is_file():
        print(
            "ERROR: provide --model path/to/en_US-libritts-high.onnx "
            "or set PIPER_DIAG_MODEL",
            file=sys.stderr,
        )
        return 2

    os.environ["PIPER_POOL_SLOTS"] = str(args.slots)
    piper_pool.clear_pool_for_tests()
    print(f"Warming {args.slots} replicas for {args.model} ...", flush=True)
    piper_pool.warm_voice_models(
        [args.model],
        request_id="diag-warm",
        wait_for_ready=True,
        timeout_seconds=600.0,
    )
    status = piper_pool.pool_status()
    print(json.dumps({"pool_status": status}, indent=2), flush=True)

    out_dir = Path(os.environ.get("TEMP") or ".") / "piper-concurrency-diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    _install_timing_patches()
    try:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [
                pool.submit(
                    _worker,
                    f"req-{i}",
                    args.model.resolve(),
                    args.text,
                    out_dir,
                )
                for i in range(args.concurrency)
            ]
            for fut in as_completed(futs):
                fut.result()
        wall = time.perf_counter() - t0
    finally:
        _restore_patches()

    analysis = _analyze(list(_TRACES))
    # Relativize timestamps to first acquire_start for readability
    starts = [t.acquire_start for t in _TRACES if t.acquire_start is not None]
    origin = min(starts) if starts else 0.0

    def rel(v: float | None) -> float | None:
        return None if v is None else round(v - origin, 6)

    rows = []
    for t in sorted(_TRACES, key=lambda x: x.request_id):
        rows.append(
            {
                **asdict(t),
                "acquire_start_rel": rel(t.acquire_start),
                "acquire_end_rel": rel(t.acquire_end),
                "phonemize_start_rel": rel(t.phonemize_start),
                "phonemize_end_rel": rel(t.phonemize_end),
                "inference_start_rel": rel(t.inference_start),
                "inference_end_rel": rel(t.inference_end),
                "release_end_rel": rel(t.release_end),
            }
        )

    report = {
        "wall_seconds": wall,
        "model": str(args.model),
        "concurrency": args.concurrency,
        "slots": args.slots,
        "ort_session_options_note": (
            "PiperVoice.load uses onnxruntime.SessionOptions() defaults: "
            "intra_op_num_threads=0 (auto), inter_op_num_threads=0, "
            "execution_mode=ORT_SEQUENTIAL"
        ),
        "espeak_lock_note": (
            "piper.voice.phonemize holds global _ESPEAK_PHONEMIZER_LOCK around "
            "the shared EspeakPhonemizer; phonemize windows should not overlap."
        ),
        "traces": rows,
        "analysis": analysis,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.json_out}", flush=True)
    return 0 if analysis["successful"] == args.concurrency else 1


if __name__ == "__main__":
    raise SystemExit(main())
