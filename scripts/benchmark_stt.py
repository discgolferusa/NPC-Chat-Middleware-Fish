#!/usr/bin/env python3
"""Compare OpenRouter STT models on a local WAV (dev/admin utility).

Usage:
  python scripts/benchmark_stt.py path/to.wav
  python scripts/benchmark_stt.py path/to.wav --models assemblyai/universal-3-5-pro,deepgram/nova-3
  python scripts/benchmark_stt.py path/to.wav --json out.json --csv out.csv

Requires OPENROUTER_API_KEY (env or llm_secrets.env). Not used by production turns.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODELS = [
    "assemblyai/universal-3-5-pro",
    "deepgram/nova-3",
    "openai/whisper-large-v3-turbo",
    "openai/gpt-transcribe",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter STT models on one WAV.")
    parser.add_argument("wav", type=Path, help="Path to a WAV/audio file")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated OpenRouter STT model ids",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON output path")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    parser.add_argument(
        "--keyterms",
        default="",
        help="Optional comma-separated keyterms to send when supported",
    )
    args = parser.parse_args()

    wav = args.wav.resolve()
    if not wav.is_file():
        print(f"File not found: {wav}", file=sys.stderr)
        return 2

    from stt_context import STTContext
    from stt_openrouter import STTProviderError, call_openrouter_transcription

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    keyterms = [k.strip() for k in args.keyterms.split(",") if k.strip()]
    context = STTContext(keyterms=keyterms, prompt="", language="en") if keyterms else None

    print(f"File:\n{wav}\n")
    rows: list[dict] = []

    for model in models:
        print(model)
        started = time.perf_counter()
        try:
            result = call_openrouter_transcription(
                wav,
                model=model,
                context=context,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            print("Transcript:")
            print(result.text or "(empty)")
            print("Latency:")
            print(f"{latency_ms} ms (http={result.transcription_ms} ms)")
            if result.cost is not None:
                print(f"Cost: ${result.cost}")
            row = {
                "file": str(wav),
                "model": model,
                "ok": True,
                "transcript": result.text,
                "latency_ms": latency_ms,
                "http_ms": result.transcription_ms,
                "cost": result.cost,
                "request_id": result.request_id,
                "error_type": None,
                # Analyst columns for RPG-specific review
                "name_error": "",
                "location_error": "",
                "faction_error": "",
                "terminology_error": "",
                "negation_error": "",
                "intent_error": "",
                "hallucination": "",
                "missed_short_utterance": "",
                "background_speech": "",
            }
        except STTProviderError as ex:
            latency_ms = int((time.perf_counter() - started) * 1000)
            print("Transcript:")
            print(f"(failed: {ex.error_type}) {ex}")
            print("Latency:")
            print(f"{latency_ms} ms")
            row = {
                "file": str(wav),
                "model": model,
                "ok": False,
                "transcript": "",
                "latency_ms": latency_ms,
                "http_ms": None,
                "cost": None,
                "request_id": None,
                "error_type": ex.error_type,
                "name_error": "",
                "location_error": "",
                "faction_error": "",
                "terminology_error": "",
                "negation_error": "",
                "intent_error": "",
                "hallucination": "",
                "missed_short_utterance": "",
                "background_speech": "",
            }
        rows.append(row)
        print()

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.json}")
    if args.csv:
        fieldnames = list(rows[0].keys()) if rows else []
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
