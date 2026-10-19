"""Isolated Fish drama-3 probe. Not imported by the server.

Live calls need FISH_API_KEY. Without it the script records the call as unverified
and exits 2. Unit tests drive ``measure_stream`` against a local fake response.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fish_config import native_model_header


def measure_stream(iterator, *, started: float | None = None) -> dict[str, Any]:
    """Time the first yielded body chunk against the last. Empty iterators have no TTFAB."""
    origin = started if started is not None else time.perf_counter()
    first = None
    last = None
    total = 0
    count = 0
    for chunk in iterator:
        if not chunk:
            continue
        now = time.perf_counter()
        if first is None:
            first = now
        last = now
        total += len(chunk)
        count += 1
    return {
        "chunk_count": count,
        "byte_count": total,
        "ttfab_ms": None if first is None else max(0, int(round((first - origin) * 1000))),
        "total_ms": None if last is None else max(0, int(round((last - origin) * 1000))),
        "incremental": bool(first is not None and last is not None and (last - first) > 0.05 and count > 1),
    }


def probe_live() -> dict[str, Any]:
    key = (os.environ.get("FISH_API_KEY") or "").strip()
    model = (os.environ.get("FISH_TTS_MODEL") or "fish-audio/drama-3").strip()
    voice = (os.environ.get("FISH_PROBE_VOICE_ID") or "").strip()
    report: dict[str, Any] = {
        "provider": "fish-audio",
        "model": model,
        "native_model_header": native_model_header(model),
        "live": False,
        "reference_id_accepted": "unverified",
        "pcm_layout": "unverified",
        "incremental": "unverified",
        "sample_url_without_key": "unverified",
        "reason": "",
    }
    if not key or not voice:
        report["reason"] = "FISH_API_KEY or FISH_PROBE_VOICE_ID is not set. No live request was sent."
        return report
    import requests

    text = "The door is shut. Keep your voice down and tell me what you saw."
    started = time.perf_counter()
    response = requests.post(
        "https://api.fish.audio/v1/tts",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "model": native_model_header(model),
        },
        json={"text": text, "reference_id": voice, "format": "pcm", "latency": "balanced"},
        stream=True,
        timeout=60,
    )
    report["live"] = True
    report["status"] = response.status_code
    report["content_type"] = response.headers.get("Content-Type")
    if response.status_code >= 400:
        report["reference_id_accepted"] = False
        report["reason"] = response.text[:300]
        return report
    measured = measure_stream(response.iter_content(4096), started=started)
    response.close()
    report["reference_id_accepted"] = True
    report["incremental"] = measured["incremental"]
    report["ttfab_ms"] = measured["ttfab_ms"]
    report["total_ms"] = measured["total_ms"]
    report["pcm_layout"] = report["content_type"] or "unverified"
    return report


def main() -> int:
    report = probe_live()
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if report.get("live") else 2


if __name__ == "__main__":
    raise SystemExit(main())
