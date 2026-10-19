"""Progressive Fish playback events. Off unless FISH_TTS_PROGRESSIVE is set.

Events are JSON-safe and carry PCM as base64. The browser plays them. Discord
keeps the finished WAV clip and does not handle these event names.
"""

from __future__ import annotations

import base64
from typing import Any

from turn_progress import append_audio_stream


def stream_id(request_id: str, index: int) -> str:
    return f"{request_id}:{int(index)}"


def start_event(request_id: str, index: int, *, sample_rate: int, channels: int = 1) -> dict[str, Any]:
    return {
        "event": "audioStreamStart",
        "stream_id": stream_id(request_id, index),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
    }


def chunk_event(request_id: str, index: int, pcm: bytes, seq: int) -> dict[str, Any]:
    return {
        "event": "audioStreamChunk",
        "stream_id": stream_id(request_id, index),
        "seq": int(seq),
        "pcm_base64": base64.b64encode(pcm or b"").decode("ascii"),
    }


def end_event(request_id: str, index: int) -> dict[str, Any]:
    return {
        "event": "audioStreamEnd",
        "stream_id": stream_id(request_id, index),
    }


def error_event(request_id: str, index: int, message: str) -> dict[str, Any]:
    return {
        "event": "audioStreamError",
        "stream_id": stream_id(request_id, index),
        "message": (message or "Fish speech failed.")[:300],
    }


def publish(request_id: str, event: dict[str, Any]) -> None:
    append_audio_stream(request_id, event)
