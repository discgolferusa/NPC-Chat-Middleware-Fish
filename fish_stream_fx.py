"""Stateful FFmpeg graph for one Fish utterance.

The live graph uses the existing presets but skips reverse-trim, spectral
denoise, and lookahead loudness. Those need the whole file or add TTFS delay.
The archived WAV can still use ``build_af_filter`` unchanged.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tts_processing import build_af_filter


def streaming_af_filter(**kwargs) -> str:
    graph = build_af_filter(trim_silence=True, ttfa_fast=True, **kwargs)
    parts = [part for part in graph.split(",") if part and not part.startswith("afftdn") and not part.startswith("dynaudnorm")]
    return ",".join(parts)


class StreamingEffect:
    """One ffmpeg process for the life of an utterance. Not safe to share across turns."""

    def __init__(self, *, sample_rate: int, fx_preset: str, pitch_semitones: float, output_path: Path) -> None:
        self.output_path = output_path
        af = streaming_af_filter(
            pitch_semitones=pitch_semitones,
            fx_preset=fx_preset,
            sample_rate=sample_rate,
        )
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-af",
                af,
                "-ar",
                "22050",
                "-ac",
                "1",
                "-f",
                "wav",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, pcm: bytes) -> None:
        if not pcm or self._proc.stdin is None:
            return
        self._proc.stdin.write(pcm)

    def finish(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read() if self._proc.stderr is not None else b""
        code = self._proc.wait(timeout=30)
        if code != 0:
            raise RuntimeError(f"streaming ffmpeg failed: {stderr.decode('utf-8', errors='replace')[:400]}")
