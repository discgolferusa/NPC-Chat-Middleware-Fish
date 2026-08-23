"""In-process Piper voice pool with background model warm-up."""

from __future__ import annotations

import os
import threading
import wave
from pathlib import Path
from typing import Any

_POOL_LOCK = threading.Lock()
_VOICES: dict[str, Any] = {}
_VOICE_LOCKS: dict[str, threading.Lock] = {}
_WARM_THREADS: dict[str, threading.Thread] = {}


def _voice_key(model_path: Path) -> str:
    return str(model_path)


def _create_piper_voice(model_path: Path) -> Any:
    from piper import PiperVoice

    if not model_path.is_file():
        raise FileNotFoundError(f"Piper voice model not found: {model_path}")

    use_cuda = str(os.environ.get("PIPER_USE_CUDA", "")).strip().lower() in ("1", "true", "yes")
    return PiperVoice.load(model_path, use_cuda=use_cuda)


def _load_voice(model_path: Path, request_id: str | None) -> None:
    key = _voice_key(model_path)
    try:
        voice = _create_piper_voice(model_path)
        with _POOL_LOCK:
            _VOICES[key] = voice
            _VOICE_LOCKS.setdefault(key, threading.Lock())
        print(
            f"[PIPER-POOL] warmed model={model_path.name} request_id={request_id or '-'}",
            flush=True,
        )
    except Exception as ex:
        print(
            f"[PIPER-POOL] warm_failed model={model_path.name} "
            f"request_id={request_id or '-'} error={ex!r}",
            flush=True,
        )


def warm_voice_models(model_paths: list[Path], *, request_id: str | None = None) -> None:
    """Start background ONNX load for each unique model path (idempotent)."""
    unique: list[Path] = []
    seen: set[str] = set()
    for raw in model_paths:
        if raw is None:
            continue
        path = Path(raw)
        key = _voice_key(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    for model_path in unique:
        key = _voice_key(model_path)
        with _POOL_LOCK:
            if key in _VOICES:
                continue
            warm_thread = _WARM_THREADS.get(key)
            if warm_thread is not None and warm_thread.is_alive():
                continue
            thread = threading.Thread(
                target=_load_voice,
                args=(model_path, request_id),
                name=f"piper-warm-{model_path.name}",
                daemon=True,
            )
            _WARM_THREADS[key] = thread
            thread.start()


def _get_voice_and_lock(model_path: Path) -> tuple[Any, threading.Lock]:
    key = _voice_key(model_path)

    with _POOL_LOCK:
        voice = _VOICES.get(key)
        if voice is not None:
            return voice, _VOICE_LOCKS.setdefault(key, threading.Lock())

    warm_thread = _WARM_THREADS.get(key)
    if warm_thread is not None and warm_thread.is_alive():
        warm_thread.join(timeout=120.0)

    with _POOL_LOCK:
        voice = _VOICES.get(key)
        if voice is not None:
            return voice, _VOICE_LOCKS.setdefault(key, threading.Lock())

        voice = _create_piper_voice(model_path)
        _VOICES[key] = voice
        lock = threading.Lock()
        _VOICE_LOCKS[key] = lock
        return voice, lock


def synthesize_wav_file(
    model_path: Path,
    output_path: Path,
    text: str,
    *,
    length_scale: float,
    noise_scale: float,
    noise_w: float,
    sentence_silence: float,
    speaker_id: int | None = None,
) -> None:
    """Synthesize text to a WAV file using a pooled in-process Piper voice."""
    from piper import SynthesisConfig

    voice, voice_lock = _get_voice_and_lock(model_path)
    syn_config = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w,
        normalize_audio=True,
    )
    silence_bytes = bytes(int(voice.config.sample_rate * sentence_silence * 2))

    with voice_lock:
        with wave.open(str(output_path), "wb") as wav_file:
            wav_params_set = False
            for chunk_index, audio_chunk in enumerate(voice.synthesize(text, syn_config)):
                if not wav_params_set:
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)
                    wav_params_set = True

                if chunk_index > 0:
                    wav_file.writeframes(silence_bytes)

                wav_file.writeframes(audio_chunk.audio_int16_bytes)


def clear_pool_for_tests() -> None:
    """Reset module state (tests only)."""
    with _POOL_LOCK:
        _VOICES.clear()
        _VOICE_LOCKS.clear()
        _WARM_THREADS.clear()
