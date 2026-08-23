"""In-process Piper voice pool with background model warm-up."""

from __future__ import annotations

import os
import threading
import time
import wave
from pathlib import Path
from typing import Any

_POOL_LOCK = threading.Lock()


class _VoiceSlot:
    __slots__ = ("voice", "lock", "ready", "loading", "error")

    def __init__(self) -> None:
        self.voice: Any | None = None
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.loading = False
        self.error: BaseException | None = None


_SLOTS: dict[str, _VoiceSlot] = {}
_WARM_THREADS: dict[str, threading.Thread] = {}


def _voice_key(model_path: Path) -> str:
    return str(model_path)


def _create_piper_voice(model_path: Path) -> Any:
    from piper import PiperVoice

    if not model_path.is_file():
        raise FileNotFoundError(f"Piper voice model not found: {model_path}")

    use_cuda = str(os.environ.get("PIPER_USE_CUDA", "")).strip().lower() in ("1", "true", "yes")
    return PiperVoice.load(model_path, use_cuda=use_cuda)


def _store_loaded_voice(key: str, voice: Any) -> None:
    with _POOL_LOCK:
        slot = _SLOTS[key]
        if slot.voice is None:
            slot.voice = voice
        slot.loading = False
        slot.error = None
        slot.ready.set()


def _store_load_error(key: str, error: BaseException) -> None:
    with _POOL_LOCK:
        slot = _SLOTS.get(key)
        if slot is None:
            return
        slot.loading = False
        slot.error = error
        slot.ready.set()


def _claim_loader(key: str) -> tuple[_VoiceSlot, bool]:
    """Return (slot, should_load). Caller loads only when should_load is True."""
    with _POOL_LOCK:
        slot = _SLOTS.get(key)
        if slot is not None and slot.voice is not None:
            return slot, False
        if slot is not None and slot.loading:
            return slot, False
        if slot is None:
            slot = _VoiceSlot()
            _SLOTS[key] = slot
        elif slot.error is not None:
            slot.error = None
            slot.ready.clear()
        slot.loading = True
        return slot, True


def _load_voice(model_path: Path, request_id: str | None) -> None:
    key = _voice_key(model_path)
    try:
        voice = _create_piper_voice(model_path)
        _store_loaded_voice(key, voice)
        print(
            f"[PIPER-POOL] warmed model={model_path.name} request_id={request_id or '-'}",
            flush=True,
        )
    except Exception as ex:
        _store_load_error(key, ex)
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
        _slot, should_start = _claim_loader(key)
        if not should_start:
            continue
        thread = threading.Thread(
            target=_load_voice,
            args=(model_path, request_id),
            name=f"piper-warm-{model_path.name}",
            daemon=True,
        )
        with _POOL_LOCK:
            _WARM_THREADS[key] = thread
        thread.start()


def _get_voice_and_lock(model_path: Path) -> tuple[Any, threading.Lock]:
    key = _voice_key(model_path)
    deadline = time.perf_counter() + 120.0

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for Piper voice load: {model_path}")

        with _POOL_LOCK:
            slot = _SLOTS.get(key)
            if slot is not None and slot.voice is not None:
                return slot.voice, slot.lock

        slot, should_load = _claim_loader(key)
        if should_load:
            try:
                voice = _create_piper_voice(model_path)
                _store_loaded_voice(key, voice)
            except Exception as ex:
                _store_load_error(key, ex)
                raise
            with _POOL_LOCK:
                loaded = _SLOTS[key]
                return loaded.voice, loaded.lock

        if not slot.ready.wait(timeout=remaining):
            raise TimeoutError(f"Timed out waiting for Piper voice load: {model_path}")

        with _POOL_LOCK:
            slot = _SLOTS[key]
            if slot.voice is not None:
                return slot.voice, slot.lock
            if slot.loading:
                # Another thread claimed a retry after our wait returned.
                continue
            if slot.error is not None:
                raise slot.error
            # Stale ready or retry claimed between wait return and lock: loop again.
            continue


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
        _SLOTS.clear()
        _WARM_THREADS.clear()
