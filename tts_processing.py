"""
Piper TTS synthesis helpers: text cleanup, option parsing, and FFmpeg post-processing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Piper CLI defaults (used when NPC / request omit the field).
DEFAULT_LENGTH_SCALE = 1.15
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_W = 0.8
DEFAULT_SENTENCE_SILENCE = 0.2
DEFAULT_PITCH_SEMITONES = 0.0
DEFAULT_FX_PRESET = "none"
# FFmpeg afftdn defaults (nr=0 skips the denoise stage).
DEFAULT_DENOISE_NR = 12.0
DEFAULT_DENOISE_NF = -25.0

# Clamp ranges for request / YAML overrides.
LENGTH_SCALE_MIN, LENGTH_SCALE_MAX = 0.5, 2.0
NOISE_SCALE_MIN, NOISE_SCALE_MAX = 0.0, 1.5
NOISE_W_MIN, NOISE_W_MAX = 0.0, 2.0
SENTENCE_SILENCE_MIN, SENTENCE_SILENCE_MAX = 0.0, 2.0
PITCH_SEMITONES_MIN, PITCH_SEMITONES_MAX = -6.0, 6.0
DENOISE_NR_MIN, DENOISE_NR_MAX = 0.0, 30.0
# FFmpeg afftdn nf is documented/validated as [-80, -20].
DENOISE_NF_MIN, DENOISE_NF_MAX = -80.0, -20.0

VALID_FX_PRESETS = frozenset(
    {"none", "clarity", "cave", "radio", "grit", "underwater", "robot", "whisper"}
)

# Global pronunciation glossary (fiction / common RPG terms). Per-NPC overrides win.
DEFAULT_PRONUNCIATION: dict[str, str] = {
    "Golarion": "go LAR ee un",
    "Absalom": "AB suh lom",
    "Cheliax": "CHELL ee ax",
    "Mwangi": "mwahn GEE",
    "Osirion": "oh SEER ee un",
    "Taldor": "TAL door",
    "Varisia": "vuh REE zha",
    "Andoran": "an DOOR an",
    "Nirmathas": "NEER muh thas",
    "Lastwall": "LAST wall",
}

_ABBREVIATIONS = [
    (re.compile(r"\bDr\.", re.I), "Doctor"),
    (re.compile(r"\bMr\.", re.I), "Mister"),
    (re.compile(r"\bMrs\.", re.I), "Missus"),
    (re.compile(r"\bMs\.", re.I), "Miss"),
    (re.compile(r"\bSt\.", re.I), "Saint"),
    (re.compile(r"\betc\.", re.I), "etcetera"),
    (re.compile(r"\bvs\.", re.I), "versus"),
    (re.compile(r"\be\.g\.", re.I), "for example"),
    (re.compile(r"\bi\.e\.", re.I), "that is"),
]

_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
}


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_fx_preset(value: Any) -> str:
    text = str(value or DEFAULT_FX_PRESET).strip().lower()
    if text in VALID_FX_PRESETS:
        return text
    return DEFAULT_FX_PRESET


def _as_pronunciation_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, replacement in value.items():
        word = str(key or "").strip()
        spoken = str(replacement or "").strip()
        if word and spoken:
            out[word] = spoken
    return out


def infer_quality_tier(path: Path | str) -> str:
    """Infer Piper quality tier from path segments (low/medium/high)."""
    text = str(path).replace("\\", "/").lower()
    for tier in ("high", "medium", "low"):
        if f"/{tier}/" in text or text.endswith(f"/{tier}") or f"-{tier}." in text:
            return tier
    return "unknown"


def infer_dataset_name(model_path: Path | str) -> str:
    """Infer Piper dataset name from its standard folder or filename layout."""
    normalized = str(model_path).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    quality = infer_quality_tier(normalized)
    if quality != "unknown" and len(parts) >= 3 and parts[-2].lower() == quality:
        dataset = parts[-3].strip()
        if dataset:
            return dataset

    filename = parts[-1] if parts else normalized
    stem = filename[:-5] if filename.lower().endswith(".onnx") else Path(filename).stem
    stem = re.sub(
        r"^(?:[a-z]{2}_[A-Z]{2}|[a-z]{2}-[A-Z]{2})-",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    stem = re.sub(r"-(?:x_low|low|medium|high)$", "", stem, flags=re.IGNORECASE)
    return stem or filename


def speaker_annotation_sidecar_path(model_path: Path) -> Path:
    """Sidecar next to the ONNX: ``model.onnx.speakers.json``."""
    return Path(str(model_path) + ".speakers.json")


def _annotation_text_from_entry(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("annotation", "descriptors", "description", "text"):
            value = entry.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _display_name_from_entry(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in ("name", "friendly_name", "display_name"):
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def load_speaker_annotation_sidecar(model_path: Path) -> dict[int, dict[str, str]]:
    """
    Load optional human annotation overlay for multi-speaker Piper models.

    Expected shape::

        {
          "speakers": {
            "0": {"name": "Benjamin Parker", "annotation": "masculine, calm, ..."},
            "1": "feminine, soft, ..."
          }
        }

    Keys may be string or int speaker ids. Annotation-only string values are allowed.
    """
    path = speaker_annotation_sidecar_path(model_path)
    if not path.is_file():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    raw = data.get("speakers") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}

    out: dict[int, dict[str, str]] = {}
    for key, entry in raw.items():
        try:
            sid = int(key)
        except (TypeError, ValueError):
            continue
        annotation = _annotation_text_from_entry(entry)
        name = _display_name_from_entry(entry)
        if not annotation and not name:
            continue
        payload: dict[str, str] = {}
        if name:
            payload["name"] = name
        if annotation:
            payload["annotation"] = annotation
        out[sid] = payload
    return out


def apply_speaker_annotation_sidecar(
    speakers: list[dict[str, Any]],
    sidecar: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Merge sidecar name/annotation onto existing inventory speaker rows (by id).

    Sidecar ids that are not already in ``speakers`` are ignored so /voices never
    lists speakers Piper cannot synthesize.
    """
    if not sidecar or not speakers:
        return speakers

    merged: list[dict[str, Any]] = []
    for row in speakers:
        try:
            sid = int(row.get("id"))
        except (TypeError, ValueError):
            merged.append(dict(row))
            continue
        out = dict(row)
        overlay = sidecar.get(sid)
        if overlay:
            if overlay.get("name"):
                # Prefer sidecar friendly name only when current name is generic.
                current = str(out.get("name") or "").strip()
                if not current or re.match(r"^speaker_\d+$", current, re.I):
                    out["name"] = overlay["name"]
            if overlay.get("annotation"):
                out["annotation"] = overlay["annotation"]
        merged.append(out)
    return merged


def load_piper_model_meta(model_path: Path) -> dict[str, Any]:
    """
    Read companion .onnx.json for speaker map / recommended defaults when present.
    Optional ``*.onnx.speakers.json`` sidecar adds annotation text for dropdowns.
    """
    meta: dict[str, Any] = {
        "dataset": infer_dataset_name(model_path),
        "quality": infer_quality_tier(model_path),
        "num_speakers": 1,
        "speakers": [],
        "has_speaker_map": False,
        "speaker_id_map": {},
        "recommended_noise_scale": None,
        "recommended_noise_w": None,
        "recommended_length_scale": None,
    }

    config_path = Path(str(model_path) + ".json")
    if not config_path.exists():
        sibling = model_path.with_suffix(".onnx.json")
        if sibling.exists():
            config_path = sibling
        else:
            # Sidecar alone must not invent a speaker inventory.
            return meta

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return meta

    dataset = str(data.get("dataset") or data.get("dataset_name") or "").strip()
    if dataset:
        meta["dataset"] = dataset

    speaker_map = data.get("speaker_id_map") or data.get("speakerIdMap") or {}
    if isinstance(speaker_map, dict) and speaker_map:
        cleaned: dict[str, int] = {}
        for name, sid in speaker_map.items():
            try:
                cleaned[str(name)] = int(sid)
            except (TypeError, ValueError):
                continue
        meta["speaker_id_map"] = cleaned
        meta["has_speaker_map"] = bool(cleaned)
        meta["speakers"] = [
            {"id": sid, "name": name}
            for name, sid in sorted(cleaned.items(), key=lambda item: item[1])
        ]
        meta["num_speakers"] = max(len(cleaned), int(data.get("num_speakers") or 1))
    else:
        num = int(data.get("num_speakers") or 1)
        meta["num_speakers"] = num
        if num > 1:
            meta["speakers"] = [{"id": i, "name": f"speaker_{i}"} for i in range(num)]

    inference = data.get("inference") if isinstance(data.get("inference"), dict) else {}
    if inference.get("noise_scale") is not None:
        meta["recommended_noise_scale"] = _as_float(inference.get("noise_scale"), DEFAULT_NOISE_SCALE)
    if inference.get("noise_w") is not None:
        meta["recommended_noise_w"] = _as_float(inference.get("noise_w"), DEFAULT_NOISE_W)
    if inference.get("length_scale") is not None:
        meta["recommended_length_scale"] = _as_float(
            inference.get("length_scale"), DEFAULT_LENGTH_SCALE
        )

    sidecar = load_speaker_annotation_sidecar(model_path)
    if sidecar:
        # Annotate only; never expand speakers / num_speakers beyond the model.
        meta["speakers"] = apply_speaker_annotation_sidecar(meta["speakers"], sidecar)

    return meta


def curated_defaults_for_quality(quality: str) -> dict[str, float]:
    """
    Suggested Piper noise defaults for weaker vs stronger models.
    Weaker (low) voices usually sound cleaner with less generator noise.
    """
    tier = (quality or "unknown").lower()
    if tier == "low":
        return {
            "noise_scale": 0.5,
            "noise_w": 0.6,
            "length_scale": 1.2,
        }
    if tier == "high":
        return {
            "noise_scale": DEFAULT_NOISE_SCALE,
            "noise_w": DEFAULT_NOISE_W,
            "length_scale": 1.1,
        }
    return {
        "noise_scale": DEFAULT_NOISE_SCALE,
        "noise_w": DEFAULT_NOISE_W,
        "length_scale": DEFAULT_LENGTH_SCALE,
    }


def apply_pronunciation(text: str, glossary: dict[str, str] | None = None) -> str:
    merged = dict(DEFAULT_PRONUNCIATION)
    if glossary:
        merged.update(glossary)
    # Longer keys first so multi-word entries win.
    for word, spoken in sorted(merged.items(), key=lambda item: len(item[0]), reverse=True):
        if not word:
            continue
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        text = pattern.sub(spoken, text)
    return text


def expand_simple_numbers(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return _NUMBER_WORDS.get(token, token)

    return re.sub(r"\b(?:[0-9]|1[0-9]|20)\b", _replace, text)


def ensure_sentence_ending(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?…\":)'" and not text.endswith("..."):
        return text + "."
    return text


def clean_for_tts(text: str, pronunciation: dict[str, str] | None = None) -> str:
    """Normalize LLM output into Piper-friendly spoken text."""
    text = re.sub(r"\*([^*]{1,500})\*", r"\1", text)
    text = re.sub(r"_([^_]{1,500})_", r"\1", text)
    text = re.sub(r"\[([^\]]{1,500})\]", r"\1", text)
    text = re.sub(r"\(([^)]{1,500})\)", r"\1", text)

    text = text.replace("*", "")
    text = text.replace("_", "")
    text = text.replace("`", "")
    text = text.replace("~", "")
    text = text.replace("#", "")
    text = text.replace("|", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("@", " ")

    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    text = expand_simple_numbers(text)
    text = apply_pronunciation(text, pronunciation)
    text = re.sub(r"\s+", " ", text).strip()
    text = ensure_sentence_ending(text)
    return text


def resolve_tts_options(
    npc: dict | None = None,
    *,
    length_scale: Any = None,
    noise_scale: Any = None,
    noise_w: Any = None,
    sentence_silence: Any = None,
    speaker_id: Any = None,
    pitch_semitones: Any = None,
    fx_preset: Any = None,
    denoise_nr: Any = None,
    denoise_nf: Any = None,
    pronunciation: Any = None,
    quality_hint: str | None = None,
) -> dict[str, Any]:
    """
    Merge request overrides → NPC YAML → quality-curated defaults → Piper defaults.
    """
    npc = npc or {}
    quality = quality_hint or infer_quality_tier(str(npc.get("voice_model") or ""))
    curated = curated_defaults_for_quality(quality)

    def pick(override: Any, yaml_key: str, curated_key: str, default: float) -> float:
        if override is not None and str(override).strip() != "":
            return _as_float(override, default)
        if yaml_key in npc and npc.get(yaml_key) is not None and str(npc.get(yaml_key)).strip() != "":
            return _as_float(npc.get(yaml_key), default)
        return _as_float(curated.get(curated_key), default)

    opts_length = pick(length_scale, "tts_length_scale", "length_scale", DEFAULT_LENGTH_SCALE)
    opts_noise = pick(noise_scale, "tts_noise_scale", "noise_scale", DEFAULT_NOISE_SCALE)
    opts_noise_w = pick(noise_w, "tts_noise_w", "noise_w", DEFAULT_NOISE_W)

    if sentence_silence is not None and str(sentence_silence).strip() != "":
        opts_silence = _as_float(sentence_silence, DEFAULT_SENTENCE_SILENCE)
    elif "tts_sentence_silence" in npc and npc.get("tts_sentence_silence") is not None:
        opts_silence = _as_float(npc.get("tts_sentence_silence"), DEFAULT_SENTENCE_SILENCE)
    else:
        opts_silence = DEFAULT_SENTENCE_SILENCE

    raw_speaker = speaker_id
    if raw_speaker is None or str(raw_speaker).strip() == "":
        raw_speaker = npc.get("tts_speaker_id")
    sid = _as_optional_int(raw_speaker)

    raw_pitch = pitch_semitones
    if raw_pitch is None or str(raw_pitch).strip() == "":
        raw_pitch = npc.get("tts_pitch_semitones")
    pitch = _clamp(
        _as_float(raw_pitch, DEFAULT_PITCH_SEMITONES),
        PITCH_SEMITONES_MIN,
        PITCH_SEMITONES_MAX,
    )

    raw_fx = fx_preset
    if raw_fx is None or str(raw_fx).strip() == "":
        raw_fx = npc.get("tts_fx")
    fx = _as_fx_preset(raw_fx)

    raw_nr = denoise_nr
    if raw_nr is None or str(raw_nr).strip() == "":
        raw_nr = npc.get("tts_denoise_nr", npc.get("denoise_nr"))
    opts_nr = _clamp(
        _as_float(raw_nr, DEFAULT_DENOISE_NR),
        DENOISE_NR_MIN,
        DENOISE_NR_MAX,
    )

    raw_nf = denoise_nf
    if raw_nf is None or str(raw_nf).strip() == "":
        raw_nf = npc.get("tts_denoise_nf", npc.get("denoise_nf"))
    opts_nf = _clamp(
        _as_float(raw_nf, DEFAULT_DENOISE_NF),
        DENOISE_NF_MIN,
        DENOISE_NF_MAX,
    )

    gloss = dict(DEFAULT_PRONUNCIATION)
    gloss.update(_as_pronunciation_map(npc.get("tts_pronunciation")))
    if pronunciation is not None:
        gloss.update(_as_pronunciation_map(pronunciation))

    return {
        "length_scale": _clamp(opts_length, LENGTH_SCALE_MIN, LENGTH_SCALE_MAX),
        "noise_scale": _clamp(opts_noise, NOISE_SCALE_MIN, NOISE_SCALE_MAX),
        "noise_w": _clamp(opts_noise_w, NOISE_W_MIN, NOISE_W_MAX),
        "sentence_silence": _clamp(opts_silence, SENTENCE_SILENCE_MIN, SENTENCE_SILENCE_MAX),
        "speaker_id": sid,
        "pitch_semitones": pitch,
        "fx_preset": fx,
        "denoise_nr": opts_nr,
        "denoise_nf": opts_nf,
        "pronunciation": gloss,
        "quality": quality,
    }


def build_af_filter(
    *,
    pitch_semitones: float = 0.0,
    fx_preset: str = "none",
    trim_silence: bool = True,
    sample_rate: int = 22050,
    denoise_nr: float = DEFAULT_DENOISE_NR,
    denoise_nf: float = DEFAULT_DENOISE_NF,
) -> str:
    """
    Build an FFmpeg -af chain: optional silence trim, clarity EQ/dynamics,
    pitch shift, and character FX presets.
    """
    filters: list[str] = []

    if trim_silence:
        # Trim only leading/trailing hush. Do NOT use stop_periods=1 — Piper's
        # sentence_silence inserts mid-clip pauses, and stop_periods would cut
        # every sentence after the first (e.g. nursery-rhyme preview).
        filters.append(
            "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB"
        )
        filters.append("areverse")
        filters.append(
            "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB"
        )
        filters.append("areverse")
        filters.append("apad=pad_dur=0.04")

    # Always-on clarity baseline (ElevenLabs-ish speaker boost / loudness feel).
    filters.append("highpass=f=80")
    nr = _clamp(float(denoise_nr if denoise_nr is not None else DEFAULT_DENOISE_NR), DENOISE_NR_MIN, DENOISE_NR_MAX)
    nf = _clamp(float(denoise_nf if denoise_nf is not None else DEFAULT_DENOISE_NF), DENOISE_NF_MIN, DENOISE_NF_MAX)
    # nr=0 disables spectral denoise (useful when comparing or if speech dulls).
    if nr > 0.05:
        filters.append(f"afftdn=nr={nr:.2f}:nf={nf:.2f}")
    filters.append("equalizer=f=3000:t=q:w=1.2:g=2.5")
    filters.append("acompressor=threshold=-18dB:ratio=2.5:attack=10:release=80:makeup=2")
    filters.append("alimiter=limit=0.95")
    filters.append("dynaudnorm=f=150:g=10:p=0.85")

    preset = _as_fx_preset(fx_preset)
    if preset == "cave":
        filters.append("aecho=0.8:0.88:60:0.3")
        filters.append("lowpass=f=3500")
    elif preset == "radio":
        filters.append("highpass=f=400")
        filters.append("lowpass=f=2800")
        filters.append("equalizer=f=1800:t=q:w=1:g=4")
        filters.append("asetrate=22050*0.98,aresample=22050")
    elif preset == "grit":
        filters.append("equalizer=f=200:t=q:w=1:g=3")
        filters.append("equalizer=f=4000:t=q:w=1:g=2")
        filters.append("acompressor=threshold=-20dB:ratio=4:attack=5:release=50:makeup=3")
    elif preset == "clarity":
        # Extra presence on top of the baseline.
        filters.append("equalizer=f=4500:t=q:w=1:g=2")
    elif preset == "underwater":
        # Muffled lows/mids + short watery echoes.
        filters.append("lowpass=f=900")
        filters.append("highpass=f=100")
        filters.append("aecho=0.6:0.7:40|90:0.28|0.18")
        filters.append("equalizer=f=380:t=q:w=1:g=2.5")
    elif preset == "robot":
        # Narrow band + vibrato + hard compression for a synthetic edge.
        # Keep makeup/EQ modest — baseline already has dynaudnorm + limiter.
        filters.append("highpass=f=280")
        filters.append("lowpass=f=3400")
        filters.append("vibrato=f=8.5:d=0.35")
        filters.append("equalizer=f=2200:t=q:w=1.4:g=2.5")
        filters.append("acompressor=threshold=-16dB:ratio=5:attack=3:release=40:makeup=1.2")
        filters.append("volume=0.78")
        filters.append("alimiter=limit=0.90")
    elif preset == "whisper":
        # Thin, hush-soft presence (FX approximation — not true whisper synthesis).
        # Applied after dynaudnorm so volume stays deliberately quiet.
        filters.append("highpass=f=450")
        filters.append("lowpass=f=4200")
        filters.append("equalizer=f=180:t=q:w=1:g=-11")
        filters.append("equalizer=f=900:t=q:w=1:g=-3")
        filters.append("equalizer=f=3800:t=q:w=1.4:g=1.5")
        filters.append("tremolo=f=5.5:d=0.12")
        filters.append("volume=0.36")
        filters.append("acompressor=threshold=-24dB:ratio=1.6:attack=20:release=150:makeup=0.7")
        filters.append("alimiter=limit=0.55")

    pitch = _clamp(float(pitch_semitones or 0.0), PITCH_SEMITONES_MIN, PITCH_SEMITONES_MAX)
    if abs(pitch) >= 0.05:
        ratio = 2 ** (pitch / 12.0)
        # Pitch without changing duration via asetrate + aTempo compensation.
        filters.append(f"asetrate={int(sample_rate * ratio)},aresample={sample_rate},atempo={1.0 / ratio:.6f}")

    return ",".join(filters)


def options_from_request_and_npc(
    request_values: dict[str, Any] | None,
    npc: dict | None,
) -> dict[str, Any]:
    """Convenience wrapper used by turn handlers / preview."""
    rv = request_values or {}
    return resolve_tts_options(
        npc,
        length_scale=rv.get("tts_length_scale", rv.get("length_scale")),
        noise_scale=rv.get("tts_noise_scale", rv.get("noise_scale")),
        noise_w=rv.get("tts_noise_w", rv.get("noise_w")),
        sentence_silence=rv.get("tts_sentence_silence", rv.get("sentence_silence")),
        speaker_id=rv.get("tts_speaker_id", rv.get("speaker_id")),
        pitch_semitones=rv.get("tts_pitch_semitones", rv.get("pitch_semitones")),
        fx_preset=rv.get("tts_fx", rv.get("fx_preset")),
        denoise_nr=rv.get("tts_denoise_nr", rv.get("denoise_nr")),
        denoise_nf=rv.get("tts_denoise_nf", rv.get("denoise_nf")),
        pronunciation=rv.get("tts_pronunciation", rv.get("pronunciation")),
    )
