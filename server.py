from flask import Flask, request, jsonify, send_file, make_response
import os
import hashlib
import secrets
import yaml
import re
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid
import subprocess
import wave
import array
import threading
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context

from request_timing import (
    attach_timing_headers,
    begin_request_timing,
    log_timing,
    note_first_audio_ms,
)
from faster_whisper import WhisperModel

from llm_router import (
    generate_llm_response,
    get_llm_models_payload,
    get_llm_status,
    get_llm_usage_cost_report,
    reset_usage_tracking,
    select_llm_model,
)
from content_import import (
    ContentImportError,
    analyze_campaign_document,
    detail_location_batch,
    detail_npc_batch,
    discover_campaign_document,
)
from required_dm_checks import (
    format_required_dm_checks_prompt_block,
    generate_npc_reply_text,
    is_language_comprehension_claim,
    resolve_npc_native_language,
)
from dm_channel import (
    build_dm_injection_messages,
    clear_dm_data,
    create_manual_ruling,
    detect_ruling,
    dismiss_pending_check,
    ensure_campaign_id_column,
    format_dm_context_for_prompt,
    get_dm_context,
    init_dm_tables,
    load_dm_messages,
    load_pending_check_by_id,
    load_pending_checks,
    load_pending_checks_for_dm,
    load_ruling_by_id,
    load_rulings,
    load_rulings_for_dm,
    log_dm_turn,
    normalize_campaign_id,
    process_npc_response,
    process_player_message,
    save_dm_message,
    strip_dm_check_tag,
    assistant_history_text,
    is_dm_wait_narration,
)
import dm_channel
from party_campaign import (
    DEFAULT_PLAYER_ID,
    build_party_context,
    format_campaign_setting_rule,
    format_party_context_for_prompt,
    format_player_message_content,
    get_character_options,
    get_session_status,
    init_party_campaign,
    list_campaigns,
    list_parties,
    load_campaign,
    load_party,
    get_active_session_ids,
    resolve_member,
    resolve_request_session,
    select_campaign,
    select_character,
)
from content_pack import (
    clear_request_owner,
    has_pack,
    install_content_pack,
    list_document_ids,
    lookup_document,
    set_request_owner,
)
from session_context import (
    clear_request_context,
    set_request_campaign,
    set_request_context,
)

from scene_state import (
    build_scene_context,
    clear_sub_location,
    clear_time_of_day,
    format_scene_context_for_prompt,
    get_scene_status,
    list_sub_location_options,
    set_sub_location,
    set_time_of_day,
)
from turn_progress import (
    TurnInProgressError,
    TurnSupersededError,
    complete_turn_progress,
    ensure_turn_in_progress,
    fail_turn_progress,
    get_active_turn_progress,
    get_turn_progress,
    get_turn_progress_catalog,
    new_request_id,
    normalize_client_request_id,
    register_turn_progress,
    set_turn_progress,
    turn_is_in_progress,
)
from server_settings import format_llm_rules, get_narrator_tts_settings, load_global_llm_rules
from prompt_caps import get_prompt_caps, save_prompt_caps, DEFAULT_PROMPT_CAPS
from npc_pronouns import get_npc_pronoun_context
from peer_context import build_peer_location_prompt_sections
from tts_processing import (
    DEFAULT_DENOISE_NF,
    DEFAULT_DENOISE_NR,
    DEFAULT_FX_PRESET,
    DEFAULT_LENGTH_SCALE,
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_W,
    DEFAULT_SENTENCE_SILENCE,
    asterisk_body_tts_type,
    build_af_filter,
    clean_for_tts,
    curated_defaults_for_quality,
    infer_model_language,
    load_piper_model_meta,
    options_from_request_and_npc,
    resolve_tts_options,
    split_text_for_streaming,
)
from piper_paths import LEGACY_WINDOWS_PIPER_EXE, resolve_piper_exe
import piper_pool
import memory_db
from memory_db import MEMORY_DB as _SHARED_MEMORY_DB
from memory_db import refresh_memory_db, refresh_memory_db_config

BASE_DIR = Path(__file__).resolve().parent

NPC_DIR = BASE_DIR / "npcs"
LOCATION_DIR = BASE_DIR / "locations"
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(BASE_DIR / "uploads")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(BASE_DIR / "outputs")))

# Kept in sync with dm_channel / memory_db after game_api.env load.
MEMORY_DB = _SHARED_MEMORY_DB
MEMORY_DB_CONFIG = memory_db.CONFIG

PARTIES_DIR = BASE_DIR / "parties"
CAMPAIGNS_DIR = BASE_DIR / "campaigns"


NPC_DIR.mkdir(exist_ok=True)
LOCATION_DIR.mkdir(exist_ok=True)
PARTIES_DIR.mkdir(exist_ok=True)
CAMPAIGNS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Resolved after game_api.env load (see PIPER_EXE assignment below).
DEFAULT_VOICE_MODEL = Path(r"C:\AI\Voices\en\en_US\ryan\medium\en_US-ryan-medium.onnx")
VOICE_PREVIEW_TEXT = os.environ.get(
    "VOICE_PREVIEW_TEXT",
    "Mary had a little lamb, whose fleece was white as snow. "
    "And everywhere that Mary went, the lamb was sure to go.",
)
SEGMENT_PAUSE_SECONDS = 0.18
PIPER_SENTENCE_SILENCE = DEFAULT_SENTENCE_SILENCE

# Standard output format for mixed Piper voice segments.
# Different Piper voices can output different WAV sample rates. Convert all
# narrator/NPC segments to this format before concatenating.
STANDARD_WAV_SAMPLE_RATE = 22050
STANDARD_WAV_CHANNELS = 1
STANDARD_WAV_SAMPLE_FORMAT = "s16"
STANDARD_WAV_CODEC = "pcm_s16le"

# Prompt/memory tuning. Turning async memory on makes the spoken response return faster
# because the extra LLM call for memory extraction happens after audio is generated.
AUTO_MEMORY_EXTRACTION_ENABLED = True
AUTO_MEMORY_EXTRACTION_ASYNC = True
MEMORY_PROMPT_LIMIT = 8
RECENT_TURN_LIMIT = 4

# Lightweight in-process caches for YAML/static context.
# YAML file loading is not the biggest bottleneck, but this avoids repeated disk reads
# and gives the Pi a /npc-preload endpoint to validate/warm context when an NPC changes.
CACHE_LOCK = threading.Lock()
NPC_CACHE = {}
LOCATION_CACHE = {}
PRELOADED_CONTEXTS = {}

# Context pieces are mostly independent I/O reads. A bounded, process-wide pool
# lets them overlap without creating a fresh set of threads for every turn.
NPC_CONTEXT_LOAD_WORKERS = max(
    4,
    int(os.environ.get("NPC_CONTEXT_LOAD_WORKERS", "8")),
)
NPC_CONTEXT_EXECUTOR = ThreadPoolExecutor(
    max_workers=NPC_CONTEXT_LOAD_WORKERS,
    thread_name_prefix="npc-context",
)

# Good first choice. If accuracy is poor, change to "small.en".
# If latency is too high, change to "tiny.en".
STT_MODEL = WhisperModel("base.en", device="cpu", compute_type="int8")

app = Flask(__name__)
# Cap uploads / JSON bodies (STT audio, content packs) to reduce DoS risk.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH_BYTES", str(16 * 1024 * 1024)))

GAME_API_ENV_FILE = BASE_DIR / "game_api.env"


def _load_game_api_env_file() -> None:
    """Load GAME_API_* and MEMORY_* from game_api.env.

    File values win for GAME_API_* / MEMORY_MYSQL_* / MEMORY_DB so a stale shell
    export cannot silently override the secrets file ops edited and verified.
    Other keys still only fill gaps (shell wins).
    """
    if not GAME_API_ENV_FILE.exists():
        return

    try:
        applied: list[str] = []
        skipped: list[str] = []
        for line in GAME_API_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip().lstrip("\ufeff")
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if key.startswith("GAME_API_") or key.startswith("MEMORY_MYSQL_") or key == "MEMORY_DB":
                os.environ[key] = value
                applied.append(key)
                continue
            if key not in os.environ:
                os.environ[key] = value
                applied.append(key)
            else:
                skipped.append(key)
        if applied:
            print(f"[GAME_API] applied from {GAME_API_ENV_FILE.name}: {', '.join(applied)}", flush=True)
        if skipped:
            print(
                f"[GAME_API] left shell values for non-GAME_API keys: {', '.join(skipped)}",
                flush=True,
            )
    except OSError as ex:
        print(f"[GAME_API] failed to read env file: {ex}", flush=True)


_load_game_api_env_file()

# Re-bind after game_api.env so MEMORY_MYSQL_* / MEMORY_DB apply to NPC + DM tables.
MEMORY_DB_CONFIG = refresh_memory_db_config()
MEMORY_DB = refresh_memory_db()
dm_channel.MEMORY_DB = MEMORY_DB
if MEMORY_DB_CONFIG.is_mysql:
    print(
        f"[MEMORY_DB] MySQL {MEMORY_DB_CONFIG.mysql_user}@"
        f"{MEMORY_DB_CONFIG.mysql_host}:{MEMORY_DB_CONFIG.mysql_port}/"
        f"{MEMORY_DB_CONFIG.mysql_database}",
        flush=True,
    )
else:
    print(f"[MEMORY_DB] SQLite {MEMORY_DB}", flush=True)

# Prefer PIPER_EXE / PATH; fall back to legacy Windows path for local Funnel/dev.
try:
    PIPER_EXE = resolve_piper_exe()
except FileNotFoundError:
    # Allow the process to start (health/LLM) even if TTS is not installed yet.
    # synthesize_speech() re-resolves and raises a clear error when TTS is used.
    PIPER_EXE = Path(os.environ.get("PIPER_EXE") or LEGACY_WINDOWS_PIPER_EXE)


def _secret_fingerprint(value: str) -> str:
    """Short non-reversible fingerprint for startup / auth diagnostics."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"len={len(value)} sha256_12={digest[:12]}"


# Alpha / Funnel: bind localhost only; require Bearer token when GAME_API_KEY is set.
GAME_API_KEY = os.environ.get("GAME_API_KEY", "").strip()
REQUIRE_GAME_API_KEY = os.environ.get("REQUIRE_GAME_API_KEY", "").strip().lower() in ("1", "true", "yes")
GAME_API_HOST = os.environ.get("GAME_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
GAME_API_PORT = int(os.environ.get("GAME_API_PORT", "5000"))
AUTH_EXEMPT_PATHS = frozenset({"/health"})

if GAME_API_KEY:
    print(f"[GAME_API] GAME_API_KEY active ({_secret_fingerprint(GAME_API_KEY)})", flush=True)
elif GAME_API_ENV_FILE.exists():
    print(f"[GAME_API] {GAME_API_ENV_FILE.name} present but GAME_API_KEY is empty", flush=True)

if REQUIRE_GAME_API_KEY and not GAME_API_KEY:
    raise SystemExit(
        "REQUIRE_GAME_API_KEY is enabled but GAME_API_KEY is empty. "
        "Set GAME_API_KEY or disable REQUIRE_GAME_API_KEY."
    )


def _extract_game_api_key():
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Game-Api-Key", "").strip()


def _api_keys_match(provided: str, expected: str) -> bool:
    if not provided or not expected or len(provided) != len(expected):
        return False
    return secrets.compare_digest(provided, expected)


def _client_error(ex: BaseException, status: int = 500, *, not_found: bool = False):
    """Log exception details server-side; return a generic client message."""
    print(f"[ERROR] {type(ex).__name__}: {ex!r}", flush=True)
    if not_found or status == 404:
        return jsonify({"error": "Not found"}), 404
    if status == 400:
        return jsonify({"error": "Bad request"}), 400
    return jsonify({"error": "Internal server error"}), status


@app.before_request
def require_game_api_key():
    if request.method == "OPTIONS":
        return None

    owner = (
        request.headers.get("X-Campaign-Owner-User-Id")
        or request.headers.get("X-Owner-User-Id")
        or ""
    ).strip()
    campaign = (
        request.headers.get("X-Campaign-Id")
        or request.headers.get("X-Active-Campaign-Id")
        or ""
    ).strip()

    if request.method in ("POST", "PUT", "PATCH"):
        # Best-effort: JSON/form may carry owner/campaign for session scoping.
        body = None
        if request.mimetype and "json" in (request.mimetype or ""):
            body = request.get_json(silent=True) or {}
            if isinstance(body, dict):
                if not owner:
                    owner = str(body.get("owner_user_id") or "").strip()
                if not campaign:
                    campaign = str(body.get("campaign_id") or body.get("id") or "").strip()
        if request.form:
            if not owner:
                owner = str(request.form.get("owner_user_id") or "").strip()
            if not campaign:
                campaign = str(request.form.get("campaign_id") or "").strip()

    if not campaign:
        campaign = (request.args.get("campaign_id") or "").strip()

    set_request_context(owner or None, campaign or None)

    if REQUIRE_GAME_API_KEY and request.path == "/llm/reset-usage":
        return jsonify({"error": "Not found"}), 404

    # Fail closed when a key is configured, or when REQUIRE_GAME_API_KEY demands one.
    if (GAME_API_KEY or REQUIRE_GAME_API_KEY) and request.path not in AUTH_EXEMPT_PATHS:
        provided = _extract_game_api_key()
        if not GAME_API_KEY or not _api_keys_match(provided, GAME_API_KEY):
            auth_hdr = request.headers.get("Authorization", "")
            print(
                f"[AUTH] rejected {request.method} {request.path} from {request.remote_addr} "
                f"(authorization={'yes' if auth_hdr else 'no'}, "
                f"x_game_api_key={'yes' if request.headers.get('X-Game-Api-Key') else 'no'}, "
                f"provided={_secret_fingerprint(provided) if provided else 'len=0'}, "
                f"expected={_secret_fingerprint(GAME_API_KEY) if GAME_API_KEY else 'len=0'})",
                flush=True,
            )
            return jsonify({"error": "Unauthorized"}), 401

    return None


def resolve_voices_dir() -> Path:
    """Root folder to scan recursively for Piper *.onnx voice models."""
    explicit = os.environ.get("VOICES_DIR", "").strip()
    if explicit:
        return Path(explicit)

    # Default: sibling Voices folder above the middleware repo (e.g. C:\AI\Voices).
    sibling = BASE_DIR.parent / "Voices"
    if sibling.exists():
        return sibling

    for parent in DEFAULT_VOICE_MODEL.parents:
        if parent.name.lower() == "voices":
            return parent

    return DEFAULT_VOICE_MODEL.parent.parent


VOICES_DIR = resolve_voices_dir()

# Cached Piper inventory — full-tree rglob on a large Voices share is slow enough
# to 504 the Yaml Editor /api/voices proxy (30s) when scanned on every request.
_VOICE_CACHE_LOCK = threading.Lock()
_VOICE_BUILD_LOCK = threading.Lock()  # single-flight full-tree rebuilds
_VOICE_CACHE: dict[str, Any] = {
    "built_at": 0.0,
    "voices": [],
    "by_name": {},
}
_VOICE_CACHE_TTL_SECONDS = float(os.environ.get("VOICES_CACHE_TTL_SECONDS", "300"))


def _voice_cache_snapshot() -> list[dict[str, Any]] | None:
    """Return a copy of cached voices when fresh, else None."""
    now = time.perf_counter()
    with _VOICE_CACHE_LOCK:
        age = now - float(_VOICE_CACHE.get("built_at") or 0.0)
        if _VOICE_CACHE["voices"] and age < _VOICE_CACHE_TTL_SECONDS:
            return list(_VOICE_CACHE["voices"])
    return None


def _build_voice_inventory() -> tuple[list[dict[str, Any]], dict[str, Path]]:
    if not VOICES_DIR.exists():
        return [], {}

    voices: list[dict[str, Any]] = []
    by_name: dict[str, Path] = {}
    seen: set[str] = set()

    # Avoid Path.resolve() here — on UNC/network trees it can be very expensive.
    for path in sorted(VOICES_DIR.rglob("*.onnx"), key=lambda item: str(item).lower()):
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)

        try:
            rel = path.relative_to(VOICES_DIR).as_posix()
        except ValueError:
            rel = path.name

        meta = load_piper_model_meta(path)
        curated = curated_defaults_for_quality(meta["quality"])
        quality = meta["quality"]
        quality_note = ""
        if quality != "unknown":
            quality_note = f" [{quality}]"
        if meta["num_speakers"] > 1:
            quality_note += f" ({meta['num_speakers']} speakers)"

        voices.append({
            "label": f"{rel}{quality_note}",
            # Relative to VOICES_DIR (posix). Never expose host absolute/UNC paths.
            "path": rel,
            "dataset": meta["dataset"],
            "quality": quality,
            # Additive: the editor derived this client-side from the path.
            "language": infer_model_language(rel),
            "num_speakers": meta["num_speakers"],
            "speakers": meta["speakers"],
            "has_speaker_map": meta["has_speaker_map"],
            "recommended": {
                "length_scale": curated["length_scale"],
                "noise_scale": curated["noise_scale"],
                "noise_w": curated["noise_w"],
            },
        })
        name_key = path.name.lower()
        by_name.setdefault(name_key, path)
        by_name.setdefault(rel.lower(), path)

    # Alphabetical by display label (case-insensitive).
    voices.sort(key=lambda item: str(item.get("label", "")).lower())

    return voices, by_name


def refresh_voice_inventory(*, force: bool = False) -> list[dict[str, Any]]:
    if not force:
        cached = _voice_cache_snapshot()
        if cached is not None:
            return cached

    # Single-flight: concurrent /voices + preview must not each rglob the tree.
    with _VOICE_BUILD_LOCK:
        if not force:
            cached = _voice_cache_snapshot()
            if cached is not None:
                return cached

        started = time.perf_counter()
        voices, by_name = _build_voice_inventory()
        elapsed = time.perf_counter() - started
        print(
            f"[VOICES] inventory rebuilt in {elapsed:.3f}s "
            f"({len(voices)} model(s) under {VOICES_DIR})",
            flush=True,
        )
        with _VOICE_CACHE_LOCK:
            _VOICE_CACHE["built_at"] = time.perf_counter()
            _VOICE_CACHE["voices"] = voices
            _VOICE_CACHE["by_name"] = by_name
            return list(voices)


def list_piper_voices() -> list[dict[str, Any]]:
    return refresh_voice_inventory(force=False)


def lookup_cached_voice_by_name(filename: str) -> Path | None:
    """Look up a voice by basename or relative posix path (case-insensitive)."""
    key = (filename or "").replace("\\", "/").strip().lower().lstrip("/")
    if not key:
        return None

    refresh_voice_inventory(force=False)
    with _VOICE_CACHE_LOCK:
        path = _VOICE_CACHE["by_name"].get(key)
        if path is None:
            path = _VOICE_CACHE["by_name"].get(Path(key).name.lower())
        return path if isinstance(path, Path) else None


def resolve_voice_model_path(voice_model: str | None) -> Path:
    """Resolve voice_model to a real *.onnx under VOICES_DIR.

    Contract: callers should pass a path relative to VOICES_DIR (posix), e.g.
    ``en/en_US/ryan/medium/en_US-ryan-medium.onnx``. Absolute/UNC paths are
    rejected for path-traversal hardening; legacy absolutes may still resolve
    via basename/relative-key inventory lookup only (never by opening the
    absolute path itself).

    Important: do **not** call Path.resolve() on the Voices tree here. On mapped
    drives / UNC shares resolve() can take many seconds per call and made
    preview/select feel like 15–20s timeouts.
    """
    if not voice_model or not str(voice_model).strip():
        return DEFAULT_VOICE_MODEL

    raw = str(voice_model).strip()

    def _from_cache(key: str) -> Path | None:
        cached = lookup_cached_voice_by_name(key)
        if cached is None:
            return None
        # Inventory paths are already under VOICES_DIR; trust them without resolve().
        return cached

    # Harden: never open caller-supplied absolute/UNC paths.
    if Path(raw).is_absolute() or raw.startswith(("\\\\", "//")):
        matched = _from_cache(raw)
        if matched is not None:
            return matched
        raise FileNotFoundError(f"Piper voice model not found: {voice_model}")

    if ".." in Path(raw).parts:
        raise FileNotFoundError(f"Piper voice model not found: {voice_model}")

    normalized = raw.replace("\\", "/").lstrip("/")
    candidate = VOICES_DIR / Path(normalized)
    # Syntactic containment (no filesystem resolve).
    try:
        candidate.relative_to(VOICES_DIR)
    except ValueError as ex:
        raise FileNotFoundError(f"Piper voice model not found: {voice_model}") from ex

    if candidate.suffix.lower() == ".onnx" and candidate.is_file():
        return candidate

    matched = _from_cache(normalized)
    if matched is not None:
        return matched

    raise FileNotFoundError(f"Piper voice model not found: {voice_model}")


if VOICES_DIR.exists():
    _startup_voices = refresh_voice_inventory(force=True)
    print(
        f"[VOICES] Piper scan root: {VOICES_DIR} ({len(_startup_voices)} model(s))",
        flush=True,
    )
else:
    print(f"[VOICES] Piper scan root not found: {VOICES_DIR}", flush=True)


@app.teardown_request
def _clear_content_owner(_exc=None):
    clear_request_context()
    clear_request_owner()


def _header_b64(value: str | None, *, max_chars: int = 6000) -> str:
    """UTF-8 text as base64url for HTTP response headers (ASCII-safe)."""
    text = (value or "").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    raw = text.encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def get_db():
    """Open memory/DM DB (MySQL when MEMORY_MYSQL_HOST is set, else SQLite)."""
    if memory_db.CONFIG.is_sqlite and MEMORY_DB != memory_db.CONFIG.sqlite_path:
        return memory_db.connect(
            memory_db.MemoryDbConfig(backend="sqlite", sqlite_path=MEMORY_DB)
        )
    return memory_db.get_db()


def init_memory_db():
    with get_db() as conn:
        memory_db.init_npc_memory_tables(conn)
        ensure_campaign_id_column(conn, "conversation_turns")
        ensure_campaign_id_column(conn, "npc_memories")
        conn.commit()

    init_dm_tables()


def save_turn(
    npc_id: str,
    player_id: str,
    role: str,
    content: str,
    campaign_id: str = "",
    *,
    connection: memory_db.DbConnection | None = None,
):
    campaign_id = normalize_campaign_id(campaign_id)
    content = (content or "").strip()
    if not content:
        return

    with (nullcontext(connection) if connection is not None else get_db()) as conn:
        conn.execute("""
            INSERT INTO conversation_turns
                (campaign_id, npc_id, player_id, role, content, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?)
        """, (
            campaign_id,
            npc_id,
            player_id,
            role,
            content,
            datetime.utcnow().isoformat()
        ))


def save_player_npc_conversation_turns(
    npc_id: str,
    player_id: str,
    player_text: str,
    npc_response: dict,
    *,
    campaign_id: str = "",
    save_user: bool = True,
    connection: memory_db.DbConnection | None = None,
) -> str:
    """Persist the player line and in-character NPC draft, never wait-narration.

    Callers that already saved the user utterance before a long LLM wait should pass
    ``save_user=False`` so a force-displaced turn still leaves the player line in
    history for DM auto-response follow-ups.
    """
    if save_user:
        save_turn(
            npc_id,
            player_id,
            "user",
            player_text,
            campaign_id=campaign_id,
            connection=connection,
        )
    history_text = assistant_history_text(npc_response)
    if history_text:
        save_turn(
            npc_id,
            player_id,
            "assistant",
            history_text,
            campaign_id=campaign_id,
            connection=connection,
        )
    return history_text


def load_recent_turns(
    npc_id: str,
    player_id: str,
    limit: int = 10,
    campaign_id: str = "",
    *,
    connection: memory_db.DbConnection | None = None,
):
    campaign_id = normalize_campaign_id(campaign_id)
    with (nullcontext(connection) if connection is not None else get_db()) as conn:
        rows = conn.execute("""
            SELECT role, content
            FROM conversation_turns
            WHERE campaign_id = ?
              AND npc_id = ?
              AND player_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (campaign_id, npc_id, player_id, max(limit * 3, 20))).fetchall()

    filtered = [
        row
        for row in reversed(rows)
        if not is_dm_wait_narration(row["content"] if row["content"] is not None else "")
    ]
    return filtered[-limit:]


def save_memory(npc_id: str, player_id: str, memory: str, campaign_id: str = ""):
    campaign_id = normalize_campaign_id(campaign_id)
    memory = (memory or "").strip()
    if not memory:
        return

    with get_db() as conn:
        conn.execute("""
            INSERT INTO npc_memories
                (campaign_id, npc_id, player_id, memory, created_at)
            VALUES
                (?, ?, ?, ?, ?)
        """, (
            campaign_id,
            npc_id,
            player_id,
            memory,
            datetime.utcnow().isoformat()
        ))
        conn.commit()


def memory_exists(npc_id: str, player_id: str, memory: str, campaign_id: str = "") -> bool:
    campaign_id = normalize_campaign_id(campaign_id)
    normalized = (memory or "").strip().lower()

    if not normalized:
        return True

    with get_db() as conn:
        rows = conn.execute("""
            SELECT memory
            FROM npc_memories
            WHERE campaign_id = ?
              AND npc_id = ?
              AND player_id = ?
        """, (campaign_id, npc_id, player_id)).fetchall()

    for row in rows:
        if row["memory"].strip().lower() == normalized:
            return True

    return False


def save_memory_if_new(npc_id: str, player_id: str, memory: str, campaign_id: str = ""):
    memory = (memory or "").strip()

    if not memory:
        return

    if memory_exists(npc_id, player_id, memory, campaign_id=campaign_id):
        return

    save_memory(npc_id, player_id, memory, campaign_id=campaign_id)


def load_memories(
    npc_id: str,
    player_id: str,
    limit: int = 20,
    campaign_id: str = "",
    *,
    connection: memory_db.DbConnection | None = None,
):
    campaign_id = normalize_campaign_id(campaign_id)
    with (nullcontext(connection) if connection is not None else get_db()) as conn:
        rows = conn.execute("""
            SELECT memory
            FROM npc_memories
            WHERE campaign_id = ?
              AND npc_id = ?
              AND player_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (campaign_id, npc_id, player_id, limit)).fetchall()

    return [row["memory"] for row in reversed(rows)]


def prepare_turn_database_phase(
    npc_id: str,
    player_id: str,
    player_text: str | None,
    *,
    campaign_id: str = "",
    save_player_turn: bool = True,
    detect_player_claim: bool = True,
) -> dict[str, Any]:
    """Run all mutable pre-LLM database work in one short transaction.

    The returned values are plain Python data. The connection is returned to the
    pool before prompt construction and, critically, before the LLM wait starts.
    """
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
        memories = load_memories(
            npc_id,
            player_id,
            limit=MEMORY_PROMPT_LIMIT,
            campaign_id=campaign_id,
            connection=conn,
        )
        recent_turns = load_recent_turns(
            npc_id,
            player_id,
            limit=RECENT_TURN_LIMIT,
            campaign_id=campaign_id,
            connection=conn,
        )
        dm_messages = load_dm_messages(
            npc_id,
            player_id,
            limit=10,
            campaign_id=campaign_id,
            connection=conn,
        )
        pending_checks = load_pending_checks(
            npc_id,
            player_id,
            status="pending",
            limit=None,
            campaign_id=campaign_id,
            connection=conn,
            for_update=True,
        )
        rulings = load_rulings(
            npc_id,
            player_id,
            limit=50,
            campaign_id=campaign_id,
            connection=conn,
        )

        player_claim = None
        if detect_player_claim and player_text:
            player_claim = process_player_message(
                npc_id,
                player_id,
                player_text,
                campaign_id=campaign_id,
                connection=conn,
                pending_checks=pending_checks,
                rulings=rulings,
            )

        if save_player_turn and player_text:
            save_turn(
                npc_id,
                player_id,
                "user",
                player_text,
                campaign_id=campaign_id,
                connection=conn,
            )

    return {
        "memories": memories,
        "recent_turns": recent_turns,
        "player_claim": player_claim,
        "dm_context": {
            "campaign_id": campaign_id,
            "messages": dm_messages[-10:],
            "pending_checks": pending_checks[-10:],
            "rulings": rulings[-10:],
        },
    }


def finalize_turn_database_phase(
    npc_id: str,
    player_id: str,
    player_text: str,
    npc_text: str,
    *,
    player_pending_id: int | None = None,
    campaign_id: str = "",
    npc: dict[str, Any] | None = None,
    create_pending_checks: bool = True,
    fallback_spoken_text: str = "",
) -> dict[str, Any]:
    """Process and persist an NPC reply, returning one pending snapshot.

    This is intentionally complete before any Piper work. Logging, JSON payloads,
    and response headers reuse the returned snapshot instead of querying again.
    """
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
        pending_checks = []
        if create_pending_checks:
            pending_checks = load_pending_checks(
                npc_id,
                player_id,
                status="pending",
                limit=None,
                campaign_id=campaign_id,
                connection=conn,
                for_update=True,
            )
        npc_response = process_npc_response(
            npc_id,
            player_id,
            player_text,
            npc_text,
            player_pending_id=player_pending_id,
            campaign_id=campaign_id,
            create_pending_checks=create_pending_checks,
            npc=npc,
            connection=conn,
            pending_checks=pending_checks,
        )
        if not npc_response.get("spoken_text") and fallback_spoken_text:
            npc_response["spoken_text"] = fallback_spoken_text
        history_npc_text = assistant_history_text(npc_response)
        if history_npc_text:
            save_turn(
                npc_id,
                player_id,
                "assistant",
                history_npc_text,
                campaign_id=campaign_id,
                connection=conn,
            )

    return {
        "npc_response": npc_response,
        "history_npc_text": history_npc_text,
        "pending_items": pending_checks,
        "pending_count": len(pending_checks),
    }


def clear_memories(
    npc_id: str,
    player_id: str = "",
    campaign_id: str = "",
    *,
    all_players: bool = False,
) -> int:
    campaign_id = normalize_campaign_id(campaign_id)
    npc_id = (npc_id or "").strip()
    player_id = (player_id or "").strip()
    with get_db() as conn:
        if all_players or not player_id:
            cur = conn.execute("""
                DELETE FROM npc_memories
                WHERE campaign_id = ?
                  AND npc_id = ?
            """, (campaign_id, npc_id))
        else:
            cur = conn.execute("""
                DELETE FROM npc_memories
                WHERE campaign_id = ?
                  AND npc_id = ?
                  AND player_id = ?
            """, (campaign_id, npc_id, player_id))
        conn.commit()
        return cur.rowcount


def clear_recent_turns(
    npc_id: str,
    player_id: str = "",
    campaign_id: str = "",
    *,
    all_players: bool = False,
) -> int:
    campaign_id = normalize_campaign_id(campaign_id)
    npc_id = (npc_id or "").strip()
    player_id = (player_id or "").strip()
    with get_db() as conn:
        if all_players or not player_id:
            cur = conn.execute("""
                DELETE FROM conversation_turns
                WHERE campaign_id = ?
                  AND npc_id = ?
            """, (campaign_id, npc_id))
        else:
            cur = conn.execute("""
                DELETE FROM conversation_turns
                WHERE campaign_id = ?
                  AND npc_id = ?
                  AND player_id = ?
            """, (campaign_id, npc_id, player_id))
        conn.commit()
        return cur.rowcount


def load_yaml_file_cached(path: Path, cache: dict, cache_key: str) -> dict:
    """
    Load a YAML file with mtime-based caching.

    If the file changes on disk, it is automatically re-read on the next request.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")

    modified_at = path.stat().st_mtime

    with CACHE_LOCK:
        cached = cache.get(cache_key)
        if cached and cached.get("modified_at") == modified_at:
            return cached["data"]

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, dict):
        from yaml_list_text import repair_string_list_fields

        repair_string_list_fields(data)

    with CACHE_LOCK:
        cache[cache_key] = {
            "modified_at": modified_at,
            "data": data
        }

    return data


def load_npc(npc_id: str) -> dict:
    npc_id = (npc_id or "").strip()
    packed = lookup_document("npc", npc_id)
    if packed is not None:
        data = dict(packed)
        data.setdefault("id", npc_id)
        data.setdefault("npc_id", npc_id)
        return data

    path = NPC_DIR / f"{npc_id}.yaml"
    if not path.exists():
        # Also support .yml for convenience.
        path = NPC_DIR / f"{npc_id}.yml"

    if not path.exists():
        raise FileNotFoundError(f"Unknown NPC: {npc_id}")

    return load_yaml_file_cached(path, NPC_CACHE, npc_id)


def load_location(location_id: str | None) -> dict:
    location_id = (location_id or "").strip()
    if not location_id:
        return {}

    packed = lookup_document("location", location_id)
    if packed is not None:
        data = dict(packed)
        data.setdefault("id", location_id)
        return data

    path = LOCATION_DIR / f"{location_id}.yaml"
    if not path.exists():
        # Also support .yml for convenience.
        path = LOCATION_DIR / f"{location_id}.yml"

    if not path.exists():
        return {
            "id": location_id,
            "name": location_id,
            "error": f"Location file not found for location_id: {location_id}"
        }

    return load_yaml_file_cached(path, LOCATION_CACHE, location_id)


def format_list_items(items, formatter=None, *, max_items: int | None = None) -> str:
    if not items:
        return ""

    lines = []
    for item in items:
        if max_items is not None and len(lines) >= max_items:
            break
        if formatter:
            rendered = formatter(item)
        elif isinstance(item, dict):
            name = item.get("name", "")
            role = item.get("role", "")
            description = item.get("description", "")
            public_knowledge = item.get("public_knowledge", "")
            parts = [x for x in [name, role, description, public_knowledge] if x]
            rendered = ": ".join(parts[:1]) + (" - " + "; ".join(parts[1:]) if len(parts) > 1 else "")
        else:
            rendered = str(item)

        rendered = rendered.strip()
        if rendered:
            lines.append(f"- {rendered}")

    return "\n".join(lines)


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1].rstrip() + "…"


def format_location_context(location: dict, caps: dict | None = None) -> str:
    if not location:
        return "No specific local setting context is loaded."

    if location.get("error"):
        return f"Location context warning: {location['error']}"

    caps = caps or get_prompt_caps()
    name = location.get("name", location.get("id", "Unknown location"))
    location_type = location.get("type", "")
    summary = _truncate_text(
        str(location.get("summary") or ""),
        caps.get("location_summary_max_chars", 300),
    )

    current_events = format_list_items(
        location.get("current_events", []),
        max_items=caps.get("location_events_max_items", 5),
    )
    important_locations = format_list_items(
        location.get("important_locations", []),
        max_items=caps.get("location_places_max_items", 5),
    )
    rumors = format_list_items(
        location.get("rumors", []),
        max_items=caps.get("location_rumors_max_items", 5),
    )
    local_tone = format_list_items(
        location.get("local_tone", []),
        max_items=caps.get("location_tone_max_items", 5),
    )
    knowledge_rules = format_list_items(
        location.get("knowledge_rules", []),
        max_items=caps.get("location_knowledge_rules_max_items", 5),
    )

    # notable_npcs removed: co-located NPCs come from peer directory + public_facts.
    return f"""
Location name: {name}
Location type: {location_type}

Summary:
{summary}

Current local events:
{current_events}

Important local places:
{important_locations}

Local rumors:
{rumors}

Local tone:
{local_tone}

Location knowledge rules:
{knowledge_rules}
""".strip()


def get_npc_context(
    npc_id: str,
    player_id: str = DEFAULT_PLAYER_ID,
    party_context: dict | None = None,
    campaign_id: str | None = None,
    *,
    include_dm_context: bool = True,
    include_database_context: bool = True,
) -> dict:
    """
    Load all context pieces used for an NPC turn.

    Independent reads overlap in the context worker pool. Location and scene reads
    begin as soon as the NPC load provides its location_id. Static YAML pieces are
    mtime-cached. Memories/recent turns stay live because they change during play.
    Turn routes can skip all database context because their pre-LLM database phase
    loads one consistent snapshot and saves the player line in one transaction.
    Preload and legacy callers retain the original defaults.
    """
    overall_started = time.perf_counter()
    party_context = party_context or build_party_context(player_id=player_id)
    resolved_campaign_id = normalize_campaign_id(
        campaign_id if campaign_id is not None else party_context.get("campaign_id")
    )

    def submit_timed(name: str, func, *args, **kwargs):
        # Context variables do not automatically cross thread boundaries. Give each
        # task its own snapshot so tenant/campaign context remains available to
        # content-pack and scene-state helpers.
        caller_context = copy_context()

        def worker():
            started = time.perf_counter()
            value = caller_context.run(func, *args, **kwargs)
            return name, value, time.perf_counter() - started

        return NPC_CONTEXT_EXECUTOR.submit(worker)

    futures = []
    task_seconds: dict[str, float] = {}

    npc_future = submit_timed("npc", load_npc, npc_id)
    memories_future = None
    recent_turns_future = None
    if include_database_context:
        memories_future = submit_timed(
            "memories",
            load_memories,
            npc_id,
            player_id,
            limit=MEMORY_PROMPT_LIMIT,
            campaign_id=resolved_campaign_id,
        )
        recent_turns_future = submit_timed(
            "recent_turns",
            load_recent_turns,
            npc_id,
            player_id,
            limit=RECENT_TURN_LIMIT,
            campaign_id=resolved_campaign_id,
        )
    dm_context_future = None
    if include_dm_context and include_database_context:
        dm_context_future = submit_timed(
            "dm_context",
            get_dm_context,
            npc_id,
            player_id,
            campaign_id=resolved_campaign_id,
        )
    futures.append(npc_future)
    if memories_future is not None:
        futures.append(memories_future)
    if recent_turns_future is not None:
        futures.append(recent_turns_future)
    if dm_context_future is not None:
        futures.append(dm_context_future)

    try:
        name, npc, elapsed = npc_future.result()
        task_seconds[name] = elapsed

        # These two reads depend only on location_id, so start them together as
        # soon as the NPC is available while the database reads continue.
        location_id = npc.get("location_id")
        location_future = submit_timed("location", load_location, location_id)
        scene_context_future = submit_timed(
            "scene_context", build_scene_context, location_id
        )
        futures.extend([location_future, scene_context_future])

        memories = []
        recent_turns = []
        if memories_future is not None:
            name, memories, elapsed = memories_future.result()
            task_seconds[name] = elapsed
        if recent_turns_future is not None:
            name, recent_turns, elapsed = recent_turns_future.result()
            task_seconds[name] = elapsed
        dm_context = {}
        if dm_context_future is not None:
            name, dm_context, elapsed = dm_context_future.result()
            task_seconds[name] = elapsed
        name, location, elapsed = location_future.result()
        task_seconds[name] = elapsed
        name, scene_context, elapsed = scene_context_future.result()
        task_seconds[name] = elapsed
    except Exception:
        for future in futures:
            future.cancel()
        raise

    wall_seconds = time.perf_counter() - overall_started
    print(
        f"[CONTEXT] npc={npc_id} player={player_id} campaign={resolved_campaign_id} "
        f"parallel_wall={wall_seconds:.3f}s "
        f"tasks={json.dumps({key: round(value, 3) for key, value in task_seconds.items()})}",
        flush=True,
    )

    context = {
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": party_context.get("party_id"),
        "campaign_id": resolved_campaign_id,
        "party_context": party_context,
        "scene_context": scene_context,
        "npc": npc,
        "location": location,
        "memories": memories,
        "recent_turns": recent_turns,
        "dm_context": dm_context,
    }

    with CACHE_LOCK:
        PRELOADED_CONTEXTS[(resolved_campaign_id, npc_id, player_id)] = context

    return context


def preload_npc_context(npc_id: str, player_id: str = DEFAULT_PLAYER_ID) -> dict:
    return get_npc_context(npc_id, player_id)


def load_turn_pre_llm_context(
    npc_id: str,
    player_id: str,
    *,
    player_text: str | None,
    party_context: dict | None,
    campaign_id: str,
    save_player_turn: bool = True,
    detect_player_claim: bool = True,
) -> dict[str, Any]:
    """Validate static context before starting the mutating pre-LLM phase.

    The database phase saves the player turn and may create a pending DM check,
    so it must never run speculatively. ``Future.cancel()`` cannot stop a worker
    that has already started and therefore cannot provide rollback semantics.
    """
    static_context = get_npc_context(
        npc_id,
        player_id,
        party_context=party_context,
        campaign_id=campaign_id,
        include_dm_context=False,
        include_database_context=False,
    )
    database_context = prepare_turn_database_phase(
        npc_id,
        player_id,
        player_text,
        campaign_id=campaign_id,
        save_player_turn=save_player_turn,
        detect_player_claim=detect_player_claim,
    )

    # Do not mutate the static-only entry stored in PRELOADED_CONTEXTS.
    return {**static_context, **database_context}


def run_memory_extraction_background(
    npc: dict,
    npc_id: str,
    player_id: str,
    player_text: str,
    npc_text: str,
    campaign_id: str = "",
) -> None:
    if not AUTO_MEMORY_EXTRACTION_ENABLED:
        return

    def worker():
        try:
            save_extracted_memories(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=npc_text,
                campaign_id=campaign_id,
            )
        except Exception:
            # Memory extraction must never break live play.
            pass

    if AUTO_MEMORY_EXTRACTION_ASYNC:
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    else:
        worker()


@app.get("/npcs")
def list_npcs():
    npcs = []
    if has_pack():
        npc_ids = list_document_ids("npc")
    else:
        from session_context import get_request_campaign, get_request_owner

        owner = (get_request_owner() or "").strip()
        campaign = (get_request_campaign() or "").strip()
        # Tenant-scoped requests must never fall back to shared middleware NPC_DIR
        # (that mixes other users' / local demo content into DM Console lists).
        if owner or campaign:
            return jsonify({
                "npcs": [],
                "warning": "No content pack installed for this owner/campaign.",
            })
        npc_ids = [
            path.stem for path in sorted(list(NPC_DIR.glob("*.yaml")) + list(NPC_DIR.glob("*.yml")))
        ]

    for npc_id in npc_ids:
        try:
            npc = load_npc(npc_id)
            display_name = npc.get("name", npc_id)
            role = npc.get("role", "")
            voice = npc.get("voice", "")
            location_id = npc.get("location_id", "")
            pronoun_context = get_npc_pronoun_context(npc)

            npcs.append({
                "id": npc_id,
                "name": display_name,
                "role": role,
                "voice": voice,
                "gender": pronoun_context["gender"],
                "pronouns": pronoun_context["display"],
                "location_id": location_id
            })

        except Exception as ex:
            npcs.append({
                "id": npc_id,
                "name": npc_id,
                "role": "",
                "voice": "",
                "error": "load_failed"
            })

    return jsonify({
        "npcs": npcs
    })



def format_npc_llm_rules(npc: dict) -> str:
    """
    Format optional NPC-specific LLM rules from YAML.

    Supported fields:
      llm_rules: list[str] or multiline string
      npc_rules: alias for llm_rules
    """
    raw = npc.get("llm_rules")
    if raw is None:
        raw = npc.get("npc_rules")

    return format_llm_rules(raw)


def build_system_prompt(
    npc: dict,
    scene: dict,
    memories: list[str] | None = None,
    location: dict | None = None,
    dm_context: dict | None = None,
    party_context: dict | None = None,
    scene_context: dict | None = None,
    player_text: str | None = None,
    about_npc_id: str | None = None,
) -> str:
    caps = get_prompt_caps()
    public_facts = "\n".join(f"- {x}" for x in npc.get("public_facts", []))
    known_facts = "\n".join(f"- {x}" for x in npc.get("known_facts", []))
    secrets = "\n".join(f"- {x}" for x in npc.get("secrets", []))
    memories_text = "\n".join(f"- {x}" for x in (memories or []))
    location_context = format_location_context(location or {}, caps=caps)
    directory_block, peer_facts_block = build_peer_location_prompt_sections(
        npc,
        player_text=player_text,
        about_npc_id=about_npc_id,
        caps=caps,
    )
    peer_sections = ""
    if directory_block:
        peer_sections += f"\n\n{directory_block}"
    if peer_facts_block:
        peer_sections += f"\n\n{peer_facts_block}"
    pronoun_context = get_npc_pronoun_context(npc)
    dm_context_text = format_dm_context_for_prompt(dm_context)
    party_context_text = format_party_context_for_prompt(party_context)
    if scene_context is None:
        scene_context = build_scene_context(npc.get("location_id"))
    scene_context_text = format_scene_context_for_prompt(scene_context)
    optional_scene_notes = (scene.get("description") or "").strip()
    optional_scene_block = optional_scene_notes or "None."
    global_rules_text = load_global_llm_rules()
    npc_rules_text = format_npc_llm_rules(npc)
    npc_rules_block = ""
    if npc_rules_text:
        npc_rules_block = f"""

NPC-specific rules for {npc["name"]} (these override conflicting global rules):
{npc_rules_text}"""

    required_checks_block = format_required_dm_checks_prompt_block(npc, dm_context)
    campaign_setting_rule = format_campaign_setting_rule(party_context)
    campaign_setting_block = f"\n{campaign_setting_rule}\n" if campaign_setting_rule else "\n"

    return f"""
You are portraying an NPC in a tabletop RPG.

NPC name: {npc["name"]}
Role: {npc["role"]}
Gender/identity: {pronoun_context["gender"]}
Pronouns: {pronoun_context["display"]}
Pronoun details: subject={pronoun_context["subject"]}, object={pronoun_context["object"]}, possessive adjective={pronoun_context["possessive_adjective"]}, possessive pronoun={pronoun_context["possessive_pronoun"]}, reflexive={pronoun_context["reflexive"]}
Self-reference rule: These gender/pronouns are YOURS. When peer NPC identity appears later in this prompt, use it only for third-person references to that other NPC — never as your own.
Voice style: {npc.get("voice", "natural spoken dialogue")}
Personality: {", ".join(npc.get("personality", []))}

Public facts:
{public_facts}

NPC-known facts:
{known_facts}

Hidden or secret facts:
{secrets}

Long-term memories about this party:
{memories_text}

Local setting context:
{location_context}{peer_sections}

Authoritative scene context:
{scene_context_text}

Optional scene notes from request:
{optional_scene_block}

Party and campaign context:
{party_context_text}

DM channel context:
{dm_context_text}
{required_checks_block}
Rules:{campaign_setting_block}{global_rules_text}{npc_rules_block}
""".strip()


def build_npc_llm_messages(
    npc: dict,
    scene: dict,
    memories: list[str] | None,
    location: dict | None,
    *,
    recent_turns=None,
    history: list[dict] | None = None,
    player_text: str | None = None,
    dm_context: dict | None = None,
    party_context: dict | None = None,
    scene_context: dict | None = None,
    about_npc_id: str | None = None,
) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                npc,
                scene,
                memories,
                location,
                dm_context=dm_context,
                party_context=party_context,
                scene_context=scene_context,
                player_text=player_text,
                about_npc_id=about_npc_id,
            ),
        }
    ]

    for dm_message in build_dm_injection_messages(dm_context):
        messages.append(dm_message)

    if history:
        for entry in history[-8:]:
            messages.append({
                "role": entry["role"],
                "content": entry["content"],
            })
    elif recent_turns:
        for turn in recent_turns:
            messages.append({
                "role": turn["role"],
                "content": turn["content"],
            })

    if player_text is not None:
        messages.append({
            "role": "user",
            "content": format_player_message_content(player_text, party_context),
        })

    return messages


def _enrich_pending_check_display(item: dict | None) -> dict:
    """Attach human-readable player/NPC names for the DM console."""
    row = dict(item or {})
    npc_id = (row.get("npc_id") or "").strip()
    player_id = (row.get("player_id") or "").strip()

    npc_name = npc_id
    if npc_id:
        try:
            npc = load_npc(npc_id)
            npc_name = (npc.get("name") or npc_id).strip() or npc_id
        except Exception:
            npc_name = npc_id

    player_name = player_id
    if player_id:
        try:
            member = resolve_member(player_id)
            player_name = (member.get("name") or player_id).strip() or player_id
        except Exception:
            player_name = player_id

    row["npc_name"] = npc_name
    row["player_name"] = player_name
    return row


def _json_body_flag(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _turn_busy_message(progress: dict) -> str:
    route = (progress.get("route") or "").strip()
    if route == "/dm-ruling-auto-response":
        return (
            progress.get("message")
            or "The DM ruled — the NPC is responding to your last exchange. Please wait."
        )
    return progress.get("message") or "Wait for the current NPC response before sending another message."


def _turn_in_progress_response(progress: dict):
    return jsonify({
        **progress,
        "error": "turn_in_progress",
        "message": _turn_busy_message(progress),
    }), 409


def _turn_superseded_response(request_id: str, progress: dict | None = None):
    payload = dict(progress or {})
    payload["request_id"] = request_id
    payload["error"] = "superseded"
    payload["message"] = payload.get("message") or "This response was replaced by a newer turn."
    return jsonify(payload), 409


def build_dm_followup_messages(
    npc: dict,
    scene: dict,
    memories: list[str] | None,
    location: dict | None,
    recent_turns,
    dm_context: dict | None,
    ruling: dict,
    party_context: dict | None = None,
    scene_context: dict | None = None,
) -> list[dict]:
    messages = build_npc_llm_messages(
        npc,
        scene,
        memories,
        location,
        recent_turns=recent_turns,
        dm_context=dm_context,
        party_context=party_context,
        scene_context=scene_context,
    )

    verdict = (ruling.get("verdict") or "unclear").strip().upper()
    claim_summary = (ruling.get("claim_summary") or "the party's claim").strip()
    ruling_text = (ruling.get("ruling_text") or "").strip()

    if verdict == "CONFIRMED":
        action_hint = (
            "The DM confirmed the claim. Accept any proof the party already presented "
            "and let them proceed in character."
        )
    elif verdict == "DENIED":
        action_hint = (
            "The DM denied the claim. Refuse the request in character without granting "
            "access or benefit."
        )
        if is_language_comprehension_claim(claim_summary):
            native = resolve_npc_native_language(npc, language_claim=claim_summary)
            if native:
                action_hint += (
                    f" Language barrier Failure is in force: do NOT speak English or Common. "
                    f"Reply ONLY in {native} (short confusion / refusal), never translate."
                )
            else:
                action_hint += (
                    " Language barrier Failure is in force: do NOT speak English or Common. "
                    "Reply only in this NPC's native/fantasy language from its rules."
                )
    else:
        action_hint = "Apply the DM's guidance in character."

    messages.append({
        "role": "user",
        "content": (
            f"[DM RULING APPLIED - {verdict}] {claim_summary}. "
            f"DM said: {ruling_text}\n\n"
            f"{action_hint} Respond now as the NPC to continue the scene from your last exchange. "
            "Keep the reply concise and suitable for spoken dialogue. "
            "Put spoken dialogue in plain text (NPC voice), never inside *asterisks*. "
            "Use *asterisks* only for a short third-person action beat "
            "(*The guard steps aside.*), which the narrator reads aloud. "
            "Include at most one such beat and put it at the very beginning of the "
            "reply — never in the middle or at the end. "
            "Do not mention the DM, pending checks, or game mechanics. "
            "Do not emit [DM CHECK] tags — this claim was already ruled."
        ),
    })

    return messages


def run_dm_ruling_auto_response(
    npc_id: str,
    player_id: str,
    ruling: dict,
    *,
    include_audio: bool = True,
    scene_description: str = "",
    party_context: dict | None = None,
    voice_model: str | None = None,
    tts_length_scale: Any = None,
    tts_options: dict[str, Any] | None = None,
    campaign_id: str = "",
) -> dict:
    verdict = (ruling.get("verdict") or "").strip().lower()
    if verdict not in ("confirmed", "denied"):
        return {
            "generated": False,
            "reason": "no_ruling_verdict",
        }

    request_id = uuid.uuid4().hex
    progress_started = False

    try:
        campaign_id = normalize_campaign_id(
            campaign_id or (party_context or {}).get("campaign_id") or ruling.get("campaign_id")
        )
        register_turn_progress(
            request_id,
            route="/dm-ruling-auto-response",
            npc_id=npc_id,
            player_id=player_id,
            mode="text",
            campaign_id=campaign_id,
            force=True,
        )
        progress_started = True
        set_turn_progress(
            request_id,
            "waiting_llm",
            message="The DM ruled — the NPC is responding to your last exchange...",
        )

        context = load_turn_pre_llm_context(
            npc_id,
            player_id,
            player_text=None,
            party_context=party_context,
            campaign_id=campaign_id,
            save_player_turn=False,
            detect_player_claim=False,
        )
        npc = context["npc"]
        location = context["location"]
        campaign_id = context["campaign_id"]
        memories = context["memories"]
        recent_turns = context["recent_turns"]
        dm_context = context["dm_context"]
        party_context = context.get("party_context") or party_context
        scene_context = context.get("scene_context")

        messages = build_dm_followup_messages(
            npc,
            {"description": scene_description},
            memories,
            location,
            recent_turns,
            dm_context,
            ruling,
            party_context=party_context,
            scene_context=scene_context,
        )

        npc_text_raw = generate_npc_reply_text(
            npc,
            dm_context,
            messages,
            ask_llm,
            player_text=None,
            enforce_required_dm_checks=False,
        )
        ensure_turn_in_progress(request_id)
        set_turn_progress(request_id, "processing_response")
        post_llm_db = finalize_turn_database_phase(
            npc_id,
            player_id,
            "",
            npc_text_raw,
            campaign_id=campaign_id,
            create_pending_checks=False,
            npc=npc,
            fallback_spoken_text="Very well.",
        )
        npc_response = post_llm_db["npc_response"]
        npc_text = npc_response["spoken_text"]

        ensure_turn_in_progress(request_id)

        tts_segments = split_narration_and_dialogue(npc_text)
        tts_text = " ".join(segment["text"] for segment in tts_segments).strip()

        audio_url = None
        audio_filename = None
        audio_clips: list[dict[str, str]] = []
        narration_text = ""
        dialogue_text = ""

        if include_audio:
            ensure_turn_in_progress(request_id)
            set_turn_progress(request_id, "synthesizing_speech")
            request_tts = dict(tts_options or {})
            if tts_length_scale is not None and "tts_length_scale" not in request_tts:
                request_tts["tts_length_scale"] = tts_length_scale
            clip_result = synthesize_segmented_speech_clips(
                npc_text=npc_text,
                npc_voice_model=pick_npc_voice_model(npc, voice_model),
                request_id=request_id,
                npc_tts_options=pick_npc_tts_options(npc, request_tts or None),
            )
            audio_clips = clip_result["clips"]
            narration_text = clip_result["narration_text"]
            dialogue_text = clip_result["dialogue_text"]
            # Legacy single URL points at speech when present, else narration.
            if clip_result.get("speech_path"):
                audio_url = turn_audio_url(request_id, "speech")
                audio_filename = turn_audio_path(request_id, "speech").name
            elif clip_result.get("narration_path"):
                audio_url = turn_audio_url(request_id, "narration")
                audio_filename = turn_audio_path(request_id, "narration").name

        print(
            f"[DM] auto-response request_id={request_id} npc={npc_id} player={player_id} "
            f"verdict={verdict} chars={len(npc_text)} audio={bool(audio_url)} clips={len(audio_clips)}",
            flush=True,
        )

        complete_turn_progress(request_id)
        progress_started = False

        return {
            "generated": True,
            "request_id": request_id,
            "npc_id": npc_id,
            "npc_name": npc.get("name", npc_id),
            "player_id": player_id,
            "ruling_id": ruling.get("id"),
            "verdict": verdict,
            "npc_text": npc_text,
            "narration_text": narration_text,
            "dialogue_text": dialogue_text,
            "tts_text": tts_text,
            "tts_segments": tts_segments,
            "audio_url": audio_url,
            "audio_filename": audio_filename,
            "audio_clips": audio_clips,
        }

    except TurnInProgressError as ex:
        print(
            f"[DM] auto-response blocked request_id={request_id} npc={npc_id} player={player_id} "
            f"active={ex.progress.get('request_id')}",
            flush=True,
        )
        return {
            "generated": False,
            "request_id": request_id,
            "reason": "turn_in_progress",
            "error": "turn_in_progress",
            "active_turn": ex.progress,
        }
    except TurnSupersededError as ex:
        print(
            f"[DM] auto-response superseded request_id={request_id} npc={npc_id} player={player_id}",
            flush=True,
        )
        return {
            "generated": False,
            "request_id": request_id,
            "reason": "superseded",
            "error": "superseded",
            "progress": ex.progress,
        }
    except Exception as ex:
        if progress_started:
            fail_turn_progress(request_id, "generation_failed")
        print(
            f"[DM] auto-response failed request_id={request_id} npc={npc_id} error={ex!r}",
            flush=True,
        )
        return {
            "generated": False,
            "request_id": request_id,
            "reason": "generation_failed",
            "error": "generation_failed",
        }


def ask_llm(messages):
    # Let llm_router pick a higher budget for reasoning models (gpt-oss / qwen3).
    return generate_llm_response(
        messages,
        temperature=0.80,
        max_tokens=None,
        top_p=0.90,
        stream=False,
    )


def save_uploaded_audio(file_storage, request_id: str) -> Path:
    """
    Persist browser uploads with a real extension, then normalize to WAV for Whisper.
    Phone clients usually send webm/opus (or mp4), not WAV.
    """
    original_name = (getattr(file_storage, "filename", None) or "").strip().lower()
    content_type = (getattr(file_storage, "content_type", None) or "").strip().lower()

    if original_name.endswith(".webm") or "webm" in content_type:
        suffix = ".webm"
    elif original_name.endswith(".m4a") or original_name.endswith(".mp4") or "mp4" in content_type:
        suffix = ".m4a"
    elif original_name.endswith(".ogg") or "ogg" in content_type:
        suffix = ".ogg"
    elif original_name.endswith(".wav") or "wav" in content_type:
        suffix = ".wav"
    else:
        suffix = ".webm"

    raw_path = UPLOAD_DIR / f"{request_id}{suffix}"
    wav_path = UPLOAD_DIR / f"{request_id}.wav"
    file_storage.save(raw_path)
    raw_bytes = raw_path.stat().st_size if raw_path.exists() else 0
    print(
        f"[PLAYER-AUDIO] request_id={request_id} saved name={original_name!r} "
        f"content_type={content_type!r} suffix={suffix} bytes={raw_bytes} path={raw_path}",
        flush=True,
    )

    if raw_bytes < 1000:
        print(
            f"[PLAYER-AUDIO] request_id={request_id} ERROR empty_or_short_upload bytes={raw_bytes}",
            flush=True,
        )

    if suffix == ".wav":
        return raw_path

    try:
        convert_wav_to_standard_format(raw_path, wav_path)
        wav_bytes = wav_path.stat().st_size if wav_path.exists() else 0
        print(
            f"[PLAYER-AUDIO] request_id={request_id} convert_ok out_bytes={wav_bytes} path={wav_path}",
            flush=True,
        )
    except Exception as ex:
        print(
            f"[PLAYER-AUDIO] request_id={request_id} ERROR convert_failed "
            f"suffix={suffix} in_bytes={raw_bytes} error={ex!r}",
            flush=True,
        )
        raise
    return wav_path


def _wav_duration_seconds(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return 0.0


def _pad_wav_for_short_speech(audio_path: Path, *, pad_seconds: float = 0.5) -> Path:
    """
    Pad short clips with silence so Whisper is more likely to catch one-word replies.
    Returns a new path when padding is applied; otherwise the original path.
    """
    duration = _wav_duration_seconds(audio_path)
    if duration <= 0 or duration >= 2.5:
        return audio_path

    padded_path = audio_path.with_name(f"{audio_path.stem}_padded{audio_path.suffix}")
    try:
        ffmpeg_exe = find_ffmpeg_exe()
        # adelay needs ms; pad both ends for short yes/no/ok utterances.
        pad_ms = int(pad_seconds * 1000)
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-i", str(audio_path),
                "-af", f"adelay={pad_ms}|{pad_ms},apad=pad_dur={pad_seconds}",
                "-ac", "1",
                "-ar", "16000",
                "-sample_fmt", "s16",
                "-c:a", "pcm_s16le",
                "-f", "wav",
                str(padded_path),
            ],
            check=True,
        )
        print(
            f"[PLAYER-AUDIO] padded short clip duration={duration:.2f}s "
            f"pad={pad_seconds}s path={padded_path}",
            flush=True,
        )
        return padded_path
    except Exception as ex:
        print(f"[PLAYER-AUDIO] pad_failed path={audio_path} error={ex!r}", flush=True)
        return audio_path


def _wav_peak_stats(audio_path: Path) -> str:
    """Return a short duration/amplitude summary for STT diagnostics."""
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate() or 1
            channels = wav_file.getnchannels() or 1
            sample_width = wav_file.getsampwidth()
            duration = frames / float(rate)
            raw = wav_file.readframes(frames)
        if sample_width != 2 or not raw:
            return f"duration={duration:.2f}s frames={frames} width={sample_width}"
        samples = array.array("h")
        samples.frombytes(raw)
        if channels > 1:
            samples = samples[::channels]
        if not samples:
            return f"duration={duration:.2f}s empty"
        peak = max(abs(s) for s in samples)
        # Sparse RMS over up to 20k samples for speed.
        step = max(1, len(samples) // 20000)
        subset = samples[::step]
        rms = (sum(s * s for s in subset) / max(1, len(subset))) ** 0.5
        return f"duration={duration:.2f}s peak={peak} rms={rms:.1f} rate={rate}"
    except Exception as ex:
        return f"stats_unavailable error={ex!r}"


def _wav_peak_value(audio_path: Path) -> int | None:
    """Return absolute PCM16 peak, or None when unavailable."""
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            channels = wav_file.getnchannels() or 1
            sample_width = wav_file.getsampwidth()
            raw = wav_file.readframes(frames)
        if sample_width != 2 or not raw:
            return None
        samples = array.array("h")
        samples.frombytes(raw)
        if channels > 1:
            samples = samples[::channels]
        if not samples:
            return 0
        return max(abs(s) for s in samples)
    except Exception:
        return None


# Whisper often invents these on near-silence / truncated clips.
# Do not list allowlisted replies here (ok/okay/yes/…) — allowlist wins first.
_SHORT_CLIP_HALLUCINATIONS = frozenset(
    {
        "you",
        "yeah",
        "thank you",
        "thanks",
        "thanks for watching",
        "thank you for watching",
        "subscribe",
        "bye",
        "um",
        "uh",
        "the",
        "a",
        "i",
        "oh",
        "hmm",
        "mb",
        "www",
    }
)

# One-word replies we still accept on short clips.
_SHORT_CLIP_ALLOWLIST = frozenset(
    {
        "yes",
        "no",
        "ok",
        "okay",
        "go",
        "hi",
        "hey",
        "stop",
        "help",
        "wait",
        "look",
        "run",
        "attack",
        "cast",
        "open",
        "close",
        "take",
        "give",
        "talk",
        "leave",
        "stay",
        "hide",
    }
)


def _is_stt_hallucination(
    text: str,
    *,
    duration_seconds: float | None = None,
    peak: int | None = None,
) -> bool:
    """
    Detect common Whisper garbage (especially when silence/truncation biases output).
    Examples: 'No. No. No. No.', 'Thank you. Thank you.', or bare 'You' on a 1s clip.
    Allowlisted short replies (ok/okay/yes/…) are kept even on short/weak clips.
    """
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").strip(), flags=re.IGNORECASE)
    words = [w.lower() for w in cleaned.split() if w]
    if not words:
        return True

    normalized = " ".join(words)
    unique = set(words)
    if len(words) >= 3 and len(unique) == 1:
        return True
    if len(words) >= 4 and len(unique) <= 2 and max(words.count(w) for w in unique) >= 3:
        return True

    # Allowlist before hallucination rejection so short ok/okay are not dropped.
    if normalized in _SHORT_CLIP_ALLOWLIST or (
        len(words) == 1 and words[0] in _SHORT_CLIP_ALLOWLIST
    ):
        return False

    short = duration_seconds is not None and duration_seconds < 1.6
    weak = peak is not None and peak < 2500
    if (short or weak) and normalized in _SHORT_CLIP_HALLUCINATIONS:
        return True
    if short and weak and len(words) == 1:
        return True
    return False


def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribe player/DM mic audio, optimized for short replies (yes/ok/go).
    Skips VAD on short clips, pads brief audio, and retries with looser thresholds.
    """
    raw_duration = _wav_duration_seconds(audio_path)
    raw_peak = _wav_peak_value(audio_path)
    print(
        f"[PLAYER-AUDIO] stt_raw duration={raw_duration:.2f}s peak={raw_peak} path={audio_path}",
        flush=True,
    )
    stt_path = _pad_wav_for_short_speech(audio_path)
    stats = _wav_peak_stats(stt_path)
    duration = _wav_duration_seconds(stt_path)
    peak = _wav_peak_value(stt_path)
    print(f"[PLAYER-AUDIO] stt_input path={stt_path} {stats}", flush=True)

    def _run(*, vad_filter: bool, beam_size: int, no_speech_threshold: float) -> str:
        segments, info = STT_MODEL.transcribe(
            str(stt_path),
            language="en",
            beam_size=beam_size,
            best_of=max(1, beam_size),
            vad_filter=vad_filter,
            condition_on_previous_text=False,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=2.4,
        )
        text_parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                text_parts.append(text)
        info_duration = getattr(info, "duration", None)
        joined = " ".join(text_parts).strip()
        print(
            f"[PLAYER-AUDIO] stt_pass vad={vad_filter} beam={beam_size} "
            f"no_speech={no_speech_threshold} duration={info_duration} "
            f"parts={len(text_parts)} text={joined[:120]!r}",
            flush=True,
        )
        return joined

    # Short clips: skip VAD first — Silero often eats one-word replies.
    prefer_no_vad = (raw_duration or duration) < 3.0
    attempts = []
    if prefer_no_vad:
        attempts.append(dict(vad_filter=False, beam_size=5, no_speech_threshold=0.35))
        attempts.append(dict(vad_filter=False, beam_size=5, no_speech_threshold=0.2))
        attempts.append(dict(vad_filter=True, beam_size=5, no_speech_threshold=0.35))
    else:
        attempts.append(dict(vad_filter=True, beam_size=1, no_speech_threshold=0.6))
        attempts.append(dict(vad_filter=False, beam_size=5, no_speech_threshold=0.35))
        attempts.append(dict(vad_filter=False, beam_size=5, no_speech_threshold=0.2))

    # Judge hallucinations against the unpadded clip — padding can hide how short it was.
    judge_duration = raw_duration if raw_duration > 0 else duration
    judge_peak = raw_peak if raw_peak is not None else peak

    text = ""
    for attempt in attempts:
        text = _run(**attempt)
        if text and not _is_stt_hallucination(
            text,
            duration_seconds=judge_duration,
            peak=judge_peak,
        ):
            break
        if text and _is_stt_hallucination(
            text,
            duration_seconds=judge_duration,
            peak=judge_peak,
        ):
            print(
                f"[PLAYER-AUDIO] stt_hallucination rejected text={text[:120]!r} "
                f"raw_duration={judge_duration:.2f}s peak={judge_peak}",
                flush=True,
            )
            text = ""

    return text


def split_into_ordered_tts_segments(
    text: str,
    pronunciation: dict[str, str] | None = None,
    *,
    stream_sentences: bool = True,
) -> list[dict]:
    """
    Split raw model output into ordered TTS segments, preserving document order.

    Asterisk-wrapped text is narration and becomes a narrator clip whatever its
    length; plain text is NPC dialogue. Bare DM-wait copy (no asterisks) is also
    narrator so Piper never voices the hold line as the NPC.

    Each run is then cut on sentence ends. A whole paragraph of dialogue is one
    Piper call otherwise, so the player waits for the last word to be
    synthesized before hearing the first; clips are published as they finish.
    """
    raw_text, _ = strip_dm_check_tag(text or "")

    # Whole-turn wait lines often lose asterisks in transit; never NPC-voice them.
    if is_dm_wait_narration(raw_text):
        cleaned = clean_for_tts(
            raw_text.strip().strip("*").strip(),
            pronunciation=pronunciation,
        )
        if cleaned:
            return [{"type": "narrator", "text": cleaned}]
        return []

    segments: list[dict] = []
    pattern = re.compile(r"\*([^*]{1,800})\*")
    last_end = 0

    def _append(segment_type: str, body: str) -> None:
        cleaned = clean_for_tts(body, pronunciation=pronunciation)
        if not cleaned:
            return
        if not stream_sentences:
            segments.append({"type": segment_type, "text": cleaned})
            return
        for chunk in split_text_for_streaming(cleaned):
            segments.append({"type": segment_type, "text": chunk})

    for match in pattern.finditer(raw_text):
        before = raw_text[last_end:match.start()].strip()
        if before:
            if is_dm_wait_narration(before):
                _append("narrator", before)
            else:
                _append("npc", before)

        stage = match.group(1).strip()
        if stage:
            _append(asterisk_body_tts_type(stage), stage)

        last_end = match.end()

    after = raw_text[last_end:].strip()
    if after:
        if is_dm_wait_narration(after):
            _append("narrator", after)
        else:
            _append("npc", after)

    if not segments:
        fallback = clean_for_tts(raw_text, pronunciation=pronunciation)
        if fallback:
            segments.append({"type": "npc", "text": fallback})

    return segments


def split_narration_and_dialogue(text: str) -> list[dict]:
    """
    Compatibility wrapper: ordered segments coalesced into at most one narrator
    and one NPC blob. Prefer split_into_ordered_tts_segments for streaming clips.
    """
    ordered = split_into_ordered_tts_segments(text)
    stage_directions = [s["text"] for s in ordered if s["type"] == "narrator"]
    dialogue_parts = [s["text"] for s in ordered if s["type"] == "npc"]

    segments = []
    if stage_directions:
        segments.append({
            "type": "narrator",
            "text": " ".join(stage_directions).strip()
        })
    if dialogue_parts:
        segments.append({
            "type": "npc",
            "text": " ".join(dialogue_parts).strip()
        })
    return segments

def find_ffmpeg_exe() -> str:
    """
    Locate ffmpeg.exe for standardizing generated WAV segments.
    """
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path:
        return ffmpeg_path

    common_paths = [
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
    ]

    for path in common_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "ffmpeg.exe was not found. Install FFmpeg and make sure ffmpeg is in PATH. "
        "For example: winget install Gyan.FFmpeg"
    )


def convert_wav_to_standard_format(
    input_path: Path,
    output_path: Path,
    *,
    pitch_semitones: float = 0.0,
    fx_preset: str = DEFAULT_FX_PRESET,
    trim_silence: bool = True,
    denoise_nr: float | None = None,
    denoise_nf: float | None = None,
) -> None:
    """
    Convert a WAV file to standard Piper-friendly PCM with clarity post-processing.

    Output is 22.05 kHz mono 16-bit PCM WAV. Applies loudness/EQ/dynamics,
    optional pitch shift, silence trim, and character FX presets.
    """
    ffmpeg_exe = find_ffmpeg_exe()
    af = build_af_filter(
        pitch_semitones=pitch_semitones,
        fx_preset=fx_preset,
        trim_silence=trim_silence,
        sample_rate=STANDARD_WAV_SAMPLE_RATE,
        denoise_nr=DEFAULT_DENOISE_NR if denoise_nr is None else denoise_nr,
        denoise_nf=DEFAULT_DENOISE_NF if denoise_nf is None else denoise_nf,
    )

    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-vn",
            "-af", af,
            "-ac", str(STANDARD_WAV_CHANNELS),
            "-ar", str(STANDARD_WAV_SAMPLE_RATE),
            "-sample_fmt", STANDARD_WAV_SAMPLE_FORMAT,
            "-c:a", STANDARD_WAV_CODEC,
            "-f", "wav",
            str(output_path)
        ],
        check=True
    )


def get_wav_format_description(path: Path) -> str:
    with wave.open(str(path), "rb") as wav_file:
        params = wav_file.getparams()
        return (
            f"channels={params.nchannels}, "
            f"sample_width={params.sampwidth}, "
            f"sample_rate={params.framerate}, "
            f"compression={params.comptype}"
        )


def create_silence_wav(output_path: Path, duration_seconds: float = SEGMENT_PAUSE_SECONDS) -> None:
    """
    Create a short silence WAV using the same standard PCM format as TTS segments.
    """
    ffmpeg_exe = find_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"anullsrc=r={STANDARD_WAV_SAMPLE_RATE}:cl=mono",
            "-t", str(duration_seconds),
            "-ac", str(STANDARD_WAV_CHANNELS),
            "-ar", str(STANDARD_WAV_SAMPLE_RATE),
            "-sample_fmt", STANDARD_WAV_SAMPLE_FORMAT,
            "-c:a", STANDARD_WAV_CODEC,
            "-f", "wav",
            str(output_path)
        ],
        check=True
    )


def concatenate_wav_files(input_paths: list[Path], output_path: Path, pause_seconds: float = SEGMENT_PAUSE_SECONDS) -> None:
    """
    Concatenate standardized WAV segments into one WAV file using FFmpeg.

    This avoids manual byte-level WAV assembly, which can produce files that
    technically have matching headers but still play as static on some devices.
    """
    if not input_paths:
        raise ValueError("No WAV files to concatenate.")

    if len(input_paths) == 1:
        convert_wav_to_standard_format(input_paths[0], output_path)
        return

    ffmpeg_exe = find_ffmpeg_exe()
    concat_list_path = output_path.with_name(f"{output_path.stem}_concat_list.txt")
    silence_path = output_path.with_name(f"{output_path.stem}_silence.wav")

    try:
        create_silence_wav(silence_path, pause_seconds)

        ordered_paths = []
        for index, input_path in enumerate(input_paths):
            if index > 0:
                ordered_paths.append(silence_path)
            ordered_paths.append(input_path)

        # FFmpeg concat demuxer list. Paths are single-quoted and backslashes escaped.
        lines = []
        for path in ordered_paths:
            escaped = str(path).replace("\\", "/").replace("'", "'\\''")
            lines.append(f"file '{escaped}'")

        concat_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-ac", str(STANDARD_WAV_CHANNELS),
                "-ar", str(STANDARD_WAV_SAMPLE_RATE),
                "-sample_fmt", STANDARD_WAV_SAMPLE_FORMAT,
                "-c:a", STANDARD_WAV_CODEC,
                "-f", "wav",
                str(output_path)
            ],
            check=True
        )

    except subprocess.CalledProcessError as ex:
        details = []
        for input_path in input_paths:
            try:
                details.append(f"{input_path}: {get_wav_format_description(input_path)}")
            except Exception as format_ex:
                details.append(f"{input_path}: could not inspect format: {format_ex}")

        raise RuntimeError(
            "FFmpeg failed while concatenating standardized WAV files. "
            + " | ".join(details)
        ) from ex

    finally:
        try:
            concat_list_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            silence_path.unlink(missing_ok=True)
        except Exception:
            pass


def turn_audio_path(request_id: str, clip: str | int) -> Path:
    """Stable on-disk path for a per-turn segment WAV (indexed) or legacy narration/speech."""
    safe_id = normalize_client_request_id(request_id) or re.sub(r"[^a-f0-9]", "", str(request_id).lower())
    clip_key = str(clip).strip().lower()
    if clip_key.isdigit():
        return OUTPUT_DIR / f"{safe_id}_seg_{int(clip_key):02d}.wav"
    if clip_key.startswith("seg_"):
        return OUTPUT_DIR / f"{safe_id}_{clip_key}.wav"
    if clip_key in ("narration", "speech"):
        return OUTPUT_DIR / f"{safe_id}_{clip_key}.wav"
    raise ValueError(f"Unsupported turn-audio clip key: {clip!r}")


def turn_audio_url(request_id: str, clip: str | int) -> str:
    safe_id = normalize_client_request_id(request_id) or str(request_id)
    clip_key = str(clip).strip().lower()
    if clip_key.isdigit():
        return f"/turn-audio/{safe_id}/{int(clip_key)}"
    return f"/turn-audio/{safe_id}/{clip_key}"


def synthesize_segmented_speech_clips(
    npc_text: str,
    npc_voice_model: str | None,
    request_id: str,
    npc_length_scale: float = DEFAULT_LENGTH_SCALE,
    npc_tts_options: dict[str, Any] | None = None,
    on_clip_ready: Any = None,
    on_narration_ready: Any = None,
    *,
    return_after_first_clip: bool = False,
    on_stream_complete: Any = None,
) -> dict[str, Any]:
    """
    Synthesize ordered narrator/NPC segments as separate WAV files (no concatenation).

    Each segment is written and published via on_clip_ready as soon as Piper finishes,
    so the editor can start SignalR playback while later segments are still generating.

    When ``return_after_first_clip`` is true and more segments remain, returns after the
    first clip with ``pending=True`` and continues remaining segments on a background
    thread (progress + ``on_stream_complete``).
    """
    npc_opts = npc_tts_options or resolve_tts_options(
        {},
        length_scale=npc_length_scale,
    )
    pronunciation = npc_opts.get("pronunciation") if isinstance(npc_opts.get("pronunciation"), dict) else None

    segments = split_into_ordered_tts_segments(npc_text, pronunciation=pronunciation)
    if not segments:
        segments = [{"type": "npc", "text": "I have nothing more to say."}]

    narration_text = " ".join(s["text"] for s in segments if s["type"] == "narrator").strip()
    dialogue_text = " ".join(s["text"] for s in segments if s["type"] == "npc").strip()

    narrator_tts = get_narrator_tts_settings()
    narrator_opts = resolve_tts_options(
        {
            "voice_model": narrator_tts.get("resolved_voice_model"),
            "tts_length_scale": narrator_tts.get("length_scale"),
            "tts_noise_scale": narrator_tts.get("noise_scale"),
            "tts_noise_w": narrator_tts.get("noise_w"),
            "tts_sentence_silence": narrator_tts.get("sentence_silence"),
            "tts_speaker_id": narrator_tts.get("speaker_id"),
            "tts_pitch_semitones": narrator_tts.get("pitch_semitones"),
            "tts_fx": narrator_tts.get("fx_preset"),
            "tts_denoise_nr": narrator_tts.get("denoise_nr"),
            "tts_denoise_nf": narrator_tts.get("denoise_nf"),
        },
        length_scale=narrator_tts.get("length_scale"),
        noise_scale=narrator_tts.get("noise_scale"),
        noise_w=narrator_tts.get("noise_w"),
        sentence_silence=narrator_tts.get("sentence_silence"),
        speaker_id=narrator_tts.get("speaker_id"),
        pitch_semitones=narrator_tts.get("pitch_semitones"),
        fx_preset=narrator_tts.get("fx_preset") or "clarity",
        denoise_nr=narrator_tts.get("denoise_nr"),
        denoise_nf=narrator_tts.get("denoise_nf"),
    )

    state: dict[str, Any] = {
        "narration_path": None,
        "speech_path": None,
        "clips": [],
        # Every clip in order. The legacy narration/speech pair keeps only the
        # first narration and the last speech, which was the whole turn back
        # when dialogue was one segment and is a fraction of it now.
        "clip_paths": [],
        "lock": threading.Lock(),
    }

    def _synthesize_one(index: int, segment: dict[str, Any]) -> dict[str, Any] | None:
        if not turn_is_in_progress(request_id):
            print(
                f"[PLAYER-AUDIO] clip_skipped_superseded request_id={request_id} index={index}",
                flush=True,
            )
            return None

        raw_path = OUTPUT_DIR / f"{request_id}_raw_{index:02d}_{segment['type']}.wav"
        out_path = turn_audio_path(request_id, index)
        clip_type = "narration" if segment["type"] == "narrator" else "speech"
        opts = narrator_opts if segment["type"] == "narrator" else npc_opts
        voice = (
            narrator_tts["resolved_voice_model"]
            if segment["type"] == "narrator"
            else npc_voice_model
        )

        try:
            synthesize_speech(
                segment["text"],
                voice,
                raw_path,
                length_scale=opts["length_scale"],
                noise_scale=opts["noise_scale"],
                noise_w=opts["noise_w"],
                sentence_silence=opts["sentence_silence"],
                speaker_id=opts.get("speaker_id"),
            )

            convert_wav_to_standard_format(
                raw_path,
                out_path,
                pitch_semitones=float(opts.get("pitch_semitones") or 0.0),
                fx_preset=str(opts.get("fx_preset") or DEFAULT_FX_PRESET),
                denoise_nr=opts.get("denoise_nr"),
                denoise_nf=opts.get("denoise_nf"),
            )
        finally:
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass

        with state["lock"]:
            if clip_type == "narration" and state["narration_path"] is None:
                legacy = turn_audio_path(request_id, "narration")
                legacy.write_bytes(out_path.read_bytes())
                state["narration_path"] = legacy
            if clip_type == "speech":
                legacy = turn_audio_path(request_id, "speech")
                legacy.write_bytes(out_path.read_bytes())
                state["speech_path"] = legacy

            clip = {
                "type": clip_type,
                "index": index,
                "url": turn_audio_url(request_id, index),
                "text": segment["text"],
            }
            state["clips"].append(clip)
            state["clip_paths"].append((index, out_path))
            clips_snapshot = list(state["clips"])

        if index == 0:
            note_first_audio_ms()

        print(
            f"[PLAYER-AUDIO] clip_ready request_id={request_id} index={index} "
            f"type={clip_type} chars={len(segment['text'])} path={out_path}",
            flush=True,
        )

        if callable(on_clip_ready):
            on_clip_ready(clip, clips_snapshot)
        if clip_type == "narration" and callable(on_narration_ready):
            on_narration_ready(out_path, segment["text"], clips_snapshot)
        return clip

    def _result(*, pending: bool) -> dict[str, Any]:
        with state["lock"]:
            clips = list(state["clips"])
            narration_path = state["narration_path"]
            speech_path = state["speech_path"]
            # Clips finish on their own thread, so order by index, not arrival.
            clip_paths = [path for _, path in sorted(state["clip_paths"], key=lambda item: item[0])]
        return {
            "segments": segments,
            "narration_text": narration_text,
            "dialogue_text": dialogue_text,
            "narration_path": narration_path,
            "speech_path": speech_path,
            "clip_paths": clip_paths,
            "clips": clips,
            "pending": pending,
        }

    # Always synthesize the first clip on the request thread.
    if _synthesize_one(0, segments[0]) is None:
        # Empty clips without raising used to make /npc-voice-turn send_file a
        # missing {request_id}_response.wav (500). Surface supersede instead.
        progress = get_turn_progress(request_id)
        raise TurnSupersededError(progress)

    if return_after_first_clip and len(segments) > 1:
        remaining = list(enumerate(segments[1:], start=1))

        def _continue_remaining() -> None:
            error: BaseException | None = None
            try:
                for index, segment in remaining:
                    if _synthesize_one(index, segment) is None:
                        break
            except BaseException as ex:
                error = ex
                print(
                    f"[PLAYER-AUDIO] stream_tts_failed request_id={request_id} error={ex!r}",
                    flush=True,
                )
            finally:
                if callable(on_stream_complete):
                    try:
                        on_stream_complete(_result(pending=False), error)
                    except Exception as cb_ex:
                        print(
                            f"[PLAYER-AUDIO] on_stream_complete failed "
                            f"request_id={request_id} error={cb_ex!r}",
                            flush=True,
                        )

        threading.Thread(
            target=_continue_remaining,
            name=f"tts-stream-{request_id[:8]}",
            daemon=True,
        ).start()
        print(
            f"[PLAYER-AUDIO] stream_early_return request_id={request_id} "
            f"first_of={len(segments)}",
            flush=True,
        )
        return _result(pending=True)

    for index, segment in enumerate(segments[1:], start=1):
        if _synthesize_one(index, segment) is None:
            break

    return _result(pending=False)


def synthesize_segmented_speech(
    npc_text: str,
    npc_voice_model: str | None,
    output_path: Path,
    npc_length_scale: float = DEFAULT_LENGTH_SCALE,
    npc_tts_options: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Legacy helper: synthesize separate clips then concatenate into output_path.
    Prefer synthesize_segmented_speech_clips for early narration playback.
    """
    request_id = output_path.stem.replace("_response", "").replace("_dm_followup", "")
    if not normalize_client_request_id(request_id):
        request_id = uuid.uuid4().hex

    result = synthesize_segmented_speech_clips(
        npc_text=npc_text,
        npc_voice_model=npc_voice_model,
        request_id=request_id,
        npc_length_scale=npc_length_scale,
        npc_tts_options=npc_tts_options,
    )

    paths = [p for p in (result.get("clip_paths") or []) if p and p.exists()]
    if not paths:
        paths = [p for p in (result.get("narration_path"), result.get("speech_path")) if p and p.exists()]
    if not paths:
        raise RuntimeError("No TTS segments were produced.")
    if len(paths) == 1:
        output_path.write_bytes(paths[0].read_bytes())
    else:
        concatenate_wav_files(paths, output_path)

    return result["segments"]


def pick_npc_voice_model(npc: dict, override: str | None = None) -> str | None:
    """Prefer request/editor override, then NPC YAML voice_model."""
    raw = (override or "").strip()
    if raw:
        return raw
    value = npc.get("voice_model")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_voice_model_paths_for_turn(
    npc_id: str,
    voice_override: str | None = None,
) -> list[Path]:
    """Resolved ONNX paths for NPC + narrator voices on a voice turn."""
    paths: list[Path] = []
    seen: set[str] = set()

    def _add(voice_model: str | Path | None) -> None:
        if not voice_model or not str(voice_model).strip():
            return
        resolved = resolve_voice_model_path(voice_model)
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        paths.append(resolved)

    try:
        npc = load_npc(npc_id)
        _add(pick_npc_voice_model(npc, voice_override) or DEFAULT_VOICE_MODEL)
    except Exception as ex:
        print(
            f"[PIPER-POOL] npc_voice_unresolved npc_id={npc_id} error={ex!r}",
            flush=True,
        )

    try:
        narrator = get_narrator_tts_settings()
        _add(narrator.get("resolved_voice_model") or narrator.get("voice_model"))
    except Exception as ex:
        print(f"[PIPER-POOL] narrator_voice_unresolved error={ex!r}", flush=True)

    return paths


_PIPER_STARTUP_WARM_LOCK = threading.Lock()
_PIPER_STARTUP_WARM_COMPLETE = False


def piper_startup_voice_directory() -> Path:
    """Folder whose ONNX models are all loaded before WSGI starts listening."""
    configured = str(os.environ.get("PIPER_WARM_VOICE_DIR", "")).strip()
    if configured:
        return Path(configured)
    return VOICES_DIR


def warm_piper_at_startup() -> None:
    """
    Load the Piper models the server will almost certainly need before any turn
    arrives.

    Warming during a turn is self-defeating: the replica loaders saturate the CPU
    that speech-to-text is running on, so the turn waits out the whole fan-out to
    save a fraction of that on synthesis. Paying the cost at startup keeps it off
    the first player's turn.
    """
    global _PIPER_STARTUP_WARM_COMPLETE

    if str(os.environ.get("PIPER_USE_SUBPROCESS", "")).strip().lower() in ("1", "true", "yes"):
        print("[PIPER-POOL] startup warm skipped: subprocess mode enabled", flush=True)
        return
    if str(os.environ.get("PIPER_WARM_AT_STARTUP", "1")).strip().lower() in ("0", "false", "no"):
        print("[PIPER-POOL] startup warm disabled by PIPER_WARM_AT_STARTUP", flush=True)
        return

    with _PIPER_STARTUP_WARM_LOCK:
        if _PIPER_STARTUP_WARM_COMPLETE:
            return

        paths: list[Path] = []
        seen: set[str] = set()

        def _add_resolved(resolved: Path) -> None:
            key = os.path.normcase(os.path.normpath(str(resolved)))
            if key in seen:
                return
            seen.add(key)
            paths.append(resolved)

        def _add(voice_model: Any) -> None:
            if not voice_model:
                return
            try:
                resolved = resolve_voice_model_path(voice_model)
            except Exception as ex:
                print(
                    f"[PIPER-POOL] startup_voice_unresolved "
                    f"voice={str(voice_model)!r} error={ex!r}",
                    flush=True,
                )
                return
            _add_resolved(resolved)

        # The configured voice folder is the authoritative startup inventory. These
        # paths already came from our configured VOICES_DIR, so add them directly
        # instead of feeding absolute paths through the request-path resolver.
        warm_voice_dir = piper_startup_voice_directory()
        if warm_voice_dir.is_dir():
            discovered = sorted(
                warm_voice_dir.rglob("*.onnx"),
                key=lambda path: str(path).lower(),
            )
            for model_path in discovered:
                _add_resolved(model_path)
            print(
                f"[PIPER-POOL] startup voice scan dir={warm_voice_dir} "
                f"models={len(discovered)}",
                flush=True,
            )
        else:
            print(
                f"[PIPER-POOL] startup voice scan directory not found: {warm_voice_dir}",
                flush=True,
            )

        # NPCs without an explicit voice fall back to this model at synthesis time.
        _add(DEFAULT_VOICE_MODEL)

        try:
            narrator = get_narrator_tts_settings()
            _add(narrator.get("resolved_voice_model") or narrator.get("voice_model"))
        except Exception as ex:
            print(f"[PIPER-POOL] startup_narrator_unresolved error={ex!r}", flush=True)

        # Any voice already assigned to an NPC is a voice a turn can ask for. There
        # is no request context at startup, so this reads the on-disk NPC directory
        # rather than the tenant-scoped listing.
        try:
            npc_files = sorted(list(NPC_DIR.glob("*.yaml")) + list(NPC_DIR.glob("*.yml")))
            for path in npc_files:
                try:
                    _add(pick_npc_voice_model(load_npc(path.stem), None))
                except Exception as ex:
                    print(
                        f"[PIPER-POOL] startup_npc_voice_unresolved "
                        f"npc={path.stem} error={ex!r}",
                        flush=True,
                    )
        except Exception as ex:
            print(f"[PIPER-POOL] startup_npc_scan_failed error={ex!r}", flush=True)

        if not paths:
            raise RuntimeError(
                "Piper startup warming is enabled, but no usable voice models were found."
            )

        try:
            timeout_seconds = max(
                1.0,
                float(os.environ.get("PIPER_WARM_TIMEOUT_SECONDS", "300")),
            )
        except ValueError:
            timeout_seconds = 300.0

        started = time.perf_counter()
        print(
            f"[PIPER-POOL] startup blocking warm begin "
            f"models={', '.join(p.name for p in paths)} "
            f"slot_budget={piper_pool.pool_slot_budget()} "
            f"timeout={timeout_seconds:.1f}s",
            flush=True,
        )
        try:
            status = piper_pool.warm_voice_models(
                paths,
                request_id="startup",
                wait_for_ready=True,
                timeout_seconds=timeout_seconds,
            ) or {}
        except Exception as ex:
            print(f"[PIPER-POOL] startup_warm_failed error={ex!r}", flush=True)
            raise

        ready_count = sum(int(item.get("ready") or 0) for item in status.values())
        target_count = sum(int(item.get("target") or 0) for item in status.values())
        _PIPER_STARTUP_WARM_COMPLETE = True
        print(
            f"[PIPER-POOL] startup blocking warm complete "
            f"replicas={ready_count}/{target_count} "
            f"models={len(status)} elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )


def schedule_piper_warm_for_voice_turn(
    npc_id: str,
    voice_override: str | None,
    *,
    request_id: str,
) -> None:
    """Load Piper ONNX models in the background while STT/LLM run."""
    if str(os.environ.get("PIPER_USE_SUBPROCESS", "")).strip().lower() in ("1", "true", "yes"):
        return
    try:
        model_paths = resolve_voice_model_paths_for_turn(npc_id, voice_override)
        if not model_paths:
            return
        piper_pool.warm_voice_models(model_paths, request_id=request_id)
        names = ", ".join(p.name for p in model_paths)
        print(
            f"[PIPER-POOL] scheduled warm request_id={request_id} models={names}",
            flush=True,
        )
    except Exception as ex:
        print(
            f"[PIPER-POOL] schedule_skipped request_id={request_id} error={ex!r}",
            flush=True,
        )


def pick_npc_length_scale(npc: dict, override: Any = None) -> float:
    opts = pick_npc_tts_options(npc, {"tts_length_scale": override} if override is not None else None)
    return float(opts["length_scale"])


def pick_npc_tts_options(npc: dict, request_values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve full Piper + post-process options for an NPC turn."""
    return options_from_request_and_npc(request_values, npc)


def _tts_request_values_from_mapping(mapping: Any) -> dict[str, Any]:
    """Pull TTS override fields from a JSON body or form-like mapping."""
    if mapping is None:
        return {}
    getters = {
        "tts_length_scale": ("tts_length_scale", "length_scale"),
        "tts_noise_scale": ("tts_noise_scale", "noise_scale"),
        "tts_noise_w": ("tts_noise_w", "noise_w"),
        "tts_sentence_silence": ("tts_sentence_silence", "sentence_silence"),
        "tts_speaker_id": ("tts_speaker_id", "speaker_id"),
        "tts_pitch_semitones": ("tts_pitch_semitones", "pitch_semitones"),
        "tts_fx": ("tts_fx", "fx_preset"),
        "tts_denoise_nr": ("tts_denoise_nr", "denoise_nr"),
        "tts_denoise_nf": ("tts_denoise_nf", "denoise_nf"),
    }
    out: dict[str, Any] = {}
    get = mapping.get if hasattr(mapping, "get") else None
    if get is None:
        return out
    for dest, keys in getters.items():
        for key in keys:
            value = get(key)
            if value is not None and str(value).strip() != "":
                out[dest] = value
                break
    pronunciation = get("tts_pronunciation") or get("pronunciation")
    if pronunciation is not None:
        out["tts_pronunciation"] = pronunciation
    return out


def synthesize_speech(
    text: str,
    voice_model: str | None,
    output_path: Path,
    length_scale: float = DEFAULT_LENGTH_SCALE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    noise_w: float = DEFAULT_NOISE_W,
    sentence_silence: float = DEFAULT_SENTENCE_SILENCE,
    speaker_id: int | None = None,
) -> None:
    # Same resolution as /voices/preview so UNC/partial/editor paths work at runtime.
    model_path = resolve_voice_model_path(voice_model)

    if not model_path.exists():
        raise FileNotFoundError(f"Piper voice model not found: {model_path}")

    use_subprocess = str(os.environ.get("PIPER_USE_SUBPROCESS", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not use_subprocess:
        piper_pool.synthesize_wav_file(
            model_path,
            output_path,
            text,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
            sentence_silence=sentence_silence,
            speaker_id=speaker_id,
        )
        return

    piper_exe = resolve_piper_exe()
    cmd = [
        str(piper_exe),
        "--model", str(model_path),
        "--length_scale", str(length_scale),
        "--noise_scale", str(noise_scale),
        "--noise_w", str(noise_w),
        "--sentence_silence", str(sentence_silence),
        "--output_file", str(output_path),
    ]
    if speaker_id is not None:
        cmd.extend(["--speaker", str(int(speaker_id))])

    subprocess.run(
        cmd,
        input=text,
        text=True,
        encoding="utf-8",
        check=True,
    )


def extract_memories_from_exchange(
    npc: dict,
    npc_id: str,
    player_id: str,
    player_text: str,
    npc_text: str
) -> list[str]:
    prompt = f"""
You are maintaining long-term memory for a tabletop RPG NPC.

NPC name: {npc.get("name", npc_id)}
NPC role: {npc.get("role", "")}

Extract only durable facts this NPC should remember for future sessions.

Good memories:
- promises, deals, debts, threats, favors, betrayals
- items bought, sold, given, stolen, lost, or repaired
- important facts learned about the party
- names, relationships, loyalties, suspicions
- unresolved tasks or quests
- emotional changes in the relationship between this NPC and the party

Bad memories:
- temporary greetings
- filler dialogue
- stage directions
- generic questions
- obvious facts already known from the NPC YAML
- secrets the NPC did not actually learn
- anything uncertain unless phrased as uncertainty

Return JSON only.
Return an empty JSON array if nothing should be remembered.

Format:
[
  "The party bought a silvered sword from Torin Ashhand.",
  "The party promised Captain Varra they would investigate the old aqueduct."
]

Player said:
{player_text}

NPC replied:
{npc_text}
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You extract concise long-term RPG NPC memories. Return valid JSON only."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        raw = ask_llm(messages).strip()

        # Some models wrap JSON in markdown fences. Strip those if present.
        raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)

        if not isinstance(parsed, list):
            return []

        memories = []

        for item in parsed:
            if not isinstance(item, str):
                continue

            memory = item.strip()

            if len(memory) < 10:
                continue

            if len(memory) > 300:
                memory = memory[:300].strip()

            memories.append(memory)

        return memories[:5]

    except Exception:
        # Do not break the voice flow if memory extraction fails.
        return []


def save_extracted_memories(
    npc: dict,
    npc_id: str,
    player_id: str,
    player_text: str,
    npc_text: str,
    campaign_id: str = "",
) -> list[str]:
    campaign_id = normalize_campaign_id(campaign_id)
    memories = extract_memories_from_exchange(
        npc=npc,
        npc_id=npc_id,
        player_id=player_id,
        player_text=player_text,
        npc_text=npc_text
    )

    saved = []

    for memory in memories:
        if not memory_exists(npc_id, player_id, memory, campaign_id=campaign_id):
            save_memory(npc_id, player_id, memory, campaign_id=campaign_id)
            saved.append(memory)

    return saved


@app.post("/npc-turn")
def npc_turn():
    body = request.get_json(force=True)

    npc_id = body.get("npc_id", "captain_varra")
    player_text = body["text"]
    scene = body.get("scene", {})
    history = body.get("history", [])
    about_npc_id = (body.get("about_npc_id") or body.get("asked_about_npc_id") or "").strip() or None
    session = resolve_request_session(body=body)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    party_context = session["party_context"]

    request_id = new_request_id(body.get("request_id"))
    begin_request_timing()
    total_start = time.perf_counter()
    step_start = total_start

    try:
        register_turn_progress(
            request_id,
            route="/npc-turn",
            npc_id=npc_id,
            player_id=player_id,
            mode="text",
            campaign_id=campaign_id,
        )
    except TurnInProgressError as ex:
        return _turn_in_progress_response(ex.progress)
    set_turn_progress(request_id, "received")

    try:
        log_dm_turn(
            request_id=request_id,
            route="/npc-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="start",
            player_text=player_text,
        )

        set_turn_progress(request_id, "loading_context")
        context = load_turn_pre_llm_context(
            npc_id,
            player_id,
            player_text=player_text,
            party_context=party_context,
            campaign_id=campaign_id,
        )
        npc = context["npc"]
        location = context["location"]
        scene_context = context["scene_context"]
        memories = context["memories"]
        player_claim = context["player_claim"]
        dm_context = context["dm_context"]
        step_start = log_timing(request_id, "load npc context", step_start)

        set_turn_progress(request_id, "preparing_prompt")
        messages = build_npc_llm_messages(
            npc,
            scene,
            memories,
            location,
            history=history,
            player_text=player_text,
            dm_context=dm_context,
            party_context=party_context,
            scene_context=scene_context,
            about_npc_id=about_npc_id,
        )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        print(
            f"[TIMING] {request_id} text prompt messages={len(messages)} chars={prompt_chars}",
            flush=True
        )
        step_start = log_timing(request_id, "build text prompt/messages", step_start)

        set_turn_progress(request_id, "waiting_llm")
        npc_text_raw = generate_npc_reply_text(
            npc, dm_context, messages, ask_llm, player_text=player_text
        )
        step_start = log_timing(request_id, "llm text npc response", step_start)

        ensure_turn_in_progress(request_id)
        set_turn_progress(request_id, "processing_response")
        post_llm_db = finalize_turn_database_phase(
            npc_id,
            player_id,
            player_text,
            npc_text_raw,
            player_pending_id=player_claim.get("pending_check_id") if player_claim else None,
            campaign_id=campaign_id,
            npc=npc,
        )
        npc_response = post_llm_db["npc_response"]
        npc_text = npc_response["spoken_text"]
        history_npc_text = post_llm_db["history_npc_text"]
        pending_items = post_llm_db["pending_items"]
        pending_count = post_llm_db["pending_count"]
        step_start = log_timing(request_id, "post-llm database phase", step_start)

        saved_memories = []
        memory_extraction_mode = "disabled"

        if history_npc_text and AUTO_MEMORY_EXTRACTION_ENABLED and AUTO_MEMORY_EXTRACTION_ASYNC:
            memory_extraction_mode = "async"
            run_memory_extraction_background(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=history_npc_text,
                campaign_id=campaign_id,
            )
        elif history_npc_text and AUTO_MEMORY_EXTRACTION_ENABLED:
            memory_extraction_mode = "sync"
            saved_memories = save_extracted_memories(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=history_npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "sync text memory extraction/save", step_start)

        total_elapsed = time.perf_counter() - total_start
        print(f"[TIMING] {request_id} TEXT TOTAL: {total_elapsed:.3f}s", flush=True)

        log_dm_turn(
            request_id=request_id,
            route="/npc-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="complete",
            player_text=player_text,
            player_claim=player_claim,
            npc_response=npc_response,
            campaign_id=campaign_id,
            pending_count=pending_count,
        )

        include_audio = _json_body_flag(body.get("include_audio"), False)
        narration_text = ""
        dialogue_text = ""
        audio_url = None
        audio_clips: list[dict[str, str]] = []

        if include_audio:
            ensure_turn_in_progress(request_id)
            # Keep progress in_progress through TTS, then complete once. Completing
            # before synthesize + reopening left clients polling forever after playback.
            set_turn_progress(request_id, "synthesizing_speech")
            clip_result = synthesize_segmented_speech_clips(
                npc_text=npc_text,
                npc_voice_model=pick_npc_voice_model(npc, body.get("voice_model")),
                request_id=request_id,
                npc_tts_options=pick_npc_tts_options(npc, _tts_request_values_from_mapping(body)),
            )
            audio_clips = clip_result["clips"]
            narration_text = clip_result["narration_text"]
            dialogue_text = clip_result["dialogue_text"]
            if clip_result.get("speech_path"):
                audio_url = turn_audio_url(request_id, "speech")
            elif clip_result.get("narration_path"):
                audio_url = turn_audio_url(request_id, "narration")
            step_start = log_timing(request_id, "synthesize text-turn speech clips", step_start)
            set_turn_progress(request_id, "finishing", clips=audio_clips)
        else:
            tts_segments = split_narration_and_dialogue(npc_text)
            narration_text = " ".join(s["text"] for s in tts_segments if s["type"] == "narrator").strip()
            dialogue_text = " ".join(s["text"] for s in tts_segments if s["type"] == "npc").strip()
            set_turn_progress(request_id, "finishing")

        complete_turn_progress(request_id)

        payload = {
            "request_id": request_id,
            "npc_id": npc_id,
            "npc_name": npc["name"],
            "player_id": player_id,
            "party_id": session["party_id"],
            "campaign_id": session["campaign_id"],
            "speaking_member": party_context.get("speaking_member"),
            "party_members": party_context.get("party_members"),
            "text": npc_text,
            "npc_text": npc_text,
            "player_text": player_text,
            "narration_text": narration_text,
            "dialogue_text": dialogue_text,
            "saved_memories": saved_memories,
            "memory_extraction_mode": memory_extraction_mode,
            "dm_check_claim": npc_response.get("dm_check_claim"),
            "dm_check_claims": npc_response.get("dm_check_claims"),
            "dm_pending_check_id": npc_response.get("pending_check_id"),
            "dm_pending_check_ids": npc_response.get("pending_check_ids"),
            "detected_player_claim": player_claim,
            "dm_pending": pending_items[-5:],
        }
        if include_audio:
            payload["audio_url"] = audio_url
            payload["audio_clips"] = audio_clips

        return attach_timing_headers(jsonify(payload), request_id)

    except TurnSupersededError as ex:
        print(
            f"[NPC-TURN] superseded request_id={request_id} npc={npc_id}",
            flush=True,
        )
        body, status = _turn_superseded_response(request_id, ex.progress)
        return attach_timing_headers(body, request_id), status

    except Exception as ex:
        fail_turn_progress(request_id, "turn_failed")
        log_dm_turn(
            request_id=request_id,
            route="/npc-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="error",
        )
        print(f"[NPC-TURN] failed request_id={request_id}: {ex!r}", flush=True)
        err = jsonify({
            "error": "Internal server error",
            "request_id": request_id,
            "npc_id": npc_id,
        })
        return attach_timing_headers(err, request_id), 500


@app.post("/npc-voice-turn")
def npc_voice_turn():
    npc_id = request.form.get("npc_id", "captain_varra")
    scene_description = request.form.get("scene", "")
    about_npc_id = (
        request.form.get("about_npc_id") or request.form.get("asked_about_npc_id") or ""
    ).strip() or None
    session = resolve_request_session(form=request.form)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    party_context = session["party_context"]

    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file field named 'audio'"}), 400

    request_id = new_request_id(request.form.get("request_id"))
    begin_request_timing()
    total_start = time.perf_counter()
    step_start = total_start

    try:
        register_turn_progress(
            request_id,
            route="/npc-voice-turn",
            npc_id=npc_id,
            player_id=player_id,
            mode="voice",
            campaign_id=campaign_id,
        )
    except TurnInProgressError as ex:
        return _turn_in_progress_response(ex.progress)
    set_turn_progress(request_id, "received")

    log_dm_turn(
        request_id=request_id,
        route="/npc-voice-turn",
        npc_id=npc_id,
        player_id=player_id,
        stage="start",
    )

    output_path = OUTPUT_DIR / f"{request_id}_response.wav"

    try:
        input_path = save_uploaded_audio(request.files["audio"], request_id)
        step_start = log_timing(request_id, "save uploaded audio", step_start)

        voice_override = (request.form.get("voice_model") or "").strip() or None
        schedule_piper_warm_for_voice_turn(
            npc_id,
            voice_override,
            request_id=request_id,
        )
        step_start = log_timing(request_id, "schedule piper warm-up", step_start)

        set_turn_progress(request_id, "transcribing")
        player_text = transcribe_audio(input_path)
        step_start = log_timing(request_id, "speech-to-text", step_start)

        if not player_text:
            print(
                f"[PLAYER-AUDIO] request_id={request_id} ERROR no_speech_detected "
                f"path={input_path} bytes={input_path.stat().st_size if input_path.exists() else 0}",
                flush=True,
            )
            fail_turn_progress(request_id, "Couldn't catch that — try again")
            err = jsonify({
                "error": "Couldn't catch that. Speak a bit clearer or hold a moment longer, then try again.",
                "request_id": request_id,
            })
            return attach_timing_headers(err, request_id, include_voice_phases=True), 400

        log_dm_turn(
            request_id=request_id,
            route="/npc-voice-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="stt",
            player_text=player_text,
        )

        set_turn_progress(request_id, "loading_context")
        context = load_turn_pre_llm_context(
            npc_id,
            player_id,
            player_text=player_text,
            party_context=party_context,
            campaign_id=campaign_id,
        )
        npc = context["npc"]
        location = context["location"]
        scene_context = context["scene_context"]
        memories = context["memories"]
        recent_turns = context["recent_turns"]
        player_claim = context["player_claim"]
        dm_context = context["dm_context"]
        step_start = log_timing(request_id, "load npc/location/memory/recent turns", step_start)

        set_turn_progress(request_id, "preparing_prompt")
        messages = build_npc_llm_messages(
            npc,
            {"description": scene_description},
            memories,
            location,
            recent_turns=recent_turns,
            player_text=player_text,
            dm_context=dm_context,
            party_context=party_context,
            scene_context=scene_context,
            about_npc_id=about_npc_id,
        )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        print(
            f"[TIMING] {request_id} prompt messages={len(messages)} chars={prompt_chars}",
            flush=True
        )
        step_start = log_timing(request_id, "build prompt/messages", step_start)

        set_turn_progress(request_id, "waiting_llm")
        npc_text_raw = generate_npc_reply_text(
            npc, dm_context, messages, ask_llm, player_text=player_text
        )
        step_start = log_timing(request_id, "llm npc response", step_start)

        ensure_turn_in_progress(request_id)
        set_turn_progress(request_id, "processing_response")
        post_llm_db = finalize_turn_database_phase(
            npc_id,
            player_id,
            player_text,
            npc_text_raw,
            player_pending_id=player_claim.get("pending_check_id") if player_claim else None,
            campaign_id=campaign_id,
            npc=npc,
        )
        npc_response = post_llm_db["npc_response"]
        npc_text = npc_response["spoken_text"]
        history_npc_text = post_llm_db["history_npc_text"]
        pending_count = post_llm_db["pending_count"]
        step_start = log_timing(request_id, "post-llm database phase", step_start)

        tts_segments = split_into_ordered_tts_segments(npc_text)
        tts_text = " ".join(segment["text"] for segment in tts_segments).strip()

        print(
            f"[TIMING] {request_id} tts segments={len(tts_segments)} "
            f"types={[s['type'] for s in tts_segments]}",
            flush=True
        )
        step_start = log_timing(request_id, "split tts segments", step_start)

        set_turn_progress(
            request_id,
            "text_ready",
            player_text=player_text,
            narration_text=" ".join(s["text"] for s in tts_segments if s["type"] == "narrator").strip(),
            dialogue_text=" ".join(s["text"] for s in tts_segments if s["type"] == "npc").strip(),
        )

        if not tts_text:
            npc_text = "I have nothing more to say."
            tts_text = npc_text
            tts_segments = [{"type": "npc", "text": tts_text}]
            set_turn_progress(
                request_id,
                "text_ready",
                player_text=player_text,
                narration_text="",
                dialogue_text=tts_text,
            )

        memory_extraction_mode = "disabled"
        if history_npc_text and AUTO_MEMORY_EXTRACTION_ENABLED and AUTO_MEMORY_EXTRACTION_ASYNC:
            memory_extraction_mode = "async"
            run_memory_extraction_background(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=history_npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "schedule async memory extraction", step_start)
        elif history_npc_text and AUTO_MEMORY_EXTRACTION_ENABLED:
            memory_extraction_mode = "sync"
            save_extracted_memories(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=history_npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "sync memory extraction/save", step_start)

        set_turn_progress(request_id, "generating_voice")

        stream_clips = (request.form.get("stream_clips") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        def _on_clip_ready(clip, clips_so_far):
            if not turn_is_in_progress(request_id):
                return
            step = "clip_ready"
            if clip.get("type") == "narration":
                step = "narration_ready"
            elif clip.get("type") == "speech":
                step = "speech_ready"
            set_turn_progress(
                request_id,
                step,
                player_text=player_text,
                narration_text=" ".join(
                    c.get("text", "") for c in clips_so_far if c.get("type") == "narration"
                ).strip(),
                dialogue_text=" ".join(
                    c.get("text", "") for c in clips_so_far if c.get("type") == "speech"
                ).strip(),
                clips=clips_so_far,
                message=f"Ready part {int(clip.get('index', 0)) + 1} of the response...",
            )

        def _on_stream_complete(final_result: dict, error: BaseException | None) -> None:
            if not turn_is_in_progress(request_id):
                return
            if error is not None:
                fail_turn_progress(request_id, "tts_stream_failed")
                return
            final_clips = final_result.get("clips") or []
            set_turn_progress(
                request_id,
                "speech_ready" if any(c.get("type") == "speech" for c in final_clips) else "narration_ready",
                player_text=player_text,
                narration_text=final_result.get("narration_text") or "",
                dialogue_text=final_result.get("dialogue_text") or "",
                clips=final_clips,
            )
            set_turn_progress(request_id, "finishing", clips=final_clips)
            complete_turn_progress(request_id)
            print(
                f"[PLAYER-AUDIO] stream_complete request_id={request_id} clips={len(final_clips)}",
                flush=True,
            )

        ensure_turn_in_progress(request_id)
        clip_result = synthesize_segmented_speech_clips(
            npc_text=npc_text,
            npc_voice_model=pick_npc_voice_model(
                npc,
                (request.form.get("voice_model") or "").strip() or None,
            ),
            request_id=request_id,
            npc_tts_options=pick_npc_tts_options(
                npc,
                _tts_request_values_from_mapping(request.form),
            ),
            on_clip_ready=_on_clip_ready,
            return_after_first_clip=stream_clips,
            on_stream_complete=_on_stream_complete if stream_clips else None,
        )
        step_start = log_timing(request_id, "piper segmented tts", step_start)

        narration_text = clip_result["narration_text"]
        dialogue_text = clip_result["dialogue_text"]
        clips = clip_result["clips"]
        speech_path = clip_result.get("speech_path")
        narration_path = clip_result.get("narration_path")
        streaming = bool(clip_result.get("pending"))
        # Prefer last segment as HTTP body; clients play via SignalR clip list.
        response_path = turn_audio_path(request_id, len(clips) - 1) if clips else (speech_path or narration_path or output_path)
        if not clips or not Path(response_path).is_file():
            # Race: superseded after synthesize returned, or missing WAV body.
            ensure_turn_in_progress(request_id)
            raise RuntimeError("TTS produced no playable audio for this turn")

        set_turn_progress(
            request_id,
            "speech_ready" if any(c.get("type") == "speech" for c in clips) else "narration_ready",
            player_text=player_text,
            narration_text=narration_text,
            dialogue_text=dialogue_text,
            clips=clips,
        )

        total_elapsed = time.perf_counter() - total_start
        print(
            f"[TIMING] {request_id} TOTAL: {total_elapsed:.3f}s "
            f"memory_mode={memory_extraction_mode} clips={len(clips)} "
            f"streaming={int(streaming)}",
            flush=True
        )

        log_dm_turn(
            request_id=request_id,
            route="/npc-voice-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="complete" if not streaming else "streaming",
            player_text=player_text,
            player_claim=player_claim,
            npc_response=npc_response,
            campaign_id=campaign_id,
            pending_count=pending_count,
        )

        if not streaming:
            set_turn_progress(request_id, "finishing", clips=clips)
            complete_turn_progress(request_id)

        response = make_response(send_file(
            response_path,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="npc_response.wav"
        ))
        response.headers["X-Turn-Request-Id"] = request_id
        if streaming:
            response.headers["X-Turn-Streaming"] = "1"
        response.headers["X-DM-Pending-Count"] = str(pending_count)
        pending_id = npc_response.get("pending_check_id") or (
            player_claim.get("pending_check_id") if player_claim else None
        )
        if pending_id:
            response.headers["X-DM-Pending-Check-Id"] = str(pending_id)
        if player_claim:
            response.headers["X-DM-Player-Claim-Detected"] = "1"
        if npc_response.get("npc_deferral_detected"):
            response.headers["X-DM-Npc-Deferral-Detected"] = "1"
        response.headers["X-Npc-Id"] = str(npc_id or "")
        response.headers["X-Npc-Name"] = str((npc or {}).get("name") or npc_id or "")
        response.headers["X-Player-Id"] = str(player_id or "")
        response.headers["X-Npc-Text"] = _header_b64(npc_text)
        response.headers["X-Player-Transcript"] = _header_b64(player_text)
        response.headers["X-Narration-Text"] = _header_b64(narration_text)
        response.headers["X-Dialogue-Text"] = _header_b64(dialogue_text)
        response.headers["X-Audio-Clips"] = "|".join(
            f"{c.get('index', i)}:{c.get('type')}" for i, c in enumerate(clips)
        )
        response.headers["X-Audio-Clip-Urls"] = "|".join(c["url"] for c in clips if c.get("url"))
        if narration_path:
            response.headers["X-Narration-Audio-Url"] = turn_audio_url(request_id, "narration")
        if speech_path:
            response.headers["X-Speech-Audio-Url"] = turn_audio_url(request_id, "speech")
        return attach_timing_headers(response, request_id, include_voice_phases=True)

    except TurnSupersededError as ex:
        print(
            f"[DM] route=/npc-voice-turn stage=superseded request_id={request_id}",
            flush=True,
        )
        body, status = _turn_superseded_response(request_id, ex.progress)
        return attach_timing_headers(body, request_id, include_voice_phases=True), status

    except Exception as ex:
        total_elapsed = time.perf_counter() - total_start
        print(f"[TIMING] {request_id} ERROR after {total_elapsed:.3f}s: {ex}", flush=True)
        log_dm_turn(
            request_id=request_id,
            route="/npc-voice-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="error",
        )
        print(f"[DM] route=/npc-voice-turn stage=error request_id={request_id} error={ex!r}", flush=True)
        fail_turn_progress(request_id, "turn_failed")
        err = jsonify({
            "error": "Internal server error",
            "request_id": request_id,
            "npc_id": npc_id
        })
        return attach_timing_headers(err, request_id, include_voice_phases=True), 500


@app.post("/npc-voice-turn-debug")
def npc_voice_turn_debug():
    npc_id = request.form.get("npc_id", "captain_varra")
    scene_description = request.form.get("scene", "")
    session = resolve_request_session(form=request.form)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    party_context = session["party_context"]

    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file field named 'audio'"}), 400

    request_id = uuid.uuid4().hex
    total_start = time.perf_counter()
    step_start = total_start

    log_dm_turn(
        request_id=request_id,
        route="/npc-voice-turn-debug",
        npc_id=npc_id,
        player_id=player_id,
        stage="start",
    )

    output_path = OUTPUT_DIR / f"{request_id}_response.wav"

    input_path = save_uploaded_audio(request.files["audio"], request_id)
    step_start = log_timing(request_id, "debug save uploaded audio", step_start)

    voice_override = (request.form.get("voice_model") or "").strip() or None
    schedule_piper_warm_for_voice_turn(
        npc_id,
        voice_override,
        request_id=request_id,
    )
    step_start = log_timing(request_id, "debug schedule piper warm-up", step_start)

    try:
        player_text = transcribe_audio(input_path)
        step_start = log_timing(request_id, "debug speech-to-text", step_start)

        if not player_text:
            return jsonify({
                "error": "Couldn't catch that. Speak a bit clearer or hold a moment longer, then try again.",
            }), 400

        log_dm_turn(
            request_id=request_id,
            route="/npc-voice-turn-debug",
            npc_id=npc_id,
            player_id=player_id,
            stage="stt",
            player_text=player_text,
        )

        context = load_turn_pre_llm_context(
            npc_id,
            player_id,
            player_text=player_text,
            party_context=party_context,
            campaign_id=campaign_id,
        )
        npc = context["npc"]
        location = context["location"]
        scene_context = context["scene_context"]
        memories = context["memories"]
        recent_turns = context["recent_turns"]
        player_claim = context["player_claim"]
        dm_context = context["dm_context"]
        step_start = log_timing(request_id, "debug load npc/location/memory/recent turns", step_start)

        messages = build_npc_llm_messages(
            npc,
            {"description": scene_description},
            memories,
            location,
            recent_turns=recent_turns,
            player_text=player_text,
            dm_context=dm_context,
            party_context=party_context,
            scene_context=scene_context,
        )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        print(
            f"[TIMING] {request_id} debug prompt messages={len(messages)} chars={prompt_chars}",
            flush=True
        )
        step_start = log_timing(request_id, "debug build prompt/messages", step_start)

        npc_text_raw = generate_npc_reply_text(
            npc, dm_context, messages, ask_llm, player_text=player_text
        )
        step_start = log_timing(request_id, "debug llm npc response", step_start)

        post_llm_db = finalize_turn_database_phase(
            npc_id,
            player_id,
            player_text,
            npc_text_raw,
            player_pending_id=player_claim.get("pending_check_id") if player_claim else None,
            campaign_id=campaign_id,
            npc=npc,
        )
        npc_response = post_llm_db["npc_response"]
        npc_text = npc_response["spoken_text"]
        history_npc_text = post_llm_db["history_npc_text"]
        pending_items = post_llm_db["pending_items"]
        pending_count = post_llm_db["pending_count"]
        step_start = log_timing(request_id, "debug post-llm database phase", step_start)

        tts_segments = split_narration_and_dialogue(npc_text)
        tts_text = " ".join(segment["text"] for segment in tts_segments).strip()

        if not tts_text:
            npc_text = "I have nothing more to say."
            tts_text = npc_text
            tts_segments = [{"type": "npc", "text": tts_text}]

        print(
            f"[TIMING] {request_id} debug tts segments={len(tts_segments)} "
            f"types={[s['type'] for s in tts_segments]}",
            flush=True
        )
        step_start = log_timing(request_id, "debug split tts segments", step_start)

        saved_memories = []
        memory_extraction_mode = "disabled"
        if history_npc_text and AUTO_MEMORY_EXTRACTION_ENABLED and AUTO_MEMORY_EXTRACTION_ASYNC:
            memory_extraction_mode = "async"
            run_memory_extraction_background(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=history_npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "debug schedule async memory extraction", step_start)
        elif history_npc_text and AUTO_MEMORY_EXTRACTION_ENABLED:
            memory_extraction_mode = "sync"
            saved_memories = save_extracted_memories(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=history_npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "debug sync memory extraction/save", step_start)

        clip_result = synthesize_segmented_speech_clips(
            npc_text=npc_text,
            npc_voice_model=pick_npc_voice_model(
                npc,
                (request.form.get("voice_model") or "").strip() or None,
            ),
            request_id=request_id,
            npc_tts_options=pick_npc_tts_options(
                npc,
                _tts_request_values_from_mapping(request.form),
            ),
        )
        paths = [p for p in (clip_result.get("clip_paths") or []) if p]
        if not paths:
            paths = [p for p in (clip_result.get("narration_path"), clip_result.get("speech_path")) if p]
        if len(paths) == 1:
            output_path.write_bytes(paths[0].read_bytes())
        elif paths:
            concatenate_wav_files(paths, output_path)
        step_start = log_timing(request_id, "debug piper segmented tts", step_start)

        total_elapsed = time.perf_counter() - total_start
        print(
            f"[TIMING] {request_id} DEBUG TOTAL: {total_elapsed:.3f}s "
            f"memory_mode={memory_extraction_mode}",
            flush=True
        )

        log_dm_turn(
            request_id=request_id,
            route="/npc-voice-turn-debug",
            npc_id=npc_id,
            player_id=player_id,
            stage="complete",
            player_text=player_text,
            player_claim=player_claim,
            npc_response=npc_response,
            campaign_id=campaign_id,
            pending_count=pending_count,
        )

        return jsonify({
            "npc_id": npc_id,
            "npc_name": npc["name"],
            "player_id": player_id,
            "party_id": session["party_id"],
            "campaign_id": session["campaign_id"],
            "speaking_member": party_context.get("speaking_member"),
            "party_members": party_context.get("party_members"),
            "transcript": player_text,
            "npc_text": npc_text,
            "narration_text": clip_result["narration_text"],
            "dialogue_text": clip_result["dialogue_text"],
            "tts_text": tts_text,
            "tts_segments": tts_segments,
            "audio_clips": clip_result["clips"],
            "location_id": npc.get("location_id", ""),
            "location_name": location.get("name", location.get("id", "")) if isinstance(location, dict) else "",
            "prompt_messages": len(messages),
            "prompt_characters": prompt_chars,
            "saved_memories": saved_memories,
            "memory_extraction_mode": memory_extraction_mode,
            "audio_file": str(output_path),
            "total_seconds": round(total_elapsed, 3),
            "dm_check_claim": npc_response.get("dm_check_claim"),
            "dm_check_claims": npc_response.get("dm_check_claims"),
            "dm_pending_check_id": npc_response.get("pending_check_id"),
            "dm_pending_check_ids": npc_response.get("pending_check_ids"),
            "detected_player_claim": player_claim,
            "dm_pending": pending_items[-5:],
        })

    except Exception as ex:
        total_elapsed = time.perf_counter() - total_start
        print(f"[TIMING] {request_id} DEBUG ERROR after {total_elapsed:.3f}s: {ex}", flush=True)
        return jsonify({
            "error": "Internal server error",
            "npc_id": npc_id
        }), 500


@app.get("/locations")
def list_locations():
    locations = []
    if has_pack():
        location_ids = list_document_ids("location")
    else:
        from session_context import get_request_campaign, get_request_owner

        owner = (get_request_owner() or "").strip()
        campaign = (get_request_campaign() or "").strip()
        if owner or campaign:
            return jsonify({
                "locations": [],
                "warning": "No content pack installed for this owner/campaign.",
            })
        location_ids = [
            path.stem
            for path in sorted(list(LOCATION_DIR.glob("*.yaml")) + list(LOCATION_DIR.glob("*.yml")))
        ]

    for location_id in location_ids:
        try:
            location = load_location(location_id)
            locations.append({
                "id": location_id,
                "name": location.get("name", location_id),
                "type": location.get("type", ""),
                "summary": location.get("summary", ""),
            })

        except Exception as ex:
            locations.append({
                "id": location_id,
                "name": location_id,
                "type": "",
                "summary": "",
                "error": "load_failed"
            })

    return jsonify({
        "locations": locations
    })


@app.post("/npc-preload")
def npc_preload():
    body = request.get_json(force=True, silent=True) or {}

    npc_id = body.get("npc_id") or request.args.get("npc_id") or "captain_varra"
    session = resolve_request_session(body=body, args=request.args)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    party_context = session["party_context"]

    try:
        context = get_npc_context(npc_id, player_id, party_context=party_context, campaign_id=campaign_id)
        npc = context["npc"]
        location = context["location"]

        # Build the prompt once so YAML/context formatting is warmed and validation happens now.
        prompt = build_system_prompt(
            npc,
            {"description": body.get("scene", "")},
            context["memories"],
            location,
            dm_context=context.get("dm_context"),
            party_context=party_context,
            scene_context=context.get("scene_context"),
        )

        return jsonify({
            "status": "preloaded",
            "npc_id": npc_id,
            "npc_name": npc.get("name", npc_id),
            "player_id": player_id,
            "party_id": session["party_id"],
            "campaign_id": session["campaign_id"],
            "speaking_member": party_context.get("speaking_member"),
            "party_members": party_context.get("party_members"),
            "location_id": npc.get("location_id", ""),
            "location_name": location.get("name", location.get("id", "")) if isinstance(location, dict) else "",
            "memory_count": len(context["memories"]),
            "recent_turn_count": len(context["recent_turns"]),
            "prompt_characters": len(prompt),
            "cache": {
                "npc_cached": npc_id in NPC_CACHE,
                "location_cached": bool(npc.get("location_id")) and npc.get("location_id") in LOCATION_CACHE
            }
        })

    except Exception as ex:
        print(f"[NPC-PRELOAD] failed npc={npc_id}: {ex!r}", flush=True)
        return jsonify({
            "status": "error",
            "npc_id": npc_id,
            "player_id": player_id,
            "error": "Internal server error"
        }), 500


@app.get("/llm/models")
def llm_models():
    return jsonify(get_llm_models_payload())


@app.get("/settings/prompt-caps")
def settings_prompt_caps_get():
    return jsonify({
        "ok": True,
        "caps": get_prompt_caps(),
        "defaults": dict(DEFAULT_PROMPT_CAPS),
    })


@app.put("/settings/prompt-caps")
@app.post("/settings/prompt-caps")
def settings_prompt_caps_put():
    body = request.get_json(force=True, silent=True) or {}
    raw = body.get("caps") if isinstance(body.get("caps"), dict) else body
    if not isinstance(raw, dict):
        return jsonify({"ok": False, "error": "Expected a caps object."}), 400
    saved = save_prompt_caps(raw)
    return jsonify({"ok": True, "caps": saved, "defaults": dict(DEFAULT_PROMPT_CAPS)})


@app.get("/llm/status")
def llm_status():
    return jsonify(get_llm_status())


@app.get("/llm/usage/report")
def llm_usage_report():
    """Alpha token/cost report for Admin charts (by model, by user, hourly series)."""
    hours_raw = request.args.get("hours", "24")
    try:
        hours = int(hours_raw)
    except (TypeError, ValueError):
        hours = 24
    return jsonify(get_llm_usage_cost_report(hours=hours))


@app.post("/llm/select")
def llm_select():
    body = request.get_json(force=True, silent=True) or {}

    model_id = body.get("model_id") or body.get("model")
    provider_mode = body.get("provider_mode")
    purpose = (body.get("purpose") or "chat").strip() or "chat"

    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400

    try:
        select_kwargs = {"provider_mode": provider_mode, "purpose": purpose}
        if "fallback_model_ids" in body:
            select_kwargs["fallback_model_ids"] = body.get("fallback_model_ids")
        result = select_llm_model(model_id, **select_kwargs)
        result["status"] = get_llm_status()
        return jsonify(result)
    except ValueError as ex:
        return _client_error(ex, 400)
    except Exception as ex:
        print(f"[LLM] select failed: {ex}", flush=True)
        return _client_error(ex, 500)


@app.post("/llm/reset-usage")
def llm_reset_usage():
    # LAN-only convenience endpoint; do not expose this middleware publicly.
    reset_usage_tracking()
    return jsonify({"status": "reset", "usage": get_llm_status().get("usage", {})})


@app.post("/content/import/analyze")
def content_import_analyze():
    body = request.get_json(force=True, silent=True) or {}

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing text"}), 400

    mode = str(body.get("mode") or "create_new").strip().lower()
    existing_campaign_id = body.get("existing_campaign_id")
    existing_location_ids = body.get("existing_location_ids") or []
    existing_npc_names = body.get("existing_npc_names") or []

    if not isinstance(existing_location_ids, list):
        return jsonify({"error": "existing_location_ids must be a list"}), 400
    if not isinstance(existing_npc_names, list):
        return jsonify({"error": "existing_npc_names must be a list"}), 400

    try:
        result = analyze_campaign_document(
            text,
            mode,
            existing_campaign_id=existing_campaign_id,
            existing_location_ids=existing_location_ids,
            existing_npc_names=existing_npc_names,
        )
        return jsonify(result)
    except ContentImportError as ex:
        # User-facing analysis failures (empty model output, bad JSON, etc.) —
        # return the message so the editor can show it instead of generic "Bad request".
        print(f"[CONTENT-IMPORT] analyze rejected: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[CONTENT-IMPORT] analyze failed: {ex!r}", flush=True)
        return _client_error(ex, 500)


@app.post("/content/import/discover")
def content_import_discover():
    """Discovery hop: campaign + location/NPC stubs with temporary keys."""
    body = request.get_json(force=True, silent=True) or {}

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing text"}), 400

    mode = str(body.get("mode") or "create_new").strip().lower()
    existing_campaign_id = body.get("existing_campaign_id")
    existing_location_ids = body.get("existing_location_ids") or []
    existing_npc_names = body.get("existing_npc_names") or []

    if not isinstance(existing_location_ids, list):
        return jsonify({"error": "existing_location_ids must be a list"}), 400
    if not isinstance(existing_npc_names, list):
        return jsonify({"error": "existing_npc_names must be a list"}), 400

    try:
        result = discover_campaign_document(
            text,
            mode,
            existing_campaign_id=existing_campaign_id,
            existing_location_ids=existing_location_ids,
            existing_npc_names=existing_npc_names,
        )
        return jsonify(result)
    except ContentImportError as ex:
        print(f"[CONTENT-IMPORT] discover rejected: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[CONTENT-IMPORT] discover failed: {ex!r}", flush=True)
        return _client_error(ex, 500)


@app.post("/content/import/outline")
def content_import_outline():
    """Short Funnel hop: thin wrapper over discover (temporary-key stubs)."""
    return content_import_discover()


@app.post("/content/import/location-details")
def content_import_location_details():
    """Short Funnel hop: detail one location batch."""
    body = request.get_json(force=True, silent=True) or {}

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing text"}), 400

    campaign = body.get("campaign")
    if not isinstance(campaign, dict):
        return jsonify({"error": "campaign object is required"}), 400

    locations = body.get("locations")
    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "locations must be a non-empty list"}), 400

    try:
        result = detail_location_batch(
            text,
            campaign,
            locations,
            is_retry=bool(body.get("retry") or body.get("is_retry")),
            allow_seed_fallback=bool(body.get("allow_seed_fallback")),
        )
        return jsonify(result)
    except ContentImportError as ex:
        print(f"[CONTENT-IMPORT] location-details rejected: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[CONTENT-IMPORT] location-details failed: {ex!r}", flush=True)
        return _client_error(ex, 500)


@app.post("/content/import/npc-details")
def content_import_npc_details():
    """Short Funnel hop: detail one NPC batch."""
    body = request.get_json(force=True, silent=True) or {}

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing text"}), 400

    campaign = body.get("campaign")
    if not isinstance(campaign, dict):
        return jsonify({"error": "campaign object is required"}), 400

    npcs = body.get("npcs")
    if not isinstance(npcs, list) or not npcs:
        return jsonify({"error": "npcs must be a non-empty list"}), 400

    try:
        result = detail_npc_batch(
            text,
            campaign,
            npcs,
            is_retry=bool(body.get("retry") or body.get("is_retry")),
            allow_seed_fallback=bool(body.get("allow_seed_fallback")),
        )
        return jsonify(result)
    except ContentImportError as ex:
        print(f"[CONTENT-IMPORT] npc-details rejected: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[CONTENT-IMPORT] npc-details failed: {ex!r}", flush=True)
        return _client_error(ex, 500)


@app.get("/session/status")
def session_status():
    return jsonify({
        **get_session_status(),
        "scene": get_scene_status(include_options=False),
    })


@app.get("/scene/status")
def scene_status():
    return jsonify(get_scene_status())


@app.post("/scene/time")
def scene_time_set():
    body = request.get_json(force=True, silent=True) or {}
    time_of_day = (body.get("time_of_day") or body.get("time") or "").strip()

    if not time_of_day:
        return jsonify({"error": "Missing time_of_day"}), 400

    try:
        return jsonify(set_time_of_day(time_of_day))
    except ValueError as ex:
        return _client_error(ex, 400)


@app.post("/scene/time/clear")
def scene_time_clear():
    return jsonify(clear_time_of_day())


@app.get("/scene/sub-locations")
def scene_sub_locations():
    location_id = (request.args.get("location_id") or "").strip()
    if not location_id:
        return jsonify({"error": "Missing location_id query parameter"}), 400

    try:
        return jsonify(list_sub_location_options(location_id))
    except ValueError as ex:
        return _client_error(ex, 400)


@app.post("/scene/sub-location")
def scene_sub_location_set():
    body = request.get_json(force=True, silent=True) or {}
    location_id = (body.get("location_id") or "").strip()
    important_location_name = (body.get("important_location_name") or body.get("name") or "").strip()
    custom = (body.get("custom") or body.get("custom_name") or "").strip()

    if not location_id:
        return jsonify({"error": "Missing location_id"}), 400

    try:
        return jsonify(set_sub_location(
            location_id=location_id,
            important_location_name=important_location_name or None,
            custom=custom or None,
        ))
    except ValueError as ex:
        return _client_error(ex, 400)


@app.post("/scene/sub-location/clear")
def scene_sub_location_clear():
    return jsonify(clear_sub_location())


@app.get("/turn/progress/steps")
def turn_progress_steps():
    return jsonify(get_turn_progress_catalog())


@app.post("/turn/progress/register")
def turn_progress_register():
    body = request.get_json(force=True, silent=True) or {}
    request_id = new_request_id(body.get("request_id"))
    route = (body.get("route") or "/npc-voice-turn").strip()
    npc_id = (body.get("npc_id") or "").strip()
    player_id = (
        body.get("player_id")
        or resolve_request_session(body=body).get("player_id")
        or DEFAULT_PLAYER_ID
    )
    mode = (body.get("mode") or "voice").strip().lower()
    if mode not in ("voice", "text"):
        mode = "voice" if "voice" in route else "text"

    session = resolve_request_session(body=body)
    campaign_id = session["campaign_id"]
    try:
        return jsonify(register_turn_progress(
            request_id,
            route=route,
            npc_id=npc_id,
            player_id=player_id,
            mode=mode,
            campaign_id=campaign_id,
        ))
    except TurnInProgressError as ex:
        return _turn_in_progress_response(ex.progress)


@app.get("/turn/progress/active")
def turn_progress_active():
    session = resolve_request_session(args=request.args)
    campaign_id = session["campaign_id"]
    npc_id = (request.args.get("npc_id") or "").strip()
    player_id = session["player_id"]
    active = get_active_turn_progress(campaign_id, npc_id, player_id)
    if not active:
        return jsonify({"status": "idle"})
    return jsonify(active)


@app.get("/turn/progress/<request_id>")
def turn_progress_status(request_id: str):
    progress = get_turn_progress(request_id)
    if not progress:
        return jsonify({
            "request_id": request_id,
            "status": "not_found",
            "message": "No active turn found for this request yet.",
        }), 404

    return jsonify(progress)


@app.get("/turn-audio/<request_id>/<clip>")
def turn_audio_clip(request_id: str, clip: str):
    """Serve per-turn segment WAV (index or legacy narration/speech) for early SignalR playback."""
    normalized = normalize_client_request_id(request_id)
    if not normalized:
        return jsonify({"error": "Invalid request_id."}), 400

    clip_key = (clip or "").strip().lower()
    try:
        path = turn_audio_path(normalized, clip_key)
    except ValueError as ex:
        return _client_error(ex, 400)

    if not path.exists():
        return jsonify({"error": "Audio clip not ready.", "request_id": normalized, "clip": clip_key}), 404

    return send_file(
        path,
        mimetype="audio/wav",
        as_attachment=False,
        download_name=path.name,
    )


@app.get("/campaigns")
def campaigns_list():
    return jsonify({
        "campaigns": list_campaigns(),
        "active_campaign_id": get_active_session_ids()["campaign_id"],
    })


@app.get("/campaign/active")
def campaign_active():
    status = get_session_status()
    campaign_id = status.get("campaign_id") or ""
    return jsonify({
        "campaign_id": campaign_id,
        "campaign": status.get("active_campaign") or load_campaign(campaign_id),
    })


@app.post("/campaign/select")
def campaign_select():
    started = time.perf_counter()
    body = request.get_json(force=True, silent=True) or {}
    campaign_id = (body.get("campaign_id") or body.get("id") or "").strip()

    if not campaign_id:
        return jsonify({"error": "Missing campaign_id"}), 400

    try:
        owner_user_id = (
            str(body.get("owner_user_id") or "").strip()
            or request.headers.get("X-Campaign-Owner-User-Id", "").strip()
            or request.headers.get("X-Owner-User-Id", "").strip()
        )
        content_pack = body.get("content_pack")
        files = body.get("files") if isinstance(body.get("files"), list) else None
        yaml_text = body.get("yaml") or body.get("content")
        if isinstance(yaml_text, str) and yaml_text.strip() and not files and not content_pack:
            files = [{
                "kind": "campaign",
                "filename": f"{campaign_id}.yaml",
                "content": yaml_text,
            }]

        pack_info = None
        if owner_user_id and (content_pack or files):
            set_request_owner(owner_user_id)
            pack_info = install_content_pack(
                owner_user_id,
                campaign_id=campaign_id,
                content_pack=content_pack if isinstance(content_pack, dict) else None,
                files=files,
            )
            print(
                f"[CONTENT] installed pack owner={owner_user_id} campaign={campaign_id} "
                f"counts={pack_info.get('counts')}",
                flush=True,
            )

        result = select_campaign(campaign_id, owner_user_id=owner_user_id or None)
        payload = {"status": "selected", **result}
        if pack_info:
            payload["content_pack"] = pack_info
        print(
            f"[SESSION] POST /campaign/select campaign={campaign_id} "
            f"in {time.perf_counter() - started:.3f}s",
            flush=True,
        )
        return jsonify(payload)
    except ValueError as ex:
        return _client_error(ex, 400)
    except Exception as ex:
        print(f"[SESSION] campaign select failed: {ex}", flush=True)
        return _client_error(ex, 500)


@app.post("/content/pack")
def content_pack_install():
    """Refresh in-memory tenant YAML without changing the active campaign session."""
    body = request.get_json(force=True, silent=True) or {}
    owner_user_id = (
        str(body.get("owner_user_id") or "").strip()
        or request.headers.get("X-Campaign-Owner-User-Id", "").strip()
        or request.headers.get("X-Owner-User-Id", "").strip()
    )
    campaign_id = str(body.get("campaign_id") or "").strip()
    if not owner_user_id:
        return jsonify({"error": "Missing owner_user_id"}), 400

    try:
        set_request_owner(owner_user_id)
        pack_info = install_content_pack(
            owner_user_id,
            campaign_id=campaign_id,
            content_pack=body.get("content_pack") if isinstance(body.get("content_pack"), dict) else None,
            files=body.get("files") if isinstance(body.get("files"), list) else None,
        )
        return jsonify({"status": "installed", **pack_info})
    except ValueError as ex:
        return _client_error(ex, 400)
    except Exception as ex:
        print(f"[CONTENT] pack install failed: {ex}", flush=True)
        return _client_error(ex, 500)


@app.get("/parties")
def parties_list():
    campaign_id = request.args.get("campaign_id") or get_active_session_ids()["campaign_id"] or None
    return jsonify({
        "campaign_id": campaign_id or "",
        "parties": list_parties(campaign_id=campaign_id or None),
    })


@app.get("/party/characters")
def party_characters():
    status = get_session_status()
    campaign_id = request.args.get("campaign_id") or status.get("campaign_id") or None
    party_id = request.args.get("party_id") or status.get("party_id") or None
    characters = get_character_options(
        campaign_id=campaign_id or None,
        party_id=party_id or None,
    )

    return jsonify({
        "campaign_id": campaign_id or "",
        "party_id": party_id or "",
        "active_player_id": status.get("player_id") or DEFAULT_PLAYER_ID,
        "characters": characters,
        "includes_default_party": any(character.get("is_default") for character in characters),
    })


@app.get("/party/active")
def party_active():
    status = get_session_status()
    party_id = status.get("party_id") or DEFAULT_PARTY_ID
    return jsonify({
        "party_id": party_id,
        "party": load_party(party_id),
        "speaking_member": status.get("speaking_member") or {},
        "party_members": status.get("active_party", {}).get("members") or [],
    })


@app.post("/party/select")
def party_select():
    body = request.get_json(force=True, silent=True) or {}
    player_id = (body.get("player_id") or body.get("character_id") or "").strip()
    party_id = (body.get("party_id") or "").strip() or None

    if not player_id:
        return jsonify({"error": "Missing player_id"}), 400

    try:
        result = select_character(player_id, party_id=party_id)
        return jsonify({"status": "selected", **result})
    except ValueError as ex:
        return _client_error(ex, 400)
    except Exception as ex:
        print(f"[SESSION] party select failed: {ex}", flush=True)
        return _client_error(ex, 500)


@app.get("/dm/status")
def dm_status():
    npc_id = request.args.get("npc_id", "captain_varra")
    session = resolve_request_session(args=request.args)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]

    return jsonify({
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "speaking_member": session["party_context"].get("speaking_member"),
        **get_dm_context(npc_id, player_id, campaign_id=campaign_id),
    })


@app.get("/dm/messages")
def dm_messages():
    npc_id = request.args.get("npc_id", "captain_varra")
    session = resolve_request_session(args=request.args)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 100))
    except (TypeError, ValueError):
        limit = 50

    return jsonify({
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "messages": load_dm_messages(npc_id, player_id, limit=limit, campaign_id=campaign_id),
    })


@app.post("/dm/message")
def dm_message_post():
    body = request.get_json(force=True, silent=True) or {}

    npc_id = body.get("npc_id", "captain_varra")
    session = resolve_request_session(body=body)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    text = (body.get("text") or body.get("content") or "").strip()
    verdict_override = body.get("verdict")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        saved = save_dm_message(
            npc_id=npc_id,
            player_id=player_id,
            content=text,
            verdict_override=verdict_override,
                campaign_id=campaign_id,
            )

        auto_response = None
        verdict = (saved.get("detected_verdict") or "").lower()
        if _json_body_flag(body.get("auto_respond"), True) and verdict in ("confirmed", "denied") and saved.get("ruling_id"):
            ruling = load_ruling_by_id(saved["ruling_id"])
            if ruling:
                auto_response = run_dm_ruling_auto_response(
                    npc_id,
                    player_id,
                    ruling,
                    include_audio=_json_body_flag(body.get("include_audio"), True),
                    scene_description=(body.get("scene") or "").strip(),
                    party_context=session["party_context"],
                    voice_model=(body.get("voice_model") or "").strip() or None,
                    tts_length_scale=body.get("tts_length_scale"),
                    tts_options=_tts_request_values_from_mapping(body),
                    campaign_id=campaign_id,
                )

        return jsonify({
            "status": "saved",
            "npc_id": npc_id,
            "player_id": player_id,
            "party_id": session["party_id"],
            "campaign_id": session["campaign_id"],
            "speaking_member": session["party_context"].get("speaking_member"),
            "message": saved,
            "auto_response": auto_response,
            "pending": load_pending_checks(npc_id, player_id, status="pending", limit=10, campaign_id=campaign_id),
            "rulings": load_rulings(npc_id, player_id, limit=10, campaign_id=campaign_id),
        })
    except ValueError as ex:
        return _client_error(ex, 400)
    except Exception as ex:
        print(f"[DM] message post failed: {ex}", flush=True)
        return _client_error(ex, 500)


@app.get("/dm/pending")
def dm_pending():
    # Empty npc_id = all NPCs in the campaign (DM console "All NPCs" filter).
    npc_id = (request.args.get("npc_id") or "").strip()
    session = resolve_request_session(args=request.args)
    campaign_id = session["campaign_id"]
    status = request.args.get("status", "pending")

    checks = [
        _enrich_pending_check_display(item)
        for item in load_pending_checks_for_dm(
            campaign_id=campaign_id,
            npc_id=npc_id or None,
            status=status or None,
            limit=50,
        )
    ]
    return jsonify({
        "npc_id": npc_id or None,
        "player_id": session["player_id"],
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "pending_checks": checks,
        # Alias used by older DM console clients.
        "pending": checks,
    })


@app.post("/dm/pending/dismiss")
def dm_pending_dismiss():
    """Ignore/delete a pending check without creating a confirm/deny ruling."""
    body = request.get_json(force=True, silent=True) or {}
    session = resolve_request_session(body=body)
    campaign_id = session["campaign_id"]

    pending_raw = body.get("pending_check_id", body.get("check_id"))
    try:
        pending_check_id = int(pending_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "Missing or invalid pending_check_id"}), 400

    pending = load_pending_check_by_id(pending_check_id)
    if not pending:
        return jsonify({"error": "Pending check not found"}), 404

    pending_campaign = normalize_campaign_id(pending.get("campaign_id"))
    if campaign_id and pending_campaign and pending_campaign != campaign_id:
        return jsonify({"error": "Pending check belongs to a different campaign"}), 400

    try:
        updated = dismiss_pending_check(pending_check_id)
    except LookupError:
        return jsonify({"error": "Pending check not found"}), 404
    except ValueError as ex:
        current = load_pending_check_by_id(pending_check_id)
        return jsonify({
            "error": str(ex),
            "status": (current or {}).get("status"),
            "pending_check_id": pending_check_id,
            "check": _enrich_pending_check_display(current) if current else None,
        }), 409

    if (updated.get("status") or "").strip().lower() != "ignored":
        return jsonify({
            "error": "Pending check could not be ignored",
            "status": updated.get("status"),
            "pending_check_id": pending_check_id,
            "check": _enrich_pending_check_display(updated),
        }), 409

    remaining = [
        _enrich_pending_check_display(item)
        for item in load_pending_checks_for_dm(
            campaign_id=pending_campaign or campaign_id,
            npc_id=None,
            status="pending",
            limit=50,
        )
    ]
    return jsonify({
        "status": "ignored",
        "pending_check_id": pending_check_id,
        "check": _enrich_pending_check_display(updated),
        "campaign_id": pending_campaign or campaign_id,
        "pending_checks": remaining,
        "pending": remaining,
    })


@app.get("/dm/rulings")
def dm_rulings():
    npc_id = (request.args.get("npc_id") or "").strip()
    session = resolve_request_session(args=request.args)
    campaign_id = session["campaign_id"]
    try:
        limit = max(1, min(int(request.args.get("limit", 20)), 100))
    except (TypeError, ValueError):
        limit = 20

    rulings = load_rulings_for_dm(
        campaign_id=campaign_id,
        npc_id=npc_id or None,
        limit=limit,
    )
    return jsonify({
        "npc_id": npc_id or None,
        "player_id": session["player_id"],
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "rulings": rulings,
    })


@app.post("/dm/ruling")
def dm_ruling_post():
    body = request.get_json(force=True, silent=True) or {}

    session = resolve_request_session(body=body)
    campaign_id = session["campaign_id"]
    npc_id = (body.get("npc_id") or "").strip() or "captain_varra"
    player_id = session["player_id"]
    claim_summary = (body.get("claim_summary") or body.get("claim") or "").strip()
    raw_verdict = (body.get("verdict") or "").strip().lower()
    verdict_aliases = {
        "confirm": "confirmed",
        "confirmed": "confirmed",
        "deny": "denied",
        "denied": "denied",
    }
    verdict = verdict_aliases.get(raw_verdict, raw_verdict)
    notes = (body.get("notes") or body.get("text") or "").strip()

    pending_raw = body.get("pending_check_id", body.get("check_id"))
    pending_check_id = None
    try:
        if pending_raw is not None and str(pending_raw).strip() != "":
            pending_check_id = int(pending_raw)
    except (TypeError, ValueError):
        pending_check_id = None

    if pending_check_id is not None:
        pending = load_pending_check_by_id(pending_check_id)
        if pending:
            # Prefer the pending row's identity so rulings match the original turn.
            npc_id = (pending.get("npc_id") or npc_id).strip() or npc_id
            player_id = (pending.get("player_id") or player_id).strip() or player_id
            if not claim_summary:
                claim_summary = (pending.get("claim_summary") or "").strip()
            pending_campaign = normalize_campaign_id(pending.get("campaign_id"))
            if pending_campaign:
                campaign_id = pending_campaign

    # Proxy strips player_id; rebuild speaker context for the claim's player.
    party_context = build_party_context(
        player_id=player_id,
        party_id=session["party_id"],
    )
    if campaign_id:
        party_context["campaign_id"] = campaign_id
        if not party_context.get("campaign"):
            party_context["campaign"] = load_campaign(campaign_id)

    if not claim_summary:
        return jsonify({"error": "Missing claim_summary"}), 400
    if verdict not in ("confirmed", "denied"):
        return jsonify({"error": "verdict must be confirmed or denied"}), 400

    try:
        result = create_manual_ruling(
            npc_id=npc_id,
            player_id=player_id,
            claim_summary=claim_summary,
            verdict=verdict,
            notes=notes,
            pending_check_id=pending_check_id,
            campaign_id=campaign_id,
        )

        auto_response = None
        if _json_body_flag(body.get("auto_respond"), True) and result.get("ruling_id"):
            ruling = load_ruling_by_id(result["ruling_id"])
            if ruling:
                auto_response = run_dm_ruling_auto_response(
                    npc_id,
                    player_id,
                    ruling,
                    include_audio=_json_body_flag(body.get("include_audio"), True),
                    scene_description=(body.get("scene") or "").strip(),
                    party_context=party_context,
                    voice_model=(body.get("voice_model") or "").strip() or None,
                    tts_length_scale=body.get("tts_length_scale"),
                    tts_options=_tts_request_values_from_mapping(body),
                    campaign_id=campaign_id,
                )

        pending_list = [
            _enrich_pending_check_display(item)
            for item in load_pending_checks_for_dm(
                campaign_id=campaign_id,
                npc_id=None,
                status="pending",
                limit=10,
            )
        ]
        rulings_list = load_rulings_for_dm(
            campaign_id=campaign_id,
            npc_id=None,
            limit=10,
        )
        return jsonify({
            "status": "saved",
            "npc_id": npc_id,
            "player_id": player_id,
            "party_id": session["party_id"],
            "campaign_id": campaign_id,
            "speaking_member": party_context.get("speaking_member"),
            **result,
            "auto_response": auto_response,
            "pending": pending_list,
            "pending_checks": pending_list,
            "rulings": rulings_list,
        })
    except ValueError as ex:
        return _client_error(ex, 400)
    except Exception as ex:
        print(f"[DM] ruling post failed: {ex}", flush=True)
        return _client_error(ex, 500)


@app.post("/dm/detect")
def dm_detect():
    """Preview automatic ruling detection without saving. LAN-only helper for the Pi DM page."""
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Missing text"}), 400

    return jsonify({
        "text": text,
        "detection": detect_ruling(text),
    })


@app.post("/dm/speak")
def dm_speak():
    """
    Speak DM-authored text as the current NPC (no LLM).

    Used for non-interactive dialogue such as speeches. Synthesizes TTS with the
    NPC voice, optionally saves the line into conversation history, and returns
    the same audio_clips shape used by DM auto-responses for SignalR fan-out.
    """
    body = request.get_json(force=True, silent=True) or {}
    npc_id = (body.get("npc_id") or "captain_varra").strip() or "captain_varra"
    text = (body.get("text") or body.get("content") or "").strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400

    session = resolve_request_session(body=body)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    include_audio = _json_body_flag(body.get("include_audio"), True)
    save_history = _json_body_flag(body.get("save_history"), True)
    request_id = normalize_client_request_id(body.get("request_id")) or uuid.uuid4().hex

    try:
        context = get_npc_context(
            npc_id,
            player_id,
            party_context=session.get("party_context"),
            campaign_id=campaign_id,
            include_dm_context=False,
        )
        npc = context["npc"]
        campaign_id = context.get("campaign_id") or campaign_id

        if save_history:
            save_turn(npc_id, player_id, "assistant", text, campaign_id=campaign_id)

        narration_text = ""
        dialogue_text = ""
        audio_url = None
        audio_filename = None
        audio_clips: list[dict[str, Any]] = []

        if include_audio:
            clip_result = synthesize_segmented_speech_clips(
                npc_text=text,
                npc_voice_model=pick_npc_voice_model(
                    npc,
                    (body.get("voice_model") or "").strip() or None,
                ),
                request_id=request_id,
                npc_tts_options=pick_npc_tts_options(
                    npc,
                    _tts_request_values_from_mapping(body),
                ),
            )
            audio_clips = clip_result["clips"]
            narration_text = clip_result["narration_text"]
            dialogue_text = clip_result["dialogue_text"]
            if clip_result.get("speech_path"):
                audio_url = turn_audio_url(request_id, "speech")
                audio_filename = turn_audio_path(request_id, "speech").name
            elif clip_result.get("narration_path"):
                audio_url = turn_audio_url(request_id, "narration")
                audio_filename = turn_audio_path(request_id, "narration").name
        else:
            segments = split_narration_and_dialogue(text)
            narration_text = " ".join(
                s["text"] for s in segments if s["type"] == "narrator"
            ).strip()
            dialogue_text = " ".join(
                s["text"] for s in segments if s["type"] == "npc"
            ).strip()

        print(
            f"[DM] speak request_id={request_id} npc={npc_id} player={player_id} "
            f"chars={len(text)} audio={bool(audio_url)} clips={len(audio_clips)}",
            flush=True,
        )

        return jsonify({
            "status": "spoken",
            "generated": True,
            "request_id": request_id,
            "npc_id": npc_id,
            "npc_name": npc.get("name", npc_id),
            "player_id": player_id,
            "party_id": session.get("party_id"),
            "campaign_id": campaign_id,
            "npc_text": text,
            "narration_text": narration_text,
            "dialogue_text": dialogue_text,
            "audio_url": audio_url,
            "audio_filename": audio_filename,
            "audio_clips": audio_clips,
            "saved_history": save_history,
        })
    except FileNotFoundError as ex:
        return _client_error(ex, 404, not_found=True)
    except ValueError as ex:
        return _client_error(ex, 400)
    except Exception as ex:
        print(f"[DM] speak failed: {ex!r}", flush=True)
        return _client_error(ex, 500)


@app.get("/dm/followup-audio/<filename>")
def dm_followup_audio(filename: str):
    safe_name = Path(filename).name
    if not re.fullmatch(r"[a-f0-9]{32}_dm_followup\.wav", safe_name):
        return jsonify({"error": "Invalid followup audio filename"}), 400

    output_path = OUTPUT_DIR / safe_name
    if not output_path.exists():
        return jsonify({"error": "Followup audio not found"}), 404

    return send_file(
        output_path,
        mimetype="audio/wav",
        as_attachment=False,
        download_name=safe_name,
    )


@app.get("/voices")
def list_voices():
    started = time.perf_counter()
    force_refresh = (request.args.get("refresh") or "").strip().lower() in ("1", "true", "yes")
    voices = refresh_voice_inventory(force=force_refresh)
    print(
        f"[VOICES] GET /voices {len(voices)} model(s) in {time.perf_counter() - started:.3f}s "
        f"(force_refresh={force_refresh})",
        flush=True,
    )
    return jsonify({
        "voices": voices,
        "voice_enabled": True,
        "voices_dir": str(VOICES_DIR),
        "cached": not force_refresh,
    })


@app.get("/voices/attribution")
def list_voice_attributions():
    """Public-facing MODEL_CARD credits for each Piper model under VOICES_DIR."""
    from voice_attribution import build_voice_attributions

    started = time.perf_counter()
    force_refresh = (request.args.get("refresh") or "").strip().lower() in ("1", "true", "yes")
    voices = refresh_voice_inventory(force=force_refresh)
    rows = build_voice_attributions(VOICES_DIR, voices)
    print(
        f"[VOICES] GET /voices/attribution {len(rows)} model(s) in "
        f"{time.perf_counter() - started:.3f}s (force_refresh={force_refresh})",
        flush=True,
    )
    return jsonify({
        "voices": rows,
        "voice_enabled": True,
        "voices_dir": str(VOICES_DIR),
        "cached": not force_refresh,
    })


@app.post("/voices/preview")
def preview_voice():
    started = time.perf_counter()
    body = request.get_json(force=True, silent=True) or {}
    voice_path = (body.get("voice_path") or body.get("voice_model") or "").strip()
    if not voice_path:
        return jsonify({"error": "Missing voice_path"}), 400

    text = (body.get("text") or VOICE_PREVIEW_TEXT).strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400

    opts = options_from_request_and_npc(
        _tts_request_values_from_mapping(body),
        {"voice_model": voice_path},
    )
    spoken = clean_for_tts(text, pronunciation=opts.get("pronunciation"))

    raw_path = OUTPUT_DIR / f"{uuid.uuid4().hex}_voice_preview_raw.wav"
    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}_voice_preview.wav"
    try:
        resolve_started = time.perf_counter()
        model_path = resolve_voice_model_path(voice_path)
        resolve_elapsed = time.perf_counter() - resolve_started
        synth_started = time.perf_counter()
        synthesize_speech(
            spoken,
            str(model_path),
            raw_path,
            length_scale=opts["length_scale"],
            noise_scale=opts["noise_scale"],
            noise_w=opts["noise_w"],
            sentence_silence=opts["sentence_silence"],
            speaker_id=opts.get("speaker_id"),
        )
        convert_wav_to_standard_format(
            raw_path,
            output_path,
            pitch_semitones=float(opts.get("pitch_semitones") or 0.0),
            fx_preset=str(opts.get("fx_preset") or DEFAULT_FX_PRESET),
            denoise_nr=opts.get("denoise_nr"),
            denoise_nf=opts.get("denoise_nf"),
        )
        print(
            f"[VOICE] preview ok resolve={resolve_elapsed:.3f}s "
            f"synth+fx={time.perf_counter() - synth_started:.3f}s "
            f"total={time.perf_counter() - started:.3f}s path={voice_path}",
            flush=True,
        )
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="voice_preview.wav",
        )
    except FileNotFoundError as ex:
        return _client_error(ex, 404, not_found=True)
    except subprocess.CalledProcessError as ex:
        return jsonify({"error": f"Piper synthesis failed: {ex}"}), 500
    except Exception as ex:
        print(f"[VOICE] preview failed: {ex!r}", flush=True)
        return _client_error(ex, 500)
    finally:
        raw_path.unlink(missing_ok=True)


@app.post("/transcribe")
def transcribe_only():
    """STT-only endpoint for Discord wake-word gating (no NPC turn)."""
    upload = request.files.get("audio") or request.files.get("file")
    if upload is None:
        return jsonify({"error": "audio is required."}), 400

    request_id = f"transcribe-{uuid.uuid4().hex}"
    try:
        # Same normalize path as /npc-voice-turn (webm/ogg/opus → WAV).
        wav_path = save_uploaded_audio(upload, request_id)
        transcript = (transcribe_audio(wav_path) or "").strip()
        return jsonify({"transcript": transcript})
    except Exception as ex:
        print(f"[TRANSCRIBE] failed: {ex!r}", flush=True)
        return jsonify({"error": f"Transcription failed: {ex}"}), 500
    finally:
        for leftover in UPLOAD_DIR.glob(f"{request_id}*"):
            leftover.unlink(missing_ok=True)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/npc-memory")
def get_npc_memory():
    npc_id = request.args.get("npc_id", "captain_varra")
    session = resolve_request_session(args=request.args)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]

    memories = load_memories(npc_id, player_id, limit=100, campaign_id=campaign_id)
    recent_turns = load_recent_turns(npc_id, player_id, limit=20, campaign_id=campaign_id)

    return jsonify({
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "memories": memories,
        "recent_turns": [
            {
                "role": row["role"],
                "content": row["content"]
            }
            for row in recent_turns
        ]
    })


@app.post("/npc-memory")
def post_npc_memory():
    body = request.get_json(force=True)

    npc_id = body.get("npc_id", "captain_varra")
    session = resolve_request_session(body=body)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    memory = body.get("memory", "").strip()

    if not memory:
        return jsonify({"error": "Missing memory"}), 400

    save_memory_if_new(npc_id, player_id, memory, campaign_id=campaign_id)

    return jsonify({
        "status": "saved",
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "memory": memory
    })


@app.post("/npc-memory/clear")
def clear_npc_memory():
    body = request.get_json(force=True)

    npc_id = body.get("npc_id", "captain_varra")
    session = resolve_request_session(body=body)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    clear_type = body.get("clear_type", "memories").strip().lower()
    # DM console clear buttons omit player_id and mean "this NPC in this campaign"
    # for every speaker. Opt out with all_players=false + an explicit player_id.
    raw_all_players = body.get("all_players", True)
    if isinstance(raw_all_players, str):
        all_players = raw_all_players.strip().lower() in ("1", "true", "yes", "on")
    else:
        all_players = bool(raw_all_players)

    valid_clear_types = ("memories", "turns", "history", "conversation", "dm", "all")
    if clear_type not in valid_clear_types:
        return jsonify({
            "error": "Invalid clear_type. Use memories, turns, history, conversation, dm, or all.",
            "valid_clear_types": list(valid_clear_types)
        }), 400

    deleted_memories = 0
    deleted_turns = 0
    deleted_dm = {}

    if clear_type in ("memories", "all"):
        deleted_memories = clear_memories(
            npc_id,
            player_id,
            campaign_id=campaign_id,
            all_players=all_players,
        )

    if clear_type in ("turns", "history", "conversation", "all"):
        deleted_turns = clear_recent_turns(
            npc_id,
            player_id,
            campaign_id=campaign_id,
            all_players=all_players,
        )

    if clear_type in ("dm", "all"):
        deleted_dm = clear_dm_data(
            npc_id,
            player_id,
            campaign_id=campaign_id,
            all_players=all_players,
        )

    return jsonify({
        "status": "cleared",
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "clear_type": clear_type,
        "all_players": all_players,
        "deleted_memories": deleted_memories,
        "deleted_turns": deleted_turns,
        "deleted_dm": deleted_dm,
    })


init_memory_db()
print(
    "[DM] channel logging enabled — look for lines starting with [DM] "
    "(stages: start, stt, complete)",
    flush=True,
)

# WSGI servers import ``app`` rather than executing this module as __main__.
# Keep the import blocked until every configured Piper replica is actually ready.
warm_piper_at_startup()

if __name__ == "__main__":
    # threaded=True keeps /health, /campaigns, etc. responsive during long voice/LLM turns
    print(
        f"[GAME_API] listening on {GAME_API_HOST}:{GAME_API_PORT} "
        f"(auth={'on' if GAME_API_KEY else 'off'}, "
        f"require_key_mode={'on' if REQUIRE_GAME_API_KEY else 'off'})",
        flush=True,
    )
    app.run(host=GAME_API_HOST, port=GAME_API_PORT, threaded=True)
