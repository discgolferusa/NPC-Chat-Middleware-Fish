from flask import Flask, request, jsonify, send_file, make_response
import os
import secrets
import yaml
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid
import subprocess
import wave
import threading
import time
import shutil

from faster_whisper import WhisperModel

from llm_router import (
    generate_llm_response,
    get_llm_models_payload,
    get_llm_status,
    reset_usage_tracking,
    select_llm_model,
)
from dm_channel import (
    build_dm_injection_messages,
    clear_dm_data,
    create_manual_ruling,
    detect_ruling,
    ensure_campaign_id_column,
    format_dm_context_for_prompt,
    get_dm_context,
    init_dm_tables,
    load_dm_messages,
    load_pending_checks,
    load_ruling_by_id,
    load_rulings,
    log_dm_turn,
    normalize_campaign_id,
    process_npc_response,
    process_player_message,
    save_dm_message,
    strip_dm_check_tag,
)
from party_campaign import (
    DEFAULT_PLAYER_ID,
    build_party_context,
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
    resolve_request_session,
    select_campaign,
    select_character,
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
    complete_turn_progress,
    fail_turn_progress,
    get_turn_progress,
    get_turn_progress_catalog,
    new_request_id,
    register_turn_progress,
    set_turn_progress,
)
from server_settings import format_llm_rules, get_narrator_tts_settings, load_global_llm_rules

BASE_DIR = Path(__file__).resolve().parent

NPC_DIR = BASE_DIR / "npcs"
LOCATION_DIR = BASE_DIR / "locations"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

MEMORY_DB = BASE_DIR / "npc_memory.db"

PARTIES_DIR = BASE_DIR / "parties"
CAMPAIGNS_DIR = BASE_DIR / "campaigns"


NPC_DIR.mkdir(exist_ok=True)
LOCATION_DIR.mkdir(exist_ok=True)
PARTIES_DIR.mkdir(exist_ok=True)
CAMPAIGNS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

PIPER_EXE = Path(r"C:\AI\npc-server\.venv\Scripts\piper.exe")
DEFAULT_VOICE_MODEL = Path(r"C:\AI\Voices\en\en_US\ryan\medium\en_US-ryan-medium.onnx")
VOICE_PREVIEW_TEXT = os.environ.get(
    "VOICE_PREVIEW_TEXT",
    "Mary had a little lamb, whose fleece was white as snow. "
    "And everywhere that Mary went, the lamb was sure to go.",
)
SEGMENT_PAUSE_SECONDS = 0.18
PIPER_SENTENCE_SILENCE = 0.2

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

# Good first choice. If accuracy is poor, change to "small.en".
# If latency is too high, change to "tiny.en".
STT_MODEL = WhisperModel("base.en", device="cpu", compute_type="int8")

app = Flask(__name__)

GAME_API_ENV_FILE = BASE_DIR / "game_api.env"


def _load_game_api_env_file() -> None:
    """Load GAME_API_* from game_api.env when not already set in the shell."""
    if not GAME_API_ENV_FILE.exists():
        return

    try:
        for line in GAME_API_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as ex:
        print(f"[GAME_API] failed to read env file: {ex}", flush=True)


_load_game_api_env_file()
if GAME_API_ENV_FILE.exists():
    print(f"[GAME_API] loaded settings from {GAME_API_ENV_FILE.name}", flush=True)


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
_VOICE_CACHE: dict[str, Any] = {
    "built_at": 0.0,
    "voices": [],
    "by_name": {},
}
_VOICE_CACHE_TTL_SECONDS = float(os.environ.get("VOICES_CACHE_TTL_SECONDS", "300"))


def _build_voice_inventory() -> tuple[list[dict[str, str]], dict[str, Path]]:
    if not VOICES_DIR.exists():
        return [], {}

    voices: list[dict[str, str]] = []
    by_name: dict[str, Path] = {}
    seen: set[str] = set()

    # Avoid Path.resolve() here — on UNC/network trees it can be very expensive.
    for path in sorted(VOICES_DIR.rglob("*.onnx"), key=lambda item: str(item).lower()):
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)

        try:
            label = path.relative_to(VOICES_DIR).as_posix()
        except ValueError:
            label = path.name

        voices.append({"label": label, "path": str(path)})
        name_key = path.name.lower()
        by_name.setdefault(name_key, path)

    return voices, by_name


def refresh_voice_inventory(*, force: bool = False) -> list[dict[str, str]]:
    now = time.perf_counter()
    with _VOICE_CACHE_LOCK:
        age = now - float(_VOICE_CACHE.get("built_at") or 0.0)
        if not force and _VOICE_CACHE["voices"] and age < _VOICE_CACHE_TTL_SECONDS:
            return list(_VOICE_CACHE["voices"])

    voices, by_name = _build_voice_inventory()
    with _VOICE_CACHE_LOCK:
        _VOICE_CACHE["built_at"] = time.perf_counter()
        _VOICE_CACHE["voices"] = voices
        _VOICE_CACHE["by_name"] = by_name
        return list(voices)


def list_piper_voices() -> list[dict[str, str]]:
    return refresh_voice_inventory(force=False)


def lookup_cached_voice_by_name(filename: str) -> Path | None:
    name = Path(filename).name.lower()
    if not name:
        return None

    refresh_voice_inventory(force=False)
    with _VOICE_CACHE_LOCK:
        path = _VOICE_CACHE["by_name"].get(name)
        return path if isinstance(path, Path) else None


if VOICES_DIR.exists():
    _startup_voices = refresh_voice_inventory(force=True)
    print(
        f"[VOICES] Piper scan root: {VOICES_DIR} ({len(_startup_voices)} model(s))",
        flush=True,
    )
else:
    print(f"[VOICES] Piper scan root not found: {VOICES_DIR}", flush=True)

# Alpha / Funnel: bind localhost only; require Bearer token when GAME_API_KEY is set.
GAME_API_KEY = os.environ.get("GAME_API_KEY", "").strip()
REQUIRE_GAME_API_KEY = os.environ.get("REQUIRE_GAME_API_KEY", "").strip().lower() in ("1", "true", "yes")
GAME_API_HOST = os.environ.get("GAME_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
GAME_API_PORT = int(os.environ.get("GAME_API_PORT", "5000"))
AUTH_EXEMPT_PATHS = frozenset({"/health"})


def _extract_game_api_key():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Game-Api-Key", "").strip()


@app.before_request
def require_game_api_key():
    if request.method == "OPTIONS":
        return None

    if REQUIRE_GAME_API_KEY and request.path == "/llm/reset-usage":
        return jsonify({"error": "Not found"}), 404

    if GAME_API_KEY and request.path not in AUTH_EXEMPT_PATHS:
        provided = _extract_game_api_key()
        if not provided or not secrets.compare_digest(provided, GAME_API_KEY):
            print(
                f"[AUTH] rejected request to {request.path} from {request.remote_addr}",
                flush=True,
            )
            return jsonify({"error": "Unauthorized"}), 401

    return None


def log_timing(request_id: str, label: str, start_time: float):
    elapsed = time.perf_counter() - start_time
    print(f"[TIMING] {request_id} {label}: {elapsed:.3f}s", flush=True)
    return time.perf_counter()


def get_db():
    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS npc_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                memory TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        ensure_campaign_id_column(conn, "conversation_turns")
        ensure_campaign_id_column(conn, "npc_memories")

        conn.commit()

    init_dm_tables()


def save_turn(npc_id: str, player_id: str, role: str, content: str, campaign_id: str = ""):
    campaign_id = normalize_campaign_id(campaign_id)
    content = (content or "").strip()
    if not content:
        return

    with get_db() as conn:
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
        conn.commit()


def load_recent_turns(npc_id: str, player_id: str, limit: int = 10, campaign_id: str = ""):
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
        rows = conn.execute("""
            SELECT role, content
            FROM conversation_turns
            WHERE campaign_id = ?
              AND npc_id = ?
              AND player_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (campaign_id, npc_id, player_id, limit)).fetchall()

    return list(reversed(rows))


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


def load_memories(npc_id: str, player_id: str, limit: int = 20, campaign_id: str = ""):
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
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


def clear_memories(npc_id: str, player_id: str, campaign_id: str = "") -> int:
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
        cur = conn.execute("""
            DELETE FROM npc_memories
            WHERE campaign_id = ?
              AND npc_id = ?
              AND player_id = ?
        """, (campaign_id, npc_id, player_id))
        conn.commit()
        return cur.rowcount


def clear_recent_turns(npc_id: str, player_id: str, campaign_id: str = "") -> int:
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
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

    with CACHE_LOCK:
        cache[cache_key] = {
            "modified_at": modified_at,
            "data": data
        }

    return data


def load_npc(npc_id: str) -> dict:
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


def format_list_items(items, formatter=None) -> str:
    if not items:
        return ""

    lines = []
    for item in items:
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


def format_location_context(location: dict) -> str:
    if not location:
        return "No specific local setting context is loaded."

    if location.get("error"):
        return f"Location context warning: {location['error']}"

    name = location.get("name", location.get("id", "Unknown location"))
    location_type = location.get("type", "")
    summary = location.get("summary", "")

    current_events = format_list_items(location.get("current_events", []))
    important_locations = format_list_items(location.get("important_locations", []))
    notable_npcs = format_list_items(location.get("notable_npcs", []))
    rumors = format_list_items(location.get("rumors", []))
    local_tone = format_list_items(location.get("local_tone", []))
    knowledge_rules = format_list_items(location.get("knowledge_rules", []))

    return f"""
Location name: {name}
Location type: {location_type}

Summary:
{summary}

Current local events:
{current_events}

Important local places:
{important_locations}

Notable local NPCs:
{notable_npcs}

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
) -> dict:
    """
    Load all context pieces used for an NPC turn.

    Static YAML pieces are mtime-cached. Memories/recent turns stay live because they
    change during play.
    """
    npc = load_npc(npc_id)
    location = load_location(npc.get("location_id"))
    party_context = party_context or build_party_context(player_id=player_id)
    resolved_campaign_id = normalize_campaign_id(
        campaign_id if campaign_id is not None else party_context.get("campaign_id")
    )
    memories = load_memories(
        npc_id, player_id, limit=MEMORY_PROMPT_LIMIT, campaign_id=resolved_campaign_id
    )
    recent_turns = load_recent_turns(
        npc_id, player_id, limit=RECENT_TURN_LIMIT, campaign_id=resolved_campaign_id
    )
    scene_context = build_scene_context(npc.get("location_id"))

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
        "dm_context": get_dm_context(npc_id, player_id, campaign_id=resolved_campaign_id),
    }

    with CACHE_LOCK:
        PRELOADED_CONTEXTS[(resolved_campaign_id, npc_id, player_id)] = context

    return context


def preload_npc_context(npc_id: str, player_id: str = DEFAULT_PLAYER_ID) -> dict:
    return get_npc_context(npc_id, player_id)


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

    for path in sorted(list(NPC_DIR.glob("*.yaml")) + list(NPC_DIR.glob("*.yml"))):
        npc_id = path.stem

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
                "error": str(ex)
            })

    return jsonify({
        "npcs": npcs
    })



def get_npc_pronoun_context(npc: dict) -> dict:
    """
    Return normalized NPC gender/pronoun context for prompt grounding.

    Existing NPC YAML files do not need these fields; old files fall back to
    neutral they/them phrasing until explicit values are added.
    """
    gender = (npc.get("gender") or npc.get("identity") or "unspecified").strip()
    pronouns = npc.get("pronouns", {}) or {}

    # Allow either a structured YAML block or a simple string like "he/him".
    if isinstance(pronouns, str):
        raw = pronouns.strip()
        lowered = raw.lower()

        if lowered in ("he/him", "he/him/his"):
            pronouns = {
                "subject": "he",
                "object": "him",
                "possessive_adjective": "his",
                "possessive_pronoun": "his",
                "reflexive": "himself",
            }
        elif lowered in ("she/her", "she/her/hers"):
            pronouns = {
                "subject": "she",
                "object": "her",
                "possessive_adjective": "her",
                "possessive_pronoun": "hers",
                "reflexive": "herself",
            }
        elif lowered in ("they/them", "they/them/theirs"):
            pronouns = {
                "subject": "they",
                "object": "them",
                "possessive_adjective": "their",
                "possessive_pronoun": "theirs",
                "reflexive": "themself",
            }
        else:
            pronouns = {"display": raw}

    subject = pronouns.get("subject", "they")
    obj = pronouns.get("object", "them")
    possessive_adjective = pronouns.get("possessive_adjective", "their")
    possessive_pronoun = pronouns.get("possessive_pronoun", "theirs")
    reflexive = pronouns.get("reflexive", "themself")

    display = pronouns.get("display")
    if not display:
        display = f"{subject}/{obj}/{possessive_adjective}/{possessive_pronoun}/{reflexive}"

    return {
        "gender": gender,
        "subject": subject,
        "object": obj,
        "possessive_adjective": possessive_adjective,
        "possessive_pronoun": possessive_pronoun,
        "reflexive": reflexive,
        "display": display,
    }


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
) -> str:
    public_facts = "\n".join(f"- {x}" for x in npc.get("public_facts", []))
    known_facts = "\n".join(f"- {x}" for x in npc.get("known_facts", []))
    secrets = "\n".join(f"- {x}" for x in npc.get("secrets", []))
    memories_text = "\n".join(f"- {x}" for x in (memories or []))
    location_context = format_location_context(location or {})
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

NPC-specific rules for {npc["name"]}:
{npc_rules_text}"""

    return f"""
You are portraying an NPC in a tabletop RPG.

NPC name: {npc["name"]}
Role: {npc["role"]}
Gender/identity: {pronoun_context["gender"]}
Pronouns: {pronoun_context["display"]}
Pronoun details: subject={pronoun_context["subject"]}, object={pronoun_context["object"]}, possessive adjective={pronoun_context["possessive_adjective"]}, possessive pronoun={pronoun_context["possessive_pronoun"]}, reflexive={pronoun_context["reflexive"]}
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
{location_context}

Authoritative scene context:
{scene_context_text}

Optional scene notes from request:
{optional_scene_block}

Party and campaign context:
{party_context_text}

DM channel context:
{dm_context_text}

Rules:
{global_rules_text}{npc_rules_block}
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


def _json_body_flag(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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
            "The DM denied the claim. Refuse the request in character without granting access or benefit."
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
            "Do not mention the DM, pending checks, or game mechanics."
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
    campaign_id: str = "",
) -> dict:
    verdict = (ruling.get("verdict") or "").strip().lower()
    if verdict not in ("confirmed", "denied"):
        return {
            "generated": False,
            "reason": "no_ruling_verdict",
        }

    request_id = uuid.uuid4().hex

    try:
        campaign_id = normalize_campaign_id(
            campaign_id or (party_context or {}).get("campaign_id") or ruling.get("campaign_id")
        )
        context = get_npc_context(
            npc_id, player_id, party_context=party_context, campaign_id=campaign_id
        )
        npc = context["npc"]
        memories = context["memories"]
        recent_turns = context["recent_turns"]
        location = context["location"]
        campaign_id = context["campaign_id"]
        dm_context = get_dm_context(npc_id, player_id, campaign_id=campaign_id)
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

        npc_text_raw = ask_llm(messages)
        npc_response = process_npc_response(
            npc_id, player_id, "", npc_text_raw, campaign_id=campaign_id
        )
        npc_text = npc_response["spoken_text"]

        if not npc_text:
            npc_text = "Very well."

        save_turn(npc_id, player_id, "assistant", npc_text, campaign_id=campaign_id)

        tts_segments = split_narration_and_dialogue(npc_text)
        tts_text = " ".join(segment["text"] for segment in tts_segments).strip()

        audio_url = None
        audio_filename = None

        if include_audio:
            audio_filename = f"{request_id}_dm_followup.wav"
            output_path = OUTPUT_DIR / audio_filename
            synthesize_segmented_speech(
                npc_text=npc_text,
                npc_voice_model=pick_npc_voice_model(npc, voice_model),
                output_path=output_path,
                npc_length_scale=pick_npc_length_scale(npc, tts_length_scale),
            )
            audio_url = f"/dm/followup-audio/{audio_filename}"

        print(
            f"[DM] auto-response request_id={request_id} npc={npc_id} player={player_id} "
            f"verdict={verdict} chars={len(npc_text)} audio={bool(audio_url)}",
            flush=True,
        )

        return {
            "generated": True,
            "request_id": request_id,
            "npc_id": npc_id,
            "npc_name": npc.get("name", npc_id),
            "player_id": player_id,
            "ruling_id": ruling.get("id"),
            "verdict": verdict,
            "npc_text": npc_text,
            "tts_text": tts_text,
            "tts_segments": tts_segments,
            "audio_url": audio_url,
            "audio_filename": audio_filename,
        }

    except Exception as ex:
        print(
            f"[DM] auto-response failed request_id={request_id} npc={npc_id} error={ex!r}",
            flush=True,
        )
        return {
            "generated": False,
            "request_id": request_id,
            "reason": "generation_failed",
            "error": str(ex),
        }


def ask_llm(messages):
    return generate_llm_response(
        messages,
        temperature=0.80,
        max_tokens=180,
        top_p=0.90,
        stream=False,
    )


def transcribe_audio(audio_path: Path) -> str:
    segments, info = STT_MODEL.transcribe(
        str(audio_path),
        language="en",
        beam_size=1,
        vad_filter=True
    )

    text_parts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            text_parts.append(text)

    return " ".join(text_parts).strip()


def clean_for_tts(text: str) -> str:
    # Keep stage direction content, but remove markdown wrappers.
    # Example: "*Varra eyes you suspiciously.*" -> "Varra eyes you suspiciously."
    text = re.sub(r"\*([^*]{1,500})\*", r"\1", text)
    text = re.sub(r"_([^_]{1,500})_", r"\1", text)

    # Keep bracket/parenthesis content too, but remove wrappers.
    text = re.sub(r"\[([^\]]{1,500})\]", r"\1", text)
    text = re.sub(r"\(([^)]{1,500})\)", r"\1", text)

    # Remove any leftover TTS-problem characters.
    text = text.replace("*", "")
    text = text.replace("_", "")
    text = text.replace("`", "")
    text = text.replace("~", "")

    # Clean up whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text




def split_narration_and_dialogue(text: str) -> list[dict]:
    """
    Split raw model output into at most two TTS segments:

    1. One narrator segment containing all asterisk-wrapped stage directions.
    2. One NPC segment containing all spoken dialogue.

    This prevents expensive NPC -> narrator -> NPC multi-segment TTS.
    """
    raw_text, _ = strip_dm_check_tag(text or "")

    stage_directions = []
    dialogue_parts = []

    pattern = re.compile(r"\*([^*]{1,800})\*")
    last_end = 0

    for match in pattern.finditer(raw_text):
        before = raw_text[last_end:match.start()].strip()
        if before:
            cleaned_before = clean_for_tts(before)
            if cleaned_before:
                dialogue_parts.append(cleaned_before)

        stage = match.group(1).strip()
        if stage:
            cleaned_stage = clean_for_tts(stage)
            if cleaned_stage:
                stage_directions.append(cleaned_stage)

        last_end = match.end()

    after = raw_text[last_end:].strip()
    if after:
        cleaned_after = clean_for_tts(after)
        if cleaned_after:
            dialogue_parts.append(cleaned_after)

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

    if not segments:
        fallback = clean_for_tts(raw_text)
        if fallback:
            segments.append({
                "type": "npc",
                "text": fallback
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


def convert_wav_to_standard_format(input_path: Path, output_path: Path) -> None:
    """
    Convert a WAV file to standard Piper-friendly PCM.

    Output is 22.05 kHz mono 16-bit PCM WAV. The Pi can still up-convert
    the final response to 48 kHz stereo for playback. Keeping this close to
    common Piper output avoids static caused by odd playback expectations.
    """
    ffmpeg_exe = find_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-vn",
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


def synthesize_segmented_speech(
    npc_text: str,
    npc_voice_model: str | None,
    output_path: Path,
    npc_length_scale: float = 1.15
) -> list[dict]:
    """
    Speak stage directions with the global narrator voice and dialogue with the NPC voice.

    Each raw Piper segment is converted to the same WAV format before concatenation,
    so different narrator/NPC voice models can be mixed safely.
    """
    segments = split_narration_and_dialogue(npc_text)

    if not segments:
        segments = [{"type": "npc", "text": "I have nothing more to say."}]

    temp_paths = []
    standardized_paths = []
    narrator_tts = get_narrator_tts_settings()

    try:
        for index, segment in enumerate(segments):
            raw_segment_path = output_path.with_name(
                f"{output_path.stem}_raw_segment_{index:02d}.wav"
            )
            standard_segment_path = output_path.with_name(
                f"{output_path.stem}_segment_{index:02d}.wav"
            )

            temp_paths.append(raw_segment_path)
            temp_paths.append(standard_segment_path)
            standardized_paths.append(standard_segment_path)

            if segment["type"] == "narrator":
                synthesize_speech(
                    segment["text"],
                    narrator_tts["resolved_voice_model"],
                    raw_segment_path,
                    narrator_tts["length_scale"],
                )
            else:
                synthesize_speech(
                    segment["text"],
                    npc_voice_model,
                    raw_segment_path,
                    npc_length_scale
                )

            convert_wav_to_standard_format(
                raw_segment_path,
                standard_segment_path
            )

        if len(standardized_paths) == 1:
            output_path.write_bytes(standardized_paths[0].read_bytes())
        else:
            concatenate_wav_files(standardized_paths, output_path)

        return segments

    finally:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


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


def pick_npc_length_scale(npc: dict, override: Any = None) -> float:
    if override is not None and str(override).strip() != "":
        try:
            return float(override)
        except (TypeError, ValueError):
            pass
    try:
        return float(npc.get("tts_length_scale", 1.15))
    except (TypeError, ValueError):
        return 1.15


def synthesize_speech(
    text: str,
    voice_model: str | None,
    output_path: Path,
    length_scale: float = 1.15
) -> None:
    # Same resolution as /voices/preview so UNC/partial/editor paths work at runtime.
    model_path = resolve_voice_model_path(voice_model)

    if not PIPER_EXE.exists():
        raise FileNotFoundError(f"Piper executable not found: {PIPER_EXE}")

    if not model_path.exists():
        raise FileNotFoundError(f"Piper voice model not found: {model_path}")

    subprocess.run(
        [
            str(PIPER_EXE),
            "--model", str(model_path),
            "--length_scale", str(length_scale),
            "--sentence_silence", str(PIPER_SENTENCE_SILENCE),
            "--output_file", str(output_path)
        ],
        input=text,
        text=True,
        encoding="utf-8",
        check=True
    )


def resolve_voice_model_path(voice_model: str | None) -> Path:
    if not voice_model or not str(voice_model).strip():
        return DEFAULT_VOICE_MODEL

    raw = str(voice_model).strip()
    model_path = Path(raw)
    if model_path.exists():
        return model_path

    normalized = raw.replace("\\", "/")
    if normalized.startswith("//"):
        unc = Path(raw)
        if unc.exists():
            return unc

    candidate = VOICES_DIR / normalized
    if candidate.exists():
        return candidate

    # Match by filename using the cached inventory (no per-request full-tree rglob).
    target_name = Path(normalized).name.lower()
    cached = lookup_cached_voice_by_name(target_name)
    if cached is not None and cached.exists():
        return cached

    raise FileNotFoundError(f"Piper voice model not found: {voice_model}")


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
    session = resolve_request_session(body=body)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    party_context = session["party_context"]

    request_id = new_request_id(body.get("request_id"))
    total_start = time.perf_counter()
    step_start = total_start

    register_turn_progress(
        request_id,
        route="/npc-turn",
        npc_id=npc_id,
        player_id=player_id,
        mode="text",
    )
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
        context = get_npc_context(npc_id, player_id, party_context=party_context, campaign_id=campaign_id)
        npc = context["npc"]
        memories = context["memories"]
        location = context["location"]
        dm_context = context["dm_context"]
        scene_context = context["scene_context"]
        step_start = log_timing(request_id, "load npc context", step_start)

        player_claim = process_player_message(npc_id, player_id, player_text, campaign_id=campaign_id)
        dm_context = get_dm_context(npc_id, player_id, campaign_id=campaign_id)

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
        )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        print(
            f"[TIMING] {request_id} text prompt messages={len(messages)} chars={prompt_chars}",
            flush=True
        )
        step_start = log_timing(request_id, "build text prompt/messages", step_start)

        set_turn_progress(request_id, "waiting_llm")
        npc_text_raw = ask_llm(messages)
        step_start = log_timing(request_id, "llm text npc response", step_start)

        set_turn_progress(request_id, "processing_response")
        npc_response = process_npc_response(
            npc_id,
            player_id,
            player_text,
            npc_text_raw,
            player_pending_id=player_claim.get("pending_check_id") if player_claim else None,
                campaign_id=campaign_id,
            )
        npc_text = npc_response["spoken_text"]

        save_turn(npc_id, player_id, "user", player_text, campaign_id=campaign_id)
        save_turn(npc_id, player_id, "assistant", npc_text, campaign_id=campaign_id)
        step_start = log_timing(request_id, "save text conversation turns", step_start)

        saved_memories = []
        memory_extraction_mode = "disabled"

        if AUTO_MEMORY_EXTRACTION_ENABLED and AUTO_MEMORY_EXTRACTION_ASYNC:
            memory_extraction_mode = "async"
            run_memory_extraction_background(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=npc_text,
                campaign_id=campaign_id,
            )
        elif AUTO_MEMORY_EXTRACTION_ENABLED:
            memory_extraction_mode = "sync"
            saved_memories = save_extracted_memories(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=npc_text,
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
        )

        set_turn_progress(request_id, "finishing")
        complete_turn_progress(request_id)

        return jsonify({
            "request_id": request_id,
            "npc_id": npc_id,
            "npc_name": npc["name"],
            "player_id": player_id,
            "party_id": session["party_id"],
            "campaign_id": session["campaign_id"],
            "speaking_member": party_context.get("speaking_member"),
            "party_members": party_context.get("party_members"),
            "text": npc_text,
            "saved_memories": saved_memories,
            "memory_extraction_mode": memory_extraction_mode,
            "dm_check_claim": npc_response.get("dm_check_claim"),
            "dm_pending_check_id": npc_response.get("pending_check_id"),
            "detected_player_claim": player_claim,
            "dm_pending": load_pending_checks(npc_id, player_id, status="pending", limit=5, campaign_id=campaign_id),
        })

    except Exception as ex:
        fail_turn_progress(request_id, str(ex))
        log_dm_turn(
            request_id=request_id,
            route="/npc-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="error",
        )
        return jsonify({
            "error": str(ex),
            "request_id": request_id,
            "npc_id": npc_id,
        }), 500


@app.post("/npc-voice-turn")
def npc_voice_turn():
    npc_id = request.form.get("npc_id", "captain_varra")
    scene_description = request.form.get("scene", "")
    session = resolve_request_session(form=request.form)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    party_context = session["party_context"]

    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file field named 'audio'"}), 400

    request_id = new_request_id(request.form.get("request_id"))
    total_start = time.perf_counter()
    step_start = total_start

    register_turn_progress(
        request_id,
        route="/npc-voice-turn",
        npc_id=npc_id,
        player_id=player_id,
        mode="voice",
    )
    set_turn_progress(request_id, "received")

    log_dm_turn(
        request_id=request_id,
        route="/npc-voice-turn",
        npc_id=npc_id,
        player_id=player_id,
        stage="start",
    )

    input_path = UPLOAD_DIR / f"{request_id}.wav"
    output_path = OUTPUT_DIR / f"{request_id}_response.wav"

    request.files["audio"].save(input_path)
    step_start = log_timing(request_id, "save uploaded audio", step_start)

    try:
        set_turn_progress(request_id, "transcribing")
        player_text = transcribe_audio(input_path)
        step_start = log_timing(request_id, "speech-to-text", step_start)

        if not player_text:
            fail_turn_progress(request_id, "No speech detected")
            return jsonify({"error": "No speech detected", "request_id": request_id}), 400

        log_dm_turn(
            request_id=request_id,
            route="/npc-voice-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="stt",
            player_text=player_text,
        )

        set_turn_progress(request_id, "loading_context")
        context = get_npc_context(npc_id, player_id, party_context=party_context, campaign_id=campaign_id)
        npc = context["npc"]
        memories = context["memories"]
        recent_turns = context["recent_turns"]
        location = context["location"]
        dm_context = context["dm_context"]
        scene_context = context["scene_context"]
        step_start = log_timing(request_id, "load npc/location/memory/recent turns", step_start)

        player_claim = process_player_message(npc_id, player_id, player_text, campaign_id=campaign_id)
        dm_context = get_dm_context(npc_id, player_id, campaign_id=campaign_id)

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
        )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        print(
            f"[TIMING] {request_id} prompt messages={len(messages)} chars={prompt_chars}",
            flush=True
        )
        step_start = log_timing(request_id, "build prompt/messages", step_start)

        set_turn_progress(request_id, "waiting_llm")
        npc_text_raw = ask_llm(messages)
        step_start = log_timing(request_id, "llm npc response", step_start)

        set_turn_progress(request_id, "processing_response")
        npc_response = process_npc_response(
            npc_id,
            player_id,
            player_text,
            npc_text_raw,
            player_pending_id=player_claim.get("pending_check_id") if player_claim else None,
                campaign_id=campaign_id,
            )
        npc_text = npc_response["spoken_text"]

        tts_segments = split_narration_and_dialogue(npc_text)
        tts_text = " ".join(segment["text"] for segment in tts_segments).strip()

        print(
            f"[TIMING] {request_id} tts segments={len(tts_segments)} "
            f"types={[s['type'] for s in tts_segments]}",
            flush=True
        )
        step_start = log_timing(request_id, "split tts segments", step_start)

        if not tts_text:
            npc_text = "I have nothing more to say."
            tts_text = npc_text
            tts_segments = [{"type": "npc", "text": tts_text}]

        save_turn(npc_id, player_id, "user", player_text, campaign_id=campaign_id)
        save_turn(npc_id, player_id, "assistant", npc_text, campaign_id=campaign_id)
        step_start = log_timing(request_id, "save conversation turns", step_start)

        memory_extraction_mode = "disabled"
        if AUTO_MEMORY_EXTRACTION_ENABLED and AUTO_MEMORY_EXTRACTION_ASYNC:
            memory_extraction_mode = "async"
            run_memory_extraction_background(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "schedule async memory extraction", step_start)
        elif AUTO_MEMORY_EXTRACTION_ENABLED:
            memory_extraction_mode = "sync"
            save_extracted_memories(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "sync memory extraction/save", step_start)

        set_turn_progress(request_id, "generating_voice")
        synthesize_segmented_speech(
            npc_text=npc_text,
            npc_voice_model=pick_npc_voice_model(
                npc,
                (request.form.get("voice_model") or "").strip() or None,
            ),
            output_path=output_path,
            npc_length_scale=pick_npc_length_scale(
                npc,
                request.form.get("tts_length_scale"),
            ),
        )
        step_start = log_timing(request_id, "piper segmented tts", step_start)

        total_elapsed = time.perf_counter() - total_start
        print(
            f"[TIMING] {request_id} TOTAL: {total_elapsed:.3f}s "
            f"memory_mode={memory_extraction_mode}",
            flush=True
        )

        pending_items = load_pending_checks(npc_id, player_id, status="pending", limit=10, campaign_id=campaign_id)
        log_dm_turn(
            request_id=request_id,
            route="/npc-voice-turn",
            npc_id=npc_id,
            player_id=player_id,
            stage="complete",
            player_text=player_text,
            player_claim=player_claim,
            npc_response=npc_response,
        )

        set_turn_progress(request_id, "finishing")
        complete_turn_progress(request_id)

        response = make_response(send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="npc_response.wav"
        ))
        response.headers["X-Turn-Request-Id"] = request_id
        response.headers["X-DM-Pending-Count"] = str(len(pending_items))
        pending_id = npc_response.get("pending_check_id") or (
            player_claim.get("pending_check_id") if player_claim else None
        )
        if pending_id:
            response.headers["X-DM-Pending-Check-Id"] = str(pending_id)
        if player_claim:
            response.headers["X-DM-Player-Claim-Detected"] = "1"
        if npc_response.get("npc_deferral_detected"):
            response.headers["X-DM-Npc-Deferral-Detected"] = "1"
        return response

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
        fail_turn_progress(request_id, str(ex))
        return jsonify({
            "error": str(ex),
            "request_id": request_id,
            "npc_id": npc_id
        }), 500


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

    input_path = UPLOAD_DIR / f"{request_id}.wav"
    output_path = OUTPUT_DIR / f"{request_id}_response.wav"

    request.files["audio"].save(input_path)
    step_start = log_timing(request_id, "debug save uploaded audio", step_start)

    try:
        player_text = transcribe_audio(input_path)
        step_start = log_timing(request_id, "debug speech-to-text", step_start)

        if not player_text:
            return jsonify({"error": "No speech detected"}), 400

        log_dm_turn(
            request_id=request_id,
            route="/npc-voice-turn-debug",
            npc_id=npc_id,
            player_id=player_id,
            stage="stt",
            player_text=player_text,
        )

        context = get_npc_context(npc_id, player_id, party_context=party_context, campaign_id=campaign_id)
        npc = context["npc"]
        memories = context["memories"]
        recent_turns = context["recent_turns"]
        location = context["location"]
        dm_context = context["dm_context"]
        scene_context = context["scene_context"]
        step_start = log_timing(request_id, "debug load npc/location/memory/recent turns", step_start)

        player_claim = process_player_message(npc_id, player_id, player_text, campaign_id=campaign_id)
        dm_context = get_dm_context(npc_id, player_id, campaign_id=campaign_id)

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

        npc_text_raw = ask_llm(messages)
        step_start = log_timing(request_id, "debug llm npc response", step_start)

        npc_response = process_npc_response(
            npc_id,
            player_id,
            player_text,
            npc_text_raw,
            player_pending_id=player_claim.get("pending_check_id") if player_claim else None,
                campaign_id=campaign_id,
            )
        npc_text = npc_response["spoken_text"]

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

        save_turn(npc_id, player_id, "user", player_text, campaign_id=campaign_id)
        save_turn(npc_id, player_id, "assistant", npc_text, campaign_id=campaign_id)
        step_start = log_timing(request_id, "debug save conversation turns", step_start)

        saved_memories = []
        memory_extraction_mode = "disabled"
        if AUTO_MEMORY_EXTRACTION_ENABLED and AUTO_MEMORY_EXTRACTION_ASYNC:
            memory_extraction_mode = "async"
            run_memory_extraction_background(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "debug schedule async memory extraction", step_start)
        elif AUTO_MEMORY_EXTRACTION_ENABLED:
            memory_extraction_mode = "sync"
            saved_memories = save_extracted_memories(
                npc=npc,
                npc_id=npc_id,
                player_id=player_id,
                player_text=player_text,
                npc_text=npc_text,
                campaign_id=campaign_id,
            )
            step_start = log_timing(request_id, "debug sync memory extraction/save", step_start)

        synthesize_segmented_speech(
            npc_text=npc_text,
            npc_voice_model=pick_npc_voice_model(
                npc,
                (request.form.get("voice_model") or "").strip() or None,
            ),
            output_path=output_path,
            npc_length_scale=pick_npc_length_scale(
                npc,
                request.form.get("tts_length_scale"),
            ),
        )
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
            "tts_text": tts_text,
            "tts_segments": tts_segments,
            "location_id": npc.get("location_id", ""),
            "location_name": location.get("name", location.get("id", "")) if isinstance(location, dict) else "",
            "prompt_messages": len(messages),
            "prompt_characters": prompt_chars,
            "saved_memories": saved_memories,
            "memory_extraction_mode": memory_extraction_mode,
            "audio_file": str(output_path),
            "total_seconds": round(total_elapsed, 3),
            "dm_check_claim": npc_response.get("dm_check_claim"),
            "dm_pending_check_id": npc_response.get("pending_check_id"),
            "detected_player_claim": player_claim,
            "dm_pending": load_pending_checks(npc_id, player_id, status="pending", limit=5, campaign_id=campaign_id),
        })

    except Exception as ex:
        total_elapsed = time.perf_counter() - total_start
        print(f"[TIMING] {request_id} DEBUG ERROR after {total_elapsed:.3f}s: {ex}", flush=True)
        return jsonify({
            "error": str(ex),
            "npc_id": npc_id
        }), 500


@app.get("/locations")
def list_locations():
    locations = []

    for path in sorted(list(LOCATION_DIR.glob("*.yaml")) + list(LOCATION_DIR.glob("*.yml"))):
        location_id = path.stem

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
                "error": str(ex)
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
        return jsonify({
            "status": "error",
            "npc_id": npc_id,
            "player_id": player_id,
            "error": str(ex)
        }), 500


@app.get("/llm/models")
def llm_models():
    return jsonify(get_llm_models_payload())


@app.get("/llm/status")
def llm_status():
    return jsonify(get_llm_status())


@app.post("/llm/select")
def llm_select():
    body = request.get_json(force=True, silent=True) or {}

    model_id = body.get("model_id") or body.get("model")
    provider_mode = body.get("provider_mode")

    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400

    try:
        result = select_llm_model(model_id, provider_mode=provider_mode)
        result["status"] = get_llm_status()
        return jsonify(result)
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[LLM] select failed: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 500


@app.post("/llm/reset-usage")
def llm_reset_usage():
    # LAN-only convenience endpoint; do not expose this middleware publicly.
    reset_usage_tracking()
    return jsonify({"status": "reset", "usage": get_llm_status().get("usage", {})})


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
        return jsonify({"error": str(ex)}), 400


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
        return jsonify({"error": str(ex)}), 400


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
        return jsonify({"error": str(ex)}), 400


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

    return jsonify(register_turn_progress(
        request_id,
        route=route,
        npc_id=npc_id,
        player_id=player_id,
        mode=mode,
    ))


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
    body = request.get_json(force=True, silent=True) or {}
    campaign_id = (body.get("campaign_id") or body.get("id") or "").strip()

    if not campaign_id:
        return jsonify({"error": "Missing campaign_id"}), 400

    try:
        result = select_campaign(campaign_id)
        return jsonify({"status": "selected", **result})
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[SESSION] campaign select failed: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 500


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
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[SESSION] party select failed: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 500


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
    limit = int(request.args.get("limit", 50))

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
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[DM] message post failed: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 500


@app.get("/dm/pending")
def dm_pending():
    npc_id = request.args.get("npc_id", "captain_varra")
    session = resolve_request_session(args=request.args)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    status = request.args.get("status", "pending")

    return jsonify({
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "pending_checks": load_pending_checks(npc_id, player_id, status=status or None, limit=50, campaign_id=campaign_id),
    })


@app.get("/dm/rulings")
def dm_rulings():
    npc_id = request.args.get("npc_id", "captain_varra")
    session = resolve_request_session(args=request.args)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    limit = int(request.args.get("limit", 20))

    return jsonify({
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "rulings": load_rulings(npc_id, player_id, limit=limit, campaign_id=campaign_id),
    })


@app.post("/dm/ruling")
def dm_ruling_post():
    body = request.get_json(force=True, silent=True) or {}

    npc_id = body.get("npc_id", "captain_varra")
    session = resolve_request_session(body=body)
    player_id = session["player_id"]
    campaign_id = session["campaign_id"]
    claim_summary = (body.get("claim_summary") or body.get("claim") or "").strip()
    verdict = (body.get("verdict") or "").strip().lower()
    notes = (body.get("notes") or body.get("text") or "").strip()
    pending_check_id = body.get("pending_check_id")

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
                    party_context=session["party_context"],
                    voice_model=(body.get("voice_model") or "").strip() or None,
                    tts_length_scale=body.get("tts_length_scale"),
                campaign_id=campaign_id,
            )

        return jsonify({
            "status": "saved",
            "npc_id": npc_id,
            "player_id": player_id,
            "party_id": session["party_id"],
            "campaign_id": session["campaign_id"],
            "speaking_member": session["party_context"].get("speaking_member"),
            **result,
            "auto_response": auto_response,
            "pending": load_pending_checks(npc_id, player_id, status="pending", limit=10, campaign_id=campaign_id),
            "rulings": load_rulings(npc_id, player_id, limit=10, campaign_id=campaign_id),
        })
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[DM] ruling post failed: {ex}", flush=True)
        return jsonify({"error": str(ex)}), 500


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
    force_refresh = (request.args.get("refresh") or "").strip().lower() in ("1", "true", "yes")
    voices = refresh_voice_inventory(force=force_refresh)
    return jsonify({
        "voices": voices,
        "voice_enabled": True,
        "voices_dir": str(VOICES_DIR),
        "cached": not force_refresh,
    })


@app.post("/voices/preview")
def preview_voice():
    body = request.get_json(force=True, silent=True) or {}
    voice_path = (body.get("voice_path") or body.get("voice_model") or "").strip()
    if not voice_path:
        return jsonify({"error": "Missing voice_path"}), 400

    text = (body.get("text") or VOICE_PREVIEW_TEXT).strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        length_scale = float(body.get("length_scale", 1.15))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid length_scale"}), 400

    raw_path = OUTPUT_DIR / f"{uuid.uuid4().hex}_voice_preview_raw.wav"
    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}_voice_preview.wav"
    try:
        model_path = resolve_voice_model_path(voice_path)
        synthesize_speech(text, str(model_path), raw_path, length_scale)
        convert_wav_to_standard_format(raw_path, output_path)
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="voice_preview.wav",
        )
    except FileNotFoundError as ex:
        return jsonify({"error": str(ex)}), 404
    except subprocess.CalledProcessError as ex:
        return jsonify({"error": f"Piper synthesis failed: {ex}"}), 500
    except Exception as ex:
        print(f"[VOICE] preview failed: {ex!r}", flush=True)
        return jsonify({"error": str(ex)}), 500
    finally:
        raw_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "routes": [
            "/npc-turn",
            "/npc-voice-turn",
            "/npc-voice-turn-debug",
            "/npc-memory",
            "/npc-memory/clear",
            "/npcs",
            "/locations",
            "/npc-preload",
            "/llm/models",
            "/llm/status",
            "/llm/select",
            "/llm/reset-usage",
            "/session/status",
            "/scene/status",
            "/scene/time",
            "/scene/time/clear",
            "/scene/sub-location",
            "/scene/sub-location/clear",
            "/scene/sub-locations",
            "/turn/progress/steps",
            "/turn/progress/register",
            "/turn/progress/<request_id>",
            "/campaigns",
            "/campaign/active",
            "/campaign/select",
            "/parties",
            "/party/characters",
            "/party/active",
            "/party/select",
            "/dm/status",
            "/dm/messages",
            "/dm/message",
            "/dm/pending",
            "/dm/rulings",
            "/dm/ruling",
            "/dm/detect",
            "/dm/followup-audio/<filename>",
            "/voices",
            "/voices/preview",
        ]
    })


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
        deleted_memories = clear_memories(npc_id, player_id, campaign_id=campaign_id)

    if clear_type in ("turns", "history", "conversation", "all"):
        deleted_turns = clear_recent_turns(npc_id, player_id, campaign_id=campaign_id)

    if clear_type in ("dm", "all"):
        deleted_dm = clear_dm_data(npc_id, player_id, campaign_id=campaign_id)

    return jsonify({
        "status": "cleared",
        "npc_id": npc_id,
        "player_id": player_id,
        "party_id": session["party_id"],
        "campaign_id": session["campaign_id"],
        "clear_type": clear_type,
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


if __name__ == "__main__":
    # threaded=True keeps /health, /campaigns, etc. responsive during long voice/LLM turns
    print(
        f"[GAME_API] listening on {GAME_API_HOST}:{GAME_API_PORT} "
        f"(auth={'on' if GAME_API_KEY else 'off'}, "
        f"require_key_mode={'on' if REQUIRE_GAME_API_KEY else 'off'})",
        flush=True,
    )
    app.run(host=GAME_API_HOST, port=GAME_API_PORT, threaded=True)
