"""
Dedicated DM channel: messages, pending verification checks, and automatic ruling detection.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import db_paths

# Reassigned by server.py after game_api.env load; tests may monkeypatch this.
MEMORY_DB = db_paths.MEMORY_DB

# Existing unscoped rows are attributed to this campaign on first migration.
LEGACY_CAMPAIGN_BACKFILL_ID = "greyhaven_intro"

_ALLOWED_SQL_TABLES = frozenset({
    "conversation_turns",
    "npc_memories",
    "dm_messages",
    "dm_pending_checks",
    "dm_rulings",
})


def normalize_campaign_id(campaign_id: str | None) -> str:
    return (campaign_id or "").strip()


def _assert_safe_table(table: str) -> str:
    if table not in _ALLOWED_SQL_TABLES:
        raise ValueError(f"Refusing unsafe SQL table name: {table}")
    return table


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    _assert_safe_table(table)
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_campaign_id_column(
    conn: sqlite3.Connection,
    table: str,
    *,
    backfill: str = LEGACY_CAMPAIGN_BACKFILL_ID,
) -> None:
    """Add campaign_id if missing and backfill legacy empty rows once."""
    _assert_safe_table(table)
    columns = _table_columns(conn, table)
    if "campaign_id" not in columns:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN campaign_id TEXT NOT NULL DEFAULT ''"
        )
        if backfill:
            conn.execute(
                f"UPDATE {table} SET campaign_id = ? WHERE campaign_id = '' OR campaign_id IS NULL",
                (backfill,),
            )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table}_campaign_npc_player "
        f"ON {table} (campaign_id, npc_id, player_id)"
    )

DM_CHECK_TAG_PATTERN = re.compile(
    r"\[DM CHECK:\s*(.+?)\]\s*$",
    re.IGNORECASE | re.DOTALL,
)

POSITIVE_RULING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(yes|yeah|yep|yup|correct|true|confirmed|granted|allowed|approved|affirmative)\b",
        r"\b(that(?:'s| is) (?:true|correct|valid|accurate))\b",
        r"\b(they (?:do|indeed) have)\b",
        r"\b(the party (?:does|do) have)\b",
        r"\b(you (?:may|can) (?:proceed|pass|enter|go ahead))\b",
        r"\b(go ahead)\b",
        r"\b(verdict:\s*confirmed)\b",
        r"\b(ruling:\s*(?:yes|confirmed|true|granted))\b",
    )
]

NEGATIVE_RULING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(no|nope|nah|false|incorrect|denied|rejected|negative|invalid)\b",
        r"\b(that(?:'s| is) (?:not true|false|incorrect|wrong|invalid))\b",
        r"\b(they (?:do not|don't) have)\b",
        r"\b(the party (?:does not|doesn't|do not|don't) have)\b",
        r"\b(not allowed|not authorized|not permitted)\b",
        r"\b(verdict:\s*denied)\b",
        r"\b(ruling:\s*(?:no|denied|false|rejected))\b",
    )
]

PLAYER_CLAIM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(we|i|our party) (?:have|got|carry|brought|were given)\b.+\b(letter|seal|badge|pass|permit|credentials|authorization|authority|order|writ|write|warrant|token|key|invitation|proof|papers|documents?)\b",
        r"\b(we|i|our party) (?:have|got)\b.+\bfrom the (?:mayor|king|queen|lord|captain|guild|guard|magistrate|governor)\b",
        r"\b(the (?:king|queen|lord|captain|guild|mayor|guard|magistrate|governor)) (?:sent|gave|authorized|vouched for|issued)\b",
        r"\b(royal|official|signed|sealed) (?:letter|decree|order|writ|write|pass|papers?)\b",
        r"\b(we are|i am) (?:authorized|permitted|allowed|expected|sent|officially)\b",
        r"\b(here(?:'s| is) (?:my|our) (?:proof|letter|badge|pass|credentials|authorization|writ|write|papers?))\b",
        r"\b(showing you|presenting) (?:my|our|the) (?:proof|letter|badge|pass|credentials|writ|write|papers?)\b",
        r"\b(writ|write|letter|pass|permit|papers?) from the (?:mayor|king|queen|lord|captain|guild|guard|magistrate|governor)\b",
        r"\b(allows?|permitting|authorizing) (?:us|me) to (?:enter|pass|go|access|proceed)\b",
        r"\b(?:mayor|king|queen|lord|captain|guard|magistrate|governor)\b.+\b(writ|write|letter|pass|permit|authorization)\b",
    )
]

# Broader fallback when STT slightly garbles keywords ("write" vs "writ", etc.).
PLAYER_CLAIM_LOOSE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(we|i|our|the party)\b.+\b(writ|write|warrant|letter|pass|permit|badge|seal|proof|papers?|credentials|authorization)\b",
        r"\b(writ|write|warrant|letter|pass|permit)\b.+\b(mayor|king|queen|lord|captain|guard|magistrate|governor)\b",
        r"\b(allows?|permitting|authorized|permission)\b.+\b(enter|pass|access|proceed|inside)\b",
        r"\b(enter|pass|access|proceed)\b.+\b(aqueduct|vault|gate|city|keep|fort|district|building)\b",
    )
]

NPC_DEFERRAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(cannot|can't|could not|won't|will not)\s+(?:simply\s+)?(?:accept|believe|trust|verify|confirm)\b",
        r"\b(need|want|must)\s+(?:the\s+)?(?:dm|dungeon master)\b",
        r"\bneed\s+(?:the\s+)?dm\s+to\s+(?:confirm|verify|rule)\b",
        r"\bneed\s+to\s+(?:see|verify|confirm|check)\b.+\b(proof|writ|write|letter|document|credentials|authorization|seal|badge|pass|papers?)\b",
        r"\b(show|present|produce)\s+(?:me\s+)?(?:your|the)\s+(?:proof|writ|write|letter|document|credentials|papers?)\b",
        r"\b(not\s+(?:so\s+)?(?:sure|certain|convinced))\b.+\b(valid|authentic|legitimate|real|genuine)\b",
        r"\b(suspicious|wary|dubious|question(?:able)?)\b.+\b(writ|write|letter|claim|story|papers?|document)\b",
        r"\bbefore\s+i\s+(?:let|allow|permit|grant|believe)\s+you\b",
        r"\b(?:have|needs?) (?:the\s+)?dm\s+(?:confirm|verify|rule)\b",
        r"\b(?:until|unless)\s+(?:the\s+)?dm\s+((?:has\s+)?confirm|verif|rules?)\b",
        r"\b(?:can't|cannot)\s+(?:simply\s+)?(?:accept|believe|grant|allow)\b",
        r"\bwithout\s+(?:proof|confirmation|verification|the dm)\b",
        r"\bnot until i (?:see|have)\b.+\b(proof|writ|write|letter|document|authorization|permission)\b",
        r"\b(?:can't|cannot)\s+verify\b.+\b(valid|authentic|legitimate|real|genuine|writ|write|letter)\b",
        r"\[DM CHECK:",
    )
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_db():
    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_dm_tables() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                content TEXT NOT NULL,
                detected_verdict TEXT,
                detected_claim TEXT,
                detection_confidence REAL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_pending_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                claim_summary TEXT NOT NULL,
                player_text TEXT,
                npc_text TEXT,
                source TEXT NOT NULL DEFAULT 'npc',
                status TEXT NOT NULL DEFAULT 'pending',
                ruling_id INTEGER,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS dm_rulings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL DEFAULT '',
                npc_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                claim_summary TEXT NOT NULL,
                verdict TEXT NOT NULL,
                ruling_text TEXT NOT NULL,
                source TEXT NOT NULL,
                pending_check_id INTEGER,
                created_at TEXT NOT NULL
            )
        """)

        for table in ("dm_messages", "dm_pending_checks", "dm_rulings"):
            ensure_campaign_id_column(conn, table)

        conn.commit()


def strip_dm_check_tag(text: str) -> tuple[str, str | None]:
    raw = (text or "").strip()
    match = DM_CHECK_TAG_PATTERN.search(raw)
    if not match:
        return raw, None

    spoken = raw[: match.start()].strip()
    claim = match.group(1).strip()
    return spoken, claim or None


def detect_ruling(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {
            "verdict": "unclear",
            "claim": None,
            "confidence": 0.0,
            "matched_phrase": None,
        }

    positive_hits = [p.pattern for p in POSITIVE_RULING_PATTERNS if p.search(content)]
    negative_hits = [p.pattern for p in NEGATIVE_RULING_PATTERNS if p.search(content)]

    claim = _extract_claim_hint(content)

    if positive_hits and not negative_hits:
        return {
            "verdict": "confirmed",
            "claim": claim,
            "confidence": min(1.0, 0.55 + 0.15 * len(positive_hits)),
            "matched_phrase": positive_hits[0],
        }

    if negative_hits and not positive_hits:
        return {
            "verdict": "denied",
            "claim": claim,
            "confidence": min(1.0, 0.55 + 0.15 * len(negative_hits)),
            "matched_phrase": negative_hits[0],
        }

    if positive_hits and negative_hits:
        return {
            "verdict": "unclear",
            "claim": claim,
            "confidence": 0.35,
            "matched_phrase": None,
        }

    return {
        "verdict": "unclear",
        "claim": claim,
        "confidence": 0.0,
        "matched_phrase": None,
    }


def detect_player_claim(text: str) -> dict[str, Any] | None:
    content = (text or "").strip()
    if not content:
        return None

    for pattern in PLAYER_CLAIM_PATTERNS:
        match = pattern.search(content)
        if match:
            return {
                "claim_summary": _summarize_claim(content),
                "matched_phrase": match.group(0).strip(),
                "confidence": 0.7,
                "detection_tier": "strong",
            }

    for pattern in PLAYER_CLAIM_LOOSE_PATTERNS:
        match = pattern.search(content)
        if match:
            return {
                "claim_summary": _summarize_claim(content),
                "matched_phrase": match.group(0).strip(),
                "confidence": 0.55,
                "detection_tier": "loose",
            }

    return None


def detect_npc_deferral(text: str) -> dict[str, Any] | None:
    content = (text or "").strip()
    if not content:
        return None

    for pattern in NPC_DEFERRAL_PATTERNS:
        match = pattern.search(content)
        if match:
            return {
                "matched_phrase": match.group(0).strip(),
                "confidence": 0.75,
            }

    return None


def infer_claim_summary(player_text: str, npc_text: str | None = None) -> str:
    detected = detect_player_claim(player_text)
    if detected:
        return detected["claim_summary"]

    content = (player_text or "").strip()
    if content:
        return _summarize_claim(content)

    if npc_text:
        return _summarize_claim(npc_text)

    return "Player claim requiring DM verification"


def _extract_claim_hint(text: str) -> str | None:
    content = text.strip()

    for pattern in (
        r"\b(?:regarding|about|that|whether)\s+(.{8,180})",
        r"\b(?:they have|the party has)\s+(.{8,180})",
    ):
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!?:;")

    return _summarize_claim(content)


def _summarize_claim(text: str, max_len: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _normalize_claim(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _claims_match(left: str, right: str) -> bool:
    a = _normalize_claim(left)
    b = _normalize_claim(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True

    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return False

    overlap = len(a_words & b_words)
    return overlap >= min(3, min(len(a_words), len(b_words)))


AUTHORIZATION_TOPIC_WORDS = (
    "writ", "write", "warrant", "letter", "pass", "permit", "badge", "seal",
    "proof", "authorized", "authorization", "permission", "mayor", "mare",
    "king", "queen", "lord", "captain", "guard", "aqueduct", "access", "enter",
)


def _topic_word_hits(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {word for word in AUTHORIZATION_TOPIC_WORDS if word in lowered}


def _claims_same_exchange(
    left: str,
    right: str,
    *,
    player_text: str | None = None,
) -> bool:
    if _claims_match(left, right):
        return True

    left_topics = _topic_word_hits(left)
    right_topics = _topic_word_hits(right)
    if left_topics and right_topics and len(left_topics & right_topics) >= 2:
        return True

    if player_text:
        player_topics = _topic_word_hits(player_text)
        if player_topics:
            if len(player_topics & left_topics) >= 2 and len(player_topics & right_topics) >= 1:
                return True
            if len(player_topics & right_topics) >= 2 and len(player_topics & left_topics) >= 1:
                return True

    return False


def _find_related_open_pending(
    npc_id: str,
    player_id: str,
    claim_summary: str,
    player_text: str | None = None,
    *,
    campaign_id: str = "",
) -> dict[str, Any] | None:
    pending_items = load_pending_checks(
        npc_id, player_id, status="pending", campaign_id=campaign_id
    )
    if not pending_items:
        return None

    for item in reversed(pending_items):
        if _claims_same_exchange(
            item["claim_summary"],
            claim_summary,
            player_text=player_text or item.get("player_text"),
        ):
            return item

        if player_text and item.get("player_text"):
            if _normalize_claim(item["player_text"]) == _normalize_claim(player_text):
                return item

    return None


def update_pending_check(
    pending_check_id: int,
    *,
    claim_summary: str | None = None,
    player_text: str | None = None,
    npc_text: str | None = None,
    source: str | None = None,
) -> int:
    fields = []
    values = []

    if claim_summary is not None:
        fields.append("claim_summary = ?")
        values.append(claim_summary.strip())
    if player_text is not None:
        fields.append("player_text = ?")
        values.append(player_text)
    if npc_text is not None:
        fields.append("npc_text = ?")
        values.append(npc_text)
    if source is not None:
        fields.append("source = ?")
        values.append(source)

    if not fields:
        return pending_check_id

    values.append(pending_check_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE dm_pending_checks SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        conn.commit()

    return pending_check_id


def save_pending_check(
    *,
    npc_id: str,
    player_id: str,
    claim_summary: str,
    player_text: str | None = None,
    npc_text: str | None = None,
    source: str = "npc",
    merge_pending_id: int | None = None,
    campaign_id: str = "",
) -> int | None:
    campaign_id = normalize_campaign_id(campaign_id)
    claim_summary = (claim_summary or "").strip()
    if not claim_summary:
        return None

    if merge_pending_id is not None:
        existing = next(
            (
                pending for pending in load_pending_checks(
                    npc_id, player_id, status="pending", campaign_id=campaign_id
                )
                if pending["id"] == merge_pending_id
            ),
            None,
        )
        merged_source = source
        if existing and existing.get("source") and existing["source"] != source:
            merged_source = f"{existing['source']}+{source}"

        update_pending_check(
            merge_pending_id,
            claim_summary=claim_summary,
            player_text=player_text,
            npc_text=npc_text,
            source=merged_source,
        )
        print(
            f"[DM] merged pending check #{merge_pending_id} npc={npc_id} player={player_id} "
            f"campaign={campaign_id or '(none)'} source={source} claim={claim_summary[:120]!r}",
            flush=True,
        )
        return merge_pending_id

    for pending in load_pending_checks(
        npc_id, player_id, status="pending", campaign_id=campaign_id
    ):
        if _claims_same_exchange(
            pending["claim_summary"],
            claim_summary,
            player_text=player_text or pending.get("player_text"),
        ):
            return update_pending_check(
                pending["id"],
                claim_summary=claim_summary,
                player_text=player_text,
                npc_text=npc_text,
                source=source if pending.get("source") == source else f"{pending.get('source')}+{source}",
            )

    related = _find_related_open_pending(
        npc_id, player_id, claim_summary, player_text, campaign_id=campaign_id
    )
    if related:
        merged_source = source if related.get("source") == source else f"{related.get('source')}+{source}"
        return update_pending_check(
            related["id"],
            claim_summary=claim_summary,
            player_text=player_text or related.get("player_text"),
            npc_text=npc_text,
            source=merged_source,
        )

    created_at = _utc_now_iso()
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO dm_pending_checks
                (campaign_id, npc_id, player_id, claim_summary, player_text, npc_text, source, status, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            campaign_id,
            npc_id,
            player_id,
            claim_summary,
            player_text,
            npc_text,
            source,
            created_at,
        ))
        conn.commit()
        pending_id = cur.lastrowid

    print(
        f"[DM] pending check #{pending_id} npc={npc_id} player={player_id} "
        f"campaign={campaign_id or '(none)'} source={source} claim={claim_summary[:120]!r}",
        flush=True,
    )
    return pending_id


def resolve_pending_check(
    pending_check_id: int,
    *,
    verdict: str,
    ruling_id: int,
) -> None:
    with get_db() as conn:
        conn.execute("""
            UPDATE dm_pending_checks
            SET status = ?, ruling_id = ?, resolved_at = ?
            WHERE id = ?
        """, (verdict, ruling_id, _utc_now_iso(), pending_check_id))
        conn.commit()


def _find_matching_pending(
    npc_id: str,
    player_id: str,
    claim_hint: str | None,
    *,
    campaign_id: str = "",
) -> dict[str, Any] | None:
    pending_items = load_pending_checks(
        npc_id, player_id, status="pending", campaign_id=campaign_id
    )
    if not pending_items:
        return None

    if claim_hint:
        for item in reversed(pending_items):
            if _claims_match(item["claim_summary"], claim_hint):
                return item

    return pending_items[-1]


def save_ruling(
    *,
    npc_id: str,
    player_id: str,
    claim_summary: str,
    verdict: str,
    ruling_text: str,
    source: str,
    pending_check_id: int | None = None,
    campaign_id: str = "",
) -> int:
    campaign_id = normalize_campaign_id(campaign_id)
    created_at = _utc_now_iso()
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO dm_rulings
                (campaign_id, npc_id, player_id, claim_summary, verdict, ruling_text, source, pending_check_id, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            campaign_id,
            npc_id,
            player_id,
            claim_summary,
            verdict,
            ruling_text,
            source,
            pending_check_id,
            created_at,
        ))
        conn.commit()
        ruling_id = cur.lastrowid

    if pending_check_id:
        resolve_pending_check(pending_check_id, verdict=verdict, ruling_id=ruling_id)

    return ruling_id


def save_dm_message(
    *,
    npc_id: str,
    player_id: str,
    content: str,
    verdict_override: str | None = None,
    campaign_id: str = "",
) -> dict[str, Any]:
    campaign_id = normalize_campaign_id(campaign_id)
    content = (content or "").strip()
    if not content:
        raise ValueError("Missing DM message content")

    detection = detect_ruling(content)
    verdict = (verdict_override or detection["verdict"] or "unclear").strip().lower()
    if verdict not in ("confirmed", "denied", "unclear"):
        raise ValueError("verdict must be confirmed, denied, or unclear")

    created_at = _utc_now_iso()
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO dm_messages
                (campaign_id, npc_id, player_id, content, detected_verdict, detected_claim, detection_confidence, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            campaign_id,
            npc_id,
            player_id,
            content,
            verdict,
            detection.get("claim"),
            float(detection.get("confidence") or 0.0),
            created_at,
        ))
        conn.commit()
        message_id = cur.lastrowid

    ruling_id = None
    pending_check_id = None
    claim_summary = detection.get("claim")

    if verdict in ("confirmed", "denied"):
        pending = _find_matching_pending(
            npc_id, player_id, claim_summary, campaign_id=campaign_id
        )
        if pending:
            pending_check_id = pending["id"]
            if not claim_summary:
                claim_summary = pending["claim_summary"]

        if not claim_summary:
            claim_summary = _summarize_claim(content)

        ruling_id = save_ruling(
            npc_id=npc_id,
            player_id=player_id,
            claim_summary=claim_summary,
            verdict=verdict,
            ruling_text=content,
            source="manual" if verdict_override else "auto",
            pending_check_id=pending_check_id,
            campaign_id=campaign_id,
        )

    return {
        "id": message_id,
        "campaign_id": campaign_id,
        "npc_id": npc_id,
        "player_id": player_id,
        "content": content,
        "detected_verdict": verdict,
        "detected_claim": claim_summary,
        "detection_confidence": detection.get("confidence"),
        "detection": detection,
        "ruling_id": ruling_id,
        "pending_check_id": pending_check_id,
        "created_at": created_at,
    }


def load_dm_messages(
    npc_id: str,
    player_id: str,
    limit: int = 50,
    *,
    campaign_id: str = "",
) -> list[dict[str, Any]]:
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, content, detected_verdict, detected_claim, detection_confidence, created_at
            FROM dm_messages
            WHERE campaign_id = ?
              AND npc_id = ?
              AND player_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (campaign_id, npc_id, player_id, limit)).fetchall()

    return [
        {
            "id": row["id"],
            "content": row["content"],
            "detected_verdict": row["detected_verdict"],
            "detected_claim": row["detected_claim"],
            "detection_confidence": row["detection_confidence"],
            "created_at": row["created_at"],
        }
        for row in reversed(rows)
    ]


def load_pending_checks(
    npc_id: str,
    player_id: str,
    *,
    status: str | None = "pending",
    limit: int = 20,
    campaign_id: str = "",
) -> list[dict[str, Any]]:
    campaign_id = normalize_campaign_id(campaign_id)
    query = """
        SELECT id, claim_summary, player_text, npc_text, source, status, ruling_id, created_at, resolved_at
        FROM dm_pending_checks
        WHERE campaign_id = ?
          AND npc_id = ?
          AND player_id = ?
    """
    params: list[Any] = [campaign_id, npc_id, player_id]

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    items = [
        {
            "id": row["id"],
            "claim_summary": row["claim_summary"],
            "player_text": row["player_text"],
            "npc_text": row["npc_text"],
            "source": row["source"],
            "status": row["status"],
            "ruling_id": row["ruling_id"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }
        for row in rows
    ]
    return list(reversed(items))


def load_rulings(
    npc_id: str,
    player_id: str,
    limit: int = 20,
    *,
    campaign_id: str = "",
) -> list[dict[str, Any]]:
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, claim_summary, verdict, ruling_text, source, pending_check_id, created_at
            FROM dm_rulings
            WHERE campaign_id = ?
              AND npc_id = ?
              AND player_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (campaign_id, npc_id, player_id, limit)).fetchall()

    return [
        {
            "id": row["id"],
            "claim_summary": row["claim_summary"],
            "verdict": row["verdict"],
            "ruling_text": row["ruling_text"],
            "source": row["source"],
            "pending_check_id": row["pending_check_id"],
            "created_at": row["created_at"],
        }
        for row in reversed(rows)
    ]


def load_ruling_by_id(ruling_id: int) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("""
            SELECT id, campaign_id, npc_id, player_id, claim_summary, verdict, ruling_text, source, pending_check_id, created_at
            FROM dm_rulings
            WHERE id = ?
        """, (ruling_id,)).fetchone()

    if not row:
        return None

    return {
        "id": row["id"],
        "campaign_id": row["campaign_id"] if "campaign_id" in row.keys() else "",
        "npc_id": row["npc_id"],
        "player_id": row["player_id"],
        "claim_summary": row["claim_summary"],
        "verdict": row["verdict"],
        "ruling_text": row["ruling_text"],
        "source": row["source"],
        "pending_check_id": row["pending_check_id"],
        "created_at": row["created_at"],
    }


def get_dm_context(npc_id: str, player_id: str, campaign_id: str = "") -> dict[str, Any]:
    campaign_id = normalize_campaign_id(campaign_id)
    return {
        "campaign_id": campaign_id,
        "messages": load_dm_messages(npc_id, player_id, limit=10, campaign_id=campaign_id),
        "pending_checks": load_pending_checks(
            npc_id, player_id, status="pending", limit=10, campaign_id=campaign_id
        ),
        "rulings": load_rulings(npc_id, player_id, limit=10, campaign_id=campaign_id),
    }


def format_dm_context_for_prompt(dm_context: dict[str, Any] | None) -> str:
    dm_context = dm_context or {}
    sections = []

    rulings = dm_context.get("rulings") or []
    if rulings:
        lines = []
        for ruling in rulings[-8:]:
            label = ruling["verdict"].upper()
            lines.append(
                f"- [{label}] {ruling['claim_summary']} — DM said: {ruling['ruling_text']}"
            )
        sections.append("Active DM rulings (authoritative; the NPC must obey these):\n" + "\n".join(lines))

    pending = dm_context.get("pending_checks") or []
    if pending:
        lines = [f"- {item['claim_summary']}" for item in pending[-5:]]
        sections.append(
            "Pending DM verification (not yet ruled; NPC must not grant these claims yet):\n"
            + "\n".join(lines)
        )

    messages = dm_context.get("messages") or []
    recent_notes = [
        msg for msg in messages[-5:]
        if (msg.get("detected_verdict") or "unclear") == "unclear"
    ]
    if recent_notes:
        lines = [f"- {msg['content']}" for msg in recent_notes]
        sections.append("Recent DM channel notes:\n" + "\n".join(lines))

    if not sections:
        return "No DM rulings or pending checks are on record for this scene."

    return "\n\n".join(sections)


def build_dm_injection_messages(dm_context: dict[str, Any] | None) -> list[dict[str, str]]:
    dm_context = dm_context or {}
    injections = []

    for ruling in (dm_context.get("rulings") or [])[-6:]:
        verdict = ruling["verdict"].upper()
        injections.append({
            "role": "user",
            "content": (
                f"[DM RULING - {verdict}] {ruling['claim_summary']}. "
                f"DM said: {ruling['ruling_text']}"
            ),
        })

    for msg in (dm_context.get("messages") or [])[-4:]:
        verdict = (msg.get("detected_verdict") or "unclear").lower()
        if verdict in ("confirmed", "denied"):
            continue
        injections.append({
            "role": "user",
            "content": f"[DM NOTE] {msg['content']}",
        })

    return injections


def process_player_message(
    npc_id: str,
    player_id: str,
    player_text: str,
    *,
    campaign_id: str = "",
) -> dict[str, Any] | None:
    detected = detect_player_claim(player_text)
    if not detected:
        return None

    pending_id = save_pending_check(
        npc_id=npc_id,
        player_id=player_id,
        claim_summary=detected["claim_summary"],
        player_text=player_text,
        source="player",
        campaign_id=campaign_id,
    )

    return {
        "pending_check_id": pending_id,
        **detected,
    }


def process_npc_response(
    npc_id: str,
    player_id: str,
    player_text: str,
    npc_text: str,
    *,
    player_pending_id: int | None = None,
    campaign_id: str = "",
) -> dict[str, Any]:
    spoken_text, dm_check_claim = strip_dm_check_tag(npc_text)
    pending_id = None
    pending_source = None
    merge_pending_id = player_pending_id

    if dm_check_claim:
        pending_id = save_pending_check(
            npc_id=npc_id,
            player_id=player_id,
            claim_summary=dm_check_claim,
            player_text=player_text,
            npc_text=spoken_text,
            source="npc",
            merge_pending_id=merge_pending_id,
            campaign_id=campaign_id,
        )
        pending_source = "npc_tag"
    else:
        deferral = detect_npc_deferral(npc_text)
        player_claim = detect_player_claim(player_text)
        if deferral and (player_claim or _looks_like_authorization_exchange(player_text)):
            claim_summary = infer_claim_summary(player_text, spoken_text)
            pending_id = save_pending_check(
                npc_id=npc_id,
                player_id=player_id,
                claim_summary=claim_summary,
                player_text=player_text,
                npc_text=spoken_text,
                source="npc_deferral",
                merge_pending_id=merge_pending_id,
                campaign_id=campaign_id,
            )
            pending_source = "npc_deferral"

    return {
        "spoken_text": spoken_text,
        "raw_text": npc_text,
        "dm_check_claim": dm_check_claim,
        "pending_check_id": pending_id,
        "pending_source": pending_source,
        "merged_player_pending_id": player_pending_id if pending_id == player_pending_id else None,
        "npc_deferral_detected": bool(detect_npc_deferral(npc_text)),
    }


def _looks_like_authorization_exchange(player_text: str) -> bool:
    content = (player_text or "").strip()
    if not content:
        return False

    if detect_player_claim(content):
        return True

    authorization_words = (
        "writ", "write", "warrant", "letter", "pass", "permit", "badge", "seal",
        "proof", "authorized", "permission", "mayor", "king", "queen", "lord",
        "captain", "guard", "enter", "access", "allow", "stand aside", "let us",
    )
    lowered = content.lower()
    hits = sum(1 for word in authorization_words if word in lowered)
    return hits >= 2


def create_manual_ruling(
    *,
    npc_id: str,
    player_id: str,
    claim_summary: str,
    verdict: str,
    notes: str = "",
    pending_check_id: int | None = None,
    campaign_id: str = "",
) -> dict[str, Any]:
    campaign_id = normalize_campaign_id(campaign_id)
    verdict = verdict.strip().lower()
    if verdict not in ("confirmed", "denied"):
        raise ValueError("verdict must be confirmed or denied")

    claim_summary = (claim_summary or "").strip()
    if not claim_summary:
        raise ValueError("Missing claim_summary")

    ruling_text = (notes or "").strip() or f"DM ruling: {verdict} for {claim_summary}"

    if pending_check_id is None:
        pending = _find_matching_pending(
            npc_id, player_id, claim_summary, campaign_id=campaign_id
        )
        if pending:
            pending_check_id = pending["id"]

    ruling_id = save_ruling(
        npc_id=npc_id,
        player_id=player_id,
        claim_summary=claim_summary,
        verdict=verdict,
        ruling_text=ruling_text,
        source="manual",
        pending_check_id=pending_check_id,
        campaign_id=campaign_id,
    )

    created_at = _utc_now_iso()
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO dm_messages
                (campaign_id, npc_id, player_id, content, detected_verdict, detected_claim, detection_confidence, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            campaign_id,
            npc_id,
            player_id,
            ruling_text,
            verdict,
            claim_summary,
            1.0,
            created_at,
        ))
        conn.commit()
        message_id = cur.lastrowid

    return {
        "ruling_id": ruling_id,
        "message_id": message_id,
        "campaign_id": campaign_id,
        "claim_summary": claim_summary,
        "verdict": verdict,
        "ruling_text": ruling_text,
        "pending_check_id": pending_check_id,
        "created_at": created_at,
    }


def log_dm_turn(
    *,
    request_id: str,
    route: str,
    npc_id: str,
    player_id: str,
    stage: str,
    player_text: str | None = None,
    player_claim: dict | None = None,
    npc_response: dict | None = None,
    campaign_id: str = "",
) -> None:
    campaign_id = normalize_campaign_id(campaign_id)
    parts = [
        "[DM]",
        f"route={route}",
        f"stage={stage}",
        f"request_id={request_id}",
        f"npc={npc_id}",
        f"player={player_id}",
        f"campaign={campaign_id or '(none)'}",
    ]

    if player_text is not None:
        parts.append(f"transcript={player_text!r}")

    if player_claim:
        parts.append("player_claim=True")
        parts.append(f"claim_tier={player_claim.get('detection_tier', 'unknown')}")
    elif stage == "complete":
        parts.append("player_claim=False")

    if npc_response:
        parts.append(f"npc_deferral={npc_response.get('npc_deferral_detected')}")
        parts.append(f"pending_source={npc_response.get('pending_source')}")
        parts.append(f"pending_check_id={npc_response.get('pending_check_id')}")

    if stage in ("complete", "summary"):
        pending = load_pending_checks(
            npc_id, player_id, status="pending", limit=50, campaign_id=campaign_id
        )
        parts.append(f"pending_count={len(pending)}")

    print(" ".join(str(part) for part in parts), flush=True)


def clear_dm_data(npc_id: str, player_id: str, campaign_id: str = "") -> dict[str, int]:
    campaign_id = normalize_campaign_id(campaign_id)
    with get_db() as conn:
        deleted_messages = conn.execute("""
            DELETE FROM dm_messages
            WHERE campaign_id = ? AND npc_id = ? AND player_id = ?
        """, (campaign_id, npc_id, player_id)).rowcount

        deleted_pending = conn.execute("""
            DELETE FROM dm_pending_checks
            WHERE campaign_id = ? AND npc_id = ? AND player_id = ?
        """, (campaign_id, npc_id, player_id)).rowcount

        deleted_rulings = conn.execute("""
            DELETE FROM dm_rulings
            WHERE campaign_id = ? AND npc_id = ? AND player_id = ?
        """, (campaign_id, npc_id, player_id)).rowcount

        conn.commit()

    return {
        "deleted_messages": deleted_messages,
        "deleted_pending_checks": deleted_pending,
        "deleted_rulings": deleted_rulings,
    }
