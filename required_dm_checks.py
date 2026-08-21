"""
DM CHECK handling: LLM judges; platform parses tags.

Enforcement is intentionally narrow:
1. Parse ``[DM CHECK: …]`` from model output (see dm_channel.extract_dm_check_tags).
2. If NPC llm_rules author a literal ``[DM CHECK: …]`` template and no matching
   DM ruling exists yet, ensure that exact tag is present on the reply.
3. Do not regex-guess player intent to invent server-side checks — global llm_rules
   tell the model when to emit tags.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable

from stage_directions import normalize_stage_directions

DM_CHECK_IN_RULES_PATTERN = re.compile(
    r"\[DM CHECK:\s*([^\]]+?)\]",
    re.IGNORECASE,
)

# YAML colon-split can leave fragments like: [DM CHECK': 'claim text]
DM_CHECK_MANGLED_IN_RULES_PATTERN = re.compile(
    r"\[DM\s*CHECK['\"\s]*:\s*['\"]?\s*([^\]]+?)\s*['\"]?\s*\]",
    re.IGNORECASE,
)

DM_RULING_FAILURE_PATTERN = re.compile(
    r"\[DM\s*Ruling:\s*Failure\]\s*(.+)",
    re.IGNORECASE,
)

_STAGE_DIRECTION_PATTERN = re.compile(r"^\*[^*]+\*\s*")

# Claim must be about speaking/comprehending a language — not generic "understand X".
_LANGUAGE_BARRIER_CLAIM_PHRASES = (
    "necessary language",
    "know the language",
    "knows the language",
    "know language",
    "speak the language",
    "speaks the language",
    "speak language",
    "language to talk",
    "language to speak",
)

_LANGUAGE_BARRIER_CLAIM_PATTERN = re.compile(
    r"language\s+([a-z][a-z\s-]{2,})",
    re.IGNORECASE,
)

# "Slaadi language", "Common language" (name before the word language).
_NAMED_LANGUAGE_BEFORE_PATTERN = re.compile(
    r"\b([a-z][a-z-]{2,})\s+language\b",
    re.IGNORECASE,
)

# "know the Slaadi language", "speak the Common language"
_KNOW_SPEAK_NAMED_LANGUAGE_PATTERN = re.compile(
    r"\b(?:know|knows|speak|speaks)\s+the\s+[a-z][a-z-]+\s+language\b",
    re.IGNORECASE,
)

# Leading words to strip from a captured language name (not the language itself).
_LANGUAGE_NAME_LEADING_FILLER = frozenset(
    {"is", "are", "was", "were", "the", "a", "an"},
)

_NATIVE_LANGUAGE_PATTERNS = (
    re.compile(r"native language\s*:\s*([a-z][a-z\s-]{2,})", re.IGNORECASE),
    re.compile(r"native language\s+is\s+([a-z][a-z\s-]{2,})", re.IGNORECASE),
    re.compile(r"speaks only ([a-z][a-z\s-]{2,})", re.IGNORECASE),
    re.compile(r"speak only ([a-z][a-z\s-]{2,})", re.IGNORECASE),
    re.compile(r"only speaks ([a-z][a-z\s-]{2,})", re.IGNORECASE),
    re.compile(r"language is ([a-z][a-z\s-]{2,})", re.IGNORECASE),
)

# Words that are not a creature language name when parsed from a claim.
_LANGUAGE_NAME_STOPWORDS = frozenset(
    {
        "the", "to", "talk", "speak", "npc", "necessary", "required",
        "understood", "unknown", "player", "speaker", "party",
    }
)

_ENGLISH_COMPREHENSION_WORDS = frozenset(
    {
        "i", "you", "we", "they", "he", "she", "it", "the", "a", "an", "is", "are", "was", "were",
        "can", "could", "will", "would", "have", "has", "had", "do", "does", "did",
        "yes", "no", "hello", "hi", "help", "please", "thank", "thanks", "what", "how", "why",
        "who", "where", "when", "understand", "understood", "speak", "english", "sure", "of", "course",
        "may", "might", "must", "shall", "should", "want", "need", "know", "think", "believe",
        "your", "my", "our", "their", "this", "that", "these", "those", "here", "there",
        "well", "very", "good", "great", "fine", "okay", "ok", "right", "indeed", "certainly",
        "welcome", "sorry", "excuse", "pardon", "tell", "say", "said", "ask", "answer",
        "perfectly", "clearly", "absolutely", "however", "perhaps", "maybe", "come", "go",
    }
)


def _npc_rules_blob(npc: dict[str, Any] | None) -> str:
    from yaml_list_text import yaml_list_item_as_text

    npc = npc or {}
    raw = npc.get("llm_rules")
    if raw is None:
        raw = npc.get("npc_rules")

    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(
            text for text in (yaml_list_item_as_text(item) for item in raw if item is not None) if text.strip()
        )
    if isinstance(raw, dict):
        nested = raw.get("llm_rules") or raw.get("rules") or []
        if isinstance(nested, list):
            return "\n".join(
                text
                for text in (yaml_list_item_as_text(item) for item in nested if item is not None)
                if text.strip()
            )
        return str(nested)
    return str(raw)


def _normalize_claim(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_dm_check_templates_from_rules(npc: dict[str, Any] | None) -> list[str]:
    """Unique DM CHECK claim strings authored inside NPC llm_rules."""
    blob = _npc_rules_blob(npc)
    claims: list[str] = []
    seen: set[str] = set()

    def _add_claim(claim: str) -> None:
        claim = (claim or "").strip()
        if not claim:
            return
        key = _normalize_claim(claim)
        if not key or key in seen:
            return
        seen.add(key)
        claims.append(claim)

    for match in DM_CHECK_IN_RULES_PATTERN.finditer(blob):
        # Preserve authored punctuation (possessives, trailing quotes in the claim).
        _add_claim(match.group(1))

    for match in DM_CHECK_MANGLED_IN_RULES_PATTERN.finditer(blob):
        # Only mangled YAML fragments may wrap the claim in leftover ' / " chars.
        _add_claim(match.group(1).strip().strip("'\""))

    return claims


_CLAIM_STOPWORDS = frozenset(
    {
        "a", "an", "the", "to", "of", "and", "or", "for", "is", "are", "was", "were",
        "be", "been", "being", "do", "does", "did", "this", "that", "these", "those",
        "it", "its", "in", "on", "at", "as", "by", "with", "from", "into", "about",
        "than", "then", "so", "if", "not", "no", "yes", "any", "all", "some",
    }
)


def _content_claim_tokens(text: str) -> set[str]:
    return {
        w for w in _normalize_claim(text).split()
        if len(w) >= 4 and w not in _CLAIM_STOPWORDS
    }


def _claim_matches(rule_claim: str, other: str) -> bool:
    """
    Match an authored template to a DM claim_summary / emitted tag.

    Equality and long substring only for exact-ish text. Near-paraphrase uses
    content tokens (stopwords excluded). Overlap thresholds scale with claim
    length so short templates (e.g. secrets probing — three content tokens)
    can still match slight rewordings.
    """
    a = _normalize_claim(rule_claim)
    b = _normalize_claim(other)
    if not a or not b:
        return False
    if a == b:
        return True

    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 20 and shorter in longer:
        return True

    ca = _content_claim_tokens(a)
    cb = _content_claim_tokens(b)
    if not ca or not cb:
        return False

    overlap = ca & cb
    shorter_n = min(len(ca), len(cb))
    # Scale: short claims need (almost) all tokens; long ones need ~90% with a floor.
    if shorter_n <= 3:
        needed = shorter_n
        distinctive_needed = 1
    elif shorter_n <= 5:
        needed = max(3, int(0.9 * shorter_n + 0.999))
        distinctive_needed = 2
    else:
        needed = max(4, int(0.9 * shorter_n + 0.999))
        distinctive_needed = 2

    if len(overlap) < needed:
        return False
    if sum(1 for w in overlap if len(w) >= 6) < distinctive_needed:
        return False
    return True


def _ruling_covers_claim(dm_context: dict[str, Any] | None, claim: str) -> bool:
    """A ruling covers a template only via claim_summary identity, not ruling_text."""
    for ruling in (dm_context or {}).get("rulings") or []:
        verdict = str(ruling.get("verdict") or "").strip().lower()
        if verdict not in ("confirmed", "denied"):
            continue
        summary = str(ruling.get("claim_summary") or "")
        if _claim_matches(claim, summary):
            return True
    return False


def required_dm_checks(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> list[str]:
    """NPC-authored DM CHECK claims that still lack a matching DM ruling."""
    return [
        claim
        for claim in extract_dm_check_templates_from_rules(npc)
        if not _ruling_covers_claim(dm_context, claim)
    ]


def _clean_language_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", (raw or "").strip()).strip(" .'\"")
    words = name.split()
    while words and words[0].lower() in _LANGUAGE_NAME_LEADING_FILLER:
        words.pop(0)
    if not words:
        return ""
    if words[0].lower() in _LANGUAGE_NAME_STOPWORDS:
        return ""
    if len(words) > 4:
        return ""
    return " ".join(words)


def _language_from_claim(claim: str) -> str:
    match = _LANGUAGE_BARRIER_CLAIM_PATTERN.search(claim or "")
    if match:
        cleaned = _clean_language_name(match.group(1))
        if cleaned:
            return cleaned
    before = _NAMED_LANGUAGE_BEFORE_PATTERN.search(claim or "")
    if before:
        return _clean_language_name(before.group(1))
    return ""


def _language_from_text_blob(text: str) -> str:
    for pattern in _NATIVE_LANGUAGE_PATTERNS:
        match = pattern.search(text or "")
        if match:
            cleaned = _clean_language_name(match.group(1))
            if cleaned:
                return cleaned
    return ""


def resolve_npc_native_language(
    npc: dict[str, Any] | None,
    *,
    language_claim: str | None = None,
) -> str:
    """
    Resolve the NPC's native language for barrier replies.

    Authors can set this via ``known_facts`` / ``public_facts`` (e.g. ``Native language:
    Slaadi``), an explicit ``llm_rules`` line, or by naming the language in the DM CHECK
    template (e.g. ``Language Slaadi``).
    """
    npc = npc or {}
    from_facts: list[str] = []
    for field in ("known_facts", "public_facts", "llm_rules", "npc_rules"):
        raw = npc.get(field)
        if isinstance(raw, list):
            from_facts.extend(str(item) for item in raw if item is not None)
        elif raw:
            from_facts.append(str(raw))

    for text in from_facts:
        resolved = _language_from_text_blob(text)
        if resolved:
            return resolved

    if language_claim:
        resolved = _language_from_claim(language_claim)
        if resolved:
            return resolved

    return ""


def _is_language_comprehension_claim(claim: str) -> bool:
    """True only for language-barrier checks, not generic 'does the player understand X'."""
    normalized = _normalize_claim(claim)
    if not normalized:
        return False
    if _language_from_claim(claim):
        return True
    if "language" not in normalized:
        return False
    if _KNOW_SPEAK_NAMED_LANGUAGE_PATTERN.search(claim or ""):
        return True
    if _NAMED_LANGUAGE_BEFORE_PATTERN.search(claim or "") and any(
        word in normalized for word in ("know", "speak", "speaker", "necessary")
    ):
        return True
    if any(phrase in normalized for phrase in _LANGUAGE_BARRIER_CLAIM_PHRASES):
        return True
    # Authored template: "Does the speaker know the necessary language to talk to the NPC"
    if "speaker" in normalized and "know" in normalized:
        return True
    return False


def is_language_comprehension_claim(claim: str) -> bool:
    """Public alias for DM follow-up / auto-response language-barrier hints."""
    return _is_language_comprehension_claim(claim)


def _language_barrier_templates(npc: dict[str, Any] | None) -> list[str]:
    return [
        claim
        for claim in extract_dm_check_templates_from_rules(npc)
        if _is_language_comprehension_claim(claim)
    ]


def _language_barrier_ruling_for_template(
    template: str,
    dm_context: dict[str, Any] | None,
) -> str | None:
    """Return the latest ``confirmed`` / ``denied`` verdict for a language template."""
    rulings = (dm_context or {}).get("rulings") or []
    for ruling in reversed(rulings):
        verdict = str(ruling.get("verdict") or "").strip().lower()
        if verdict not in ("confirmed", "denied"):
            continue
        summary = str(ruling.get("claim_summary") or "")
        if _claim_matches(template, summary):
            return verdict
    return None


def _language_barrier_enforcement_context(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Decide whether to rewrite English barrier replies.

    Returns None when Success is ruled (English allowed). Otherwise returns metadata
    for unruled (append DM CHECK tags) or denied Failure (rewrite only).
    """
    if not _npc_has_language_barrier_rules(npc):
        return None

    templates = _language_barrier_templates(npc)
    if not templates:
        return None

    for template in templates:
        verdict = _language_barrier_ruling_for_template(template, dm_context)
        if verdict == "confirmed":
            return None

    pending = [
        claim
        for claim in required_dm_checks(npc, dm_context)
        if _is_language_comprehension_claim(claim)
    ]
    if pending:
        return {
            "mode": "unruled",
            "language_claim": pending[0],
            "append_checks": True,
            "pending": pending,
        }

    for template in templates:
        if _language_barrier_ruling_for_template(template, dm_context) == "denied":
            return {
                "mode": "failure",
                "language_claim": template,
                "append_checks": False,
                "pending": [],
            }

    return None


def _npc_has_language_barrier_rules(npc: dict[str, Any] | None) -> bool:
    if _language_barrier_templates(npc):
        return True
    blob = _npc_rules_blob(npc).lower()
    if "do not treat the speaker as understood" in blob:
        return True
    if "fantasy-language" in blob:
        return True
    prose_markers = (
        "does not understand common",
        "do not understand common",
        "never switch to common",
        "only understands language",
        "do not answer the meaning",
        "not answer the meaning of their words",
        "keep failing to understand",
        "confirms a speaker knows",
    )
    if any(marker in blob for marker in prose_markers):
        return True
    failure_hint = _extract_failure_ruling_hint(npc).lower()
    return bool(
        failure_hint
        and (
            "fantasy-language" in failure_hint
            or "do not understand them" in failure_hint
            or "cannot understand them" in failure_hint
            or "show you cannot understand" in failure_hint
        )
    )


def _extract_failure_ruling_hint(npc: dict[str, Any] | None) -> str:
    for line in _npc_rules_blob(npc).splitlines():
        match = DM_RULING_FAILURE_PATTERN.search(line)
        if match:
            return match.group(1).strip().strip("'\"")
    return ""


def _split_stage_direction(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    match = _STAGE_DIRECTION_PATTERN.match(text)
    if not match:
        return "", text
    return match.group(0).strip(), text[match.end():].strip()


def _looks_like_english_comprehension(spoken: str) -> bool:
    """Heuristic: did the NPC reply as if they understood English?"""
    _stage, dialogue = _split_stage_direction(spoken)
    check_text = (dialogue or spoken).strip()
    if not check_text:
        return False
    words = re.findall(r"[a-zA-Z']+", check_text.lower())
    if len(words) < 2:
        return False
    english_hits = sum(1 for word in words if word in _ENGLISH_COMPREHENSION_WORDS)
    if english_hits >= 2:
        return True
    return len(words) >= 3 and english_hits >= 1 and len(words) <= 8


def _default_fantasy_language_line(_npc: dict[str, Any] | None) -> str:
    return "Skraa-vek?"


def _generate_native_language_barrier_reply(
    npc: dict[str, Any] | None,
    *,
    native_language: str,
    player_text: str | None,
    stage_direction: str,
    ask_llm_fn: Callable[[list[dict[str, str]]], str],
) -> str:
    """Ask the LLM for a short reply in the NPC's native language only."""
    npc = npc or {}
    npc_name = (npc.get("name") or npc.get("id") or "The NPC").strip()
    player_line = (player_text or "").strip() or "(The player spoke, but you cannot understand them.)"
    stage_hint = (
        f"You may begin with this nonverbal stage direction exactly: {stage_direction} "
        "Then put spoken words in plain text (no asterisks around speech)."
        if stage_direction
        else (
            "You may begin with one short nonverbal *stage direction* "
            "(action only, like *cocks its head.*). "
            "Put all spoken words in plain text after that — never wrap speech in asterisks."
        )
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You write one brief in-character NPC line for a tabletop RPG. "
                "Output spoken content only in the requested language — never English. "
                "Asterisks are only for nonverbal stage directions, never for spoken words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"NPC: {npc_name}\n"
                f"Native language: {native_language}\n"
                f"The player said (probably in English, which this NPC does NOT speak): "
                f"{player_line}\n\n"
                f"The NPC does not understand the player yet. Write ONE short reply showing "
                f"confusion or that they cannot understand — using ONLY {native_language}. "
                f"Invent appropriate words or sounds for {native_language}; do not translate into English.\n"
                f"{stage_hint}\n"
                "Do not include [DM CHECK] tags."
            ),
        },
    ]
    raw = (ask_llm_fn(messages) or "").strip()
    from dm_channel import extract_dm_check_tags

    spoken, _claims = extract_dm_check_tags(raw)
    spoken = spoken.strip()
    if spoken and not _looks_like_english_comprehension(spoken):
        return spoken
    # LLM still answered in English — fall back to generic gibberish.
    return _rewrite_language_barrier_spoken(stage_direction or spoken, npc)


def _rewrite_language_barrier_spoken(
    spoken: str,
    npc: dict[str, Any] | None,
    *,
    native_language: str = "",
    player_text: str | None = None,
    ask_llm_fn: Callable[[list[dict[str, str]]], str] | None = None,
) -> str:
    stage, _dialogue = _split_stage_direction(spoken)
    if ask_llm_fn and native_language:
        return _generate_native_language_barrier_reply(
            npc,
            native_language=native_language,
            player_text=player_text,
            stage_direction=stage,
            ask_llm_fn=ask_llm_fn,
        )
    fantasy = _default_fantasy_language_line(npc)
    if stage:
        return f"{stage} {fantasy}".strip()
    return fantasy


def enforce_unruled_check_spoken_behavior(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    raw_text: str,
    *,
    player_text: str | None = None,
    ask_llm_fn: Callable[[list[dict[str, str]]], str] | None = None,
) -> str:
    """
    When a language-barrier DM CHECK is still unruled (or Failure was ruled),
    strip English comprehension dialogue the model may have added before/after tags.

    After a Failure ruling, also strip OOC wait narration the model may echo from
    earlier held turns — that line must not play again without opening a new check.
    """
    barrier = _language_barrier_enforcement_context(npc, dm_context)
    if not barrier:
        return raw_text

    from dm_channel import extract_dm_check_tags, is_dm_wait_narration

    spoken, claims = extract_dm_check_tags(raw_text)
    needs_rewrite = _looks_like_english_comprehension(spoken)
    spoken_for_rewrite = spoken
    if barrier.get("mode") == "failure" and is_dm_wait_narration(spoken):
        # Deny already settled the language question; never re-hold on wait copy,
        # and never feed wait text back as a stage direction for the rewrite LLM.
        needs_rewrite = True
        spoken_for_rewrite = ""
    if not needs_rewrite:
        return raw_text

    native_language = resolve_npc_native_language(
        npc,
        language_claim=barrier["language_claim"],
    )
    rewritten = _rewrite_language_barrier_spoken(
        spoken_for_rewrite,
        npc,
        native_language=native_language,
        player_text=player_text,
        ask_llm_fn=ask_llm_fn,
    )
    result = rewritten.rstrip()
    if not barrier.get("append_checks"):
        return result

    tag_claims = list(claims)
    pending = barrier.get("pending") or []
    for claim in pending:
        if not any(_claim_matches(claim, existing) for existing in tag_claims):
            tag_claims.append(claim)
    for claim in tag_claims:
        result += f"\n[DM CHECK: {claim}]"
    return result


def format_required_dm_checks_prompt_block(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> str:
    """Remind the model of NPC-authored tags that are still waiting on a ruling."""
    pending = required_dm_checks(npc, dm_context)
    barrier_templates = _language_barrier_templates(npc)
    failure_templates = [
        template
        for template in barrier_templates
        if _language_barrier_ruling_for_template(template, dm_context) == "denied"
    ]
    language_waiting = any(_is_language_comprehension_claim(claim) for claim in pending)
    language_block = ""
    if language_waiting and _npc_has_language_barrier_rules(npc):
        failure_hint = _extract_failure_ruling_hint(npc)
        failure_line = failure_hint or "Use a short fantasy-language line only; do not reply in English."
        native_language = resolve_npc_native_language(
            npc,
            language_claim=next(
                (claim for claim in pending if _is_language_comprehension_claim(claim)),
                "",
            ),
        )
        native_line = (
            f"This NPC's native language is {native_language}. Reply ONLY in {native_language}, never English."
            if native_language
            else "Reply only in this NPC's native language (see NPC facts/rules), never English."
        )
        language_block = f"""

LANGUAGE BARRIER (still waiting on DM ruling):
Until a matching confirmed Success ruling exists, do NOT treat the speaker as understood.
{native_line} {failure_line}"""
    elif failure_templates and _npc_has_language_barrier_rules(npc):
        failure_hint = _extract_failure_ruling_hint(npc)
        failure_line = failure_hint or "Use a short fantasy-language line only; do not reply in English."
        native_language = resolve_npc_native_language(
            npc,
            language_claim=failure_templates[0],
        )
        native_line = (
            f"This NPC's native language is {native_language}. Reply ONLY in {native_language}, never English."
            if native_language
            else "Reply only in this NPC's native language (see NPC facts/rules), never English."
        )
        language_block = f"""

LANGUAGE BARRIER (DM ruled Failure — do not use English):
The DM already ruled the speaker does NOT know this NPC's language.
{native_line} {failure_line}"""

    if not pending and not language_block:
        return ""

    lines = "\n".join(f"- [DM CHECK: {claim}]" for claim in pending)
    tags_intro = ""
    if pending:
        tags_intro = f"""

NPC-AUTHORED DM CHECK TAGS (still required until the DM rules):
This NPC's llm_rules declare the exact checks below. Prefer these tag wordings over a
global paraphrase for the same situation. Until DM channel context shows a matching
confirmed (Success) or denied (Failure) ruling for each claim, end your reply with
those exact tags (one per line, after spoken dialogue / stage directions). Follow any
NPC [DM Ruling: Success] / [DM Ruling: Failure] instructions after a matching ruling.
Do not invent DM rulings yourself."""
    return f"{tags_intro}{language_block}\n{lines}".rstrip()


def ensure_required_dm_check_tags(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    raw_text: str,
) -> str:
    """
    Append still-unruled language-barrier DM CHECK tags the model omitted.

    Situational templates (secrets, writs, authority) stay prompt guidance only.
    Force-appending those on every turn would mute ordinary greetings.
    """
    pending = [
        claim
        for claim in required_dm_checks(npc, dm_context)
        if _is_language_comprehension_claim(claim)
    ]
    if not pending:
        return raw_text

    text = (raw_text or "").rstrip()
    existing = {
        _normalize_claim(m.group(1))
        for m in DM_CHECK_IN_RULES_PATTERN.finditer(text)
    }

    missing = [
        claim for claim in pending
        if _normalize_claim(claim) not in existing
        and not any(_claim_matches(claim, existing_claim) for existing_claim in existing)
    ]
    if not missing:
        return text

    suffix = "\n".join(f"[DM CHECK: {claim}]" for claim in missing)
    if text:
        return f"{text}\n{suffix}"
    return suffix


_LLM_PROMPT_LOG_LOCK = threading.Lock()


def log_llm_prompt_to_console(messages: list[dict[str, str]] | None) -> None:
    """
    Print the full NPC-reply prompt to the middleware process console.

    Emits one write under a lock so Flask ``threaded=True`` workers cannot
    interleave headers/body lines from concurrent turns.
    """
    msgs = messages or []
    total_chars = sum(len((m or {}).get("content") or "") for m in msgs)
    parts = [
        f"[LLM-PROMPT] npc-reply messages={len(msgs)} chars={total_chars}",
    ]
    for index, message in enumerate(msgs):
        role = (message or {}).get("role") or "?"
        content = (message or {}).get("content") or ""
        parts.append(f"[LLM-PROMPT] --- [{index}] role={role} chars={len(content)} ---")
        parts.append(content)
    parts.append("[LLM-PROMPT] === end ===")
    block = "\n".join(parts) + "\n"
    with _LLM_PROMPT_LOG_LOCK:
        # One print keeps the whole block contiguous in the console stream.
        print(block, end="", flush=True)


def generate_npc_reply_text(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    messages: list[dict[str, str]],
    ask_llm_fn: Callable[[list[dict[str, str]]], str],
    *,
    player_text: str | None = None,
    enforce_required_dm_checks: bool = True,
) -> str:
    """
    Call the LLM, then ensure NPC-authored required DM CHECK tags are present.

    When a language-barrier check is still unruled, skip the NPC reply LLM and
    return narrator wait-copy so the NPC does not speak English before the DM
    rules. Situational authored templates (secrets, writs) and already-open
    pending rows do not skip the LLM on their own — ``process_npc_response``
    still replaces spoken copy with wait narration when a check is pending.

    ``player_text`` is accepted for call-site compatibility; intent classification is
    left to the model via global/NPC llm_rules.

    Set ``enforce_required_dm_checks=False`` for post-ruling auto-replies so a just-
    resolved claim is not force-appended again. Language-barrier **Failure** rewrites
    still run so a Deny cannot leave the NPC speaking English.

    Stage-direction placement is normalized on the way out: every reply path here
    ends up as ``npc_text``, so this is the one place that can hold the llm_rules
    contract for all of text, voice, and post-ruling turns.
    """
    return normalize_stage_directions(
        _generate_npc_reply_text(
            npc,
            dm_context,
            messages,
            ask_llm_fn,
            player_text=player_text,
            enforce_required_dm_checks=enforce_required_dm_checks,
        )
    )


def _generate_npc_reply_text(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
    messages: list[dict[str, str]],
    ask_llm_fn: Callable[[list[dict[str, str]]], str],
    *,
    player_text: str | None = None,
    enforce_required_dm_checks: bool = True,
) -> str:
    if enforce_required_dm_checks and _should_skip_llm_for_pending_dm(npc, dm_context):
        from dm_channel import DM_PENDING_WAIT_NARRATION

        return ensure_required_dm_check_tags(npc, dm_context, DM_PENDING_WAIT_NARRATION)

    log_llm_prompt_to_console(messages)
    raw = ask_llm_fn(messages)
    if not enforce_required_dm_checks:
        barrier = _language_barrier_enforcement_context(npc, dm_context)
        if barrier and barrier.get("mode") == "failure":
            return enforce_unruled_check_spoken_behavior(
                npc,
                dm_context,
                raw,
                player_text=player_text,
                ask_llm_fn=ask_llm_fn,
            )
        return raw
    tagged = ensure_required_dm_check_tags(npc, dm_context, raw)
    return enforce_unruled_check_spoken_behavior(
        npc,
        dm_context,
        tagged,
        player_text=player_text,
        ask_llm_fn=ask_llm_fn,
    )


def _should_skip_llm_for_pending_dm(
    npc: dict[str, Any] | None,
    dm_context: dict[str, Any] | None,
) -> bool:
    # Only language-barrier templates skip the LLM. A pending writ/secret/player
    # claim still generates a held in-character draft; spoken wait copy is applied
    # later in process_npc_response.
    return any(
        _is_language_comprehension_claim(claim)
        for claim in required_dm_checks(npc, dm_context)
    )
