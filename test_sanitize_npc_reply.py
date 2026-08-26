"""Unit tests for NPC reply sanitization (meta leaks, name prefixes, dupes)."""

from __future__ import annotations

from sanitize_npc_reply import sanitize_npc_reply_text, sanitize_npc_spoken_body


MATEO = {"id": "mateo", "name": "Mateo"}

LEAKED = (
    "Mercy shut the launch hard-wired. The port's sealed. No way to fire the engines. "
    "You can't get out that way. You either load the manual override or stay."
    "Check for DM tag none."
    "Mateo: Mercy shut the launch hard-wired. The port's sealed. No way to fire the engines. "
    "You can't get out that way. You either load the manual override or stay."
)

CLEAN = (
    "Mercy shut the launch hard-wired. The port's sealed. No way to fire the engines. "
    "You can't get out that way. You either load the manual override or stay."
)


def test_strips_dm_meta_leak_and_name_prefix_and_duplicate():
    assert sanitize_npc_spoken_body(LEAKED, MATEO) == CLEAN


def test_preserves_clean_dialogue():
    assert sanitize_npc_spoken_body(CLEAN, MATEO) == CLEAN


def test_strips_leading_script_name_only():
    assert (
        sanitize_npc_spoken_body("Mateo: Hold the line.", MATEO)
        == "Hold the line."
    )


def test_does_not_strip_unrelated_name_mention_in_dialogue():
    # Mid-sentence name without colon stays.
    text = "Tell Mateo to seal the port."
    assert sanitize_npc_spoken_body(text, MATEO) == text


def test_preserves_real_dm_check_machine_tag():
    raw = f"{CLEAN}\n[DM CHECK: player claims engine override access]"
    out = sanitize_npc_reply_text(raw, MATEO)
    assert out.startswith(CLEAN)
    assert "[DM CHECK: player claims engine override access]" in out
    assert "Check for DM tag" not in out


def test_strips_meta_without_npc_name_alias():
    text = "Stay put. Check for DM tag none. Stay put."
    # Without aliases, still remove meta and collapse the duplicate.
    assert sanitize_npc_spoken_body(text, None) == "Stay put."


def test_full_name_and_first_token_aliases():
    npc = {"name": "Mateo Rios"}
    assert sanitize_npc_spoken_body("Mateo Rios: Go.", npc) == "Go."
    assert sanitize_npc_spoken_body("Mateo: Go.", npc) == "Go."
