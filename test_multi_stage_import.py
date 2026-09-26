"""Unit tests for multi-stage import purposes, campaign hop, and repair."""
from __future__ import annotations

import content_import
from llm_router import (
    PURPOSE_CONTENT_CAMPAIGN,
    PURPOSE_CONTENT_CREATION,
    PURPOSE_CONTENT_DISCOVERY,
    PURPOSE_CONTENT_LOCATION,
    PURPOSE_CONTENT_NPC,
    PURPOSE_CONTENT_REPAIR,
    import_stage_to_purpose,
)


def test_import_stage_to_purpose_mapping():
    assert import_stage_to_purpose("discovery") == PURPOSE_CONTENT_DISCOVERY
    assert import_stage_to_purpose("outline") == PURPOSE_CONTENT_DISCOVERY
    assert import_stage_to_purpose("campaign") == PURPOSE_CONTENT_CAMPAIGN
    assert import_stage_to_purpose("location_details") == PURPOSE_CONTENT_LOCATION
    assert import_stage_to_purpose("npc_details") == PURPOSE_CONTENT_NPC
    assert import_stage_to_purpose("repair") == PURPOSE_CONTENT_REPAIR
    assert import_stage_to_purpose("unknown") == PURPOSE_CONTENT_CREATION


def test_generate_campaign_info_uses_campaign_purpose(monkeypatch):
    seen = {}

    def fake_call(system, user, *, warnings, step, usage_acc=None, response_schema=None, schema_name=None, max_schema_retries=None):
        seen["step"] = step
        seen["schema_name"] = schema_name
        from llm_router import import_stage_to_purpose as map_purpose

        seen["purpose"] = map_purpose(step)
        if usage_acc is not None:
            usage_acc["prompt_tokens"] = 10
            usage_acc["completion_tokens"] = 5
        return {
            "temporary_key": "campaign_001",
            "title": "Test Module",
            "name": "Test Module",
            "description": "A short summary.",
            "genre": "horror",
            "rpg_system": "Call of Cthulhu",
            "premise": "Something stirs.",
            "tone": "tense",
            "themes": ["mystery"],
            "factions": [],
            "starting_situation": "You arrive at dusk.",
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call)
    monkeypatch.setattr(content_import, "content_creation_max_input_chars", lambda: 200_000)

    result = content_import.generate_campaign_info(
        "=== PAGE 1 ===\nIntro to the haunted manor.\n",
        {"temporary_key": "campaign_001", "title": "Seed", "name": "Seed", "genre": "horror"},
        source_index_summary="pages=1",
    )
    assert seen["step"] == "campaign"
    assert seen["purpose"] == PURPOSE_CONTENT_CAMPAIGN
    assert seen["schema_name"] == "campaign_info_response"
    assert result["campaign"]["title"]
    assert "usage" in result


def test_repair_does_not_include_full_document_when_excerpts_provided(monkeypatch):
    captured = {}

    def fake_call(system, user, *, warnings, step, usage_acc=None, response_schema=None, schema_name=None, max_schema_retries=None):
        captured["user"] = user
        captured["step"] = step
        return {
            "action": "NO_CHANGE",
            "patches": [],
            "rationale": "insufficient",
            "source_refs": [],
            "warnings": [],
        }

    monkeypatch.setattr(content_import, "_call_llm_json", fake_call)
    full = "FULL_MODULE_MARKER " + ("x" * 50_000)
    result = content_import.repair_import_entity(
        problem="dangling_ref",
        entities=[{"temporary_key": "npc_001", "name": "Ada"}],
        excerpts="cited page only",
        text=full,
    )
    assert captured["step"] == "repair"
    assert "cited page only" in captured["user"]
    assert "FULL_MODULE_MARKER" not in captured["user"]
    assert result["action"] == "NO_CHANGE"


def test_call_llm_json_mechanical_retry_stays_on_same_purpose(monkeypatch):
    calls = []

    def fake_generate(messages, *, purpose="chat", max_tokens=None, temperature=None, response_format=None, **kwargs):
        calls.append({"purpose": purpose, "response_format": response_format})
        if len(calls) == 1:
            return "not-json {{{", {"prompt_tokens": 1, "completion_tokens": 1}
        return '{"locations": []}', {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(
        "llm_router.generate_llm_response_with_usage",
        fake_generate,
    )
    # Also patch the import used inside _call_llm_json's local import path
    import llm_router

    monkeypatch.setattr(llm_router, "generate_llm_response_with_usage", fake_generate)
    monkeypatch.setattr(llm_router, "IMPORT_MAX_SCHEMA_RETRIES", 1)

    warnings: list[str] = []
    out = content_import._call_llm_json(
        "sys",
        "user",
        warnings=warnings,
        step="location_details",
        response_schema=content_import.LOCATION_BATCH_RESPONSE_SCHEMA,
        schema_name="location_batch_response",
    )
    assert out == {"locations": []}
    assert all(c["purpose"] == PURPOSE_CONTENT_LOCATION for c in calls)
    assert calls[0]["response_format"] is not None
    assert calls[1]["response_format"] is None  # retry drops schema
    assert not any(c["purpose"] == PURPOSE_CONTENT_REPAIR for c in calls)


def test_slim_discovery_prompt_omits_long_campaign_prose_fields():
    prompt = content_import._build_outline_prompt(
        "Some module text",
        "create_new",
        None,
        [],
        [],
    )
    assert "SLIM discovery" in prompt
    assert "Do NOT write full campaign description" in prompt
    assert '"premise"' not in prompt


def test_location_prompt_uses_location_knowledge_rules_not_npc_fields():
    prompt = content_import._build_location_detail_prompt(
        [{"temporary_key": "location_001", "id": "location_001", "name": "Bridge", "type": "ship"}],
        "Bridge prose.",
        {"name": "Test", "temporary_key": "campaign_001"},
    )
    assert "knowledge_rules" in prompt
    assert "public_knowledge" in prompt
    assert "THIS NPC" not in prompt
    assert "public_facts" not in prompt or "public_facts:" not in prompt.split("KNOWLEDGE RULES:")[-1]


def test_normalize_outline_preserves_location_significance():
    campaign, locations, npcs = content_import._normalize_outline(
        {
            "campaign": {"temporary_key": "campaign_001", "title": "T", "name": "T", "genre": "sci_fi"},
            "locations": [
                {
                    "temporary_key": "location_001",
                    "name": "Bridge",
                    "type": "ship",
                    "significance": "minor",
                }
            ],
            "npcs": [],
        },
        mode="create_new",
        existing_campaign_id=None,
        existing_location_ids=[],
        existing_npc_names=[],
        warnings=[],
    )
    assert locations[0]["significance"] == "minor"


def test_json_schema_response_format_is_not_strict():
    fmt = content_import._json_schema_response_format(
        "discovery_response",
        content_import.DISCOVERY_RESPONSE_SCHEMA,
    )
    assert fmt["json_schema"]["strict"] is False


def test_import_purpose_override_resolves_via_content_creation_eligibility(monkeypatch):
    import llm_router

    monkeypatch.setitem(
        llm_router.IMPORT_PURPOSE_MODEL_IDS,
        llm_router.PURPOSE_CONTENT_REPAIR,
        "openrouter-gpt-oss-20b",
    )
    # Catalog entry typically lists content_creation, not content_repair.
    entry = llm_router.LLM_MODELS.get("openrouter-gpt-oss-20b")
    if entry is not None:
        entry["purposes"] = [llm_router.PURPOSE_CONTENT_CREATION]
        entry["enabled"] = True
    mid = llm_router._get_purpose_model_id(llm_router.PURPOSE_CONTENT_REPAIR)
    assert mid == "openrouter-gpt-oss-20b"
    assert mid != "local"