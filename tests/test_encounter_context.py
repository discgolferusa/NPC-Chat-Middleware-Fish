"""Shared encounter context: audience, whispers, rulings, and scene place."""

from __future__ import annotations

import threading

import encounter_context as ec
import scene_state


NPC = "captain_varra"
OTHER_NPC = "blacksmith_torin"
CAMPAIGN = "greyhaven_intro"
ALDEN = "player_alden"
SERA = "player_sera"


def _party(speaker: str) -> dict:
    members = [
        {"player_id": ALDEN, "name": "Alden Mor", "race": "Human", "class": "Fighter"},
        {"player_id": SERA, "name": "Sera Vale", "race": "Elf", "class": "Wizard"},
    ]
    for member in members:
        member["is_speaking"] = member["player_id"] == speaker
    speaking = next(member for member in members if member["is_speaking"])
    return {
        "player_id": speaker,
        "speaking_member": speaking,
        "party_members": members,
        "campaign_id": CAMPAIGN,
    }


def _prompt_for(server, speaker: str) -> str:
    turns = ec.load_encounter_turns(NPC, CAMPAIGN)
    return ec.format_encounter_prompt(
        turns,
        current_player_id=speaker,
        party_context=_party(speaker),
    )


def _heard_section(prompt: str, name: str) -> str:
    marker = f"What {name} heard:"
    heard = prompt.split(marker, 1)[1]
    return heard.split("Do not reveal", 1)[0]


def test_legacy_null_visibility_stays_with_original_player(api):
    _, server, _ = api
    server.save_turn(NPC, ALDEN, "user", "Legacy secret handshake.", campaign_id=CAMPAIGN)

    alden_prompt = _prompt_for(server, ALDEN)
    sera_prompt = _prompt_for(server, SERA)
    assert "Legacy secret handshake." in alden_prompt
    assert "Legacy secret handshake." not in sera_prompt


def test_scene_answer_is_available_to_the_next_character(api):
    _, server, _ = api
    record = ec.build_turn_record(
        visibility=None,
        explicit_audience=None,
        speaker_id=ALDEN,
        party_context=_party(ALDEN),
        scene_context={},
    )
    server.save_turn(
        NPC, ALDEN, "user", "The well is dry.",
        campaign_id=CAMPAIGN, **record,
    )

    prompt = _prompt_for(server, SERA)
    assert "The well is dry." in prompt
    assert "Alden Mor" in prompt
    assert "The well is dry." in _heard_section(prompt, "Sera Vale")


def test_absent_character_does_not_count_as_having_heard(api):
    client, server, headers = api
    marked = client.post(
        "/scene/presence",
        headers=headers,
        json={"player_id": SERA, "present": False},
    )
    assert marked.status_code == 200
    assert SERA in (marked.get_json().get("absent_player_ids") or [])

    record = ec.build_turn_record(
        visibility=None,
        explicit_audience=None,
        speaker_id=ALDEN,
        party_context=_party(ALDEN),
        scene_context={},
    )
    assert SERA not in record["audience_player_ids"]
    server.save_turn(
        NPC, ALDEN, "user", "The well is dry.",
        campaign_id=CAMPAIGN, **record,
    )

    client.post(
        "/scene/presence",
        headers=headers,
        json={"player_id": SERA, "present": True},
    )
    prompt = _prompt_for(server, SERA)
    assert "The well is dry." in prompt
    assert "The well is dry." not in _heard_section(prompt, "Sera Vale")
    assert "did not hear" in prompt
    party = server.format_party_context_for_prompt(_party(ALDEN))
    # Presence was cleared, so the roster is present again.
    assert "Sera Vale" in party
    assert "absent" not in party.split("Sera Vale", 1)[1].split("\n", 1)[0]


def test_absent_label_while_marked_away(api):
    client, server, headers = api
    client.post("/scene/presence", headers=headers, json={"player_id": SERA, "present": False})
    prompt = server.format_party_context_for_prompt(_party(ALDEN))
    assert "Sera Vale" in prompt
    assert "absent" in prompt.split("Sera Vale", 1)[1].split("\n", 1)[0]


def test_whisper_stays_player_only(api):
    _, server, _ = api
    record = ec.build_turn_record(
        visibility="whisper",
        explicit_audience=None,
        speaker_id=ALDEN,
        party_context=_party(ALDEN),
        scene_context={},
    )
    assert record["visibility"] == "private"
    assert record["audience_player_ids"] == [ALDEN]
    server.save_turn(
        NPC, ALDEN, "user", "The password is lantern.",
        campaign_id=CAMPAIGN, **record,
    )

    sera = _prompt_for(server, SERA)
    assert "The password is lantern." in sera
    assert "private" in sera
    assert "The password is lantern." not in _heard_section(sera, "Sera Vale")

    alden = _prompt_for(server, ALDEN)
    assert "The password is lantern." in _heard_section(alden, "Alden Mor")


def test_private_ruling_is_omitted_for_the_other_character(api, monkeypatch):
    client, server, headers = api
    import dm_channel

    record = ec.build_turn_record(
        visibility="whisper",
        explicit_audience=None,
        speaker_id=ALDEN,
        party_context=_party(ALDEN),
        scene_context={},
    )
    server.save_turn(
        NPC, ALDEN, "user", "The password is lantern.",
        campaign_id=CAMPAIGN, **record,
    )
    dm_channel.create_manual_ruling(
        npc_id=NPC,
        player_id=ALDEN,
        claim_summary="Alden asks the guard to ignore the password",
        verdict="confirmed",
        notes="The guard ignores the password for Alden only.",
        campaign_id=CAMPAIGN,
        npc=server.load_npc(NPC),
        party_context=_party(ALDEN),
    )

    captured = {}
    monkeypatch.setattr(
        server, "ask_llm", lambda messages: captured.update(messages=messages) or "I cannot say.",
    )
    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": NPC,
            "text": "What did they decide?",
            "history": [],
            "player_id": SERA,
            "campaign_id": CAMPAIGN,
            "include_audio": False,
        },
    )
    assert turn.status_code == 200
    blob = "\n".join(message.get("content") or "" for message in captured["messages"])
    assert "The guard ignores the password for Alden only." not in blob
    assert "[DM RULING - CONFIRMED]" not in blob
    assert "The password is lantern." in blob
    assert "The password is lantern." not in _heard_section(blob, "Sera Vale")


def test_scene_ruling_consequence_reaches_the_next_speaker(api, monkeypatch):
    client, server, headers = api
    import dm_channel

    dm_channel.create_manual_ruling(
        npc_id=NPC,
        player_id=ALDEN,
        claim_summary="Alden persuades the guard to open the gate",
        verdict="confirmed",
        notes="The guard opens the gate for Alden. He does not obey further orders.",
        campaign_id=CAMPAIGN,
        npc=server.load_npc(NPC),
        party_context=_party(ALDEN),
    )

    captured = {}

    def _ask(messages):
        captured["messages"] = messages
        return "The gate is open."

    monkeypatch.setattr(server, "ask_llm", _ask)
    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": NPC,
            "text": "What did the guard decide?",
            "history": [],
            "player_id": SERA,
            "campaign_id": CAMPAIGN,
            "include_audio": False,
        },
    )
    assert turn.status_code == 200
    blob = "\n".join(message.get("content") or "" for message in captured["messages"])
    assert "The guard opens the gate for Alden. He does not obey further orders." in blob
    assert "player_alden" in blob or "Alden Mor" in blob
    assert "[DM RULING - CONFIRMED]" in blob
    assert "Apply only the recorded consequence." in blob
    assert "Outcome: confirmed" in blob


def test_denied_ruling_stays_denied_for_the_next_speaker(api, monkeypatch):
    client, server, headers = api
    import dm_channel

    dm_channel.create_manual_ruling(
        npc_id=NPC,
        player_id=ALDEN,
        claim_summary="Alden persuades the guard to open the gate",
        verdict="denied",
        notes="The guard keeps the gate shut.",
        campaign_id=CAMPAIGN,
        npc=server.load_npc(NPC),
        party_context=_party(ALDEN),
    )
    captured = {}
    monkeypatch.setattr(server, "ask_llm", lambda messages: captured.update(messages=messages) or "No.")
    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": NPC,
            "text": "Did it work?",
            "history": [],
            "player_id": SERA,
            "campaign_id": CAMPAIGN,
            "include_audio": False,
        },
    )
    assert turn.status_code == 200
    blob = "\n".join(message.get("content") or "" for message in captured["messages"])
    assert "[DM RULING - DENIED]" in blob
    assert "The guard keeps the gate shut." in blob


def test_item_handoff_consequence_is_distinct_from_player_dialogue(api):
    _, server, _ = api
    import dm_channel

    dm_channel.create_manual_ruling(
        npc_id=NPC,
        player_id=ALDEN,
        claim_summary="Alden gives the lantern to the guard",
        verdict="confirmed",
        notes="Alden hands the lantern to the guard. The guard now holds it.",
        campaign_id=CAMPAIGN,
        npc=server.load_npc(NPC),
        party_context=_party(ALDEN),
    )
    prompt = _prompt_for(server, SERA)
    assert "[GM RESOLUTION" in prompt
    assert "Alden hands the lantern to the guard." in prompt


def test_other_npc_does_not_receive_the_party_discovery(api):
    _, server, _ = api
    record = ec.build_turn_record(
        visibility=None,
        explicit_audience=None,
        speaker_id=ALDEN,
        party_context=_party(ALDEN),
        scene_context={},
    )
    server.save_turn(
        NPC, ALDEN, "user", "We found the sealed stair.",
        campaign_id=CAMPAIGN, **record,
    )
    other = ec.load_encounter_turns(OTHER_NPC, CAMPAIGN)
    prompt = ec.format_encounter_prompt(
        other, current_player_id=SERA, party_context=_party(SERA),
    )
    assert "sealed stair" not in prompt


def test_concurrent_turns_both_append(api):
    _, server, _ = api
    errors = []

    def _write(player_id: str, text: str) -> None:
        try:
            ec.insert_turn(
                NPC, player_id, "user", text, CAMPAIGN,
                visibility="scene",
                audience_player_ids=[ALDEN, SERA],
                event_kind="dialogue",
            )
        except Exception as ex:  # pragma: no cover - assertion below
            errors.append(ex)

    threads = [
        threading.Thread(target=_write, args=(ALDEN, "Alden speaks first.")),
        threading.Thread(target=_write, args=(SERA, "Sera speaks second.")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    turns = ec.load_encounter_turns(NPC, CAMPAIGN)
    contents = [turn["content"] for turn in turns]
    assert "Alden speaks first." in contents
    assert "Sera speaks second." in contents


def test_text_turn_includes_previous_line_without_client_history(api, monkeypatch):
    client, server, headers = api
    captured = {}

    def _ask(messages):
        captured["messages"] = messages
        return "I heard you."

    monkeypatch.setattr(server, "ask_llm", _ask)
    first = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": NPC,
            "text": "My name is Alden.",
            "history": [],
            "player_id": ALDEN,
            "campaign_id": CAMPAIGN,
            "include_audio": False,
        },
    )
    assert first.status_code == 200
    second = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": NPC,
            "text": "What is my name?",
            "history": [],
            "player_id": ALDEN,
            "campaign_id": CAMPAIGN,
            "include_audio": False,
        },
    )
    assert second.status_code == 200
    system = captured["messages"][0]["content"]
    assert "My name is Alden." in system


def test_set_location_reaches_the_prompt_without_moving_other_npcs(api, monkeypatch):
    client, server, headers = api
    placed = client.post(
        "/scene/sub-location",
        headers=headers,
        json={
            "location_id": "greyhaven",
            "important_location_name": "Crooked Lantern Inn",
        },
    )
    assert placed.status_code == 200

    captured = {}
    monkeypatch.setattr(
        server,
        "ask_llm",
        lambda messages: captured.update(messages=messages) or "We are at the inn.",
    )
    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": NPC,
            "text": "Where are we?",
            "history": [],
            "player_id": ALDEN,
            "campaign_id": CAMPAIGN,
            "include_audio": False,
        },
    )
    assert turn.status_code == 200
    system = captured["messages"][0]["content"]
    assert "Crooked Lantern Inn" in system

    elsewhere = scene_state.format_scene_context_for_prompt(
        scene_state.build_scene_context("not_greyhaven")
    )
    assert "not set for this NPC's region" in elsewhere
    routes = {str(rule) for rule in server.app.url_map.iter_rules()}
    assert not any("accompan" in route for route in routes)


def test_private_memory_does_not_load_for_another_character(api):
    _, server, _ = api
    server.save_memory(
        NPC, ALDEN, "Alden Mor: the password is lantern.",
        campaign_id=CAMPAIGN, visibility="private",
    )
    server.save_memory(
        NPC, ALDEN, "Alden Mor promised to return the lantern.",
        campaign_id=CAMPAIGN, visibility="scene",
    )
    alden = server.load_encounter_memories(NPC, ALDEN, CAMPAIGN, limit=8)
    sera = server.load_encounter_memories(NPC, SERA, CAMPAIGN, limit=8)
    assert any("password" in item for item in alden)
    assert any("promised" in item for item in alden)
    assert any("promised" in item for item in sera)
    assert not any("password" in item for item in sera)
    other = server.load_encounter_memories(OTHER_NPC, SERA, CAMPAIGN, limit=8)
    assert other == []


def test_behavior_rules_mention_audience_and_consequence():
    from pathlib import Path

    text = Path("ServerSettings/llm_rules.yaml").read_text(encoding="utf-8")
    assert "still in the audience" in text
    assert "whisper was heard only by the speaking character" in text
    assert "recorded consequence" in text
