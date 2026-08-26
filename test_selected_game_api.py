"""Functional coverage for the selected game API routes (mocked LLM/TTS/STT)."""

from __future__ import annotations

import io
import json

import party_campaign
import scene_state


def test_api_fixture_isolates_on_disk_paths(api):
    client, server, headers = api

    assert server.MEMORY_DB_CONFIG.is_sqlite
    assert server.MEMORY_DB.resolve() != (server.BASE_DIR / "npc_memory.db").resolve()
    assert server.UPLOAD_DIR.resolve() != (server.BASE_DIR / "uploads").resolve()
    assert server.OUTPUT_DIR.resolve() != (server.BASE_DIR / "outputs").resolve()
    assert scene_state.SCENE_RUNTIME_CONFIG_FILE.resolve().parent != server.BASE_DIR.resolve()
    assert party_campaign.SESSION_RUNTIME_CONFIG_FILE.resolve().parent != server.BASE_DIR.resolve()

    repo_scene = server.BASE_DIR / "scene_runtime_config.json"
    repo_session = server.BASE_DIR / "session_runtime_config.json"
    scene_before = repo_scene.read_bytes() if repo_scene.exists() else None
    session_before = repo_session.read_bytes() if repo_session.exists() else None
    upload_before = {
        p.name for p in (server.BASE_DIR / "uploads").glob("*")
    } if (server.BASE_DIR / "uploads").is_dir() else set()

    assert client.post(
        "/scene/time", headers=headers, json={"time_of_day": "Evening"}
    ).status_code == 200
    assert client.post(
        "/party/select",
        headers=headers,
        json={"player_id": "player_alden", "party_id": "default_party"},
    ).status_code == 200
    voice = client.post(
        "/npc-voice-turn",
        headers=headers,
        data={
            "npc_id": "captain_varra",
            "scene": "Barracks",
            "player_id": "player_alden",
            "audio": (io.BytesIO(b"RIFF" + (b"\x00" * 2048)), "clip.wav"),
        },
        content_type="multipart/form-data",
    )
    assert voice.status_code == 200

    assert scene_state.SCENE_RUNTIME_CONFIG_FILE.exists()
    assert party_campaign.SESSION_RUNTIME_CONFIG_FILE.exists()
    assert any(server.UPLOAD_DIR.iterdir())

    scene_after = repo_scene.read_bytes() if repo_scene.exists() else None
    session_after = repo_session.read_bytes() if repo_session.exists() else None
    upload_after = {
        p.name for p in (server.BASE_DIR / "uploads").glob("*")
    } if (server.BASE_DIR / "uploads").is_dir() else set()

    assert scene_after == scene_before
    assert session_after == session_before
    assert upload_after == upload_before


def test_session_status(api):
    client, _, headers = api
    response = client.get("/session/status", headers=headers)
    assert response.status_code == 200
    body = response.get_json()
    assert "campaign_id" in body
    assert "scene" in body


def test_party_select(api):
    client, _, headers = api
    response = client.post(
        "/party/select",
        headers=headers,
        json={"player_id": "player_alden", "party_id": "default_party"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "selected"
    assert body.get("player_id") == "player_alden" or body.get("speaking_member")


def test_scene_time_and_sub_location(api):
    client, _, headers = api

    set_time = client.post("/scene/time", headers=headers, json={"time_of_day": "Evening"})
    assert set_time.status_code == 200
    assert set_time.get_json().get("time_of_day") == "Evening"

    clear_time = client.post("/scene/time/clear", headers=headers, json={})
    assert clear_time.status_code == 200

    set_sub = client.post(
        "/scene/sub-location",
        headers=headers,
        json={"location_id": "greyhaven", "important_location_name": "Old Aqueduct"},
    )
    assert set_sub.status_code == 200
    sub = set_sub.get_json().get("sub_location") or set_sub.get_json()
    assert "Aqueduct" in json.dumps(sub)

    clear_sub = client.post("/scene/sub-location/clear", headers=headers, json={})
    assert clear_sub.status_code == 200

    options = client.get("/scene/sub-locations?location_id=greyhaven", headers=headers)
    assert options.status_code == 200
    options_body = options.get_json()
    assert options_body["location_id"] == "greyhaven"
    assert options_body.get("location_name")
    important = options_body.get("important_locations")
    assert isinstance(important, list)
    assert any(
        isinstance(item, dict) and item.get("name") == "Old Aqueduct"
        for item in important
    )


def test_npcs_and_voices(api):
    client, _, headers = api

    npcs = client.get("/npcs", headers=headers)
    assert npcs.status_code == 200
    payload = npcs.get_json()
    assert "npcs" in payload
    assert any(n.get("id") == "captain_varra" for n in payload["npcs"])

    voices = client.get("/voices", headers=headers)
    assert voices.status_code == 200
    body = voices.get_json()
    assert body.get("voice_enabled") is True
    assert len(body.get("voices") or []) >= 1


def test_npc_preload_and_npc_turn(api):
    client, _, headers = api

    preload = client.post(
        "/npc-preload",
        headers=headers,
        json={"npc_id": "captain_varra", "scene": "Watch barracks at night"},
    )
    assert preload.status_code == 200
    assert preload.get_json()["status"] == "preloaded"
    assert preload.get_json()["npc_id"] == "captain_varra"

    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": "captain_varra",
            "text": "Any news from the aqueduct?",
            "scene": {"description": "Rainy barracks"},
            "history": [],
            "player_id": "player_alden",
        },
    )
    assert turn.status_code == 200
    body = turn.get_json()
    assert "request_id" in body
    # Response shape varies slightly; accept common reply fields.
    reply = body.get("npc_text") or body.get("text") or body.get("reply") or body.get("spoken_text")
    assert reply
    assert "watch" in reply.lower() or "firm" in reply.lower()


def test_npc_turn_with_audio_completes_progress_after_tts(api):
    """include_audio must not leave progress stuck on synthesizing_speech after complete."""
    client, _, headers = api
    request_id = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": "captain_varra",
            "text": "Any news from the aqueduct?",
            "scene": {"description": "Rainy barracks"},
            "history": [],
            "player_id": "player_alden",
            "include_audio": True,
            "request_id": request_id,
        },
    )
    assert turn.status_code == 200
    body = turn.get_json()
    assert body.get("audio_clips") or body.get("audio_url")

    progress = client.get(f"/turn/progress/{request_id}", headers=headers)
    assert progress.status_code == 200
    progress_body = progress.get_json()
    assert progress_body["status"] == "complete"
    assert progress_body["step"] == "complete"


def test_npc_voice_turn(api):
    client, server, headers = api
    data = {
        "npc_id": "captain_varra",
        "scene": "Barracks",
        "player_id": "player_alden",
    }
    # Larger than the empty/short-upload guard in save_uploaded_audio.
    audio = (io.BytesIO(b"RIFF" + (b"\x00" * 2048)), "clip.wav")
    response = client.post(
        "/npc-voice-turn",
        headers=headers,
        data={**data, "audio": audio},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.mimetype == "audio/wav"
    assert response.headers.get("X-Turn-Request-Id")
    assert response.headers.get("X-Player-Transcript")
    assert response.headers.get("X-Npc-Text")

    import base64

    def _decode_header(value: str) -> str:
        padded = value + ("=" * (-len(value) % 4))
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")

    assert _decode_header(response.headers["X-Player-Transcript"]) == "Hello captain."
    npc_text = _decode_header(response.headers["X-Npc-Text"])
    assert "watch" in npc_text.lower() or "firm" in npc_text.lower()


def test_npc_voice_turn_returns_409_when_tts_superseded_before_first_clip(api, monkeypatch):
    """Confirm/Deny during narrator setup must not send_file a missing response WAV."""
    client, server, headers = api
    import turn_progress as tp

    request_id = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    npc_id = "captain_varra"
    player_id = "player_alden"
    campaign_id = "greyhaven_intro"

    def _supersede_on_tts(**_kwargs):
        tp.register_turn_progress(
            "ffffffffffffffffffffffffffffffff",
            route="/dm-ruling-auto-response",
            npc_id=npc_id,
            player_id=player_id,
            mode="text",
            campaign_id=campaign_id,
            force=True,
        )
        return {
            "clips": [],
            "narration_text": "",
            "dialogue_text": "",
            "speech_path": None,
            "narration_path": None,
            "pending": False,
        }

    monkeypatch.setattr(server, "synthesize_segmented_speech_clips", _supersede_on_tts)

    audio = (io.BytesIO(b"RIFF" + (b"\x00" * 2048)), "clip.wav")
    response = client.post(
        "/npc-voice-turn",
        headers=headers,
        data={
            "npc_id": npc_id,
            "scene": "Barracks",
            "player_id": player_id,
            "campaign_id": campaign_id,
            "request_id": request_id,
            "audio": audio,
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 409
    body = response.get_json()
    assert body.get("error") == "superseded"
    assert response.mimetype != "audio/wav"


def test_npc_voice_turn_upload_failure_releases_lock(api, monkeypatch):
    client, server, headers = api

    def _boom(*_args, **_kwargs):
        raise RuntimeError("convert failed")

    monkeypatch.setattr(server, "save_uploaded_audio", _boom)
    first = client.post(
        "/npc-voice-turn",
        headers=headers,
        data={
            "npc_id": "captain_varra",
            "player_id": "player_alden",
            "audio": (io.BytesIO(b"RIFF" + (b"\x00" * 2048)), "clip.wav"),
        },
        content_type="multipart/form-data",
    )
    assert first.status_code == 500

    active = client.get(
        "/turn/progress/active?npc_id=captain_varra&player_id=player_alden",
        headers=headers,
    )
    assert active.status_code == 200
    assert (active.get_json() or {}).get("status") == "idle"

    text = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": "captain_varra",
            "text": "Hello.",
            "scene": {"description": "Barracks"},
            "history": [],
            "player_id": "player_alden",
            "include_audio": False,
        },
    )
    assert text.status_code != 409


def test_dm_channel_endpoints(api):
    client, _, headers = api
    npc = "captain_varra"
    player = "player_alden"
    qs = f"npc_id={npc}&player_id={player}"

    status = client.get(f"/dm/status?{qs}", headers=headers)
    assert status.status_code == 200
    assert status.get_json()["npc_id"] == npc

    detect = client.post("/dm/detect", headers=headers, json={"text": "Yes, that is correct."})
    assert detect.status_code == 200
    assert detect.get_json()["detection"]["verdict"] == "confirmed"

    msg = client.post(
        "/dm/message",
        headers=headers,
        json={
            "npc_id": npc,
            "player_id": player,
            "text": "Yes, the door is unlocked.",
            "auto_respond": False,
            "include_audio": False,
        },
    )
    assert msg.status_code == 200
    assert msg.get_json()["status"] == "saved"

    messages = client.get(f"/dm/messages?{qs}", headers=headers)
    assert messages.status_code == 200
    assert isinstance(messages.get_json()["messages"], list)
    assert len(messages.get_json()["messages"]) >= 1

    ruling = client.post(
        "/dm/ruling",
        headers=headers,
        json={
            "npc_id": npc,
            "player_id": player,
            "claim_summary": "party has the mayor's writ",
            "verdict": "confirmed",
            "notes": "Granted for this scene.",
            "auto_respond": False,
            "include_audio": False,
        },
    )
    assert ruling.status_code == 200

    pending = client.get(f"/dm/pending?{qs}", headers=headers)
    assert pending.status_code == 200
    pending_body = pending.get_json()
    assert "pending_checks" in pending_body
    assert pending_body.get("pending") == pending_body.get("pending_checks")

    rulings = client.get(f"/dm/rulings?{qs}", headers=headers)
    assert rulings.status_code == 200
    assert isinstance(rulings.get_json()["rulings"], list)
    assert len(rulings.get_json()["rulings"]) >= 1

    speak = client.post(
        "/dm/speak",
        headers=headers,
        json={
            "npc_id": npc,
            "player_id": player,
            "text": "Stand aside.",
            "include_audio": False,
            "save_history": True,
        },
    )
    assert speak.status_code == 200
    assert speak.get_json()["status"] == "spoken"


def test_content_pack_install(api):
    client, _, headers = api
    response = client.post(
        "/content/pack",
        headers={
            **headers,
            "X-Campaign-Owner-User-Id": "owner-test-1",
        },
        json={
            "owner_user_id": "owner-test-1",
            "campaign_id": "test_campaign",
            "files": [
                {
                    "kind": "npc",
                    "filename": "scout_mira.yaml",
                    "content": "id: scout_mira\nname: Scout Mira\nrole: scout\n",
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "installed"

    # Tenant-scoped /npcs should see pack content when owner/campaign headers are present.
    listed = client.get(
        "/npcs",
        headers={
            **headers,
            "X-Campaign-Owner-User-Id": "owner-test-1",
            "X-Campaign-Id": "test_campaign",
        },
    )
    assert listed.status_code == 200
    ids = [n.get("id") for n in listed.get_json().get("npcs") or []]
    assert "scout_mira" in ids


def test_content_import_routes(api):
    client, _, headers = api
    sample = "Chapter 1\nThe outpost holds against the fog."

    analyze = client.post(
        "/content/import/analyze",
        headers=headers,
        json={"text": sample, "mode": "create_new"},
    )
    assert analyze.status_code == 200
    assert analyze.get_json()["campaign"]["id"] == "test_campaign"

    outline = client.post(
        "/content/import/outline",
        headers=headers,
        json={"text": sample, "mode": "create_new"},
    )
    assert outline.status_code == 200
    assert "campaign" in outline.get_json()

    loc = client.post(
        "/content/import/location-details",
        headers=headers,
        json={
            "text": sample,
            "campaign": {"id": "test_campaign", "name": "Test"},
            "locations": [{"id": "loc_1", "name": "Outpost"}],
        },
    )
    assert loc.status_code == 200
    assert loc.get_json()["locations"]

    npc = client.post(
        "/content/import/npc-details",
        headers=headers,
        json={
            "text": sample,
            "campaign": {"id": "test_campaign", "name": "Test"},
            "npcs": [{"id": "npc_1", "name": "Guard"}],
        },
    )
    assert npc.status_code == 200
    assert npc.get_json()["npcs"]


def test_npc_turn_authored_dm_check_is_enforced(api, monkeypatch):
    """Literal [DM CHECK:…] templates in llm_rules must open a pending check."""
    client, server, headers = api
    npc_id = "language_gate_npc"
    player_id = "player_alden"

    rule_npc = {
        "id": npc_id,
        "name": "Xill",
        "role": "chaos beast",
        "location_id": "greyhaven",
        "personality": ["alien"],
        "public_facts": [],
        "known_facts": [],
        "secrets": [],
        "voice": ["rasping"],
        "llm_rules": [
            "Until the DM rules, do not treat the speaker as understood.",
            "End unsettled replies with "
            "[DM CHECK: Does the speaker know the necessary language to talk to the NPC]",
            "[DM Ruling: Success] Respond to the speaker in English.",
            "[DM Ruling: Failure] Show you cannot understand them.",
        ],
    }

    llm_calls = {"n": 0}

    def _ask(_messages):
        llm_calls["n"] += 1
        return "*Xill cocks its head.* Krah-vesh?"

    monkeypatch.setattr(server, "load_npc", lambda _id: dict(rule_npc))
    monkeypatch.setattr(server, "ask_llm", _ask)

    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": npc_id,
            "text": "Hello, can you help us?",
            "scene": {"description": "Cave mouth"},
            "history": [],
            "player_id": player_id,
            "include_audio": False,
        },
    )
    assert turn.status_code == 200
    body = turn.get_json()
    reply = body.get("npc_text") or body.get("text") or ""
    assert llm_calls["n"] == 0
    assert "waiting on a decision from game master" in reply.lower()
    assert "Krah-vesh" not in reply
    assert body.get("dm_check_claim")
    assert "necessary language" in str(body.get("dm_check_claim")).lower()
    assert body.get("dm_pending_check_id")

    pending = client.get(
        f"/dm/pending?npc_id={npc_id}&player_id={player_id}",
        headers=headers,
    )
    assert pending.status_code == 200
    checks = pending.get_json().get("pending_checks") or []
    assert any("necessary language" in str(item.get("claim_summary", "")).lower() for item in checks)


def test_npc_turn_slaad_strips_english_when_language_check_unruled(api, monkeypatch):
    """Slaad-style NPC must not speak comprehensible English before DM Success ruling."""
    client, server, headers = api
    npc_id = "slaad"
    player_id = "player_alden"

    rule_npc = {
        "id": npc_id,
        "name": "Slaad",
        "role": "chaos beast",
        "location_id": "demo",
        "personality": ["aggressive"],
        "public_facts": ["dislikes humans"],
        "known_facts": ["Native language: Slaadi"],
        "secrets": [],
        "voice": [],
        "llm_rules": [
            "Until the DM rules, do not treat the speaker as understood.",
            "'End unsettled replies with [DM CHECK: Does the speaker know the necessary language to talk to the NPC]'",
            "'[DM Ruling: Success] Respond to the speaker in English.'",
            "'[DM Ruling: Failure] You do not understand them; reply only to show that, in a short fantasy-language line.'",
        ],
    }

    llm_calls = {"n": 0}

    def _ask(_messages):
        llm_calls["n"] += 1
        if llm_calls["n"] == 1:
            return "*Snarls.* Yes, I understand you. What do you want?"
        return "*Snarls.* Krah-thok slaad-vek?"

    monkeypatch.setattr(server, "load_npc", lambda _id: dict(rule_npc))
    monkeypatch.setattr(server, "ask_llm", _ask)

    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": npc_id,
            "text": "Hello, can you help us?",
            "scene": {"description": "Cave"},
            "history": [],
            "player_id": player_id,
            "include_audio": False,
        },
    )
    assert turn.status_code == 200
    body = turn.get_json()
    reply = body.get("npc_text") or body.get("text") or ""
    assert llm_calls["n"] == 0
    assert "waiting on a decision from game master" in reply.lower()
    assert "understand you" not in reply.lower()
    assert "Krah-thok" not in reply
    assert body.get("dm_check_claim")
    assert "necessary language" in str(body.get("dm_check_claim")).lower()


def test_npc_turn_without_authored_template_does_not_invent_check(api, monkeypatch):
    """Without an authored template, omitted tags are not force-appended from player text."""
    client, server, headers = api
    npc_id = "captain_varra"
    player_id = "player_alden"

    rule_npc = {
        "id": npc_id,
        "name": "Captain Varra",
        "role": "watch captain",
        "location_id": "greyhaven",
        "personality": ["suspicious"],
        "public_facts": [],
        "known_facts": [],
        "secrets": [],
        "voice": ["stern"],
        "llm_rules": [
            "Stay in character as the watch captain.",
        ],
    }

    llm_calls = {"n": 0}

    def _ask(_messages):
        llm_calls["n"] += 1
        return "I cannot simply take your word for that."

    monkeypatch.setattr(server, "load_npc", lambda _id: dict(rule_npc))
    monkeypatch.setattr(server, "ask_llm", _ask)

    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": npc_id,
            "text": "We have a letter from the mayor allowing us to enter the aqueduct.",
            "scene": {"description": "Barracks gate"},
            "history": [],
            "player_id": player_id,
            "include_audio": False,
        },
    )
    assert turn.status_code == 200
    body = turn.get_json()
    # spoken npc_text is always tag-stripped; invention shows up on claim fields.
    assert llm_calls["n"] == 1
    assert body.get("dm_check_claims") in (None, [])
    assert body.get("dm_check_claim") is None
    assert "waiting on a decision from game master" in (body.get("npc_text") or "").lower()
    campaign_id = body.get("campaign_id") or "greyhaven_intro"
    history = server.load_recent_turns(npc_id, player_id, campaign_id=campaign_id)
    assistant = " ".join(
        (row["content"] or "")
        for row in history
        if row["role"] == "assistant"
    )
    assert "cannot simply take your word" in assistant.lower()
    assert "waiting on a decision from game master" not in assistant.lower()


def test_npc_turn_situational_template_greeting_still_speaks(api, monkeypatch):
    """Secrets/writ templates must not mute ordinary greetings or invent a pending check."""
    client, server, headers = api
    npc_id = "keeper_npc"
    player_id = "player_alden"

    rule_npc = {
        "id": npc_id,
        "name": "Keeper",
        "role": "archivist",
        "location_id": "greyhaven",
        "personality": ["guarded"],
        "public_facts": [],
        "known_facts": [],
        "secrets": ["the vault combination"],
        "voice": ["quiet"],
        "llm_rules": [
            "If asked about secrets, end with "
            "[DM CHECK: the player is probing one of this NPC's secrets]",
        ],
    }

    llm_calls = {"n": 0}

    def _ask(_messages):
        llm_calls["n"] += 1
        return "Good day. The reading room is open."

    monkeypatch.setattr(server, "load_npc", lambda _id: dict(rule_npc))
    monkeypatch.setattr(server, "ask_llm", _ask)

    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": npc_id,
            "text": "Hello there.",
            "scene": {"description": "Archive"},
            "history": [],
            "player_id": player_id,
            "include_audio": False,
        },
    )
    assert turn.status_code == 200
    body = turn.get_json()
    assert llm_calls["n"] == 1
    assert "reading room is open" in (body.get("npc_text") or "").lower()
    assert "waiting on a decision from game master" not in (body.get("npc_text") or "").lower()
    assert body.get("dm_check_claim") is None
    assert not body.get("dm_pending_check_id")


def test_dm_pending_and_ruling_are_campaign_wide_without_player_id(api, monkeypatch):
    """DM console proxies strip player_id; pending/rulings must still resolve by campaign."""
    client, server, headers = api
    import dm_channel

    monkeypatch.setattr(server, "ask_llm", lambda _messages: "Hold.")
    monkeypatch.setattr(
        server,
        "run_dm_ruling_auto_response",
        lambda *args, **kwargs: None,
    )

    pending_id = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player claims official authority or permission",
        player_text="Captain let us pass",
        npc_text="I need the DM.",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert pending_id

    listed = client.get(
        "/dm/pending?campaign_id=greyhaven_intro",
        headers=headers,
    )
    assert listed.status_code == 200
    body = listed.get_json()
    checks = body.get("pending_checks") or []
    assert any(item.get("id") == pending_id for item in checks)
    assert body.get("pending") == checks

    ruling = client.post(
        "/dm/ruling",
        headers=headers,
        json={
            "campaign_id": "greyhaven_intro",
            "check_id": pending_id,
            "verdict": "confirm",
            "text": "Yes, that is approved.",
            "auto_respond": False,
            "include_audio": False,
        },
    )
    assert ruling.status_code == 200, ruling.get_json()
    saved = ruling.get_json()
    assert saved.get("status") == "saved"
    assert saved.get("verdict") == "confirmed"
    assert saved.get("claim_summary")
    assert "official authority" in saved["claim_summary"].lower()

    history = client.get(
        "/dm/rulings?campaign_id=greyhaven_intro",
        headers=headers,
    )
    assert history.status_code == 200
    rulings = history.get_json().get("rulings") or []
    assert any(row.get("id") == saved.get("ruling_id") for row in rulings)
    assert any("official authority" in str(row.get("claim_summary", "")).lower() for row in rulings)

    remaining = client.get(
        "/dm/pending?campaign_id=greyhaven_intro",
        headers=headers,
    )
    assert remaining.status_code == 200
    still_open = remaining.get_json().get("pending_checks") or []
    assert all(item.get("id") != pending_id for item in still_open)


def test_npc_memory_clear_dm_wipes_other_players_language_rulings(api):
    """DM Knowledge clear must remove Deny rulings under the speaking player_id."""
    client, server, headers = api
    import dm_channel

    campaign = "greyhaven_intro"
    npc = "captain_varra"
    pending_id = dm_channel.save_pending_check(
        npc_id=npc,
        player_id="player_alden",
        claim_summary="Does the speaker know the necessary language to talk to the NPC",
        player_text="Hello",
        npc_text="Waiting.",
        source="npc",
        campaign_id=campaign,
    )
    ruling_id = dm_channel.save_ruling(
        npc_id=npc,
        player_id="player_alden",
        claim_summary="Does the speaker know the necessary language to talk to the NPC",
        verdict="denied",
        ruling_text="Failure",
        source="dm",
        pending_check_id=pending_id,
        campaign_id=campaign,
    )
    assert ruling_id
    assert dm_channel.load_rulings(npc, "player_alden", campaign_id=campaign)

    # Session speaker is not player_alden — old clear would no-op on these rows.
    cleared = client.post(
        "/npc-memory/clear",
        headers=headers,
        json={
            "npc_id": npc,
            "campaign_id": campaign,
            "clear_type": "dm",
            "player_id": "dm_console",
            "all_players": True,
        },
    )
    assert cleared.status_code == 200, cleared.get_json()
    body = cleared.get_json()
    assert body.get("all_players") is True
    assert (body.get("deleted_dm") or {}).get("deleted_rulings", 0) >= 1
    assert dm_channel.load_rulings(npc, "player_alden", campaign_id=campaign) == []


def test_dm_ruling_auto_respond_uses_pending_player_party_context(api, monkeypatch):
    """After check_id override, auto_respond must use the pending claim's player speaker."""
    client, server, headers = api
    import dm_channel

    captured = {}

    def _auto(npc_id, player_id, ruling, **kwargs):
        captured["npc_id"] = npc_id
        captured["player_id"] = player_id
        captured["party_context"] = kwargs.get("party_context")
        return {"status": "spoken", "player_id": player_id}

    monkeypatch.setattr(server, "run_dm_ruling_auto_response", _auto)

    # Without a selected character the DM proxy session speaker is DEFAULT_PLAYER_ID,
    # which must not leak into auto_respond after pending-row player override.
    pending_id = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player claims official authority or permission",
        player_text="Captain let us pass",
        npc_text="I need the DM.",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert pending_id

    ruling = client.post(
        "/dm/ruling",
        headers=headers,
        json={
            "campaign_id": "greyhaven_intro",
            "check_id": pending_id,
            "verdict": "confirm",
            "auto_respond": True,
            "include_audio": False,
        },
    )
    assert ruling.status_code == 200, ruling.get_json()
    body = ruling.get_json()
    assert body.get("player_id") == "player_alden"
    assert (body.get("speaking_member") or {}).get("player_id") == "player_alden"
    assert captured.get("player_id") == "player_alden"
    assert (captured.get("party_context") or {}).get("player_id") == "player_alden"
    assert (captured.get("party_context") or {}).get("speaking_member", {}).get("player_id") == "player_alden"


def test_dm_ruling_auto_respond_does_not_reopen_pending_check(api, monkeypatch):
    """Confirm + auto_respond must not recreate the same pending check from follow-up tags."""
    client, server, headers = api
    import dm_channel

    monkeypatch.setattr(
        server,
        "ask_llm",
        lambda _messages: (
            "Very well, you may pass.\n"
            "[DM CHECK: player claims official authority or permission]"
        ),
    )
    monkeypatch.setattr(
        server,
        "load_npc",
        lambda _id: {
            "id": "captain_varra",
            "name": "Captain Varra",
            "role": "watch captain",
            "llm_rules": [
                "[DM CHECK: player claims official authority or permission]",
            ],
        },
    )

    pending_id = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player claims official authority or permission",
        player_text="Captain let us pass",
        npc_text="I need confirmation.",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert pending_id

    ruling = client.post(
        "/dm/ruling",
        headers=headers,
        json={
            "campaign_id": "greyhaven_intro",
            "check_id": pending_id,
            "verdict": "confirm",
            "text": "Yes, that is approved.",
            "auto_respond": True,
            "include_audio": False,
        },
    )
    assert ruling.status_code == 200, ruling.get_json()
    body = ruling.get_json()
    assert body.get("status") == "saved"
    assert body.get("verdict") == "confirmed"
    auto = body.get("auto_response") or {}
    assert auto.get("generated") is True
    assert "may pass" in (auto.get("npc_text") or "").lower()
    assert "[DM CHECK:" not in (auto.get("npc_text") or "")

    remaining = client.get(
        "/dm/pending?campaign_id=greyhaven_intro",
        headers=headers,
    )
    assert remaining.status_code == 200
    checks = remaining.get_json().get("pending_checks") or []
    assert checks == []
    assert all(item.get("id") != pending_id for item in checks)


def test_dm_pending_includes_display_names_and_can_be_dismissed(api, monkeypatch):
    client, server, headers = api
    import dm_channel

    monkeypatch.setattr(
        server,
        "load_npc",
        lambda npc_id: {"id": npc_id, "name": "Captain Varra"},
    )
    monkeypatch.setattr(
        server,
        "resolve_member",
        lambda player_id, party_id=None: {
            "player_id": player_id,
            "name": "Alden",
        },
    )

    pending_id = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="player claims official authority or permission",
        player_text="Captain let us pass",
        npc_text="I need the DM.",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert pending_id

    listed = client.get(
        "/dm/pending?campaign_id=greyhaven_intro",
        headers=headers,
    )
    assert listed.status_code == 200
    checks = listed.get_json().get("pending_checks") or []
    match = next(item for item in checks if item.get("id") == pending_id)
    assert match.get("npc_name") == "Captain Varra"
    assert match.get("player_name") == "Alden"
    assert match.get("claim_summary")

    dismissed = client.post(
        "/dm/pending/dismiss",
        headers=headers,
        json={
            "campaign_id": "greyhaven_intro",
            "check_id": pending_id,
        },
    )
    assert dismissed.status_code == 200, dismissed.get_json()
    body = dismissed.get_json()
    assert body.get("status") == "ignored"
    assert body.get("pending_check_id") == pending_id
    assert all(item.get("id") != pending_id for item in (body.get("pending_checks") or []))

    remaining = client.get(
        "/dm/pending?campaign_id=greyhaven_intro",
        headers=headers,
    )
    assert remaining.status_code == 200
    assert all(
        item.get("id") != pending_id
        for item in (remaining.get_json().get("pending_checks") or [])
    )

    # Second dismiss must not report success while status stays ignored.
    again = client.post(
        "/dm/pending/dismiss",
        headers=headers,
        json={
            "campaign_id": "greyhaven_intro",
            "check_id": pending_id,
        },
    )
    assert again.status_code == 409
    again_body = again.get_json()
    assert again_body.get("status") == "ignored"
    assert "already" in str(again_body.get("error") or "").lower()
    assert (again_body.get("check") or {}).get("status") == "ignored"


def test_dm_pending_dismiss_rejects_already_confirmed(api):
    client, _, headers = api
    import dm_channel

    pending_id = dm_channel.save_pending_check(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="already confirmed claim",
        player_text="claim",
        npc_text="defer",
        source="npc",
        campaign_id="greyhaven_intro",
    )
    assert pending_id

    dm_channel.create_manual_ruling(
        npc_id="captain_varra",
        player_id="player_alden",
        claim_summary="already confirmed claim",
        verdict="confirmed",
        notes="Approved.",
        pending_check_id=pending_id,
        campaign_id="greyhaven_intro",
    )

    response = client.post(
        "/dm/pending/dismiss",
        headers=headers,
        json={
            "campaign_id": "greyhaven_intro",
            "check_id": pending_id,
        },
    )
    assert response.status_code == 409
    body = response.get_json()
    assert body.get("status") == "confirmed"
    assert "already confirmed" in str(body.get("error") or "").lower()
    assert (body.get("check") or {}).get("status") == "confirmed"


def test_npc_turn_strips_dm_meta_leak_and_script_name_for_tts(api, monkeypatch):
    """Leaked DM-tag checklist / Name: prefixes must not reach npc_text or Piper."""
    client, server, headers = api
    npc_id = "mateo_leak_npc"
    player_id = "player_alden"

    mateo = {
        "id": npc_id,
        "name": "Mateo",
        "role": "crew",
        "location_id": "greyhaven",
        "personality": ["tense"],
        "public_facts": [],
        "known_facts": [],
        "secrets": [],
        "voice": ["gruff"],
        "llm_rules": [],
    }

    leaked = (
        "Mercy shut the launch hard-wired. The port's sealed. No way to fire the engines. "
        "You can't get out that way. You either load the manual override or stay."
        "Check for DM tag none."
        "Mateo: Mercy shut the launch hard-wired. The port's sealed. No way to fire the engines. "
        "You can't get out that way. You either load the manual override or stay."
    )
    clean = (
        "Mercy shut the launch hard-wired. The port's sealed. No way to fire the engines. "
        "You can't get out that way. You either load the manual override or stay."
    )

    monkeypatch.setattr(server, "load_npc", lambda _id: dict(mateo))
    monkeypatch.setattr(server, "ask_llm", lambda _messages: leaked)

    tts_bodies: list[str] = []

    def _fake_tts(*, npc_text, **_kwargs):
        tts_bodies.append(npc_text)
        return {
            "clips": [],
            "narration_text": "",
            "dialogue_text": npc_text,
            "speech_path": None,
            "narration_path": None,
            "pending": False,
        }

    monkeypatch.setattr(server, "synthesize_segmented_speech_clips", _fake_tts)

    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": npc_id,
            "text": "What do you mean, Mercy stopped it?",
            "scene": {"description": "Launch bay"},
            "history": [],
            "player_id": player_id,
            "include_audio": True,
        },
    )
    assert turn.status_code == 200
    body = turn.get_json()
    reply = (body.get("npc_text") or body.get("dialogue_text") or "").strip()
    assert reply == clean
    assert "Check for DM tag" not in reply
    assert not reply.lower().startswith("mateo:")
    assert "Mateo:" not in reply
    assert tts_bodies
    assert tts_bodies[0] == clean
    # Clients (Discord captions / browser) still receive npc_name to label the speaker.
    assert (body.get("npc_name") or "").strip() == "Mateo"

