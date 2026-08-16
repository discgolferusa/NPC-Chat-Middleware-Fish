"""Functional coverage for the selected game API routes (mocked LLM/TTS/STT)."""

from __future__ import annotations

import io
import json

import party_campaign
import scene_state


def test_api_fixture_isolates_on_disk_paths(api):
    client, server, headers = api

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
    assert "pending_checks" in pending.get_json()

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


def test_npc_turn_rule_dm_check_is_enforced(api, monkeypatch):
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
    assert llm_calls["n"] == 1
    assert "Krah-vesh" in reply
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

