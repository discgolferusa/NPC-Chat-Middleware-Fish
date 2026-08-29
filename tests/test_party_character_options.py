"""Character options survive a play session that lost its party id (restart)."""

from __future__ import annotations

import json

import party_campaign as pc

HEROES = {
    "party_id": "heroes",
    "campaign_id": "greyhaven_intro",
    "members": [
        {
            "player_id": "player_alden",
            "party_id": "heroes",
            "campaign_id": "greyhaven_intro",
            "name": "Alden",
            "race": "Human",
            "class": "Fighter",
        },
        {
            "player_id": "player_mira",
            "party_id": "heroes",
            "campaign_id": "greyhaven_intro",
            "name": "Mira",
            "race": "Halfling",
            "class": "Rogue",
        },
    ],
}


def _stub_parties(monkeypatch, parties: dict[str, dict]):
    monkeypatch.setattr(
        pc,
        "load_party",
        lambda party_id: parties.get(party_id, {"party_id": party_id, "members": []}),
    )
    monkeypatch.setattr(
        pc,
        "list_parties",
        lambda campaign_id=None: [
            {"party_id": party_id, "campaign_id": party.get("campaign_id", "")}
            for party_id, party in parties.items()
            if not campaign_id or not party.get("campaign_id") or party.get("campaign_id") == campaign_id
        ],
    )


def test_named_party_lists_its_members(monkeypatch):
    _stub_parties(monkeypatch, {"heroes": HEROES})

    options = pc.get_character_options(campaign_id="greyhaven_intro", party_id="heroes")

    assert [option["player_id"] for option in options] == ["player_alden", "player_mira"]


def test_placeholder_party_id_widens_to_the_campaign(monkeypatch):
    # A middleware restart rebuilds the play session with DEFAULT_PARTY_ID, which
    # resolves to no party at all.
    _stub_parties(monkeypatch, {"heroes": HEROES})

    options = pc.get_character_options(
        campaign_id="greyhaven_intro",
        party_id=pc.DEFAULT_PARTY_ID,
    )

    assert [option["player_id"] for option in options] == ["player_alden", "player_mira"]
    assert not any(option.get("is_default") for option in options)


def test_default_party_is_the_last_resort(monkeypatch):
    _stub_parties(monkeypatch, {})

    options = pc.get_character_options(campaign_id="greyhaven_intro", party_id="default_party")

    assert len(options) == 1
    assert options[0]["is_default"] is True
    assert options[0]["name"] == "The Party"


def test_party_characters_endpoint_recovers_after_a_restart(api, monkeypatch):
    client, server, headers = api
    _stub_parties(monkeypatch, {"heroes": HEROES})

    with pc._STATE_LOCK:
        pc._PLAY_SESSIONS.clear()

    response = client.get(
        "/party/characters",
        headers={
            **headers,
            "X-Owner-User-Id": "owner-1",
            "X-Campaign-Id": "greyhaven_intro",
        },
    )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert [character["player_id"] for character in data["characters"]] == [
        "player_alden",
        "player_mira",
    ]
    assert data["includes_default_party"] is False
