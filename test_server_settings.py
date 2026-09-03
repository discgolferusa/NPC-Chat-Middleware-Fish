"""Unit tests for narrator TTS defaults in server_settings."""

import pytest

import server_settings
from content_pack import install_content_pack
from server_settings import format_llm_rules, get_narrator_tts_settings, load_global_llm_rules
from session_context import clear_request_context, set_request_campaign, set_request_owner


def test_narrator_defaults_use_libritts_dylan_howard(tmp_path):
    missing = tmp_path / "missing-tts.yaml"
    settings = get_narrator_tts_settings(missing)

    assert settings["voice_model"].endswith("en_US-libritts-high.onnx")
    assert "libritts" in settings["voice_model"].replace("\\", "/").lower()
    assert settings["speaker_id"] == 3
    assert settings["speaker_name"] == "Dylan Howard"


def test_narrator_reads_speaker_fields_from_yaml(tmp_path):
    path = tmp_path / "tts.yaml"
    path.write_text(
        "\n".join(
            [
                "narrator:",
                "  voice_model: en/en_US/Licensed/libritts/en_US-libritts-high.onnx",
                "  speaker_id: 3",
                "  speaker_name: Dylan Howard",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_narrator_tts_settings(path)
    assert settings["speaker_id"] == 3
    assert settings["speaker_name"] == "Dylan Howard"
    assert "libritts" in settings["voice_model"]


def test_speaker_id_without_name_is_not_labeled_dylan(tmp_path):
    path = tmp_path / "tts.yaml"
    path.write_text(
        "\n".join(
            [
                "narrator:",
                "  voice_model: en/en_US/Licensed/libritts/en_US-libritts-high.onnx",
                "  speaker_id: 10",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_narrator_tts_settings(path)
    assert settings["speaker_id"] == 10
    assert settings["speaker_name"] == ""


def test_overridden_voice_without_speaker_id_uses_zero(tmp_path):
    path = tmp_path / "tts.yaml"
    path.write_text(
        "\n".join(
            [
                "narrator:",
                "  voice_model: en/en_US/amy/medium/en_US-amy-medium.onnx",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_narrator_tts_settings(path)
    assert "amy" in settings["voice_model"].replace("\\", "/").lower()
    assert settings["speaker_id"] == 0
    assert settings["speaker_name"] == ""


def test_load_global_llm_rules_prefers_content_pack(tmp_path, monkeypatch):
    local_path = tmp_path / "llm_rules.yaml"
    local_path.write_text(
        "llm_rules:\n  - Local fallback rule\n",
        encoding="utf-8",
    )

    install_content_pack(
        "owner-a",
        campaign_id="camp-1",
        content_pack={
            "campaigns": {"camp-1": {"campaign_id": "camp-1", "name": "Camp"}},
            "llm_rules": "llm_rules:\n  - Pack rule wins\n",
        },
    )

    set_request_owner("owner-a")
    set_request_campaign("camp-1")
    formatted = load_global_llm_rules(local_path)

    assert "- Pack rule wins" in formatted
    assert "Local fallback rule" not in formatted


def test_load_global_llm_rules_falls_back_to_local_file(tmp_path):
    clear_request_context()
    local_path = tmp_path / "llm_rules.yaml"
    local_path.write_text(
        "llm_rules:\n  - Local fallback rule\n",
        encoding="utf-8",
    )

    formatted = load_global_llm_rules(local_path)

    assert "- Local fallback rule" in formatted
    assert format_llm_rules({"llm_rules": ["Local fallback rule"]}) == formatted


@pytest.mark.parametrize(
    "empty_pack_rules",
    [
        {},
        [],
        {"llm_rules": None},
    ],
    ids=["empty-object", "empty-list", "yaml-with-empty-llm-rules"],
)
def test_load_global_llm_rules_falls_back_when_pack_rules_are_empty(
    tmp_path, monkeypatch, empty_pack_rules
):
    local_path = tmp_path / "llm_rules.yaml"
    local_path.write_text(
        "llm_rules:\n  - Local fallback rule\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        server_settings,
        "_resolve_pack_llm_rules",
        lambda: empty_pack_rules,
    )

    formatted = load_global_llm_rules(local_path)

    assert formatted == "- Local fallback rule"
