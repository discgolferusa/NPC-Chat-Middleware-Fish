"""Unit tests for narrator TTS defaults in server_settings."""

from server_settings import get_narrator_tts_settings


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
