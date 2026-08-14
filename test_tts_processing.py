"""Unit tests for Piper TTS text prep and option resolution."""

import json

from tts_processing import (
    build_af_filter,
    clean_for_tts,
    curated_defaults_for_quality,
    infer_dataset_name,
    infer_quality_tier,
    load_piper_model_meta,
    resolve_tts_options,
)


def test_clean_for_tts_expands_and_pronounces():
    text = clean_for_tts("Dr. Torin walked to Golarion.")
    assert "Doctor" in text
    assert "go LAR ee un" in text
    assert text.endswith(".")


def test_curated_low_quality_is_less_noisy():
    low = curated_defaults_for_quality("low")
    medium = curated_defaults_for_quality("medium")
    assert low["noise_scale"] < medium["noise_scale"]


def test_infer_quality_tier():
    assert infer_quality_tier(r"C:\AI\Voices\en\en_US\ryan\low\x.onnx") == "low"
    assert infer_quality_tier(r"C:\AI\Voices\en\en_US\ryan\high\x.onnx") == "high"


def test_infer_dataset_name_from_standard_path():
    model = r"C:\AI\Voices\en\en_US\lessac\medium\en_US-lessac-medium.onnx"
    assert infer_dataset_name(model) == "lessac"


def test_load_model_meta_exposes_dataset_and_named_speakers(tmp_path):
    model_dir = tmp_path / "en" / "en_US" / "characters" / "medium"
    model_dir.mkdir(parents=True)
    model = model_dir / "en_US-characters-medium.onnx"
    model.touch()
    config = {
        "dataset": "campaign_characters",
        "num_speakers": 2,
        "speaker_id_map": {"Narrator": 0, "Guard": 1},
    }
    (model_dir / "en_US-characters-medium.onnx.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    meta = load_piper_model_meta(model)

    assert meta["dataset"] == "campaign_characters"
    assert meta["has_speaker_map"] is True
    assert meta["speakers"] == [
        {"id": 0, "name": "Narrator"},
        {"id": 1, "name": "Guard"},
    ]


def test_resolve_prefers_yaml_override():
    opts = resolve_tts_options(
        {"voice_model": r"C:\x\low\a.onnx", "tts_noise_scale": 0.2},
    )
    assert opts["noise_scale"] == 0.2
    assert opts["quality"] == "low"


def test_af_filter_includes_clarity_baseline():
    af = build_af_filter(fx_preset="cave", pitch_semitones=1)
    assert "dynaudnorm" in af
    assert "afftdn=" in af
    assert "aecho" in af
    assert "asetrate" in af
    # Mid-sentence pauses must not be treated as clip end.
    assert "stop_periods" not in af
    assert "areverse" in af


def test_af_filter_underwater_robot_whisper_presets():
    underwater = build_af_filter(fx_preset="underwater", trim_silence=False)
    robot = build_af_filter(fx_preset="robot", trim_silence=False)
    whisper = build_af_filter(fx_preset="whisper", trim_silence=False)
    assert "lowpass=f=900" in underwater
    assert "aecho=" in underwater
    assert "vibrato=" in robot
    assert "equalizer=f=2200" in robot
    assert "highpass=f=350" in whisper
    assert "volume=0.72" in whisper


def test_resolve_accepts_new_fx_presets():
    for preset in ("underwater", "robot", "whisper"):
        opts = resolve_tts_options({"tts_fx": preset})
        assert opts["fx_preset"] == preset
    assert resolve_tts_options({"tts_fx": "not-a-real-fx"})["fx_preset"] == "none"


def test_af_filter_always_includes_denoise_by_default():
    af = build_af_filter(fx_preset="none", trim_silence=False)
    assert "afftdn=nr=12.00:nf=-25.00" in af
    assert af.index("highpass=") < af.index("afftdn=")
    assert af.index("afftdn=") < af.index("equalizer=")


def test_af_filter_denoise_severity_is_tunable():
    mild = build_af_filter(denoise_nr=6, denoise_nf=-20, trim_silence=False)
    strong = build_af_filter(denoise_nr=20, denoise_nf=-35, trim_silence=False)
    assert "afftdn=nr=6.00:nf=-20.00" in mild
    assert "afftdn=nr=20.00:nf=-35.00" in strong


def test_af_filter_clamps_nf_to_ffmpeg_afftdn_range():
    # Values above -20 (e.g. -5) must not reach FFmpeg — afftdn rejects them.
    mild = build_af_filter(denoise_nr=12, denoise_nf=-5, trim_silence=False)
    deep = build_af_filter(denoise_nr=12, denoise_nf=-90, trim_silence=False)
    assert "afftdn=nr=12.00:nf=-20.00" in mild
    assert "afftdn=nr=12.00:nf=-80.00" in deep


def test_resolve_clamps_denoise_nf_to_ffmpeg_range():
    opts = resolve_tts_options({"tts_denoise_nf": -10})
    assert opts["denoise_nf"] == -20
    opts_low = resolve_tts_options({"tts_denoise_nf": -100})
    assert opts_low["denoise_nf"] == -80


def test_af_filter_denoise_disabled_when_nr_zero():
    af = build_af_filter(denoise_nr=0, denoise_nf=-25, trim_silence=False)
    assert "afftdn=" not in af


def test_resolve_reads_denoise_from_yaml():
    opts = resolve_tts_options(
        {"tts_denoise_nr": 8, "tts_denoise_nf": -22},
    )
    assert opts["denoise_nr"] == 8
    assert opts["denoise_nf"] == -22


def test_af_filter_trim_still_removes_edges_only():
    af = build_af_filter(trim_silence=True)
    assert af.count("silenceremove=") == 2
    assert "stop_periods" not in af
