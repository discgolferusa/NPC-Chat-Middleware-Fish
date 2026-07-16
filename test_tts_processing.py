"""Unit tests for Piper TTS text prep and option resolution."""

from tts_processing import (
    build_af_filter,
    clean_for_tts,
    curated_defaults_for_quality,
    infer_quality_tier,
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


def test_resolve_prefers_yaml_override():
    opts = resolve_tts_options(
        {"voice_model": r"C:\x\low\a.onnx", "tts_noise_scale": 0.2},
    )
    assert opts["noise_scale"] == 0.2
    assert opts["quality"] == "low"


def test_af_filter_includes_clarity_baseline():
    af = build_af_filter(fx_preset="cave", pitch_semitones=1)
    assert "dynaudnorm" in af
    assert "aecho" in af
    assert "asetrate" in af
    # Mid-sentence pauses must not be treated as clip end.
    assert "stop_periods" not in af
    assert "areverse" in af


def test_af_filter_trim_still_removes_edges_only():
    af = build_af_filter(trim_silence=True)
    assert af.count("silenceremove=") == 2
    assert "stop_periods" not in af
