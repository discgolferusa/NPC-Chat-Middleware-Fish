"""Unit tests for Piper TTS text prep and option resolution."""

import json
import re
import shutil
import subprocess

import pytest

from tts_processing import (
    VALID_FX_PRESETS,
    asterisk_body_tts_type,
    build_af_filter,
    clean_for_tts,
    curated_defaults_for_quality,
    infer_dataset_name,
    infer_quality_tier,
    is_nonverbal_stage_direction,
    load_piper_model_meta,
    options_from_request_and_npc,
    resolve_tts_options,
)


def test_clean_for_tts_expands_and_pronounces():
    text = clean_for_tts("Dr. Torin walked to Golarion.")
    assert "Doctor" in text
    assert "go LAR ee un" in text
    assert text.endswith(".")


def test_asterisk_body_tts_type_keeps_stage_beats_on_narrator():
    assert is_nonverbal_stage_direction("Varra narrows her eyes.")
    assert is_nonverbal_stage_direction("Snarls.")
    assert is_nonverbal_stage_direction("Waiting on a decision from game master.")
    assert is_nonverbal_stage_direction("Waiting on a DM decision.")
    assert asterisk_body_tts_type("Xill cocks its head.") == "narrator"


def test_asterisk_body_tts_type_puts_spoken_lines_on_npc():
    # Language-barrier / denied-claim dialogue often arrives fully italicized.
    assert asterisk_body_tts_type("Krah-vesh?") == "npc"
    assert asterisk_body_tts_type("Skraa-vek krah.") == "npc"
    assert asterisk_body_tts_type("I do not understand your tongue.") == "npc"
    assert not is_nonverbal_stage_direction("Krah-vesh?")


def test_asterisk_body_tts_type_rejects_mixed_action_and_dialogue():
    # A stage verb anywhere must not force the whole italics span onto narrator.
    assert asterisk_body_tts_type("Snarls. I will not open the gate.") == "npc"
    assert asterisk_body_tts_type("He looks at you coldly. Leave now.") == "npc"
    assert asterisk_body_tts_type("I raise my voice and refuse.") == "npc"
    assert asterisk_body_tts_type("She turns away and says the vault stays sealed.") == "npc"
    assert asterisk_body_tts_type("You shake the latch, but it holds.") == "npc"
    assert asterisk_body_tts_type(
        "He looks toward the door and refuses without another word."
    ) == "npc"


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


def test_load_model_meta_merges_speaker_annotation_sidecar(tmp_path):
    model_dir = tmp_path / "Clean 100" / "medium"
    model_dir.mkdir(parents=True)
    model = model_dir / "clean100.onnx"
    model.touch()
    (model_dir / "clean100.onnx.json").write_text(
        json.dumps(
            {
                "dataset": "clean100",
                "num_speakers": 2,
                "speaker_id_map": {"Benjamin Parker": 0, "Teresa Richardson": 1},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "clean100.onnx.speakers.json").write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {
                    "0": {
                        "name": "Benjamin Parker",
                        "annotation": "masculine, adult-like, cool, sincere",
                    },
                    "1": "feminine, soft, kind",
                },
            }
        ),
        encoding="utf-8",
    )

    meta = load_piper_model_meta(model)
    assert meta["speakers"] == [
        {
            "id": 0,
            "name": "Benjamin Parker",
            "annotation": "masculine, adult-like, cool, sincere",
        },
        {
            "id": 1,
            "name": "Teresa Richardson",
            "annotation": "feminine, soft, kind",
        },
    ]
    assert meta["num_speakers"] == 2


def test_sidecar_does_not_invent_speakers_missing_from_model(tmp_path):
    model_dir = tmp_path / "Clean 100" / "medium"
    model_dir.mkdir(parents=True)
    model = model_dir / "clean100.onnx"
    model.touch()
    (model_dir / "clean100.onnx.json").write_text(
        json.dumps(
            {
                "dataset": "clean100",
                "num_speakers": 2,
                "speaker_id_map": {"Benjamin Parker": 0, "Teresa Richardson": 1},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "clean100.onnx.speakers.json").write_text(
        json.dumps(
            {
                "speakers": {
                    "0": {"annotation": "cool"},
                    "99": {"name": "Ghost", "annotation": "should not appear"},
                }
            }
        ),
        encoding="utf-8",
    )

    meta = load_piper_model_meta(model)
    assert [s["id"] for s in meta["speakers"]] == [0, 1]
    assert meta["num_speakers"] == 2
    assert meta["speakers"][0]["annotation"] == "cool"
    assert all(s["id"] != 99 for s in meta["speakers"])


def test_build_speaker_annotation_sidecar_skips_unknown_ids(tmp_path):
    import importlib.util
    from pathlib import Path

    try:
        import openpyxl
    except ImportError:
        import pytest

        pytest.skip("openpyxl not installed")

    onnx_json = tmp_path / "model.onnx.json"
    onnx_json.write_text(
        json.dumps({"speaker_id_map": {"Alice": 0, "Bob": 1}}),
        encoding="utf-8",
    )
    xlsx = tmp_path / "desc.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(
        [
            "Filename",
            "Friendly Name",
            "Piper Speaker ID",
            "Annotator 1",
            "Annotator 2",
            "Annotator 3",
        ]
    )
    ws.append(["a", "Alice", 0, "", "", "calm"])
    ws.append(["ghost", "Ghost", 99, "", "", "nope"])
    wb.save(xlsx)

    script = Path(__file__).resolve().parent / "scripts" / "build_speaker_annotation_sidecar.py"
    spec = importlib.util.spec_from_file_location("build_speaker_annotation_sidecar", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    payload = mod.build_sidecar(xlsx, onnx_json)
    assert set(payload["speakers"].keys()) == {"0"}
    assert payload["speakers"]["0"]["annotation"] == "calm"


def test_build_speaker_annotation_sidecar_priority():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent / "scripts" / "build_speaker_annotation_sidecar.py"
    spec = importlib.util.spec_from_file_location("build_speaker_annotation_sidecar", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    idx = {"Annotator 1": 0, "Annotator 2": 1, "Annotator 3": 2}
    assert mod.pick_annotation(("a1", "a2", "a3"), idx) == "a3"
    assert mod.pick_annotation(("a1", "a2", ""), idx) == "a1"
    assert mod.pick_annotation(("", "a2", None), idx) == "a2"
    assert mod.pick_annotation(("", "", ""), idx) == ""


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
    assert "makeup=1.2" in robot
    assert "volume=0.78" in robot
    assert "alimiter=limit=0.90:level=disabled" in robot
    assert "highpass=f=450" in whisper
    assert "lowpass=f=4200" in whisper
    assert "tremolo=" in whisper
    assert "volume=0.36" in whisper
    assert "makeup=1," in whisper
    assert "alimiter=limit=0.55:level=disabled" in whisper


def test_af_filter_acompressor_makeup_in_ffmpeg_range():
    """FFmpeg acompressor makeup must be in [1, 64]; values <1 fail graph init."""
    for preset in sorted(VALID_FX_PRESETS):
        af = build_af_filter(fx_preset=preset, trim_silence=False)
        for match in re.finditer(r"makeup=([0-9.]+)", af):
            makeup = float(match.group(1))
            assert 1.0 <= makeup <= 64.0, f"{preset}: makeup={makeup} out of range"


def test_af_filter_preset_limiters_disable_auto_level():
    """Default alimiter level=enabled multiplies by 1/limit and undoes volume cuts."""
    for preset in ("robot", "whisper"):
        af = build_af_filter(fx_preset=preset, trim_silence=False)
        for match in re.finditer(r"alimiter=[^,]+", af):
            stage = match.group(0)
            if "limit=0.95" in stage:
                continue  # shared baseline before dynaudnorm
            assert "level=disabled" in stage, f"{preset}: {stage}"


def test_af_filter_robot_whisper_ffmpeg_accepts_graph():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not on PATH")
    for preset in ("robot", "whisper"):
        af = build_af_filter(fx_preset=preset, trim_silence=False)
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=22050:cl=mono",
                "-t",
                "0.05",
                "-af",
                af,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{preset} af rejected: {proc.stderr}"


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


def test_resolve_uses_default_denoise_when_request_omits_knobs():
    opts = resolve_tts_options({})
    assert opts["denoise_nr"] == 12
    assert opts["denoise_nf"] == -25


def test_options_from_request_treat_null_denoise_as_default():
    # Preview proxy used to forward denoise_nr: null; resolve must keep defaults.
    opts = options_from_request_and_npc(
        {"tts_length_scale": 1.2, "tts_denoise_nr": None, "tts_denoise_nf": None},
        None,
    )
    assert opts["length_scale"] == 1.2
    assert opts["denoise_nr"] == 12
    assert opts["denoise_nf"] == -25


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
