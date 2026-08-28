"""Unit + functional tests for structured speaker traits on /voices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("flask")

from tts_processing import (  # noqa: E402
    apply_speaker_annotation_sidecar,
    infer_model_language,
    load_piper_model_meta,
    load_speaker_annotation_sidecar,
    normalize_speaker_traits,
    parse_speaker_traits,
)


# --- parser ---------------------------------------------------------------


def test_parses_a_librittsp_style_descriptor_list():
    traits = parse_speaker_traits("masculine, calm, low-pitched, adult")
    assert traits["gender"] == "masculine"
    assert traits["age"] == "adult"
    assert traits["pitch"] == "low"
    assert traits["tags"] == ["calm"]


def test_pitch_wins_over_age_for_ambiguous_wording():
    # "very low-pitched" must not be mistaken for an age descriptor.
    traits = parse_speaker_traits("feminine, very low-pitched, elderly")
    assert traits["pitch"] == "low"
    assert traits["age"] == "elderly"
    assert traits["gender"] == "feminine"


@pytest.mark.parametrize(
    "annotation, expected",
    [
        ("female, high-pitched", "feminine"),
        ("male voice", "masculine"),
        ("androgynous", "neutral"),
        # LibriTTS-P's own hyphenated label. The word-level fallback splits on
        # spaces only, so this has to match as a whole descriptor or the
        # speaker gets a free-form tag and the gender filter loses them.
        ("gender-neutral", "neutral"),
        ("gender neutral, calm", "neutral"),
    ],
)
def test_gender_synonyms_collapse(annotation, expected):
    assert parse_speaker_traits(annotation)["gender"] == expected


def test_hyphenated_gender_neutral_is_a_facet_not_a_tag():
    traits = parse_speaker_traits("gender-neutral, warm")
    assert traits["gender"] == "neutral"
    assert "gender-neutral" not in traits["tags"]
    assert traits["tags"] == ["warm"]


def test_unknown_descriptors_survive_as_tags():
    traits = parse_speaker_traits("masculine, swashbuckling, calm")
    # The vocabulary is not exhaustive, so nothing may be silently dropped.
    assert "swashbuckling" in traits["tags"]
    # Curated tags sort ahead of free-form ones.
    assert traits["tags"] == ["calm", "swashbuckling"]


def test_descriptors_split_on_semicolons_slashes_and_and():
    traits = parse_speaker_traits("feminine; soft / warm and bright")
    assert traits["gender"] == "feminine"
    assert traits["tags"] == ["bright", "soft", "warm"]


@pytest.mark.parametrize(
    "annotation, key, expected",
    [
        ("adult-like", "age", "adult"),
        ("youthful-sounding", "age", "young"),
        ("male voice", "gender", "masculine"),
    ],
)
def test_hedged_descriptors_still_resolve(annotation, key, expected):
    # LibriTTS-P hedges freely; "adult-like" is the same facet as "adult".
    assert parse_speaker_traits(annotation)[key] == expected


def test_duplicate_descriptors_are_collapsed():
    assert parse_speaker_traits("calm, calm, calm")["tags"] == ["calm"]


@pytest.mark.parametrize("value", [None, "", "   ", [], {}])
def test_empty_annotations_yield_empty_traits(value):
    assert parse_speaker_traits(value) == {
        "gender": None,
        "age": None,
        "pitch": None,
        "tags": [],
    }


def test_only_the_first_descriptor_of_each_facet_wins():
    traits = parse_speaker_traits("masculine, feminine")
    assert traits["gender"] == "masculine"
    # The loser is not lost, it becomes a tag.
    assert "feminine" in traits["tags"]


# --- explicit traits normalization ----------------------------------------


def test_explicit_traits_are_lowercased_and_deduped():
    traits = normalize_speaker_traits(
        {"gender": "Feminine", "pitch": " High ", "tags": ["Warm", "warm", "Sweet"]}
    )
    assert traits == {
        "gender": "feminine",
        "age": None,
        "pitch": "high",
        "tags": ["warm", "sweet"],
    }


def test_explicit_traits_accept_a_comma_separated_tag_string():
    assert normalize_speaker_traits({"tags": "warm, bright"})["tags"] == ["warm", "bright"]


@pytest.mark.parametrize("value", [None, "text", {}, {"gender": "  "}, []])
def test_unusable_explicit_traits_return_none(value):
    assert normalize_speaker_traits(value) is None


# --- language -------------------------------------------------------------


@pytest.mark.parametrize(
    "path, expected",
    [
        ("en/en_US/Licensed/Clean 100/medium/clean100.onnx", "en_US"),
        ("de/de_DE/thorsten/high/thorsten.onnx", "de_DE"),
        ("misc/en_US-ryan-high.onnx", "en_US"),
        ("misc/unlabelled.onnx", ""),
    ],
)
def test_model_language_comes_from_the_path_layout(path, expected):
    assert infer_model_language(path) == expected


# --- sidecar merge --------------------------------------------------------


def _write_model(tmp_path: Path, sidecar: dict) -> Path:
    model = tmp_path / "en" / "en_US" / "pack" / "medium" / "pack.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    (model.parent / "pack.onnx.json").write_text(
        json.dumps({"speaker_id_map": {"alice": 0, "bob": 1}, "num_speakers": 2}),
        encoding="utf-8",
    )
    (model.parent / "pack.onnx.speakers.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return model


def test_sidecar_derives_traits_from_the_annotation(tmp_path: Path):
    model = _write_model(
        tmp_path,
        {"speakers": {"0": {"name": "Alice", "annotation": "feminine, warm, high-pitched"}}},
    )

    loaded = load_speaker_annotation_sidecar(model)
    assert loaded[0]["traits"]["gender"] == "feminine"
    assert loaded[0]["traits"]["pitch"] == "high"
    assert loaded[0]["traits"]["tags"] == ["warm"]


def test_explicit_sidecar_traits_beat_the_parser(tmp_path: Path):
    model = _write_model(
        tmp_path,
        {
            "speakers": {
                "0": {
                    "name": "Alice",
                    # The annotation says masculine; the hand-corrected traits win.
                    "annotation": "masculine, calm",
                    "traits": {"gender": "feminine", "tags": ["regal"]},
                }
            }
        },
    )

    loaded = load_speaker_annotation_sidecar(model)
    assert loaded[0]["traits"]["gender"] == "feminine"
    assert loaded[0]["traits"]["tags"] == ["regal"]


def test_traits_reach_the_model_meta_speaker_rows(tmp_path: Path):
    model = _write_model(
        tmp_path,
        {"speakers": {"1": {"annotation": "masculine, gravelly, low-pitched"}}},
    )

    meta = load_piper_model_meta(model)
    by_id = {row["id"]: row for row in meta["speakers"]}
    assert by_id[1]["traits"]["gender"] == "masculine"
    assert by_id[1]["traits"]["pitch"] == "low"
    # The un-annotated speaker is untouched rather than given empty traits.
    assert "traits" not in by_id[0]


def test_merge_ignores_sidecar_ids_the_model_cannot_synthesize():
    speakers = [{"id": 0, "name": "alice"}]
    merged = apply_speaker_annotation_sidecar(
        speakers,
        {9: {"annotation": "feminine", "traits": {"gender": "feminine"}}},
    )
    assert merged == [{"id": 0, "name": "alice"}]


# --- functional /voices ---------------------------------------------------


def _scan_for_real(server, monkeypatch):
    """The shared api fixture stubs the inventory; these tests need a real scan."""
    monkeypatch.setattr(
        server,
        "refresh_voice_inventory",
        lambda force=False: server._build_voice_inventory()[0],
    )


def test_voices_endpoint_serves_traits_and_language(api, tmp_path, monkeypatch):
    client, server, headers = api

    root = tmp_path / "Voices"
    model = root / "en" / "en_US" / "pack" / "medium" / "pack.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    (model.parent / "pack.onnx.json").write_text(
        json.dumps({"speaker_id_map": {"alice": 0, "bob": 1}, "num_speakers": 2}),
        encoding="utf-8",
    )
    (model.parent / "pack.onnx.speakers.json").write_text(
        json.dumps(
            {
                "speakers": {
                    "0": {"name": "Alice", "annotation": "feminine, warm, high-pitched"},
                    "1": {"name": "Bob", "annotation": "masculine, gravelly, low-pitched"},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "VOICES_DIR", root)
    _scan_for_real(server, monkeypatch)

    res = client.get("/voices?refresh=1", headers=headers)
    assert res.status_code == 200
    voices = res.get_json()["voices"]
    assert len(voices) == 1

    voice = voices[0]
    assert voice["language"] == "en_US"
    # Additive only: the fields the current editor reads are unchanged.
    assert voice["path"] == "en/en_US/pack/medium/pack.onnx"
    assert voice["quality"] == "medium"
    assert voice["num_speakers"] == 2

    # Keyed by id: the model's own speaker_id_map names win over the sidecar's,
    # which only fills in generic speaker_N placeholders.
    by_id = {row["id"]: row for row in voice["speakers"]}
    assert by_id[0]["traits"] == {
        "gender": "feminine",
        "age": None,
        "pitch": "high",
        "tags": ["warm"],
    }
    assert by_id[1]["traits"]["pitch"] == "low"


def test_voices_endpoint_omits_traits_when_there_is_no_sidecar(api, tmp_path, monkeypatch):
    client, server, headers = api

    root = tmp_path / "Voices"
    model = root / "en" / "en_US" / "solo" / "low" / "solo.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    (model.parent / "solo.onnx.json").write_text(
        json.dumps({"num_speakers": 1}), encoding="utf-8"
    )

    monkeypatch.setattr(server, "VOICES_DIR", root)
    _scan_for_real(server, monkeypatch)

    res = client.get("/voices?refresh=1", headers=headers)
    assert res.status_code == 200
    voice = res.get_json()["voices"][0]
    # The editor must degrade gracefully rather than see empty facets.
    assert voice["speakers"] == []
    assert voice["language"] == "en_US"
