"""Unit tests for the llm_rules stage-direction placement contract."""

from __future__ import annotations

from stage_directions import normalize_stage_directions


def test_leading_direction_is_returned_untouched():
    text = "*Varra narrows her eyes.* State your business."
    assert normalize_stage_directions(text) == text


def test_plain_dialogue_is_returned_untouched():
    text = "Aye, the forge is open. What do you need?"
    assert normalize_stage_directions(text) == text


def test_trailing_direction_moves_to_the_front():
    assert normalize_stage_directions(
        "Aye, come in. *He turns back to the forge.*"
    ) == "*He turns back to the forge.* Aye, come in."


def test_middle_direction_moves_to_the_front():
    assert normalize_stage_directions(
        "Aye. *Torin wipes his hands.* What do you need?"
    ) == "*Torin wipes his hands.* Aye. What do you need?"


def test_multiple_directions_merge_in_original_order():
    assert normalize_stage_directions(
        "*Torin looks up.* Aye. *He turns back to the forge.*"
    ) == "*Torin looks up. He turns back to the forge.* Aye."


def test_merged_directions_get_sentence_punctuation():
    assert normalize_stage_directions(
        "Hold there. *snarls* Who goes?"
    ) == "*snarls.* Hold there. Who goes?"


def test_direction_lifted_from_mid_sentence_cleans_the_seam():
    assert normalize_stage_directions(
        "Aye, *he nods*, what now?"
    ) == "*he nods.* Aye, what now?"


def test_direction_only_reply_keeps_a_single_wrapper():
    assert normalize_stage_directions(
        "*The gate swings open.* *The guard steps aside.*"
    ) == "*The gate swings open. The guard steps aside.*"


def test_dm_check_tag_stays_on_its_own_final_line():
    normalized = normalize_stage_directions(
        "You will need a writ. *He folds his arms.*\n"
        "[DM CHECK: whether the party carries a council writ]"
    )
    assert normalized == (
        "*He folds his arms.* You will need a writ.\n"
        "[DM CHECK: whether the party carries a council writ]"
    )
    assert normalized.splitlines()[-1].startswith("[DM CHECK:")


def test_dm_check_tag_keeps_its_line_when_the_reply_is_only_beats():
    """The tag's newline is the whole remainder's first character, not seam debris."""
    normalized = normalize_stage_directions(
        "*Torin looks up.* *He sets down the hammer.*\n"
        "[DM CHECK: whether the party carries a writ]"
    )
    assert normalized == (
        "*Torin looks up. He sets down the hammer.*\n"
        "[DM CHECK: whether the party carries a writ]"
    )
    assert normalized.splitlines()[-1].startswith("[DM CHECK:")


def test_leading_hesitation_survives_a_misplaced_beat():
    """An opening ellipsis is the model's pacing, not leftovers from the lifted beat."""
    assert normalize_stage_directions(
        "... *he hesitates* I cannot say."
    ) == "*he hesitates.* ... I cannot say."


def test_leading_dash_survives_a_misplaced_beat():
    assert normalize_stage_directions(
        "— *she pauses* Aye."
    ) == "*she pauses.* — Aye."


def test_seam_debris_is_still_cleaned_when_the_reply_opens_with_a_beat():
    """The leading strip still has to run where the remainder really starts at a seam."""
    assert normalize_stage_directions(
        "*Torin looks up.*, and then he left. *He turns away.*"
    ) == "*Torin looks up. He turns away.* and then he left."


def test_dm_wait_narration_is_preserved_exactly():
    """Wait copy is a whole-turn narrator line; reflowing it would break matching."""
    from dm_channel import DM_PENDING_WAIT_NARRATION

    assert normalize_stage_directions(DM_PENDING_WAIT_NARRATION) == DM_PENDING_WAIT_NARRATION


def test_empty_and_blank_directions_are_ignored():
    assert normalize_stage_directions("") == ""
    assert normalize_stage_directions("Just words.") == "Just words."


def test_normalized_output_has_one_leading_direction():
    """The whole point of the contract: exactly one beat, and it opens the reply."""
    from stage_directions import STAGE_DIRECTION_PATTERN

    normalized = normalize_stage_directions(
        "*Looks up.* Aye. *Wipes hands.* Well? *Turns away.*"
    )
    assert normalized.startswith("*")
    assert len(STAGE_DIRECTION_PATTERN.findall(normalized)) == 1
