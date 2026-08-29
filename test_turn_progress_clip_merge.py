"""
Published clips must never disappear from turn progress.

Clips are synthesized on a background thread while the request thread is still
finishing its own bookkeeping, so an older snapshot can be written last. The
poller fans out clips as it sees them, so a clip that vanishes before it is
polled is never played and the reply just stops mid-way.
"""

from __future__ import annotations

import turn_progress as tp


def _register(request_id: str) -> None:
    with tp._PROGRESS_LOCK:
        tp._PROGRESS.clear()
        tp._ACTIVE_TURNS.clear()
    tp.register_turn_progress(
        request_id,
        route="/npc-voice-turn",
        npc_id="mira_smallfoot",
        player_id="steven",
        mode="voice",
        campaign_id="watchvale",
    )


def _clip(index: int) -> dict:
    return {"type": "speech", "index": index, "url": f"/turn-audio/req/{index}", "text": f"part {index}"}


def test_a_stale_snapshot_cannot_drop_a_published_clip():
    rid = "a" * 32
    _register(rid)

    tp.set_turn_progress(rid, "speech_ready", clips=[_clip(0)])
    # Background thread publishes the rest of the reply...
    tp.set_turn_progress(rid, "speech_ready", clips=[_clip(0), _clip(1)])
    # ...while the request thread writes the snapshot it captured earlier.
    progress = tp.set_turn_progress(rid, "speech_ready", clips=[_clip(0)])

    assert [clip["index"] for clip in progress["clips"]] == [0, 1]


def test_clips_stay_in_play_order_however_they_arrive():
    rid = "b" * 32
    _register(rid)

    tp.set_turn_progress(rid, "speech_ready", clips=[_clip(2)])
    progress = tp.set_turn_progress(rid, "speech_ready", clips=[_clip(0), _clip(1)])

    assert [clip["index"] for clip in progress["clips"]] == [0, 1, 2]


def test_a_republished_clip_is_updated_not_duplicated():
    rid = "c" * 32
    _register(rid)

    tp.set_turn_progress(rid, "speech_ready", clips=[_clip(0)])
    revised = {**_clip(0), "text": "corrected"}
    progress = tp.set_turn_progress(rid, "speech_ready", clips=[revised])

    assert len(progress["clips"]) == 1
    assert progress["clips"][0]["text"] == "corrected"


def test_progress_without_clips_leaves_the_published_list_alone():
    rid = "d" * 32
    _register(rid)

    tp.set_turn_progress(rid, "speech_ready", clips=[_clip(0), _clip(1)])
    progress = tp.set_turn_progress(rid, "finishing")

    assert [clip["index"] for clip in progress["clips"]] == [0, 1]
