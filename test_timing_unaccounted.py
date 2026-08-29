"""
The phase buckets only sum steps wrapped in log_timing, while the turn total is
wall clock. A turn can therefore report a small llm and tts while taking far
longer, with nothing in the breakdown showing where the time went. These cover
the gap being reported explicitly.
"""

from __future__ import annotations

import json

import flask

from request_timing import (
    HeaderPhases,
    HeaderTotalMs,
    attach_timing_headers,
    begin_request_timing,
    build_voice_timing_phases,
)


def test_tracked_ms_counts_only_the_labelled_steps():
    phases = build_voice_timing_phases(
        [
            "build prompt/messages=200",
            "llm npc response=1220",
            "piper segmented tts=835",
        ]
    )

    assert phases["tracked_ms"] == 200 + 1220 + 835


def _phases_for(steps: list[str], *, sleep_ms: int) -> dict:
    app = flask.Flask(__name__)
    with app.test_request_context("/"):
        begin_request_timing()
        flask.g._timing_steps = list(steps)
        # Stand in for wall clock the steps never measured.
        flask.g._timing_start -= sleep_ms / 1000.0
        response = attach_timing_headers(
            flask.Response("{}"), "req-1", include_voice_phases=True
        )
    return json.loads(response.headers[HeaderPhases]), response


def test_time_no_step_measures_is_reported():
    phases, response = _phases_for(["llm npc response=1220"], sleep_ms=11_900)

    # The turn took ~11.9s while the steps only explain 1.22s of it.
    assert int(response.headers[HeaderTotalMs]) >= 11_900
    assert phases["tracked_ms"] == 1220
    assert phases["unaccounted_ms"] >= 10_000


def test_a_fully_measured_turn_reports_no_gap():
    phases, _ = _phases_for(["llm npc response=1220"], sleep_ms=0)

    assert phases["unaccounted_ms"] == 0
