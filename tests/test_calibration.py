"""The refusal threshold must stay tied to the measurement that produced it.

Two failure modes this guards, both real:

1. Someone re-runs eval/combine.py, gets a new threshold, and never applies it
   to answer.py — so the shipped gate and the calibration disagree silently.
2. Someone edits the threshold in answer.py by hand, and the recorded
   provenance in eval/calibration.json no longer describes what ships.

Deliberately NOT skipped when the corpus is absent. These tests read two files
that are both committed, so they run everywhere and cannot quietly stop
checking. The corpus-dependent behaviour test below is separate and is honest
about needing an index.
"""
import json
import os

import pytest

from podrag.answer import DEFAULT_MIN_CONFIDENCE

CAL = os.path.join(os.path.dirname(__file__), "..", "eval", "calibration.json")


@pytest.fixture(scope="module")
def calibration():
    with open(CAL) as f:
        return json.load(f)


def test_shipped_threshold_matches_calibration(calibration):
    """answer.py and the calibration artifact must not drift apart."""
    # Read the module default rather than the env var, which a developer may
    # have set locally.
    shipped = float(os.environ.get("PODRAG_MIN_CONFIDENCE") or DEFAULT_MIN_CONFIDENCE)
    recorded = calibration["threshold"]
    assert abs(shipped - recorded) <= 0.011, (
        f"shipped floor {shipped} does not match eval/calibration.json "
        f"({recorded}). Either apply the re-calibrated value to "
        f"podrag/answer.py, or re-run `python eval/combine.py` if the code "
        f"is the newer of the two. They must never disagree silently."
    )


def test_calibration_is_precision_weighted(calibration):
    """beta < 1 for a refusal gate.

    The costly error is answering a question the corpus cannot support
    (a false positive). beta > 1 weights recall and would optimise for the
    cheap error instead. This asserts the direction of the tradeoff, which is
    a design decision, not a tuning detail.
    """
    assert calibration["beta"] < 1.0, (
        f"beta={calibration['beta']} weights recall. A refusal gate must "
        f"favour precision: fabricating an answer costs more than declining "
        f"a question it could have answered."
    )


def test_calibration_beat_the_previous_floor(calibration):
    """The change must be justified by its own numbers."""
    assert calibration["in_sample"]["precision"] > \
        calibration["previous_precision_on_this_set"], (
            "the calibrated threshold does not improve precision over the "
            "previous floor on the labelled set — do not ship it")


def test_alternatives_were_measured_not_assumed(calibration):
    """A rejected alternative must carry the number that rejected it."""
    alts = calibration.get("rejected_alternatives") or {}
    assert alts, "no alternatives recorded; the choice of signal is unevidenced"
    best_alt = max(alts.values())
    assert calibration["cross_validated_f_beta"] >= best_alt, (
        f"a rejected alternative ({best_alt}) cross-validates better than the "
        f"shipped signal ({calibration['cross_validated_f_beta']}). Ship the "
        f"better one or record why not.")


def test_threshold_is_not_the_old_uncalibrated_value(calibration):
    """0.25 was set by inspection and measured at 0.56 precision.

    It let through 17 of 20 unanswerable questions, including recipe and chess
    questions. Reverting to it is a regression, not a preference.
    """
    assert calibration["threshold"] != 0.25


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "podrag.lance")),
    reason="needs a built index (podrag.lance is gitignored; run ingest first)",
)
def test_off_topic_probes_are_refused():
    """Behaviour, not configuration: these must score below the floor.

    The torque-spec probe is the original F-1 series (0.105 -> 0.188 -> 0.227).
    The other two scored 0.265 and 0.267 — ABOVE the old 0.25 floor, which is
    the concrete evidence that it had already failed.
    """
    from podrag.answer import score_confidence
    from podrag.search import search

    for q in (
        "What is the optimal torque spec for a 1997 Honda Civic transmission?",
        "How long should I braise a lamb shoulder and at what temperature?",
        "What are the rules for castling in chess?",
    ):
        hits = search(q, k=5)
        top = max(score_confidence(q, hits)) if hits else 0.0
        assert top < DEFAULT_MIN_CONFIDENCE, (
            f"off-topic question scored {top:.3f} against a floor of "
            f"{DEFAULT_MIN_CONFIDENCE} — the gate would answer it")
