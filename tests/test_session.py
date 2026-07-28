"""Session memory and follow-up detection. No network."""
from podrag.session import Session


def test_history_is_bounded():
    s = Session(max_history=3)
    for i in range(6):
        s.add(f"q{i}", f"a{i}")
    assert len(s.exchanges) == 3
    assert s.exchanges[0].question == "q3"


def test_empty_session_has_no_followups():
    assert Session().looks_like_followup("what about for men?") is False


def test_detects_common_followup_shapes():
    s = Session(); s.add("what are the signs of menopause", "…")
    for q in ["what about for men?", "and the side effects?", "why is that?"]:
        assert s.looks_like_followup(q), q


def test_long_standalone_question_is_not_a_followup():
    s = Session(); s.add("prior", "…")
    q = ("what does the research say about mitochondrial dysfunction "
         "and its relationship to chronic fatigue in adults over fifty")
    assert not s.looks_like_followup(q)


def test_context_renders_prior_exchanges():
    s = Session(); s.add("q1", "a1"); s.add("q2", "a2")
    c = s.as_context()
    assert "q1" in c and "a2" in c


def test_refusals_are_recorded_as_such():
    s = Session(); s.add("q", "", refused=True)
    assert "(refused)" in s.as_context()


def test_standalone_questions_are_not_swept_up_as_followups():
    """The earlier heuristic substring-matched 'why', 'that', 'more', 'it '
    anywhere, which glued fresh questions to prior context."""
    s = Session(); s.add("what happens during menopause", "…")
    for q in ["why do mitochondria matter",
              "what causes gray hair",
              "how much exercise is enough",
              "what should women know about hormone therapy"]:
        assert not s.looks_like_followup(q), q


def test_genuine_followups_still_detected():
    s = Session(); s.add("what happens during menopause", "…")
    for q in ["what about for men?", "and exercise?", "does it affect sleep?",
              "but why?", "what else?"]:
        assert s.looks_like_followup(q), q
