"""Refusal and citation integrity. No network, no LLM, no spend."""
from podrag.answer import Answer, DEFAULT_MIN_CONFIDENCE
from podrag.search import Hit


def _hit(t="text", start=90.0, guid="REDACTED"):
    return Hit(text=t, show="s", episode_guid=guid, episode_title="Ep",
               published="2026", start_s=start, end_s=start + 10, score=0.5)


def test_refusal_renders_without_inventing_an_answer():
    a = Answer("q", "", [], 0.1, refused=True, reason="below floor")
    out = a.render()
    assert "No confident answer" in out
    assert "below floor" in out


def test_answered_render_lists_every_source():
    a = Answer("q", "some answer", [_hit(start=90.0), _hit(start=3725.0)], 0.6)
    out = a.render()
    assert "some answer" in out
    assert "1:30" in out and "1:02:05" in out
    assert out.count("youtu.be") == 2


def test_citations_are_playable_deep_links_at_the_right_offset():
    h = _hit(start=1591.0)
    assert h.url() == "https://youtu.be/REDACTED?t=1591"


def test_non_youtube_guid_falls_back_to_time_fragment():
    h = _hit(guid="https://example.com/ep.mp3", start=60.0)
    assert h.url().endswith("#t=60")


def test_confidence_floor_is_explicit_not_hidden():
    """The floor is a named constant that can be argued with, not a magic
    number buried in a conditional."""
    assert 0.0 < DEFAULT_MIN_CONFIDENCE < 1.0


def test_refused_answer_carries_no_sources():
    """A refusal must not leak the low-relevance chunks as if they supported
    an answer."""
    a = Answer("q", "", [], 0.1, refused=True, reason="r")
    assert a.hits == []
    assert "Sources:" not in a.render()


# --- regressions for hardcoding found 2026-07-27 -------------------------

def test_embed_dim_is_derived_from_the_model_not_asserted():
    """Was EMBED_DIM = 384. Swapping PODRAG_EMBED_MODEL would then either
    corrupt the schema or fail far from the cause."""
    from podrag.index import embed_dim, schema
    d = embed_dim()
    assert d > 0
    field = schema().field("vector")
    assert field.type.list_size == d


def test_youtube_detection_uses_a_character_class_not_a_length_guess():
    """Was `len(guid) == 11`, which turned any 11-char id into a YouTube link."""
    assert _hit(guid="REDACTED").url().startswith("https://youtu.be/")
    # 11 chars, but not a YouTube id shape
    assert _hit(guid="episode/12.").url().endswith("#t=90")


def test_missing_api_key_returns_hits_instead_of_raising(monkeypatch):
    """Retrieval succeeded; only synthesis is unavailable. Losing the hits to
    a KeyError throws away work already done (and paid for in compute)."""
    import podrag.answer as A
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(A, "search", lambda *a, **k: [_hit("passage text")])
    monkeypatch.setattr(A, "score_confidence", lambda q, h: [0.9])
    out = A.ask("q", db_path="/nonexistent")
    assert out.error and "OPENAI_API_KEY" in out.error
    assert out.hits and not out.refused
