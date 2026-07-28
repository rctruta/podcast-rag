"""Refusal and citation integrity. No network, no LLM, no spend."""
from podrag.answer import Answer, DEFAULT_MIN_CONFIDENCE
from podrag.search import Hit


def _hit(t="text", start=90.0, guid="https://example.com/ep/1"):
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
    assert out.count("#t=") == 2, "each source renders a time-anchored link"


def test_citations_anchor_at_the_right_offset():
    h = _hit(start=1591.0)
    assert h.url() == "https://example.com/ep/1#t=1591"


def test_any_guid_shape_produces_a_time_anchor():
    """RSS guids are opaque publisher strings; none are special-cased."""
    for g in ("https://example.com/ep.mp3", "urn:uuid:abc", "episode-12"):
        assert _hit(guid=g, start=60.0).url() == f"{g}#t=60"


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


def test_no_source_specific_url_special_cases_remain():
    """A YouTube branch here guessed at video ids by string shape. With that
    source withdrawn (docs/findings.md F-0), no guid shape gets special
    treatment — RSS guids are opaque publisher strings."""
    import inspect

    from podrag import search
    assert "youtu" not in inspect.getsource(search).lower()

def test_provider_failure_returns_hits_instead_of_raising(monkeypatch):
    """Retrieval succeeded; only synthesis failed. Losing the hits to an
    exception discards work already done. Provider-agnostic since the switch
    to litellm — any provider error must degrade the same way."""
    import podrag.answer as A
    monkeypatch.setattr(A, "search", lambda *a, **k: [_hit("passage text")])
    monkeypatch.setattr(A, "score_confidence", lambda q, h: [0.9])
    import litellm
    def boom(*a, **k):
        raise RuntimeError("no credentials for provider")
    monkeypatch.setattr(litellm, "completion", boom)
    out = A.ask("q", db_path="/nonexistent")
    assert out.error and "synthesis unavailable" in out.error
    assert out.hits and not out.refused, "hits must survive a provider failure"
