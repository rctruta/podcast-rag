"""The invariant: provenance survives every transform.

These are the tests that make the citation claim defensible. They need no
network, no API keys, no audio.
"""
import pytest

from podrag.chunks import Chunk, chunk_words
from podrag.feeds import parse_episodes

FEED = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel><item>
  <title>How to manage in times of uncertainty</title>
  <guid>ep-abc-123</guid>
  <pubDate>Mon, 27 Jul 2026 04:00:00 GMT</pubDate>
  <itunes:duration>37:28</itunes:duration>
  <enclosure url="https://example.com/a.mp3" type="audio/mpeg"/>
</item>
<item><title>No audio, skipped</title><guid>ep-no-audio</guid></item>
</channel></rss>"""


def _words(n, start=0.0, dt=0.5):
    return [{"word": f"w{i}", "start": start + i * dt, "end": start + (i + 1) * dt}
            for i in range(n)]


def test_feed_yields_metadata_a_citation_needs():
    eps = list(parse_episodes(FEED, show="fixable"))
    assert len(eps) == 1, "episodes without audio must be skipped"
    e = eps[0]
    assert e.guid == "ep-abc-123"
    assert e.title.startswith("How to manage")
    assert e.audio_url.endswith(".mp3")
    assert e.published and e.duration


def test_every_chunk_carries_its_source():
    cs = chunk_words(_words(500), show="fixable", episode_guid="ep-abc-123",
                     episode_title="T", published="P")
    assert cs
    for c in cs:
        assert c.episode_guid == "ep-abc-123"
        assert c.end_s > c.start_s
        assert c.text


def test_timestamps_are_monotonic_and_ordered():
    cs = chunk_words(_words(500), show="s", episode_guid="g",
                     episode_title="T", published="P")
    starts = [c.start_s for c in cs]
    assert starts == sorted(starts)
    assert [c.chunk_index for c in cs] == list(range(len(cs)))


def test_chunks_overlap_so_boundary_passages_stay_retrievable():
    cs = chunk_words(_words(400), show="s", episode_guid="g",
                     episode_title="T", published="P",
                     target_words=100, overlap_words=20)
    assert len(cs) >= 2
    assert cs[1].start_s < cs[0].end_s, "no overlap — boundary passages lost"


def test_citation_is_human_readable_and_playable():
    c = Chunk(show="s", episode_guid="g", episode_title="Ep Title", published="P",
              chunk_index=0, text="t", start_s=3725.0, end_s=3730.0)
    assert c.timestamp == "1:02:05"
    assert c.citation() == "Ep Title · 1:02:05"


def test_short_timestamp_has_no_hour_component():
    c = Chunk(show="s", episode_guid="g", episode_title="T", published="P",
              chunk_index=0, text="t", start_s=95.0, end_s=99.0)
    assert c.timestamp == "1:35"


def test_empty_input_yields_no_chunks():
    assert chunk_words([], show="s", episode_guid="g", episode_title="T", published="P") == []


def test_bad_overlap_config_fails_loudly():
    with pytest.raises(ValueError):
        chunk_words(_words(10), show="s", episode_guid="g", episode_title="T",
                    published="P", target_words=50, overlap_words=50)


# --- transcript adaptation (no network) -----------------------------------

from podrag.transcripts import Segment, segments_to_words


def test_segments_expand_to_words_covering_the_segment_span():
    segs = [Segment(text="a b c d", start=10.0, duration=4.0)]
    ws = segments_to_words(segs)
    assert len(ws) == 4
    assert ws[0]["start"] == 10.0
    assert abs(ws[-1]["end"] - 14.0) < 1e-6, "words must span the segment exactly"


def test_word_timestamps_stay_ordered_across_segments():
    segs = [Segment("one two", 0.0, 2.0), Segment("three four", 2.0, 2.0)]
    ws = segments_to_words(segs)
    starts = [w["start"] for w in ws]
    assert starts == sorted(starts)


def test_empty_and_whitespace_segments_are_dropped():
    segs = [Segment("   ", 0.0, 1.0), Segment("real", 1.0, 1.0)]
    ws = segments_to_words(segs)
    assert [w["word"] for w in ws] == ["real"]


def test_zero_duration_segment_does_not_divide_by_zero():
    ws = segments_to_words([Segment("a b", 5.0, 0.0)])
    assert len(ws) == 2 and all(w["start"] == 5.0 for w in ws)


def test_end_to_end_shape_from_segments_to_citable_chunks():
    segs = [Segment(f"word{i} " * 20, float(i * 10), 10.0) for i in range(20)]
    chunks = chunk_words(segments_to_words(segs), show="s", episode_guid="vid123",
                         episode_title="Ep", published="2026")
    assert chunks
    for c in chunks:
        assert c.episode_guid == "vid123"
        assert c.end_s > c.start_s
        assert ":" in c.timestamp
