"""Bronze layer: raw capture is persisted, idempotent, and re-derivable."""
import tempfile

from podrag.store import load_segments, save_segments, stored_episodes
from podrag.transcripts import Segment


def _segs(n=3, start=0.0):
    return [Segment(text=f"line {i}", start=start + i, duration=1.0) for i in range(n)]


def test_roundtrip_preserves_text_and_timing():
    with tempfile.TemporaryDirectory() as d:
        save_segments(_segs(5), show="s", episode_guid="e1",
                      episode_title="T", published="2026", root=d)
        rows = load_segments("s", root=d)
        assert len(rows) == 5
        assert rows[0]["text"] == "line 0"
        assert rows[4]["start"] == 4.0
        assert rows[0]["episode_title"] == "T"


def test_resaving_an_episode_replaces_rather_than_duplicates():
    """Ingest must be idempotent — re-running should not double the corpus."""
    with tempfile.TemporaryDirectory() as d:
        save_segments(_segs(5), show="s", episode_guid="e1",
                      episode_title="T", published="", root=d)
        save_segments(_segs(3), show="s", episode_guid="e1",
                      episode_title="T", published="", root=d)
        assert len(load_segments("s", root=d)) == 3


def test_multiple_episodes_coexist_in_one_show_file():
    with tempfile.TemporaryDirectory() as d:
        save_segments(_segs(2), show="s", episode_guid="e1",
                      episode_title="A", published="", root=d)
        save_segments(_segs(4), show="s", episode_guid="e2",
                      episode_title="B", published="", root=d)
        assert len(load_segments("s", root=d)) == 6
        assert len(load_segments("s", root=d, episode_guid="e2")) == 4
        assert stored_episodes("s", root=d) == {"e1", "e2"}


def test_segments_come_back_in_order():
    with tempfile.TemporaryDirectory() as d:
        save_segments(_segs(10), show="s", episode_guid="e1",
                      episode_title="T", published="", root=d)
        seqs = [r["seq"] for r in load_segments("s", root=d)]
        assert seqs == sorted(seqs)


def test_capture_records_when_it_was_taken():
    """Provenance of the raw tier itself — YouTube captions can change."""
    with tempfile.TemporaryDirectory() as d:
        save_segments(_segs(2), show="s", episode_guid="e1",
                      episode_title="T", published="", root=d)
        r = load_segments("s", root=d)[0]
        assert r["fetched_at"] and r["source"] == "youtube_captions"


def test_missing_store_fails_loudly():
    import pytest
    with tempfile.TemporaryDirectory() as d:
        assert stored_episodes("nope", root=d) == set()
        with pytest.raises(FileNotFoundError):
            load_segments("nope", root=d)
