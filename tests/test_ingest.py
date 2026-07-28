"""URL parsing and ingest reporting. No network."""
import pytest

from podrag.youtube import slug, video_id


@pytest.mark.parametrize("raw,expected", [
    ("REDACTED", "REDACTED"),
    ("https://youtu.be/REDACTED", "REDACTED"),
    ("https://youtu.be/REDACTED?si=IHuNHIFiu35rE-Ed", "REDACTED"),
    ("https://www.youtube.com/watch?v=REDACTED", "REDACTED"),
    ("https://www.youtube.com/watch?v=REDACTED&t=16s", "REDACTED"),
    ("  REDACTED  ", "REDACTED"),
])
def test_video_id_survives_real_share_link_shapes(raw, expected):
    """Share links carry ?si= tracking and &t= offsets; both must be stripped."""
    assert video_id(raw) == expected


@pytest.mark.parametrize("bad", ["", "not-a-url", "https://youtu.be/", "https://example.com/x"])
def test_unparseable_input_fails_loudly(bad):
    with pytest.raises(ValueError):
        video_id(bad)


def test_channel_name_becomes_a_stable_filter_key():
    assert slug("a wellness podcast") == "redacted-show"
    assert slug("TED Audio Collective!") == "tedaudiocollective"
    assert slug("") == "unknown"
