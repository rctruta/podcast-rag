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


# --- metadata: API path vs keyless fallback (no network) ------------------

from podrag.youtube import VideoMeta, _iso8601_to_seconds


@pytest.mark.parametrize("iso,secs", [
    ("PT1H3M21S", 3801),
    ("PT45M32S", 2732),
    ("PT30S", 30),
    ("PT2H", 7200),
    ("", 0),
    ("garbage", 0),
])
def test_iso8601_durations_parse(iso, secs):
    assert _iso8601_to_seconds(iso) == secs


def test_topic_text_combines_the_fields_a_filter_needs():
    m = VideoMeta(video_id="v", title="Menopause 101", channel="c",
                  description="hormones and sleep", tags=("health", "women"),
                  source="api")
    t = m.topic_text
    assert "Menopause" in t and "health" in t and "hormones" in t


def test_oembed_metadata_is_thin_and_says_so():
    """The keyless path yields title only — the reason the API path exists.
    Asserted so nobody later assumes topic filtering works without a key."""
    m = VideoMeta(video_id="v", title="T", channel="c", source="oembed")
    assert m.topic_text == "T"
    assert m.tags == () and m.description == "" and m.duration_s == 0
