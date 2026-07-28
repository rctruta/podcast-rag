"""RSS transcript parsing — the sanctioned source. No network."""
import pytest
import xml.etree.ElementTree as ET

from podrag.rss_transcripts import (has_timings, parse, parse_json, parse_vtt,
                                    slug, transcripts_for_item)

VTT = """WEBVTT

00:00:01.000 --> 00:00:04.500
Welcome to the show.

00:00:04.500 --> 00:00:09.250
Today we discuss <i>data pipelines</i>.
"""

ITEM = """<item xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <title>Ep 1</title>
  <podcast:transcript url="https://x/t.html" type="text/html"/>
  <podcast:transcript url="https://x/t.vtt" type="text/vtt"/>
  <podcast:transcript url="https://x/t.json" type="application/json"/>
</item>"""


def test_vtt_cues_become_timed_segments():
    segs = parse_vtt(VTT)
    assert len(segs) == 2
    assert segs[0].start == 1.0 and abs(segs[0].duration - 3.5) < 1e-6
    assert "Welcome" in segs[0].text
    assert "<i>" not in segs[1].text, "inline markup must be stripped"


def test_timestamped_formats_are_preferred_over_untimed():
    """Citation fidelity is the ranking criterion — VTT before JSON before HTML."""
    refs = transcripts_for_item(ET.fromstring(ITEM))
    assert refs[0].mime == "text/vtt"
    assert refs[-1].mime == "text/html"


def test_json_transcript_shape():
    segs = parse_json('{"segments":[{"startTime":5,"endTime":9,"body":"hello"}]}')
    assert len(segs) == 1 and segs[0].start == 5.0 and segs[0].duration == 4.0


def test_json_tolerates_alternate_keys():
    segs = parse_json('[{"start":2,"end":3,"text":"hi"}]')
    assert segs and segs[0].text == "hi"


def test_untimed_formats_declare_themselves_untimed():
    """HTML/plain have no timings. Citations must degrade to episode-level
    rather than claim a position in the audio that was never known."""
    segs = parse("<p>one</p>\n\n<p>two</p>", "text/html")
    assert len(segs) == 2
    assert not has_timings(segs)


def test_timed_formats_report_timings():
    assert has_timings(parse(VTT, "text/vtt"))


def test_item_without_transcript_tag_yields_nothing():
    assert transcripts_for_item(ET.fromstring("<item><title>x</title></item>")) == []


@pytest.mark.parametrize("name,expected", [
    ("Data Engineering Podcast", "dataengineeringpodcast"),
    ("Talk Python To Me!", "talkpythontome"),
    ("", "unknown"),
])
def test_show_slug(name, expected):
    assert slug(name) == expected
