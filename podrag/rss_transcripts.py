"""Transcripts from the podcast feed itself — the sanctioned source.

The Podcasting 2.0 <podcast:transcript> tag exists so clients can fetch a
transcript the publisher has chosen to distribute. Reading it is what the tag
is FOR: no scraping, no undocumented endpoint, no access control routed around.

WHY THIS REPLACED THE YOUTUBE PATH (2026-07-28). Verified, not assumed:
  * YouTube's Data API `captions.download` "requires the user to have
    permission to edit the video" — there is NO sanctioned way to fetch a
    third party's captions.
  * Therefore a third-party caption library must use the undocumented timedtext
    endpoint, which is circumvention. YouTube's ToS prohibits "automated
    means (such as robots, botnets or scrapers)" and "circumvent, disable...
    or otherwise interfere with any part of the Service".
  * The prohibition is on ACCESS, so keeping the data local does not cure it.
  * All episodes previously indexed reported license: "youtube" (Standard
    YouTube License) — no Creative Commons exemption.

Format preference is by timestamp fidelity: WebVTT and SubRip carry cue
timings, which is what makes a citation playable. HTML and plain text usually
do not, so they are last and yield a single untimed block.
"""
from __future__ import annotations

import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from podrag.transcripts import Segment

NS = {"podcast": "https://podcastindex.org/namespace/1.0"}

# best first — timestamped formats win
PREFERRED = ["text/vtt", "application/x-subrip", "application/srt",
             "application/json", "text/html", "text/plain"]


@dataclass(frozen=True)
class TranscriptRef:
    url: str
    mime: str
    language: str = ""


def transcripts_for_item(item: ET.Element) -> list[TranscriptRef]:
    refs = [TranscriptRef(url=t.get("url", ""), mime=(t.get("type") or "").lower(),
                          language=t.get("language", "") or "")
            for t in item.findall("podcast:transcript", NS) if t.get("url")]
    def rank(r: TranscriptRef) -> int:
        for i, m in enumerate(PREFERRED):
            if r.mime.startswith(m):
                return i
        return len(PREFERRED)
    return sorted(refs, key=rank)


def fetch(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "podrag/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------- parsers

_TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})")


def _secs(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_vtt(text: str) -> list[Segment]:
    """WebVTT. Cue timings give real start/end."""
    segs, start, end, buf = [], None, None, []
    for line in text.splitlines():
        line = line.strip()
        if "-->" in line:
            if start is not None and buf:
                segs.append(Segment(" ".join(buf).strip(), start, max(end - start, 0.0)))
            ts = _TS.findall(line)
            if len(ts) >= 2:
                start, end = _secs(*ts[0]), _secs(*ts[1])
            buf = []
        elif not line or line.upper().startswith(("WEBVTT", "NOTE", "STYLE")) or line.isdigit():
            continue
        else:
            buf.append(re.sub(r"<[^>]+>", "", line))
    if start is not None and buf:
        segs.append(Segment(" ".join(buf).strip(), start, max((end or start) - start, 0.0)))
    return [s for s in segs if s.text]


def parse_srt(text: str) -> list[Segment]:
    """SubRip — same cue structure, comma decimal separator."""
    return parse_vtt(text)


def parse_json(text: str) -> list[Segment]:
    """Podcasting 2.0 JSON transcript: {"segments":[{"startTime","endTime","body"}]}.
    Publishers vary, so key lookups are tolerant."""
    import json
    d = json.loads(text)
    raw = d.get("segments") if isinstance(d, dict) else d
    if not isinstance(raw, list):
        return []
    out = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        body = seg.get("body") or seg.get("text") or ""
        st = seg.get("startTime", seg.get("start", 0)) or 0
        et = seg.get("endTime", seg.get("end", st)) or st
        if body:
            out.append(Segment(str(body).strip(), float(st), max(float(et) - float(st), 0.0)))
    return out


def parse_plain(text: str) -> list[Segment]:
    """HTML or plain text — no timings available. One untimed block per
    paragraph, so citations degrade to episode-level rather than lying about
    a position in the audio."""
    body = re.sub(r"<[^>]+>", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    return [Segment(p, 0.0, 0.0) for p in paras]


def parse(text: str, mime: str) -> list[Segment]:
    m = mime.lower()
    if m.startswith("text/vtt"):
        return parse_vtt(text)
    if "subrip" in m or m.endswith("srt"):
        return parse_srt(text)
    if "json" in m:
        return parse_json(text)
    return parse_plain(text)


def has_timings(segs: list[Segment]) -> bool:
    return any(s.duration > 0 or s.start > 0 for s in segs)


def slug(name: str) -> str:
    """Show title -> stable filter key. Lives here now that youtube.py is gone."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower()) or "unknown"
