"""Podcast feed ingestion — corpus-agnostic.

A show is a parameter: give it any RSS feed URL and it yields episodes with
the metadata a citation needs. Verified 2026-07-27 against TED Audio Collective
shows (Fixable, ReThinking) — both Acast-hosted, same structure. Neither
publishes `podcast:transcript` tags, so audio + Whisper is the transcript path,
which is what gives us word-level timestamps anyway.
"""
from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Iterator

NS = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "podcast": "https://podcastindex.org/namespace/1.0",
}

# Known-good feeds. Not a closed list — any RSS URL works.
KNOWN_SHOWS = {
    "fixable": "https://feeds.acast.com/public/shows/67572068d59c6635eea1c5fc",
    "rethinking": "https://feeds.acast.com/public/shows/675858676d1777b3683ec351",
}


@dataclass(frozen=True)
class Episode:
    """Everything a citation needs, captured at ingestion.

    guid is the stable identity: chunks reference it, the cache keys on it,
    and re-ingesting the same episode must not create a second identity.
    """
    show: str
    guid: str
    title: str
    published: str
    audio_url: str
    duration: str | None
    link: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def _text(el, path: str) -> str | None:
    found = el.find(path, NS)
    return found.text if found is not None and found.text else None


def fetch_feed(feed_url: str) -> bytes:
    req = urllib.request.Request(feed_url, headers={"User-Agent": "podrag/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def parse_episodes(raw: bytes, show: str) -> Iterator[Episode]:
    root = ET.fromstring(raw)
    for item in root.findall(".//item"):
        enc = item.find("enclosure")
        if enc is None or not enc.get("url"):
            continue  # no audio, nothing to transcribe
        guid = _text(item, "guid") or enc.get("url")
        yield Episode(
            show=show,
            guid=guid,
            title=_text(item, "title") or "(untitled)",
            published=_text(item, "pubDate") or "",
            audio_url=enc.get("url"),
            duration=_text(item, "itunes:duration"),
            link=_text(item, "link"),
        )


def load_show(show: str, feed_url: str | None = None, limit: int | None = None) -> list[Episode]:
    url = feed_url or KNOWN_SHOWS.get(show)
    if not url:
        raise ValueError(f"unknown show {show!r}; pass feed_url explicitly "
                         f"(known: {sorted(KNOWN_SHOWS)})")
    eps = list(parse_episodes(fetch_feed(url), show))
    return eps[:limit] if limit else eps


def has_feed_transcripts(raw: bytes) -> bool:
    """Podcasting 2.0 transcript tags, if a show ever publishes them —
    then we can skip Whisper for that show."""
    return b"podcast:transcript" in raw
