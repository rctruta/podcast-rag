"""Ingest from podcast RSS — publisher-provided transcripts only.

Replaces the YouTube path. An episode is indexed only when its feed declares a
<podcast:transcript>; otherwise it is skipped and the reason recorded. Nothing
is scraped and no access control is worked around.

Citations degrade honestly: with a timestamped format (VTT/SRT/JSON) they point
at a moment; with HTML or plain text they point at the episode, and
`timed=False` is recorded so the difference is visible rather than implied.
"""
from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET

from podrag.chunks import Chunk, chunk_words
from podrag.manifest import record as record_manifest
from podrag.rss_transcripts import (fetch, has_timings, parse, slug,
                                    transcripts_for_item)
from podrag.store import save_segments, stored_episodes
from podrag.transcripts import Segment, segments_to_words


NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}


def _text(el, path):
    f = el.find(path, NS)
    return (f.text or "").strip() if f is not None and f.text else ""


def fetch_feed(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "podrag/0.1"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read()


def ingest_feed(feed_url: str, *, limit: int = 5, show: str | None = None,
                transcripts_root: str = "./transcripts", refetch: bool = False,
                verbose: bool = True) -> tuple[list[Chunk], list[dict]]:
    raw = fetch_feed(feed_url)
    root = ET.fromstring(raw)
    channel = root.find("channel")
    show_title = _text(channel, "title") or "unknown"
    link = _text(channel, "link")
    show_key = show or slug(show_title)

    chunks: list[Chunk] = []
    report: list[dict] = []

    for item in root.findall(".//item")[:limit]:
        title = _text(item, "title") or "(untitled)"
        guid = _text(item, "guid") or title
        published = _text(item, "pubDate")
        enc = item.find("enclosure")
        episode_url = _text(item, "link") or (enc.get("url") if enc is not None else "")

        cached = (not refetch) and guid in stored_episodes(show_key, transcripts_root)
        if cached:
            from podrag.store import load_segments
            rows = load_segments(show_key, transcripts_root, episode_guid=guid)
            segs = [Segment(r["text"], r["start"], r["duration"]) for r in rows]
            origin, mime = "bronze", "cached"
        else:
            refs = transcripts_for_item(item)
            if not refs:
                report.append({"guid": guid, "title": title, "status": "no_transcript_tag"})
                if verbose:
                    print(f"  SKIP  {title[:52]} — publisher provides no transcript")
                continue
            ref = refs[0]
            try:
                segs = parse(fetch(ref.url), ref.mime)
            except Exception as e:
                report.append({"guid": guid, "title": title, "status": "fetch_failed",
                               "detail": type(e).__name__})
                if verbose:
                    print(f"  FAIL  {title[:52]} — {type(e).__name__}")
                continue
            if not segs:
                report.append({"guid": guid, "title": title, "status": "empty_transcript"})
                continue
            save_segments(segs, show=show_key, episode_guid=guid, episode_title=title,
                          published=published, root=transcripts_root,
                          source=f"rss:{ref.mime}")
            origin, mime = "fetched", ref.mime

        record_manifest(slug=show_key, show_name=show_title, channel_url=link,
                        episode_guid=guid, episode_title=title,
                        episode_url=episode_url, published=published,
                        root=transcripts_root)

        cs = chunk_words(segments_to_words(segs), show=show_key, episode_guid=guid,
                         episode_title=title, published=published)
        chunks += cs
        timed = has_timings(segs)
        report.append({"guid": guid, "title": title, "status": "ok", "origin": origin,
                       "mime": mime, "segments": len(segs), "chunks": len(cs),
                       "timed": timed})
        if verbose:
            print(f"  OK    {origin:>7} · {len(segs):>5} segs -> {len(cs):>3} chunks · "
                  f"{'timed' if timed else 'UNTIMED':<7} · {title[:40]}")

    return chunks, report
