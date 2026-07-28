#!/usr/bin/env python3
"""Which podcasts publish <podcast:transcript>? Re-runnable survey.

    python scripts/survey_feeds.py "Talk Python To Me" "Darknet Diaries"
    python scripts/survey_feeds.py --file shows.txt

Results as of 2026-07-28 are recorded in docs/sources.md. Re-run rather than
trusting that file if it matters — publishers add and remove the tag.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"podcast": "https://podcastindex.org/namespace/1.0"}


def probe(term: str, checked: int = 20):
    """(show_name, feed_url, episodes_with_tag, episodes_checked, formats)."""
    q = urllib.parse.quote(term)
    d = json.load(urllib.request.urlopen(
        f"https://itunes.apple.com/search?term={q}&entity=podcast&limit=1",
        timeout=20))
    if not d["results"]:
        return term, None, 0, 0, set()
    r = d["results"][0]
    name, feed = r["collectionName"], r.get("feedUrl")
    if not feed:
        return name, None, 0, 0, set()
    raw = urllib.request.urlopen(
        urllib.request.Request(feed, headers={"User-Agent": "podrag/0.1"}),
        timeout=30).read()
    items = ET.fromstring(raw).findall(".//item")[:checked]
    n, fmts = 0, set()
    for it in items:
        ts = it.findall("podcast:transcript", NS)
        if ts:
            n += 1
        for t in ts:
            fmts.add((t.get("type") or "?").split("/")[-1])
    return name, feed, n, len(items), fmts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("shows", nargs="*")
    p.add_argument("--file", help="newline-separated show names")
    a = p.parse_args()
    shows = list(a.shows)
    if a.file:
        shows += [l.strip() for l in open(a.file) if l.strip()]
    if not shows:
        p.error("give show names or --file")

    hits = []
    print(f"{'show':<38}{'coverage':<12}formats")
    print("-" * 72)
    for s in shows:
        try:
            name, feed, n, tot, fmts = probe(s)
            print(f"{name[:37]:<38}{(f'{n}/{tot}' if feed else 'no feed'):<12}"
                  f"{','.join(sorted(fmts)) or '-'}")
            if n:
                hits.append((name, feed))
        except Exception as e:
            print(f"{s[:37]:<38}{'ERR':<12}{type(e).__name__}")

    print(f"\n{len(hits)}/{len(shows)} publish transcripts")
    for n, f in hits:
        print(f"  {n[:44]:<46}{f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
