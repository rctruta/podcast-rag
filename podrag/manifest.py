"""Show/episode manifest — presentation metadata, kept out of the index.

Display name, channel link and episode links are per-SHOW and per-EPISODE, not
per-chunk. Storing them on every chunk would duplicate them ~1000x and force a
re-embed whenever a title changed. They live here instead, written at ingest
time and read by the UI.

The index keeps only what retrieval and citation need.
"""
from __future__ import annotations

import json
from pathlib import Path

FILENAME = "shows.json"


def path_for(root: str = "./transcripts") -> Path:
    return Path(root) / FILENAME


def load(root: str = "./transcripts") -> dict:
    p = path_for(root)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def record(*, slug: str, show_name: str, channel_url: str, episode_guid: str,
           episode_title: str, episode_url: str, published: str,
           duration_s: int = 0, root: str = "./transcripts") -> None:
    """Upsert one episode under its show. Idempotent."""
    data = load(root)
    show = data.setdefault(slug, {"name": show_name, "channel_url": channel_url,
                                  "episodes": {}})
    # refresh show-level fields in case metadata improved (oembed -> API)
    if show_name:
        show["name"] = show_name
    if channel_url:
        show["channel_url"] = channel_url
    show["episodes"][episode_guid] = {
        "title": episode_title, "url": episode_url,
        "published": published, "duration_s": duration_s,
    }
    p = path_for(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def show_of(slug: str, root: str = "./transcripts") -> dict:
    return load(root).get(slug, {})
