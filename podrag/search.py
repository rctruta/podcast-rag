"""Retrieval — vector, keyword, hybrid. Every hit carries its citation.

Mirrors the kw/vector/hybrid comparison from the 2024 course project, but the
three strategies now live in one embedded store instead of a hosted cluster.

The design rule: a hit that cannot state its source is not returned. Provenance
is not decoration added at answer time — it travels in the row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from podrag.index import _encoder, open_index

SearchType = Literal["vector", "keyword", "hybrid"]

# YouTube video ids: exactly 11 chars from [A-Za-z0-9_-].
_YOUTUBE_ID = re.compile(r"[A-Za-z0-9_-]{11}")


@dataclass(frozen=True)
class Hit:
    text: str
    show: str
    episode_guid: str
    episode_title: str
    published: str
    start_s: float
    end_s: float
    score: float

    @property
    def timestamp(self) -> str:
        m, s = divmod(int(self.start_s), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def citation(self) -> str:
        return f"{self.episode_title} · {self.timestamp}"

    def url(self) -> str:
        """Playable deep link.

        Was `len(guid) == 11` — a guess that any 11-character id is a YouTube
        video. Now an explicit character-class check, so an 11-char guid from
        some other source is not silently turned into a YouTube link.
        """
        if _YOUTUBE_ID.fullmatch(self.episode_guid):
            return f"https://youtu.be/{self.episode_guid}?t={int(self.start_s)}"
        return f"{self.episode_guid}#t={int(self.start_s)}"


def _to_hits(rows: list[dict]) -> list[Hit]:
    out = []
    for r in rows:
        score = r.get("_relevance_score", r.get("_distance", r.get("_score", 0.0)))
        # vector search reports distance (lower better); normalise to "higher better"
        if "_distance" in r and "_relevance_score" not in r:
            score = 1.0 / (1.0 + float(score))
        out.append(Hit(
            text=r["text"], show=r["show"], episode_guid=r["episode_guid"],
            episode_title=r["episode_title"], published=r["published"],
            start_s=float(r["start_s"]), end_s=float(r["end_s"]),
            score=float(score),
        ))
    return out


def search(query: str, *, search_type: SearchType = "hybrid", k: int = 5,
           db_path: str = "./podrag.lance", show: str | None = None) -> list[Hit]:
    tbl = open_index(db_path)

    if search_type == "keyword":
        q = tbl.search(query, query_type="fts")
    elif search_type == "vector":
        vec = _encoder().encode(query, normalize_embeddings=True).tolist()
        q = tbl.search(vec, query_type="vector")
    elif search_type == "hybrid":
        vec = _encoder().encode(query, normalize_embeddings=True).tolist()
        q = tbl.search(query_type="hybrid").vector(vec).text(query)
    else:
        raise ValueError(f"unknown search_type {search_type!r}")

    if show:
        q = q.where(f"show = '{show}'")
    return _to_hits(q.limit(k).to_list())
