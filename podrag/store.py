"""Bronze layer — raw transcripts persisted before anything derives from them.

WHY THIS EXISTS. Without it the pipeline is fetch -> chunk -> embed -> discard,
which makes every downstream parameter unchangeable in practice:

  * chunk size and overlap are tuning knobs, but re-tuning meant re-fetching
    every transcript
  * PODRAG_EMBED_MODEL is env-overridable, but swapping it meant re-fetching
  * YouTube can remove a video or regenerate its captions; the fetched text is
    then unrecoverable
  * "which transcript produced index v4?" had no answer

Parquet because the data is columnar, repeated, and read far more than written:
it compresses well, is typed, and is readable by DuckDB, Polars and pandas
without a loader. One file per show keeps re-ingest of a single show cheap.

This is the raw tier of a medallion layout:
    bronze  transcripts/<show>.parquet   raw segments, immutable
    silver  chunks (in memory / derived)  re-derivable at any chunk size
    gold    podrag.lance                  embedded + indexed, queryable

Bronze is the only tier that costs a network call. Everything else rebuilds
from it offline.
"""
from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from podrag.transcripts import Segment

SCHEMA = pa.schema([
    pa.field("show", pa.string()),
    pa.field("episode_guid", pa.string()),
    pa.field("episode_title", pa.string()),
    pa.field("published", pa.string()),
    pa.field("seq", pa.int32()),          # segment order within the episode
    pa.field("text", pa.string()),
    pa.field("start", pa.float32()),
    pa.field("duration", pa.float32()),
    pa.field("fetched_at", pa.string()),  # when the raw capture was taken
    pa.field("source", pa.string()),      # youtube_captions | whisper
])


def path_for(show: str, root: str = "./transcripts") -> Path:
    return Path(root) / f"{show}.parquet"


def save_segments(segments: list[Segment], *, show: str, episode_guid: str,
                  episode_title: str, published: str, root: str = "./transcripts",
                  source: str = "youtube_captions") -> Path:
    """Append one episode's raw segments. Re-saving an episode replaces its
    rows rather than duplicating them, so ingest is idempotent."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = {
        "show": [show] * len(segments),
        "episode_guid": [episode_guid] * len(segments),
        "episode_title": [episode_title] * len(segments),
        "published": [published] * len(segments),
        "seq": list(range(len(segments))),
        "text": [s.text for s in segments],
        "start": [s.start for s in segments],
        "duration": [s.duration for s in segments],
        "fetched_at": [now] * len(segments),
        "source": [source] * len(segments),
    }
    new = pa.table(rows, schema=SCHEMA)

    p = path_for(show, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = pq.read_table(p)
        keep = existing.filter(
            pa.compute.not_equal(existing["episode_guid"], episode_guid))
        new = pa.concat_tables([keep, new])
    pq.write_table(new, p, compression="zstd")
    return p


def load_segments(show: str, root: str = "./transcripts",
                  episode_guid: str | None = None) -> list[dict]:
    """Read raw segments back — no network."""
    p = path_for(show, root)
    if not p.exists():
        raise FileNotFoundError(f"no bronze transcripts at {p}")
    t = pq.read_table(p)
    if episode_guid:
        t = t.filter(pa.compute.equal(t["episode_guid"], episode_guid))
    return t.sort_by([("episode_guid", "ascending"), ("seq", "ascending")]).to_pylist()


def stored_episodes(show: str, root: str = "./transcripts") -> set[str]:
    """Which episodes are already captured — lets ingest skip refetching."""
    p = path_for(show, root)
    if not p.exists():
        return set()
    return set(pq.read_table(p, columns=["episode_guid"])["episode_guid"].to_pylist())
