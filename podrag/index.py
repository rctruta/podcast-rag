"""LanceDB index — embedded, no server, no API key.

Carries forward the structure of a 2024 Weaviate course project (same
embedding model, all-MiniLM-L6-v2; same kw/vector/hybrid comparison) onto an
embedded store. The 2024 version needed a hosted Weaviate cluster and an API
key; this runs from a local directory, which means the repo is clonable and
runnable by anyone.

LanceDB gives three things that matter here (verified against docs.lancedb.com):
  * full-text + vector + hybrid search in one store, so retrieval strategies
    are comparable without a second system
  * versioned tables, so "which index version produced this answer" is
    answerable — the property the answer cache depends on
  * Lance/Arrow columnar format, continuous with the DuckDB side of this stack
"""
from __future__ import annotations

import os
from typing import Iterable

import lancedb
import pyarrow as pa

from podrag.chunks import Chunk

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
TABLE = "chunks"

SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    pa.field("text", pa.string()),
    # provenance — every field here exists so a hit can cite itself
    pa.field("show", pa.string()),
    pa.field("episode_guid", pa.string()),
    pa.field("episode_title", pa.string()),
    pa.field("published", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("start_s", pa.float32()),
    pa.field("end_s", pa.float32()),
])


def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def connect(db_path: str = "./podrag.lance"):
    return lancedb.connect(db_path)


def build_index(chunks: Iterable[Chunk], db_path: str = "./podrag.lance",
                mode: str = "overwrite"):
    """Embed and store. `mode='add'` appends a new show to an existing index."""
    chunks = list(chunks)
    if not chunks:
        raise ValueError("no chunks to index")

    model = _encoder()
    vectors = model.encode([c.text for c in chunks],
                           show_progress_bar=False, normalize_embeddings=True)

    rows = [{
        "vector": v.tolist(),
        "text": c.text,
        "show": c.show,
        "episode_guid": c.episode_guid,
        "episode_title": c.episode_title,
        "published": c.published,
        "chunk_index": c.chunk_index,
        "start_s": c.start_s,
        "end_s": c.end_s,
    } for c, v in zip(chunks, vectors)]

    db = connect(db_path)
    if mode == "add" and TABLE in db.table_names():
        tbl = db.open_table(TABLE)
        tbl.add(rows)
    else:
        tbl = db.create_table(TABLE, data=rows, schema=SCHEMA, mode="overwrite")

    # Full-text index enables keyword and hybrid search alongside vector.
    try:
        tbl.create_fts_index("text", replace=True)
    except Exception as e:  # non-fatal: vector search still works
        print(f"  (fts index skipped: {type(e).__name__}: {e})")
    return tbl


def open_index(db_path: str = "./podrag.lance"):
    db = connect(db_path)
    if TABLE not in db.table_names():
        raise FileNotFoundError(f"no index at {db_path}; run build_index first")
    return db.open_table(TABLE)


def index_version(tbl) -> int:
    """LanceDB table version — the identity an answer-cache entry pins to."""
    return tbl.version
