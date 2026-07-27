# podrag — podcast RAG with verifiable citations

Ask a question, get an answer, get sent to the exact moment someone said it.

```
[26:31] manipulation and they use it quite strategically...
        https://youtu.be/REDACTED?t=1591
```

Corpus-agnostic: the podcast is a parameter, not the product.

## Design rule

**A retrieval hit that cannot state its source is not returned.** Provenance
is not added at answer time — episode id, title, publish date and start/end
offsets travel in the row, through every transform, and are pinned by tests.

## Stack

| layer | choice | why |
|---|---|---|
| transcripts | YouTube captions (free) · Whisper fallback | segment-level timestamps at zero cost |
| chunking | word-level with overlap | boundary passages stay retrievable |
| index | **LanceDB** (embedded) | vector + full-text + hybrid in one store; **no server, no API key** — the repo is clonable and runnable |
| embeddings | `all-MiniLM-L6-v2` | same model as the 2024 predecessor, for comparability |

LanceDB's **versioned tables** make "which index version produced this answer"
answerable — the property an answer-cache needs to invalidate correctly.

## Status — working

- [x] RSS ingestion, any feed (verified: Fixable 173 eps, ReThinking 214)
- [x] Free timestamped transcripts (verified: 1,646 segments / 1h03m episode)
- [x] Timestamp-preserving chunking with overlap
- [x] LanceDB index with vector / keyword / hybrid search
- [x] Playable deep-link citations
- [x] 13 tests, no network, no keys, no spend

## Not done

- [ ] Answer generation over retrieved chunks (retrieval works; synthesis isn't wired)
- [ ] Refuse-to-answer below a retrieval-confidence threshold
- [ ] Answer cache in DuckDB keyed on query + index version
- [ ] Incremental reindex (CocoIndex candidate — untested here)
- [ ] Retrieval evaluation: hit-rate / MRR over a **hand-built** golden set.
      Not claimed until that set exists.

**Known issue:** hybrid scores are on a different scale from vector and keyword
scores (~0.016 vs ~0.5 vs ~4.6). Blending works, comparability across modes does
not yet. Needs normalisation before any cross-mode measurement.

## Lineage

Structure carried forward from a 2024 Weaviate course project — same embedding
model, same kw/vector/hybrid comparison — rebuilt on an embedded store. The
2024 version required a hosted cluster and an API key; this runs from a local
directory. Transcripts are fetched at index time on the user's machine and are
never redistributed.

## Run

```bash
uv venv && uv pip install -e .
pytest tests/ -q
```
