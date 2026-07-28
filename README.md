# podrag

Ask a question about a podcast. Get an answer, and get sent to the exact
second someone said it.

```
Q: what happens to hormones during menopause and what can help?

During menopause, estrogen levels decrease significantly, leading to bone
loss, increased cardiovascular risk, and changes in body composition...

Sources:
  · The #1 Menopause Doctor · 43:40    https://youtu.be/REDACTED?t=2620
  · The Women's Hormone Health Episode · 1:01:31  https://youtu.be/REDACTED?t=3691
  · The Most Important Conversation About Women's Health · 9:04  https://youtu.be/REDACTED?t=544
```

Three episodes, one answer, every claim clickable.

---

## Why I built this

<!-- YOUR VOICE — the notes below are your own words from the design
     conversation, left as a scaffold to rewrite, not as final prose.

     - there is an enormous amount of health information in long-form podcasts
       (another show, a wellness podcast, DOAC, and others) and no way to keep track of it
     - the information conflicts across sources, and you cannot verify a claim
       without re-listening to hours of audio
     - AI-generated fake content now sits alongside the real thing
     - you have personally seen rip-offs of famous podcast episodes passed off
       as original
     - so: provenance was the point from the start. an answer that cannot show
       you where it came from is worse than no answer
     - and the value is across shows, not within one: the same question answered
       by several experts is what you actually want
-->

**[write this section]**

---

## What it does

- **Cites everything.** Every answer carries episode, timestamp, and a playable
  link. Citations come from the stored rows, never from the model, so they
  cannot be invented.
- **Refuses when it should.** Below a confidence floor it says so instead of
  synthesising from irrelevant context. Asked about Honda transmissions against
  a health corpus, it returns *"no confident answer, best similarity 0.105"* —
  and makes no LLM call.
- **Works across shows.** The corpus is a parameter. One question, answers drawn
  from whichever episodes actually cover it.
- **Runs from a directory.** Embedded vector store, no server, no account.

## How it's built

```
 RSS / YouTube ids
        │
        ▼
   bronze   transcripts/<show>.parquet     raw segments, immutable, fetched once
        │
        ▼
   silver   chunks                          derived at any size, offline
        │
        ▼
   gold     podrag.lance                    embedded, versioned, queryable
```

Only bronze touches the network. Re-chunking or swapping the embedding model
rebuilds from disk — measured: 514 chunks in 1.25s, and 855/514/210 chunks at
target sizes 120/180/400, all offline.

| layer | choice | why |
|---|---|---|
| transcripts | YouTube captions | free, timestamped, no key |
| metadata | YouTube Data API v3 | description + tags + dates; oembed fallback when no key |
| raw store | Parquet + zstd | columnar, 9.5× compression, readable by DuckDB/Polars/pandas |
| index | LanceDB | vector + full-text + hybrid in one embedded store; versioned tables |
| embeddings | `all-MiniLM-L6-v2` | local, free, swappable via `PODRAG_EMBED_MODEL` |

**Provenance is structural, not decorative.** `show`, `episode_guid`,
`episode_title`, `published`, `start_s`, `end_s` are columns on every chunk,
carried through every transform and pinned by tests. A hit that cannot state
its source is not returned.

## Use

```bash
uv venv && uv pip install -e .
echo 'YT_API_KEY=...' > .env          # optional; falls back to oembed
python -m podrag.cli index REDACTED --show redacted-show
python -m podrag.cli ask "what are the signs of manipulation?"
```

`--no-synth` returns retrieval only, with no LLM call and no spend.

## Tests

```bash
pytest tests/ -q      # 46 tests, no network, no API keys, no spend
```

They pin the invariants rather than the implementation: provenance survives
chunking, timestamps stay ordered, chunks overlap so boundary passages aren't
lost, refusals carry no sources, citations resolve to real offsets, bronze
saves are idempotent, and the embedding dimension is derived rather than
asserted.

## Not done

- Answer cache keyed on query + index version
- Incremental reindex for new episodes
- Episode-level topic filtering (metadata is captured; the filter isn't wired)
- **Retrieval evaluation.** Hit-rate/MRR need a hand-built golden set. Not
  claimed until that exists.
- The 0.25 confidence floor is set by inspection, not calibrated.

**Known issue:** LanceDB's vector, full-text and hybrid searches return scores
on different scales (~0.5 / ~4.6 / ~0.016). Blending works; cross-mode score
comparison does not. Confidence is measured separately by cosine similarity for
exactly this reason.

## Lineage

Structure carried forward from a 2024 Weaviate course project — same embedding
model, same keyword/vector/hybrid comparison. That version needed a hosted
cluster and an API key; this runs from a local directory.

Transcripts are fetched at index time on the user's machine and are never
redistributed. Point it at a corpus you have the rights to use.
