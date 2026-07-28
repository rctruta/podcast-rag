# podrag

Ask a question about a podcast. Get an answer, and get sent to the exact
second someone said it.

```
Q: what should women know about menopause and hormone therapy?

Hormone replacement therapy is an evolving conversation, particularly
regarding its use during perimenopause, not just menopause...

Sources:
  · The Diary Of A CEO — Dr Rachel Rubin · 24:45
  · The Diary Of A CEO — Dr Rachel Rubin · 27:09
```

Every claim clickable, straight to the second it was said.

*Corpus: 99 episodes across 9 shows, 7,377 chunks — every transcript published
by the show itself. Every number below is measured on it.*

---

## Why I built this

There is an enormous amount of health information in long-form podcasts ([a wellness podcast](https://www.youtube.com/@redacted-show), [Andrew another show](https://www.youtube.com/@hubermanlab), [The Diary Of A CEO](https://www.youtube.com/@TheDiaryOfACEO), and others) and no way to keep track of it. 

Moreover, sometimes the information conflicts across sources, and you cannot verify a claim without re-listening to hours of audio.

AI-generated fake content now sits alongside the original content; I have personally seen rip-offs of famous podcast episodes passed off as original content.

Provenance was the point from the start: an answer that cannot show you where it came from is worse than no answer.

The value is across shows: the same question answered by several experts, across various different podcasts. Similarly, one expert appearing in multiple podcasts may provide additional content/answer to a similar question. Podcasts vary in length, and an expert may be constrained on time in one, but not other.
---

## The corpus

Two halves, deliberately. **Nothing here is scraped** — every show below
publishes its own transcripts via the Podcasting 2.0 `<podcast:transcript>`
RSS tag, which exists so clients can read them.

**Health & long-form interviews**
- *The Diary Of A CEO* — 25 episodes. Deep interviews with clinicians and
  researchers; substantial coverage of women's health, hormones, menopause,
  and longevity.

**Engineering & AI**
- *Data Engineering Podcast* · *Talk Python To Me* · *Practical AI* ·
  *Oxide and Friends* · *Python Bytes* · *Screaming in the Cloud* ·
  *Darknet Diaries* · *The Joe Reis Show*

So the same system answers *"what should women know about hormone therapy?"*
and *"how do you keep a data pipeline maintainable as it grows?"* — from
different shows, with citations either way.

Full survey of which shows publish transcripts, and which don't:
**[docs/sources.md](docs/sources.md)**

## What it does

- **Cites everything.** Every answer carries episode, timestamp, and a playable
  link. Citations come from the stored rows, never from the model, so they
  cannot be invented.
- **Refuses when it should.** Below a confidence floor it says so instead of
  synthesising from irrelevant context. Asked about Honda transmissions against
  this corpus, it returns *"no confident answer"* and makes no LLM call.
- **Works across shows.** The corpus is a parameter. One question, answers drawn
  from whichever episodes actually cover it.
- **Runs from a directory.** Embedded vector store, no server, no account.

## The cross-show claim, tested

The motivation above says the same expert can give more in one show than
another. That is measurable with this tool, so it was measured rather than
asserted.

**Dr. a repeat guest on mitochondria, two shows:**
Diary Of A CEO (2h38m, 206 chunks) and another show (3h16m, 244 chunks).
Seven questions, top-8 retrieval each, counting which episode the chunks came
from:

| | DOAC | another show |
|---|---|---|
| chunks retrieved (of 50) | 18 | **32** |
| questions where it ranked #1 | **4** | 3 |
| share of Picard chunks in index | 46% | 54% |

**Both shows are worth having, for different reasons.** another show returned 64%
of retrieved chunks from 54% of the indexed material — broader coverage,
consistent with the longer runtime. But DOAC produced the single best passage
on more questions.

The asymmetry is by topic, not just volume. *"What role does sleep play in
cellular energy"* returned 8 another show chunks and 0 from DOAC in the top 8 —
though a keyword search confirms DOAC **does** discuss sleep (4 hits). It is
covered less substantively there, not absent. Conversely *"how does stress
affect mitochondria"* favoured DOAC 5–3.

So the useful version of the claim is stronger than "longer episodes say more":
**different shows draw different material out of the same expert**, which is
the case for indexing across shows rather than picking one.

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
rebuilds from disk — measured on the 7-episode corpus: 964 chunks rebuilt in
1.72s, and 1605/964/393 chunks at target sizes 120/180/400, all offline.

| layer | choice | why |
|---|---|---|
| transcripts | publisher RSS `<podcast:transcript>` | the sanctioned source — VTT / SubRip / JSON, all timed |
| raw store | Parquet + zstd | columnar, 6.4× compression (59,832 segments → 2.8 MB), readable by DuckDB/Polars/pandas |
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

### UI

```bash
streamlit run app.py
```

Sidebar shows the corpus grouped by show, and exposes retrieval mode, passage
count, and the confidence floor — so the refusal threshold is something you can
move and observe rather than a hidden constant. Citations render as links that
open the episode at the cited second.

## Tests

```bash
pytest tests/ -q      # 46 tests, no network, no API keys, no spend
```

They pin the invariants rather than the implementation: provenance survives
chunking, timestamps stay ordered, chunks overlap so boundary passages aren't
lost, refusals carry no sources, citations resolve to real offsets, bronze
saves are idempotent, and the embedding dimension is derived rather than
asserted.

## Bring your own model — or none

**Retrieval is entirely local and free.** Embeddings run on your machine,
LanceDB is embedded. Search, citations and refusal all work with no key and no
account.

Written answers are optional and provider-agnostic (via `litellm`). Set any
one of these and the app finds it:

| provider | env var |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GEMINI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| **Ollama (local)** | none — just run it |

Ollama costs nothing and needs no account, which is what makes this shareable:
**nobody has to spend money to run it, and no operator ends up paying for
other people's queries.**

## When it costs money

The model is called in exactly two places, both visible in the UI:

| call | when | cost |
|---|---|---|
| follow-up rewrite | only when a question is detected as a follow-up | ~100 tokens |
| answer synthesis | only when "synthesize answer" is on and confidence clears the floor | 1 call over the retrieved passages |

So a question costs **0, 1, or 2 calls**. Each answer shows which, and the
sidebar keeps a running total for the session. Turn synthesis off and the app
is free to run — retrieval, citations and refusal all still work.

**If you host this with your own key, you pay for every visitor.** The repo
defaults to retrieval-only unless a key is present, and the provider list above
means a visitor can supply their own — or use Ollama and pay nothing.

## Findings

Observations with architectural consequences, each marked MEASURED or ASSUMED:
**[docs/findings.md](docs/findings.md)** · legitimate sources: **[docs/sources.md](docs/sources.md)**

- **F-1** — a fixed similarity threshold does not survive corpus growth. The
  same off-topic query scored 0.105 at 514 chunks, 0.188 at 964, and **0.227
  at 7,377** — against a 0.25 floor. Predicted at the second measurement,
  confirmed at the third. Four remedies, with tradeoffs and a position.
- **F-2** — different shows draw different material out of the same expert.
  Measured across two a repeat guest interviews; the asymmetry is by topic, not
  runtime.
- **F-3** — persisting raw capture is what makes chunk size and embedding model
  actually tunable.
- **F-4** — incremental indexing, measured.

## Not done

- Answer cache keyed on query + index version
- Incremental reindex for new episodes
- Episode-level topic filtering (metadata is captured; the filter isn't wired)
- **Retrieval evaluation.** Hit-rate/MRR need a hand-built golden set. Not
  claimed until that exists.
- **The 0.25 confidence floor is now the most likely next defect.** It is set
  by inspection, not calibrated, and it drifts with corpus size: 0.105 → 0.188
  → 0.227 on an identical off-topic query as the corpus grew to 7,377 chunks.
  The margin is down to 0.023. See F-1 for the four remedies; the relative-signal
  approach is the one to implement.

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

