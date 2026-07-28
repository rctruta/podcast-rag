# podrag — findings

Observations with architectural consequences. Numbers live in the README and
go stale; what belongs here is what was *learned*, and what it implies for the
design. Each entry states whether it was MEASURED or ASSUMED.

---

## F-1. A fixed similarity threshold does not survive corpus growth

**MEASURED.** The same off-topic query — *"optimal torque spec for a 1997 Honda
Civic transmission"* against a health/podcast corpus — scored:

| corpus | best cosine similarity |
|---|---|
| 5 episodes (514 chunks) | 0.105 |
| 7 episodes (964 chunks) | **0.188** |

Same query, same embedding model, same floor (0.25). The margin to refusal
narrowed by ~44% purely because the corpus grew. Extrapolating, a large enough
corpus will push an unrelated query over any fixed floor and the system will
start answering questions it should refuse.

### Why this happens

Cosine similarity is **relative, not absolute**. It has no calibrated meaning —
0.19 is not "19% relevant". As a corpus grows, the chance that *something* is
superficially close to any query rises, so the maximum-over-corpus statistic
drifts upward regardless of whether real relevance improved. The threshold is
measuring the wrong thing: it treats a ranking score as a probability.

### Is this a deficit of RAG?

Yes, and a general one — not specific to this implementation or to LanceDB.
Any retrieve-then-generate system that decides "do I have enough to answer?"
from a raw embedding-similarity threshold has it. It is under-discussed
relative to retrieval quality itself, because most demos never implement
refusal at all and so never encounter it.

### What to do about it — options, with tradeoffs

1. **Relative rather than absolute signals.** Instead of `top_score > T`, use
   the *shape* of the score distribution: the gap between hit #1 and hit #k, or
   top score versus the median of retrieved. A genuinely relevant hit stands
   apart from its neighbours; an off-topic query produces a flat, undifferentiated
   distribution. Cheap, no extra model, and corpus-size-robust because it is
   scale-free. **Weakness:** fails when a corpus genuinely contains many equally
   good answers.
2. **Cross-encoder reranking.** A reranker scores (query, passage) jointly
   rather than comparing two independent embeddings, and its scores are far
   better calibrated. Standard practice, and it improves ordering as well as
   the refusal decision. **Cost:** a second model at query time, and latency.
3. **Calibrate on a labelled set, and re-calibrate.** Hold a set of answerable
   and unanswerable questions; set the floor where it separates them; re-run
   whenever the corpus changes materially. Turns a guess into a measurement.
   **Cost:** the labelled set is manual work, and it must be maintained.
4. **Ask the model.** A cheap LLM call — *"do these excerpts answer this
   question?"* — before synthesis. Robust to corpus size. **Weakness:** costs a
   call, and delegates the integrity decision to the component least able to
   report its own uncertainty, which is the thing this design is trying to
   avoid.

**Position for this project:** (1) plus (3). Relative signals for the mechanism
because they are scale-free and add no dependency; a labelled calibration set
because a threshold without one is an opinion. (2) is the obvious upgrade if
retrieval quality itself becomes the bottleneck. (4) is deliberately last —
the point of the refusal gate is to be mechanical, and re-introducing an LLM
judgement into it reintroduces exactly the failure mode being defended against.

**Status: unimplemented.** The floor is still fixed at 0.25 and still by
inspection. Recorded so the limitation is known rather than discovered.

---

## F-2. Different shows draw different material out of the same expert

**MEASURED.** Dr. a repeat guest on mitochondria, two shows — Diary Of A CEO
(2h38m, 206 chunks) and another show (3h16m, 244 chunks). Seven questions,
top-8 retrieval:

| | DOAC | another show |
|---|---|---|
| chunks retrieved (of 50) | 18 | 32 |
| ranked #1 | 4 | 3 |
| share of indexed material | 46% | 54% |

another show contributed 64% of retrieved chunks from 54% of the material — broader
coverage. But DOAC produced the single best passage more often.

**The asymmetry is by topic, not volume.** *"Sleep and cellular energy"*
returned 8 another show chunks and 0 DOAC in the top 8 — yet a keyword search
confirms DOAC **does** discuss sleep (4 hits). Covered less substantively, not
absent. Conversely *"how does stress affect mitochondria"* favoured DOAC 5–3.

**Architectural consequence:** this is the argument for indexing across shows
rather than picking the "best" one, and it argues against de-duplicating
overlapping episodes. It also means **per-show retrieval quotas** may be worth
testing — a naive top-k lets the longer show crowd out the better passage.
Untested.

---

## F-3. Persisting raw capture is what makes parameters tunable

**MEASURED.** Before a bronze layer, the pipeline was fetch → chunk → embed →
discard. Chunk size and embedding model were nominally configurable and
practically frozen, because changing either meant re-downloading everything.

With raw segments in parquet: re-chunking the full corpus at target sizes
120/180/400 yields 1605/964/393 chunks **entirely offline**, and warm re-ingest
of 964 chunks takes 1.72s with zero network calls. 22,067 segments compress to
567 KB (9.2×).

**Consequence:** the expensive, rate-limited, *externally-owned* step (someone
else's captions, which can change or vanish) happens exactly once. Everything
downstream is a local rebuild. This is the bronze tier of a medallion layout,
and the reason to adopt it here is not tidiness — it is that the upstream is
not under our control.

---

## F-4. Incremental indexing is no longer the hard part

**MEASURED.** Adding one new episode to a 964-chunk index: fetch + embed +
append took 10.2s, moved the index 964 → 967 chunks, and left the existing 964
untouched. LanceDB `add()` plus the bronze layer's record of already-captured
episodes makes "index the new episodes" a normal operation.

Worth noting because the 2024 predecessor made this awkward — delta handling
was manual. **ASSUMED:** that the difficulty was inherent to the earlier stack
rather than to how it was used; not re-tested.

---

## Open / worth testing

- **Per-show retrieval quotas** (from F-2) — does capping chunks-per-episode
  improve answer quality when one show is much longer?
- **Actian Vector as a retrieval backend.** Actian has moved into vector search;
  comparing it against LanceDB on the same corpus and queries would be a real
  benchmark rather than a vendor claim — and it reuses existing Actian
  familiarity from sqlbenchdag. Scope risk: this is a separate study, not a
  feature.
- **Relative-signal refusal** (from F-1) — implement and compare against the
  fixed floor on a labelled answerable/unanswerable set.
