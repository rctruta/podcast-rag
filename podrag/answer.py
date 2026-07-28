"""Answer synthesis over retrieved chunks — with a refusal path.

Two rules, both enforced here rather than requested in a prompt:

1. **Refuse below a confidence floor.** If nothing retrieved is close enough to
   the question, return a refusal instead of a synthesis. A RAG system that
   always answers is a system that will confidently answer from irrelevant
   context, and on health/advice content that is the failure that matters.

2. **Every answer carries its sources.** Citations are attached from the
   retrieved rows, not requested from the model, so they cannot be invented.

CONFIDENCE IS MEASURED INDEPENDENTLY OF SEARCH MODE. LanceDB's vector,
full-text and hybrid searches return scores on different scales (observed:
~0.5 vector, ~4.6 keyword, ~0.016 hybrid) which are not comparable — using
them as a threshold would make the refusal behaviour depend on which retrieval
mode was chosen. Instead we re-measure cosine similarity between the query and
each returned chunk with the same normalised embeddings. That number means the
same thing regardless of how the chunk was found.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from podrag.index import _encoder
from podrag.search import Hit, search

# Cosine similarity on normalised MiniLM embeddings.
#
# Was 0.25, chosen by inspection. Calibrated 2026-07-28 against
# eval/questions.yaml (42 labelled questions, 99-episode corpus) — see F-1 in
# docs/findings.md. The old value was not merely drifting toward failure; it
# had already failed: it opened the gate on 17 of 20 questions the corpus
# cannot answer, including "how long should I braise a lamb shoulder" (0.265)
# and "rules for castling in chess" (0.267). Precision 0.56.
#
# 0.54 is the F0.5-optimal threshold on that set (precision-weighted, because
# fabricating an answer costs more than refusing a good question). It beat a
# cross-encoder reranker and a cosine-AND-reranker rule under 6-fold
# cross-validation, both of which scored higher in-sample and generalised worse.
#
# This value is CORPUS-SPECIFIC and expires. Re-run `python eval/calibrate.py`
# after any material change to the corpus or the embedding model; the value
# here is checked against eval/calibration.json by tests/test_calibration.py,
# so the two cannot silently diverge.
DEFAULT_MIN_CONFIDENCE = float(os.environ.get("PODRAG_MIN_CONFIDENCE", "0.54"))

SYSTEM = """You answer questions using ONLY the podcast excerpts provided.

Rules:
- Use only what the excerpts say. Do not add outside knowledge.
- If the excerpts do not answer the question, say so plainly.
- Refer to speakers as the excerpts do. Do not invent names.
- Be concise. Do not pad.
- Do not write citations or timestamps yourself; they are attached separately."""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    reason: str | None = None
    error: str | None = None

    def render(self) -> str:
        if self.refused:
            return f"No confident answer.\n  reason: {self.reason}\n  best similarity: {self.confidence:.3f}"
        lines = [self.text, "", "Sources:"]
        for h in self.hits:
            lines.append(f"  · {h.citation()}  {h.url()}")
        return "\n".join(lines)


def score_confidence(question: str, hits: list[Hit]) -> list[float]:
    """Cosine similarity, query vs each hit. Mode-independent by construction."""
    if not hits:
        return []
    model = _encoder()
    qv = model.encode(question, normalize_embeddings=True)
    hv = model.encode([h.text for h in hits], normalize_embeddings=True)
    return [float(qv @ v) for v in hv]  # normalised -> dot product is cosine


def ask(question: str, *, k: int = 5, search_type: str = "hybrid",
        db_path: str = "./podrag.lance", show: str | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        model: str = "gpt-4o-mini", synthesize: bool = True) -> Answer:
    hits = search(question, search_type=search_type, k=k, db_path=db_path, show=show)
    if not hits:
        return Answer(question, "", [], 0.0, refused=True,
                      reason="retrieval returned nothing")

    sims = score_confidence(question, hits)
    ranked = sorted(zip(hits, sims), key=lambda t: t[1], reverse=True)
    best = ranked[0][1]

    if best < min_confidence:
        return Answer(question, "", [], best, refused=True,
                      reason=f"best similarity {best:.3f} < floor {min_confidence}")

    # keep only chunks that clear the floor — never pad context with noise
    kept = [h for h, s in ranked if s >= min_confidence]

    if not synthesize:
        return Answer(question, "(synthesis disabled)", kept, best)

    excerpts = "\n\n".join(
        f"[{h.episode_title} @ {h.timestamp}]\n{h.text}" for h in kept)
    user = f"Question: {question}\n\nExcerpts:\n\n{excerpts}"

    # Provider-agnostic via litellm: OpenAI, Anthropic, Gemini, Groq,
    # OpenRouter, or a LOCAL Ollama model. Local means a public deployment
    # costs the operator nothing, which is what makes this shareable.
    import litellm
    litellm.suppress_debug_info = True
    try:
        resp = litellm.completion(
            model=model, temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}])
    except Exception as e:
        # Retrieval already succeeded; losing the hits to a provider error
        # throws away work that was already done.
        return Answer(question, "", kept, best,
                      error=f"synthesis unavailable ({type(e).__name__}). "
                            f"Retrieval results shown below. Check the model "
                            f"name and that its key is set: {str(e)[:160]}")
    return Answer(question, (resp.choices[0].message.content or "").strip(),
                  kept, best)
