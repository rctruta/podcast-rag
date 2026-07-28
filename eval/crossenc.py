"""Does a cross-encoder separate the near-misses that cosine cannot? (F-1)

calibrate.py measured a hard ceiling on every signal derived from bi-encoder
cosine: no threshold separates "the corpus answers this" from "the corpus is
ABOUT this but does not answer it". The clearest case is a question about the
2019 Stanford AI Index scoring 0.692 against a corpus containing the 2026
report — topically near-identical, propositionally wrong.

That is not a tuning problem. A bi-encoder embeds the question and the passage
independently, so nothing in the comparison can represent "same subject, wrong
year". A cross-encoder reads both together and can.

This script tests whether that theoretical advantage shows up in measurement.
Model runs locally; no API, no spend.
"""
import sys

import lancedb
import numpy as np
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer

from calibrate import POSITIVE, prf, separation, sweep

DB_PATH, TABLE, QUESTIONS = "podrag.lance", "chunks", "eval/questions.yaml"
BI = "all-MiniLM-L6-v2"
CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_K = 20


def main() -> int:
    db = lancedb.connect(DB_PATH)
    df = db.open_table(TABLE).to_pandas()
    mat = np.vstack(df["vector"].to_numpy()).astype("float32")
    texts = df["text"].tolist()

    spec = yaml.safe_load(open(QUESTIONS))["questions"]
    qs = [r["q"] for r in spec]
    labels = [r["label"] in POSITIVE for r in spec]

    bi = SentenceTransformer(BI)
    qvs = bi.encode(qs, normalize_embeddings=True)
    ce = CrossEncoder(CROSS)

    top_cos, ce_max = [], []
    for q, qv in zip(qs, qvs):
        sims = mat @ qv
        idx = np.argpartition(sims, -RERANK_K)[-RERANK_K:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        top_cos.append(float(sims[idx[0]]))
        # Max over the reranked candidates: "is there ANY passage here that
        # actually answers this?" — the question the gate needs settled.
        ce_max.append(float(max(ce.predict([(q, texts[i]) for i in idx]))))

    print(f"corpus {len(texts)} chunks | {len(qs)} questions "
          f"({sum(labels)} answerable, {len(labels)-sum(labels)} should-refuse)\n")

    print("=" * 70)
    print("SEPARATION: worst answerable vs best should-refuse")
    print("=" * 70)
    for name, vals in (("top_cos", top_cos), ("cross_enc", ce_max)):
        lo, hi = separation(vals, labels)
        print(f"  {name:>10}: worst-answerable {lo:>8.3f} | best-refuse {hi:>8.3f} "
              f"| margin {lo-hi:>8.3f}  {'SEPARABLE' if lo > hi else 'overlapping'}")

    print("\n" + "=" * 70)
    print("SWEEP (positive = gate opens; beta<1 favours precision)")
    print("=" * 70)
    for beta in (0.5, 1.0):
        print(f"\n--- F{beta} ---")
        print(f"{'signal':>10} {'thresh':>9} {'prec':>6} {'recall':>7} {'F':>6} "
              f"{'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
        for name, vals in (("top_cos", top_cos), ("cross_enc", ce_max)):
            fb, t, p, r, tp, fp, fn, tn = sweep(vals, labels, beta)
            print(f"{name:>10} {t:>9.3f} {p:>6.2f} {r:>7.2f} {fb:>6.3f} "
                  f"{tp:>3} {fp:>3} {fn:>3} {tn:>3}")

    print("\nPer-question (sorted by cross-encoder score):")
    for rec, c, tc in sorted(zip(spec, ce_max, top_cos),
                             key=lambda x: x[1], reverse=True):
        print(f"  {rec['label']:>10} | ce {c:>8.3f} | cos {tc:.3f} | {rec['q'][:56]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
