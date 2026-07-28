"""Which gate should ship? Honest, out-of-sample comparison. (F-1)

calibrate.py and crossenc.py both pick a threshold on the same 42 questions
they then score it with. That is in-sample: it reports the best this signal
could POSSIBLY do on this exact set, which is an optimistic bound, not an
estimate of behaviour on the next question a user asks.

This script fits the threshold on training folds and scores on held-out folds,
repeated over shuffles. The gap between the in-sample and cross-validated
numbers is itself the finding — it says how much of the apparent quality was
threshold-fitting.

Strategies compared:
  current      raw cosine >= 0.25 (what ships today; no fitting, so CV == in-sample)
  cos          raw cosine, threshold fitted
  cross        cross-encoder max over reranked candidates, threshold fitted
  cos AND ce   open only if BOTH clear their own fitted thresholds

The AND rule is motivated by the measured error structure, not by taste: the
cosine false positives are topical near-misses (Rust internals, gRPC-vs-GraphQL)
that the cross-encoder scores very low, while the cross-encoder's misses are
conversational passages that do answer the question but do not look like the
short factoid passages it was trained on. The errors land in different places.
"""
import json
import sys

import lancedb
import numpy as np
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer

from calibrate import POSITIVE, prf

DB_PATH, TABLE, QUESTIONS = "podrag.lance", "chunks", "eval/questions.yaml"
RERANK_K = 20
BETA = 0.5          # precision-weighted: a fabricated answer costs more than a refusal
FOLDS, REPEATS = 6, 40


def fit_threshold(vals, labels, beta=BETA):
    """Best threshold on the training fold."""
    order = sorted(set(vals))
    cands = [order[0] - 1e-6] + [(order[i] + order[i+1]) / 2
                                 for i in range(len(order)-1)] + [order[-1] + 1e-6]
    best = (-1.0, cands[0])
    for t in cands:
        tp = sum(1 for v, y in zip(vals, labels) if v >= t and y)
        fp = sum(1 for v, y in zip(vals, labels) if v >= t and not y)
        fn = sum(1 for v, y in zip(vals, labels) if v < t and y)
        f = prf(tp, fp, fn, beta)
        if f > best[0]:
            best = (f, t)
    return best[1]


def score(pred, labels, beta=BETA):
    tp = sum(1 for p, y in zip(pred, labels) if p and y)
    fp = sum(1 for p, y in zip(pred, labels) if p and not y)
    fn = sum(1 for p, y in zip(pred, labels) if not p and y)
    tn = sum(1 for p, y in zip(pred, labels) if not p and not y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return prf(tp, fp, fn, beta), p, r, tp, fp, fn, tn


def main() -> int:
    db = lancedb.connect(DB_PATH)
    df = db.open_table(TABLE).to_pandas()
    mat = np.vstack(df["vector"].to_numpy()).astype("float32")
    texts = df["text"].tolist()

    spec = yaml.safe_load(open(QUESTIONS))["questions"]
    qs = [r["q"] for r in spec]
    labels = np.array([r["label"] in POSITIVE for r in spec])

    bi = SentenceTransformer("all-MiniLM-L6-v2")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    qvs = bi.encode(qs, normalize_embeddings=True)

    cos, cem = [], []
    for q, qv in zip(qs, qvs):
        sims = mat @ qv
        idx = np.argpartition(sims, -RERANK_K)[-RERANK_K:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        cos.append(float(sims[idx[0]]))
        cem.append(float(max(ce.predict([(q, texts[i]) for i in idx]))))
    cos, cem = np.array(cos), np.array(cem)

    n = len(qs)
    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("cos", "cross", "and")}

    for _ in range(REPEATS):
        order = rng.permutation(n)
        for f in range(FOLDS):
            te = order[f::FOLDS]
            tr = np.array([i for i in order if i not in set(te.tolist())])
            if labels[tr].all() or not labels[tr].any():
                continue

            tc = fit_threshold(cos[tr].tolist(), labels[tr].tolist())
            te_ce = fit_threshold(cem[tr].tolist(), labels[tr].tolist())

            acc["cos"].append(score(cos[te] >= tc, labels[te])[0])
            acc["cross"].append(score(cem[te] >= te_ce, labels[te])[0])
            acc["and"].append(
                score((cos[te] >= tc) & (cem[te] >= te_ce), labels[te])[0])

    print(f"n={n} questions ({int(labels.sum())} answerable, "
          f"{int((~labels).sum())} should-refuse) | "
          f"F{BETA} | {FOLDS}-fold x {REPEATS} shuffles\n")

    print("=" * 68)
    print(f"{'strategy':>12} {'in-sample':>10} {'cross-val':>10} {'gap':>7}")
    print("=" * 68)

    cur = score(cos >= 0.25, labels)
    print(f"{'current .25':>12} {cur[0]:>10.3f} {cur[0]:>10.3f} {0.0:>7.3f}   "
          f"(no fitting: prec {cur[1]:.2f} rec {cur[2]:.2f} "
          f"TP{cur[3]} FP{cur[4]} FN{cur[5]} TN{cur[6]})")

    for name, vals in (("cos", cos), ("cross", cem)):
        t = fit_threshold(vals.tolist(), labels.tolist())
        ins = score(vals >= t, labels)
        cv = float(np.mean(acc[name]))
        print(f"{name:>12} {ins[0]:>10.3f} {cv:>10.3f} {ins[0]-cv:>7.3f}   "
              f"(thresh {t:.3f}, prec {ins[1]:.2f} rec {ins[2]:.2f} "
              f"TP{ins[3]} FP{ins[4]} FN{ins[5]} TN{ins[6]})")

    tc = fit_threshold(cos.tolist(), labels.tolist())
    te_ = fit_threshold(cem.tolist(), labels.tolist())
    ins = score((cos >= tc) & (cem >= te_), labels)
    cv = float(np.mean(acc["and"]))
    print(f"{'cos AND ce':>12} {ins[0]:>10.3f} {cv:>10.3f} {ins[0]-cv:>7.3f}   "
          f"(cos>={tc:.3f} & ce>={te_:.3f}, prec {ins[1]:.2f} rec {ins[2]:.2f} "
          f"TP{ins[3]} FP{ins[4]} FN{ins[5]} TN{ins[6]})")

    print("\nRead: 'cross-val' is the honest number. A large 'gap' means the")
    print("in-sample figure was mostly threshold-fitting to these 42 questions.")

    # Emit the shipping value as an artifact rather than leaving a human to
    # transcribe it into answer.py. tests/test_calibration.py asserts the two
    # agree, so a re-calibration that is never applied fails the suite.
    cos_t = fit_threshold(cos.tolist(), labels.tolist())
    cos_ins = score(cos >= cos_t, labels)
    out = {
        "generated_by": "eval/combine.py",
        "beta": BETA,
        "signal": "top cosine, query vs chunk (bi-encoder, all-MiniLM-L6-v2)",
        "threshold": round(float(cos_t), 3),
        "corpus_chunks": int(len(texts)),
        "questions": int(n),
        "in_sample": {"f_beta": round(cos_ins[0], 3),
                      "precision": round(cos_ins[1], 3),
                      "recall": round(cos_ins[2], 3)},
        "cross_validated_f_beta": round(float(np.mean(acc["cos"])), 3),
        "rejected_alternatives": {
            "cross_encoder": round(float(np.mean(acc["cross"])), 3),
            "cos_and_cross_encoder": round(float(np.mean(acc["and"])), 3),
        },
        "previous_threshold": 0.25,
        "previous_precision_on_this_set": round(score(cos >= 0.25, labels)[1], 3),
    }
    with open("eval/calibration.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"\nwrote eval/calibration.json (threshold {out['threshold']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
