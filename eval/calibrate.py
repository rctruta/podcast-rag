"""Calibrate the refusal gate against a labelled set (F-1).

The gate currently opens when raw cosine similarity clears a fixed 0.25. F-1
measured that floor drifting: the same off-topic probe scored 0.105 / 0.188 /
0.227 as the corpus grew 514 -> 964 -> 7,377 chunks. The margin collapsed 84%
without anything about the query, the model, or the floor changing.

This script does two things:

  1. MECHANISM. Tests why the drift happens, by subsampling the live corpus to
     the historical sizes and re-measuring. If the cause is the maximum-order
     statistic rising with N (an extreme-value effect), then a signal expressed
     in units of the corpus's own similarity spread should hold roughly still
     where raw cosine climbs.

  2. CALIBRATION. Sweeps a threshold over each candidate signal against
     eval/questions.yaml and reports precision / recall / F-beta, so the floor
     is chosen by measurement rather than inspection.

On the choice of beta. Define the positive class as "the gate OPENS and we
answer". Then:

    false positive = answered a question the corpus cannot support -> fabrication
    false negative = refused a question we could have answered  -> mild annoyance

Those costs are wildly asymmetric, and precision is the metric that punishes the
expensive one. F-beta weights RECALL by beta: beta > 1 favours recall, beta < 1
favours precision. So a refusal gate wants **beta < 1** (F0.5), not beta > 1.
All betas are reported so the asymmetry is visible rather than asserted.

Note also that F-beta is a way to SCORE a threshold, not a mechanism to replace
one. It selects the operating point; it does not make the underlying signal
robust to corpus growth. That is what part (1) is for.
"""
import math
import statistics
import sys
from dataclasses import dataclass

import lancedb
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

DB_PATH = "podrag.lance"
TABLE = "chunks"
QUESTIONS = "eval/questions.yaml"
MODEL = "all-MiniLM-L6-v2"

# The corpus sizes F-1 recorded, so the mechanism test reproduces that series.
HISTORICAL_N = [514, 964, 7377]

# Labels that mean "the gate should open".
POSITIVE = {"answerable"}


@dataclass
class Signals:
    """Every candidate refusal signal, computed for one query."""
    top_cos: float      # current gate: raw cosine of the best chunk
    z_bg: float         # (top1 - mean) / stdev, in units of this corpus's spread
    z_excess: float     # z_bg minus the z you'd EXPECT from N draws by chance
    gap12: float        # top1 - top2
    gap_med: float      # top1 - median of the top 20


def expected_max_z(n: int) -> float:
    """Expected maximum of n standard normal draws.

    This is the whole mechanism in one line. Take n samples from a distribution
    and the LARGEST one grows with n, all by itself, with no change in the
    underlying distribution. So "the best chunk scores 0.227" says as much about
    how many chunks there are as about whether any of them is relevant.

    Standard extreme-value approximation; accurate to a few percent for n > 100.
    """
    if n < 2:
        return 0.0
    a = math.sqrt(2.0 * math.log(n))
    return a - (math.log(math.log(n)) + math.log(4.0 * math.pi)) / (2.0 * a)


def signals_for(qv: np.ndarray, mat: np.ndarray) -> Signals:
    """Compute all candidate signals for one query against a chunk matrix.

    Uses the STORED chunk vectors, i.e. the same space retrieval ranks in.
    (podrag.answer.score_confidence re-encodes hit text at query time; for
    calibration we stay in the index's own space so the numbers describe the
    retrieval that actually happened.)
    """
    sims = mat @ qv                      # both sides L2-normalised -> cosine
    n = len(sims)
    top = np.partition(sims, -20)[-20:]
    top = np.sort(top)[::-1]

    mean = float(sims.mean())
    sd = float(sims.std())
    top1 = float(top[0])

    z = (top1 - mean) / sd if sd > 0 else 0.0
    return Signals(
        top_cos=top1,
        z_bg=z,
        # The correction: subtract off the elevation that corpus SIZE alone
        # buys you. What remains is how much better the best chunk is than the
        # best chance match you'd expect from a corpus this large.
        z_excess=z - expected_max_z(n),
        gap12=top1 - float(top[1]),
        gap_med=top1 - float(np.median(top)),
    )


def prf(tp: int, fp: int, fn: int, beta: float) -> float:
    """F-beta. beta > 1 weights recall, beta < 1 weights precision."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p == 0 and r == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom else 0.0


def sweep(values: list[float], labels: list[bool], beta: float):
    """Best threshold for one signal under one beta.

    Gate opens when signal >= threshold. Candidate thresholds are the midpoints
    between observed values, so every distinct partition of the data is tried.
    """
    order = sorted(set(values))
    cands = [order[0] - 1e-6] + [
        (order[i] + order[i + 1]) / 2 for i in range(len(order) - 1)
    ] + [order[-1] + 1e-6]

    best = None
    for t in cands:
        tp = sum(1 for v, y in zip(values, labels) if v >= t and y)
        fp = sum(1 for v, y in zip(values, labels) if v >= t and not y)
        fn = sum(1 for v, y in zip(values, labels) if v < t and y)
        tn = sum(1 for v, y in zip(values, labels) if v < t and not y)
        f = prf(tp, fp, fn, beta)
        if best is None or f > best[0]:
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            best = (f, t, p, r, tp, fp, fn, tn)
    return best


def separation(values: list[float], labels: list[bool]) -> tuple[float, float]:
    """Worst answerable vs best should-refuse.

    A threshold exists that is perfect on this set iff the first exceeds the
    second. This is threshold-free, so it measures the SIGNAL rather than a
    particular operating point.
    """
    pos = [v for v, y in zip(values, labels) if y]
    neg = [v for v, y in zip(values, labels) if not y]
    return (min(pos) if pos else float("nan"),
            max(neg) if neg else float("nan"))


def main() -> int:
    db = lancedb.connect(DB_PATH)
    df = db.open_table(TABLE).to_pandas()
    mat_full = np.vstack(df["vector"].to_numpy()).astype("float32")
    # Guard: the cosine identity below assumes unit-norm rows.
    norms = np.linalg.norm(mat_full, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"stored vectors are not unit-norm (min {norms.min():.4f}, "
        f"max {norms.max():.4f}); dot product would not be cosine"
    )

    spec = yaml.safe_load(open(QUESTIONS))["questions"]
    model = SentenceTransformer(MODEL)
    qs = [r["q"] for r in spec]
    labels = [r["label"] in POSITIVE for r in spec]
    qvs = model.encode(qs, normalize_embeddings=True)

    n_total = len(mat_full)
    print(f"corpus: {n_total} chunks | questions: {len(qs)} "
          f"({sum(labels)} answerable, {len(labels) - sum(labels)} should-refuse)\n")

    # --- part 1: mechanism ---------------------------------------------------
    print("=" * 72)
    print("MECHANISM: does raw cosine drift with corpus size where z_excess does not?")
    print("Off-topic probe, corpus subsampled to the sizes F-1 recorded.")
    print("=" * 72)
    probe = "What is the optimal torque spec for a 1997 Honda Civic transmission?"
    pv = model.encode(probe, normalize_embeddings=True)
    rng = np.random.default_rng(0)

    print(f"{'N':>7} {'top_cos':>9} {'z_bg':>8} {'E[max z]':>9} {'z_excess':>9}")
    for n in HISTORICAL_N:
        n = min(n, n_total)
        # Average over draws so the row reports the effect, not one lucky subset.
        acc = []
        for _ in range(15):
            idx = rng.choice(n_total, size=n, replace=False)
            acc.append(signals_for(pv, mat_full[idx]))
        print(f"{n:>7} {statistics.mean(s.top_cos for s in acc):>9.3f} "
              f"{statistics.mean(s.z_bg for s in acc):>8.2f} "
              f"{expected_max_z(n):>9.2f} "
              f"{statistics.mean(s.z_excess for s in acc):>9.2f}")
    print("\nRead: if top_cos climbs down the column while z_excess stays flat,")
    print("the drift is a corpus-size artifact and z_excess removes it.\n")

    # --- part 2: calibration -------------------------------------------------
    all_sig = [signals_for(qv, mat_full) for qv in qvs]
    fields = ["top_cos", "z_bg", "z_excess", "gap12", "gap_med"]

    print("=" * 72)
    print("SEPARATION (threshold-free): worst answerable vs best should-refuse")
    print("=" * 72)
    for f in fields:
        vals = [getattr(s, f) for s in all_sig]
        lo, hi = separation(vals, labels)
        ok = "SEPARABLE" if lo > hi else "overlapping"
        print(f"  {f:>9}: worst-answerable {lo:>7.3f} | best-refuse {hi:>7.3f} "
              f"| margin {lo - hi:>7.3f}  {ok}")

    print("\n" + "=" * 72)
    print("THRESHOLD SWEEP  (positive class = gate OPENS)")
    print("beta<1 favours precision (avoid fabricating) — the right side for a")
    print("refusal gate. beta>1 favours recall — the wrong side. Both shown.")
    print("=" * 72)
    for beta in (0.5, 1.0, 2.0):
        print(f"\n--- F{beta} ---")
        print(f"{'signal':>9} {'thresh':>9} {'prec':>6} {'recall':>7} "
              f"{'F':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
        for f in fields:
            vals = [getattr(s, f) for s in all_sig]
            fb, t, p, r, tp, fp, fn, tn = sweep(vals, labels, beta)
            print(f"{f:>9} {t:>9.3f} {p:>6.2f} {r:>7.2f} {fb:>6.3f} "
                  f"{tp:>3} {fp:>3} {fn:>3} {tn:>3}")

    # --- part 3: where the current gate stands -------------------------------
    print("\n" + "=" * 72)
    print("CURRENT GATE (top_cos >= 0.25) on this set")
    print("=" * 72)
    vals = [s.top_cos for s in all_sig]
    tp = sum(1 for v, y in zip(vals, labels) if v >= 0.25 and y)
    fp = sum(1 for v, y in zip(vals, labels) if v >= 0.25 and not y)
    fn = sum(1 for v, y in zip(vals, labels) if v < 0.25 and y)
    tn = sum(1 for v, y in zip(vals, labels) if v < 0.25 and not y)
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  precision {tp/(tp+fp) if tp+fp else 0:.2f}  "
          f"recall {tp/(tp+fn) if tp+fn else 0:.2f}  "
          f"F0.5 {prf(tp, fp, fn, 0.5):.3f}")

    print("\nPer-question detail (sorted by z_excess):")
    rows = sorted(zip(spec, all_sig), key=lambda x: x[1].z_excess, reverse=True)
    for rec, s in rows:
        mark = "OPEN " if s.top_cos >= 0.25 else "refuse"
        print(f"  {rec['label']:>10} | cos {s.top_cos:.3f} | zx {s.z_excess:>6.2f} "
              f"| now:{mark} | {rec['q'][:58]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
