"""Threshold sweep for jev_conf_router+rerank, computed offline from one steelman run.

The router returns hybrid+rerank's list when Jev's best document scores at or above τ, and
hybrid_diverse+rerank's list otherwise. So for any τ its per-query NDCG is one of the two
arms' values, and the whole curve comes from a single run with --conf-router (which records
each query's signal). The run's own router arm is checked against this reconstruction.

Reported per τ: rewrite rate, NDCG@10, share of the full fusion gain kept, and a permutation
test against random routing at the same rate: the observed router is compared with 10,000
random choices of the same number of queries to fuse. A small one-sided p means the signal
picks the queries that need fusion better than chance. τ=0.5 is the pre-registered threshold
(the evidence gate's); the rest of the curve is descriptive.

  python -m eval.conf_router_sweep experiments/jev-confidence-routing/results/<run>.json
"""

import argparse
import json
import random

from eval.bootstrap_ci import percentile

HYBRID, DIVERSE, ROUTER = "hybrid+rerank", "hybrid_diverse+rerank", "jev_conf_router+rerank"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def permutation_test(delta, fuse, b=10000, seed=42):
    """delta[i] = diverse − hybrid for query i. Returns (observed − mean random,
    95% range of random − mean random, one-sided p) for fusing the same number of queries."""
    rng, n, m = random.Random(seed), len(delta), sum(fuse)
    observed = sum(x for x, f in zip(delta, fuse) if f) / n
    rand = [sum(rng.sample(delta, m)) / n for _ in range(b)]
    centre = sum(rand) / b
    p = (1 + sum(r >= observed for r in rand)) / (b + 1)
    return observed - centre, (percentile(rand, 0.025) - centre, percentile(rand, 0.975) - centre), p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--pre-registered", type=float, default=0.5)
    args = parser.parse_args()

    run = json.load(open(args.run))
    pq, sig = run["per_query"], run["jev"]["decisions"]["conf_route"]
    qids = list(pq[HYBRID])
    h = [pq[HYBRID][q][args.metric] for q in qids]
    d = [pq[DIVERSE][q][args.metric] for q in qids]
    top = [sig[q]["max_relevance"] for q in qids]
    n = len(qids)
    mean = lambda xs: sum(xs) / len(xs)
    gain = mean(d) - mean(h)

    if ROUTER in pq:
        tau = run["config"]["conf_threshold"]
        recon = [d[i] if top[i] < tau else h[i] for i in range(n)]
        actual = [pq[ROUTER][q][args.metric] for q in qids]
        mismatched = sum(abs(a - b) > 1e-9 for a, b in zip(recon, actual))
        print(f"router arm at τ={tau}: {mean(actual):.4f}, reconstruction {mean(recon):.4f}, "
              f"{mismatched} queries differ")

    print(f"\nhybrid+rerank {mean(h):.4f}   hybrid_diverse+rerank {mean(d):.4f}   "
          f"full fusion gain {gain:+.4f}   oracle {mean([max(a, b) for a, b in zip(h, d)]):.4f}")
    delta = [b - a for a, b in zip(h, d)]
    print(f"\n{'τ':>5} | rewrite rate | {args.metric} | gain kept | vs random routing "
          f"(random 95% range) | p")
    print("-" * 92)
    rows = []
    for tau in sorted(set(THRESHOLDS + [args.pre_registered])):
        fuse = [t < tau for t in top]
        rate = sum(fuse) / n
        routed = [d[i] if fuse[i] else h[i] for i in range(n)]
        diff, (lo, hi), p = permutation_test(delta, fuse)
        kept = (mean(routed) - mean(h)) / gain if gain else float("nan")
        mark = " *" if tau == args.pre_registered else ""
        print(f"{tau:>5} | {rate:>11.0%} | {mean(routed):.4f}  | {kept:>8.0%} | "
              f"{diff:+.4f} [{lo:+.4f}, {hi:+.4f}] | {p:.3f}{mark}")
        rows.append({"threshold": tau, "rewrite_rate": rate, "metric": mean(routed),
                     "gain_kept": kept, "vs_random": diff, "random_range": [lo, hi], "p": p})
    print("\n* pre-registered threshold")
    return rows


if __name__ == "__main__":
    main()
