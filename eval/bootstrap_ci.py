"""Bootstrap 95% confidence intervals on the steelman per-query metrics.

Loads a steelman JSON, resamples queries with replacement, recomputes per-method
means, and reports both per-method CIs and paired (fusion - baseline) lift CIs.
The paired bootstrap shares query indices across methods so within-query
correlation is preserved — that's what makes the lift CI tight.
"""

import argparse
import json
import random
import statistics


def percentile(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def bootstrap_paired(per_q, qids, methods, metric_key, b=10000, seed=42):
    """Return dict {method: (mean, lo, hi)} for each method, using shared resampled
    indices across methods. Also returns paired (method - baseline) CIs."""
    rng = random.Random(seed)
    n = len(qids)
    # Pre-extract per-query metric values for each method
    values = {m: [per_q[m][q][metric_key] for q in qids] for m in methods}

    resampled_means = {m: [] for m in methods}
    for _ in range(b):
        indices = [rng.randrange(n) for _ in range(n)]
        for m in methods:
            v = values[m]
            resampled_means[m].append(sum(v[i] for i in indices) / n)

    out = {}
    for m in methods:
        rs = resampled_means[m]
        out[m] = {
            "mean": statistics.mean(values[m]),
            "ci_lo": percentile(rs, 0.025),
            "ci_hi": percentile(rs, 0.975),
        }

    # Paired lift: each method - baseline, using SAME resampled indices
    if "baseline+rerank" in methods:
        for m in methods:
            if m == "baseline+rerank":
                continue
            diffs = [resampled_means[m][i] - resampled_means["baseline+rerank"][i]
                     for i in range(b)]
            mean_lift = out[m]["mean"] - out["baseline+rerank"]["mean"]
            out[m]["lift_mean"] = mean_lift
            out[m]["lift_ci_lo"] = percentile(diffs, 0.025)
            out[m]["lift_ci_hi"] = percentile(diffs, 0.975)
    return out


def fmt_ci(d, key="mean", lo="ci_lo", hi="ci_hi"):
    return f"{d[key]:+.3f} [{d[lo]:+.3f}, {d[hi]:+.3f}]"


def fmt_metric(d):
    return f"{d['mean']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("steelman_json", type=str)
    parser.add_argument("--metrics", nargs="+",
                        default=["ndcg@10", "recall@10", "mrr"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    data = json.load(open(args.steelman_json))
    per_q = data["per_query"]
    methods = list(per_q.keys())
    all_qids = list(next(iter(per_q.values())).keys())
    rich = data["buckets"]["rich_qids"]
    scarce = data["buckets"]["scarce_qids"]

    print(f"\n=== {args.steelman_json} ===")
    print(f"reranker: {data['config'].get('rerank_model', '?')}")
    print(f"n_total={len(all_qids)}  n_rich={len(rich)}  n_scarce={len(scarce)}")

    for metric in args.metrics:
        print(f"\n----- {metric} (95% paired-bootstrap CI, B={args.bootstrap}) -----")
        for bucket_name, qids in [("ALL", all_qids), ("RICH", rich), ("SCARCE", scarce)]:
            if not qids:
                continue
            cis = bootstrap_paired(per_q, qids, methods, metric, b=args.bootstrap)
            print(f"\n  {bucket_name} (n={len(qids)})")
            print(f"    {'method':<38} | {'mean [95% CI]':<28} | lift over baseline [95% CI]")
            print("    " + "-" * 100)
            for m in methods:
                d = cis[m]
                lift = (f"{fmt_ci(d, 'lift_mean', 'lift_ci_lo', 'lift_ci_hi')}"
                        if "lift_mean" in d else "—")
                print(f"    {m:<38} | {fmt_metric(d):<28} | {lift}")


if __name__ == "__main__":
    main()
