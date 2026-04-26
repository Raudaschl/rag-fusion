"""Headline-table evaluation with paired-bootstrap 95% CIs.

Re-runs the 6-method retrieval-only comparison at large samples and reports
each cell with a 95% CI on the lift over baseline. Outputs a markdown table
ready to drop into the top-level README.

Note: this is RETRIEVAL-ONLY (no cross-encoder rerank). For the full
production-relevant comparison (rerank + answer quality) see
experiments/arxiv-2603-02153-replication/.
"""

import argparse
import json
import os
import random

from tqdm import tqdm

import main
from eval.dataset import download_nfcorpus, load_nfcorpus, load_into_chromadb, sample_queries
from eval.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr
from eval.query_cache import cached_generate
from eval.retrieval import (
    bm25_retrieve, single_query_retrieve, hybrid_retrieve, rag_fusion_retrieve,
    rag_fusion_diverse_retrieve, hybrid_diverse_retrieve,
)


# Patch in-process so the legacy retrieval functions go through the disk cache.
# Cache key is the literal query text, which is a safe surrogate for qid here
# (rewrites are deterministic given (query, diverse)).
_original_generate = main.generate_queries_chatgpt
def _patched_generate(query, diverse=False):
    return cached_generate(qid=f"text:{query}", query=query, diverse=diverse)
main.generate_queries_chatgpt = _patched_generate


def percentile(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def per_query_value(retrieved, qrel_scores, metric, k):
    relevant = {d for d, s in qrel_scores.items() if s > 0}
    if metric == "precision":
        return precision_at_k(retrieved, relevant, k)
    if metric == "recall":
        return recall_at_k(retrieved, relevant, k)
    if metric == "ndcg":
        return ndcg_at_k(retrieved, qrel_scores, k)
    if metric == "mrr":
        return mrr(retrieved, relevant)
    raise ValueError(metric)


def paired_bootstrap_lift(method_vals, baseline_vals, b=10000, seed=42):
    rng = random.Random(seed)
    n = len(method_vals)
    diffs = []
    for _ in range(b):
        indices = [rng.randrange(n) for _ in range(n)]
        m = sum(method_vals[i] for i in indices) / n
        bs = sum(baseline_vals[i] for i in indices) / n
        diffs.append(m - bs)
    return percentile(diffs, 0.025), percentile(diffs, 0.975)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--out", type=str, default="./eval_with_ci.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()

    download_nfcorpus(data_dir=args.data_dir)
    corpus, queries, qrels = load_nfcorpus(data_dir=args.data_dir)
    db_path = os.path.join(args.data_dir, "chroma_eval_db")
    collection = load_into_chromadb(corpus, db_path=db_path)

    query_ids = sample_queries(queries, qrels, n=args.sample)
    max_k = max(args.k)

    methods = [
        ("BM25", bm25_retrieve),
        ("Baseline", single_query_retrieve),
        ("Hybrid", hybrid_retrieve),
        ("RAG-Fusion", rag_fusion_retrieve),
        ("+Diverse", rag_fusion_diverse_retrieve),
        ("Hybrid+Diverse", hybrid_diverse_retrieve),
    ]

    print(f"n_queries={len(query_ids)}; methods={[n for n, _ in methods]}")

    retrievals = {}
    for name, fn in methods:
        print(f"\n=== {name} ===")
        retrievals[name] = {}
        for qid in tqdm(query_ids):
            retrievals[name][qid] = fn(queries[qid], collection, k=max_k)

    metric_specs = [(m, k) for k in args.k for m in ("precision", "recall", "ndcg")]
    metric_specs.append(("mrr", None))

    per_query_vals = {}
    for name, _ in methods:
        for metric, k in metric_specs:
            key = f"{metric}@{k}" if k else "mrr"
            per_query_vals[(name, key)] = [
                per_query_value(retrievals[name][qid], qrels.get(qid, {}), metric, k or 0)
                for qid in query_ids
            ]

    rows = []
    for metric, k in metric_specs:
        key = f"{metric}@{k}" if k else "mrr"
        row = {"metric": metric, "k": k or "-", "key": key}
        for name, _ in methods:
            vals = per_query_vals[(name, key)]
            row[name] = sum(vals) / len(vals)
        baseline_vals = per_query_vals[("Baseline", key)]
        for name, _ in methods:
            if name == "Baseline":
                continue
            vals = per_query_vals[(name, key)]
            lo, hi = paired_bootstrap_lift(vals, baseline_vals, b=args.bootstrap)
            row[f"{name}_lift_mean"] = sum(v - b for v, b in zip(vals, baseline_vals)) / len(vals)
            row[f"{name}_lift_ci"] = (lo, hi)
        rows.append(row)

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows, "qids": query_ids}, f, indent=2,
                  default=lambda o: list(o) if isinstance(o, tuple) else o)
    print(f"\nWrote {args.out}")

    # Markdown table
    print("\n# Headline table (n={})\n".format(args.sample))
    headers = ["Metric", "k"] + [n for n, _ in methods] + ["Hybrid+Diverse lift over Baseline [95% CI]"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = [r["metric"].capitalize(), str(r["k"])]
        for name, _ in methods:
            cells.append(f"{r[name]:.3f}")
        lo, hi = r["Hybrid+Diverse_lift_ci"]
        m = r["Hybrid+Diverse_lift_mean"]
        sig = "**" if (lo > 0 or hi < 0) else ""
        cells.append(f"{sig}{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main_cli()
