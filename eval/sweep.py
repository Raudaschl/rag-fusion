"""Sweep driver for arxiv 2603.02153 replication.

Two independent sweeps:
  (A) candidate_pool size with fixed N=4 rewrites — find where rerank's selection
      capacity stops absorbing fusion's added candidates.
  (B) number of LLM rewrites with fixed candidate_pool=50 — test whether N=1 (the
      paper's setting) reproduces the "fusion gains evaporate" effect even with a
      generous candidate pool.

Caches generated queries per (qid, diverse) pair so the N sweep is single-cost.
"""

import argparse
import json
import os
import time
from functools import lru_cache

from eval.dataset import download_nfcorpus, load_nfcorpus, load_into_chromadb, sample_queries
from eval.query_cache import cached_generate
from eval.retrieval import single_query_retrieve, with_rerank, run_evaluation
from main import vector_search, reciprocal_rank_fusion


def make_rag_fusion_n(n, diverse=True, qid_lookup=None):
    """Build a retrieve fn that uses original_query + first n cached LLM rewrites."""
    def retrieve(query, collection, k=10):
        qid = qid_lookup.get(query) if qid_lookup else query
        rewrites = cached_generate(qid, query, diverse=diverse)[:n]
        all_results = {query: vector_search(query, collection, n_results=k)}
        for q in rewrites:
            all_results[q] = vector_search(q, collection, n_results=k)
        fused = reciprocal_rank_fusion(all_results, verbose=False)
        return list(fused.keys())[:k]
    return retrieve


def sweep_pool(query_ids, queries, qrels, collection, pool_values, k_values,
               qid_lookup, n_rewrites=4):
    """Vary candidate_pool with fixed N rewrites."""
    rows = {}
    for pool in pool_values:
        baseline_fn = with_rerank(single_query_retrieve, candidate_pool=pool)
        fusion_fn = with_rerank(
            make_rag_fusion_n(n_rewrites, diverse=True, qid_lookup=qid_lookup),
            candidate_pool=pool,
        )
        print(f"\n[pool={pool}] baseline+rerank ...")
        rows[(pool, "baseline+rerank")] = run_evaluation(
            query_ids, queries, qrels, collection, baseline_fn, k_values
        )
        print(f"[pool={pool}] fusion(N={n_rewrites})+rerank ...")
        rows[(pool, f"fusion(N={n_rewrites})+rerank")] = run_evaluation(
            query_ids, queries, qrels, collection, fusion_fn, k_values
        )
    return rows


def sweep_n(query_ids, queries, qrels, collection, n_values, k_values,
            qid_lookup, candidate_pool=50):
    """Vary N rewrites with fixed candidate_pool."""
    rows = {}
    baseline_fn = with_rerank(single_query_retrieve, candidate_pool=candidate_pool)
    print(f"\n[N=0 baseline] +rerank ...")
    rows[(0, "baseline+rerank")] = run_evaluation(
        query_ids, queries, qrels, collection, baseline_fn, k_values
    )
    for n in n_values:
        fn = with_rerank(
            make_rag_fusion_n(n, diverse=True, qid_lookup=qid_lookup),
            candidate_pool=candidate_pool,
        )
        print(f"[N={n}] fusion+rerank ...")
        rows[(n, f"fusion(N={n})+rerank")] = run_evaluation(
            query_ids, queries, qrels, collection, fn, k_values
        )
    return rows


def fmt_table(rows, k_values, headline_metric="ndcg@10"):
    """Compact one-line-per-config table for the headline metric."""
    keys = sorted(rows.keys(), key=lambda x: (x[0], x[1]))
    lines = [f"{'config':<32} | {headline_metric:<10} | recall@10 | mrr"]
    lines.append("-" * len(lines[0]))
    for k in keys:
        m = rows[k]
        lines.append(
            f"{str(k):<32} | {m.get(headline_metric, 0):<10.3f} "
            f"| {m.get('recall@10', 0):<9.3f} | {m.get('mrr', 0):.3f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sweep pool size and N rewrites for fusion+rerank.")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--pool-values", type=int, nargs="+", default=[10, 20, 30, 50, 75])
    parser.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--n-fixed-for-pool", type=int, default=4,
                        help="N rewrites used during pool sweep (default 4)")
    parser.add_argument("--pool-fixed-for-n", type=int, default=50,
                        help="candidate_pool used during N sweep (default 50)")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--out", type=str, default="./eval_sweep_results.json")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    download_nfcorpus(data_dir=args.data_dir)
    corpus, queries, qrels = load_nfcorpus(data_dir=args.data_dir)
    db_path = os.path.join(args.data_dir, "chroma_eval_db")
    collection = load_into_chromadb(corpus, db_path=db_path)

    query_ids = sample_queries(queries, qrels, n=args.sample)
    qid_lookup = {queries[qid]: qid for qid in query_ids}
    print(f"Sampled {len(query_ids)} queries.")

    t0 = time.time()
    print("\n==== SWEEP A: candidate_pool ====")
    pool_rows = sweep_pool(query_ids, queries, qrels, collection,
                           args.pool_values, args.k, qid_lookup,
                           n_rewrites=args.n_fixed_for_pool)

    print("\n==== SWEEP B: N rewrites ====")
    n_rows = sweep_n(query_ids, queries, qrels, collection,
                     args.n_values, args.k, qid_lookup,
                     candidate_pool=args.pool_fixed_for_n)

    elapsed = time.time() - t0
    print(f"\nTotal sweep time: {elapsed:.1f}s")

    print("\n==== POOL SWEEP RESULTS ====")
    print(fmt_table(pool_rows, args.k))
    print("\n==== N SWEEP RESULTS ====")
    print(fmt_table(n_rows, args.k))

    out = {
        "pool_sweep": {f"{p}|{name}": v for (p, name), v in pool_rows.items()},
        "n_sweep": {f"{n}|{name}": v for (n, name), v in n_rows.items()},
        "config": vars(args),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
