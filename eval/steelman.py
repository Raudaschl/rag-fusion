"""Steelman tests for arXiv 2603.02153 — three head-on probes of the paper's argument.

Test 1: Pipeline ordering. The paper's ordering is per-query retrieve → per-query
        rerank → fuse → truncate. Mine is per-query retrieve → fuse → rerank → truncate.
        Both share the same rerank compute budget (~50 cross-encoder pairs per query)
        so the only thing that varies is where the budget is spent.
Test 2: Truncation depth. Report NDCG at K=1, 3, 5, 10. The user-facing context window
        is K≤5; if fusion's lift only shows up at K=10/20 the paper's truncation argument
        lands.
Test 3: Difficulty stratification. Bucket queries into "recall-rich" (baseline+rerank
        finds ≥1 relevant in top-10) and "recall-scarce". The paper conceded fusion
        helps the latter; testing whether the former (the production majority) sees
        any benefit.
"""

import argparse
import json
import os
import time

from tqdm import tqdm

from eval.dataset import download_nfcorpus, load_nfcorpus, load_into_chromadb, sample_queries
from eval.query_cache import cached_generate
from eval.retrieval import single_query_retrieve, with_rerank, bm25_search
from eval.rerank import rerank
from eval.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr
from main import vector_search, reciprocal_rank_fusion


def make_paper_pipeline(n_rewrites=4, per_query_pool=10, qid_lookup=None,
                         rerank_model="BAAI/bge-reranker-base"):
    """Paper's ordering: for each (Q1, Q_rewrite_i), retrieve a small pool, rerank,
    keep the reranked list, then RRF-fuse the reranked lists, truncate to k.
    """
    def retrieve(query, collection, k=10):
        qid = qid_lookup.get(query) if qid_lookup else query
        rewrites = cached_generate(qid, query, diverse=True)[:n_rewrites]
        all_queries = [query] + rewrites
        per_query_reranked = {}
        for q in all_queries:
            search = vector_search(q, collection, n_results=per_query_pool)
            ids = list(search.keys())
            ranked = rerank(q, ids, collection, top_k=per_query_pool, model_name=rerank_model)
            per_query_reranked[q] = {d: per_query_pool - i for i, d in enumerate(ranked)}
        fused = reciprocal_rank_fusion(per_query_reranked, verbose=False)
        return list(fused.keys())[:k]
    return retrieve


def make_fuse_then_rerank(n_rewrites=4, candidate_pool=50, qid_lookup=None,
                           rerank_model="BAAI/bge-reranker-base"):
    """Our default: per-query retrieve → fuse → rerank fused pool → truncate."""
    def base(query, collection, k=10):
        qid = qid_lookup.get(query) if qid_lookup else query
        rewrites = cached_generate(qid, query, diverse=True)[:n_rewrites]
        all_results = {query: vector_search(query, collection, n_results=k)}
        for q in rewrites:
            all_results[q] = vector_search(q, collection, n_results=k)
        fused = reciprocal_rank_fusion(all_results, verbose=False)
        return list(fused.keys())[:k]
    return with_rerank(base, candidate_pool=candidate_pool, model_name=rerank_model)


def make_hybrid_baseline(candidate_pool=50, rerank_model="BAAI/bge-reranker-base"):
    """Hybrid baseline: BM25 + vector for the original query, fused via RRF, then reranked.
    No LLM rewrites — this is the 'free lunch' baseline that adds BM25 to the dense retriever.
    """
    def base(query, collection, k=10):
        all_results = {
            f"bm25:{query}": bm25_search(query, collection, n_results=k),
            f"vector:{query}": vector_search(query, collection, n_results=k),
        }
        fused = reciprocal_rank_fusion(all_results, verbose=False)
        return list(fused.keys())[:k]
    return with_rerank(base, candidate_pool=candidate_pool, model_name=rerank_model)


def make_hybrid_diverse_fuse_then_rerank(n_rewrites=4, candidate_pool=50, qid_lookup=None,
                                          rerank_model="BAAI/bge-reranker-base"):
    """The strongest fusion variant in the repo: for each query (Q1 + N rewrites), do BOTH
    BM25 and vector search; fuse all 2*(N+1) result lists via RRF; rerank the fused pool.
    """
    def base(query, collection, k=10):
        qid = qid_lookup.get(query) if qid_lookup else query
        rewrites = cached_generate(qid, query, diverse=True)[:n_rewrites]
        all_queries = [query] + rewrites
        all_results = {}
        for q in all_queries:
            all_results[f"bm25:{q}"] = bm25_search(q, collection, n_results=k)
            all_results[f"vector:{q}"] = vector_search(q, collection, n_results=k)
        fused = reciprocal_rank_fusion(all_results, verbose=False)
        return list(fused.keys())[:k]
    return with_rerank(base, candidate_pool=candidate_pool, model_name=rerank_model)


def make_hybrid_per_query_rerank_then_fuse(n_rewrites=4, per_query_pool=10, qid_lookup=None,
                                            rerank_model="BAAI/bge-reranker-base"):
    """Cross-encoder integrated into each sub-query's pipeline before fusion.

    For each Qi in (Q1 + N rewrites):
      1. candidates_i = RRF(BM25(Qi), vector(Qi))         [hybrid retrieval per query]
      2. reranked_i   = cross_encoder.rerank(Qi, candidates_i)[:per_query_pool]
    RRF across all reranked_i lists; truncate to k.

    This is the "fusion that uses a cross-encoder inside, not just as a post-hoc filter"
    variant — the hybrid extension of the paper's pipeline ordering.
    """
    def retrieve(query, collection, k=10):
        qid = qid_lookup.get(query) if qid_lookup else query
        rewrites = cached_generate(qid, query, diverse=True)[:n_rewrites]
        all_queries = [query] + rewrites
        per_query_reranked = {}
        for q in all_queries:
            local = {
                f"bm25:{q}": bm25_search(q, collection, n_results=per_query_pool),
                f"vector:{q}": vector_search(q, collection, n_results=per_query_pool),
            }
            local_fused = reciprocal_rank_fusion(local, verbose=False)
            local_ids = list(local_fused.keys())[:per_query_pool]
            ranked = rerank(q, local_ids, collection, top_k=per_query_pool, model_name=rerank_model)
            per_query_reranked[q] = {d: per_query_pool - i for i, d in enumerate(ranked)}
        fused = reciprocal_rank_fusion(per_query_reranked, verbose=False)
        return list(fused.keys())[:k]
    return retrieve


def per_query_metrics(retrieved_by_qid, qrels, k_values):
    out = {}
    for qid, retrieved in retrieved_by_qid.items():
        qrel_scores = qrels.get(qid, {})
        relevant = {d for d, s in qrel_scores.items() if s > 0}
        m = {"mrr": mrr(retrieved, relevant)}
        for k in k_values:
            m[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            m[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            m[f"ndcg@{k}"] = ndcg_at_k(retrieved, qrel_scores, k)
        out[qid] = m
    return out


def aggregate(per_q, qids, k_values):
    if not qids:
        return None
    keys = ["mrr"] + [f"{m}@{k}" for m in ("precision", "recall", "ndcg") for k in k_values]
    agg = {kk: 0.0 for kk in keys}
    for qid in qids:
        for kk in keys:
            agg[kk] += per_q[qid][kk]
    return {kk: agg[kk] / len(qids) for kk in keys}


def collect(query_ids, queries, method_fn, collection, max_k):
    out = {}
    for qid in tqdm(query_ids):
        out[qid] = method_fn(queries[qid], collection, k=max_k)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--n-rewrites", type=int, default=4)
    parser.add_argument("--per-query-pool", type=int, default=10,
                        help="Pool retrieved per sub-query in the paper's pipeline (default 10 = paper's K).")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--rerank-model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--out", type=str, default="./eval_steelman.json")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()

    download_nfcorpus(data_dir=args.data_dir)
    corpus, queries, qrels = load_nfcorpus(data_dir=args.data_dir)
    collection = load_into_chromadb(corpus, db_path=os.path.join(args.data_dir, "chroma_eval_db"))

    query_ids = sample_queries(queries, qrels, n=args.sample)
    qid_lookup = {queries[qid]: qid for qid in query_ids}
    max_k = max(args.k)

    methods = {
        "baseline+rerank":
            with_rerank(single_query_retrieve, candidate_pool=args.candidate_pool,
                        model_name=args.rerank_model),
        "hybrid+rerank":
            make_hybrid_baseline(candidate_pool=args.candidate_pool,
                                 rerank_model=args.rerank_model),
        "fuse_then_rerank (mine)":
            make_fuse_then_rerank(n_rewrites=args.n_rewrites,
                                  candidate_pool=args.candidate_pool, qid_lookup=qid_lookup,
                                  rerank_model=args.rerank_model),
        "hybrid_diverse+rerank":
            make_hybrid_diverse_fuse_then_rerank(n_rewrites=args.n_rewrites,
                                                  candidate_pool=args.candidate_pool,
                                                  qid_lookup=qid_lookup,
                                                  rerank_model=args.rerank_model),
        "rerank_per_query_then_fuse (paper)":
            make_paper_pipeline(n_rewrites=args.n_rewrites,
                                per_query_pool=args.per_query_pool, qid_lookup=qid_lookup,
                                rerank_model=args.rerank_model),
        "hybrid_per_query_rerank_then_fuse":
            make_hybrid_per_query_rerank_then_fuse(n_rewrites=args.n_rewrites,
                                                    per_query_pool=args.per_query_pool,
                                                    qid_lookup=qid_lookup,
                                                    rerank_model=args.rerank_model),
    }
    print(f"reranker: {args.rerank_model}")

    print(f"Sampled {len(query_ids)} queries.")
    retrieved = {}
    per_q = {}
    for name, fn in methods.items():
        print(f"\n=== {name} ===")
        t0 = time.time()
        retrieved[name] = collect(query_ids, queries, fn, collection, max_k)
        per_q[name] = per_query_metrics(retrieved[name], qrels, args.k)
        print(f"  done in {time.time()-t0:.1f}s")

    # ---- TEST 1: pipeline ordering ----
    print("\n==== TEST 1: PIPELINE ORDERING (all queries) ====")
    print(f"{'method':<38} | NDCG@10 | Recall@10 | MRR")
    print("-" * 80)
    for name in methods:
        agg = aggregate(per_q[name], query_ids, [10])
        print(f"{name:<38} | {agg['ndcg@10']:.3f}   | {agg['recall@10']:.3f}     | {agg['mrr']:.3f}")

    # ---- TEST 2: K sweep ----
    print("\n==== TEST 2: TRUNCATION DEPTH (NDCG@K, all queries) ====")
    header = f"{'method':<38} | " + " | ".join(f"k={k:<3}" for k in args.k)
    print(header)
    print("-" * len(header))
    for name in methods:
        agg = aggregate(per_q[name], query_ids, args.k)
        line = f"{name:<38} | " + " | ".join(f"{agg[f'ndcg@{k}']:.3f}" for k in args.k)
        print(line)

    # ---- TEST 3: difficulty stratification (bucket by baseline+rerank top-10) ----
    baseline_ret = retrieved["baseline+rerank"]
    rich, scarce = [], []
    for qid in query_ids:
        relevant = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        if any(d in relevant for d in baseline_ret[qid][:10]):
            rich.append(qid)
        else:
            scarce.append(qid)

    print(f"\n==== TEST 3: DIFFICULTY STRATIFICATION ====")
    print(f"recall-rich n={len(rich)}   recall-scarce n={len(scarce)}")
    print(f"\n{'method':<38} | rich NDCG@10 | scarce NDCG@10 | rich R@10 | scarce R@10")
    print("-" * 100)
    for name in methods:
        ra = aggregate(per_q[name], rich, [10]) or {}
        sa = aggregate(per_q[name], scarce, [10]) or {}
        print(f"{name:<38} | "
              f"{ra.get('ndcg@10', 0):.3f}        | "
              f"{sa.get('ndcg@10', 0):.3f}          | "
              f"{ra.get('recall@10', 0):.3f}     | "
              f"{sa.get('recall@10', 0):.3f}")

    out = {
        "config": vars(args),
        "buckets": {"rich_qids": rich, "scarce_qids": scarce,
                    "n_rich": len(rich), "n_scarce": len(scarce)},
        "metrics_all": {n: aggregate(per_q[n], query_ids, args.k) for n in methods},
        "metrics_rich": {n: aggregate(per_q[n], rich, args.k) for n in methods},
        "metrics_scarce": {n: aggregate(per_q[n], scarce, args.k) for n in methods},
        "per_query": per_q,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
