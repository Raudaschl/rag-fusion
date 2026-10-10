"""Where fusion's gain goes: relevant documents in the 50-candidate pool, and in each reranker's top 10.

Pool counts come from re-running retrieval without a reranker (single-query vector top 50 vs the
hybrid_diverse fused top 50, same cached rewrites). Top-10 counts are precision@10 × 10 from the
steelman result files, so they match the published NDCG runs exactly.

    python -m eval.pool_recall --out experiments/jev-in-the-pipeline/pool_recall.json
"""

import argparse
import json
import os
import statistics

from eval.dataset import load_into_chromadb, load_nfcorpus, sample_queries
from eval.query_cache import cached_generate
from eval.retrieval import bm25_search
from main import reciprocal_rank_fusion, vector_search

RESULTS = "experiments/arxiv-2603-02153-replication/results"
RUNS = {
    "bge-base": "steelman_base_n200_v2.json",
    "MiniLM": "steelman_flashrank_n200_v2.json",
    "bge-large": "steelman_large_n200_v2.json",
    "Jev run 1": "steelman_jev_n200_run1.json",
    "Jev run 2": "steelman_jev_n200_run2.json",
    "Luna score": "../../openai-decisions-api/results/steelman_decisions_score_n200_run1.json",
    "Luna predicate": "../../openai-decisions-api/results/steelman_decisions_predicate_n200_run1.json",
    "MSD1 yes/no": "../../microsoft-decision-1/results/steelman_msd1_yesno_n200_run1.json",
    "MSD1 rubric": "../../microsoft-decision-1/results/steelman_msd1_rubric_n200_run1.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./datasets")
    ap.add_argument("--out", default="./pool_recall.json")
    args = ap.parse_args()

    corpus, queries, qrels = load_nfcorpus(data_dir=args.data_dir)
    collection = load_into_chromadb(corpus, db_path=os.path.join(args.data_dir, "chroma_eval_db"))
    qids = sample_queries(queries, qrels, n=200)
    relevant = {q: {d for d, s in qrels[q].items() if s > 0} for q in qids}

    single, fused = [], []
    for q in qids:
        text = queries[q]
        single.append(len(relevant[q] & set(list(vector_search(text, collection, n_results=50))[:50])))
        lists = {}
        for s in [text] + cached_generate(q, text, diverse=True)[:4]:
            lists[f"bm25:{s}"] = bm25_search(s, collection, n_results=50)
            lists[f"vector:{s}"] = vector_search(s, collection, n_results=50)
        fused.append(len(relevant[q] & set(list(reciprocal_rank_fusion(lists, verbose=False))[:50])))
    pool = {"single_query": round(statistics.mean(single), 3), "fusion": round(statistics.mean(fused), 3)}
    out = {"n_queries": len(qids), "relevant_in_pool_of_50": pool, "relevant_in_top10": {}}

    for name, fname in RUNS.items():
        pq = json.load(open(os.path.join(RESULTS, fname)))["per_query"]
        alone = statistics.mean(10 * pq["baseline+rerank"][q]["precision@10"] for q in qids)
        with_f = statistics.mean(10 * pq["hybrid_diverse+rerank"][q]["precision@10"] for q in qids)
        out["relevant_in_top10"][name] = {
            "alone": round(alone, 3), "with_fusion": round(with_f, 3), "gain": round(with_f - alone, 3),
            "share_of_pool_gain": round((with_f - alone) / (pool["fusion"] - pool["single_query"]), 3),
            "pool_to_top10_rate_alone": round(alone / pool["single_query"], 3),
            "pool_to_top10_rate_fusion": round(with_f / pool["fusion"], 3),
        }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
