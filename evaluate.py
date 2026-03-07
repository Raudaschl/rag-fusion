"""CLI entry point for RAG-Fusion evaluation against NFCorpus."""

import argparse
import os

from tabulate import tabulate

from eval.dataset import download_nfcorpus, load_nfcorpus, load_into_chromadb, sample_queries
from eval.retrieval import (
    bm25_retrieve, single_query_retrieve, hybrid_retrieve, rag_fusion_retrieve,
    rag_fusion_diverse_retrieve, rag_fusion_weighted_retrieve,
    hybrid_diverse_retrieve, run_evaluation,
)


def positive_int(value):
    """Argparse type for integers greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("k values must be greater than 0")
    return parsed


def build_comparison_table(all_metrics, k_values):
    """Build a comparison table of metrics across methods.

    Args:
        all_metrics: dict of {method_name: metrics_dict}
        k_values: list of k values used in evaluation
    """
    method_names = list(all_metrics.keys())
    rows = []
    metric_names = ["Precision", "Recall", "NDCG"]

    for name in metric_names:
        key_prefix = name.lower()
        for k in k_values:
            key = f"{key_prefix}@{k}"
            row = [name, k]
            for method in method_names:
                m = all_metrics[method]
                val = m.get(key) if m else None
                row.append(f"{val:.3f}" if val is not None else "N/A")
            rows.append(row)

    # MRR row
    row = ["MRR", "-"]
    for method in method_names:
        m = all_metrics[method]
        val = m.get("mrr") if m else None
        row.append(f"{val:.3f}" if val is not None else "N/A")
    rows.append(row)

    headers = ["Metric", "k"] + method_names
    return tabulate(rows, headers=headers, tablefmt="grid")


def _format_delta(baseline_val, fusion_val):
    if baseline_val is None or fusion_val is None:
        return "N/A"
    if baseline_val == 0:
        return "+inf%" if fusion_val > 0 else "0.0%"
    pct = (fusion_val - baseline_val) / baseline_val * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def show_example_queries(query_ids, queries, qrels, collection, methods, method_registry):
    """Display top-5 docs from each method for 3 example queries, marking relevant ones."""
    example_ids = query_ids[:3]

    for qid in example_ids:
        query_text = queries[qid]
        relevant = set(
            doc_id for doc_id, score in qrels.get(qid, {}).items() if score > 0
        )
        print(f"\nQuery [{qid}]: {query_text}")
        print("-" * 60)

        for method_key in methods:
            display_name, method_fn, needs_api_key = method_registry[method_key]
            if needs_api_key and not os.getenv("OPENAI_API_KEY"):
                continue
            docs = method_fn(query_text, collection, k=5)
            print(f"  {display_name} top-5:")
            for i, doc_id in enumerate(docs, 1):
                marker = " *" if doc_id in relevant else ""
                print(f"    {i}. {doc_id}{marker}")

        print(f"  (* = relevant document)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG-Fusion against baseline retrieval on NFCorpus.")
    parser.add_argument("--sample", type=int, default=50, help="Number of queries to sample (default: 50)")
    parser.add_argument("--k", type=positive_int, nargs="+", default=[5, 10, 20],
                        help="k values for evaluation; each must be > 0 (default: 5 10 20)")
    parser.add_argument("--data-dir", type=str, default="./datasets", help="Data directory (default: ./datasets)")
    parser.add_argument("--methods", type=str, nargs="+", default=["baseline", "rag-fusion"],
                        choices=["bm25", "baseline", "hybrid", "rag-fusion", "rag-fusion-diverse",
                                 "rag-fusion-weighted", "hybrid-diverse"],
                        help="Methods to evaluate (default: baseline rag-fusion)")
    args = parser.parse_args()

    # Download and load dataset
    download_nfcorpus(data_dir=args.data_dir)
    corpus, queries, qrels = load_nfcorpus(data_dir=args.data_dir)

    # Keep the persistent index scoped to the selected dataset directory.
    db_path = os.path.join(args.data_dir, "chroma_eval_db")
    collection = load_into_chromadb(corpus, db_path=db_path)

    # Sample queries
    query_ids = sample_queries(queries, qrels, n=args.sample)
    print(f"\nSampled {len(query_ids)} queries for evaluation.\n")

    method_registry = {
        "bm25": ("BM25", bm25_retrieve, False),
        "baseline": ("Baseline", single_query_retrieve, False),
        "hybrid": ("Hybrid", hybrid_retrieve, False),
        "rag-fusion": ("RAG-Fusion", rag_fusion_retrieve, True),
        "rag-fusion-diverse": ("+Diverse", rag_fusion_diverse_retrieve, True),
        "rag-fusion-weighted": ("+Diverse+Weighted", rag_fusion_weighted_retrieve, True),
        "hybrid-diverse": ("Hybrid+Diverse", hybrid_diverse_retrieve, True),
    }

    all_metrics = {}
    for method_key in args.methods:
        display_name, method_fn, needs_api_key = method_registry[method_key]
        if needs_api_key and not os.getenv("OPENAI_API_KEY"):
            print(f"WARNING: OPENAI_API_KEY not set. Skipping {display_name}.\n")
            all_metrics[display_name] = None
            continue
        print(f"Running {display_name} retrieval ...")
        all_metrics[display_name] = run_evaluation(query_ids, queries, qrels, collection, method_fn, args.k)
        print(f"{display_name} evaluation complete.\n")

    # Display comparison table
    print(build_comparison_table(all_metrics, args.k))

    # Show example queries if any method produced results
    if any(v is not None for v in all_metrics.values()):
        print("\n--- Example Queries (top-5 results) ---")
        show_example_queries(query_ids, queries, qrels, collection, args.methods, method_registry)


if __name__ == "__main__":
    main()
