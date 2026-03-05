"""CLI entry point for RAG-Fusion evaluation against NFCorpus."""

import argparse
import os

from tabulate import tabulate

from eval.dataset import download_nfcorpus, load_nfcorpus, load_into_chromadb, sample_queries
from eval.retrieval import single_query_retrieve, rag_fusion_retrieve, run_evaluation


def positive_int(value):
    """Argparse type for integers greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("k values must be greater than 0")
    return parsed


def build_comparison_table(baseline_metrics, fusion_metrics, k_values):
    """Build a comparison table of metrics across methods."""
    rows = []
    metric_names = ["Precision", "Recall", "NDCG"]

    for name in metric_names:
        key_prefix = name.lower()
        for k in k_values:
            key = f"{key_prefix}@{k}"
            b_val = baseline_metrics.get(key) if baseline_metrics else None
            f_val = fusion_metrics.get(key) if fusion_metrics else None
            delta = _format_delta(b_val, f_val)
            rows.append([
                name,
                k,
                f"{b_val:.3f}" if b_val is not None else "N/A",
                f"{f_val:.3f}" if f_val is not None else "N/A",
                delta,
            ])

    # MRR row
    b_mrr = baseline_metrics.get("mrr") if baseline_metrics else None
    f_mrr = fusion_metrics.get("mrr") if fusion_metrics else None
    delta = _format_delta(b_mrr, f_mrr)
    rows.append([
        "MRR",
        "-",
        f"{b_mrr:.3f}" if b_mrr is not None else "N/A",
        f"{f_mrr:.3f}" if f_mrr is not None else "N/A",
        delta,
    ])

    headers = ["Metric", "k", "Baseline", "RAG-Fusion", "Delta"]
    return tabulate(rows, headers=headers, tablefmt="grid")


def _format_delta(baseline_val, fusion_val):
    if baseline_val is None or fusion_val is None:
        return "N/A"
    if baseline_val == 0:
        return "+inf%" if fusion_val > 0 else "0.0%"
    pct = (fusion_val - baseline_val) / baseline_val * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def show_example_queries(query_ids, queries, qrels, collection, baseline_metrics, fusion_metrics):
    """Display top-5 docs from each method for 3 example queries, marking relevant ones."""
    example_ids = query_ids[:3]

    for qid in example_ids:
        query_text = queries[qid]
        relevant = set(
            doc_id for doc_id, score in qrels.get(qid, {}).items() if score > 0
        )
        print(f"\nQuery [{qid}]: {query_text}")
        print("-" * 60)

        if baseline_metrics is not None:
            baseline_docs = single_query_retrieve(query_text, collection, k=5)
            print("  Baseline top-5:")
            for i, doc_id in enumerate(baseline_docs, 1):
                marker = " *" if doc_id in relevant else ""
                print(f"    {i}. {doc_id}{marker}")

        if fusion_metrics is not None:
            fusion_docs = rag_fusion_retrieve(query_text, collection, k=5)
            print("  RAG-Fusion top-5:")
            for i, doc_id in enumerate(fusion_docs, 1):
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
                        choices=["baseline", "rag-fusion"], help="Methods to evaluate (default: baseline rag-fusion)")
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

    baseline_metrics = None
    fusion_metrics = None

    # Run baseline
    if "baseline" in args.methods:
        print("Running baseline (single-query) retrieval ...")
        baseline_metrics = run_evaluation(query_ids, queries, qrels, collection, single_query_retrieve, args.k)
        print("Baseline evaluation complete.\n")

    # Run RAG-Fusion
    if "rag-fusion" in args.methods:
        if not os.getenv("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not set. Skipping RAG-Fusion evaluation.\n")
        else:
            print("Running RAG-Fusion retrieval ...")
            fusion_metrics = run_evaluation(query_ids, queries, qrels, collection, rag_fusion_retrieve, args.k)
            print("RAG-Fusion evaluation complete.\n")

    # Display comparison table
    print(build_comparison_table(baseline_metrics, fusion_metrics, args.k))

    # Show example queries
    if baseline_metrics is not None or fusion_metrics is not None:
        print("\n--- Example Queries (top-5 results) ---")
        show_example_queries(query_ids, queries, qrels, collection, baseline_metrics, fusion_metrics)


if __name__ == "__main__":
    main()
