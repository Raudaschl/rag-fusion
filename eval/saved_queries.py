"""Compute the 'saved queries' metric: fraction of queries where baseline+rerank
finds zero relevant docs in top-K but fusion finds ≥1.

This is the kohlrabi-class statistic — direct measure of binary answer recovery.
Aggregate metrics like NDCG@10 dilute these wins against ties; this metric exposes them.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("steelman_json", type=str)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    data = json.load(open(args.steelman_json))
    per_q = data["per_query"]
    methods = list(per_q.keys())
    all_qids = list(next(iter(per_q.values())).keys())
    rich_qids = set(data["buckets"]["rich_qids"])
    scarce_qids = set(data["buckets"]["scarce_qids"])

    print(f"\n=== {args.steelman_json} ===")
    print(f"reranker: {data['config'].get('rerank_model', '?')}")
    print(f"n_total={len(all_qids)}  n_rich={len(rich_qids)}  n_scarce={len(scarce_qids)}")

    baseline = "baseline+rerank"
    metric_key = f"recall@{args.k}"

    print(f"\n----- saved queries ({metric_key} > 0) -----")
    print(f"{'method':<38} | {'saved (of all 200)':<22} | {'saved (of {} scarce)'.format(len(scarce_qids)):<28}")
    print("-" * 100)

    for m in methods:
        if m == baseline:
            continue

        # All-bucket: queries where baseline=0 but this method >0
        saved_all = sum(
            1 for qid in all_qids
            if per_q[baseline][qid][metric_key] == 0 and per_q[m][qid][metric_key] > 0
        )
        # Scarce-bucket only (definitionally where baseline finds 0)
        saved_scarce = sum(
            1 for qid in scarce_qids
            if per_q[m][qid][metric_key] > 0
        )

        # Also count "lost queries" — where baseline >0 but this method =0
        lost_all = sum(
            1 for qid in all_qids
            if per_q[baseline][qid][metric_key] > 0 and per_q[m][qid][metric_key] == 0
        )

        print(f"{m:<38} | "
              f"{saved_all}/{len(all_qids)} ({100*saved_all/len(all_qids):.1f}%)         | "
              f"{saved_scarce}/{len(scarce_qids)} ({100*saved_scarce/len(scarce_qids):.1f}%)  "
              f"  [lost: {lost_all}]")


if __name__ == "__main__":
    main()
