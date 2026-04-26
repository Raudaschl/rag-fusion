"""Qualitative end-to-end eval for the arXiv 2603.02153 replication.

For a small selection of queries (mix of recall-rich and recall-scarce, biased
toward those where retrievals diverge across methods), runs each method, fetches
document texts, generates an answer using each method's retrievals as context,
and writes a verbose log for direct human/agent inspection.

Also computes cost-adjusted comparisons (NDCG/Recall lift per extra LLM call)
from per-query metrics computed inline.
"""

import argparse
import json
import os

from eval.dataset import download_nfcorpus, load_nfcorpus, load_into_chromadb, sample_queries
from eval.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr
from eval.retrieval import single_query_retrieve, with_rerank
from eval.steelman import make_paper_pipeline, make_fuse_then_rerank
from main import get_client


SYSTEM_PROMPT = (
    "You are a careful biomedical research assistant. Answer the user's question "
    "using only the provided context. Cite the context numbers you used like [1], [2]. "
    "Be concise. If the context doesn't contain enough information to answer, say so explicitly."
)


def generate_answer(query, doc_texts, model="gpt-5.1-chat-latest"):
    ctx = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(doc_texts))
    prompt = f"Question: {query}\n\nContext:\n{ctx}\n\nAnswer:"
    resp = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def per_query_metrics_for_ids(retrieved, qrel_scores, k_values):
    relevant = {d for d, s in qrel_scores.items() if s > 0}
    m = {"mrr": mrr(retrieved, relevant)}
    for k in k_values:
        m[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        m[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        m[f"ndcg@{k}"] = ndcg_at_k(retrieved, qrel_scores, k)
    return m


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--n-judge", type=int, default=8,
                        help="Number of queries to evaluate end-to-end")
    parser.add_argument("--top-k", type=int, default=5,
                        help="K used for retrieval truncation and as answer context")
    parser.add_argument("--metric-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--n-rewrites", type=int, default=4)
    parser.add_argument("--per-query-pool", type=int, default=10)
    parser.add_argument("--rerank-model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--out", type=str, default="./qualitative_eval.txt")
    parser.add_argument("--snippet-chars", type=int, default=350)
    parser.add_argument("--ctx-chars", type=int, default=1500)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()

    download_nfcorpus(data_dir=args.data_dir)
    corpus, queries, qrels = load_nfcorpus(data_dir=args.data_dir)
    collection = load_into_chromadb(corpus, db_path=os.path.join(args.data_dir, "chroma_eval_db"))

    query_ids = sample_queries(queries, qrels, n=args.sample)
    qid_lookup = {queries[qid]: qid for qid in query_ids}
    metric_max_k = max(args.metric_k)

    methods = {
        "baseline+rerank":
            with_rerank(single_query_retrieve, candidate_pool=args.candidate_pool,
                        model_name=args.rerank_model),
        "fuse_then_rerank":
            make_fuse_then_rerank(n_rewrites=args.n_rewrites,
                                  candidate_pool=args.candidate_pool, qid_lookup=qid_lookup,
                                  rerank_model=args.rerank_model),
        "rerank_per_query_then_fuse":
            make_paper_pipeline(n_rewrites=args.n_rewrites,
                                per_query_pool=args.per_query_pool, qid_lookup=qid_lookup,
                                rerank_model=args.rerank_model),
    }
    print(f"reranker: {args.rerank_model}")

    print(f"Retrieving (k={metric_max_k}) for {len(query_ids)} queries × {len(methods)} methods ...")
    retrieved = {name: {} for name in methods}
    for name, fn in methods.items():
        for qid in query_ids:
            retrieved[name][qid] = fn(queries[qid], collection, k=metric_max_k)

    # Per-query metrics for cost analysis
    per_q = {name: {qid: per_query_metrics_for_ids(retrieved[name][qid],
                                                    qrels.get(qid, {}), args.metric_k)
                    for qid in query_ids}
             for name in methods}

    # Bucket by baseline top-10
    rich, scarce = [], []
    for qid in query_ids:
        relevant = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        if any(d in relevant for d in retrieved["baseline+rerank"][qid][:10]):
            rich.append(qid)
        else:
            scarce.append(qid)

    # Cost-adjusted summary (one extra LLM call per query for fusion methods)
    EXTRA_CALLS = {"baseline+rerank": 0, "fuse_then_rerank": 1, "rerank_per_query_then_fuse": 1}

    def lift_per_call(metric_key, name, qids):
        if not qids:
            return 0.0, 0.0
        base = avg([per_q["baseline+rerank"][q][metric_key] for q in qids])
        new = avg([per_q[name][q][metric_key] for q in qids])
        delta = new - base
        cost = EXTRA_CALLS[name] - EXTRA_CALLS["baseline+rerank"]
        return delta, (delta / cost if cost else 0.0)

    cost_lines = []
    cost_lines.append("\n" + "=" * 100)
    cost_lines.append("COST-ADJUSTED ANALYSIS")
    cost_lines.append("=" * 100)
    cost_lines.append("Each fusion method costs 1 extra LLM call/query (for query rewrites).")
    cost_lines.append("Below: ΔNDCG@10 / Δ(LLM calls). Higher = more lift per dollar.")
    cost_lines.append("")
    cost_lines.append(f"{'method':<32} | {'bucket':<8} | n  | ΔNDCG@10 | per-call | ΔRecall@10 | per-call")
    cost_lines.append("-" * 100)
    for name in methods:
        if name == "baseline+rerank":
            continue
        for bucket_name, qids in [("all", query_ids), ("rich", rich), ("scarce", scarce)]:
            d_ndcg, pc_ndcg = lift_per_call("ndcg@10", name, qids)
            d_rec, pc_rec = lift_per_call("recall@10", name, qids)
            cost_lines.append(f"{name:<32} | {bucket_name:<8} | {len(qids):<3}| "
                              f"{d_ndcg:+.3f}   | {pc_ndcg:+.3f}   | "
                              f"{d_rec:+.3f}     | {pc_rec:+.3f}")

    # Pick queries to judge: prefer top-3 divergence, mix rich/scarce
    def divergence(qid):
        sets = [tuple(retrieved[m][qid][:3]) for m in methods]
        return len(set(sets))

    rich_sorted = sorted(rich, key=lambda q: (-divergence(q), q))
    scarce_sorted = sorted(scarce, key=lambda q: (-divergence(q), q))
    half = args.n_judge // 2
    judge_qids = rich_sorted[:half] + scarce_sorted[:args.n_judge - half]

    # Fetch all needed doc texts in one batch
    needed_ids = set()
    for qid in judge_qids:
        for name in methods:
            needed_ids.update(retrieved[name][qid][:args.top_k])
        needed_ids.update(d for d, s in qrels.get(qid, {}).items() if s > 0)
    fetched = collection.get(ids=list(needed_ids))
    text_by_id = dict(zip(fetched["ids"], fetched["documents"]))

    lines = []
    lines.append("=" * 100)
    lines.append(f"QUALITATIVE END-TO-END EVAL")
    lines.append(f"  total queries scored: {len(query_ids)} (n_rich={len(rich)}, n_scarce={len(scarce)})")
    lines.append(f"  judging: {len(judge_qids)} queries "
                 f"({sum(1 for q in judge_qids if q in rich)} rich + "
                 f"{sum(1 for q in judge_qids if q in scarce)} scarce)")
    lines.append(f"  retrieval top-K = {args.top_k}; answer context = top-{args.top_k} docs each")
    lines.append("=" * 100)
    lines.extend(cost_lines)

    for qid in judge_qids:
        bucket = "RICH" if qid in rich else "SCARCE"
        query = queries[qid]
        relevant = {d: s for d, s in qrels.get(qid, {}).items() if s > 0}

        lines.append("\n" + "=" * 100)
        lines.append(f"[{bucket}] qid={qid}")
        lines.append(f"QUERY: {query}")
        lines.append(f"\nGOLD RELEVANT DOCS ({len(relevant)} total, showing up to 5):")
        for did, score in list(relevant.items())[:5]:
            t = text_by_id.get(did, "")[:args.snippet_chars].replace("\n", " ")
            lines.append(f"  - {did} (rel={score}): {t}...")

        for name in methods:
            ids = retrieved[name][qid][:args.top_k]
            lines.append(f"\n--- {name} top-{args.top_k} ---")
            ctx_texts = []
            for i, did in enumerate(ids):
                marker = " [REL]" if did in relevant else ""
                t = text_by_id.get(did, "")
                snippet = t[:args.snippet_chars].replace("\n", " ")
                lines.append(f"  [{i+1}] {did}{marker}: {snippet}...")
                ctx_texts.append(t[:args.ctx_chars])

            print(f"  generating answer: {name} for {qid} ...")
            try:
                ans = generate_answer(query, ctx_texts)
            except Exception as e:
                ans = f"[error: {e}]"
            lines.append(f"\nGENERATED ANSWER ({name}):")
            for ln in ans.split("\n"):
                lines.append(f"  {ln}")

    text = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"\nWrote {args.out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
