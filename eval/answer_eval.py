"""End-to-end answer-quality eval at scale via LLM-as-judge.

For each query, generates an answer from each method's top-K retrievals using
a synthesizer LLM, then sends the (query, gold-doc-text, anonymised-answers)
to a separate judge LLM that ranks the answers. Aggregates win/tie/loss rates
per method.

The point of this eval: NDCG@10 measures retrieval ranking; what matters in
production is whether the user got a correct answer. Aggregate retrieval metrics
average kohlrabi-style binary recoveries against ties and dilute them. Direct
answer scoring captures the recovery effect.
"""

import argparse
import json
import os
import random
import time
from collections import Counter

from tqdm import tqdm

from eval.dataset import download_nfcorpus, load_nfcorpus, load_into_chromadb
from main import get_client


SYNTHESIZER_SYSTEM = (
    "You are a careful biomedical research assistant. Answer the user's question "
    "using only the provided context. Cite the context numbers you used like [1], [2]. "
    "Be concise. If the context doesn't contain enough information to answer, say so explicitly."
)


JUDGE_SYSTEM = (
    "You are an impartial expert evaluator of biomedical-question answers. "
    "You will see a user question, the text of the gold relevant document(s), and three "
    "candidate answers labelled A/B/C. Score each answer on a 0-3 scale for whether it "
    "correctly answers the user's question (0 = wrong or absent, 1 = partial/peripheral, "
    "2 = mostly correct, 3 = correct and well-grounded). Output ONLY a JSON object of the "
    "form {\"A\": 2, \"B\": 0, \"C\": 3, \"reason\": \"<one short sentence>\"}. "
    "Do not include any other text."
)


def synthesize(query, doc_texts, model="gpt-5.1-chat-latest"):
    if not doc_texts:
        return "[no context provided]"
    ctx = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(doc_texts))
    resp = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{ctx}\n\nAnswer:"},
        ],
    )
    return resp.choices[0].message.content.strip()


def judge(query, gold_text, answers, model="gpt-5.1-chat-latest"):
    """Score three labelled answers; returns dict {label: int 0-3}."""
    labelled = [(label, ans) for label, ans in answers.items()]
    body = (
        f"QUESTION: {query}\n\n"
        f"GOLD RELEVANT DOCUMENT TEXT:\n{gold_text}\n\n"
        + "\n\n".join(f"ANSWER {label}:\n{ans}" for label, ans in labelled)
    )
    resp = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": body},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # robust to ```json fences
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("steelman_json", type=str,
                        help="Path to steelman JSON with per-query retrievals")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Limit to this many queries (for cheaper smoke runs)")
    parser.add_argument("--ctx-chars", type=int, default=1500)
    parser.add_argument("--gold-chars", type=int, default=2500)
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--out", type=str, default="./answer_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Subset of methods to evaluate (default: 3-method legacy set)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()

    download_nfcorpus(data_dir=args.data_dir)
    corpus, queries, qrels = load_nfcorpus(data_dir=args.data_dir)
    collection = load_into_chromadb(corpus, db_path=os.path.join(args.data_dir, "chroma_eval_db"))

    # Re-run retrievals from the steelman config — we need top-K=5 (the steelman
    # JSON saves only metrics, not retrieved IDs). Easiest: re-execute the methods
    # using the same query set via the same pipeline.
    from eval.retrieval import single_query_retrieve, with_rerank
    from eval.steelman import (make_paper_pipeline, make_fuse_then_rerank,
                                make_hybrid_baseline, make_hybrid_diverse_fuse_then_rerank,
                                make_hybrid_per_query_rerank_then_fuse)

    sm = json.load(open(args.steelman_json))
    rerank_model = sm["config"]["rerank_model"]
    candidate_pool = sm["config"]["candidate_pool"]
    n_rewrites = sm["config"]["n_rewrites"]
    per_query_pool = sm["config"]["per_query_pool"]

    qids = list(sm["per_query"][next(iter(sm["per_query"]))].keys())
    if args.n_queries:
        rng = random.Random(args.seed)
        qids = rng.sample(qids, args.n_queries)
    qid_lookup = {queries[qid]: qid for qid in qids}

    all_methods = {
        "baseline+rerank":
            with_rerank(single_query_retrieve, candidate_pool=candidate_pool, model_name=rerank_model),
        "hybrid+rerank":
            make_hybrid_baseline(candidate_pool=candidate_pool, rerank_model=rerank_model),
        "fuse_then_rerank":
            make_fuse_then_rerank(n_rewrites=n_rewrites, candidate_pool=candidate_pool,
                                  qid_lookup=qid_lookup, rerank_model=rerank_model),
        "hybrid_diverse+rerank":
            make_hybrid_diverse_fuse_then_rerank(n_rewrites=n_rewrites, candidate_pool=candidate_pool,
                                                  qid_lookup=qid_lookup, rerank_model=rerank_model),
        "rerank_per_query_then_fuse":
            make_paper_pipeline(n_rewrites=n_rewrites, per_query_pool=per_query_pool,
                                qid_lookup=qid_lookup, rerank_model=rerank_model),
        "hybrid_per_query_rerank_then_fuse":
            make_hybrid_per_query_rerank_then_fuse(n_rewrites=n_rewrites, per_query_pool=per_query_pool,
                                                    qid_lookup=qid_lookup, rerank_model=rerank_model),
    }
    if args.methods:
        methods = {m: all_methods[m] for m in args.methods if m in all_methods}
    else:
        methods = {m: all_methods[m] for m in
                   ["baseline+rerank", "fuse_then_rerank", "rerank_per_query_then_fuse"]}
    method_labels = {m: chr(ord("A") + i) for i, m in enumerate(methods)}

    print(f"reranker: {rerank_model}; n_queries={len(qids)}; top_k={args.top_k}")
    print("Retrieving + generating + judging ...")

    results = []
    score_totals = {m: 0 for m in methods}
    score_counts = {m: 0 for m in methods}
    bucket_results = {"rich": [], "scarce": []}
    rich_set = set(sm["buckets"]["rich_qids"])

    for qid in tqdm(qids):
        query_text = queries[qid]
        bucket = "rich" if qid in rich_set else "scarce"

        gold_ids = [d for d, s in qrels.get(qid, {}).items() if s > 0]
        if not gold_ids:
            continue
        gold_fetch = collection.get(ids=gold_ids[:3])
        gold_text = "\n\n".join(t[:args.gold_chars] for t in gold_fetch["documents"])

        # retrieve + synthesise per method
        method_answers = {}
        retrieved_ids = {}
        for name, fn in methods.items():
            ids = fn(query_text, collection, k=args.top_k)
            retrieved_ids[name] = ids
            doc_fetch = collection.get(ids=ids)
            by_id = dict(zip(doc_fetch["ids"], doc_fetch["documents"]))
            doc_texts = [by_id.get(d, "")[:args.ctx_chars] for d in ids]
            try:
                ans = synthesize(query_text, doc_texts)
            except Exception as e:
                ans = f"[error: {e}]"
            method_answers[name] = ans

        # judge
        labelled_answers = {method_labels[m]: method_answers[m] for m in methods}
        try:
            judge_scores = judge(query_text, gold_text, labelled_answers)
        except Exception as e:
            judge_scores = {"error": str(e)}

        row = {
            "qid": qid,
            "bucket": bucket,
            "query": query_text,
            "scores": judge_scores,
            "answers": method_answers,
            "retrieved": retrieved_ids,
        }
        results.append(row)
        bucket_results[bucket].append(row)

        for m, label in method_labels.items():
            if label in judge_scores and isinstance(judge_scores[label], int):
                score_totals[m] += judge_scores[label]
                score_counts[m] += 1

    # Aggregate
    print("\n=== Mean judge score (0-3) per method ===")
    print(f"{'method':<40} | {'all':<14} | {'rich':<14} | {'scarce':<14}")
    for m, label in method_labels.items():
        all_scores = [r["scores"].get(label) for r in results
                      if isinstance(r["scores"].get(label), int)]
        rich_scores = [r["scores"].get(label) for r in bucket_results["rich"]
                       if isinstance(r["scores"].get(label), int)]
        scarce_scores = [r["scores"].get(label) for r in bucket_results["scarce"]
                         if isinstance(r["scores"].get(label), int)]
        def fmt(xs):
            if not xs:
                return "n/a"
            return f"{sum(xs)/len(xs):.2f} (n={len(xs)})"
        print(f"{m:<40} | {fmt(all_scores):<14} | {fmt(rich_scores):<14} | {fmt(scarce_scores):<14}")

    # Pairwise win rates: fusion vs baseline
    print("\n=== Win/tie/loss vs baseline+rerank (judge score comparison) ===")
    for m, label in method_labels.items():
        if m == "baseline+rerank":
            continue
        for bucket_name, bucket_rows in [("ALL", results),
                                         ("RICH", bucket_results["rich"]),
                                         ("SCARCE", bucket_results["scarce"])]:
            wins = ties = losses = 0
            for r in bucket_rows:
                a = r["scores"].get("A")
                me = r["scores"].get(label)
                if not (isinstance(a, int) and isinstance(me, int)):
                    continue
                if me > a:
                    wins += 1
                elif me == a:
                    ties += 1
                else:
                    losses += 1
            total = wins + ties + losses
            if total == 0:
                continue
            print(f"  {m:<38} {bucket_name:<7} W/T/L = {wins}/{ties}/{losses} "
                  f"({100*wins/total:.0f}% wins, {100*losses/total:.0f}% losses)")

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results,
                   "score_totals": score_totals, "score_counts": score_counts}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
