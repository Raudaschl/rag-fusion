"""Latency and cost per query for each pipeline piece, measured one at a time on an idle machine.

Pieces: vector and BM25 retrieval, the 10-list hybrid_diverse retrieval (cached rewrites),
reranking a 50-document pool with each local cross-encoder and with Jev, the LLM rewrite
call, and Jev's query-type router call. Hosted calls record their token usage, which is
priced from the published per-token rates below.

    python -m eval.latency_bench --n 30 --out experiments/jev-in-the-pipeline/latency_bench.json
"""

import argparse
import json
import os
import random
import statistics
import time

from eval.dataset import load_into_chromadb, load_nfcorpus, sample_queries
from eval.query_cache import cached_generate
from eval.retrieval import bm25_search
from main import LLM_MODEL, REASONING_EFFORT, get_client, reciprocal_rank_fusion, vector_search

# USD per token, as published on 2026-09-27: OpenRouter's listing for openai/gpt-6-luna, and
# TypeSafe's own "$42 per billion input tokens" for Jev (typesafe.ai lists no output price)
PRICE = {
    "gpt-6-luna": {"in": 0.10e-6, "out": 0.50e-6},
    "jev": {"in": 0.042e-6, "out": 0.0},
}
# The cached rewrites came from gpt-5.1-chat-latest, which OpenAI has since retired; time the
# repo's current model instead.
REWRITE_MODEL = LLM_MODEL


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000


def summary(ms):
    s = sorted(ms)
    return {"p50_ms": round(statistics.median(s), 1), "p90_ms": round(s[int(0.9 * (len(s) - 1))], 1), "n": len(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--data-dir", default="./datasets")
    ap.add_argument("--out", default="./latency_bench.json")
    args = ap.parse_args()

    corpus, queries, qrels = load_nfcorpus(data_dir=args.data_dir)
    collection = load_into_chromadb(corpus, db_path=os.path.join(args.data_dir, "chroma_eval_db"))
    qids = random.Random(7).sample(sample_queries(queries, qrels, n=200), args.n)
    import torch
    res = {"n_queries": args.n, "prices_usd_per_token": PRICE, "rewrite_model": f"{REWRITE_MODEL} ({REASONING_EFFORT})",
           "machine": "Apple M2 Max laptop", "torch_mps_available": torch.backends.mps.is_available()}

    # --- retrieval ---
    vec, bm, hyb = [], [], []
    bm25_search("warm up", collection, n_results=50)
    vector_search("warm up", collection, n_results=50)
    pools = {}
    for q in qids:
        text = queries[q]
        _, t = timed(lambda: vector_search(text, collection, n_results=50)); vec.append(t)
        _, t = timed(lambda: bm25_search(text, collection, n_results=50)); bm.append(t)
        rewrites = cached_generate(q, text, diverse=True)[:4]

        def hybrid_diverse():
            lists = {}
            for s in [text] + rewrites:
                lists[f"bm25:{s}"] = bm25_search(s, collection, n_results=50)
                lists[f"vector:{s}"] = vector_search(s, collection, n_results=50)
            return list(reciprocal_rank_fusion(lists, verbose=False))[:50]
        pools[q], t = timed(hybrid_diverse); hyb.append(t)
    res["retrieval"] = {"vector_top50": summary(vec), "bm25_top50": summary(bm), "hybrid_diverse_10_lists": summary(hyb)}

    # --- rerank a 50-doc pool ---
    from eval.rerank import _fetch_doc_texts, _get_backend
    texts = {q: _fetch_doc_texts(pools[q], collection) for q in qids}
    res["rerank_50"] = {}
    for name in ["flashrank", "BAAI/bge-reranker-base", "BAAI/bge-reranker-large"]:
        score = _get_backend(name)
        device = "onnx-cpu" if name == "flashrank" else str(getattr(score, "__closure__", None) and next((c.cell_contents.model.device for c in score.__closure__ if hasattr(c.cell_contents, "model")), "?"))
        score("warm up", texts[qids[0]][:5])
        ms = [timed(lambda: score(queries[q], texts[q]))[1] for q in qids]
        res["rerank_50"][name] = {**summary(ms), "device": device}
        print(name, res["rerank_50"][name])

    # Jev: fresh calls (no cache), 50 docs = two concurrent calls of 30 + 20, as in eval/jev.py
    from concurrent.futures import ThreadPoolExecutor
    from eval.jev import RELEVANCE_CRITERIA, RELEVANCE_QUESTION, _get_client
    from typesafe_sdk import Noul
    client = _get_client()

    def jev_call(query, docs):
        ids = [f"D{j:02d}" for j in range(len(docs))]
        r = client.system_one(state={"query": query, "documents": dict(zip(ids, docs))},
                              questions={i: Noul(instructions=RELEVANCE_QUESTION.format(id=i), criteria=RELEVANCE_CRITERIA) for i in ids},
                              model="jev-latest")
        return r.usage.input_tokens

    jev_ms, jev_tok = [], []
    for q in qids:
        docs = texts[q]
        with ThreadPoolExecutor(2) as ex:
            toks, t = timed(lambda: list(ex.map(lambda part: jev_call(queries[q], part), [docs[:30], docs[30:]])))
        jev_ms.append(t); jev_tok.append(sum(toks))
    res["rerank_50"]["jev"] = {**summary(jev_ms), "input_tokens_mean": round(statistics.mean(jev_tok)),
                               "usd_per_1000_queries": round(statistics.mean(jev_tok) * PRICE["jev"]["in"] * 1000, 3)}
    print("jev", res["rerank_50"]["jev"])

    # --- LLM rewrite call (same prompt as main.generate_queries_chatgpt, diverse) ---
    rw_ms, rw_cost = [], []
    for q in qids[:15]:
        msgs = [
            {"role": "system", "content": "You are a search expert. Generate diverse search queries that explore different aspects of the user's question. Each query should target a different angle: use synonyms, vary specificity (broader/narrower), and consider related sub-topics. Avoid generating queries that are just minor rewordings of each other."},
            {"role": "user", "content": f"Generate 4 diverse search queries for: {queries[q]}"},
            {"role": "user", "content": "OUTPUT (4 queries):"},
        ]
        r, t = timed(lambda: get_client().chat.completions.create(model=REWRITE_MODEL, reasoning_effort=REASONING_EFFORT, messages=msgs))
        rw_ms.append(t)
        rw_cost.append(r.usage.prompt_tokens * PRICE["gpt-6-luna"]["in"] + r.usage.completion_tokens * PRICE["gpt-6-luna"]["out"])
    res["rewrite_llm"] = {**summary(rw_ms), "usd_per_1000_queries": round(statistics.mean(rw_cost) * 1000, 3)}
    print("rewrite", res["rewrite_llm"])

    # --- Jev query-type router call ---
    from typesafe_sdk import Choice
    from eval.jev import KIND_CRITERIA, KIND_QUESTION
    rt_ms, rt_tok = [], []
    for q in qids[:15]:
        r, t = timed(lambda: client.system_one(state={"query": queries[q]},
                                               questions={"kind": Choice(instructions=KIND_QUESTION, criteria=KIND_CRITERIA)},
                                               model="jev-latest"))
        rt_ms.append(t); rt_tok.append(r.usage.input_tokens)
    res["jev_router"] = {**summary(rt_ms), "usd_per_1000_queries": round(statistics.mean(rt_tok) * PRICE["jev"]["in"] * 1000, 4)}
    print("router", res["jev_router"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
