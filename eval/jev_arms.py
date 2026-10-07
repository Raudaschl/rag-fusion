"""Fusion variants that put Jev inside the pipeline (Jev as reranker is `--rerank-model jev`).

  - hybrid_diverse_orig3x+rerank:      static control — original query's lists get 3x RRF weight
  - hybrid_diverse_jevweighted+rerank: original keeps weight 1.0 (a code rule, never Jev's call);
                                       each rewrite's lists are weighted by Jev's P(keeps intent)
  - jev_router+rerank:                 Jev classifies the query; fuse only when it is broad or
                                       multi-faceted, otherwise run hybrid+rerank (no LLM rewrites)
  - jev_conf_router+rerank:           run hybrid → Jev first; rewrite and fuse (hybrid_diverse)
                                       only when Jev's best document scores below a threshold.
                                       Cost-aware routing from arXiv 2609.05637, gated on
                                       retrieval confidence instead of query type
  - with_jev_gate:                     after retrieval, if Jev gives every top-k document
                                       P(relevant) below a threshold, return nothing so the
                                       synthesizer abstains ("evidence is thin")

Every Jev decision is recorded in DECISIONS so runs can report weights, routes and gate rates.
"""

from eval.jev import intent_weights, query_kind, relevance
from eval.query_cache import cached_generate
from eval.rerank import _fetch_doc_texts
from eval.retrieval import bm25_search, with_rerank
from main import reciprocal_rank_fusion, vector_search

DECISIONS = {"intent": {}, "route": {}, "gate": {}, "conf_route": {}}

FUSE_KINDS = ("broad", "multi_faceted")


def _hybrid_lists(queries, collection, k):
    results = {}
    for q in queries:
        results[f"bm25:{q}"] = bm25_search(q, collection, n_results=k)
        results[f"vector:{q}"] = vector_search(q, collection, n_results=k)
    return results


def _weights_for(per_query_weight):
    return {f"{src}:{q}": w for q, w in per_query_weight.items() for src in ("bm25", "vector")}


def make_hybrid_diverse_weighted(weighting, n_rewrites=4, candidate_pool=50, qid_lookup=None,
                                 rerank_model="BAAI/bge-reranker-base"):
    """hybrid_diverse with per-query RRF weights. weighting: "orig3x" or "jev"."""
    def base(query, collection, k=10):
        qid = qid_lookup.get(query) if qid_lookup else query
        rewrites = cached_generate(qid, query, diverse=True)[:n_rewrites]
        if weighting == "orig3x":
            per_query = {query: 3.0}
        elif weighting == "jev":
            probs = intent_weights(query, rewrites)
            DECISIONS["intent"][qid] = dict(zip(rewrites, probs))
            per_query = {query: 1.0, **dict(zip(rewrites, probs))}
        else:
            raise ValueError(weighting)
        fused = reciprocal_rank_fusion(_hybrid_lists([query] + rewrites, collection, k),
                                       verbose=False, query_weights=_weights_for(per_query))
        return list(fused.keys())[:k]
    return with_rerank(base, candidate_pool=candidate_pool, model_name=rerank_model)


def make_jev_router(n_rewrites=4, candidate_pool=50, qid_lookup=None,
                    rerank_model="BAAI/bge-reranker-base"):
    """Fuse (hybrid_diverse) only on queries Jev calls broad or multi-faceted."""
    def base(query, collection, k=10):
        qid = qid_lookup.get(query) if qid_lookup else query
        kind = query_kind(query)
        fuse = kind["choice"] in FUSE_KINDS
        DECISIONS["route"][qid] = {**kind, "fused": fuse}
        queries = [query]
        if fuse:
            queries += cached_generate(qid, query, diverse=True)[:n_rewrites]
        fused = reciprocal_rank_fusion(_hybrid_lists(queries, collection, k), verbose=False)
        return list(fused.keys())[:k]
    return with_rerank(base, candidate_pool=candidate_pool, model_name=rerank_model)


def make_jev_confidence_router(fuse_fn, threshold=0.5, candidate_pool=50, qid_lookup=None):
    """hybrid → Jev rerank; if Jev's best document has P(relevant) < threshold, return
    fuse_fn's result instead (the hybrid_diverse+rerank arm). The hybrid pool and its scores
    are the same request hybrid+rerank makes, so with Jev as reranker they come from cache and
    the confident path costs nothing extra. Every query's signal is recorded, so thresholds
    can be swept offline from the two arms' per-query results."""
    def retrieve(query, collection, k=10):
        qid = qid_lookup.get(query) if qid_lookup else query
        pool = max(candidate_pool, k)
        fused = reciprocal_rank_fusion(_hybrid_lists([query], collection, pool), verbose=False)
        candidates = list(fused.keys())[:pool]
        scores = relevance(query, _fetch_doc_texts(candidates, collection))
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top = ranked[0][1] if ranked else 0.0
        fuse = top < threshold
        DECISIONS["conf_route"][qid] = {"max_relevance": top, "fused": fuse}
        if fuse:
            return fuse_fn(query, collection, k=k)
        return [d for d, _ in ranked[:k]]
    return retrieve


def with_jev_gate(method_fn, threshold=0.5, qid_lookup=None):
    """Return no documents when Jev judges none of the top-k relevant (max P < threshold)."""
    def gated(query, collection, k=10):
        ids = method_fn(query, collection, k=k)
        if not ids:
            return ids
        res = collection.get(ids=ids)
        by_id = dict(zip(res["ids"], res["documents"]))
        scores = relevance(query, [by_id.get(d, "") for d in ids])
        top = max(scores)
        qid = qid_lookup.get(query) if qid_lookup else query
        DECISIONS["gate"][qid] = {"max_relevance": top, "fired": top < threshold}
        return [] if top < threshold else ids
    return gated
