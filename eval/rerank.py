"""Cross-encoder reranker for evaluating retrieval methods under production-style constraints.

Models the post-retrieval stage that arxiv 2603.02153 argues absorbs fusion's recall gains:
each method retrieves a candidate pool, the cross-encoder rescores (query, doc) pairs,
results are truncated to top-k.

Three reranker backends:
  - sentence-transformers CrossEncoder (default; bge-reranker-{base,large}, etc.)
  - FlashRank (the cross-encoder used in arxiv 2603.02153) — invoked when model_name
    starts with "flashrank:" (e.g. "flashrank:ms-marco-MiniLM-L-12-v2") or is the
    literal string "flashrank" (which uses FlashRank's default model).
  - Jev (TypeSafe's decision model, hosted) — invoked when model_name is "jev". Scores
    are calibrated P(relevant) from eval/jev.py. "jev-score" asks the four-level rubric
    instead of yes/no and scores each document by its expected level.
  - Voyage (hosted) — invoked as "voyage:<model>", e.g. "voyage:rerank-3" or
    "voyage:rerank-3-lite". Scores are Voyage's relevance_score from eval/voyage.py.
"""

_backend = None
_loaded_model_name = None


def _get_backend(model_name):
    """Return a callable scorer (query, list[text]) -> list[float], cached on model_name."""
    global _backend, _loaded_model_name
    if _backend is not None and _loaded_model_name == model_name:
        return _backend

    if model_name == "jev":
        from eval.jev import relevance

        def score(query, texts):
            return relevance(query, texts)
    elif model_name == "jev-score":
        from eval.jev import relevance_score

        def score(query, texts):
            return relevance_score(query, texts)
    elif model_name.startswith("voyage:"):
        from eval.voyage import relevance as voyage_relevance
        voyage_model = model_name.split(":", 1)[1]

        def score(query, texts):
            return voyage_relevance(query, texts, model=voyage_model)
    elif model_name == "flashrank" or model_name.startswith("flashrank:"):
        from flashrank import Ranker, RerankRequest
        flash_model = (model_name.split(":", 1)[1]
                       if ":" in model_name else "ms-marco-MiniLM-L-12-v2")
        ranker = Ranker(model_name=flash_model)

        def score(query, texts):
            passages = [{"id": str(i), "text": t} for i, t in enumerate(texts)]
            req = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(req)
            # Map back into original-input order so the caller can zip with doc_ids
            score_by_idx = {int(r["id"]): float(r["score"]) for r in results}
            return [score_by_idx.get(i, 0.0) for i in range(len(texts))]
    else:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model_name)

        def score(query, texts):
            return [float(s) for s in ce.predict([(query, t) for t in texts])]

    _backend = score
    _loaded_model_name = model_name
    return _backend


def _fetch_doc_texts(doc_ids, collection):
    if not doc_ids:
        return []
    res = collection.get(ids=list(doc_ids))
    by_id = dict(zip(res["ids"], res["documents"]))
    return [by_id.get(d, "") for d in doc_ids]


def rerank(query, doc_ids, collection, top_k=10, model_name="BAAI/bge-reranker-base"):
    """Score (query, doc) pairs with a cross-encoder; return doc_ids reordered, truncated to top_k."""
    if not doc_ids:
        return []
    score_fn = _get_backend(model_name)
    texts = _fetch_doc_texts(doc_ids, collection)
    scores = score_fn(query, texts)
    ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]
