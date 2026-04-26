"""Cross-encoder reranker for evaluating retrieval methods under production-style constraints.

Models the post-retrieval stage that arxiv 2603.02153 argues absorbs fusion's recall gains:
each method retrieves a candidate pool, the cross-encoder rescores (query, doc) pairs,
results are truncated to top-k.
"""

_cross_encoder = None
_loaded_model_name = None


def _get_cross_encoder(model_name):
    global _cross_encoder, _loaded_model_name
    if _cross_encoder is None or _loaded_model_name != model_name:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
        _loaded_model_name = model_name
    return _cross_encoder


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
    model = _get_cross_encoder(model_name)
    texts = _fetch_doc_texts(doc_ids, collection)
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs)
    ranked = sorted(zip(doc_ids, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked[:top_k]]
