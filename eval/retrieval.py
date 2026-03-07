from main import vector_search, generate_queries_chatgpt, reciprocal_rank_fusion
from tqdm import tqdm

_bm25_index = None
_bm25_doc_ids = None


def get_bm25_index(collection):
    """Lazy-build a BM25 index from the ChromaDB collection."""
    global _bm25_index, _bm25_doc_ids
    if _bm25_index is None:
        from rank_bm25 import BM25Okapi
        all_docs = collection.get()
        _bm25_doc_ids = all_docs["ids"]
        tokenized = [doc.lower().split() for doc in all_docs["documents"]]
        _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_doc_ids


def bm25_search(query, collection, n_results=10):
    """BM25 keyword search returning {doc_id: score} dict (same format as vector_search)."""
    bm25, doc_ids = get_bm25_index(collection)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return {doc_ids[i]: scores[i] for i in top_indices}


def bm25_retrieve(query, collection, k=10):
    """Classic BM25 keyword search."""
    results = bm25_search(query, collection, n_results=k)
    return list(results.keys())


def single_query_retrieve(query, collection, k=10):
    results = vector_search(query, collection, n_results=k)
    return list(results.keys())


def hybrid_retrieve(query, collection, k=10):
    """Hybrid BM25 + vector search fused via RRF (no LLM calls)."""
    all_results = {
        f"bm25:{query}": bm25_search(query, collection, n_results=k),
        f"vector:{query}": vector_search(query, collection, n_results=k),
    }
    fused = reciprocal_rank_fusion(all_results, verbose=False)
    return list(fused.keys())[:k]


def rag_fusion_retrieve(query, collection, k=10):
    generated_queries = generate_queries_chatgpt(query)
    all_results = {}
    all_results[query] = vector_search(query, collection, n_results=k)
    for q in generated_queries:
        all_results[q] = vector_search(q, collection, n_results=k)
    fused = reciprocal_rank_fusion(all_results, verbose=False)
    return list(fused.keys())[:k]


def rag_fusion_diverse_retrieve(query, collection, k=10):
    """RAG-Fusion with diverse query generation prompt."""
    generated_queries = generate_queries_chatgpt(query, diverse=True)
    all_results = {}
    all_results[query] = vector_search(query, collection, n_results=k)
    for q in generated_queries:
        all_results[q] = vector_search(q, collection, n_results=k)
    fused = reciprocal_rank_fusion(all_results, verbose=False)
    return list(fused.keys())[:k]


def rag_fusion_weighted_retrieve(query, collection, k=10):
    """RAG-Fusion with diverse queries + 3x weight on original query."""
    generated_queries = generate_queries_chatgpt(query, diverse=True)
    all_results = {}
    all_results[query] = vector_search(query, collection, n_results=k)
    for q in generated_queries:
        all_results[q] = vector_search(q, collection, n_results=k)
    query_weights = {query: 3.0}
    fused = reciprocal_rank_fusion(all_results, verbose=False, query_weights=query_weights)
    return list(fused.keys())[:k]


def hybrid_diverse_retrieve(query, collection, k=10):
    """Hybrid (BM25+vector) for each query + diverse LLM-generated queries."""
    generated_queries = generate_queries_chatgpt(query, diverse=True)
    all_queries = [query] + generated_queries
    all_results = {}
    for q in all_queries:
        all_results[f"bm25:{q}"] = bm25_search(q, collection, n_results=k)
        all_results[f"vector:{q}"] = vector_search(q, collection, n_results=k)
    fused = reciprocal_rank_fusion(all_results, verbose=False)
    return list(fused.keys())[:k]


def run_evaluation(query_ids, queries, qrels, collection, method_fn, k_values):
    max_k = max(k_values)
    results = {}
    for qid in tqdm(query_ids):
        query_text = queries[qid]
        retrieved = method_fn(query_text, collection, k=max_k)
        results[qid] = retrieved
    from eval.metrics import compute_all_metrics
    metrics = compute_all_metrics(results, qrels, k_values)
    return metrics
