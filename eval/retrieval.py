from main import vector_search, generate_queries_chatgpt, reciprocal_rank_fusion
from tqdm import tqdm


def single_query_retrieve(query, collection, k=10):
    results = vector_search(query, collection, n_results=k)
    return list(results.keys())


def rag_fusion_retrieve(query, collection, k=10):
    generated_queries = generate_queries_chatgpt(query)
    all_results = {}
    all_results[query] = vector_search(query, collection, n_results=k)
    for q in generated_queries:
        all_results[q] = vector_search(q, collection, n_results=k)
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
