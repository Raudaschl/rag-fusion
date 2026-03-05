import math


def precision_at_k(retrieved, relevant, k):
    """Precision at rank k."""
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return relevant_count / k


def recall_at_k(retrieved, relevant, k):
    """Recall at rank k."""
    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return relevant_count / len(relevant)


def mrr(retrieved, relevant):
    """Mean Reciprocal Rank. Returns 1/(rank of first relevant doc) or 0."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved, qrel_scores, k):
    """Normalized Discounted Cumulative Gain at k.

    qrel_scores is a dict {doc_id: relevance_score} with graded relevance (0/1/2).
    """
    retrieved_at_k = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_at_k):
        rel = qrel_scores.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # IDCG: sort all relevance scores descending, take top k
    ideal_rels = sorted(qrel_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all_metrics(all_results, qrels, k_values):
    """Compute averaged metrics across all queries.

    all_results: dict {query_id: [list of retrieved doc IDs]}
    qrels: dict {query_id: {doc_id: relevance_score}}
    k_values: list like [5, 10, 20]

    Returns dict with keys like "precision@5", "recall@10", "ndcg@5", "mrr".
    """
    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = 0.0
        metrics[f"recall@{k}"] = 0.0
        metrics[f"ndcg@{k}"] = 0.0
    metrics["mrr"] = 0.0

    query_ids = list(all_results.keys())
    n_queries = len(query_ids)
    if n_queries == 0:
        return metrics

    for qid in query_ids:
        retrieved = all_results[qid]
        qrel_scores = qrels.get(qid, {})
        relevant = set(doc_id for doc_id, score in qrel_scores.items() if score > 0)

        metrics["mrr"] += mrr(retrieved, relevant)

        for k in k_values:
            metrics[f"precision@{k}"] += precision_at_k(retrieved, relevant, k)
            metrics[f"recall@{k}"] += recall_at_k(retrieved, relevant, k)
            metrics[f"ndcg@{k}"] += ndcg_at_k(retrieved, qrel_scores, k)

    # Average across queries
    for key in metrics:
        metrics[key] /= n_queries

    return metrics
