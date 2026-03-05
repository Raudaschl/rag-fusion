from main import create_collection, vector_search, reciprocal_rank_fusion, generate_output


def test_create_collection():
    collection = create_collection()
    assert collection.count() == 10
    result = collection.get()
    assert result["ids"] == [f"doc{i+1}" for i in range(10)]


def test_vector_search():
    collection = create_collection()
    scores = vector_search("climate change effects", collection)

    assert len(scores) == 5
    score_values = list(scores.values())
    # Scores should be sorted descending
    assert score_values == sorted(score_values, reverse=True)
    # All scores should be between 0 and 1 (exclusive of 0)
    for s in score_values:
        assert 0 < s <= 1


def test_vector_search_relevance():
    collection = create_collection()
    scores = vector_search("economic impact of climate change", collection)
    top_docs = list(scores.keys())
    # doc1 is "Climate change and economic impact." — should rank high
    assert "doc1" in top_docs[:3]


def test_reciprocal_rank_fusion():
    search_results = {
        "query1": {"doc1": 0.9, "doc2": 0.8, "doc3": 0.7},
        "query2": {"doc2": 0.95, "doc1": 0.85, "doc4": 0.6},
    }
    fused = reciprocal_rank_fusion(search_results, k=60)

    # All docs from both queries should appear
    assert set(fused.keys()) == {"doc1", "doc2", "doc3", "doc4"}

    # Results should be sorted descending by fused score
    fused_values = list(fused.values())
    assert fused_values == sorted(fused_values, reverse=True)

    # doc1 and doc2 appear in both queries at top ranks, so they should score highest
    top_two = list(fused.keys())[:2]
    assert set(top_two) == {"doc1", "doc2"}

    # Verify RRF math: doc1 is rank 0 in query1, rank 1 in query2
    expected_doc1 = 1 / (0 + 60) + 1 / (1 + 60)
    assert abs(fused["doc1"] - expected_doc1) < 1e-9


def test_generate_output():
    reranked = {"doc2": 0.5, "doc1": 0.3}
    queries = ["q1", "q2"]
    output = generate_output(reranked, queries)
    assert "doc2" in output
    assert "doc1" in output
    assert "q1" in output
    assert "q2" in output


def test_generate_output_accepts_new_params():
    """Verify generate_output accepts the new parameters and still works with use_llm=False."""
    reranked = {"doc2": 0.5, "doc1": 0.3}
    queries = ["q1", "q2"]
    collection = create_collection()
    output = generate_output(reranked, queries, collection=collection, original_query="test query", use_llm=False)
    assert "doc2" in output
    assert "doc1" in output
