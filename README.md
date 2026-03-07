# RAG-Fusion: The Next Frontier of Search Technology

## Overview

RAG-Fusion is a search methodology that aims to bridge the gap between traditional search paradigms and the multifaceted dimensions of human queries. Where Retrieval Augmented Generation (RAG) fuses vector search with generative models, RAG-Fusion goes a step further — employing multiple query generation and Reciprocal Rank Fusion to re-rank search results. The overarching goal is to move closer to unearthing that elusive 90% of transformative knowledge that often remains hidden behind top search results.

For the full story behind the approach, see the article: [Forget RAG, the Future is RAG-Fusion](https://adrianraudaschl.com/blog/forget-rag-the-future-is-rag-fusion/).

## How It Works

```
                    ┌─────────────────┐
                    │  Original Query  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   LLM generates  │
                    │  multiple queries │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │  Vector    │ │  Vector    │ │  Vector    │
        │  Search 1  │ │  Search 2  │ │  Search N  │
        └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Reciprocal Rank │
                    │     Fusion       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Re-ranked Docs   │
                    └──────────────────┘
```

1. **Query Generation** — Takes a user's query and uses OpenAI's GPT to generate multiple search query variations that capture different facets of the original intent.

2. **Vector Search** — Conducts vector-based searches using ChromaDB on each query, casting a wider net across the document space.

3. **Reciprocal Rank Fusion** — Combines the ranked results from all searches, boosting documents that appear consistently across multiple query perspectives.

4. **Output Generation** — Produces a final re-ranked list of documents, optionally synthesised into a natural language answer via LLM.

## Project Structure

```
├── main.py              # Core RAG-Fusion pipeline
├── evaluate.py          # Evaluation CLI entry point
├── test_main.py         # Unit tests
├── eval/
│   ├── dataset.py       # NFCorpus download & loading
│   ├── metrics.py       # IR metrics (Precision, Recall, NDCG, MRR)
│   └── retrieval.py     # Retrieval methods (BM25, vector, hybrid, RAG-Fusion variants)
└── .env.example         # Environment template
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install openai chromadb python-dotenv tqdm tabulate rank_bm25
   ```

2. Set up your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and replace `your-key-here` with your actual key.

3. Run the demo:
   ```bash
   python main.py
   ```

4. Run the tests (no API key needed):
   ```bash
   python -m pytest test_main.py -v
   ```

## Evaluation

To move beyond toy examples, the repo includes a quantitative evaluation harness that compares multiple retrieval strategies on a real dataset. It uses [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) (3,633 medical/nutrition documents, 323 test queries with graded relevance judgments) from the [BEIR benchmark](https://github.com/beir-cellar/beir).

### Results (50 queries, seed=42)

| Metric    | k  | BM25  | Baseline | Hybrid | RAG-Fusion | +Diverse | Hybrid+Diverse | vs Baseline |
|-----------|----|-------|----------|--------|------------|----------|----------------|-------------|
| Precision | 5  | 0.264 | 0.272    | 0.264  | 0.276      | 0.288    | **0.312**      | +14.7%      |
| Precision | 10 | 0.202 | 0.226    | 0.228  | 0.226      | 0.242    | **0.254**      | +12.4%      |
| Precision | 20 | 0.146 | 0.183    | 0.175  | 0.194      | 0.197    | **0.203**      | +10.9%      |
| Recall    | 5  | 0.135 | 0.130    | 0.126  | 0.118      | 0.145    | **0.169**      | +30.0%      |
| Recall    | 10 | 0.156 | 0.153    | 0.164  | 0.185      | 0.182    | **0.214**      | +39.9%      |
| Recall    | 20 | 0.172 | 0.205    | 0.192  | 0.231      | 0.225    | **0.249**      | +21.5%      |
| NDCG      | 5  | 0.337 | 0.329    | 0.341  | 0.325      | 0.359    | **0.402**      | +22.2%      |
| NDCG      | 10 | 0.304 | 0.309    | 0.326  | 0.312      | 0.341    | **0.381**      | +23.3%      |
| NDCG      | 20 | 0.276 | 0.302    | 0.304  | 0.311      | 0.330    | **0.364**      | +20.5%      |
| MRR       | -  | 0.463 | 0.461    | 0.500  | 0.443      | 0.481    | **0.578**      | +25.4%      |

Six methods are compared:

- **BM25** — classic keyword search using BM25Okapi. Competitive on NDCG@5 thanks to exact term matching on NFCorpus's medical vocabulary, but falls behind at higher k values where semantic understanding matters more.
- **Baseline** — single vector search with the original query using ChromaDB's default embedding model (`all-MiniLM-L6-v2`).
- **Hybrid** — BM25 + vector search fused via RRF (no LLM calls). A strong "free lunch" — runs as fast as baseline with no API costs, and notably improves MRR.
- **RAG-Fusion** — original + 4 LLM-generated queries, combined via Reciprocal Rank Fusion.
- **RAG-Fusion +Diverse** — RAG-Fusion with an improved prompt that explicitly asks for different angles, synonyms, and varied specificity. Improves recall and NDCG over standard RAG-Fusion.
- **Hybrid+Diverse** — the best of both: runs RAG-Fusion (diverse prompt) but searches each query with both BM25 and vector search, then fuses all results via RRF. Best overall performer with **+22% NDCG@5**, **+40% recall@10**, and **+25% MRR** over baseline.

Three key insights emerge. First, **hybrid search is a free lunch** — fusing BM25 and vector results via RRF costs nothing extra and improves ranking quality, especially MRR. Second, the **diverse prompt** outperforms standard RAG-Fusion by forcing the LLM to explore genuinely different angles rather than generating semantically close variations. Third, **the two techniques are fully complementary** — hybrid's keyword precision and diverse's semantic breadth combine cleanly through RRF, producing the strongest results across every metric.

### Running the evaluation

```bash
# Baseline only (no API key needed)
python evaluate.py --sample 10 --methods baseline

# Default comparison (requires OPENAI_API_KEY)
python evaluate.py --sample 50

# All methods
python evaluate.py --sample 50 --methods bm25 baseline hybrid rag-fusion rag-fusion-diverse hybrid-diverse

# Custom parameters
python evaluate.py --sample 100 --k 5 10 --data-dir ./datasets
```

The NFCorpus dataset (~3MB) is downloaded automatically on first run. ChromaDB embeddings are persisted locally so subsequent runs skip ingestion.