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
│   └── retrieval.py     # Baseline & RAG-Fusion retrieval wrappers
└── .env.example         # Environment template
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install openai chromadb python-dotenv tqdm tabulate
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

To move beyond toy examples, the repo includes a quantitative evaluation harness that compares single-query baseline retrieval against RAG-Fusion on a real dataset. It uses [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) (3,633 medical/nutrition documents, 323 test queries with graded relevance judgments) from the [BEIR benchmark](https://github.com/beir-cellar/beir).

### Results (50 queries, seed=42)

| Metric    | k  | Baseline | RAG-Fusion | +Diverse | +Diverse+Weighted |
|-----------|----|----------|------------|----------|-------------------|
| Precision | 5  | 0.272    | 0.276      | **0.276**| 0.272             |
| Precision | 10 | 0.226    | 0.226      | **0.244**| 0.242             |
| Precision | 20 | 0.182    | 0.194      | **0.207**| 0.191             |
| Recall    | 5  | 0.130    | 0.118      | **0.142**| 0.126             |
| Recall    | 10 | 0.153    | 0.185      | **0.176**| 0.184             |
| Recall    | 20 | 0.205    | 0.231      | **0.221**| 0.210             |
| NDCG      | 5  | 0.329    | 0.325      | **0.344**| 0.330             |
| NDCG      | 10 | 0.309    | 0.312      | **0.333**| 0.327             |
| NDCG      | 20 | 0.301    | 0.311      | **0.328**| 0.312             |
| MRR       | -  | 0.460    | 0.443      | **0.470**| 0.461             |

Four methods are compared:

- **Baseline** — single vector search with the original query.
- **RAG-Fusion** — original + 4 LLM-generated queries, combined via Reciprocal Rank Fusion.
- **+Diverse** — same as RAG-Fusion but with a prompt that explicitly asks for different angles, synonyms, and varied specificity. Best overall performer.
- **+Diverse+Weighted** — diverse queries + 3× RRF weight on the original query. The extra weight narrows the net and pulls numbers back toward baseline.

The **diverse prompt** is the biggest win — it consistently outperforms both baseline and standard RAG-Fusion across precision, NDCG, and MRR. The key insight is that the default prompt produces semantically close query variations that search the same embedding neighborhood, while the diverse prompt forces the LLM to explore genuinely different angles.

### Running the evaluation

```bash
# Baseline only (no API key needed)
python evaluate.py --sample 10 --methods baseline

# Default comparison (requires OPENAI_API_KEY)
python evaluate.py --sample 50

# All four methods
python evaluate.py --sample 50 --methods baseline rag-fusion rag-fusion-diverse rag-fusion-weighted

# Custom parameters
python evaluate.py --sample 100 --k 5 10 --data-dir ./datasets
```

The NFCorpus dataset (~3MB) is downloaded automatically on first run. ChromaDB embeddings are persisted locally so subsequent runs skip ingestion.