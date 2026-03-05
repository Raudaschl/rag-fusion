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

| Metric    | k  | Baseline | RAG-Fusion | Delta   |
|-----------|----|----------|------------|---------|
| Precision | 5  | 0.264    | 0.276      | +4.5%   |
| Precision | 10 | 0.222    | 0.246      | +10.8%  |
| Precision | 20 | 0.180    | 0.199      | +10.6%  |
| Recall    | 5  | 0.110    | 0.165      | +50.8%  |
| Recall    | 10 | 0.133    | 0.195      | +46.9%  |
| Recall    | 20 | 0.184    | 0.234      | +27.1%  |
| NDCG      | 5  | 0.304    | 0.348      | +14.3%  |
| NDCG      | 10 | 0.286    | 0.339      | +18.5%  |
| NDCG      | 20 | 0.280    | 0.327      | +16.9%  |
| MRR       | -  | 0.440    | 0.476      | +8.1%   |

RAG-Fusion shows consistent improvements across all metrics, with the largest gains in **recall** (+27-51%) — generating multiple query variations surfaces relevant documents that a single query misses. This is exactly the kind of hidden knowledge retrieval that the approach was designed to unlock.

### Running the evaluation

```bash
# Baseline only (no API key needed)
python evaluate.py --sample 10 --methods baseline

# Full comparison (requires OPENAI_API_KEY)
python evaluate.py --sample 50

# Custom parameters
python evaluate.py --sample 100 --k 5 10 --data-dir ./datasets
```

The NFCorpus dataset (~3MB) is downloaded automatically on first run. ChromaDB embeddings are persisted locally so subsequent runs skip ingestion.