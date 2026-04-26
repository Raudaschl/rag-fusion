# RAG-Fusion: The Next Frontier of Search Technology

## Overview

RAG-Fusion is a search methodology that aims to bridge the gap between traditional search paradigms and the multifaceted dimensions of human queries. Where Retrieval Augmented Generation (RAG) fuses vector search with generative models, RAG-Fusion goes a step further — employing multiple query generation and Reciprocal Rank Fusion to re-rank search results. The aim is to surface relevant material a single phrasing of the query would miss, particularly when the user's vocabulary doesn't match how the corpus is indexed.

For the full story behind the approach, see the article: [Forget RAG, the Future is RAG-Fusion](https://adrianraudaschl.com/blog/forget-rag-the-future-is-rag-fusion/).

> **Where this technique fits, in one line:** RAG-Fusion is a precision tool for the recall-scarce tail — queries where the user's words don't match the corpus's words. It's not a default-on quality boost for well-formed queries against well-curated corpora.
>
> The full empirical case for that read — including a replication of arXiv 2603.02153v1, a stronger-reranker steelman, and end-to-end answer-quality eval — lives in [`experiments/arxiv-2603-02153-replication/`](./experiments/arxiv-2603-02153-replication/README.md).

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

## When to use RAG-Fusion

The technique earns its compute when three conditions hold:

1. **Terminology mismatch between user queries and indexed text** (lay vs technical names, jargon, paraphrase).
2. **Recall matters more than precision** — missing a relevant document is more costly than including a marginal one.
3. **The downstream consumer can handle topically-broad context** — either a strong synthesis LLM, or a UI that surfaces multiple candidates rather than one canonical answer.

Strong-fit examples:
- Academic / scientific literature search, biomedical research
- Patent prior-art search, legal e-discovery, regulatory review
- Long-tail e-commerce ("phone holder thing for car" → "magnetic vent mount")
- Cold-start retrieval over specialist corpora the embedding model hasn't seen
- Exploratory / "show me what's out there" workflows

Poor-fit examples:
- FAQ chatbots and curated customer-support knowledge bases
- Latency-critical retrieval (voice, autocomplete, sub-second-p95 chat)
- High-volume / margin-thin consumer search
- Code or identifier search (precision-dominated)
- Structured data, knowledge graphs, SQL-backed retrieval

For mixed workloads — most production retrieval — the right pattern is **adaptive routing**: run baseline+rerank on every query, fire fusion only when a cheap weakness signal trips. This captures the long-tail wins, eliminates the regression cases on easy queries, and pays for fusion's compute only on traffic where it earns it. See [`experiments/arxiv-2603-02153-replication/`](./experiments/arxiv-2603-02153-replication/README.md) for the data behind these recommendations, including cost and latency analysis across corpus sizes and data types.

## Project Structure

```
├── main.py                 # Core RAG-Fusion pipeline
├── evaluate.py             # Evaluation CLI (baseline + fusion variants, optional --rerank)
├── test_main.py            # Unit tests
├── eval/
│   ├── dataset.py          # NFCorpus download & loading
│   ├── metrics.py          # IR metrics (Precision, Recall, NDCG, MRR)
│   ├── retrieval.py        # Retrieval methods (BM25, vector, hybrid, RAG-Fusion variants)
│   ├── rerank.py           # Cross-encoder reranking stage
│   ├── sweep.py            # Pool-size and N-rewrites sweep driver
│   ├── steelman.py         # Pipeline-ordering / truncation / difficulty tests
│   └── qualitative.py      # End-to-end answer-quality eval driver
├── experiments/
│   └── arxiv-2603-02153-replication/  # Full replication writeup + raw results
└── .env.example            # Environment template
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

### Beyond retrieval-only metrics

The table above measures retrieval quality in isolation. A more honest production picture requires layering in a cross-encoder reranker, varying the candidate-pool size, and reading the actual generated answers — because retrieval lifts at NDCG@10 don't always survive into the answers a user sees. Adding a `--rerank` flag changes the comparison materially: a strong reranker (`bge-reranker-large`) absorbs ~68% of fusion's lift on this corpus, fusion goes net-negative on the easy ~67% of queries, and fusion's value concentrates almost entirely in the recall-scarce tail. See [`experiments/arxiv-2603-02153-replication/`](./experiments/arxiv-2603-02153-replication/README.md) for the full picture: pool-size and N-rewrites sweeps, pipeline-ordering steelman, end-to-end answer eval, and cost/latency analysis by corpus size and data type.

```bash
# Production-style comparison: candidate pool of 50, then reranked + truncated
python evaluate.py --sample 50 --rerank --candidate-pool 50 \
  --methods baseline hybrid rag-fusion rag-fusion-diverse hybrid-diverse
```

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