# OpenAI's Decisions API as a reranker

October 2026. NFCorpus, n=200, the same sample and arms as the Jev runs in
[`../jev-in-the-pipeline/`](../jev-in-the-pipeline/README.md), where section 8 summarises this page.
The code is [`eval/decisions.py`](../../eval/decisions.py).

## What it is, and why test it

OpenAI announced the Decisions API at DevDay on 29 September 2026. It runs on GPT-6 Luna. You send one input
string and a list of questions with fixed answers, and each comes back as a `predicate` (a probability), a
`choice`, a `score` (an expected level over levels you label), or a `refusal`. There's no generated text.

That's the same contract as Jev, which TypeSafe calls a System One model, after Kahneman's fast,
automatic kind of thinking. OpenAI doesn't use that term, so I'll call them both decision models. The
reason to test a second one is the result in section 2 of the Jev write-up. Fusion's lift doubled under
Jev, and my explanation was that a model reading each document as a question copes with fusion's wider
pool better than a cross-encoder scoring similarity. If that's right, it should hold for any decision
model, not just Jev.

There are two practical differences from Jev. The input is plain text rather than structured state, so
each document gets an inline label (`[D00] ...`) and each question names its label. And all 50 candidates
fit in one request, where Jev's documented batch is 30.

I asked it two kinds of question:

- **yes/no** (`predicate`): "Document [Dxx] is relevant to the query", worded as in `eval/jev.py`.
- **rubric** (`score`): the four levels from Jev's Score mode (off-topic, related but doesn't supply the
  answer, partly supplies it, fully supplies it). The API returns the expected level, scaled here to 0 to 1.

## Results

NDCG@10, with 95% paired-bootstrap CIs on the lift. Bold means the CI excludes zero.

| Reranker | Baseline | Hybrid lift | Fusion lift (hybrid_diverse) |
|---|---|---|---|
| bge-reranker-large | 0.331 | +0.012 | **+0.025** [+0.012, +0.041] |
| Luna, rubric | 0.370 | -0.002 [-0.021, +0.017] | **+0.037** [+0.014, +0.062] |
| Luna, yes/no | 0.373 | -0.010 [-0.031, +0.011] | **+0.039** [+0.019, +0.061] |
| Jev, yes/no (run 1 / run 2) | 0.382 / 0.384 | +0.015 / +0.013 | **+0.050 / +0.052** |
| Jev, rubric | 0.388 | +0.010 | **+0.050** |

Luna sits between bge-large and Jev, both on its own and in how much fusion adds, which makes it the third
reranker in a row where a stronger reranker gets a bigger lift from fusion. Arm for arm, Jev is ahead by
0.025 to 0.030 on the hybrid and fusion pipelines, with the CI excluding zero against all three Jev runs,
and by 0.012 to 0.018 on the single-query baseline, which is borderline. Both question types beat bge-large on all three
arms compared (baseline, hybrid, hybrid_diverse; every CI excludes zero, though for yes/no on hybrid only just).

### Where fusion's extra documents go

Fusion puts more relevant documents into the pool of 50 (6.3 per query against 5.6 from one query). The
question is how many of those the reranker moves into the top ten. Same method as section 2 of the Jev
write-up (`eval/pool_recall.py`, output in [`../jev-in-the-pipeline/pool_recall.json`](../jev-in-the-pipeline/pool_recall.json)):

| Reranker | Relevant in top 10, alone | With fusion | Share of fusion's extra finds reaching the top 10 | Hit rate on the pool, alone to fusion |
|---|---|---|---|---|
| MiniLM | 2.27 | 2.37 | 14% | 40% to 38% |
| bge-base | 2.20 | 2.33 | 19% | 39% to 37% |
| bge-large | 2.38 | 2.55 | 25% | 42% to 40% |
| Luna, rubric | 2.70 | 3.01 | 47% | 48% to 48% |
| Luna, yes/no | 2.66 | 3.01 | 52% | 47% to 48% |
| Jev (run 1 / run 2) | 2.77 | 3.13 / 3.17 | 54% / 60% | 49% to 50% |

This is the result I was looking for. Every cross-encoder gets a little worse at picking relevant documents
out of the wider fusion pool, and both decision models don't. Luna moves about half of fusion's extra finds
into its top ten, close to Jev and about twice what bge-large manages. In the Jev write-up I called the
explanation interpretation rather than something I'd measured. It's still two models on one corpus, but it
now looks like a property of decision models rather than of one product.

### Other things worth knowing

- **Pipeline order.** Reranking each list and then fusing gives -0.008 (rubric) and -0.016 (yes/no), both
  n.s., against +0.037 and +0.039 for fusing first. That fits the pattern from section 3 of the Jev
  write-up, where reranking each list first stops helping as the reranker gets stronger. The paper's ordering
  without BM25 (vector only, four rewrites) hurts outright: -0.023 and -0.036, both significant.
- **Hybrid on its own does nothing under Luna,** where it adds +0.010 to +0.015 under Jev and the
  cross-encoders. I don't know why yet.
- **Yes/no and rubric come out the same.** Arm by arm the differences are within ±0.006 and every CI spans zero.
  The yes/no probabilities come back rounded to two decimals and are close to binary (two to six distinct
  values in a pool of 50), so most of a pool ties and keeps its retrieval order. On NFCorpus that cost
  nothing I could measure.

## How it behaves

- **Deterministic.** In a probe, two identical calls gave identical scores for all 50 documents in both
  modes. Jev moves 48% of its scores between identical calls.
- **No refusals** in any call across both runs.
- **About 25,000 input tokens per 50-document pool, and zero output tokens.** Latency in the probe was 0.5 to
  1.8 seconds per call.
- **One read timeout** (120 seconds) across both runs. `_post` now retries on timeouts and connection errors.

## Cost

OpenAI hasn't published a separate price for the Decisions API. At GPT-6 Luna's standard input rate of
$0.10 per million tokens, the rubric run used 30.0M input tokens (about $3.00) and the yes/no run 36.4M
(about $3.64, including pools scored again after the timeout restart). That's about $2.50 per 1,000
queries to rerank 50 abstracts, against about $0.95 for Jev.

## Caveats

- One corpus and one run per question type. Luna is deterministic, so a second run would give the same
  scores. Retrieval isn't quite: Chroma's HNSW index returns a different top 50 for about 29% of queries
  each time it's loaded. After reranking, that moved NDCG@10 by at most 0.0005 in a bge test, so it doesn't
  touch the comparisons above.
- The Jev runs are from 26 and 27 September (`jev-1.13.0`) and the Luna runs from 7 October. A same-day run
  of both would make the head-to-head cleaner.
- The API is in limited preview, so the model, price and rounding could all change.

## Reproduce

```bash
python -m eval.steelman --sample 200 --rerank-model decisions-score:gpt-6-luna \
  --out experiments/openai-decisions-api/results/steelman_decisions_score_n200_run1.json
python -m eval.steelman --sample 200 --rerank-model decisions:gpt-6-luna \
  --out experiments/openai-decisions-api/results/steelman_decisions_predicate_n200_run1.json
python -m eval.bootstrap_ci experiments/openai-decisions-api/results/steelman_decisions_score_n200_run1.json
```

Needs `OPENAI_API_KEY` in `.env` and preview access to the Decisions API. Answers are cached locally in
`decisions_cache.json` (not committed). The client calls the endpoint over plain HTTP, because openai
SDK versions before 3.26.0 don't include it.
