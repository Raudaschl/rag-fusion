# Gemini Decision 1 as a reranker

November 2026. NFCorpus, n=200, the same sample and arms as the Jev, OpenAI Decisions API and
Microsoft-Decision-1 runs. Section 8 of
[`../jev-in-the-pipeline/`](../jev-in-the-pipeline/README.md#8-three-more-decision-models-openai-microsoft-and-google)
puts the four decision models side by side; this page has the detail. The code is
[`eval/gemini_decisions.py`](../../eval/gemini_decisions.py).

## What it is

Google launched the Gemini Decisions API on 11 November 2026, with a new model, Gemini Decision 1. The contract
is the same as the other decision models: one shared input, up to 1,000 questions with fixed answers (yes/no,
which Google calls `noul`, then `choice` and `score`), and a calibrated probability back for each answer instead
of generated text. It has a 262,144-token context and costs $0.10 per million input tokens with no output charge,
the same as OpenAI's Decisions API.

I called it through OpenRouter's Decisions endpoint (`/api/alpha/decisions`), which serves it from Google AI
Studio as the snapshot `gemini-decision-1-20261111`. Calling Google directly needs prepaid credit on the Gemini
key, and mine had none (HTTP 402). The client handles both routes: a model name with a slash
(`google/gemini-decision-1`) goes through OpenRouter, and a bare name (`gemini-decision-1`) goes to Google's
`v1beta/decisions`.

The client copies the one I wrote for OpenAI's Decisions API, so the two compare like for like. It sends the same
input string (the query, then each document labelled `[D00]`, `[D01]`, ...), the same question wording, and all
50 candidates in one request. I asked both kinds of question used for the other three models:

- **yes/no** (`noul`): "Document [Dxx] is relevant to the query", worded as in `eval/decisions.py`.
- **rubric** (`score`): the four levels from Jev's Score mode. The API returns the probability-weighted expected
  level, scaled here to 0 to 1.

## Results

NDCG@10, with 95% paired-bootstrap CIs on the lift. Bold means the CI excludes zero.

| Arm | Yes/no | Lift | Rubric | Lift |
|---|---|---|---|---|
| baseline+rerank | 0.385 | | 0.392 | |
| hybrid+rerank | 0.392 | +0.007 [-0.009, +0.023] | 0.392 | +0.001 [-0.017, +0.017] |
| fuse_then_rerank (vector only) | 0.401 | +0.016 [-0.001, +0.036] | 0.405 | +0.013 [-0.008, +0.034] |
| hybrid_diverse+rerank | 0.437 | **+0.052** [+0.031, +0.074] | 0.433 | **+0.041** [+0.022, +0.061] |
| rerank each list, then fuse (paper's order) | 0.347 | **-0.038** [-0.062, -0.014] | 0.348 | **-0.044** [-0.070, -0.019] |
| hybrid, rerank each list, then fuse | 0.364 | -0.021 [-0.045, +0.002] | 0.362 | **-0.030** [-0.055, -0.006] |

Against the other rerankers, paired by query on the same arm:

- **Jev:** a tie. On the baseline, hybrid and fusion arms, against all three Jev runs, every difference is
  within ±0.009 and every CI spans zero. With fusion the gaps are +0.005, +0.001 and -0.001 (yes/no) and +0.000,
  -0.004 and -0.005 (rubric).
- **Luna (OpenAI's Decisions API):** ahead by 0.021 to 0.030 with fusion and 0.023 to 0.030 on hybrid, with the
  CI excluding zero in all eight pairings. On the baseline it's ahead by 0.012 to 0.022, significant in three of
  the four pairings.
- **Microsoft-Decision-1:** ahead by 0.016 to 0.024 with fusion (significant in all four pairings). On the
  baseline it's ahead by 0.007 (yes/no, n.s.) or 0.013 (rubric, just significant), and on hybrid they tie.
- **bge-large:** ahead by 0.054 to 0.061 on the baseline, 0.049 on hybrid and 0.077 to 0.081 with fusion.
  Every CI excludes zero.

The yes/no and rubric questions come out the same, as they did for the other three decision models. Arm by arm
the differences are within ±0.007, and the difference in fusion lift (+0.052 against +0.041) is +0.011
[-0.005, +0.028].

### Where fusion's extra documents go

Same method as section 2 of the Jev write-up, against the same pool counts (5.6 relevant documents per query in the
single-query pool of 50, 6.3 with fusion). The data is in [`../jev-in-the-pipeline/pool_recall.json`](../jev-in-the-pipeline/pool_recall.json).

| Reranker | Relevant in top 10, alone | With fusion | Share of fusion's extra finds reaching the top 10 | Hit rate on the pool, alone to fusion |
|---|---|---|---|---|
| Cross-encoders (MiniLM, bge-base, bge-large) | 2.20 to 2.38 | 2.33 to 2.55 | 14% to 25% | drops 2 to 3 points |
| Microsoft-Decision-1 (yes/no, rubric) | 2.72, 2.75 | 2.97, 3.00 | 37%, 37% | 48% to 47%, 49% to 48% |
| Luna (rubric, yes/no) | 2.70, 2.66 | 3.01, 3.01 | 47%, 52% | flat |
| Gemini Decision 1, yes/no | 2.76 | 3.16 | 59% | 49% to 50% |
| Gemini Decision 1, rubric | 2.82 | 3.14 | 49% | 50% to 50% |
| Jev (run 1, run 2) | 2.77 | 3.13, 3.17 | 54%, 60% | 49% to 50% |

Gemini Decision 1 sits with Jev at the top. It moves about half or more of fusion's extra finds into its top
ten, and its accuracy on the pool holds when fusion widens it. Until now Jev was alone there. With a second model
there, built by a different company on a different base, what Jev does with fusion looks less like something
particular to Jev. The gradient from section 8 is still there, since Microsoft-Decision-1 is still partway, but
Jev isn't the only model at the top of it any more.

## How it behaves

- **Deterministic.** In a probe on two pools, two identical calls a few seconds apart gave identical scores for
  all 50 documents, in both modes. The second call took as long as the first, so it doesn't look like a cached
  response.
- **Fine-grained scores.** 49 or 50 distinct values in every pool of 50, in both modes. The yes/no probability is
  clipped to [0.01, 0.99], but ties within a pool are almost nonexistent. Luna's yes/no answers, by comparison,
  come back rounded and close to binary.
- **No blocks or refusals** in 2,800 calls per run, on a medical corpus.
- **Fast.** 0.5 to 1 second per 50-document call through OpenRouter. Each run took about 30 minutes, with the two
  running side by side.
- **About 20,000 input tokens per 50-document pool for yes/no, and 23,000 for the rubric.** That's slightly less
  than Luna (about 25,000) for the same text.

## Cost

$0.10 per million input tokens, with no output charge. OpenRouter's billed cost on a probe call matched that rate
exactly. The yes/no run used 23.8M input tokens (about $2.38) and the rubric run 27.5M (about $2.75). Reranking 50
abstracts comes to about $2.00 per 1,000 queries for yes/no and $2.30 for the rubric. That compares with about
$2.50 for Luna and $0.95 for Jev and Microsoft-Decision-1.

## Caveats

- One corpus and one run per question type. Gemini Decision 1 was deterministic in the probe, so a second run
  would mostly re-check retrieval. Chroma's HNSW index returns a different top 50 for about 29% of queries each
  time it's loaded, which moved NDCG@10 by at most 0.0005 in a bge test.
- These runs went through OpenRouter, not Google's own endpoint. It's the same model snapshot, served by Google
  AI Studio, so I'd expect the same scores, but I haven't checked.
- The Jev comparisons are against the September runs on `jev-1.13`. Jev 1.14 came out on 15 October and hasn't
  been tested here.
- The model came out eleven days before these runs and the API is `v1beta`, so the model, price and behaviour
  could all change.

## Reproduce

```bash
# Through OpenRouter (OPENROUTER_API_KEY must be set). Answers are cached in the local cache files, which aren't committed.
GEMINI_DECISIONS_CACHE_PATH=./gemini_decisions_cache_noul.json python -m eval.steelman --sample 200 \
  --rerank-model gemini-decisions:google/gemini-decision-1 \
  --out experiments/gemini-decision-1/results/steelman_gemini_yesno_n200_run1.json
GEMINI_DECISIONS_CACHE_PATH=./gemini_decisions_cache_score.json python -m eval.steelman --sample 200 \
  --rerank-model gemini-decisions-score:google/gemini-decision-1 \
  --out experiments/gemini-decision-1/results/steelman_gemini_rubric_n200_run1.json
python -m eval.bootstrap_ci experiments/gemini-decision-1/results/steelman_gemini_yesno_n200_run1.json

# Or directly from Google (GEMINI_API_KEY, with prepaid credit): drop the "google/" prefix,
# e.g. --rerank-model gemini-decisions:gemini-decision-1
```

Use a separate cache path per process. Each process loads the cache when it starts and rewrites it when it saves,
so two runs sharing one file would overwrite each other's answers.
