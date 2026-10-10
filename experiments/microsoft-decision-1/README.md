# Microsoft-Decision-1 as a reranker

October 2026. NFCorpus, n=200, the same sample and arms as the Jev and OpenAI Decisions API runs. Section 8 of
[`../jev-in-the-pipeline/`](../jev-in-the-pipeline/README.md#8-two-more-decision-models-openais-decisions-api-and-microsoft-decision-1)
puts the three decision models side by side; this page has the detail.

## What it is

Microsoft released Microsoft-Decision-1 on 9 October 2026, in Foundry and on OpenRouter. It's a decision model:
you send some context and questions with fixed answers, and it returns a probability for each answer rather than
generating text. Microsoft says it's post-trained from Qwen3.5-9B. It has a 32,768-token context, costs $0.042
per million input tokens with free output (the same as Jev), and on OpenRouter it's served by Azure.

It speaks TypeSafe's SystemOne request format exactly (state, plus yes/no, choice and score questions), and
OpenRouter exposes that format at `/api/v1/systemone`. So no new code was needed. I pointed the existing Jev
client (`eval/jev.py`) at OpenRouter and changed the model name, which also means it got exactly the same
prompts and the same 30-document batches as Jev.

## Results

NDCG@10, with 95% paired-bootstrap CIs on the lift. Bold means the CI excludes zero.

| Arm | Yes/no | Lift | Rubric | Lift |
|---|---|---|---|---|
| baseline+rerank | 0.378 | | 0.378 | |
| hybrid+rerank | 0.386 | +0.007 [-0.007, +0.022] | 0.387 | +0.009 [-0.006, +0.024] |
| fuse_then_rerank (vector only) | 0.397 | **+0.018** [+0.000, +0.037] | 0.404 | **+0.026** [+0.008, +0.045] |
| hybrid_diverse+rerank | 0.417 | **+0.038** [+0.019, +0.059] | 0.414 | **+0.035** [+0.016, +0.056] |
| rerank each list, then fuse (paper's order) | 0.346 | **-0.032** [-0.057, -0.008] | 0.348 | **-0.031** [-0.054, -0.007] |
| hybrid, rerank each list, then fuse | 0.363 | -0.015 [-0.038, +0.007] | 0.363 | -0.016 [-0.038, +0.006] |

Against the other rerankers, on the same arm (yes/no):

- **bge-large:** ahead by 0.047 on the baseline, 0.043 on hybrid and 0.061 with fusion. Every CI excludes zero.
- **Luna (OpenAI's Decisions API):** tied on the baseline and with fusion, and ahead by 0.017 to 0.023 on hybrid.
- **Jev:** within noise on the baseline (0.004 to 0.010 behind) and behind by 0.016 to 0.021 with fusion, where
  the CI excludes zero against all three Jev runs.

The yes/no and rubric questions come out the same, arm for arm, as they did for Jev and Luna.

### Where fusion's extra documents go

Same method as section 2 of the Jev write-up. The data is in [`../jev-in-the-pipeline/pool_recall.json`](../jev-in-the-pipeline/pool_recall.json).

| Reranker | Share of fusion's extra finds reaching the top 10 | Hit rate on the pool, alone to fusion |
|---|---|---|
| Cross-encoders (MiniLM, bge-base, bge-large) | 14% to 25% | drops 2 to 3 points |
| Microsoft-Decision-1 (yes/no, rubric) | 37%, 37% | 48% to 47%, 49% to 48% |
| Luna (rubric, yes/no) | 47%, 52% | flat |
| Jev (run 1, run 2) | 54%, 60% | 49% to 50% |

Microsoft-Decision-1 sits partway. It does better with fusion's extra documents than any cross-encoder, but not as
well as Luna or Jev, and its accuracy slips a little on the wider pool where theirs holds. That's why section 8 now
describes this as a gradient that decision models sit high on rather than a clean split between kinds of model.

## How it behaves

- **Fine-grained scores.** In a three-query probe it gave 15 to 19 distinct yes/no probabilities per 30
  documents and 28 to 29 distinct rubric scores. Luna's yes/no answers are close to binary by comparison.
- **Nearly deterministic.** In the same probe, most scores moved between two identical calls, but by 0.03
  at most. That's about the size of Jev's wobble.
- **Fast.** 0.7 to 1.3 seconds per 30-document call through OpenRouter.
- **Neither run failed.**

## Cost

At $0.042 per million input tokens, the same as Jev, reranking 50 abstracts should come to roughly $0.95 per
1,000 queries, assuming token counts similar to Jev's. I didn't log tokens for these runs, so treat that as an
estimate.

## Caveats

- One corpus and one run per question type. Given the small wobble, a second run would likely move the averages
  by no more than Jev's two runs differ (0.002 to 0.004), but I haven't checked.
- Retrieval drifts a little between processes (Chroma's HNSW index), which moved NDCG@10 by at most 0.0005 in a
  bge test.
- The model came out the day before these runs, and OpenRouter lists one provider. It may change.

## Reproduce

```bash
# OPENROUTER_API_KEY must be set. The SDK appends /v1, so the base URL stops at /api.
export TYPESAFE_BASE_URL=https://openrouter.ai/api JEV_API_KEY=$OPENROUTER_API_KEY \
       JEV_MODEL=microsoft/microsoft-decision-1 JEV_TIMEOUT=120
JEV_CACHE_PATH=./msd1_cache_noul.json  python -m eval.steelman --sample 200 --rerank-model jev \
  --out experiments/microsoft-decision-1/results/steelman_msd1_yesno_n200_run1.json
JEV_CACHE_PATH=./msd1_cache_score.json python -m eval.steelman --sample 200 --rerank-model jev-score \
  --out experiments/microsoft-decision-1/results/steelman_msd1_rubric_n200_run1.json
python -m eval.bootstrap_ci experiments/microsoft-decision-1/results/steelman_msd1_yesno_n200_run1.json
```
