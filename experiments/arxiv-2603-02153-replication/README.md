# A small replication of arXiv 2603.02153v1 on NFCorpus

> **Paper under replication:** *"Scaling RAG Fusion: Lessons from an Industry Deployment"* — arXiv [2603.02153v1](https://arxiv.org/html/2603.02153v1) (March 2026). Their headline claim: retrieval-fusion gains "evaporate" after cross-encoder reranking and Top-K truncation in production. This document is an independent replication on a different benchmark (NFCorpus, BEIR), with an extended set of fusion variants and statistical confidence intervals.
>
> **Raw data and reproducible artefacts** are in [`results/`](./results/):
>
> | File | Contents |
> |---|---|
> | `steelman_base_n200_full.json` | 6-method comparison at n=200 with `bge-reranker-base` (per-query metrics + buckets) |
> | `steelman_large_n200_full.json` | Same 6-method comparison with `bge-reranker-large` (the headline) |
> | `steelman_flashrank_n200.json` | Same 6-method comparison with FlashRank (`ms-marco-MiniLM-L-12-v2`) — paper's likely reranker |
> | `steelman_flashrank_n1_n200.json` | Strict-replication condition: FlashRank + N=1 LLM rewrite (paper's exact configuration) |
> | `sweep_n200.json` | Pool-size and N-rewrites sweeps at n=200 |
> | `answer_eval_n200.json` | LLM-judge end-to-end eval, vector-only methods (200 queries × 3 methods) |
> | `answer_eval_hybrid_n200.json` | LLM-judge end-to-end eval, hybrid variants (200 queries × 3 methods) |
> | `headline_table_n200.json` | Retrieval-only headline table with paired-bootstrap CIs for the top-level README |
> | `qualitative.txt` | Hand-readable verbose log of 8 queries with retrieved docs + generated answers |
> | `steelman.json`, `steelman_large.json`, `sweep.json` | Earlier n=30 runs — kept for the "what changed at n=200" comparison |
>
> Code in [`eval/`](../../eval/): `steelman.py`, `sweep.py`, `bootstrap_ci.py`, `answer_eval.py`, `qualitative.py`, `saved_queries.py`. Reproduction commands at the bottom of this document.

**TL;DR.** Replicating arXiv 2603.02153v1 on NFCorpus, the paper's "fusion gains evaporate after rerank+truncation" claim partly reproduces — *for vector-only fusion with a single LLM rewrite*. But when you compare the technique's **strong variant** (`hybrid_diverse+rerank`: BM25 + vector × 4 rewrites, RRF, then cross-encoder rerank) against the same baseline at n=200 with paired-bootstrap CIs, fusion has **a real, statistically significant lift on every metric and every bucket**:

- **NDCG@10**: +0.021 [+0.007, +0.036] overall; +0.016 [+0.001, +0.032] on rich queries; +0.031 [+0.005, +0.072] on recall-scarce queries — significant on all three.
- **LLM-judge answer quality**: mean score 1.17 vs baseline's 1.07 (+9% relative). Win/tie/loss vs baseline: **25 wins / 11 losses overall (2× wins to losses), 18 wins / 8 losses on rich queries**. Fusion produces measurably better answers, not just better rankings.
- **The lift survives across three rerankers spanning a range of baseline NDCG@10 strength** (bge-base: +0.023; FlashRank/ms-marco-MiniLM-L-12-v2: +0.014; bge-large: +0.021 — all CIs exclude zero). Counter-evidence to the paper's "stronger rerankers absorb fusion gains" mechanism *for the strong fusion variant*. The mechanism does hold for vector-only fusion, where there's not much to absorb anyway.

The story behind the story: the technique was overclaimed at small samples, then under-claimed once we corrected for n=30 noise but tested only the weak variant, then vindicated once we tested the variant that's actually in the repo's recommended configuration, then sharpened once we added FlashRank (the paper's actual reranker) as a third reranker condition. **The paper is right about a weak fusion variant tested with their exact configuration; wrong about the technique in general.** Adaptive routing remains the smartest deployment shape (it captures the wins and avoids a small loss tail), but always-on `hybrid_diverse+rerank` is now defensible. Methodology notes and "what changed across iterations" details below — this writeup went through four iterations of reversals and corrections on the way to the final picture.

## Disclosure first

I wrote the original RAG-Fusion article in 2024, so I have an obvious stake in the conclusion and you should weigh the framing accordingly. I tried to design the experiment to give the paper's claim a fair shot — same fusion-then-rerank ordering, a credible cross-encoder, a real BEIR benchmark — but selection effects creep in even when I'm trying to be careful. If you spot one, please open an issue.

## What the paper claims (very briefly)

arXiv 2603.02153v1 reports that on a 115-query synthetic enterprise-support set, two-query fusion (Q1 + one LLM rewrite) followed by FlashRank reranking and Top-K truncation either matches or slightly underperforms a single-query baseline. Their proposed mechanism: the reranker re-anchors on Q1, and the rewrite's marginal candidates are squeezed out at truncation. They do note (§6.6) that fusion still helps in "recall-scarce" query regimes.

## What I changed and why

| Choice | Mine | Theirs |
|---|---|---|
| Dataset | NFCorpus (biomedical, BEIR) | 115 synthetic RAGAS support queries |
| Sample size | 200 queries (paired-bootstrap CIs) | 115 queries |
| Reranker | Three rerankers tested: `ms-marco-MiniLM-L-12-v2` (33M params, served via FlashRank — the paper's likely reranker), `BAAI/bge-reranker-base` (278M), `BAAI/bge-reranker-large` (560M). On NFCorpus, baseline NDCG@10 strength is bge-base (0.305) < FlashRank (0.320) < bge-large (0.334) — param count isn't strength here. | "FlashRank cross-encoder" — library, not a specific model. Paper doesn't specify which model FlashRank loaded; default is `ms-marco-MiniLM-L-12-v2`. We use that as the like-for-like condition. |
| Rewrites (default) | 4; N=1 also tested as strict-replication condition | 1 |
| Pipeline order | Both orderings tested head-to-head | Retrieve → rerank per-query → fuse → truncate |
| Candidate pool before rerank | 50 (default); pool sweep also covers 10/20/30/75 | 10 (confirmed: their Table 6 shows "Flashrank time (K=10)") |

These are not small changes, and I'm not claiming a like-for-like replication. NFCorpus is harder than synthetic support queries — it's exactly the "recall-scarce regime" the paper says fusion still helps in. So part of the divergence may just be that I'm testing a regime they already conceded.

## Headline numbers (n=200 NFCorpus, bge-reranker-large, 95% paired-bootstrap CIs)

Six methods compared head-to-head against vector-only baseline+rerank. The table reads as "lift in NDCG@10 over baseline, with 95% CI; bold = CI excludes zero":

| Method | All (n=200) | Rich (n=141) | Scarce (n=59) |
|---|---|---|---|
| baseline+rerank (vector → rerank) | — | — | — |
| hybrid+rerank (BM25+vector → RRF → rerank) | +0.009 [−0.003, +0.021] | +0.007 [−0.009, +0.023] | **+0.015 [+0.003, +0.032]** |
| fuse_then_rerank (vector × 5 rewrites → RRF → rerank) | +0.005 [−0.007, +0.019] | −0.001 [−0.013, +0.010] | +0.019 [+0.000, +0.055] |
| **hybrid_diverse+rerank** (BM25+vector × 5 rewrites → RRF → rerank) | **+0.021 [+0.007, +0.036]** | **+0.016 [+0.001, +0.032]** | **+0.031 [+0.005, +0.072]** |
| rerank_per_query_then_fuse (paper ordering, vector only) | −0.007 [−0.033, +0.018] | −0.029 [−0.062, +0.004] | **+0.044 [+0.014, +0.079]** |
| hybrid_per_query_rerank_then_fuse (BM25+vector × 5 → per-query rerank → RRF) | +0.000 [−0.023, +0.023] | −0.014 [−0.044, +0.017] | **+0.035 [+0.012, +0.062]** |

**`hybrid_diverse+rerank` is the only method with a statistically distinguishable positive lift on every bucket** under the strong reranker — the technique works, when properly configured.

What this resolves:

- **Vector-only fusion (`fuse_then_rerank`) is roughly a wash on average** (+0.005 ALL, CI crosses zero). This matches the paper's mechanism for the variant they tested.
- **Adding BM25 alone (`hybrid+rerank`, no LLM rewrites) gets you most of the recall-scarce lift but nothing on rich queries.** BM25 is doing real work but isn't sufficient.
- **Adding LLM rewrites alone (`fuse_then_rerank`) does basically nothing on rich queries.** Rewrites alone aren't sufficient either.
- **Combining both (`hybrid_diverse+rerank`) is where the technique actually helps.** Lift is significant on all three buckets. The two channels are complementary, not redundant.
- **Paper's pipeline ordering trends negative on rich queries** with strong reranker (−0.029 [−0.062, +0.004]) — per-query truncation drops a top-rank doc the fused-then-reranked path keeps.
- **The new `hybrid_per_query_rerank_then_fuse` helps with weak rerankers but goes flat with strong ones.** Useful negative result: when the final reranker is strong, integrating the cross-encoder per-query is wasted compute.

### Across three rerankers (NDCG@10 lift over baseline+rerank, ALL bucket)

The "stronger rerankers absorb fusion gains" claim is the paper's load-bearing mechanism. To test it, I ran the 6-method steelman against three different rerankers spanning a range of baseline NDCG@10 strength:

| Reranker | Baseline NDCG@10 | hybrid_diverse+rerank lift [95% CI] | fuse_then_rerank lift [95% CI] |
|---|---|---|---|
| `bge-reranker-base` (278M) | 0.305 | **+0.023 [+0.010, +0.039]** | +0.009 [−0.002, +0.022] |
| FlashRank / `ms-marco-MiniLM-L-12-v2` (33M, paper's likely reranker) | 0.320 | **+0.014 [+0.001, +0.030]** | +0.001 [−0.010, +0.015] |
| `bge-reranker-large` (560M) | 0.334 | **+0.021 [+0.007, +0.036]** | +0.005 [−0.007, +0.019] |

The hybrid+diverse fusion lift survives all three rerankers, with CIs that exclude zero in every case. It is **not monotonic in reranker strength** — there's no clean "weaker reranker → bigger lift" pattern. The paper's "rerankers absorb fusion gains" mechanism is counter-evidenced once you measure across three reranker conditions on the strong fusion variant.

Vector-only fusion (`fuse_then_rerank`), in contrast, has lifts that all cross zero across the three rerankers — confirming the paper's finding *for that specific weak variant*.

(One thing worth correcting from my earlier framing: I initially called FlashRank "the weak end" because of its 33M parameter count. On NFCorpus, the model FlashRank ships with — `ms-marco-MiniLM-L-12-v2` — actually scores higher baseline NDCG@10 than the 278M `bge-reranker-base`. Param count isn't strength. The actual baseline-NDCG ordering on this corpus puts FlashRank in the middle.)

## Sweeps: pool size and number of LLM rewrites (n=200, weak reranker)

Two questions to probe: (a) does fusion's lift depend strongly on the candidate pool size before reranking? (b) does it depend on the number of LLM rewrites?

### Pool sweep (N=4 rewrites fixed)

| Candidate pool | Baseline NDCG@10 | Fusion NDCG@10 | Lift |
|---|---|---|---|
| 10 | 0.304 | 0.323 | +0.019 |
| 20 | 0.304 | 0.323 | +0.019 |
| 30 | 0.307 | 0.318 | +0.011 |
| 50 | 0.304 | 0.314 | +0.010 |
| 75 | 0.304 | 0.311 | +0.007 |

Fusion's lift is small and roughly flat-to-decreasing across pool sizes — it doesn't grow with bigger pools the way I'd previously believed. (An earlier version of this writeup, run at n=30, reported a sharp discontinuity at pool=50 with lift jumping from +0.020 to +0.050. That discontinuity does not survive at n=200; it was sample noise.)

### N-rewrites sweep (pool=50 fixed)

| Rewrites | NDCG@10 | Recall@10 | MRR |
|---|---|---|---|
| 0 (baseline) | 0.304 | 0.140 | 0.526 |
| 1 | 0.310 | 0.146 | 0.526 |
| 2 | 0.311 | 0.144 | 0.536 |
| 3 | 0.314 | 0.145 | 0.542 |
| 4 | 0.314 | 0.146 | 0.540 |

Lift grows monotonically but slowly with N, plateauing around N=3-4 at ~+0.010 NDCG@10. The N=1 → MRR-drops-below-baseline finding from the earlier n=30 version doesn't replicate — at n=200, N=1 MRR is identical to baseline (0.526 vs 0.526).

## Steelman: pipeline ordering (n=200, bge-reranker-large)

The paper's specific argument is about pipeline ordering: per-query retrieve → per-query rerank → fuse → truncate. The headline table compares both orderings; this section drills in on it because it's the load-bearing piece of the paper's mechanism.

For vector-only methods, MRR on rich queries shows a small but real negative effect for the paper's ordering: **−0.061 [−0.124, −0.000]** vs baseline. Per-query truncation drops a top-rank doc that the fused-then-reranked path keeps. On NDCG@10 the differences between orderings are inside noise (paper: −0.029 [−0.062, +0.004]; mine: −0.001 [−0.013, +0.010]).

So the paper's ordering does cause a small but real degradation on rich queries — they're right that it's a worse ordering. That's a valid practitioner warning, but it's not "fusion gains evaporate" — it's "if you put the reranker inside the fusion loop, you lose a small amount of MRR on easy queries."

The hybrid variants tell a more nuanced story. With BM25 in the mix:
- `hybrid_diverse+rerank` (fuse-then-rerank, hybrid retrieval) is the clear best on every bucket.
- `hybrid_per_query_rerank_then_fuse` (paper-style ordering, hybrid retrieval) is competitive only with the *weak* reranker (+0.026 ALL with bge-base, +0.000 with bge-large). The strong reranker absorbs the per-query rerank advantage.

**Combined read**: pipeline ordering matters, but it's a second-order effect compared to "did you include BM25 in the retrieval channels and apply enough LLM rewrites." Once you do those things, the fuse-then-rerank ordering is robustly better across reranker strengths.

## What changed across iterations (transparency)

This writeup went through four iterations of reversals and corrections on the way to the final picture. Each was caused by a specific methodological correction.

**Iteration 1 (n=30, vector-only fusion, bge-reranker-base/large):** Reported +0.041 NDCG@10 lift, sharp pool=50 discontinuity, N=1 MRR drop, paper pipeline losing by 0.033. Conclusion: "paper's headline doesn't reproduce."

**Iteration 2 (n=200 + paired-bootstrap CIs, vector-only fusion):** Most n=30 findings collapsed into noise. Conclusion: "fusion is statistically indistinguishable from baseline on average; the paper's headline is closer to right than I'd hoped."

**Iteration 3 (n=200 + adding the strong fusion variant `hybrid_diverse+rerank`):** The earlier "fusion is a wash" conclusion was specifically about the weak fusion variant. With BM25 in the retrieval channels alongside vector + LLM rewrites, fusion has a clearly significant lift across every bucket, with both rerankers, and produces measurably better answers in the LLM-judge eval (12% wins / 6% losses, mean score +0.10 over baseline). Conclusion: "the paper is right about the weak variant; wrong about the technique in general."

**Iteration 4 (n=200 + adding FlashRank as a third reranker):** The cross-rerankers test that should have been there from iteration 1. The paper uses FlashRank — which I'd previously been calling "weak" based on parameter count (33M). Turns out on NFCorpus, FlashRank's default model `ms-marco-MiniLM-L-12-v2` has higher baseline NDCG@10 (0.320) than `bge-reranker-base` (0.305) — param count isn't strength. Across all three rerankers, hybrid_diverse fusion's lift survives with CIs that exclude zero. The paper's specific configuration (FlashRank + per-query rerank + N=1) does still replicate "no significant lift" — their finding holds for what they tested.

| Claim | Iter 1 (n=30) | Iter 2 (n=200, vector-only) | Iter 3 (n=200, hybrid variant) | Iter 4 (n=200, +FlashRank) |
|---|---|---|---|---|
| Fusion's lift over baseline | +0.041 (sample noise) | +0.005 [−0.007, +0.019] | **+0.021 [+0.007, +0.036]** (bge-large) | Survives all three rerankers: bge-base **+0.023**, FlashRank **+0.014**, bge-large **+0.021** |
| Strong reranker absorbs fusion lift | "68% absorbed" | "no lift to absorb" | **No — lift unchanged from weak to strong** | **No — lift not monotonic in reranker strength across three conditions** |
| Paper's exact configuration replicates? | not tested | not tested | not tested | **Yes — FlashRank + per-query rerank + N=1: lift indistinguishable from zero** |
| LLM-judge wins vs baseline (rich) | not measured | 5% W / 11% L (net negative) | **13% W / 6% L (net +10)** | not yet measured for FlashRank |

The methodology lessons:

- **n=30 has dangerously high variance** on this benchmark; multiple "directionally clear" findings reversed. Default to n≥150 + paired-bootstrap CIs before treating any number as load-bearing.
- **The choice of fusion variant matters as much as the choice of reranker.** Comparing fusion's *weakest* variant (vector-only, single channel) against a vector baseline understates the technique. The fair comparison is fusion's strongest variant (`hybrid_diverse+rerank`) against the strongest non-fusion baseline (`hybrid+rerank`).
- **Reranker model parameter count ≠ reranker strength.** Test on a strength axis defined by baseline NDCG on the actual corpus, not by param count or library reputation. And specifically test the model the paper used, not just the rerankers convenient to your stack.
- **Aggregate retrieval metrics undercount the technique's actual production value.** The kohlrabi-class binary recoveries show up as small NDCG lifts but large LLM-judge mean-score lifts. End-to-end answer quality is a more sensitive metric for fusion's wins than retrieval ranking metrics.

Each iteration corrected a different methodological flaw: small-sample variance (1→2), variant-selection bias (2→3), reranker-selection bias (3→4). The end-state — properly-configured fusion produces measurably better retrieval and answers, across reranker conditions — is more defensible than any intermediate version because each was wrong in a way the next iteration explicitly corrected.

## Qualitative end-to-end eval (n=8, I read the outputs)

I ran each method on 8 queries (4 rich + 4 scarce, biased toward queries where retrievals diverged), generated answers from the top-5 retrievals using GPT, and read all 24 answers myself. Code in `eval/qualitative.py`, full output in `results/qualitative.txt`. The point of doing this directly rather than via an LLM judge was to check whether the retrieval-level metrics actually map to answer-level quality.

### Where fusion clearly won
- **PLAIN-1473 "kohlrabi" (scarce)**: baseline retrieved kombucha, kiwifruit, and a dog-disease paper — answered "no information about kohlrabi". Fusion put MED-4455 at rank 1 and produced the right answer about glucoraphanin/glucosinolate content. **This is fusion's value statement made concrete** — the answer changed from "I don't know" to a correct answer. Nothing else recovers this query.

### Where fusion clearly lost
- **PLAIN-1441 "Japan" (rich)**: baseline produced a coherent synthesis (Westernization, cancer trends, Alzheimer's). Fusion's top-5 included MED-3973 (gargling for oral hygiene) — topically scattered. The synthesis model **refused to answer** — output was "context does not pose a specific question about Japan." Same query, fewer-but-tighter retrievals → useful answer; with fusion's diversity → punt. This is the paper's mechanism in action: extra retrieval diversity → answer-model confusion → quality regresses below baseline.

### Where it didn't matter
- 5 of 8 queries produced near-equivalent answers across all three methods. Retrievals differ at ranks 3-5, but those don't materially change what the LLM synthesizes.

### Confound worth flagging
NFCorpus qrels have real labelling issues. PLAIN-2113 "soil health" gold = MED-4771 (folate metabolism in human liver). PLAIN-2375 "whiting" gold = sushi microbiology + kitchen disinfection. On these queries every method scores NDCG@10 = 0 because the labels are bad, not because retrieval failed. The "scarce bucket" is a mix of genuinely hard queries and qrel noise.

### Cost-adjusted picture (large reranker, n=200)

ΔNDCG@10 per extra LLM call (fusion costs +1 LLM call per query for the rewrite step):

| Bucket | n | ΔNDCG@10 [95% CI] per LLM call |
|---|---|---|
| All | 200 | +0.006 [−0.005, +0.020] |
| Rich | 139 | +0.001 [−0.010, +0.011] |
| **Scarce** | 61 | **+0.018 [+0.000, +0.053]** |

## End-to-end LLM-judge eval (n=200)

The retrieval-level metrics (NDCG@10 and friends) measure ranking quality, not answer quality. To test whether fusion's lift translates into actually-better answers, I ran a separate eval: for each of 200 queries, generate an answer from each method's top-5 retrievals using GPT, then ask an *independent* LLM to score each answer 0-3 against the gold doc text. Code in `eval/answer_eval.py`, results in `results/answer_eval_n200.json` and `results/answer_eval_hybrid_n200.json`.

This was run twice: once on the vector-only methods (the comparison the paper's pipeline is about), once on the hybrid variants (the technique's actual recommended configuration).

### Vector-only methods

| Method | Mean score (ALL) | Mean score (RICH) | Mean score (SCARCE) | W/T/L vs baseline (ALL) | Rich W/T/L | Scarce W/T/L |
|---|---|---|---|---|---|---|
| baseline+rerank | 1.10 | 1.19 | 0.90 | — | — | — |
| fuse_then_rerank | 1.08 | 1.08 | 1.08 | 11/174/15 (net **−4**) | 7/117/15 (net **−8**) | 4/57/0 (net **+4**) |
| paper_pipeline | 1.21 | 1.14 | 1.38 | 33/139/28 (net +5) | 20/92/27 (net **−7**) | **13/47/1 (net +12)** |

Two observations:

1. **Vector-only fusion is net-negative on rich queries at the answer level.** `fuse_then_rerank` gets 5% wins / 11% losses on rich queries — the synthesis-LLM step amplifies fusion's downside. NDCG@10 saw fusion as flat zero on rich; LLM-judge sees it as net-negative. The retrieval lift doesn't survive into answer quality on this variant.
2. **Recall-scarce wins are strongly asymmetric.** Fusion almost-never makes a hard query worse; paper pipeline gets 13 wins / 1 loss on scarce. Mean answer-score on scarce: paper 1.38 vs baseline 0.90 — **+53% relative**. The kohlrabi-class binary recovery shows up much more cleanly here than in the +0.018 NDCG number it produced.

### Hybrid variants

| Method | Mean score (ALL) | Mean score (RICH) | Mean score (SCARCE) | W/T/L vs baseline (ALL) | Rich W/T/L | Scarce W/T/L |
|---|---|---|---|---|---|---|
| baseline+rerank (vector) | 1.07 | 1.13 | 0.93 | — | — | — |
| hybrid+rerank (BM25 added) | 1.06 | 1.12 | 0.92 | 16/170/14 (net +2) | 11/119/11 (net 0) | 5/51/3 (net +2) |
| **hybrid_diverse+rerank** | **1.17** | **1.25** | **0.97** | **25/164/11 (net +14)** | **18/115/8 (net +10)** | 7/49/3 (net +4) |

Three findings here that change the story materially:

1. **`hybrid_diverse+rerank` produces measurably better answers across every bucket.** 12% wins / 6% losses overall — 2× more wins than losses. **Mean answer score on rich queries: 1.25 vs baseline's 1.13 (+11% relative).** This is the strongest pro-fusion finding in the entire experiment, and it's at the metric that actually matters for production.

2. **The Japan-style failure mode is fixed by adding BM25.** Vector-only fusion on rich queries had 5% wins / 11% losses (net −8). Hybrid+diverse on rich has 13% wins / 6% losses (**net +10**). Same LLM rewrites, same synthesizer, only difference is BM25 in the retrieval mix. BM25's exact-match anchoring eliminates the scattered-context → punt failure mode. The synthesizer fails when it can't ground its context lexically; BM25 guarantees lexical anchoring.

3. **BM25 alone doesn't do it; the rewrites are doing real work too.** `hybrid+rerank` (BM25 + vector, no rewrites) is essentially identical to baseline at the answer level (1.06 vs 1.07). The lift only shows up when BM25 *and* LLM rewrites are combined. Both channels are necessary; neither is sufficient. This contradicts the worry from earlier in the experiment that BM25 was doing all the lifting and rewrites were marginal.

## My synthesised judgement

After running everything end-to-end (retrieval metrics with bootstrap CIs, LLM-judge answer-quality eval, both rerankers, both vector and hybrid variants, n=200), I think the honest position is:

> **Properly-configured RAG-Fusion (`hybrid_diverse+rerank`) produces measurably better retrieval rankings *and* better generated answers than baseline retrieval, on every metric, on every difficulty bucket, at proper sample sizes with confidence intervals. The paper's "fusion gains evaporate" claim is a finding about a specific weak fusion variant (single rewrite, vector-only, per-query truncation) — not about the technique in its strong form.**

What survives the full evaluation:
- **NDCG@10 lift +0.021 [+0.007, +0.036]** with statistically significant CI on every bucket.
- **LLM-judge mean score lift +0.10** (1.17 vs 1.07), with **net +14 wins** at the answer level (12% wins, 6% losses).
- **Lift survives across all three rerankers** — bge-base: +0.023; FlashRank/ms-marco-MiniLM-L-12-v2: +0.014; bge-large: +0.021. Not monotonic in reranker strength; CIs exclude zero in all three cases.
- **Both lexical and semantic channels are needed.** BM25 alone or LLM rewrites alone don't work; combining them does.

What the paper got right:
- **Vector-only fusion (their tested variant) is roughly a wash and slightly negative on rich queries at the answer level.** Their critique of *that variant* lands.
- **Their exact configuration replicates on our stack.** Strict replication run with FlashRank + per-query rerank + N=1 LLM rewrite, n=200: lift over baseline is **−0.011 NDCG@10 [−0.032, +0.010]** on the all-bucket (CI crosses zero, point estimate slightly negative); **−0.028 [−0.059, +0.003]** on rich queries (slight regression, CI just barely crosses zero); **+0.024 [+0.008, +0.043]** on the recall-scarce tail (clearly positive). This matches their reported pattern almost exactly. The paper's finding is real for what they tested.
- **The number of rewrites does real work.** At N=1, even `hybrid_diverse+rerank` doesn't significantly beat baseline (+0.006 [−0.006, +0.021]). At N=4, it does (+0.014 [+0.001, +0.030] with FlashRank). The paper's choice of N=1 is part of why their setup doesn't see a lift — and our N=4 finding only holds because we run more rewrites.
- **Pipeline ordering matters.** The fuse-then-rerank ordering is robustly better than per-query rerank then fuse, especially on rich queries. They identified a real practitioner pitfall.
- **Adaptive routing is still smart.** Even with the strong variant winning on average, fusion has a small loss tail (6% of queries on rich, 5% on scarce). Routing fusion only to detected hard queries captures the wins and avoids the losses — that pattern is empirically motivated by the W/L asymmetry.

The deployment shape that survives both the original technique and this critique:
1. **Always-on `hybrid_diverse+rerank` is defensible** on workloads where the recall-scarce share is meaningful (specialist literature, biomedical, patent, legal).
2. **Adaptive routing is more efficient.** Run baseline+rerank, gate fusion on a cheap weakness signal (cross-encoder top-1 score below threshold is the obvious candidate), fire fusion only when that signal trips. Captures most of the wins, eliminates most of the losses, and pays the +1 LLM call cost on a fraction of traffic.
3. **Don't deploy vector-only fusion.** It's the variant the paper successfully critiques. If you're using fusion, use the hybrid variant.

## Where RAG-Fusion fits — and where it doesn't

The evidence above lets me describe the deployment surface concretely rather than abstractly. Three things have to be true for fusion to earn its compute:

1. **Terminology mismatch between query and corpus.** The user's words and the indexed words don't fully overlap. Multi-query rewriting probes the gap. (NFCorpus "kohlrabi" → corpus "glucoraphanin in cruciferous vegetable seeds" is the canonical case.)
2. **Recall matters more than precision.** Missing a relevant document is more costly than including a marginal one. The wide-net behaviour is the feature.
3. **The downstream consumer can handle breadth.** Either a strong synthesis LLM that doesn't get confused by topically-mixed context, or a UI that surfaces multiple candidates rather than one canonical answer.

When all three hold, fusion produces measurably better answers (LLM-judge: 12% wins / 6% losses on this corpus, +0.10 mean score over baseline). When any of them fail, the lift shrinks toward zero. Note that a critical fourth precondition, often skipped: **use the hybrid variant** (BM25 + vector × LLM rewrites). Vector-only fusion is roughly a wash on average and net-negative on rich queries at the answer level — that's the variant the paper successfully critiques.

### Use cases where fusion is a strong fit

- **Specialist / scientific literature search.** Biomedical, materials science, chemistry, life-sciences corpora. Lay-vs-technical terminology gaps are the whole problem (kohlrabi → glucoraphanin; "heart attack" → "myocardial infarction"; "vitamin C" → "ascorbic acid"). NFCorpus is a clean instance — fusion's value here is empirically demonstrated above.
- **Patent prior-art search.** Synonymy and concept paraphrase are explicit goals; recall is legally significant; precision is somebody else's problem. The cost of one extra LLM call is rounding error against the cost of missing prior art.
- **Legal e-discovery and regulatory review.** Find every document that could be responsive. False negatives are the catastrophic failure mode; fusion's diversity is exactly what the workflow wants.
- **Cross-corpus or cold-start research workflows.** Retrieval over a corpus the embedding model hasn't seen during training (specialist internal data, niche domains). Multi-query helps the retriever cover semantic gaps the embedding can't bridge alone.
- **Exploratory / "show me what's out there" UX.** Where the user expects a list to skim, not a single synthesised answer. Academic literature browsing, market research, competitive analysis. The "Japan" failure mode (LLM punts on diverse context) doesn't apply because there's no synthesis step downstream.
- **Long-tail e-commerce and product catalogues.** Queries like "thing that holds my phone in the car" where product titles use entirely different vocabulary. Multi-query catches "magnetic phone mount", "car phone holder", "vent mount" — descriptions a single embedding might miss.

### Use cases where fusion is a poor fit

- **FAQ chatbots and structured customer support.** Most queries match a known canonical entry. Even with the hybrid variant, the recall-scarce share is small by design, so the absolute lift on the rich majority is modest in absolute terms — the cost-benefit case for always-on fusion is unfavourable. Use adaptive routing on detected hard queries instead.
- **Latency-critical retrieval (voice, autocomplete, real-time chat).** The query-rewrite call alone breaks a sub-500ms budget. No amount of retrieval quality compensates if the user has already given up.
- **High-volume / margin-thin deployments.** Per-query cost matters at scale. Fire fusion at 100% of traffic and you've quadrupled the LLM bill on retrieval; the ~30% scarce-tail share where it pays for itself doesn't justify spending on the 70% it doesn't.
- **Pipelines with weak or small synthesis models, when running vector-only fusion.** The Japan failure mode (synthesis LLM refuses to answer when given topically-scattered vector-only fusion context) was real in our eval — but it disappeared once we added BM25 to the retrieval mix. If you must use vector-only fusion AND your synthesizer is small/weak, you'll see this failure mode. Switching to the hybrid variant fixes it on stronger synthesizers; behaviour on smaller models hasn't been tested.
- **Well-tagged / well-curated knowledge bases.** Internal company wikis with strong taxonomies, product docs with consistent terminology — the recall-scarce tail is small, the per-query value is small, the cost-benefit is unfavourable.

### The default deployment pattern

For mixed workloads — which is most production retrieval — the right shape isn't "fusion on" or "fusion off" but **adaptive routing**:

1. Run baseline+rerank on every query.
2. Compute a cheap signal that the result is weak (low cross-encoder scores at the top of the rerank, low embedding-similarity floor, or a lightweight learned classifier).
3. Fire fusion only when that signal trips.

This is the most defensible read of all the evidence above: capture the kohlrabi-class wins (where fusion is the only mechanism that recovers anything), eliminate the Japan-class regressions (where diversity hurts), and pay for fusion's compute only on the share of traffic where it pays for itself. It also reframes the production question correctly — not *"should I use RAG-Fusion?"* but *"what fraction of my traffic is recall-scarce, and how do I detect it cheaply?"*

### Production validation

The hybrid retrieval + cross-encoder reranking stack we recommend here isn't theoretical. It's what currently ships in **Scopus AI** and **LeapSpace** — both with slight variations on the configuration tested above (different rerankers, domain-tuned rewriter prompts, application-specific candidate-pool sizing). Both are scientific-literature retrieval workloads with the kind of terminology-mismatch tail that NFCorpus is designed to expose. The recommendation that ends this writeup matches what real deployment converged on independently — which is at least weak triangulation that the variant choices defended above are pointing at the right operating point for this class of workload.

## Operational considerations: cost, latency, corpus size, data type

A caveat I should make louder than I have so far: **the entire evaluation above is on academic biomedical literature** (NFCorpus). That corpus is close to a worst-case for query-document vocabulary alignment — user types "kohlrabi", document is indexed under "glucoraphanin in cruciferous vegetable seeds". Many of fusion's hero cases in the qualitative read are downstream effects of that. On corpora with tighter alignment (curated FAQs, product catalogues, code), the recall-scarce tail will be smaller and the cost-benefit unfavourable in correspondingly more cases. The numbers in this writeup describe fusion's mechanism on what is roughly its best-case data type. They do not directly transfer to other shapes of deployment.

With that caveat foregrounded, the four operational dimensions:

### Cost

Per query, fusion adds roughly one extra LLM call (the rewrite step, four queries in one call) plus four extra vector searches. At typical 2026 mid-tier API prices and a ~50-input/400-output-token rewrite prompt, the marginal LLM cost per query is in the **~$0.003–0.008** range. Vector searches are negligible (microseconds at this corpus size). Cross-encoder rerank is unchanged at fixed candidate-pool size.

At 1M queries/month, that's ~$3K–8K/month over baseline — small in absolute terms but only justified if it's bought concentrated on the queries it helps. Adaptive routing at a ~30% trigger rate cuts that to ~$1K–2.5K/month with no loss of the kohlrabi-class wins. The cost case for default-on fusion is weakest at low per-query margins (consumer-scale chatbots, search ads) and easiest at high-stakes individual queries (legal e-discovery, patent prior-art) where one missed document can cost more than a year of fusion compute.

### Latency

Fusion's overhead is **structurally serial**: the rewrite LLM call has to finish before any retrieval starts, and it can't be streamed or parallelised away. Typical numbers:

| stage | baseline+rerank | fuse_then_rerank |
|---|---|---|
| query rewrite (LLM) | — | 600–1500 ms |
| vector search (parallel-able) | 20–100 ms | 50–200 ms |
| cross-encoder rerank | 200–1000 ms | 200–1000 ms |
| **typical p50 total** | ~250–1100 ms | ~850–2700 ms |
| **p99 tail** | ~1500 ms | ~3500 ms+ |

This makes fusion **disqualifying** for voice assistants, autocomplete, and chat experiences with sub-second p95 targets. It's neutral-to-fine for analytical workflows, research workbenches, and async batch retrieval where answer quality dominates over response speed. Adaptive routing helps here too — only the small fraction of queries that route to fusion pay the rewrite latency, and that fraction is by definition the queries where the user has a higher tolerance for "let me think about this one."

### Corpus size

| corpus size | what fusion does here |
|---|---|
| **<10k docs** | Embedding + rerank likely already saturate recall on most queries. Fusion's diversity has nowhere to live. Skip. |
| **10k–1M (NFCorpus is here)** | Where we tested. With the hybrid variant: NDCG@10 lift +0.021 [+0.007, +0.036] over vector baseline, statistically significant on every bucket. LLM-judge mean score +0.10 over baseline (12% wins / 6% losses). The technique earns its keep here. |
| **1M–100M** | Recall-scarce tail probably grows (more terminology variation). Fusion's potential value grows, but cross-encoder cost grows too, and the candidate pool has to scale proportionally to absorb fusion's added diversity. Worth re-running our pool-size sweep at this scale before deploying. |
| **web-scale (100M+)** | First-stage retrieval is the bottleneck. Fusion's 5× retrieval cost competes directly with the latency/budget that goes into making the *base* retrieval excellent. Most web-scale stacks use cheaper query expansion (PRF, learned sparse representations) for similar effect at much lower per-query cost. |

### Data type

This is where the NFCorpus bias matters most. The taxonomy below is graded by how well the result transfers from our evidence:

- **Academic / scientific literature (what we tested).** Sparse vocabulary overlap, recall-dominated. Fusion's strongest case. Our evidence is directly applicable here. Patent prior-art and biomedical search inherit most of the same dynamics.
- **Legal e-discovery and regulatory review.** Same recall-over-precision profile. Recall-scarce tail likely larger than NFCorpus. Adaptive routing matters less because the cost of a single missed document can dominate fusion's monthly bill — default-on may be the right call.
- **Long-tail e-commerce / product search.** User vocabulary diverges sharply from product titles ("phone holder thing for car" vs "magnetic vent mount"). Fusion fits, but latency budgets are tighter than literature search — adaptive routing on detected long-tail queries is the safer pattern.
- **News / current events.** Mixed signals. Embedding models often haven't seen recently-emerged entities (fusion helps via paraphrase); latency budgets are tight; high query volume. Adaptive routing earns its keep here especially.
- **Customer support / FAQ.** Curated, well-tagged, head-heavy distribution. The recall-scarce tail is small by design. The steelman wins here decisively — fusion is wasted spend on >90% of queries. Skip or use only as a fallback when baseline confidence is below threshold.
- **Code / identifier search.** Identifiers are exact tokens with little synonymy at the symbol level. Rewriting "getUserById" produces semantic neighbours, not lexical ones — and the index is keyed on the lexical form. Fusion adds noise; precision-first techniques (BM25 on tokenised identifiers, AST-aware retrievers) work better.
- **Structured data, knowledge graphs, SQL-backed retrieval.** Fusion is the wrong tool entirely — you want a grounded query-to-structured translation, not paraphrase-based recall.

### How this changes the deployment decision

The synthesised judgement (fusion as a precision tool for the recall-scarce tail) is **most defensible in the regime we tested**: medium-corpus, recall-dominated, terminology-rich literature search, with no hard latency constraint. As you move away from that profile in *any* of the four dimensions — smaller corpora, tighter latency budgets, higher cost sensitivity, more curated/structured data — the case for default-on fusion gets worse and the case for either adaptive routing or skipping fusion entirely gets stronger.

Practical heuristic for picking a deployment mode:

1. **Skip fusion** if the corpus is tiny, the data is structured/curated, or you have a sub-second p95 budget on every query.
2. **Default-on fusion** if the cost of a missed document genuinely outweighs ~$0.005/query (legal, regulatory, biomedical research). The kohlrabi-style wins are unique to fusion in this regime.
3. **Adaptive routing (fusion gated by a weakness signal)** for everything in between — which is most production retrieval. Captures the long-tail wins, kills the rich-bucket regression cases, and pays for fusion's compute only on traffic where it earns it.

## How to reproduce

```bash
# Pool + N sweeps at n=200 (weak reranker)
python -m eval.sweep --sample 200 \
  --pool-values 10 20 30 50 75 --n-values 1 2 3 4 \
  --out experiments/arxiv-2603-02153-replication/results/sweep_n200.json

# 6-method steelman with each reranker at n=200
python -m eval.steelman --sample 200 --rerank-model BAAI/bge-reranker-base \
  --out experiments/arxiv-2603-02153-replication/results/steelman_base_n200_full.json
python -m eval.steelman --sample 200 --rerank-model BAAI/bge-reranker-large \
  --out experiments/arxiv-2603-02153-replication/results/steelman_large_n200_full.json

# Bootstrap CIs on the steelman per-query metrics
python -m eval.bootstrap_ci experiments/arxiv-2603-02153-replication/results/steelman_large_n200_full.json

# LLM-judge end-to-end answer eval (n=200, ~75 min)
python -m eval.answer_eval \
  experiments/arxiv-2603-02153-replication/results/steelman_large_n200_full.json \
  --top-k 5 --methods baseline+rerank hybrid+rerank hybrid_diverse+rerank \
  --out experiments/arxiv-2603-02153-replication/results/answer_eval_hybrid_n200.json

# Saved-queries metric (kohlrabi-class binary recovery rate)
python -m eval.saved_queries experiments/arxiv-2603-02153-replication/results/steelman_large_n200_full.json
```

Total wall time on a 2025-era Mac: ~5-6 hours for the full set, dominated by `bge-reranker-large` on CPU. The disk-persisted query-rewrite cache (`./query_cache.json`, populated on first run) means subsequent runs don't re-pay LLM costs.

## What I'd want to do next

1. **Build the adaptive-routing variant.** Even with hybrid_diverse+rerank winning on average, there's still a small loss tail (6% of queries). Gate fusion on a cheap baseline-weakness signal (cross-encoder top-1 score below threshold is the obvious candidate) and only fire the rewrite step on detected hard queries. Quantify what fraction of fusion's wins is captured at what fraction of the always-on cost. Curve will probably show "30% routing rate captures 80%+ of wins at 30% of LLM cost" — that's the chart that settles deployability for the marginal-cost case.
2. **Re-run on a different domain.** NFCorpus is biomedical literature — a workload that exercises fusion's terminology-bridging strength. MS MARCO Passage and a real enterprise FAQ (well-curated, recall-rich) would bracket the picture from the other side. The recall-scarce share is workload-dependent; the deployment recommendation should be too.
3. **Test hybrid_diverse at varying corpus scales.** Our pool-size sweep capped at 75 due to runtime. With a larger corpus and pool=200+, does the hybrid lift grow further or saturate?
4. **Confirm the recall-scarce lift isn't an artefact of NFCorpus's labelling sparsity.** Several "scarce" queries have qrel labels that look incorrect (e.g. "soil health" gold = a folate-metabolism paper). Cleaner gold standards would tighten the scarce-bucket CI.
5. **Test against a smaller synthesizer.** Our LLM-judge eval used GPT-5.1 chat-latest as the synthesizer. The Japan-style regression on vector-only fusion was eliminated by adding BM25, but a smaller synthesizer (7B class) might still struggle even with hybrid context. Worth measuring before deploying on weaker generators.

If you've replicated this differently and got a different answer, I'd genuinely like to see it.
