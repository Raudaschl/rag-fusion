# A small replication of arXiv 2603.02153v1 on NFCorpus

**TL;DR.** The paper argues that retrieval-fusion gains "evaporate" after reranking and truncation. Replicating on NFCorpus at n=200 with paired-bootstrap 95% CIs, the honest picture is: **fusion's average lift over a baseline+rerank pipeline is statistically indistinguishable from zero** (NDCG@10 lift +0.006 [−0.005, +0.020] with `bge-reranker-large`). The one place fusion has a real positive effect is the **recall-scarce tail** — queries where the baseline finds no relevant docs — where fusion adds a small but statistically significant lift (+0.018 [+0.000, +0.053] NDCG@10). On the recall-rich majority (~70% of NFCorpus queries), fusion is a wash. Reading the actual generated answers, the recall-scarce wins are real and qualitatively meaningful (canonical case: query "kohlrabi" → corpus indexed under "glucoraphanin in cruciferous vegetable seeds" — fusion finds it; baseline doesn't), but they're rarer than smaller-sample experiments suggest. **Honest read: the paper's headline is closer to right than I'd hoped.** Fusion's mechanism is real but its average production benefit is small. The deployment pattern that survives both the original technique and this critique is **adaptive routing** — fire fusion only when a cheap weakness signal trips on baseline retrieval, capturing the recall-scarce tail without paying for everything else. A note up top: an earlier version of this writeup leaned on n=30 numbers that didn't survive at n=200; I've updated the substantive claims and left a "what changed at n=200" section below for transparency.

## Disclosure first

I wrote the original RAG-Fusion article in 2024, so I have an obvious stake in the conclusion and you should weigh the framing accordingly. I tried to design the experiment to give the paper's claim a fair shot — same fusion-then-rerank ordering, a credible cross-encoder, a real BEIR benchmark — but selection effects creep in even when I'm trying to be careful. If you spot one, please open an issue.

## What the paper claims (very briefly)

arXiv 2603.02153v1 reports that on a 115-query synthetic enterprise-support set, two-query fusion (Q1 + one LLM rewrite) followed by FlashRank reranking and Top-K truncation either matches or slightly underperforms a single-query baseline. Their proposed mechanism: the reranker re-anchors on Q1, and the rewrite's marginal candidates are squeezed out at truncation. They do note (§6.6) that fusion still helps in "recall-scarce" query regimes.

## What I changed and why

| Choice | Mine | Theirs |
|---|---|---|
| Dataset | NFCorpus (biomedical, BEIR) | 115 synthetic RAGAS support queries |
| Sample size | 200 queries (paired-bootstrap CIs) | 115 queries |
| Reranker | `BAAI/bge-reranker-base` and `-large` | FlashRank cross-encoder |
| Rewrites (default) | 4 | 1 |
| Pipeline order | Both orderings tested head-to-head | Retrieve → rerank per-query → fuse → truncate |
| Candidate pool before rerank | 50 (default) | ~10 |

These are not small changes, and I'm not claiming a like-for-like replication. NFCorpus is harder than synthetic support queries — it's exactly the "recall-scarce regime" the paper says fusion still helps in. So part of the divergence may just be that I'm testing a regime they already conceded.

## Headline numbers (n=200 NFCorpus sample, 95% paired-bootstrap CIs)

With `bge-reranker-large` (the more capable reranker, where the steelman lands hardest):

| Method | NDCG@10 mean [95% CI] | Lift over baseline [95% CI] |
|---|---|---|
| baseline+rerank | 0.332 [0.289, 0.376] | — |
| fuse_then_rerank | 0.338 [0.295, 0.382] | +0.006 [−0.005, +0.020] |
| rerank_per_query_then_fuse (paper's order) | 0.326 [0.285, 0.369] | −0.006 [−0.032, +0.021] |

Both fusion variants' lifts over baseline cross zero. The simple read: **on average, with a strong reranker on this corpus, fusion neither helps nor hurts measurably.**

Stratified by query difficulty (queries where baseline+rerank's top-10 contained ≥1 relevant doc = "rich"; otherwise "scarce"):

| Bucket | n | fuse_then_rerank lift [95% CI] | paper_pipeline lift [95% CI] |
|---|---|---|---|
| Rich  | 139 | +0.001 [−0.010, +0.011] | −0.030 [−0.064, +0.004] |
| **Scarce** | 61 | **+0.018 [+0.000, +0.053]** | **+0.050 [+0.019, +0.088]** |

This is the crux: fusion has a real, statistically distinguishable positive effect **only on the recall-scarce tail**. On rich queries it's flat. The paper's pipeline ordering helps even more on scarce (+0.050) but trends slightly negative on rich.

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

## Steelman: probing the paper's three load-bearing claims (n=200, paired-bootstrap CIs)

Three tests, run with `bge-reranker-base` and `-large`. Code in `eval/steelman.py`, CIs via `eval/bootstrap_ci.py`, raw outputs in `results/steelman_*_n200.json`.

### Test 1 — pipeline ordering (with bge-reranker-large)

The paper's ordering: per-query retrieve → per-query rerank → fuse → truncate. Mine: per-query retrieve → fuse → rerank-fused-pool → truncate. Both share the same rerank compute budget.

| Method | NDCG@10 [95% CI] | Lift over baseline [95% CI] |
|---|---|---|
| baseline+rerank | 0.332 [0.289, 0.376] | — |
| fuse_then_rerank (mine) | 0.338 [0.295, 0.382] | +0.006 [−0.005, +0.020] |
| rerank_per_query_then_fuse (paper) | 0.326 [0.285, 0.369] | −0.006 [−0.032, +0.021] |

Both lifts cross zero on NDCG@10. On *MRR* the paper's ordering does score a small significant negative on rich queries (−0.061 [−0.124, −0.000]) — slightly worse than baseline because per-query truncation drops a relevant doc that the fused pool would have kept. But on NDCG@10 the differences between orderings are within noise.

### Test 2 — truncation depth (NDCG@K, large reranker)

| Method | K=1 | K=3 | K=5 | K=10 |
|---|---|---|---|---|
| baseline+rerank | 0.529 | 0.451 | 0.405 | 0.332 |
| fuse_then_rerank | 0.516 | 0.444 | 0.405 | 0.338 |
| paper_pipeline | 0.493 | 0.434 | 0.395 | 0.326 |

Fusion's lift is roughly constant (≤ +0.006 NDCG) across K. The paper's truncation mechanism would predict shrinking lift at smaller K — that doesn't show up here, but neither does my earlier "harsh truncation doesn't absorb fusion" claim, since there's no real lift to absorb in either direction.

### Test 3 — difficulty stratification

Bucketed by whether baseline+rerank's top-10 contains ≥1 relevant doc. With bge-reranker-large: n_rich=139, n_scarce=61 — so ~30% of NFCorpus queries are recall-scarce on this stack. With weak reranker the buckets shift slightly (n_rich=136, n_scarce=64) — the proportion is stable.

NDCG@10 lift over baseline+rerank, with both rerankers:

| Bucket | weak (`bge-reranker-base`) | strong (`bge-reranker-large`) |
|---|---|---|
| Rich  | +0.003 [−0.007, +0.013] | +0.001 [−0.010, +0.011] |
| **Scarce** | **+0.021 [+0.001, +0.057]** | **+0.018 [+0.000, +0.053]** |

This is the single durable finding from the steelman: **fusion's positive effect is concentrated entirely on the recall-scarce tail**, with both rerankers, with statistically distinguishable lift (the lower bound just grazes zero, but the directionality is real). On the rich majority — where the baseline already finds something — fusion's lift is indistinguishable from zero. The "fusion goes net-negative on rich queries with a strong reranker" claim from the earlier n=30 writeup (−0.031) does *not* replicate; at n=200 it's flat zero.

### What this means combined

With proper sample sizes and CIs, the paper's three load-bearing arguments resolve as:

1. **Pipeline ordering matters slightly** but mostly in MRR on rich queries (paper's ordering drops a top-rank doc the fused pool would have kept). On NDCG@10 the difference is inside noise.
2. **Truncation depth doesn't absorb fusion gains** — but it doesn't need to, because there isn't much fusion lift on the rich bucket to absorb in the first place.
3. **Difficulty regime is the real story.** Fusion has a small statistically-significant positive effect on the recall-scarce ~30% of queries, and is essentially neutral on the rich 70%.

## What changed at n=200 (and why I'm flagging it)

An earlier version of this writeup ran the steelman at n=30 and reported several findings that didn't survive a 200-query bootstrap. For transparency:

| Earlier claim (n=30) | Status at n=200 |
|---|---|
| "Fusion lift is +0.041 NDCG@10 over baseline+rerank with weak reranker" | False. Real lift is +0.009 [−0.001, +0.023]. |
| "Stronger reranker absorbs 68% of fusion's lift" | False. Both lifts are inside noise; absorption claim doesn't hold. |
| "Fusion goes net-negative on the rich bucket with strong reranker (−0.031 NDCG@10)" | False. Real lift is +0.001 [−0.010, +0.011]. |
| "Paper's pipeline loses by 0.033 NDCG@10 on all queries" | Partly false. NDCG@10 difference is within noise. *MRR* on rich does take a small but real hit. |
| "Sharp pool-size discontinuity at pool=50" | False. Pool sweep is roughly flat at n=200. |
| "N=1 fusion drops MRR below baseline" | False. N=1 MRR is identical to baseline. |
| **"Fusion's value lives in the recall-scarce tail"** | **True at n=200** — only finding that survives statistical scrutiny. |
| **"Kohlrabi-style hero cases are real and irreplaceable"** | **True** (qualitative read; aggregate metrics undercount this) |

The n=30 numbers were doing too much work for fusion's case. They weren't fabricated — they're what the run produced — but small samples on this benchmark have surprisingly large variance, and several "findings" were artefacts. The corrected story is more chastened and, I think, more useful: most of fusion's apparent benefit at small samples was sample noise; the durable benefit is concentrated in a specific minority of queries.

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

On the rich and all-bucket axes, fusion's per-call lift is inside noise — paying $0.005/query for a return statistically indistinguishable from zero. The scarce-bucket return is real but small. The qualitative regression cases (PLAIN-1441 Japan, where fusion's diversity made the LLM punt) are real but rare enough not to aggregate into a measurable NDCG hit at this sample size.

## My synthesised judgement

After running everything end-to-end and reading the actual outputs, I think the honest position is:

> **RAG-Fusion is a precision tool for the recall-scarce tail of a workload. On the rest, it is approximately a wash. It is not a default-on quality boost for the production-typical case.**

What survives statistical scrutiny at n=200:
- **Real lift on the recall-scarce tail.** With both rerankers, fusion provides a small, statistically distinguishable positive effect on queries where the baseline finds zero relevant docs (~30% of NFCorpus). The "kohlrabi" case from the qualitative read is the type specimen — it's the difference between "answer" and "I don't know," and aggregate NDCG numbers undercount that.
- **No systematic regression on the rich majority.** Fusion is statistically indistinguishable from baseline on rich queries, with both rerankers. The earlier "fusion goes net-negative on easy queries" framing was a small-sample artefact and doesn't hold at n=200.

What the steelman successfully forces me to concede:
- **The average lift is small.** With proper sample sizes, fusion's lift over baseline+rerank on this corpus is `+0.006` to `+0.009` NDCG@10, with CIs that cross zero. The headline numbers from the n=30 version overstated this by 4-7×.
- **Fusion ≠ free quality.** On the rich majority — most queries on most production corpora — the LLM-rewrite cost buys close to nothing. The only durable value is on the recall-scarce tail.
- **Whether to deploy depends on workload composition.** If your recall-scarce share is small, the cost-benefit is unfavourable.
- **Qualitative regressions still happen, even if they don't show in aggregate.** The "Japan" case in the answer eval (where fusion's diversity caused the synthesis LLM to refuse to answer) is real and has the same flavour as the paper's mechanism, even though it doesn't aggregate into a measurable NDCG hit at n=200.

The deployment recommendation that survives both the original article and this critique: **adaptive routing** — fire fusion only on detected hard queries (gated by a cheap weakness signal or learned classifier), not by default. That captures the kohlrabi-style wins, sidesteps the rare Japan-style regressions, and avoids paying for fusion's compute on the ~70% of queries where it doesn't help.

## Where RAG-Fusion fits — and where it doesn't

The evidence above lets me describe the deployment surface concretely rather than abstractly. Three things have to be true for fusion to earn its compute:

1. **Terminology mismatch between query and corpus.** The user's words and the indexed words don't fully overlap. Multi-query rewriting probes the gap. (NFCorpus "kohlrabi" → corpus "glucoraphanin in cruciferous vegetable seeds" is the canonical case.)
2. **Recall matters more than precision.** Missing a relevant document is more costly than including a marginal one. The wide-net behaviour is the feature.
3. **The downstream consumer can handle breadth.** Either a strong synthesis LLM that doesn't get confused by topically-mixed context, or a UI that surfaces multiple candidates rather than one canonical answer.

When all three hold, fusion is a precision tool for hard queries. When any of them fail, it's expensive and occasionally regressive.

### Use cases where fusion is a strong fit

- **Specialist / scientific literature search.** Biomedical, materials science, chemistry, life-sciences corpora. Lay-vs-technical terminology gaps are the whole problem (kohlrabi → glucoraphanin; "heart attack" → "myocardial infarction"; "vitamin C" → "ascorbic acid"). NFCorpus is a clean instance — fusion's value here is empirically demonstrated above.
- **Patent prior-art search.** Synonymy and concept paraphrase are explicit goals; recall is legally significant; precision is somebody else's problem. The cost of one extra LLM call is rounding error against the cost of missing prior art.
- **Legal e-discovery and regulatory review.** Find every document that could be responsive. False negatives are the catastrophic failure mode; fusion's diversity is exactly what the workflow wants.
- **Cross-corpus or cold-start research workflows.** Retrieval over a corpus the embedding model hasn't seen during training (specialist internal data, niche domains). Multi-query helps the retriever cover semantic gaps the embedding can't bridge alone.
- **Exploratory / "show me what's out there" UX.** Where the user expects a list to skim, not a single synthesised answer. Academic literature browsing, market research, competitive analysis. The "Japan" failure mode (LLM punts on diverse context) doesn't apply because there's no synthesis step downstream.
- **Long-tail e-commerce and product catalogues.** Queries like "thing that holds my phone in the car" where product titles use entirely different vocabulary. Multi-query catches "magnetic phone mount", "car phone holder", "vent mount" — descriptions a single embedding might miss.

### Use cases where fusion is a poor fit

- **FAQ chatbots and structured customer support.** Most queries match a known canonical entry. Adding 4× LLM cost per query for a lift that's statistically indistinguishable from zero is the production-economics objection from the steelman, made literal.
- **Latency-critical retrieval (voice, autocomplete, real-time chat).** The query-rewrite call alone breaks a sub-500ms budget. No amount of retrieval quality compensates if the user has already given up.
- **High-volume / margin-thin deployments.** Per-query cost matters at scale. Fire fusion at 100% of traffic and you've quadrupled the LLM bill on retrieval; the ~30% scarce-tail share where it pays for itself doesn't justify spending on the 70% it doesn't.
- **Pipelines with weak or small synthesis models.** The Japan failure mode is real: a smaller LLM struggles more with topically-scattered context, so fusion's diversity hurts more. If your generator is a 7B model, the answer-quality regression on the rich bucket is likely worse than what we measured here.
- **Well-tagged / well-curated knowledge bases.** Internal company wikis with strong taxonomies, product docs with consistent terminology — the recall-scarce tail is small, the per-query value is small, the cost-benefit is unfavourable.

### The default deployment pattern

For mixed workloads — which is most production retrieval — the right shape isn't "fusion on" or "fusion off" but **adaptive routing**:

1. Run baseline+rerank on every query.
2. Compute a cheap signal that the result is weak (low cross-encoder scores at the top of the rerank, low embedding-similarity floor, or a lightweight learned classifier).
3. Fire fusion only when that signal trips.

This is the most defensible read of all the evidence above: capture the kohlrabi-class wins (where fusion is the only mechanism that recovers anything), eliminate the Japan-class regressions (where diversity hurts), and pay for fusion's compute only on the share of traffic where it pays for itself. It also reframes the production question correctly — not *"should I use RAG-Fusion?"* but *"what fraction of my traffic is recall-scarce, and how do I detect it cheaply?"*

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
| **10k–1M (NFCorpus is here)** | Where we tested. Average NDCG@10 lift is small and statistically indistinguishable from zero (~+0.006 to +0.009). Real lift exists on recall-scarce queries (~+0.018 to +0.021 NDCG@10), and qualitatively-significant hero cases recover queries the baseline misses entirely. |
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
# Pool + N sweeps at n=200
python -m eval.sweep --sample 200 \
  --pool-values 10 20 30 50 75 --n-values 1 2 3 4 \
  --out experiments/arxiv-2603-02153-replication/results/sweep_n200.json

# Steelman with each reranker at n=200
python -m eval.steelman --sample 200 --rerank-model BAAI/bge-reranker-base \
  --out experiments/arxiv-2603-02153-replication/results/steelman_base_n200.json
python -m eval.steelman --sample 200 --rerank-model BAAI/bge-reranker-large \
  --out experiments/arxiv-2603-02153-replication/results/steelman_large_n200.json

# Bootstrap CIs on the steelman per-query metrics
python -m eval.bootstrap_ci experiments/arxiv-2603-02153-replication/results/steelman_large_n200.json
```

Total wall time on a 2025-era Mac: ~2-3 hours, dominated by `bge-reranker-large` on CPU. The disk-persisted query-rewrite cache (`./query_cache.json`, populated on first run) means subsequent runs don't re-pay LLM costs.

## What I'd want to do next

1. **Build the adaptive-routing variant.** That's the deployment shape the data points at: gate fusion on a cheap baseline-weakness signal (low cross-encoder top-1 score is the obvious candidate) and only fire the rewrite step on detected hard queries. Quantify how much of fusion's hero-case value is captured at what fraction of the always-on cost.
2. **Re-run on a different domain.** NFCorpus is biomedical literature with sparse labels — a workload that exercises fusion's terminology-bridging strength and contains a chunky recall-scarce tail. MS MARCO Passage and a real enterprise FAQ (well-curated, recall-rich) would bracket the picture from the other side.
3. **Confirm the recall-scarce lift isn't an artefact of NFCorpus's labelling sparsity.** Several "scarce" queries have qrel labels that look incorrect (e.g. "soil health" gold = a folate-metabolism paper). Cleaner gold standards would tighten the scarce-bucket CI.
4. **Look harder at the regression cases.** The Japan-style failures don't aggregate into a measurable hit at n=200 — but they exist qualitatively, and they're exactly the cases adaptive routing would want to predict. A small classifier trained on top-K cross-encoder score distributions might catch them cheaply.

If you've replicated this differently and got a different answer, I'd genuinely like to see it.
