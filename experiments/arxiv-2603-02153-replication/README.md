# A small replication of arXiv 2603.02153v1 on NFCorpus

**TL;DR.** The paper argues that retrieval-fusion gains "evaporate" after reranking and truncation. With a weak reranker on a 30-query NFCorpus sample, I couldn't reproduce that — fusion's lift over baseline actually *grew* under rerank. But upgrading to `bge-reranker-large` absorbs ~68% of fusion's lift, and on the easy ~67% of queries fusion goes **net-negative** (-0.031 NDCG@10) — the paper's mechanism validates there. Reading the actual generated answers across 8 queries by hand, fusion's remaining value lives almost entirely in a recall-scarce tail where it can recover queries the baseline misses outright (the canonical case: query "kohlrabi" → corpus indexed under "glucoraphanin in cruciferous vegetable seeds") — wins no reranker can absorb. My honest read: **RAG-Fusion is a precision tool for hard queries, not a default-on quality boost.** The paper is half-right; the technique is half-defensible. The deployment pattern that survives both is **adaptive routing** — fire fusion only when a cheap weakness signal trips on baseline retrieval.

## Disclosure first

I wrote the original RAG-Fusion article in 2024, so I have an obvious stake in the conclusion and you should weigh the framing accordingly. I tried to design the experiment to give the paper's claim a fair shot — same fusion-then-rerank ordering, a credible cross-encoder, a real BEIR benchmark — but selection effects creep in even when I'm trying to be careful. If you spot one, please open an issue.

## What the paper claims (very briefly)

arXiv 2603.02153v1 reports that on a 115-query synthetic enterprise-support set, two-query fusion (Q1 + one LLM rewrite) followed by FlashRank reranking and Top-K truncation either matches or slightly underperforms a single-query baseline. Their proposed mechanism: the reranker re-anchors on Q1, and the rewrite's marginal candidates are squeezed out at truncation. They do note (§6.6) that fusion still helps in "recall-scarce" query regimes.

## What I changed and why

| Choice | Mine | Theirs |
|---|---|---|
| Dataset | NFCorpus (biomedical, BEIR) | 115 synthetic RAGAS support queries |
| Sample size | 30 queries | 115 queries |
| Reranker | `BAAI/bge-reranker-base` | FlashRank cross-encoder |
| Rewrites (default) | 4 | 1 |
| Pipeline order | Retrieve → fuse → rerank → truncate | Retrieve → rerank per-query → fuse → truncate |
| Candidate pool before rerank | 50 (default) | ~10 |

These are not small changes, and I'm not claiming a like-for-like replication. NFCorpus is harder than synthetic support queries — it's exactly the "recall-scarce regime" the paper says fusion still helps in. So part of the divergence may just be that I'm testing a regime they already conceded.

## Headline numbers (NDCG@10, 30-query NFCorpus sample)

| Method | No rerank | + Rerank |
|---|---|---|
| Baseline | 0.291 | 0.343 |
| RAG-Fusion (4 rewrites) | 0.289 | **0.390** |
| Diverse RAG-Fusion | 0.316 | **0.393** |
| Hybrid + Diverse | 0.337 | **0.397** |

Reranking lifts every method, but it lifts fusion *more* than it lifts baseline. Plain RAG-Fusion gains +0.101 NDCG@10 from rerank vs baseline's +0.052.

## Where the paper's effect does show up

I ran two follow-up sweeps to find conditions where their claim might still hold.

### Pool sweep (4 rewrites fixed)

| Candidate pool | Baseline+rerank | Fusion+rerank | Lift |
|---|---|---|---|
| 10 | 0.338 | 0.354 | +0.016 |
| 20 | 0.338 | 0.354 | +0.016 |
| 30 | 0.337 | 0.357 | +0.020 |
| **50** | 0.338 | **0.388** | **+0.050** |
| 75 | 0.345 | 0.393 | +0.048 |

There's a clear discontinuity between pool=30 and pool=50. Below it, fusion's added candidates can't survive truncation upstream, so the reranker never sees them. Above it, the diversity has room to live. Fusion is still positive at small pools — but barely. This is the half of the paper's mechanism that I think is right.

### N-rewrites sweep (pool=50 fixed)

| Rewrites | NDCG@10 | Recall@10 | MRR |
|---|---|---|---|
| 0 (baseline) | 0.338 | 0.200 | 0.529 |
| **1** | 0.354 | 0.217 | **0.520** ↓ |
| 2 | 0.374 | 0.236 | 0.590 |
| 3 | **0.391** | **0.252** | 0.593 |
| 4 | 0.388 | 0.251 | 0.591 |

At **N=1 — the paper's exact rewrite count — MRR drops below baseline.** A single LLM rewrite mostly produces a near-paraphrase of Q1, the reranker treats it as redundant noise, and the result is worse than just running Q1. That's the paper's claim made local. The crossover where fusion clearly dominates is between N=2 and N=3. By N=3 the lift saturates.

## What I think this means

The paper identifies a real failure surface for fusion: when the candidate pool entering the reranker is tight **and** the rewrites don't add real diversity, fusion can underperform a single query. That's not a strawman — it's the exact regime their pipeline operates in.

But I don't think the headline ("fusion gains evaporate in production") generalises. In every setting I tested with pool ≥ 50 **and** N ≥ 3, fusion+rerank cleanly beat baseline+rerank by 5+ NDCG@10 points. Both thresholds sit comfortably inside what I'd consider standard production parameters today.

So my best current read: the paper's mechanism is correct; the framing is too strong. It would be more accurately stated as *"fusion underperforms when its diversity budget is below the reranker's noise floor."*

## Things I'm not sure about

A non-exhaustive list of ways this could be wrong:

- **Sample size is small.** 30 queries with no significance testing. The N=1 MRR drop is 0.009 — easily inside noise. I'd want bootstrap CIs and a 200+ query run before treating any single number as load-bearing.
- **One dataset, one domain.** NFCorpus is biomedical. The paper uses enterprise support queries. I haven't tested whether the discontinuity at pool=50 reproduces on, say, MS MARCO or a real internal corpus.
- **One reranker.** `BAAI/bge-reranker-base` is a reasonable default but not the only choice. A larger reranker or one trained on the target domain could shift the trade-off.
- **Pipeline ordering.** I run fuse-then-rerank; they run rerank-per-query-then-fuse. Their ordering might genuinely be worse for fusion in ways my pipeline doesn't expose.
- **Rewrite quality.** I use GPT to generate four rewrites with a "diverse" prompt. The paper uses LLaMA with a different prompt. The rewrites might just be better in my setup.
- **Selection effects on the sample.** The 30 queries are random with seed=42 — but I haven't checked whether they're skewed toward hard or easy ones.
- **The cross-encoder I use was trained on MS MARCO** which has overlap concerns I haven't audited.

## How to reproduce

```bash
# Headline comparison — no-rerank vs +rerank
python evaluate.py --sample 30 --k 5 10 20 \
  --methods baseline hybrid rag-fusion rag-fusion-diverse hybrid-diverse
python evaluate.py --sample 30 --k 5 10 20 --rerank --candidate-pool 50 \
  --methods baseline hybrid rag-fusion rag-fusion-diverse hybrid-diverse

# Pool + N sweeps (the data backing the failure-corner claim)
python -m eval.sweep \
  --sample 30 --pool-values 10 20 30 50 75 --n-values 1 2 3 4 \
  --out experiments/arxiv-2603-02153-replication/results/sweep.json
```

Raw sweep output is in `results/sweep.json` for re-analysis.

## Steelman: where does the paper's argument actually hold?

To find the strongest version of the paper's argument, I ran three more tests probing its three load-bearing pieces: pipeline ordering, truncation depth, and difficulty regime. Code in `eval/steelman.py`, raw output in `results/steelman.json`.

### Test 1 — pipeline ordering (n=30)

The paper's ordering is per-query retrieve → per-query rerank → fuse → truncate. Mine is per-query retrieve → fuse → rerank-fused-pool → truncate. Both share the same rerank compute budget (~50 cross-encoder pairs per query).

| Method | NDCG@10 | Recall@10 | MRR |
|---|---|---|---|
| baseline+rerank | 0.338 | 0.200 | 0.527 |
| fuse_then_rerank (mine) | **0.379** | 0.238 | 0.559 |
| rerank_per_query_then_fuse (paper) | 0.305 | 0.220 | 0.399 |

The paper's ordering loses to baseline by 0.033 NDCG@10 and a huge 0.128 MRR (0.527 → 0.399). With my ordering, fusion wins by +0.041. Same rewrites, same compute, same data — only the order changes. **This reproduces a paper-shaped negative result on NFCorpus.** Their negative finding may be an artefact of putting the reranker inside the fusion loop, where it truncates each sub-query's pool to 10 *before* RRF can aggregate the diversity. Read charitably, this is a real warning to practitioners about where the reranker belongs.

### Test 2 — truncation depth (NDCG@K)

| Method | K=1 | K=3 | K=5 | K=10 |
|---|---|---|---|---|
| baseline+rerank | 0.500 | 0.401 | 0.381 | 0.338 |
| fuse_then_rerank | 0.533 | 0.449 | 0.418 | 0.379 |
| paper_pipeline | 0.333 | 0.310 | 0.311 | 0.305 |

Fusion's lift is +0.033 / +0.048 / +0.037 / +0.041 at K=1/3/5/10. The paper's truncation argument predicts the lift should *shrink* as K decreases — instead it's roughly K-independent, marginally larger at K=3. On this stack, harsh truncation doesn't absorb fusion gains.

### Test 3 — difficulty stratification (n_rich=19, n_scarce=11)

Bucketed by whether baseline+rerank's top-10 contains ≥1 relevant doc.

| Method | Rich NDCG@10 | Scarce NDCG@10 | Rich R@10 | Scarce R@10 |
|---|---|---|---|---|
| baseline+rerank | 0.534 | 0.000 | 0.316 | 0.000 |
| fuse_then_rerank | 0.545 (+0.011) | 0.091 | 0.324 | 0.091 |
| paper_pipeline | 0.464 (-0.070) | 0.029 | 0.295 | 0.091 |

This is the part where the paper has a real point. **On the easy 63% of queries, fusion's lift is +0.011 NDCG@10 — within noise.** The 4× LLM cost buys essentially nothing on the recall-rich majority. Fusion's value is concentrated entirely in the hard 37%, where baseline gets a flat zero and fusion is the only recovery mechanism. On a workload dominated by easy queries, the cost-benefit math is genuinely unfavourable for fusion.

### The strongest version of the paper's argument

After running the steelman, I'd phrase their case this way — narrower than their headline but still load-bearing:

> *On a production workload dominated by recall-rich queries — those a single dense+sparse retrieval already handles — the LLM-rewrite and extra-retrieval cost of RAG-Fusion isn't justified. The per-query rerank step absorbs whatever marginal diversity fusion adds, the lift on the easy majority is statistical noise, and the cost-benefit ratio collapses. Fusion's value lives in the recall-scarce tail; deployers should size that tail before adopting.*

Two things sustain that version:
- The +0.011 NDCG@10 lift on recall-rich queries is hard to defend against 4× LLM cost.
- The paper's exact pipeline ordering does produce a negative result on this data, so practitioners who get the order wrong reproduce their finding.

Two things it can't claim:
- The truncation mechanism (Test 2) doesn't show up on this data.
- The recall-scarce tail is 37% of NFCorpus — not a minority. On harder corpora, fusion's tail-recovery role is doing real work.

## Stronger reranker (`bge-reranker-large`)

Same setup, swapping `bge-reranker-base` → `bge-reranker-large`:

| Metric | base | large | Δ |
|---|---|---|---|
| baseline+rerank NDCG@10 | 0.338 | 0.382 | +0.044 |
| fuse_then_rerank NDCG@10 | 0.379 | 0.395 | +0.016 |
| **Fusion's lift over baseline** | **+0.041** | **+0.013** | **−68%** |

The stronger reranker **absorbs 68% of fusion's lift on this dataset.** Baseline gained more from the upgrade than fusion did, because the better reranker now finds good docs in the same candidate pool that fusion was previously needed to surface. This is a clean validation of the paper's "rerankers absorb fusion gains" claim.

By bucket (large reranker):

| Method | Rich NDCG@10 (n=20) | Scarce NDCG@10 (n=10) |
|---|---|---|
| baseline+rerank | 0.573 | 0.000 |
| fuse_then_rerank | 0.542 (−0.031) | 0.100 |

**Fusion is now net-negative on the easy 67% of queries** with a strong reranker. It still rescues the recall-scarce tail (where baseline scores a flat zero). The rich-bucket regression is small but consistent.

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

### Cost-adjusted picture (large reranker)

ΔNDCG@10 per extra LLM call (fusion costs +1 LLM call per query for the rewrite step):

| Bucket | n | Δ per LLM call |
|---|---|---|
| All | 30 | +0.010 |
| **Rich** | 20 | **−0.035** |
| **Scarce** | 10 | **+0.100** |

The negative rich-bucket lift isn't a metric artefact — the qualitative read confirms it (PLAIN-1441 Japan is exactly this regression made visible).

## My synthesised judgement

After running everything end-to-end and reading the actual outputs, I think the honest position is:

> **RAG-Fusion is a precision tool for the recall-scarce tail of a workload. It is not a default-on quality boost for the production-typical case.**

What survives:
- On hard queries the baseline misses entirely, fusion is **irreplaceable**. The kohlrabi case isn't a +2pp lift — it's the difference between "answer" and "I don't know." This is the technique's core value, and it's not absorbed by any reranker.
- The original technique works as designed.

What the steelman successfully forces me to concede:
- On the easy majority of queries, with a strong reranker, fusion's compute cost is not justified. The marginal NDCG lift collapses to within noise.
- Worse: fusion's extra diversity occasionally **regresses answer quality** by giving the synthesis model topically scattered context. Strong rerankers can't filter this because the issue is the input distribution to the LLM, not the retrieval.
- Whether to deploy fusion depends on workload composition — specifically the share of recall-scarce queries.

The deployment recommendation that survives both the original article and this critique: **adaptive routing** — fire fusion only on detected hard queries (gated by a cheap difficulty classifier or a confidence threshold on the baseline retrieval), not by default. That captures the kohlrabi-style wins, eliminates the Japan-style regressions, and removes the bulk of the cost.

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

- **FAQ chatbots and structured customer support.** Most queries match a known canonical entry. Adding 4× LLM cost per query for a +0.011 NDCG lift is the production-economics objection from the steelman, made literal.
- **Latency-critical retrieval (voice, autocomplete, real-time chat).** The query-rewrite call alone breaks a sub-500ms budget. No amount of retrieval quality compensates if the user has already given up.
- **High-volume / margin-thin deployments.** Per-query cost matters at scale. Fire fusion at 100% of traffic and you've quadrupled the LLM bill on retrieval; the 37% scarce-tail share where it pays for itself doesn't justify spending on the 63% it doesn't.
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
| **<10k docs** | Embedding + rerank likely already saturate recall on most queries. The pool=50 discontinuity we observed doesn't even fit a small corpus. Fusion's diversity has nowhere to live. Skip. |
| **10k–1M (NFCorpus is here)** | Where we tested. Fusion's mechanism works as described — ~+5pp NDCG@10 with weak rerankers, ~+1pp with strong ones, hero cases on the recall-scarce tail. |
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

## What I'd want to do next

1. Re-run on MS MARCO and a real enterprise-support corpus to check whether recall-scarce share is workload-dependent (central question for the steelman).
2. Bootstrap CIs on the +0.011 rich-bucket lift and the −0.031 large-reranker version — they're both inside plausible noise on n=20.
3. Build the adaptive-routing variant: cheap classifier or baseline-confidence threshold gating fusion only when needed. The point of this whole experiment is figuring out *when to deploy*, and the answer points at "selectively, on hard queries."
4. Larger sample (200+ queries) with a cleaner gold-standard than NFCorpus.

If you've replicated this differently and got a different answer, I'd genuinely like to see it.
