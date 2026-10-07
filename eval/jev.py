"""Jev (TypeSafe's System One decision model) as a component inside the fusion pipeline.

Three uses, each one Jev call:
  - relevance(query, texts): calibrated P(relevant) per document — reranker and evidence gate
  - intent_weights(query, variants): P(variant keeps the user's intent) — fusion weights
  - query_kind(query): navigational / specific / broad / multi_faceted — fusion router

Jev is not deterministic (identical inputs can move scores by a few hundredths), so answers
are cached on disk keyed by JEV_RUN. Within one run, steelman and answer_eval see identical
Jev outputs; a repeat run (JEV_RUN=2) draws fresh ones, which is how run-to-run variance
is measured.
"""

import atexit
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
JEV_RUN = os.getenv("JEV_RUN", "1")
_CACHE_PATH = os.getenv("JEV_CACHE_PATH", "./jev_cache.json")
# Per-request timeout in seconds. The SDK default (10 s) suits the hosted API; a local
# System One server (e.g. decider via TYPESAFE_BASE_URL) can need much longer per call.
JEV_TIMEOUT = float(os.getenv("JEV_TIMEOUT", "0")) or None

# TypeSafe documents 30 documents per call as the tested batch size and a ~64k-token
# request budget; keep well under it.
MAX_DOCS_PER_CALL = 30
MAX_STATE_CHARS = 100_000

RELEVANCE_QUESTION = ("Document `documents.{id}` is relevant to `query`: it contains information "
                      "that answers or directly addresses it.")
RELEVANCE_CRITERIA = {
    "true": "The document contains information that answers the query or directly addresses what it asks about.",
    "false": "The document is only loosely related, on a similar topic, or does not address what the query asks.",
}

# Four-level rubric, worded as in anessbelbati/jev-rerank-bench (rerankers/systemone.py,
# JevScoreBatch), where it scored 0.692 nDCG@10 against 0.685 for 30 yes/no questions.
SCORE_QUESTION = ("How well does document `documents.{id}` supply the information needed to "
                  "answer or verify `query`?")
SCORE_RUBRIC = ["The passage is off-topic for the query.",
                "The passage is on a related topic but does not supply what the query asks for.",
                "The passage partly supplies the information needed to answer or verify the query.",
                "The passage fully supplies the information needed to answer or verify the query."]

INTENT_QUESTION = ("Search variant `variants.{id}` keeps the intent of `query`: searching for it would "
                   "find material that helps answer what the user asked.")
INTENT_CRITERIA = {
    "true": "It searches for the same need, possibly from a different angle, narrower, broader, or with synonyms.",
    "false": "It shifts to a different need or topic than the one the user asked about.",
}

KIND_QUESTION = "What kind of search is `query`?"
KIND_CRITERIA = {
    "navigational": "Looking for one specific known item, name, or entity page.",
    "specific": "One precise question with a single focused answer.",
    "broad": "A topic with no specific question; the user wants an overview.",
    "multi_faceted": "Several distinct sub-questions or angles that each need their own evidence.",
}

_client = None
_cache = None
_dirty = 0
_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from typesafe_sdk import RetryPolicy, TypeSafeClient
        load_dotenv(".env")
        api_key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
        if not api_key:
            raise Exception("No Jev API key found. Set JEV_API_KEY in .env.")
        _client = TypeSafeClient(api_key=api_key, timeout=JEV_TIMEOUT,
                                 retry=RetryPolicy(max_retries=6, backoff_max=30.0,
                                                   timeout=max(90.0, JEV_TIMEOUT or 0)))
    return _client


def _load():
    global _cache
    if _cache is None:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH) as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _save():
    global _dirty
    with _lock:
        if _cache is None or not _dirty:
            return
        tmp = _CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_cache, f)
        os.replace(tmp, _CACHE_PATH)
        _dirty = 0


atexit.register(_save)


def _cached(kind, payload, compute):
    global _dirty
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    key = f"{JEV_RUN}|{JEV_MODEL}|{kind}|{digest}"
    cache = _load()
    if key in cache:
        return cache[key]
    value = compute()
    with _lock:
        cache[key] = value
        _dirty += 1
        flush = _dirty >= 25
    if flush:
        _save()
    return value


def _noul_batch(state_key, items, question, criteria, extra_state):
    """One Jev call: a Noul per item in `items` (list of texts). Returns probabilities in order."""
    from typesafe_sdk import Noul
    ids = [f"{state_key[0].upper()}{j:02d}" for j in range(len(items))]
    state = {**extra_state, state_key: dict(zip(ids, items))}
    questions = {i: Noul(instructions=question.format(id=i), criteria=criteria) for i in ids}
    r = _get_client().system_one(state=state, questions=questions, model=JEV_MODEL)
    return [float(r.answers[i].noul) for i in ids]


def _chunks(texts):
    batch, chars = [], 0
    for i, t in enumerate(texts):
        if batch and (len(batch) >= MAX_DOCS_PER_CALL or chars + len(t) > MAX_STATE_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(i)
        chars += len(t)
    if batch:
        yield batch


def relevance(query, texts):
    """P(relevant to query) per text. Pools over 30 docs are split into concurrent calls;
    the scores are calibrated probabilities, so they compare across calls."""
    if not texts:
        return []

    def compute():
        batches = list(_chunks(texts))

        def score(idx):
            return _noul_batch("documents", [texts[i] for i in idx], RELEVANCE_QUESTION,
                               RELEVANCE_CRITERIA, {"query": query})

        with ThreadPoolExecutor(4) as ex:
            parts = list(ex.map(score, batches))
        out = [0.0] * len(texts)
        for idx, part in zip(batches, parts):
            for i, s in zip(idx, part):
                out[i] = s
        return out

    return _cached("relevance", {"q": query, "t": texts}, compute)


def _expected_level(answer):
    """Expected rubric level scaled to 0..1 (0 = off-topic, 1 = fully supplies)."""
    legend, probs = dict(answer.legend or {}), dict(answer.probabilities or {})
    if legend and probs:
        def level(k):
            text = legend[k]
            return SCORE_RUBRIC.index(text) if text in SCORE_RUBRIC else int(k)
        return sum(p * level(k) for k, p in probs.items()) / (len(SCORE_RUBRIC) - 1)
    return float(answer.score)


def _score_batch(items, query):
    """One Jev call: a four-level Score question per document. Returns 0..1 in order."""
    from typesafe_sdk import Score
    ids = [f"D{j:02d}" for j in range(len(items))]
    state = {"query": query, "documents": dict(zip(ids, items))}
    questions = {i: Score(instructions=SCORE_QUESTION.format(id=i), criteria=SCORE_RUBRIC)
                 for i in ids}
    r = _get_client().system_one(state=state, questions=questions, model=JEV_MODEL)
    return [_expected_level(r.answers[i]) for i in ids]


def relevance_score(query, texts):
    """Expected rubric level (0..1) per text, batched like relevance()."""
    if not texts:
        return []

    def compute():
        batches = list(_chunks(texts))
        with ThreadPoolExecutor(4) as ex:
            parts = list(ex.map(lambda idx: _score_batch([texts[i] for i in idx], query), batches))
        out = [0.0] * len(texts)
        for idx, part in zip(batches, parts):
            for i, s in zip(idx, part):
                out[i] = s
        return out

    return _cached("relevance_score", {"q": query, "t": texts}, compute)


def intent_weights(query, variants):
    """P(variant keeps the user's intent) per variant, one call for all variants."""
    if not variants:
        return []
    return _cached("intent", {"q": query, "v": variants},
                   lambda: _noul_batch("variants", variants, INTENT_QUESTION,
                                       INTENT_CRITERIA, {"query": query}))


def query_kind(query):
    """Returns {"choice": str, "probabilities": {kind: p}}."""
    def compute():
        from typesafe_sdk import Choice
        r = _get_client().system_one(
            state={"query": query},
            questions={"kind": Choice(instructions=KIND_QUESTION, criteria=KIND_CRITERIA)},
            model=JEV_MODEL)
        a = r.answers["kind"]
        return {"choice": a.choice, "probabilities": dict(a.probabilities)}

    return _cached("kind", {"q": query}, compute)
