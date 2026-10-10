"""OpenAI's Decisions API (POST /v1/decisions, GPT-6 Luna) as a backend for eval/rerank.py.

Two question types, one request per pool of up to MAX_DOCS_PER_CALL documents:
  - "predicate": P(document is relevant), worded like eval/jev.py's yes/no question.
    Probabilities come back rounded to two decimals and are nearly binary on NFCorpus
    (2-6 distinct values in a 50-document pool), so most of a pool ties and keeps its
    retrieval order under rerank()'s stable sort.
  - "score": the four-level rubric from eval/jev.py (SCORE_RUBRIC). The API returns the
    expected level as `score` (0..3), scaled here to 0..1. 17-36 distinct values per pool.

The input is a single string, so each document is labelled inline ([D00], [D01], ...) and
each question names its label. A question can come back as a refusal; it scores 0 and is
counted in the cache entry.

Called over plain HTTP because the installed openai SDK (3.19.2) predates the endpoint; the
request and response shapes follow openai 3.26.0's resources/decisions.py.

Answers are cached on disk keyed by DECISIONS_RUN. Two identical calls returned identical
scores in a probe (2026-10-07), so one run per configuration is enough; a second run only
re-checks that. Usage per call is kept so cost can be summed. Price, per OpenAI's public-beta
announcement (6 Oct 2026): $0.10 per million input tokens, no output charge, no caching.
"""

import atexit
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from eval.jev import SCORE_QUESTION, SCORE_RUBRIC

DECISIONS_URL = "https://api.openai.com/v1/decisions"
DECISIONS_RUN = os.getenv("DECISIONS_RUN", "1")
_CACHE_PATH = os.getenv("DECISIONS_CACHE_PATH", "./decisions_cache.json")
MAX_DOCS_PER_CALL = int(os.getenv("DECISIONS_MAX_DOCS", "50"))

PREDICATE_QUESTION = ("Document [{id}] is relevant to the query: it contains information that "
                      "answers or directly addresses it.")

_cache = None
_dirty = 0
_lock = threading.Lock()


def _load():
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH) as f:
                _cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
        atexit.register(_save)
    return _cache


def _save():
    global _dirty
    if _cache is None:
        return
    with _lock:
        tmp = _CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_cache, f)
        os.replace(tmp, _CACHE_PATH)
        _dirty = 0


def _api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        from dotenv import load_dotenv
        load_dotenv(".env")
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise Exception("No OpenAI API key found. Set OPENAI_API_KEY in .env.")
    return key


def _post(payload, max_retries=6):
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(DECISIONS_URL, json=payload, timeout=120,
                              headers={"Authorization": f"Bearer {_api_key()}"})
        except (requests.Timeout, requests.ConnectionError):
            if attempt == max_retries:
                raise
            time.sleep(min(30.0, 2 ** attempt))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == max_retries:
                r.raise_for_status()
            time.sleep(min(30.0, 2 ** attempt))
            continue
        if r.status_code != 200:
            raise Exception(f"Decisions API {r.status_code}: {r.text[:500]}")
        return r.json()


def _question(kind, doc_id):
    if kind == "predicate":
        return {"type": "predicate", "name": doc_id,
                "instructions": PREDICATE_QUESTION.format(id=doc_id)}
    return {"type": "score", "name": doc_id,
            "instructions": SCORE_QUESTION.replace("`documents.{id}`", "[{id}]")
                                          .replace("`query`", "the query").format(id=doc_id),
            "levels": [{"label": text} for text in SCORE_RUBRIC]}


def _call(query, texts, kind, model):
    ids = [f"D{j:02d}" for j in range(len(texts))]
    body = _post({"model": model,
                  "input": f"Query: {query}\n\n" + "\n\n".join(
                      f"[{i}] {t}" for i, t in zip(ids, texts)),
                  "questions": [_question(kind, i) for i in ids]})
    by_name = {a.get("name"): a for a in body["answers"]}
    scores, refusals = [], 0
    for i in ids:
        a = by_name.get(i, {"type": "refusal"})
        if a["type"] == "predicate":
            scores.append(float(a["probability"]))
        elif a["type"] == "score":
            scores.append(float(a["score"]) / (len(SCORE_RUBRIC) - 1))
        else:
            scores.append(0.0)
            refusals += 1
    return scores, refusals, body.get("usage", {}).get("input_tokens", 0)


def relevance(query, texts, kind="predicate", model="gpt-6-luna"):
    """Decisions-API relevance per text, in input order (0..1 for both kinds)."""
    global _dirty
    if not texts:
        return []
    digest = hashlib.sha1(json.dumps({"q": query, "t": texts}, sort_keys=True).encode()).hexdigest()
    key = f"{DECISIONS_RUN}|{model}|{kind}@{MAX_DOCS_PER_CALL}|{digest}"
    cache = _load()
    if key in cache:
        return cache[key]["scores"]

    batches = [list(range(s, min(s + MAX_DOCS_PER_CALL, len(texts))))
               for s in range(0, len(texts), MAX_DOCS_PER_CALL)]
    with ThreadPoolExecutor(4) as ex:
        parts = list(ex.map(lambda idx: _call(query, [texts[i] for i in idx], kind, model),
                            batches))
    scores = [0.0] * len(texts)
    for idx, (part, _, _) in zip(batches, parts):
        for i, s in zip(idx, part):
            scores[i] = s
    with _lock:
        cache[key] = {"scores": scores,
                      "refusals": sum(p[1] for p in parts),
                      "input_tokens": sum(p[2] for p in parts)}
        _dirty += 1
        flush = _dirty >= 25
    if flush:
        _save()
    return scores
