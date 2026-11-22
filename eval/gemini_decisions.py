"""Google's Gemini Decisions API (POST /v1beta/decisions, gemini-decision-1) as a backend for
eval/rerank.py.

Built to mirror eval/decisions.py (OpenAI's Decisions API) so the two compare like for like:
the same labelled input string ("Query: ...", then [D00] ..., [D01] ...), the same question
wording, and one request per pool of up to MAX_DOCS_PER_CALL documents. Two question types:
  - "noul": Google's yes/no type, worded as decisions.py's predicate. The answer's `noul` is a
    calibrated P(true), clipped by the API to [0.01, 0.99].
  - "score": the four-level rubric from eval/jev.py (SCORE_RUBRIC). The API returns the
    probability-weighted expected level as `score` (0..3), scaled here to 0..1.

Two routes, picked by the model name:
  - "gemini-decision-1": Google directly (GEMINI_API_KEY). Request and response shapes follow
    google-genai 3.6.1's _gaos/types/decisions/; the installed SDK (1.74.0) predates the
    endpoint, so it's called over plain HTTP.
  - "google/gemini-decision-1": OpenRouter's /api/alpha/decisions (OPENROUTER_API_KEY), which
    takes the shared text as `state` instead of `input` and can return a `refusal` answer.
Either way questions are a map keyed by name and answers come back under the same keys. A
blocked input (prompt_feedback with no answers, HTTP 400 with a safety code, or a refusal)
scores 0, so it keeps its retrieval order, and is counted in the cache entry.

Answers are cached on disk keyed by GEMINI_DECISIONS_RUN. Usage per call is kept so cost can be
summed. Price, per the launch coverage (11 Nov 2026): $0.10 per million input tokens, no output
charge.
"""

import atexit
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from eval.decisions import PREDICATE_QUESTION
from eval.jev import SCORE_QUESTION, SCORE_RUBRIC

GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
GEMINI_DECISIONS_RUN = os.getenv("GEMINI_DECISIONS_RUN", "1")
_CACHE_PATH = os.getenv("GEMINI_DECISIONS_CACHE_PATH", "./gemini_decisions_cache.json")
MAX_DOCS_PER_CALL = int(os.getenv("GEMINI_DECISIONS_MAX_DOCS", "50"))
_WORKERS = int(os.getenv("GEMINI_DECISIONS_WORKERS", "4"))

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


def _api_key(names):
    key = next((os.getenv(n) for n in names if os.getenv(n)), None)
    if not key:
        from dotenv import load_dotenv
        load_dotenv(".env")
        key = next((os.getenv(n) for n in names if os.getenv(n)), None)
    if not key:
        raise Exception(f"No API key found. Set {names[0]}.")
    return key


def _route(model):
    """(url, headers, name of the shared-text field) for this model name."""
    if "/" in model:
        return (OPENROUTER_URL,
                {"Authorization": f"Bearer {_api_key(['OPENROUTER_API_KEY'])}"}, "state")
    return GOOGLE_URL, {"x-goog-api-key": _api_key(["GEMINI_API_KEY", "GOOGLE_API_KEY"])}, "input"


class Blocked(Exception):
    pass


def _post(url, headers, payload, max_retries=8):
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=120, headers=headers)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == max_retries:
                raise
            time.sleep(min(60.0, 2 ** attempt))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == max_retries:
                r.raise_for_status()
            time.sleep(min(60.0, 2 ** attempt))
            continue
        if r.status_code == 400 and "safety" in r.text.lower():
            raise Blocked(r.text[:300])
        if r.status_code != 200:
            raise Exception(f"Gemini Decisions API {r.status_code}: {r.text[:500]}")
        return r.json()


def _question(kind, doc_id):
    if kind == "noul":
        return {"type": "noul", "instructions": PREDICATE_QUESTION.format(id=doc_id)}
    return {"type": "score",
            "instructions": SCORE_QUESTION.replace("`documents.{id}`", "[{id}]")
                                          .replace("`query`", "the query").format(id=doc_id),
            "criteria": list(SCORE_RUBRIC)}


def _call(query, texts, kind, model):
    ids = [f"D{j:02d}" for j in range(len(texts))]
    url, headers, field = _route(model)
    try:
        body = _post(url, headers, {
            "model": model,
            field: f"Query: {query}\n\n" + "\n\n".join(f"[{i}] {t}" for i, t in zip(ids, texts)),
            "questions": {i: _question(kind, i) for i in ids}})
    except Blocked:
        return [0.0] * len(texts), len(texts), 0
    answers = body.get("answers") or {}
    scores, blocked = [], 0
    for i in ids:
        a = answers.get(i)
        if a and a.get("type") == "noul":
            scores.append(float(a["noul"]))
        elif a and a.get("type") == "score":
            scores.append(float(a["score"]) / (len(SCORE_RUBRIC) - 1))
        else:
            scores.append(0.0)
            blocked += 1
    usage = body.get("usage") or {}
    return scores, blocked, usage.get("total_input_tokens", usage.get("input_tokens", 0))


def relevance(query, texts, kind="noul", model="gemini-decision-1"):
    """Gemini-Decisions relevance per text, in input order (0..1 for both kinds)."""
    global _dirty
    if not texts:
        return []
    digest = hashlib.sha1(json.dumps({"q": query, "t": texts}, sort_keys=True).encode()).hexdigest()
    key = f"{GEMINI_DECISIONS_RUN}|{model}|{kind}@{MAX_DOCS_PER_CALL}|{digest}"
    cache = _load()
    if key in cache:
        return cache[key]["scores"]

    batches = [list(range(s, min(s + MAX_DOCS_PER_CALL, len(texts))))
               for s in range(0, len(texts), MAX_DOCS_PER_CALL)]
    with ThreadPoolExecutor(_WORKERS) as ex:
        parts = list(ex.map(lambda idx: _call(query, [texts[i] for i in idx], kind, model),
                            batches))
    scores = [0.0] * len(texts)
    for idx, (part, _, _) in zip(batches, parts):
        for i, s in zip(idx, part):
            scores[i] = s
    with _lock:
        cache[key] = {"scores": scores,
                      "blocked": sum(p[1] for p in parts),
                      "input_tokens": sum(p[2] for p in parts)}
        _dirty += 1
        flush = _dirty >= 25
    if flush:
        _save()
    return scores
