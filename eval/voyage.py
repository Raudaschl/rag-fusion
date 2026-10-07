"""Voyage AI rerankers (rerank-3, rerank-3-lite) as a backend for eval/rerank.py.

One POST per query to https://api.voyageai.com/v1/rerank with every candidate in the pool
(up to 1,000 documents; 32K-token context per query-document pair, truncation on). Scores are
Voyage's relevance_score, returned here in input order so the caller can zip them with doc_ids.

Scores are cached on disk keyed by VOYAGE_RUN, the same way eval/jev.py keys Jev draws, so a
second run (VOYAGE_RUN=2) measures whether identical requests give identical scores.

Billing (docs.voyageai.com/docs/pricing): total_tokens = query tokens x number of documents
+ all document tokens. The per-call total is kept in the cache so cost can be summed later.
"""

import atexit
import hashlib
import json
import os
import threading
import time

import requests

VOYAGE_URL = "https://api.voyageai.com/v1/rerank"
VOYAGE_RUN = os.getenv("VOYAGE_RUN", "1")
_CACHE_PATH = os.getenv("VOYAGE_CACHE_PATH", "./voyage_cache.json")

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
    key = os.getenv("VOYAGE_API_KEY")
    if not key:
        from dotenv import load_dotenv
        load_dotenv(".env")
        key = os.getenv("VOYAGE_API_KEY")
    if not key:
        raise Exception("No Voyage API key found. Set VOYAGE_API_KEY in .env.")
    return key


def _post(payload, max_retries=6):
    for attempt in range(max_retries + 1):
        r = requests.post(VOYAGE_URL, json=payload, timeout=90,
                          headers={"Authorization": f"Bearer {_api_key()}"})
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == max_retries:
                r.raise_for_status()
            time.sleep(min(30.0, 2 ** attempt))
            continue
        r.raise_for_status()
        return r.json()


def relevance(query, texts, model="rerank-3"):
    """Voyage relevance_score per text, in input order."""
    global _dirty
    if not texts:
        return []
    digest = hashlib.sha1(json.dumps({"q": query, "t": texts}, sort_keys=True).encode()).hexdigest()
    key = f"{VOYAGE_RUN}|{model}|rerank|{digest}"
    cache = _load()
    if key in cache:
        return cache[key]["scores"]

    body = _post({"query": query, "documents": texts, "model": model, "truncation": True})
    scores = [0.0] * len(texts)
    for item in body["data"]:
        scores[item["index"]] = float(item["relevance_score"])
    with _lock:
        cache[key] = {"scores": scores, "total_tokens": body.get("usage", {}).get("total_tokens")}
        _dirty += 1
        flush = _dirty >= 25
    if flush:
        _save()
    return scores
