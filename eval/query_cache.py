"""Disk-persisted cache for LLM-generated query rewrites.

Lets the sweep, steelman, and qualitative runs share rewrite results across
processes — without it, each separate run pays N fresh LLM calls.
"""

import json
import os
import threading

from main import generate_queries_chatgpt

_CACHE_PATH = os.environ.get("QUERY_CACHE_PATH", "./query_cache.json")
_cache = None
_lock = threading.Lock()


def _serialize_key(qid, diverse):
    return f"{qid}|{int(bool(diverse))}"


def _load():
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            _cache = json.load(f)
    else:
        _cache = {}
    return _cache


def _save():
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_cache, f)
    os.replace(tmp, _CACHE_PATH)


def cached_generate(qid, query, diverse=True):
    """Return cached LLM-generated rewrites, or generate + persist them."""
    cache = _load()
    key = _serialize_key(qid, diverse)
    if key in cache:
        return cache[key]
    with _lock:
        # double-check after acquiring lock
        if key in cache:
            return cache[key]
        rewrites = generate_queries_chatgpt(query, diverse=diverse)
        cache[key] = rewrites
        _save()
    return rewrites
