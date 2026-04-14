from __future__ import annotations

import hashlib
import sqlite3
from typing import Callable


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _init_db(db_path: str) -> None:
    """Create the cache table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()


class _MemoryCache:
    """Simple in-process dict cache. One instance = one isolated cache."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value


def make_cached_llm(
    llm_fn: Callable[[str], str],
    cache: str,
) -> Callable[[str], str]:
    """Wrap llm_fn with a prompt cache.

    Identical prompts return the cached response without calling llm_fn again.
    Cache key is SHA-256(prompt) — deterministic across processes.

    Args:
        llm_fn: The LLM callable to wrap. Contract: str → str.
        cache: Either ":memory:" for an in-process dict, or a file path
               for a persistent SQLite database (e.g. "ragcheck.cache.db").

    Returns:
        A wrapped callable with the same signature as llm_fn.

    Example:
        cached_llm = make_cached_llm(my_llm, cache=":memory:")
        result = ragcheck.evaluate(..., llm_fn=cached_llm)
    """
    if cache == ":memory:":
        mem = _MemoryCache()

        def cached(prompt: str) -> str:
            key = _sha256(prompt)
            hit = mem.get(key)
            if hit is not None:
                return hit
            response = llm_fn(prompt)
            mem.set(key, response)
            return response

        return cached

    # SQLite-backed cache — open/close per call so Windows releases the file handle.
    db_path = cache
    _init_db(db_path)

    def cached(prompt: str) -> str:
        key = _sha256(prompt)
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        finally:
            conn.close()
        if row:
            return row[0]
        response = llm_fn(prompt)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
                (key, response),
            )
            conn.commit()
        finally:
            conn.close()
        return response

    return cached
