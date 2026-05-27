from __future__ import annotations

import time
import sqlite3
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class RateLimitDecision:
    allowed: bool
    qps: float
    burst: int
    remaining: int
    retry_after_ms: int = 0
    reason: str = ""
    key: str = ""


class RateLimiter(Protocol):
    def allow(self, *, session_id: str, endpoint: str, options: dict | None = None) -> RateLimitDecision:
        ...


@dataclass(slots=True)
class _TokenBucket:
    tokens: float
    last_refill_ts: float


class InMemoryRateLimiter:
    """Simple in-memory token bucket limiter for per-session endpoint throttling."""

    def __init__(
        self,
        default_qps: float = 0.0,
        default_burst: int = 0,
        gc_ttl_sec: float = 120.0,
    ) -> None:
        self.default_qps = max(0.0, float(default_qps))
        self.default_burst = max(0, int(default_burst))
        self.gc_ttl_sec = max(5.0, float(gc_ttl_sec))
        self._buckets: dict[str, _TokenBucket] = {}

    @staticmethod
    def _resolve_qps(options: dict | None, default_qps: float) -> float:
        if not options:
            return default_qps
        try:
            return max(0.0, float(options.get("rate_limit_qps", default_qps)))
        except (TypeError, ValueError):
            return default_qps

    @staticmethod
    def _resolve_burst(options: dict | None, qps: float, default_burst: int) -> int:
        if options and "rate_limit_burst" in options:
            try:
                return max(0, int(options.get("rate_limit_burst", default_burst)))
            except (TypeError, ValueError):
                pass
        if default_burst > 0:
            return default_burst
        return max(1, int(round(qps))) if qps > 0 else 0

    def _gc(self, now: float) -> None:
        cutoff = now - self.gc_ttl_sec
        stale = [key for key, bucket in self._buckets.items() if bucket.last_refill_ts < cutoff]
        for key in stale:
            del self._buckets[key]

    def allow(self, *, session_id: str, endpoint: str, options: dict | None = None) -> RateLimitDecision:
        now = time.monotonic()
        self._gc(now)

        qps = self._resolve_qps(options, self.default_qps)
        burst = self._resolve_burst(options, qps, self.default_burst)

        if qps <= 0.0 or burst <= 0:
            return RateLimitDecision(
                allowed=True,
                qps=qps,
                burst=burst,
                remaining=burst,
                reason="rate-limit-disabled",
                key=f"{session_id}:{endpoint}",
            )

        key = f"{session_id}:{endpoint}"
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _TokenBucket(tokens=float(burst), last_refill_ts=now)
            self._buckets[key] = bucket

        elapsed = max(0.0, now - bucket.last_refill_ts)
        bucket.tokens = min(float(burst), bucket.tokens + (elapsed * qps))
        bucket.last_refill_ts = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return RateLimitDecision(
                allowed=True,
                qps=qps,
                burst=burst,
                remaining=max(0, int(bucket.tokens)),
                reason="within-rate-limit",
                key=key,
            )

        deficit = 1.0 - bucket.tokens
        retry_after_ms = int(max(1.0, (deficit / max(1e-9, qps)) * 1000.0))
        return RateLimitDecision(
            allowed=False,
            qps=qps,
            burst=burst,
            remaining=0,
            retry_after_ms=retry_after_ms,
            reason="rate-limit-exceeded",
            key=key,
        )


class SQLiteRateLimiter:
    """SQLite token bucket limiter for single-host multi-process deployments."""

    def __init__(
        self,
        db_path: str,
        default_qps: float = 0.0,
        default_burst: int = 0,
    ) -> None:
        self.db_path = db_path
        self.default_qps = max(0.0, float(default_qps))
        self.default_burst = max(0, int(default_burst))
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limit_buckets (
                    bucket_key TEXT PRIMARY KEY,
                    tokens REAL NOT NULL,
                    last_refill_ts REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _resolve_qps(options: dict | None, default_qps: float) -> float:
        if not options:
            return default_qps
        try:
            return max(0.0, float(options.get("rate_limit_qps", default_qps)))
        except (TypeError, ValueError):
            return default_qps

    @staticmethod
    def _resolve_burst(options: dict | None, qps: float, default_burst: int) -> int:
        if options and "rate_limit_burst" in options:
            try:
                return max(0, int(options.get("rate_limit_burst", default_burst)))
            except (TypeError, ValueError):
                pass
        if default_burst > 0:
            return default_burst
        return max(1, int(round(qps))) if qps > 0 else 0

    def allow(self, *, session_id: str, endpoint: str, options: dict | None = None) -> RateLimitDecision:
        now = time.monotonic()
        qps = self._resolve_qps(options, self.default_qps)
        burst = self._resolve_burst(options, qps, self.default_burst)
        key = f"{session_id}:{endpoint}"

        if qps <= 0.0 or burst <= 0:
            return RateLimitDecision(
                allowed=True,
                qps=qps,
                burst=burst,
                remaining=burst,
                reason="rate-limit-disabled",
                key=key,
            )

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT tokens, last_refill_ts FROM rate_limit_buckets WHERE bucket_key = ?",
                (key,),
            ).fetchone()

            if row is None:
                tokens = float(burst)
                last_refill_ts = now
                conn.execute(
                    "INSERT INTO rate_limit_buckets(bucket_key, tokens, last_refill_ts) VALUES(?, ?, ?)",
                    (key, tokens, last_refill_ts),
                )
            else:
                tokens = float(row[0])
                last_refill_ts = float(row[1])

            elapsed = max(0.0, now - last_refill_ts)
            tokens = min(float(burst), tokens + (elapsed * qps))

            if tokens >= 1.0:
                tokens -= 1.0
                conn.execute(
                    "UPDATE rate_limit_buckets SET tokens = ?, last_refill_ts = ? WHERE bucket_key = ?",
                    (tokens, now, key),
                )
                return RateLimitDecision(
                    allowed=True,
                    qps=qps,
                    burst=burst,
                    remaining=max(0, int(tokens)),
                    reason="within-rate-limit",
                    key=key,
                )

            conn.execute(
                "UPDATE rate_limit_buckets SET tokens = ?, last_refill_ts = ? WHERE bucket_key = ?",
                (tokens, now, key),
            )

        deficit = 1.0 - tokens
        retry_after_ms = int(max(1.0, (deficit / max(1e-9, qps)) * 1000.0))
        return RateLimitDecision(
            allowed=False,
            qps=qps,
            burst=burst,
            remaining=0,
            retry_after_ms=retry_after_ms,
            reason="rate-limit-exceeded",
            key=key,
        )
