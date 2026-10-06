"""Rate limiting: at most N requests per key per window (fixed window).

    key "site-1:ip-1.2.3.4", limit 20, window 60 s
    request 1..20 inside the same minute -> allowed;   request 21 -> refused, "retry in 37 s"

Fixed window = one counter per (key, minute-number). Simple and cheap; the weakness is a burst of
2N at a window boundary, which is fine for protecting a public chat from abuse.

Two backends behind one interface: Redis (shared by every API process, what production uses) and
in memory (tests, single-process development).
"""

import time
from typing import Any, Protocol


class RateLimiter(Protocol):
    def hit(self, key: str, limit: int, window_s: int = 60) -> tuple[bool, int]:
        """Count one request. Returns (allowed, seconds until the window resets)."""


class MemoryRateLimiter:
    def __init__(self, clock: Any = time.time) -> None:
        self.clock = clock
        self.counts: dict[tuple[str, int], int] = {}

    def hit(self, key: str, limit: int, window_s: int = 60) -> tuple[bool, int]:
        now = self.clock()
        window = int(now // window_s)
        self.counts = {k: v for k, v in self.counts.items() if k[1] >= window}  # drop old windows
        count = self.counts[(key, window)] = self.counts.get((key, window), 0) + 1
        return count <= limit, int((window + 1) * window_s - now) + 1


class RedisRateLimiter:
    def __init__(self, redis: Any, clock: Any = time.time) -> None:
        self.redis, self.clock = redis, clock

    def hit(self, key: str, limit: int, window_s: int = 60) -> tuple[bool, int]:
        now = self.clock()
        window = int(now // window_s)
        redis_key = f"rl:{key}:{window}"
        pipe = self.redis.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, window_s * 2)  # old windows clean themselves up
        count = pipe.execute()[0]
        return count <= limit, int((window + 1) * window_s - now) + 1


def default_limiter(redis_url: str) -> RateLimiter:
    """Redis if it answers (shared by all API processes), else per-process memory with a warning."""
    import logging

    try:
        from redis import Redis

        client = Redis.from_url(redis_url, socket_connect_timeout=1)
        client.ping()
        return RedisRateLimiter(client)
    except Exception:
        logging.getLogger(__name__).warning("Redis unavailable: rate limits are per process only")
        return MemoryRateLimiter()
