"""C1 — Redis-backed per-user token budget.

Two-step reserve-then-reconcile protocol:
  1. check_and_reserve(user_id, est_tokens) — atomic INCRBY on reserved counter;
     raises BudgetExceededError if the user's limit would be exceeded.
  2. record_actual(user_id, est_tokens, actual_tokens) — reconcile: refund
     over-reservation or charge the delta, clamp at 0.

State lives in Redis so multiple BudgetTracker instances share counters.

Atomicity:
  - check_and_reserve uses a Redis transaction (WATCH + MULTI/EXEC) to
    atomically check the window, reset if expired, and INCRBY the reserved
    counter. This is compatible with both real Redis and fakeredis.
  - record_actual uses a similar WATCH/MULTI/EXEC transaction.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import redis

from cost_aware_gateway.models import BudgetExceededError

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Redis-backed per-user token budget with two-step reserve-then-reconcile."""

    def __init__(
        self,
        redis_url: str,
        default_limit: int = 50_000,
        default_window: float = 3600.0,
        _redis_client: redis.Redis | None = None,
    ) -> None:
        if _redis_client is not None:
            self._redis = _redis_client
        else:
            self._redis = redis.from_url(redis_url, decode_responses=True)
        self._default_limit = default_limit
        self._default_window = default_window
        self._key_prefix = "cow:budget"

    def _keys(self, user_id: str) -> tuple[str, str, str]:
        uid = user_id.replace(":", "_")
        return (
            f"{self._key_prefix}:{uid}:reserved",
            f"{self._key_prefix}:{uid}:spent",
            f"{self._key_prefix}:{uid}:window_start",
        )

    def set_budget(
        self, user_id: str, limit_tokens: int, window_seconds: float | None = None
    ) -> None:
        """Set (or update) a user's budget limit and window."""
        key_reserved, key_spent, key_window = self._keys(user_id)
        uid = user_id.replace(":", "_")
        pipe = self._redis.pipeline()
        pipe.set(f"{self._key_prefix}:{uid}:limit", limit_tokens)
        if window_seconds is not None:
            pipe.set(f"{self._key_prefix}:{uid}:window", window_seconds)
        pipe.set(key_window, time.time(), ex=int(window_seconds or self._default_window))
        pipe.execute()

    def check_and_reserve(self, user_id: str, est_tokens: int) -> None:
        """Atomically reserve est_tokens for user_id.

        The enforced invariant is: spent + reserved + est <= limit.
        (spent = committed in this window; reserved = in-flight; est = this call.)
        Uses WATCH + MULTI/EXEC for atomicity.
        """
        if est_tokens <= 0:
            return
        key_reserved, key_spent, key_window = self._keys(user_id)
        limit = self._get_limit(user_id)
        window = self._get_window(user_id)
        now = time.time()

        while True:
            try:
                with self._redis.pipeline() as pipe:
                    pipe.watch(key_window, key_reserved, key_spent)
                    current_window = pipe.get(key_window)
                    if current_window is None:
                        pipe.multi()
                        pipe.set(key_window, now, ex=int(window))
                        pipe.set(key_reserved, est_tokens)
                        pipe.execute()
                        return
                    window_start = float(current_window)
                    if now - window_start >= window:
                        pipe.multi()
                        pipe.delete(key_reserved, key_spent)
                        pipe.set(key_window, now, ex=int(window))
                        pipe.set(key_reserved, est_tokens)
                        pipe.execute()
                        return
                    reserved = int(pipe.get(key_reserved) or 0)
                    spent = int(pipe.get(key_spent) or 0)
                    # The invariant: spent + reserved + est must not exceed limit.
                    if spent + reserved + est_tokens > limit:
                        raise BudgetExceededError(user_id, spent + reserved, limit)
                    pipe.multi()
                    pipe.set(key_reserved, reserved + est_tokens)
                    pipe.execute()
                    return
            except redis.exceptions.WatchError:
                continue

    def record_actual(
        self, user_id: str, est_tokens: int, actual_tokens: int
    ) -> None:
        """Reconcile after the call: charge actual spend, release reserved."""
        if est_tokens <= 0 and actual_tokens <= 0:
            return
        key_reserved, key_spent, _ = self._keys(user_id)

        while True:
            try:
                with self._redis.pipeline() as pipe:
                    pipe.watch(key_reserved, key_spent)
                    reserved = int(pipe.get(key_reserved) or 0)
                    spent = int(pipe.get(key_spent) or 0)
                    new_spent = spent + actual_tokens
                    new_reserved = max(0, reserved - est_tokens)
                    pipe.multi()
                    pipe.set(key_spent, new_spent)
                    pipe.set(key_reserved, new_reserved)
                    pipe.execute()
                    return
            except redis.exceptions.WatchError:
                continue

    def status(self, user_id: str) -> dict | None:
        """Return current budget status for user_id, or None if no budget set."""
        key_reserved, key_spent, key_window = self._keys(user_id)
        limit = self._get_limit(user_id)
        window = self._get_window(user_id)
        reserved = int(self._redis.get(key_reserved) or 0)
        spent = int(self._redis.get(key_spent) or 0)
        window_start = self._redis.get(key_window)
        now = time.time()
        if window_start is None:
            return None
        window_start_f = float(window_start)
        remaining = max(0, limit - spent)
        return {
            "user_id": user_id,
            "limit_tokens": limit,
            "window_seconds": window,
            "reserved": reserved,
            "spent": spent,
            "remaining": remaining,
            "window_start": window_start_f,
            "window_resets_in": max(0.0, window - (now - window_start_f)),
        }

    def _get_limit(self, user_id: str) -> int:
        val = self._redis.get(f"{self._key_prefix}:{user_id}:limit")
        return int(val) if val is not None else self._default_limit

    def _get_window(self, user_id: str) -> float:
        val = self._redis.get(f"{self._key_prefix}:{user_id}:window")
        return float(val) if val is not None else self._default_window

    def _get_spent(self, user_id: str) -> int:
        _, key_spent, _ = self._keys(user_id)
        return int(self._redis.get(key_spent) or 0)

    def reset(self, user_id: str) -> None:
        """Force-reset a user's budget counters (test helper)."""
        key_reserved, key_spent, key_window = self._keys(user_id)
        self._redis.delete(key_reserved, key_spent, key_window)
