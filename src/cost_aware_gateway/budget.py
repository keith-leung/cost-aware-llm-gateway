"""C1 — Redis-backed per-user dollar budget.

Two-step reserve-then-reconcile protocol (denominated in USD):
  1. check_and_reserve(user_id, est_cost_usd) — atomic INCRBY on reserved;
     raises BudgetExceededError if the user's dollar limit would be exceeded.
  2. record_actual(user_id, est_cost_usd, actual_cost_usd) — reconcile.

Internally all amounts are stored as micro-dollars ($×1e6, integer) in
Redis. This keeps the existing WATCH/MULTI/EXEC + INCRBY atomic pattern
(integer arithmetic) while exposing a float-dollar API to callers.

State lives in Redis so multiple BudgetTracker instances share counters.

Atomicity:
  - check_and_reserve uses WATCH + MULTI/EXEC.
  - record_actual uses WATCH + MULTI/EXEC.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import redis

from cost_aware_gateway.models import BudgetExceededError

logger = logging.getLogger(__name__)

# 1 dollar = 1_000_000 micro-dollars. All Redis-stored values are int µ$.
_MICRO = 1_000_000


def _to_micro(usd: float) -> int:
    """Convert dollar float to integer micro-dollar for Redis storage."""
    return max(0, round(usd * _MICRO))


def _from_micro(micro: int) -> float:
    """Convert integer micro-dollar from Redis back to dollar float."""
    return micro / _MICRO


class BudgetTracker:
    """Redis-backed per-user dollar budget with two-step reserve-then-reconcile.

    All public methods accept/return float USD. Internally, Redis stores
    integer micro-dollars so INCRBY and WATCH/MULTI/EXEC work with integer
    arithmetic.
    """

    def __init__(
        self,
        redis_url: str,
        default_limit: float = 10.0,
        default_window: float = 3600.0,
        _redis_client: redis.Redis | None = None,
    ) -> None:
        if _redis_client is not None:
            self._redis = _redis_client
        else:
            self._redis = redis.from_url(redis_url, decode_responses=True)
        self._default_limit = _to_micro(default_limit)
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
        self, user_id: str, limit_usd: float, window_seconds: float | None = None
    ) -> None:
        """Set (or update) a user's dollar budget limit and window."""
        key_reserved, key_spent, key_window = self._keys(user_id)
        uid = user_id.replace(":", "_")
        pipe = self._redis.pipeline()
        pipe.set(f"{self._key_prefix}:{uid}:limit", _to_micro(limit_usd))
        if window_seconds is not None:
            pipe.set(f"{self._key_prefix}:{uid}:window", window_seconds)
        pipe.set(key_window, time.time(), ex=int(window_seconds or self._default_window))
        pipe.execute()

    def check_and_reserve(self, user_id: str, est_cost_usd: float) -> None:
        """Atomically reserve est_cost_usd for user_id.

        The enforced invariant is: spent + reserved + est <= limit (all µ$).
        Raises BudgetExceededError (in USD) on overflow.
        """
        est_micro = _to_micro(est_cost_usd)
        if est_micro <= 0:
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
                        pipe.set(key_reserved, est_micro)
                        pipe.execute()
                        return
                    window_start = float(current_window)
                    if now - window_start >= window:
                        pipe.multi()
                        pipe.delete(key_reserved, key_spent)
                        pipe.set(key_window, now, ex=int(window))
                        pipe.set(key_reserved, est_micro)
                        pipe.execute()
                        return
                    reserved = int(pipe.get(key_reserved) or 0)
                    spent = int(pipe.get(key_spent) or 0)
                    if spent + reserved + est_micro > limit:
                        raise BudgetExceededError(
                            user_id,
                            _from_micro(spent + reserved),
                            _from_micro(limit),
                        )
                    pipe.multi()
                    pipe.set(key_reserved, reserved + est_micro)
                    pipe.execute()
                    return
            except redis.exceptions.WatchError:
                continue

    def record_actual(
        self, user_id: str, est_cost_usd: float, actual_cost_usd: float
    ) -> None:
        """Reconcile after the call: charge actual cost, release reserved."""
        est_micro = _to_micro(est_cost_usd)
        actual_micro = _to_micro(actual_cost_usd)
        if est_micro <= 0 and actual_micro <= 0:
            return
        key_reserved, key_spent, _ = self._keys(user_id)

        while True:
            try:
                with self._redis.pipeline() as pipe:
                    pipe.watch(key_reserved, key_spent)
                    reserved = int(pipe.get(key_reserved) or 0)
                    spent = int(pipe.get(key_spent) or 0)
                    new_spent = spent + actual_micro
                    new_reserved = max(0, reserved - est_micro)
                    pipe.multi()
                    pipe.set(key_spent, new_spent)
                    pipe.set(key_reserved, new_reserved)
                    pipe.execute()
                    return
            except redis.exceptions.WatchError:
                continue

    def status(self, user_id: str) -> dict | None:
        """Return current budget status for user_id (in USD), or None."""
        key_reserved, key_spent, key_window = self._keys(user_id)
        limit_micro = self._get_limit(user_id)
        window = self._get_window(user_id)
        reserved_micro = int(self._redis.get(key_reserved) or 0)
        spent_micro = int(self._redis.get(key_spent) or 0)
        window_start = self._redis.get(key_window)
        now = time.time()
        if window_start is None:
            return None
        window_start_f = float(window_start)
        remaining_micro = max(0, limit_micro - spent_micro)
        return {
            "user_id": user_id,
            "limit_usd": round(_from_micro(limit_micro), 6),
            "window_seconds": window,
            "reserved_usd": round(_from_micro(reserved_micro), 6),
            "spent_usd": round(_from_micro(spent_micro), 6),
            "remaining_usd": round(_from_micro(remaining_micro), 6),
            "window_start": window_start_f,
            "window_resets_in": max(0.0, window - (now - window_start_f)),
        }

    def _get_limit(self, user_id: str) -> int:
        """Return limit in micro-dollars (int)."""
        val = self._redis.get(f"{self._key_prefix}:{user_id}:limit")
        return int(val) if val is not None else self._default_limit

    def _get_window(self, user_id: str) -> float:
        val = self._redis.get(f"{self._key_prefix}:{user_id}:window")
        return float(val) if val is not None else self._default_window

    def reset(self, user_id: str) -> None:
        """Force-reset a user's budget counters (test helper)."""
        key_reserved, key_spent, key_window = self._keys(user_id)
        self._redis.delete(key_reserved, key_spent, key_window)
