"""M2 — Cost-aware routing: difficulty classification + cascade escalation.

Two injectable interfaces drive routing decisions:

  * ``DifficultyClassifier`` — classifies a query as "easy" or "hard".
    Easy → cheap tier; hard → expensive tier. The router uses this to
    pick the cheapest tier that can handle the query.

  * ``QualityChecker`` — given a reply, decides if the answer is good
    enough. The cascade tries cheap first; if quality fails, it escalates
    to the expensive tier.

Both have deterministic implementations for tests (no LLM needed) and
are injectable so production can swap in LLM-backed versions without
touching the router.

This module does NOT import litellm — it operates on an injectable
``call_fn(provider, tier, system, user_message, ...)`` callable, so it
works with or without the real SDK.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from cost_aware_gateway.config import GatewayConfig
from cost_aware_gateway.cost import CostCalculator
from cost_aware_gateway.models import GatewayReply

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Difficulty classifier interface.
# ---------------------------------------------------------------------------

class DifficultyClassifier(ABC):
    """Classifies a query's difficulty to select the cheapest adequate tier."""

    @abstractmethod
    def classify(self, system: str, user_message: str) -> str:
        """Return 'easy' or 'hard'.

        'easy' → the query can be served by the cheapest tier.
        'hard' → requires the expensive tier for acceptable quality.
        """
        ...


class KeywordDifficultyClassifier(DifficultyClassifier):
    """Deterministic keyword-based classifier for tests.

    Queries containing any keyword from ``hard_keywords`` are 'hard';
    everything else is 'easy'. This is NOT a real LLM classifier — it's
    a testable proxy. Production swaps in an LLM-backed implementation.
    """

    def __init__(self, hard_keywords: tuple[str, ...] = ("complex", "analyze", "compare", "derive")) -> None:
        self.hard_keywords = tuple(k.lower() for k in hard_keywords)

    def classify(self, system: str, user_message: str) -> str:
        text = (system + " " + user_message).lower()
        if any(kw in text for kw in self.hard_keywords):
            return "hard"
        return "easy"


# ---------------------------------------------------------------------------
# Quality checker interface.
# ---------------------------------------------------------------------------

class QualityChecker(ABC):
    """Checks whether a model's reply meets quality bar."""

    @abstractmethod
    def check(self, reply_content: str, system: str, user_message: str) -> bool:
        """Return True if quality is acceptable, False to escalate."""
        ...


class KeywordQualityChecker(QualityChecker):
    """Deterministic quality checker for tests.

    A reply is acceptable unless it contains any ``failure_markers``.
    This simulates a real quality check (e.g. judge LLM) without the
    network call — production swaps in a real judge.
    """

    def __init__(self, failure_markers: tuple[str, ...] = ("BAD_REPLY", "I_CANNOT", "LOW_QUALITY")) -> None:
        self.failure_markers = tuple(m.upper() for m in failure_markers)

    def check(self, reply_content: str, system: str, user_message: str) -> bool:
        text = (reply_content or "").upper()
        return not any(m in text for m in self.failure_markers)


# ---------------------------------------------------------------------------
# Routing result — carries the reply + routing metadata.
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """Outcome of a cost-aware routing call."""
    reply: GatewayReply
    difficulty: str           # "easy" | "hard"
    tiers_tried: list[str]    # e.g. ["low", "high"] if escalated
    escalated: bool           # True if cheap was tried but quality failed
    total_cost_usd: float     # sum of all calls (cheap + expensive if cascade)


# ---------------------------------------------------------------------------
# CostAwareRouter — difficulty routing + cascade escalation.
# ---------------------------------------------------------------------------

class CostAwareRouter:
    """Routes queries to the cheapest adequate tier.

    Two modes:
      * **Direct tier** (backward compat): caller passes ``tier=`` explicitly;
        the router uses that tier as before.
      * **Cost-aware** (new): caller omits tier (or passes ``tier="auto"``);
        the router classifies difficulty and picks the cheapest tier. If
        quality is insufficient, it cascades to the expensive tier.

    ``call_fn(provider, tier, system, user_message, ...) -> GatewayReply``
    is the injectable model-call function. In production it wraps litellm;
    in tests it's a deterministic stub.
    """

    def __init__(
        self,
        config: GatewayConfig,
        cost_calc: CostCalculator,
        *,
        classifier: DifficultyClassifier | None = None,
        quality_checker: QualityChecker | None = None,
        call_fn: Callable[..., GatewayReply] | None = None,
        cheap_tier: str = "low",
        expensive_tier: str = "high",
    ) -> None:
        self._config = config
        self._cost_calc = cost_calc
        self._classifier = classifier or KeywordDifficultyClassifier()
        self._quality_checker = quality_checker or KeywordQualityChecker()
        self._call_fn = call_fn
        self._cheap_tier = cheap_tier
        self._expensive_tier = expensive_tier

    def route(
        self,
        *,
        system: str,
        user_message: str,
        tier: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> RoutingResult:
        """Route a query through cost-aware tier selection.

        If ``tier`` is None or "auto", the router classifies difficulty
        and picks the tier. If ``tier`` is explicit, it uses that tier
        directly (backward compat).
        """
        if tier is not None and tier != "auto":
            # Explicit tier — backward compat, no classification.
            reply = self._call(system, user_message, tier, max_tokens, temperature)
            return RoutingResult(
                reply=reply,
                difficulty="explicit",
                tiers_tried=[tier],
                escalated=False,
                total_cost_usd=reply.cost_usd,
            )

        # Cost-aware mode: classify → pick tier → optional cascade.
        difficulty = self._classifier.classify(system, user_message)
        tiers_tried: list[str] = []
        total_cost = 0.0

        if difficulty == "hard":
            # Hard query → go straight to expensive tier.
            tiers_tried.append(self._expensive_tier)
            reply = self._call(system, user_message, self._expensive_tier, max_tokens, temperature)
            total_cost += reply.cost_usd
            return RoutingResult(
                reply=reply, difficulty=difficulty,
                tiers_tried=tiers_tried, escalated=False,
                total_cost_usd=total_cost,
            )

        # Easy query → try cheap first, cascade if quality fails.
        tiers_tried.append(self._cheap_tier)
        reply = self._call(system, user_message, self._cheap_tier, max_tokens, temperature)
        total_cost += reply.cost_usd

        if self._quality_checker.check(reply.content, system, user_message):
            # Cheap model was good enough.
            return RoutingResult(
                reply=reply, difficulty=difficulty,
                tiers_tried=tiers_tried, escalated=False,
                total_cost_usd=total_cost,
            )

        # Quality failed → escalate to expensive.
        tiers_tried.append(self._expensive_tier)
        expensive_reply = self._call(
            system, user_message, self._expensive_tier, max_tokens, temperature,
        )
        total_cost += expensive_reply.cost_usd
        return RoutingResult(
            reply=expensive_reply, difficulty=difficulty,
            tiers_tried=tiers_tried, escalated=True,
            total_cost_usd=total_cost,
        )

    def _call(
        self, system: str, user_message: str, tier: str,
        max_tokens: int, temperature: float,
    ) -> GatewayReply:
        """Invoke the model via the injectable call_fn."""
        if self._call_fn is None:
            raise RuntimeError("CostAwareRouter requires a call_fn to invoke models")
        return self._call_fn(
            provider=self._config.default_provider,
            tier=tier,
            system=system,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
