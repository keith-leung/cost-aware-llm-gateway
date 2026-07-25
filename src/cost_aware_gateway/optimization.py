"""M3 — Cost reduction: semantic cache + prompt compression.

Two injectable mechanisms that sit OUTSIDE the M2 routing layer:

  * ``SemanticCache`` — ``get(query)`` returns a cached reply if a similar
    query was seen before. A hit means ZERO model calls (cost ≈ 0).

  * ``PromptCompressor`` — shrinks the input before sending it to the model.
    Fewer input tokens → lower dollar cost (M1 pricing × fewer tokens).

Both are interfaces with deterministic test implementations. Production
swaps in embedding-based similarity / A2-style structural compression.

The ``CachedRouter`` wraps M2's ``CostAwareRouter``:
    cache.get → hit? return (0 model calls)
              → miss? compress → route → model call → cache.put → return

Compression preserves a stable prefix (doesn't rewrite the whole prompt)
to avoid maximally disrupting provider prefix-cache — the tension A2
identified is real but M3 only proves the basic mechanism.
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
from cost_aware_gateway.routing import CostAwareRouter, RoutingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic cache interface.
# ---------------------------------------------------------------------------

class SemanticCache(ABC):
    """Interface: get a cached reply by semantic similarity to the query."""

    @abstractmethod
    def get(self, system: str, user_message: str) -> GatewayReply | None:
        """Return a cached reply if a similar query was seen, else None."""
        ...

    @abstractmethod
    def put(self, system: str, user_message: str, reply: GatewayReply) -> None:
        """Store a reply for future similarity matches."""
        ...


class InMemorySemanticCache(SemanticCache):
    """Deterministic similarity cache for tests.

    Similarity is determined by an injectable ``similarity_fn(a, b) -> float``
    and a ``threshold``. Two queries are "the same" if their similarity ≥
    threshold. Default similarity = Jaccard coefficient on word sets (no
    embeddings, deterministic, testable).

    This is NOT a real semantic cache (no embeddings) — it's a testable
    proxy. Production swaps in embedding-based similarity.
    """

    def __init__(
        self,
        *,
        similarity_fn: Callable[[str, str], float] | None = None,
        threshold: float = 0.7,
    ) -> None:
        self._similarity_fn = similarity_fn or self._jaccard_similarity
        self._threshold = threshold
        self._store: list[tuple[str, str, GatewayReply]] = []

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Word-level Jaccard coefficient. Deterministic, no embeddings."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa and not wb:
            return 1.0
        union = wa | wb
        if not union:
            return 0.0
        return len(wa & wb) / len(union)

    def get(self, system: str, user_message: str) -> GatewayReply | None:
        query_key = system + " " + user_message
        best_score = 0.0
        best_reply: GatewayReply | None = None
        for stored_sys, stored_msg, reply in self._store:
            stored_key = stored_sys + " " + stored_msg
            score = self._similarity_fn(query_key, stored_key)
            if score >= self._threshold and score > best_score:
                best_score = score
                best_reply = reply
        return best_reply

    def put(self, system: str, user_message: str, reply: GatewayReply) -> None:
        self._store.append((system, user_message, reply))


# ---------------------------------------------------------------------------
# Prompt compressor interface.
# ---------------------------------------------------------------------------

class PromptCompressor(ABC):
    """Interface: compress the prompt before sending to the model."""

    @abstractmethod
    def compress(self, system: str, user_message: str) -> tuple[str, str]:
        """Return (compressed_system, compressed_user_message).

        The returned pair should have fewer tokens than the input.
        Implementations should preserve a stable prefix to avoid
        maximally disrupting provider prefix-cache.
        """
        ...


class TailTruncationCompressor(PromptCompressor):
    """Deterministic compressor: keep the first N words, drop the rest.

    This preserves the prefix (stable for prefix-cache) and truncates the
    tail. NOT a real summarizer — it's a testable proxy that genuinely
    reduces token count. Production swaps in A2-style structural compression.
    """

    def __init__(self, *, max_system_words: int = 50, max_user_words: int = 200) -> None:
        self._max_system = max_system_words
        self._max_user = max_user_words

    def compress(self, system: str, user_message: str) -> tuple[str, str]:
        sys_words = system.split()
        usr_words = user_message.split()
        compressed_sys = " ".join(sys_words[: self._max_system])
        compressed_usr = " ".join(usr_words[: self._max_user])
        return compressed_sys, compressed_usr


class NoOpCompressor(PromptCompressor):
    """Pass-through compressor (disables compression)."""

    def compress(self, system: str, user_message: str) -> tuple[str, str]:
        return system, user_message


# ---------------------------------------------------------------------------
# CachedRouter — wraps M2 router with cache + compression.
# ---------------------------------------------------------------------------

@dataclass
class CachedRoutingResult(RoutingResult):
    """Extends RoutingResult with cache/compression metadata."""
    cache_hit: bool = False
    compressed: bool = False
    original_input_tokens: int = 0
    compressed_input_tokens: int = 0


class CachedRouter:
    """Cost-aware router with semantic cache + prompt compression.

    Flow:
      1. Check cache → hit? return cached reply (ZERO model calls).
      2. Miss → compress prompt → delegate to inner CostAwareRouter →
         store result in cache → return.

    ``call_spy`` is an optional callable that fires on every model
    invocation (not on cache hits). Tests use it to prove a cache hit
    made zero model calls.
    """

    def __init__(
        self,
        inner_router: CostAwareRouter,
        *,
        cache: SemanticCache | None = None,
        compressor: PromptCompressor | None = None,
        call_spy: Callable[[str, str], None] | None = None,
    ) -> None:
        self._inner = inner_router
        self._cache = cache or InMemorySemanticCache()
        self._compressor = compressor or NoOpCompressor()
        self._call_spy = call_spy

    def route(
        self,
        *,
        system: str,
        user_message: str,
        tier: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CachedRoutingResult:
        # 1. Cache check — BEFORE any model call.
        cached = self._cache.get(system, user_message)
        if cached is not None:
            return CachedRoutingResult(
                reply=cached,
                difficulty="cached",
                tiers_tried=[],
                escalated=False,
                total_cost_usd=0.0,  # zero model cost
                cache_hit=True,
                compressed=False,
            )

        # 2. Compress prompt.
        comp_sys, comp_usr = self._compressor.compress(system, user_message)
        was_compressed = (comp_sys != system) or (comp_usr != user_message)

        # Count original vs compressed tokens (rough estimate).
        orig_tokens = max(0, len(system + user_message) // 4)
        comp_tokens = max(0, len(comp_sys + comp_usr) // 4)

        # 3. Wrap the inner router's call_fn with a spy so we can count
        #    model invocations.
        orig_call_fn = self._inner._call_fn
        if self._call_spy is not None and orig_call_fn is not None:
            def spied_call_fn(**kwargs):
                self._call_spy(kwargs.get("tier", "?"), kwargs.get("user_message", ""))
                return orig_call_fn(**kwargs)
            self._inner._call_fn = spied_call_fn

        try:
            inner_result = self._inner.route(
                system=comp_sys,
                user_message=comp_usr,
                tier=tier,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        finally:
            # Restore original call_fn.
            self._inner._call_fn = orig_call_fn

        # 4. Store in cache.
        self._cache.put(system, user_message, inner_result.reply)

        return CachedRoutingResult(
            reply=inner_result.reply,
            difficulty=inner_result.difficulty,
            tiers_tried=inner_result.tiers_tried,
            escalated=inner_result.escalated,
            total_cost_usd=inner_result.total_cost_usd,
            cache_hit=False,
            compressed=was_compressed,
            original_input_tokens=orig_tokens,
            compressed_input_tokens=comp_tokens,
        )
