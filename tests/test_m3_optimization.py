"""M3 cost reduction tests — semantic cache + prompt compression.

Load-bearing assertions:
  1. Cache hit on similar query → ZERO model calls (call_spy count = 0).
  2. Compression reduces input tokens → cost_usd lower than uncompressed.

Each has paired mutations that must FAIL.
"""

from __future__ import annotations

import pytest

from cost_aware_gateway.config import (
    BudgetConfig, BreakerConfig, GatewayConfig, JudgeConfig,
    LiteLLMConfig, ProviderConfig, RedisConfig, TierConfig,
)
from cost_aware_gateway.cost import CostCalculator
from cost_aware_gateway.models import GatewayReply
from cost_aware_gateway.routing import (
    CostAwareRouter, KeywordDifficultyClassifier, KeywordQualityChecker,
)
from cost_aware_gateway.optimization import (
    CachedRouter, InMemorySemanticCache, TailTruncationCompressor,
    NoOpCompressor, SemanticCache, PromptCompressor,
)


# ---------------------------------------------------------------------------
# Config + deterministic call_fn (reuse from M2 pattern).
# ---------------------------------------------------------------------------

def _config() -> GatewayConfig:
    return GatewayConfig(
        mode="mock",
        default_provider="p",
        providers={
            "p": ProviderConfig(
                base_url="http://mock", api_key="mock",
                tiers={
                    "low": TierConfig(model="cheap", input_price_per_token=0.000001,
                                      output_price_per_token=0.000002, cache_read_price_per_token=0.0000001),
                    "high": TierConfig(model="expensive", input_price_per_token=0.000010,
                                       output_price_per_token=0.000020, cache_read_price_per_token=0.000001),
                },
            ),
        },
        judge=JudgeConfig(provider="p", tier="low"),
        redis=RedisConfig(url="redis://localhost:6379/0"),
        litellm=LiteLLMConfig(timeout=10, num_retries=1),
        budget=BudgetConfig(default_limit_usd=100.0, window_seconds=3600),
        breaker=BreakerConfig(failure_threshold=3, recovery_seconds=30),
    )


def _make_call_fn(input_tokens: int = 100, output_tokens: int = 50):
    """call_fn whose cost reflects actual input_tokens from pricing."""
    cfg = _config()
    calc = CostCalculator(cfg)

    def call_fn(*, provider, tier, system, user_message, max_tokens=1024, temperature=0.0):
        # Recompute input tokens from the actual prompt length sent.
        actual_input = max(1, len(system + user_message) // 4)
        cost = calc.cost_usd(provider, tier, actual_input, output_tokens)
        return GatewayReply(
            content=f"reply from {tier}",
            usage={"prompt_tokens": actual_input, "completion_tokens": output_tokens,
                   "total_tokens": actual_input + output_tokens},
            tier_used=tier, model_used=f"{provider}:{tier}",
            budget_after={}, breaker_state_after="closed",
            latency_ms=10.0, call_id=f"test:{tier}",
            cost_usd=cost, est_cost_usd=cost,
        )
    return call_fn


def _build_cached_router(
    *,
    cache: SemanticCache | None = None,
    compressor: PromptCompressor | None = None,
    call_spy=None,
) -> CachedRouter:
    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn()
    inner = CostAwareRouter(
        cfg, calc,
        classifier=KeywordDifficultyClassifier(),
        quality_checker=KeywordQualityChecker(),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )
    return CachedRouter(
        inner,
        cache=cache or InMemorySemanticCache(threshold=0.5),
        compressor=compressor or NoOpCompressor(),
        call_spy=call_spy,
    )


# ---------------------------------------------------------------------------
# 1. Semantic cache — hit means ZERO model calls.
# ---------------------------------------------------------------------------

def test_cache_hit_zero_model_calls():
    """Q1 misses → model called once. Similar Q2 hits → model called zero
    more times. Total call count = 1, not 2."""
    call_log: list[str] = []
    def spy(tier, msg):
        call_log.append(tier)

    router = _build_cached_router(call_spy=spy)

    # Q1: cache miss → model called.
    r1 = router.route(system="sys", user_message="what is python")
    assert r1.cache_hit is False
    assert len(call_log) == 1

    # Q2: semantically similar (Jaccard ≥ threshold) → cache hit → 0 calls.
    r2 = router.route(system="sys", user_message="what is python language")
    assert r2.cache_hit is True
    assert r2.total_cost_usd == 0.0  # zero model cost
    assert len(call_log) == 1, (
        f"cache hit should make 0 additional model calls, "
        f"but call_log has {len(call_log)} entries"
    )


def test_cache_miss_different_queries_still_call_model():
    """Dissimilar queries → cache miss → model called each time."""
    call_log: list[str] = []
    def spy(tier, msg):
        call_log.append(tier)

    router = _build_cached_router(call_spy=spy)
    router.route(system="sys", user_message="what is python")
    router.route(system="sys", user_message="completely different topic about cooking")
    assert len(call_log) == 2  # both missed


def test_cache_saves_money_vs_no_cache():
    """Q1 + similar Q2 via cache: total cost < 2 × single-call cost."""
    router = _build_cached_router()

    r1 = router.route(system="sys", user_message="what is python")
    single_cost = r1.total_cost_usd
    assert single_cost > 0

    r2 = router.route(system="sys", user_message="what is python language")
    total = r1.total_cost_usd + r2.total_cost_usd
    assert total < 2 * single_cost, (
        f"cached total (${total:.6f}) should be < 2×single "
        f"(${2*single_cost:.6f})"
    )
    assert r2.total_cost_usd == 0.0


def test_mutation_disable_cache_both_queries_call_model():
    """MUTATION: cache.get always returns None. Both Q1 and Q2 call the
    model. call_log = 2, not 1."""
    call_log: list[str] = []
    def spy(tier, msg):
        call_log.append(tier)

    class DisabledCache(SemanticCache):
        def get(self, system, user_message):
            return None  # mutation: never hit
        def put(self, system, user_message, reply):
            pass  # mutation: never store

    router = _build_cached_router(cache=DisabledCache(), call_spy=spy)
    router.route(system="sys", user_message="what is python")
    router.route(system="sys", user_message="what is python language")
    assert len(call_log) == 2, (
        f"disabled cache: both queries should call model (count=2), "
        f"got {len(call_log)}"
    )
    # The load-bearing "len(call_log) == 1 for similar queries" FAILS here.


# ---------------------------------------------------------------------------
# 2. Prompt compression — fewer tokens → lower cost.
# ---------------------------------------------------------------------------

def test_compression_reduces_tokens_and_cost():
    """A long prompt compressed → fewer input tokens → cost_usd lower
    than sending the full prompt."""
    long_system = "word " * 500    # ~125 tokens
    long_user = "question " * 500  # ~125 tokens

    call_log: list[GatewayReply] = []

    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn()

    # With compression (truncate to 10 system words + 10 user words).
    inner_compressed = CostAwareRouter(
        cfg, calc,
        classifier=KeywordDifficultyClassifier(),
        quality_checker=KeywordQualityChecker(),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )
    router_compressed = CachedRouter(
        inner_compressed,
        cache=InMemorySemanticCache(threshold=0.99),  # high threshold = no false hits
        compressor=TailTruncationCompressor(max_system_words=10, max_user_words=10),
    )
    r_compressed = router_compressed.route(
        system=long_system, user_message=long_user,
    )
    assert r_compressed.compressed is True
    assert r_compressed.compressed_input_tokens < r_compressed.original_input_tokens, (
        f"compressed tokens ({r_compressed.compressed_input_tokens}) should be < "
        f"original ({r_compressed.original_input_tokens})"
    )

    # Without compression (NoOp).
    inner_full = CostAwareRouter(
        cfg, calc,
        classifier=KeywordDifficultyClassifier(),
        quality_checker=KeywordQualityChecker(),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )
    router_full = CachedRouter(
        inner_full,
        cache=InMemorySemanticCache(threshold=0.99),
        compressor=NoOpCompressor(),
    )
    r_full = router_full.route(
        system=long_system, user_message=long_user,
    )

    assert r_compressed.reply.cost_usd < r_full.reply.cost_usd, (
        f"compressed cost (${r_compressed.reply.cost_usd:.6f}) should be < "
        f"uncompressed (${r_full.reply.cost_usd:.6f})"
    )


def test_mutation_disable_compression_no_cost_reduction():
    """MUTATION: compressor is NoOp → compressed == original tokens →
    cost same as uncompressed."""
    long_system = "word " * 500
    long_user = "question " * 500

    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn()

    # Mutation: NoOp compressor (same as "disabled").
    inner = CostAwareRouter(
        cfg, calc,
        classifier=KeywordDifficultyClassifier(),
        quality_checker=KeywordQualityChecker(),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )
    router_noop = CachedRouter(
        inner,
        cache=InMemorySemanticCache(threshold=0.99),
        compressor=NoOpCompressor(),  # mutation: no compression
    )
    r = router_noop.route(system=long_system, user_message=long_user)
    assert r.compressed is False, "mutation: NoOp should not compress"
    assert r.original_input_tokens == r.compressed_input_tokens, (
        "mutation: tokens should be equal (no compression)"
    )
    # The load-bearing "compressed cost < full cost" would FAIL here
    # because cost is the same.
