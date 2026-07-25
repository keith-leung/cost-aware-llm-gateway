"""M2 cost-aware routing tests — difficulty routing + cascade escalation.

Load-bearing assertions:
  1. Mixed batch via difficulty routing costs less than always-expensive,
     AND hard queries still get the expensive tier (quality preserved).
  2. Cascade: cheap-sufficient → 1 call at cheap; cheap-insufficient →
     escalate to expensive, get correct answer.

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
    CostAwareRouter,
    DifficultyClassifier,
    KeywordDifficultyClassifier,
    KeywordQualityChecker,
    QualityChecker,
    RoutingResult,
)


# ---------------------------------------------------------------------------
# Config + deterministic call_fn helpers.
# ---------------------------------------------------------------------------

def _config() -> GatewayConfig:
    """Two-tier config: low (cheap) and high (expensive)."""
    return GatewayConfig(
        mode="mock",
        default_provider="p",
        providers={
            "p": ProviderConfig(
                base_url="http://mock",
                api_key="mock",
                tiers={
                    "low": TierConfig(
                        model="cheap-model",
                        input_price_per_token=0.000001,
                        output_price_per_token=0.000002,
                        cache_read_price_per_token=0.0000001,
                    ),
                    "high": TierConfig(
                        model="expensive-model",
                        input_price_per_token=0.000010,
                        output_price_per_token=0.000020,
                        cache_read_price_per_token=0.000001,
                    ),
                },
            ),
        },
        judge=JudgeConfig(provider="p", tier="low"),
        redis=RedisConfig(url="redis://localhost:6379/0"),
        litellm=LiteLLMConfig(timeout=10, num_retries=1),
        budget=BudgetConfig(default_limit_usd=100.0, window_seconds=3600),
        breaker=BreakerConfig(failure_threshold=3, recovery_seconds=30),
    )


def _make_call_fn(
    cheap_reply: str = "cheap good answer",
    expensive_reply: str = "expensive great answer",
    input_tokens: int = 100,
    output_tokens: int = 50,
):
    """Build a deterministic call_fn that returns different replies per tier.

    The cheap model gives ``cheap_reply``; the expensive gives
    ``expensive_reply``. Cost is computed from the tier's pricing.
    """
    cfg = _config()
    calc = CostCalculator(cfg)

    def call_fn(*, provider, tier, system, user_message, max_tokens=1024, temperature=0.0):
        if tier == "low":
            content = cheap_reply
        else:
            content = expensive_reply
        cost = calc.cost_usd(provider, tier, input_tokens, output_tokens)
        return GatewayReply(
            content=content,
            usage={"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                   "total_tokens": input_tokens + output_tokens},
            tier_used=tier,
            model_used=f"{provider}:{tier}",
            budget_after={},
            breaker_state_after="closed",
            latency_ms=10.0,
            call_id=f"test:{tier}",
            cost_usd=cost,
            est_cost_usd=cost,
        )
    return call_fn


# ---------------------------------------------------------------------------
# 1. Difficulty routing — saves money vs always-expensive, quality preserved.
# ---------------------------------------------------------------------------

def test_difficulty_routing_costs_less_than_always_expensive():
    """Mixed batch (3 easy + 2 hard) via difficulty routing costs less
    than sending all 5 to the expensive tier."""
    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn()
    router = CostAwareRouter(
        cfg, calc,
        classifier=KeywordDifficultyClassifier(hard_keywords=("complex", "analyze")),
        quality_checker=KeywordQualityChecker(),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )

    queries = [
        ("system", "hello"),                 # easy
        ("system", "what is 2+2"),           # easy
        ("system", "tell me a joke"),        # easy
        ("system", "analyze the data"),      # hard
        ("system", "complex derivation"),    # hard
    ]

    # Difficulty routing.
    routing_total = 0.0
    routing_tiers = []
    for sys_msg, user_msg in queries:
        result = router.route(system=sys_msg, user_message=user_msg)
        routing_total += result.total_cost_usd
        routing_tiers.append(result.reply.tier_used)

    # Always-expensive baseline.
    expensive_total = 0.0
    for sys_msg, user_msg in queries:
        reply = call_fn(provider="p", tier="high", system=sys_msg, user_message=user_msg)
        expensive_total += reply.cost_usd

    assert routing_total < expensive_total, (
        f"routing (${routing_total:.6f}) should be cheaper than "
        f"always-expensive (${expensive_total:.6f})"
    )

    # Hard queries must still get the expensive tier (quality preserved).
    hard_results = [
        router.route(system=s, user_message=u)
        for s, u in queries if "complex" in u or "analyze" in u
    ]
    for r in hard_results:
        assert r.reply.tier_used == "high", (
            f"hard query got tier={r.reply.tier_used}, expected 'high' (quality preserved)"
        )


def test_mutation_always_hard_no_savings():
    """MUTATION: classifier always returns 'hard'. Every query goes to
    expensive tier. Total cost == always-expensive baseline."""
    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn()

    class AlwaysHard(DifficultyClassifier):
        def classify(self, system, user_message):
            return "hard"

    router = CostAwareRouter(
        cfg, calc,
        classifier=AlwaysHard(),
        quality_checker=KeywordQualityChecker(),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )
    queries = [("s", "hello"), ("s", "what is 2+2"), ("s", "joke")]

    routing_total = sum(
        router.route(system=s, user_message=u).total_cost_usd
        for s, u in queries
    )
    expensive_total = sum(
        call_fn(provider="p", tier="high", system=s, user_message=u).cost_usd
        for s, u in queries
    )
    assert routing_total == expensive_total, (
        f"always-hard mutation: routing (${routing_total}) should equal "
        f"always-expensive (${expensive_total})"
    )
    # The load-bearing "routing < expensive" assertion would FAIL here.


def test_mutation_always_easy_breaks_quality():
    """MUTATION: classifier always returns 'easy'. Hard queries go to
    cheap tier — quality not preserved (tier != 'high' for hard queries)."""
    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn()

    class AlwaysEasy(DifficultyClassifier):
        def classify(self, system, user_message):
            return "easy"

    router = CostAwareRouter(
        cfg, calc,
        classifier=AlwaysEasy(),
        quality_checker=KeywordQualityChecker(),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )

    # Hard query that the cheap model CAN handle (quality check passes) →
    # stays on cheap. The "hard → expensive" invariant is broken.
    result = router.route(system="s", user_message="analyze complex data")
    assert result.reply.tier_used == "low", (
        f"always-easy mutation: expected cheap tier, got {result.reply.tier_used}"
    )
    # The load-bearing "hard → high" assertion would FAIL here.


# ---------------------------------------------------------------------------
# 2. Cascade — cheap-sufficient saves money, cheap-insufficient escalates.
# ---------------------------------------------------------------------------

def test_cascade_cheap_sufficient_stays_cheap():
    """Cheap model gives good answer → no escalation, 1 call, low cost."""
    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn(cheap_reply="good answer", expensive_reply="great answer")
    router = CostAwareRouter(
        cfg, calc,
        classifier=KeywordDifficultyClassifier(),
        quality_checker=KeywordQualityChecker(failure_markers=("BAD_REPLY",)),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )

    result = router.route(system="s", user_message="hello")
    assert result.escalated is False
    assert result.tiers_tried == ["low"]
    assert len(result.tiers_tried) == 1
    assert result.total_cost_usd == result.reply.cost_usd


def test_cascade_cheap_insufficient_escalates():
    """Cheap model gives bad answer → escalate to expensive, get good answer."""
    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn(
        cheap_reply="BAD_REPLY I cannot help",   # fails quality check
        expensive_reply="correct detailed answer",
    )
    router = CostAwareRouter(
        cfg, calc,
        classifier=KeywordDifficultyClassifier(),  # "hello" → easy
        quality_checker=KeywordQualityChecker(failure_markers=("BAD_REPLY", "I_CANNOT")),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )

    result = router.route(system="s", user_message="hello")
    assert result.escalated is True
    assert result.tiers_tried == ["low", "high"]
    assert result.reply.tier_used == "high"
    assert result.reply.content == "correct detailed answer"
    # Paid for both calls.
    assert result.total_cost_usd > result.reply.cost_usd


def test_mutation_no_escalate_breaks_quality():
    """MUTATION: quality checker always returns True → never escalates.
    Bad cheap reply is returned as-is → quality assertion fails."""
    cfg = _config()
    calc = CostCalculator(cfg)
    call_fn = _make_call_fn(
        cheap_reply="BAD_REPLY I cannot help",
        expensive_reply="correct answer",
    )

    class AlwaysPassQuality(QualityChecker):
        def check(self, reply_content, system, user_message):
            return True  # mutation: never escalate

    router = CostAwareRouter(
        cfg, calc,
        classifier=KeywordDifficultyClassifier(),
        quality_checker=AlwaysPassQuality(),
        call_fn=call_fn,
        cheap_tier="low", expensive_tier="high",
    )

    result = router.route(system="s", user_message="hello")
    assert result.escalated is False, "mutation: should not escalate"
    assert result.reply.content == "BAD_REPLY I cannot help", (
        "mutation: bad cheap reply returned without escalation"
    )
    # The load-bearing "escalated + correct answer" assertion would FAIL here.
