"""CLI entrypoint for cost-aware-gateway.

Usage:
  python -m cost_aware_gateway.run --all          # run all demos
  python -m cost_aware_gateway.run --budget       # C1 only
  python -m cost_aware_gateway.run --breaker      # C2 only
  python -m cost_aware_gateway.run --router       # C3 only
  python -m cost_aware_gateway.run --redact       # C5 only
  python -m cost_aware_gateway.run --config <path>
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

from cost_aware_gateway.config import load_config, GatewayConfig
from cost_aware_gateway.budget import BudgetTracker
from cost_aware_gateway.breaker import CircuitBreaker
from cost_aware_gateway.router import TieredRouter
from cost_aware_gateway.redact import redact_tool_output
from cost_aware_gateway.models import CircuitState, GatewayReply

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _print_pass(name: str, detail: str) -> None:
    print(f"  [PASS] {name}: {detail}")


def _print_fail(name: str, detail: str) -> None:
    print(f"  [FAIL] {name}: {detail}")


def demo_C1_budget(config: GatewayConfig) -> bool:
    """C1 — two BudgetTrackers share Redis state; concurrent reserve is atomic."""
    try:
        budget = BudgetTracker(
            redis_url=config.redis.url,
            default_limit=config.budget.default_limit_tokens,
            default_window=config.budget.window_seconds,
        )
        budget.reset("demo-user")
        budget.set_budget("demo-user", limit_tokens=1000)

        budget.check_and_reserve("demo-user", 600)
        _print_pass("C1 reserve-1", "reserved 600 tokens")

        try:
            budget.check_and_reserve("demo-user", 500)
            _print_fail("C1 reserve-2", "should have exceeded limit")
            return False
        except Exception:
            _print_pass("C1 reserve-2", "correctly blocked second reserve (limit=1000)")

        # Second tracker sees the same state
        budget2 = BudgetTracker(
            redis_url=config.redis.url,
            default_limit=config.budget.default_limit_tokens,
            default_window=config.budget.window_seconds,
        )
        status = budget2.status("demo-user")
        if status and status.get("reserved", 0) >= 600:
            _print_pass("C1 shared-state", f"second tracker sees reserved={status['reserved']}")
        else:
            _print_fail("C1 shared-state", "second tracker did not see first reserve")
            return False

        # Record actual and check reconcile
        budget.record_actual("demo-user", 600, 300)
        status2 = budget.status("demo-user")
        if status2 and status2.get("spent", 0) == 300:
            _print_pass("C1 reconcile", f"spent updated to {status2['spent']}")
        else:
            _print_fail("C1 reconcile", f"spent={status2}")
            return False

        budget.reset("demo-user")
        return True
    except Exception as exc:
        _print_fail("C1", f"Redis connection or operation failed: {exc}")
        return False


def demo_C2_breaker(config: GatewayConfig) -> bool:
    """C2 — two CircuitBreakers share Redis state; SETNX half-open lock."""
    try:
        b1 = CircuitBreaker(
            name="demo-provider",
            redis_url=config.redis.url,
            failure_threshold=config.breaker.failure_threshold,
            recovery_seconds=config.breaker.recovery_seconds,
        )
        b1.reset()
        b2 = CircuitBreaker(
            name="demo-provider",
            redis_url=config.redis.url,
            failure_threshold=config.breaker.failure_threshold,
            recovery_seconds=config.breaker.recovery_seconds,
        )

        # Trip breaker via b1
        for _ in range(config.breaker.failure_threshold):
            b1.record_failure()

        if b1.state == CircuitState.OPEN:
            _print_pass("C2 trip", "b1 tripped breaker OPEN")
        else:
            _print_fail("C2 trip", f"b1 state={b1.state}")
            return False

        # b2 should see the same state
        if b2.state == CircuitState.OPEN:
            _print_pass("C2 shared-state", "b2 sees OPEN")
        else:
            _print_fail("C2 shared-state", f"b2 state={b2.state}")
            return False

        # b2.allow() should be False while open
        if not b2.allow():
            _print_pass("C2 allow-blocked", "b2 blocks call while OPEN")
        else:
            _print_fail("C2 allow-blocked", "b2 allowed call while OPEN")
            return False

        b1.reset()
        return True
    except Exception as exc:
        _print_fail("C2", f"Redis connection or operation failed: {exc}")
        return False


def demo_C3_router(config: GatewayConfig) -> bool:
    """C3 — LiteLLM Router routes a real LLM call."""
    if config.mode == "mock":
        _print_pass("C3 router (mock)", "Router instantiated in mock mode")
        return True

    try:
        budget = BudgetTracker(
            redis_url=config.redis.url,
            default_limit=config.budget.default_limit_tokens,
            default_window=config.budget.window_seconds,
        )
        breakers = {
            name: CircuitBreaker(
                name=f"provider:{name}",
                redis_url=config.redis.url,
                failure_threshold=config.breaker.failure_threshold,
                recovery_seconds=config.breaker.recovery_seconds,
            )
            for name in config.providers
        }
    except Exception as exc:
        _print_fail("C3", f"Redis connection failed: {exc}")
        return False

    router = TieredRouter(config, budget, breakers)

    try:
        reply = router.complete(
            user_id="demo-user",
            system="You are a helpful assistant. Reply in one short sentence.",
            user_message="What is 2+2?",
            tier="medium",
            max_tokens=64,
        )
        _print_pass(
            "C3 router",
            f"tier={reply.tier_used} model={reply.model_used} latency={reply.latency_ms}ms",
        )
        return True
    except Exception as exc:
        _print_fail("C3 router", str(exc))
        return False


def demo_C5_redaction(config: GatewayConfig) -> bool:
    """C5 — configured API key is redacted from tool output."""
    provider = config.providers.get(config.default_provider)
    if not provider:
        _print_fail("C5 redaction", "no default provider configured")
        return False

    api_key = provider.api_key
    if not api_key or api_key == "FILL_IN_CONFIG_YAML":
        api_key = "sk-abcdef1234567890abcdef1234567890"

    payload = f"Here is your key: {api_key}. Use it wisely."
    result = redact_tool_output(payload, sensitive={api_key})

    if api_key not in result.cleaned_text:
        _print_pass("C5 redaction", "configured key stripped from output")
    else:
        _print_fail("C5 redaction", "key still present in output")
        return False

    if "[REDACTED]" in result.cleaned_text:
        _print_pass("C5 placeholder", "redaction placeholder present")
    else:
        _print_fail("C5 placeholder", "no redaction placeholder found")
        return False

    # Clean output should pass through unchanged
    clean = "Hello, world!"
    clean_result = redact_tool_output(clean, sensitive={api_key})
    if clean_result.cleaned_text == clean:
        _print_pass("C5 passthrough", "clean output unchanged")
    else:
        _print_fail("C5 passthrough", "clean output was modified")
        return False

    return True


def run_all(config_path: str | None = None) -> int:
    config = load_config(config_path)
    results: dict[str, bool] = {}

    print("=" * 60)
    print(f" cost-aware-gateway demos (mode={config.mode})")
    print("=" * 60)

    results["C1"] = demo_C1_budget(config)
    results["C2"] = demo_C2_breaker(config)
    results["C3"] = demo_C3_router(config)
    results["C5"] = demo_C5_redaction(config)

    print("-" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print("-" * 60)
    if all_pass:
        print("All demos passed.")
        return 0
    else:
        print("Some demos FAILED.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="cost-aware-gateway demo runner")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--budget", action="store_true", help="Run C1 budget demo")
    parser.add_argument("--breaker", action="store_true", help="Run C2 breaker demo")
    parser.add_argument("--router", action="store_true", help="Run C3 router demo")
    parser.add_argument("--redact", action="store_true", help="Run C5 redaction demo")
    args = parser.parse_args()

    config = load_config(args.config)

    # If no specific demo selected, default to --all
    if not any([args.all, args.budget, args.breaker, args.router, args.redact]):
        args.all = True

    results: dict[str, bool] = {}
    if args.all or args.budget:
        results["C1"] = demo_C1_budget(config)
    if args.all or args.breaker:
        results["C2"] = demo_C2_breaker(config)
    if args.all or args.router:
        results["C3"] = demo_C3_router(config)
    if args.all or args.redact:
        results["C5"] = demo_C5_redaction(config)

    print("-" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("-" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
