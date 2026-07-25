"""cost_aware_gateway — Redis-backed cost-aware LLM gateway.

Components:
  C1  BudgetTracker   — per-user dollar budget with two-step reserve-then-reconcile
  C2  CircuitBreaker  — 3-state breaker with Redis backing + SETNX single-probe
  C3  TieredRouter    — LiteLLM Router declarative multi-model routing
  C5  Redactor        — AgentLeak runtime credential redaction
  M1  CostCalculator  — Per-model dollar cost from config-driven pricing
"""

from cost_aware_gateway.config import load_config
from cost_aware_gateway.models import (
    BudgetExceededError,
    CircuitState,
    GatewayReply,
    RedactionResult,
)
from cost_aware_gateway.budget import BudgetTracker
from cost_aware_gateway.breaker import CircuitBreaker
from cost_aware_gateway.cost import CostCalculator
from cost_aware_gateway.redact import redact_tool_output

# TieredRouter requires litellm which may not be installable in all
# environments (Rust toolchain needed for some transitive deps).
# Import lazily so the rest of the package works without it.
try:
    from cost_aware_gateway.router import TieredRouter
except ImportError:
    TieredRouter = None  # type: ignore[assignment,misc]

__all__ = [
    "load_config",
    "BudgetExceededError",
    "CircuitState",
    "GatewayReply",
    "RedactionResult",
    "BudgetTracker",
    "CircuitBreaker",
    "CostCalculator",
    "TieredRouter",
    "redact_tool_output",
]
