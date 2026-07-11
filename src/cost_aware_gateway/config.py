"""Configuration loader for cost-aware-gateway.

Reads YAML config files; mode is determined by the `mode:` field in the
config (real / mock / ci), NOT by environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BudgetConfig:
    default_limit_tokens: int = 50_000
    window_seconds: float = 3600.0


@dataclass
class BreakerConfig:
    failure_threshold: int = 3
    recovery_seconds: float = 30.0


@dataclass
class RedisConfig:
    url: str = "redis://localhost:6379/0"


@dataclass
class LiteLLMConfig:
    timeout: int = 120
    num_retries: int = 2


@dataclass
class ProviderConfig:
    base_url: str
    api_key: str
    tiers: dict[str, dict[str, str]]


@dataclass
class JudgeConfig:
    provider: str
    tier: str


@dataclass
class GatewayConfig:
    mode: str
    default_provider: str
    providers: dict[str, ProviderConfig]
    judge: JudgeConfig
    redis: RedisConfig
    litellm: LiteLLMConfig
    budget: BudgetConfig
    breaker: BreakerConfig
    routed_model: str = "step-3.7-flash"
    config_path: Path = field(default_factory=lambda: Path("config.yaml"))


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> GatewayConfig:
    """Load config from a YAML file.

    The `mode` field in the YAML determines real/mock/ci behavior.
    No env-var switching is used.
    """
    if path is None:
        # Default: config.yaml (canonical) or config.ci.yaml (CI)
        cwd = Path.cwd()
        for candidate in ("config.yaml", "config.ci.yaml"):
            p = cwd / candidate
            if p.exists():
                path = p
                break
        if path is None:
            path = cwd / "config.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # Merge defaults from config.example.yaml if present
    example = Path("config.example.yaml")
    if example.exists():
        example_raw: dict[str, Any] = yaml.safe_load(example.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(example_raw, raw)

    # Parse providers
    providers_raw = raw.get("providers", {})
    providers: dict[str, ProviderConfig] = {}
    for name, cfg in providers_raw.items():
        providers[name] = ProviderConfig(
            base_url=cfg.get("base_url", ""),
            api_key=cfg.get("api_key", ""),
            tiers=cfg.get("tiers", {}),
        )

    # Parse judge
    judge_raw = raw.get("judge", {})
    judge = JudgeConfig(
        provider=judge_raw.get("provider", "minimax"),
        tier=judge_raw.get("tier", "medium"),
    )

    # Parse redis
    redis_raw = raw.get("redis", {})
    redis_cfg = RedisConfig(url=redis_raw.get("url", "redis://localhost:6379/0"))

    # Parse litellm
    litellm_raw = raw.get("litellm", {})
    litellm_cfg = LiteLLMConfig(
        timeout=int(litellm_raw.get("timeout", 120)),
        num_retries=int(litellm_raw.get("num_retries", 2)),
    )

    # Parse budget
    budget_raw = raw.get("budget", {})
    budget_cfg = BudgetConfig(
        default_limit_tokens=int(budget_raw.get("default_limit_tokens", 50_000)),
        window_seconds=float(budget_raw.get("window_seconds", 3600.0)),
    )

    # Parse breaker
    breaker_raw = raw.get("breaker", {})
    breaker_cfg = BreakerConfig(
        failure_threshold=int(breaker_raw.get("failure_threshold", 3)),
        recovery_seconds=float(breaker_raw.get("recovery_seconds", 30.0)),
    )

    return GatewayConfig(
        mode=str(raw.get("mode", "real")),
        default_provider=str(raw.get("default_provider", "minimax")),
        providers=providers,
        judge=judge,
        redis=redis_cfg,
        litellm=litellm_cfg,
        budget=budget_cfg,
        breaker=breaker_cfg,
        routed_model=str(raw.get("routed_model", "step-3.7-flash")),
        config_path=path,
    )
