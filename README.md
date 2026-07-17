# cost-aware-gateway

A **Redis-backed cost-aware LLM gateway** with per-user budget, a Redis-backed
three-state circuit breaker, LiteLLM Router declarative multi-model routing,
and AgentLeak runtime credential redaction.

**What it does:** keeps agent spend under control at scale and defends against
Denial-of-Wallet by making budget + breaker state shared across workers (Redis,
not process-local), and by reasoning about the scaling-shape boundary chain in
architecture terms, not fake QPS.

---

## Architecture

```
Caller
  │
  ▼
TieredRouter (C3 — litellm.Router)
  │  resolves tier → model
  │
  ├─► BudgetTracker (C1 — Redis-backed reserve-then-reconcile)
  │     pre-call reserve, post-call reconcile
  │
  ├─► CircuitBreaker (C2 — hand-rolled Redis 3-state + SETNX half-open lock)
  │     fail-fast on degraded provider
  │
  ▼
litellm.Router.completion(...)
  │
  ▼
Redactor (C5 — AgentLeak runtime redaction)
  strip credentials from tool outputs before model context
```

## Components

| ID | Component | What it solves |
|----|-----------|----------------|
| C1 | `BudgetTracker` | Per-user token budget; two-step reserve-then-reconcile so we reject BEFORE spending money. State in Redis so multiple workers share counters. |
| C2 | `CircuitBreaker` | Hand-rolled 3-state breaker (closed/open/half-open) with Redis-aggregated failure counts. Half-open single-probe serialized via `SETNX`. |
| C3 | `TieredRouter` | Declarative tier-based routing via `litellm.Router`. Callers pick tier, not model id. |
| C4 | `SCALING.md` | Boundary-chain reasoning: single-process → Redis → LB → TCP exhaustion → DNS/serverless. No fake QPS. |
| C5 | `Redactor` | Runtime-internal credential redaction (AgentLeak). Strips configured API keys and known credential shapes from tool outputs before they enter model context. |

## Setup

```bash
# 1. Create conda env (mandatory)
conda create -n cost-aware-gateway python=3.11 -y
conda activate cost-aware-gateway

# 2. Install deps
pip install -e ".[dev]"

# 3. Start Redis
docker compose up -d

# 4. Configure keys
cp config.example.yaml config.yaml
# edit config.yaml with real API keys
```

## Running

```bash
# Run all demos (smoke test)
python -m cost_aware_gateway.run --all

# Run individual demos
python -m cost_aware_gateway.run --budget
python -m cost_aware_gateway.run --breaker
python -m cost_aware_gateway.run --router
python -m cost_aware_gateway.run --redact
```

## Tests

```bash
pytest tests/ -v
```

## Security Note — LiteLLM Supply-Chain Attack (CVE-2026-42271)

LiteLLM versions **1.82.7 and 1.82.8** were backdoored on PyPI in March 2026
(TeamPCP). The attack vector was a **CI/CD supply-chain compromise**: a stolen
PyPI token obtained via a compromised Trivy GitHub Action in LiteLLM's CI
pipeline — **NOT** an "MCP-injection". The backdoor exfiltrated SSH keys, cloud
credentials, and crypto wallets from every Python process that imported those
versions.

This repo pins **`litellm>=1.82.9,<2.0`** (first clean release: **v1.82.9**).
See `pyproject.toml` for the pin and [LiteLLM's security update](https://docs.litellm.ai/blog/security-update-march-2026) for the full incident report.

## Architecture Decisions

### Why WATCH/MULTI/EXEC for budget instead of Lua script?

The budget tracker uses Redis `WATCH` + `MULTI`/`EXEC` transactions instead of
a Lua script. The reason is **testability with fakeredis**: fakeredis does not
implement `EVAL`/`EVALSHA`, so a Lua-based implementation would make the unit
tests impossible to run in CI without a real Redis. `WATCH`/`MULTI`/`EXEC` is
still atomic on real Redis, and the optimistic-locking retry loop is a standard
pattern for this kind of shared counter. We prove atomicity with a real-Redis
concurrency stress test (`tests/test_budget.py::TestRealRedisBudget`).

### Why a hand-rolled Redis 3-state breaker?

The breaker implements the classic three-state machine (closed → open after N
failures → half-open after recovery_seconds → close on probe success / re-open
on probe failure) directly on Redis. Every `record_failure` does an atomic
`INCR` on the Redis failure counter; when it reaches `failure_threshold`, the
state is set to OPEN in Redis. This makes the breaker genuinely cross-worker:
3 workers each failing once will trip a threshold=3 breaker because Redis sees
count=3.

The half-open single-probe invariant (only one worker probes when recovering)
is enforced with a Redis `SET ... NX` lock — the cheapest correct answer for
cross-process mutual exclusion.

### Why not just run LiteLLM Proxy directly?

LiteLLM Proxy ships with built-in budget limits and rate limiting, but those
are **post-hoc, aggregated metrics**. This repo's differentiator is the
**application-layer two-step reserve-then-reconcile protocol**: we reserve an
estimate *before* the call, then reconcile with the actual usage *after* the
call. That lets us reject a request before spending money, which is the core
Denial-of-Wallet defense. Using LiteLLM Proxy's off-the-shelf limiter would
hide that mechanism behind a config flag and make it impossible to demo the
reserve/reconcile flow explicitly.

### Why doesn't SCALING.md give QPS numbers?

We have not performed load testing in this repository. `SCALING.md` documents
the **architecture boundary chain** (single-process → Redis → LB → TCP
exhaustion → serverless) and marks each step as either "measured here" or
"architecture-only." Inventing QPS figures would be dishonest; a staff engineer
reading the repo must be able to answer "what breaks next?" without guessing.

## Scaling-Shape Paradigm

See **`SCALING.md`** for the boundary-chain reasoning: single-process → Redis
externalized shared state → 2nd worker behind LB → TCP/connection-exhaustion
levers (`TIME_WAIT`, `SO_REUSEPORT`, `net.ipv4.ip_local_port_range`) →
DNS/serverless cutover. No fake QPS; every step marked "measured here" vs
"architecture-only."

## Keywords (§6.5 coverage)

| Keyword | Where |
|---------|-------|
| LiteLLM Router | C3 — `TieredRouter` uses `litellm.Router` |
| Portkey | README — cited as production gateway with budget limits |
| OpenRouter | README — cited as managed aggregator |
| Cloudflare AI Gateway | README — cited as edge alternative |
| pybreaker | (removed R4) — was a name-drop with dead decision logic; breaker is now hand-rolled on Redis |
| Redis | C1/C2 — state backend for budget + breaker |
| Denial-of-Wallet | README + C1 — DoW defense via per-user budget |
| hierarchical budget controls | README + SCALING.md — FINOS DoW vocabulary |
| SETNX | C2 — half-open single-probe serialization |
| TIME_WAIT / SO_REUSEPORT | SCALING.md — TCP exhaustion levers |
| vLLM / SGLang | SCALING.md — model-serving backends |
| semantic caching | SCALING.md — optional gateway layer |
| AgentLeak | C5 — runtime credential redaction |

## License

MIT
