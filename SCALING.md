# SCALING.md — Scaling-shape paradigm for cost-aware-gateway

> No fake QPS. Every step is marked "measured here" or "architecture-only."

## The boundary chain

A cost-aware LLM gateway scales through a sequence of architectural boundaries.
At each boundary, a specific resource exhausts; the correct response is to
externalize state or add capacity, not to "optimize harder."

```
Single-process
    │
    ▼  (state/coordination force shared persistence)
Redis externalized (C1 + C2)
    │
    ▼  (more call volume than one process can handle)
2nd worker behind LB
    │
    ▼  (LB / worker TCP table exhausts)
TCP / connection-exhaustion levers
    │
    ▼  (ephemeral-port / connection limit reached)
DNS-level / serverless scaling cutover
```

### Step 1: Single-process

**What works:** one Python process, in-memory dict + lock for budget/breaker.
Fine for prototyping and single-user dev.

**What breaks:** the moment a second worker is added, each worker's breaker is
blind to the other's failures. A DoW attacker hitting worker-1 is invisible to
worker-2. The per-user budget counter sees only half the spend.

**Measurement:** this step is measured implicitly — the multi-instance proofs
(C1/C2) run against real Redis and demonstrate the gap when state is process-local.

### Step 2: Redis externalized shared state (C1 + C2)

**The fix:** budget + breaker state live in Redis. Multiple `BudgetTracker` and
`CircuitBreaker` instances pointed at the same Redis share counters and state.

**Atomicity mechanisms (measured here):**
- Budget reserve uses **Redis `WATCH` + `MULTI`/`EXEC`** (optimistic locking)
  that checks window expiry and `INCRBY` the reserved counter in one atomic
  round-trip. Concurrent reserve races cannot overspend.
- Budget reconcile uses **Redis `WATCH` + `MULTI`/`EXEC`** to adjust reserved vs
  spent in one atomic round-trip.
- We use `WATCH`/`MULTI`/`EXEC` instead of a Lua script because
  **fakeredis does not implement `EVAL`/`EVALSHA`**. The optimistic-locking
  retry loop is still atomic on real Redis and keeps the unit tests runnable
  in CI without a live Redis instance.
- Breaker half-open single-probe uses **Redis `SET ... NX`** (SETNX) to ensure
  only one worker across all instances can probe the half-open state.

**Why this is the correct boundary:** Redis is the cheapest correct answer for
shared mutable state across < 100 workers. It is measured here: the two-worker
demo in `tests/` and `python -m cost_aware_gateway.run --all` proves shared
state across OS processes.

### Step 3: 2nd worker behind a load balancer

**What works:** two (or more) gateway processes behind an L4/L7 LB. Each worker
is stateless except for the Redis-backed budget/breaker. The LB distributes
connections.

**What breaks:** at some connection count, the LB's TCP table or the worker's
ephemeral-port range exhausts.

**Measurement:** the two-worker shared-state demo IS this step (C1/C2 proofs
run as two OS processes). Beyond two workers, we enter architecture-only.

### Step 4: TCP / connection-exhaustion levers

These are kernel / LB knobs that push the TCP-exhaustion boundary further.

| Lever | What it does | Measured here? |
|-------|--------------|----------------|
| `net.ipv4.ip_local_port_range` | Widens the ephemeral-port range (default: 32768-60999, ~28k ports) | Architecture-only |
| `net.ipv4.tcp_tw_reuse` | Allows reusing `TIME_WAIT` sockets for new connections | Architecture-only |
| `SO_REUSEPORT` | Multiple workers can bind the same port; kernel load-balances | Architecture-only |
| LB max-conn | Caps concurrent connections per worker; sheds excess | Architecture-only |
| Keep-alive / HTTP/2 multiplexing | Reduces connection churn by reusing one TCP conn per client | Architecture-only |

**Honesty statement:** none of these have been load-tested in this repo. They
are named here because a staff engineer reading the repo must be able to answer
"what breaks next?" without guessing.

### Step 5: DNS-level / serverless scaling cutover

When even the TCP levers are exhausted, the next step is to stop thinking about
"workers" and start thinking about request-level scaling.

| Lever | What it does | Tradeoffs |
|-------|--------------|-----------|
| DNS TTL propagation | Cut over traffic by changing DNS | Propagation delay; stale caches |
| Serverless cold-start | Spin up on demand, no warm pool | Cold-start latency; connection draining |
| Connection draining | Graceful shutdown of draining workers | Needs LB support |

**Honesty statement:** serverless cold-start latency for LLM gateway workloads
is not measured here. The boundary chain documents where measurement stops and
architecture-reasoning begins.

## Model-serving backends (landscape awareness)

In production, `litellm.Router` routes to a **model-serving backend** serving
open-weight or proprietary models via OpenAI-compatible API:

- **vLLM** — high-throughput continuous batching, PagedAttention, OpenAI-compatible server.
- **SGLang** — RadixAttention, structured generation, fast decoding for long outputs.
- **Ollama** — local inference for dev/testing; no GPU cluster required.
- **Streaming responses** — the gateway passes through streaming; no buffering.
- **Batching / continuous batching** — the serving backend's throughput mechanism; gateway is backend-agnostic.
- **Semantic caching** — optional layer (GPTCache / similar) that short-circuits repeated semantically-identical requests. Mentioned, not built in v1.

## Why not in-memory + a load balancer?

A reader might ask: *"Why not just run in-memory dicts behind a load balancer?
That's what everyone does."*

The answer is the multi-instance correctness question:

1. **Budget correctness.** If worker-1 spends 600 tokens and worker-2 spends
   500 tokens, the shared limit is 1000. With in-memory dicts, each worker sees
   only its own spend. Neither blocks. The DoW attacker wins.

2. **Breaker correctness.** If worker-1 sees 3 failures from provider X and
   trips the breaker, worker-2 is still hammering X. The breaker's purpose
   (protect a degraded backend) is defeated.

3. **Redis is the cheapest correct answer.** Redis is a single-threaded,
   in-memory key-value store with atomic operations and persistence. It is
   faster than any SQL database for this workload and simpler than a consensus
   system (etcd, Consul). For < 100 workers, Redis is the industry-standard
   choice for shared mutable state — not because it's trendy, but because it's
   the correct tool for the job.

## CVE-2026-42271 — LiteLLM Supply-Chain Attack

LiteLLM versions **1.82.7 and 1.82.8** were backdoored on PyPI in March 2026.
The attack vector was a **CI/CD compromise** (stolen PyPI token via a compromised
Trivy GitHub Action), **NOT** an "MCP-injection." The backdoor exfiltrated SSH
keys, cloud credentials, and crypto wallets. First clean release: **v1.82.9**.

This repo pins `litellm>=1.82.9,<2.0`. See README security note and
[LiteLLM's incident report](https://docs.litellm.ai/blog/security-update-march-2026).
