---
section: Core Concepts
---

# Incentive Mechanism

Kinitro distributes incentives by translating evaluation outcomes into miner scores and network weights. The backend service owns the full loop: it aggregates results, applies competition rules, and broadcasts weight updates to validators so they can call `set_node_weights` on Bittensor.

## Scoring Pipeline
1. **Aggregate results** – Evaluation metrics land in the backend database.
2. **Eligibility checks** – Each competition declares minimum average reward, minimum success rate, and win margins. The backend filters results against those thresholds before considering them for scoring.
3. **Leader updates** – When a miner beats the active leader by the configured margin, the backend records the new leader and timestamp.
4. **Score normalization** –  Winning submissions are converted into normalized scores so that weight broadcasts sum to one.
5. **Schedule** – Periodic scoring is performed on a configurable cadence and caches the latest scores for broadcasting.

## Weight Broadcasting
1. **WebSocket broadcast** – The backend pushes weight messages to every connected validator.
2. **Chain update** – Validators commit weights on the Bittensor chain.
3. **Cache warmup** – The backend keeps a copy of the latest weights in memory so reconnecting validators receive immediate updates even before the next scoring cycle.

## Configuration
- `score_evaluation_interval` and `weight_broadcast_interval` can be tuned per deployment through the backend configuration.
- Default thresholds for minimum average reward, minimum success rate, and win margin ship with safe baselines, but each competition can override them.
- Competitions award `points`, which can be used to weight scores across multiple benchmarks during aggregation.

## Operational Considerations
- Scores depend on trusted evaluation results. Validators should ensure evaluators are running the same container images and configuration to avoid inconsistent outcomes.
- When no miner satisfies a competition’s success criteria, the backend skips weight updates for that competition and previously broadcast weights decay to zero.
- Validators without an active API key will fail to receive weight updates.
