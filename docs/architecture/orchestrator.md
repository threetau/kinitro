# Orchestrator

The evaluator orchestrator is the control plane that turns queued evaluation jobs into running rollouts. It listens to pgqueuer events, provisions isolated submission containers, wires up Ray workers, and makes sure results flow back to the backend through the validator.

## Responsibilities
- **Queue consumption** – `PgQueuer` watches the validator database and invokes the orchestrator whenever a new `add_job` event appears (`src/evaluator/orchestrator.py`).
- **Concurrency control** – Track active jobs and defer new work until there is capacity (`src/evaluator/orchestrator.py`).
- **Environment provisioning** – Spin up Kubernetes pods that host the miner submission and expose an RPC endpoint for workers (`src/evaluator/orchestrator.py`).
- **Worker orchestration** – Create Ray rollout workers, attach benchmark specs, and stream observations / rewards back (`src/evaluator/orchestrator.py`).
- **Result collation** – Collect evaluation summaries, persist them via `EvalResultMessage`, and queue them for validator delivery (`src/evaluator/orchestrator.py`).
- **Cleanup & recovery** – Tear down pods, close queues, and reclaim Ray resources even on failure (`src/evaluator/orchestrator.py`).

## Job Lifecycle
1. The validator enqueues a serialized `EvalJobMessage` when it receives a job from the backend (`src/validator/websocket_validator.py`).
2. `PgQueuer` triggers the orchestrator’s `process` handler with the job payload (`src/evaluator/orchestrator.py`).
3. The job is recorded in the validator database with `EvaluationStatus.STARTING`, guaranteeing visibility and retries (`src/evaluator/orchestrator.py`).
4. A submission container is created via the `Containers` helper, exposing a service the worker can reach over TCP (`src/evaluator/orchestrator.py`).
5. A `RolloutCluster` worker runs the benchmark episodes, talking to the submission container through an RPC bridge (`src/evaluator/orchestrator.py`).
6. Episode-level and step-level telemetry is queued by the `EpisodeLogger`, which uses the same pgqueuer channel so the validator can forward data to the backend (`src/evaluator/rollout/episode_logger.py`).
7. When the rollout completes, scores and aggregates are serialized into an `EvalResultMessage` and queued for delivery (`src/evaluator/orchestrator.py`).
8. Cleanup routines ensure pods are removed, Ray actors stopped, and lingering queues closed (`src/evaluator/orchestrator.py`).

## Resilience Features
- **Durable queues** – All commands and results flow through pgq tables, so reconnecting services pick up exactly where they left off.
- **Timeout handling** – The orchestrator checks elapsed time per job and can mark stale work for cleanup (`src/evaluator/orchestrator.py`).
- **Health monitoring** – Background tasks watch running jobs for completion or timeout signals and remove them when necessary (`src/evaluator/orchestrator.py`).

Refer to the [Evaluator internals](evaluator.md) for details on the worker side of this pipeline.
