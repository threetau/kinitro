# Orchestrator

The evaluator orchestrator is the control plane that turns queued evaluation jobs into running rollouts. It listens to pgqueuer events, provisions isolated submission containers, wires up Ray workers, and makes sure results flow back to the backend through the validator.

## Responsibilities
- **Queue consumption** – `PgQueuer` watches the validator database and invokes the orchestrator whenever a new `add_job` event appears.
- **Concurrency control** – Track active jobs and defer new work until there is capacity.
- **Environment provisioning** – Spin up Kubernetes pods that host the miner submission and expose an RPC endpoint for workers.
- **Worker orchestration** – Create Ray rollout workers, attach benchmark specs, and stream observations and other metrics back.
- **Result collation** – Collect evaluation summaries, persist them via `EvalResultMessage`, and queue them for validator delivery.
- **Cleanup & recovery** – Tear down pods, close queues, and reclaim Ray resources even on failure.

## Job Lifecycle
1. The validator enqueues a job from the backend.
2. `PgQueuer` triggers the orchestrator’s `process` handler with the job payload.
3. The job is recorded in the validator database with `EvaluationStatus.STARTING`, guaranteeing visibility and retries.
4. A submission container is created via the `Containers` helper, exposing a service the worker can reach over TCP.
5. A `RolloutCluster` worker runs the benchmark episodes, talking to the submission container through an RPC bridge.
6. Episode-level and step-level telemetry is queued by the `EpisodeLogger`, which uses the same pgqueuer channel so the validator can forward data to the backend.
7. When the rollout completes, scores and aggregates queued for delivery.
8. Cleanup routines ensure pods are removed, Ray actors stopped, and lingering queues closed.

## Resilience Features
- **Durable queues** – All commands and results flow through pgq tables, so reconnecting services pick up exactly where they left off.
- **Timeout handling** – The orchestrator checks elapsed time per job and can mark stale work for cleanup.
- **Health monitoring** – Background tasks watch running jobs for completion or timeout signals and remove them when necessary.

Refer to the [Evaluator internals](evaluator.md) for details on the worker side of this pipeline.
