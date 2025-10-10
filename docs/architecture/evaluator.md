# Evaluator

The evaluator executes benchmark episodes for each miner submission. It is responsible for spawning rollout workers, connecting them to containers which run agents, and collecting metrics that flow back to the backend through the validator.

## Runtime Components
- **Rollout cluster** – `RolloutCluster` manages Ray actors and creates workers that can process multiple benchmark specs.
- **Benchmark specs** – Each evaluation job declares an environment provider, benchmark name, and config that turn into `BenchmarkSpec` objects.
- **Rollout worker** – Performs the actual environment interaction, calls the miner agent over RPC, tracks rewards, and records key statistics.
- **RPC bridge** – A lightweight process that proxies actions and observations between the worker and the submission container over TCP.
- **Episode logger** – Streams step data, episode summaries, and uploaded artifacts to pgqueuer while handling retries and back-pressure.

## Episode Execution Flow
1. The orchestrator builds a `BenchmarkSpec` from the evaluation job and creates a rollout worker.
2. When the worker requests an action, it forwards the observation to the submission container via the RPC bridge.
3. The container invokes the miner-provided policy, returns actions, and the worker steps the environment.
4. The episode logger records rewards, success flags, uploaded observation images, and any extra metrics and throttles writes according to `episode_log_interval` and `step_log_interval`.
5. Episode completions trigger an `EpisodeDataMessage`; step-level logs trigger `EpisodeStepDataMessage`s. Both messages are persisted through pgqueuer for the validator to publish.
6. Once all benchmarks finish, the worker assembles aggregate metrics (success rate, average reward, total episodes) that feed into the orchestrator’s result payload.

## Storage and Artifacts
- **Cloudflare R2 / S3** – Image observations and other heavy artifacts upload through the logger’s executor so the backend can serve signed URLs later.
- **Validator database** – Results and telemetry are queued to the validator Postgres instance so that validator and backend connectivity issues do not drop data.

## Configuration Highlights
- `episode_log_interval` and `step_log_interval` control how frequently detailed telemetry is enqueued (`config/evaluator.toml.example`).
- `max_concurrent_jobs` guards resource usage when multiple evaluations are queued concurrently.
- `r2_config` includes bucket credentials used for artifact uploads.

See the [orchestrator guide](orchestrator.md) for how jobs arrive and are scheduled.
