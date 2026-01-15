# Evaluator

The evaluator executes benchmarks for each miner submission. It is responsible for spawning rollout workers, connecting them to containers which run agents, and collecting metrics that stream directly back to the backend.

## Runtime Components

- **Rollout cluster** - `RolloutCluster` manages Ray actors and creates workers that can process multiple benchmark specs.
- **Benchmark specs** - Each evaluation job declares an environment provider, benchmark name, and config that turn into `BenchmarkSpec` objects.
- **Rollout worker** - Performs the actual environment interaction, calls the miner agent over RPC, tracks rewards, and records key statistics.
- **RPC bridge** - A lightweight process that proxies actions and observations between the worker and the submission container over TCP.
- **Episode logger** - Streams step data, episode summaries, and uploaded artifacts back to the backend via WebSocket.

## Episode Execution Flow

1. The orchestrator receives an `EvalJobMessage` from the backend via WebSocket and builds a `BenchmarkSpec`.
2. A rollout worker is created to execute the benchmark.
3. When the worker requests an action, it forwards the observation to the submission container via the RPC bridge.
4. The container invokes the miner-provided policy, returns actions, and the worker steps the environment.
5. The episode logger records rewards, success flags, uploaded observation images, and any extra metrics. Logging frequency is controlled by `episode_log_interval` and `step_log_interval`.
6. Episode completions trigger telemetry that streams directly to the backend over the WebSocket connection.
7. Once all benchmarks finish, the worker assembles aggregate metrics (success rate, average reward, total episodes) that feed into the orchestrator's `EvalResultMessage`.

## Storage and Artifacts

- **S3 Storage** - Image observations and other heavy artifacts upload through the logger's executor so the backend can serve signed URLs later.
- **Direct streaming** - Results and telemetry stream directly to the backend over WebSocket, eliminating intermediate queues.

## Configuration Highlights

Key settings in `evaluator.toml`:

- `backend_url` - WebSocket URL for the backend (e.g., `wss://api.kinitro.ai/ws/evaluator`).
- `api_key` - Authentication key for the evaluator.
- `episode_log_interval` and `step_log_interval` - Control how frequently detailed telemetry is streamed.
- `max_concurrent_jobs` - Guards resource usage when multiple evaluations run concurrently.
- `s3_config` - Bucket credentials used for artifact uploads.
- `ray_num_cpus`, `ray_num_gpus`, `ray_memory_gb` - Ray head resources.
- `worker_num_cpus`, `worker_num_gpus`, `worker_memory_gb` - Per-worker resource requests.

See the [orchestrator guide](orchestrator.md) for how jobs arrive and are scheduled.
