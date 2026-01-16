# Kinitro Development Guide

Kinitro is a Bittensor subnet for reinforcement learning evaluation. Miners submit trained agents, validators evaluate them against standardized benchmarks, and the network rewards top performers.

## Quick Reference

```bash
# Activate virtual environment
cd /root/dev/kinitro && source .venv/bin/activate

# Run tests
python3 -m pytest src/backend/tests/ src/backend/scoring/tests/ src/evaluator/executors/tests/ src/evaluator/providers/tests/ -v

# Run specific component
python3 -m backend          # Start backend service
python3 -m validator        # Start validator node
python3 -m evaluator        # Start evaluator (requires Ray + Kubernetes)
```

## Architecture Overview

```
                                    ┌─────────────────────┐
                                    │   Bittensor Chain   │
                                    │   (Commitments +    │
                                    │    Weights)         │
                                    └──────────┬──────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
           ┌────────────────┐         ┌────────────────┐         ┌────────────────┐
           │     Miner      │         │    Backend     │         │   Validator    │
           │  (Submission)  │         │   (FastAPI)    │         │(Polls /weights)│
           └───────┬────────┘         └───────┬────────┘         └───────┬────────┘
                   │                          │                          │
                   │   upload artifact        │   GET /weights           │
                   └─────────────────────────►│◄─────────────────────────┘
                                              │
                                              │   WebSocket (direct)
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   Evaluator     │
                                    │   (Ray + K8s)   │
                                    └───────┬─────────┘
                                            │
                                            │  RPC
                                            ▼
                                   ┌────────────────┐
                                   │  Miner Pod     │
                                   │  (Agent Code)  │
                                   └────────────────┘
```

**Key Points:**
- **Evaluators connect directly** to backend via WebSocket (no pgqueuer relay)
- **Validators poll** the `/weights` endpoint and set weights on chain (no WebSocket)
- **Simplified architecture** with fewer moving parts

## Directory Structure

```
src/
├── backend/           # Central orchestration service
│   ├── service.py         # Main BackendService (composition pattern)
│   ├── scoring_engine.py  # Eligibility + scoring + weights
│   ├── scoring/           # Pluggable scoring strategies
│   │   ├── strategies.py      # ScoringStrategy protocol + RLRolloutScoringStrategy
│   │   └── registry.py        # ScoringStrategyRegistry
│   ├── evaluator_hub.py   # Direct evaluator WebSocket connections
│   ├── chain_monitor.py   # Bittensor commitment scanning
│   ├── job_scheduler.py   # Job creation + broadcasting to evaluators
│   ├── endpoints.py       # FastAPI routes
│   └── models.py          # SQLModel database models
│
├── validator/         # Weight-setting service (polls backend)
│   ├── websocket_validator.py  # Polls /weights, sets on chain
│   └── db/                     # Local state cache
│
├── evaluator/         # Job execution engine (connects to backend)
│   ├── orchestrator.py    # Main job lifecycle management
│   ├── backend_client.py  # WebSocket client for backend connection
│   ├── executors/         # Task type implementations
│   │   ├── registry.py        # ExecutorRegistry
│   │   └── rl_rollout.py      # RLRolloutExecutor
│   ├── providers/         # Environment providers
│   │   ├── registry.py        # ProviderRegistry
│   │   ├── metaworld_provider.py
│   │   └── swarm_provider.py
│   ├── rollout/           # Ray actors for RL execution
│   ├── containers/        # Kubernetes pod management
│   └── rpc/               # Cap'n Proto agent communication
│
├── core/              # Shared modules
│   ├── tasks.py           # TaskSpec, TaskResult, TaskExecutor protocol
│   ├── messages.py        # WebSocket message types
│   ├── db/models.py       # SnowflakeId, EvaluationStatus
│   └── chain.py           # Bittensor helpers
│
└── miner/             # Submission CLI
    └── commands/          # upload, commit, local-eval
```

## Key Abstractions

### Task Abstraction Layer (`src/core/tasks.py`)

Universal interface for different task types (RL rollouts, training, browser tasks, etc.):

```python
class TaskType(StrEnum):
    RL_ROLLOUT = "rl_rollout"
    # Future: TRAINING_RUN, BROWSER_TASK, DATASET_EVAL

@dataclass
class TaskSpec:
    task_type: TaskType
    task_id: str
    config: dict[str, Any]
    timeout: timedelta
    resources: ResourceSpec
    submission_id: int
    competition_id: str
    artifact_url: str

@dataclass
class TaskResult:
    task_id: str
    success: bool
    metrics: dict[str, float]  # Task-specific metrics
    artifacts: dict[str, str]  # name -> S3 key

class TaskExecutor(Protocol):
    @property
    def task_type(self) -> TaskType: ...
    async def validate_spec(self, spec: TaskSpec) -> list[str]: ...
    async def setup(self, spec: TaskSpec) -> TaskContext: ...
    async def execute(self, context: TaskContext) -> TaskResult: ...
    async def teardown(self, context: TaskContext) -> None: ...
```

### Executor Registry (`src/evaluator/executors/registry.py`)

```python
# Register executors
ExecutorRegistry.register(RLRolloutExecutor(config))

# Dispatch by task type
executor = ExecutorRegistry.get(TaskType.RL_ROLLOUT)
context = await executor.setup(spec)
result = await executor.execute(context)
```

### Scoring Strategies (`src/backend/scoring/`)

```python
class ScoringStrategy(Protocol):
    @property
    def task_type(self) -> TaskType: ...
    def extract_metrics(self, result) -> ScoringMetrics: ...
    def check_eligibility(self, metrics, competition) -> EligibilityResult: ...
    def compute_score(self, metrics, competition) -> float: ...
    def compare(self, a, b) -> int: ...  # For ranking

# Usage in ScoringEngine
strategy = self.strategy_registry.get(competition.task_type)
metrics = strategy.extract_metrics(result)
if strategy.check_eligibility(metrics, competition).eligible:
    score = strategy.compute_score(metrics, competition)
```

### Provider Registry (`src/evaluator/providers/registry.py`)

```python
class EnvironmentProvider(Protocol):
    @property
    def name(self) -> str: ...
    def get_benchmark_specs(self, config) -> list[BenchmarkSpec]: ...
    def get_env_specs(self, benchmark_spec) -> list[EnvSpec]: ...
    def make_env(self, spec) -> gym.Env: ...

# Built-in providers
ProviderRegistry.register(MetaWorldProvider())
ProviderRegistry.register(SwarmProvider())
```

## Database Models

### Backend (`src/backend/models.py`)

```
Competition
├── id, name, description
├── benchmarks: JSON (list of benchmark specs)
├── points: int (weight allocation)
├── task_type: str (default "rl_rollout")
├── min_success_rate, min_avg_reward (thresholds)
├── current_leader_hotkey, current_leader_reward
└── Relationships: submissions, evaluation_jobs, leader_candidates

MinerSubmission
├── miner_hotkey, competition_id
├── hf_repo_id, version, commitment_block
├── artifact_object_key, artifact_sha256
├── holdout_release_at, released_at
└── Relationships: evaluation_jobs

BackendEvaluationJob
├── submission_id, competition_id
├── env_provider, benchmark_name, config
└── Relationships: results, status_updates

BackendEvaluationResult
├── job_id, validator_hotkey
├── score, success_rate, avg_reward, total_episodes
└── Relationships: leader_candidates

CompetitionLeaderCandidate
├── competition_id, miner_hotkey, evaluation_result_id
├── status: pending | approved | rejected
└── reviewed_by_api_key_id, reviewed_at
```

## Message Types (`src/core/messages.py`)

### Backend → Evaluator (via WebSocket)
- `EvalJobMessage`: New evaluation job (includes task_type, task_spec)
- `HeartbeatAckMessage`, `RegistrationAckMessage`, `ResultAckMessage`

### Evaluator → Backend (via WebSocket)
- `EvaluatorRegisterMessage`: Registration with API key and capabilities
- `HeartbeatMessage`: Keepalive
- `EvalResultMessage`: Completed evaluation with metrics
- `JobStatusUpdateMessage`: Status transitions

### Backend → Validator (via REST API)
- `GET /weights`: Returns `WeightsSnapshot` with UID→weight mapping

## Communication Patterns

### WebSocket (Backend ↔ Evaluator)
- Direct connection for job dispatch and result collection
- Evaluators register with supported task types
- Jobs broadcast to all connected evaluators

### HTTP Polling (Validator → Backend)
- Validator polls `GET /weights` periodically (default: 5 min)
- Sets weights on Bittensor chain when changed
- Simple polling avoids WebSocket complexity

### RPC (Evaluator ↔ Miner Pod)
- Cap'n Proto schema (`agent.capnp`)
- Methods: `reset()`, `act(observation) -> action`
- Avoids pickle for security

## Scoring Logic

1. **Eligibility**: Result must meet `min_success_rate` and `min_avg_reward`
2. **Leader Candidates**: Eligible results that beat current leader are queued for admin review
3. **Admin Approval**: Candidates require manual approval to become leader
4. **Winner-Takes-All**: One winner per competition gets points
5. **Burn Mechanism**: Configurable burn_pct (default 98%), remainder to owner UID
6. **Weight Calculation**: Normalized scores across all competitions

## Adding New Task Types

### 1. Define TaskType
```python
# src/core/tasks.py
class TaskType(StrEnum):
    RL_ROLLOUT = "rl_rollout"
    TRAINING_RUN = "training_run"  # Add new type
```

### 2. Implement TaskExecutor
```python
# src/evaluator/executors/training_executor.py
class TrainingExecutor:
    task_type = TaskType.TRAINING_RUN
    
    async def validate_spec(self, spec: TaskSpec) -> list[str]: ...
    async def setup(self, spec: TaskSpec) -> TaskContext: ...
    async def execute(self, context: TaskContext) -> TaskResult: ...
    async def teardown(self, context: TaskContext) -> None: ...
```

### 3. Register Executor
```python
# src/evaluator/executors/registry.py
ExecutorRegistry.register(TrainingExecutor(config))
```

### 4. Implement ScoringStrategy
```python
# src/backend/scoring/strategies.py
class TrainingScoringStrategy:
    task_type = TaskType.TRAINING_RUN
    
    def extract_metrics(self, result) -> ScoringMetrics: ...
    def check_eligibility(self, metrics, competition) -> EligibilityResult: ...
    def compute_score(self, metrics, competition) -> float: ...
    def compare(self, a, b) -> int: ...
```

### 5. Register Strategy
```python
# src/backend/scoring/registry.py
ScoringStrategyRegistry.register(TrainingScoringStrategy())
```

## Testing

```bash
# All tests
python3 -m pytest src/backend/tests/ src/backend/scoring/tests/ \
    src/evaluator/executors/tests/ src/evaluator/providers/tests/ -v

# Specific test files
python3 -m pytest src/backend/scoring/tests/test_strategies.py -v  # Scoring strategies
python3 -m pytest src/backend/tests/test_scoring.py -v             # Scoring engine
python3 -m pytest src/evaluator/executors/tests/test_registry.py -v # Executor registry
```

## Configuration

### Backend (`config/backend.toml.example`)
- Database URL, WebSocket settings
- Chain sync intervals, commitment scanning

### Validator (`config/validator.toml.example`)
- Backend URL, API key
- Wallet/hotkey configuration
- Reconnect intervals

### Evaluator (`config/evaluator.toml.example`)
- Database, Ray cluster settings
- Worker resources, timeouts
- Kubernetes namespace

## Deployment

### Docker Compose (`deploy/docker/`)
```bash
docker compose up -d postgres migrator validator evaluator
```

### Key Environment Variables
- `KINITRO_API_KEY`: Validator authentication
- `KUBECONFIG`: Kubernetes config for evaluator
- `DB_HOST`, `DB_USER`, `DB_PASSWORD`: Database credentials

## Development Workflow

1. **Make changes** in appropriate module
2. **Run tests** to verify nothing broke
3. **Update migrations** if models changed (`alembic revision --autogenerate`)
4. **Test locally** with `python3 -m <module>`
5. **Commit** with conventional commits (feat:, fix:, refactor:, etc.)

## Recent Refactoring (Phases 1-6)

### Phase 1: Decomposed BackendService
- Extracted `ScoringEngine`, `ChainMonitor`, `JobScheduler`
- Created `ProviderRegistry` for environment providers

### Phase 2: Task Abstraction Layer
- Added `TaskSpec`, `TaskResult`, `TaskContext`, `TaskExecutor` protocol
- Created `ExecutorRegistry` for task type dispatch
- Implemented `RLRolloutExecutor` wrapping existing rollout infrastructure

### Phase 3: Scoring Strategies
- Added `ScoringStrategy` protocol for pluggable scoring
- Implemented `RLRolloutScoringStrategy`
- Created `ScoringStrategyRegistry` for strategy dispatch
- Updated `ScoringEngine` to use strategies

### Phase 4-6: Architecture Simplification
- **Direct evaluator connection**: Evaluators connect to backend via WebSocket
- **Removed pgqueuer relay**: No more validator-as-middleman for jobs
- **Polling-based validator**: Validators poll `/weights` endpoint instead of WebSocket
- **Consolidated orchestrator**: Single `orchestrator.py` (removed legacy code)

## Important Files Reference

| Purpose | File |
|---------|------|
| Task interfaces | `src/core/tasks.py` |
| Executor registry | `src/evaluator/executors/registry.py` |
| RL executor | `src/evaluator/executors/rl_rollout.py` |
| Scoring engine | `src/backend/scoring_engine.py` |
| Scoring strategies | `src/backend/scoring/strategies.py` |
| Strategy registry | `src/backend/scoring/registry.py` |
| WebSocket messages | `src/core/messages.py` |
| Database models | `src/backend/models.py` |
| Backend service | `src/backend/service.py` |
| Evaluator hub | `src/backend/evaluator_hub.py` |
| Orchestrator | `src/evaluator/orchestrator.py` |
