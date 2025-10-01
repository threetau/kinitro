---
section: Core Concepts
---

# Introduction

Kinitro links miners, validators, evaluators, and external infrastructure to deliver verifiable scores for embodied AI submissions. Miners publish models and commit them on-chain, the backend turns those commitments into evaluation jobs, validators coordinate execution, and result streams feed back into the system in real time.

## Platform Flow
1. **Submission** – A miner uploads an agent to Hugging Face and publishes the commitment on the Bittensor chain.
2. **Ingestion** – The backend monitors the chain, records new submissions, and schedules evaluation jobs.
3. **Distribution** – Validators connect to the backend over WebSocket, receive jobs, and persist them to a durable queue.
4. **Evaluation** – The evaluator orchestrator pulls queued jobs, spins up submission containers, runs Ray rollout workers, and logs every episode.
5. **Results & Incentives** – Validators forward metrics back to the backend, which stores them, emits realtime updates, and computes miner scores for weight broadcasts.

## System Architecture

```mermaid
graph LR
    %% External Infrastructure
    subgraph "External Services"
        BT[Bittensor Chain]
        HF[Hugging Face Hub]
        R2[Cloudflare R2 Storage]
    end

    %% Miner
    subgraph Miner
        MC[Miner CLI]
        SB[Submission Bundle]
    end

    %% Backend
    subgraph "Backend Service"
        CM[Chain Monitor & Scheduler]
        API[FastAPI REST + Validator WS]
        RT[Realtime Broadcaster]
        SCORE[Scoring & Weight Engine]
        BDB[(Backend PostgreSQL)]
    end

    %% Validator
    subgraph "Validator Node"
        WV[WebSocket Client]
        VDB[(Validator PostgreSQL)]
        PQ[pgqueuer Runner]
    end

    %% Evaluator
    subgraph "Evaluator Cluster"
        ORC[Evaluator Orchestrator]
        POD[Submission Pods]
        RAY[Ray Rollout Workers]
        LOG[Episode Logger]
    end

    %% Observers
    subgraph "Realtime Clients"
        DASH[Dashboards & Tools]
    end

    MC -->|Package model| SB
    SB -->|Upload artifact| HF
    MC -->|Commit metadata| BT

    CM -->|Scan commitments| BT
    CM -->|Create jobs| BDB
    CM -->|Send EvalJob| WV
    API -->|Persist & expose| BDB
    API -->|Emit events| RT
    RT -->|Subscriptions| DASH

    WV -->|Queue job| VDB
    VDB -->|pgq event| PQ
    PQ -->|Dispatch job| ORC

    ORC -->|Start pod| POD
    ORC -->|Coordinate| RAY
    RAY -->|RPC requests| POD
    RAY -->|Log episodes| LOG
    LOG -->|Upload artifacts| R2
    LOG -->|Queue telemetry| VDB
    ORC -->|Queue results| VDB

    VDB -->|pgq event| PQ
    PQ -->|Send results| WV
    WV -->|EvalResult & telemetry| API
    API -->|Store updates| BDB
    API -->|Broadcast events| RT

    SCORE -->|Read metrics| BDB
    SCORE -->|SetWeights message| WV
    WV -->|set\_node\_weights| BT

    classDef external fill:#0277bd,color:#fff,stroke:#01579b,stroke-width:2px
    classDef miner fill:#6a1b9a,color:#fff,stroke:#4a148c,stroke-width:2px
    classDef backend fill:#2e7d32,color:#fff,stroke:#1b5e20,stroke-width:2px
    classDef validator fill:#ef6c00,color:#fff,stroke:#e65100,stroke-width:2px
    classDef evaluator fill:#ad1457,color:#fff,stroke:#880e4f,stroke-width:2px
    classDef clients fill:#546e7a,color:#fff,stroke:#37474f,stroke-width:2px

    class BT,HF,R2 external
    class MC,SB miner
    class CM,API,RT,SCORE,BDB backend
    class WV,VDB,PQ validator
    class ORC,POD,RAY,LOG evaluator
    class DASH clients
```

## Component Responsibilities

**Backend Service**
- **FastAPI REST / Admin**: Hosts competition CRUD, submission views, stats, validator management, and WebSocket endpoints (`src/backend/endpoints.py`).
- **Chain Monitor & Scheduler**: Tracks Bittensor commitments, turns them into `BackendEvaluationJob` records, and watches for stale jobs (`src/backend/service.py`).
- **Realtime Broadcaster**: Manages client subscriptions and pushes structured events such as job updates, episode completions, and live stats (`src/backend/realtime.py`).
- **Scoring & Weight Engine**: Periodically recalculates miner scores and pushes weight updates back to validators for on-chain emission.
- **Backend PostgreSQL**: Source of truth for competitions, submissions, jobs, job status, results, stats, and validator connections (`src/backend/models.py`).

**Validator Node**
- **WebSocket Client**: Authenticates with the backend, receives `EvalJobMessage` payloads, and streams results back (`src/validator/websocket_validator.py`).
- **pgqueuer Runner**: Persists jobs/results/episode logs in PostgreSQL so work survives restarts and can be retried (`src/validator/websocket_validator.py`).
- **Validator PostgreSQL**: Stores pgq queues plus normalized tables for jobs, results, and metrics consumed by the evaluator (`src/validator/db`).

**Evaluator Cluster**
- **Evaluator Orchestrator**: Listens to the pgqueuer queue, enforces concurrency caps, and coordinates job lifecycles (`src/evaluator/orchestrator.py`).
- **Submission Pods**: Kubernetes pods created per submission to run miner containers in isolation (`src/evaluator/containers`).
- **Ray Rollout Workers**: Execute benchmark episodes, communicate with submission pods via RPC, and track success metrics (`src/evaluator/rollout`).
- **Episode Logger**: Captures per-episode and per-step data, uploads media to R2, and enqueues telemetry for validator forwarding (`src/evaluator/rollout/episode_logger.py`).

**Miner Tooling**
- **Miner CLI**: Packages models, uploads to Hugging Face, and notarizes commitments on-chain so the backend can discover them (`src/miner/__main__.py`).

**Real-time Clients**
- Subscribe to the backend’s public WebSocket endpoint to monitor competitions, validator connectivity, and evaluation progress live (`src/core/messages.py`).

## Next Steps
- Dive into the [Validator architecture notes](orchestrator.md) to see how the queue, database, and message formats interact.
- Review the [Evaluator internals](evaluator.md) for details on Ray workers, RPC bridges, and logging pipelines.
- Check the [Incentive mechanism](incentive.md) to understand how scores flow into weight updates.
