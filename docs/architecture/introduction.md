---
section: Core Concepts
---

# Introduction

Kinitro incentivizes the emergence of agents that can conquer various tasks across different environments. Miners publish agents to compete, validators peform rollouts and evaluate the agents, and reward miners based on the results. All this happens in real-time and can easily be viewed by anyone through our [dashboard](https://kinitro.ai/dashboard).

## Platform Flow

1. **Submission** – The miner CLI packages the agent, requests a presigned upload slot from the backend, pushes the archive directly to the private vault, and commits the submission ID on Bittensor.
2. **Ingestion** – The backend ties the chain commitment to the uploaded artifact, records the submission with its hold-out window, and schedules evaluation jobs.
3. **Distribution** – Evaluators connect directly to the backend via WebSocket, receive jobs (including signed artifact URLs) and execute them.
4. **Evaluation** – The orchestrator launches a Kubernetes pod per submission, Ray rollout workers evaluate the agent via RPC, and telemetry is logged.
5. **Results & Incentives** – Evaluators stream results back to the backend, which stores metrics, queues leader candidates for admin approval, emits realtime updates, and recalculates scores/weights once an approved leader exists. Validators poll the `/weights` endpoint and set weights on the Bittensor chain. After hold-out expiry, the backend issues time-limited release URLs for public access.

## System Architecture

```mermaid
flowchart TD

  %% External systems
  subgraph EXT[External]
    BT([Bittensor]):::external
    S3([Private Vault - S3-Compatible]):::external
  end

  %% Core systems
  subgraph MNR[Miner]
    MIN["Miner CLI"]:::miner
  end
  subgraph BE[Backend]
    BEC["Backend Service"]:::backend
  end
  subgraph VAL[Validator]
    VALN["Validator Node"]:::validator
  end
  subgraph EVAL[Evaluator]
    EVN["Evaluator Cluster"]:::evaluator
  end
  subgraph OBS[Clients]
    CLN["Dashboards / Tools"]:::clients
  end

  %% Submission + hold-out
  MIN -- "request upload + presign" --> BEC
  BEC -- "presigned PUT" --> MIN
  MIN -- "upload artifact" --> S3
  MIN -- "commit submission id" --> BT
  BEC -- "monitor commitments" --> BT

  %% Evaluation loop (direct backend-evaluator connection)
  BEC -- "WebSocket: dispatch jobs" --> EVN
  EVN -- "download via presigned GET" --> S3
  EVN -- "WebSocket: stream results" --> BEC

  %% Weight setting (validator polls backend)
  VALN -- "HTTP: poll /weights" --> BEC
  VALN -- "set weights" --> BT

  %% Outputs
  BEC -- "broadcast updates" --> CLN
  BEC -- "release presigned URL on hold-out expiry" --> CLN

  %% Styles
  classDef external fill:#0288d1,stroke:#01579b,color:#fff
  classDef miner fill:#7c3aed,stroke:#4c1d95,color:#fff
  classDef backend fill:#16a34a,stroke:#166534,color:#fff
  classDef validator fill:#fb8c00,stroke:#e65100,color:#fff
  classDef evaluator fill:#db2777,stroke:#9d174d,color:#fff
  classDef clients fill:#64748b,stroke:#334155,color:#fff
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant MinerCLI
    participant BackendAPI
    participant SubmissionStorage as S3 Vault
    participant Chain
    participant EvaluatorCluster as Evaluator
    participant K8sPod as Evaluation Pod
    participant ValidatorNode as Validator
    participant AdminConsole as Admin Console
    participant ReleaseTask as Hold-out Release Task

    MinerCLI->>BackendAPI: POST /submissions/request-upload<br/>(signed payload)
    BackendAPI-->>MinerCLI: Presigned PUT URL + submission_id
    MinerCLI->>SubmissionStorage: PUT submission.tar.gz (presigned)
    MinerCLI->>Chain: Commit (provider=S3, submission_id, comp_id)

    BackendAPI->>BackendAPI: Match commitment with upload<br/>create MinerSubmission + jobs
    BackendAPI->>EvaluatorCluster: WebSocket: Broadcast EvalJobMessage (artifact URL, hash, holdout info)

    EvaluatorCluster->>K8sPod: Launch pod (init + runner)
    K8sPod->>SubmissionStorage: GET submission.tar.gz (presigned)
    K8sPod->>EvaluatorCluster: RPC evaluation results
    EvaluatorCluster->>BackendAPI: WebSocket: EvalResultMessage & status updates
    BackendAPI->>BackendAPI: Store metrics, update job status
    BackendAPI->>AdminConsole: Surface pending leader candidates
    AdminConsole->>BackendAPI: Approve / reject candidate (optional note)
    BackendAPI->>BackendAPI: Apply approved leader + cache scores

    loop periodic (every 5 min)
        ValidatorNode->>BackendAPI: GET /weights
        BackendAPI-->>ValidatorNode: WeightsSnapshot
        ValidatorNode->>Chain: set_weights (if changed)
    end

    loop periodic
        ReleaseTask->>BackendAPI: Scan for expired hold-outs
        BackendAPI->>SubmissionStorage: Presign release GET URL
        BackendAPI->>ReleaseTask: Save release URL + expiry
    end

```

## Component Responsibilities

### Backend Service

- **FastAPI REST / Admin**: Hosts competition CRUD, submission uploads, stats, and admin endpoints.
- **Evaluator Hub**: Manages WebSocket connections from evaluators via `/ws/evaluator`, dispatches jobs, and receives results.
- **Chain Monitor & Scheduler**: Tracks Bittensor commitments, ties them to uploaded artifacts, creates `BackendEvaluationJob` records, and watches for stale work.
- **Hold-out & Vault Manager**: Issues presigned URLs for uploads and releases, enforces per-competition hold-out windows, and keeps artifacts private until expiry.
- **Realtime Broadcaster**: Manages client subscriptions and pushes structured events such as job updates, episode completions, and live stats.
- **Scoring & Weight Engine**: Periodically recalculates miner scores and serves weight snapshots via the `/weights` endpoint for validators to poll.
- **Backend PostgreSQL**: Source of truth for competitions, submissions, jobs, job status, results, stats, and evaluator connections.

### Validator Node

- **HTTP Poller**: Periodically fetches `GET /weights` from the backend to retrieve the latest weight snapshot.
- **Weight Setter**: Compares fetched weights against the last committed values and calls `set_weights` on the Bittensor chain when changes occur.
- **Lightweight Design**: No database, no evaluator, no WebSocket - just polls and sets weights.

### Evaluator Cluster

- **Evaluator Orchestrator**: Connects directly to the backend via WebSocket, receives jobs, enforces concurrency caps, and coordinates job lifecycles.
- **Submission Pods**: Kubernetes pods created per submission to run miner containers in isolation.
- **Ray Rollout Workers**: Execute benchmark episodes, communicate with submission pods via RPC, and track success metrics.
- **Episode Logger**: Captures per-episode and per-step data, uploads media to S3-compatible storage, and streams telemetry back to the backend.

### Miner Tooling

- **Miner CLI**: Packages submissions, requests vault upload slots, pushes artifacts directly to the backend-controlled storage, and notarizes submissions on-chain.

### Real-time Clients

- Subscribe to the backend's public WebSocket endpoint to monitor competitions, evaluator connectivity, and evaluation progress live.

## Next Steps

- Dive into the [Evaluator architecture notes](orchestrator.md) to see how jobs flow from backend to evaluators.
- Review the [Evaluator internals](evaluator.md) for details on Ray workers, RPC bridges, and logging pipelines.
- Check the [Incentive mechanism](incentive.md) to understand how scores flow into weight updates.
