---
section: Core Concepts
---

# Introduction

Kinitro incentivizes the emergence of agents that can conquer various tasks across different environments. Miners publish agents to compete, validators peform rollouts and evaluate the agents, and reward miners based on the results. All this happens in real-time and can easily be viewed by anyone through our [dashboard](https://kinitro.ai/dashboard).

## Platform Flow
1. **Submission** – The miner CLI packages the agent, requests a presigned upload slot from the backend, pushes the archive directly to the private vault, and commits the submission ID on Bittensor.
2. **Ingestion** – The backend ties the chain commitment to the uploaded artifact, records the submission with its hold-out window, and schedules evaluation jobs.
3. **Distribution** – Validators connect via WebSocket, receive jobs (including signed artifact URLs) and queue them for execution.
4. **Evaluation** – The orchestrator launches a Kubernetes pod per submission, Ray rollout workers evaluate the agent via RPC, and telemetry is logged.
5. **Results & Incentives** – Validators stream results back to the backend, which stores metrics, queues leader candidates for admin approval, emits realtime updates, and recalculates scores/weights once an approved leader exists. After hold-out expiry, the backend issues time-limited release URLs for public access.

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
  BEC -- "link upload & create jobs" --> VALN

  %% Evaluation loop
  VALN -- "dispatch eval" --> EVN
  EVN -- "download via presigned GET" --> S3
  EVN -- "results" --> VALN
  VALN -- "report results" --> BEC

  %% Outputs
  BEC -- "broadcast updates" --> CLN
  BEC -- "weight updates" --> BT
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
    participant ValidatorOrchestrator as Validator Orchestrator
    participant K8sPod as Evaluation Pod
    participant AdminConsole as Admin Console
    participant ReleaseTask as Hold-out Release Task

    MinerCLI->>BackendAPI: POST /submissions/request-upload<br/>(signed payload)
    BackendAPI-->>MinerCLI: Presigned PUT URL + submission_id
    MinerCLI->>SubmissionStorage: PUT submission.tar.gz (presigned)
    MinerCLI->>Chain: Commit (provider=S3, submission_id, comp_id)

    BackendAPI->>BackendAPI: Match commitment with upload<br/>create MinerSubmission + jobs
    BackendAPI->>ValidatorOrchestrator: Broadcast EvalJobMessage (artifact URL, hash, holdout info)

    ValidatorOrchestrator->>K8sPod: Launch pod (init + runner)
    K8sPod->>SubmissionStorage: GET submission.tar.gz (presigned)
    K8sPod->>ValidatorOrchestrator: RPC evaluation results
    ValidatorOrchestrator->>BackendAPI: EvalResultMessage & status updates
    BackendAPI->>BackendAPI: Store metrics, update job status
    BackendAPI->>AdminConsole: Surface pending leader candidates
    AdminConsole->>BackendAPI: Approve / reject candidate (optional note)
    BackendAPI->>BackendAPI: Apply approved leader + cache scores

    loop periodic
        ReleaseTask->>BackendAPI: Scan for expired hold-outs
        BackendAPI->>SubmissionStorage: Presign release GET URL
        BackendAPI->>ReleaseTask: Save release URL + expiry
    end

```

## Component Responsibilities

**Backend Service**
- **FastAPI REST / Admin**: Hosts competition CRUD, submission uploads, stats, validator management, and WebSocket endpoints.
- **Chain Monitor & Scheduler**: Tracks Bittensor commitments, ties them to uploaded artifacts, creates `BackendEvaluationJob` records, and watches for stale work.
- **Hold-out & Vault Manager**: Issues presigned URLs for uploads and releases, enforces per-competition hold-out windows, and keeps artifacts private until expiry.
- **Realtime Broadcaster**: Manages client subscriptions and pushes structured events such as job updates, episode completions, and live stats.
- **Scoring & Weight Engine**: Periodically recalculates miner scores and pushes weight updates back to validators for on-chain emission.
- **Backend PostgreSQL**: Source of truth for competitions, submissions, jobs, job status, results, stats, and validator connections.

**Validator Node**
- **WebSocket Client**: Authenticates with the backend, receives `EvalJobMessage` payloads, and streams results back.
- **pgqueuer Runner**: Persists jobs/results/episode logs in PostgreSQL so work survives restarts and can be retried.
- **Validator PostgreSQL**: Stores pgq queues plus normalized tables for jobs, results, and metrics consumed by the evaluator.

**Evaluator Cluster**
- **Evaluator Orchestrator**: Listens to the pgqueuer queue, enforces concurrency caps, and coordinates job lifecycles.
- **Submission Pods**: Kubernetes pods created per submission to run miner containers in isolation.
- **Ray Rollout Workers**: Execute benchmark episodes, communicate with submission pods via RPC, and track success metrics.
- **Episode Logger**: Captures per-episode and per-step data, uploads media to S3-compatible storage, and enqueues telemetry for validator forwarding.

**Miner Tooling**
- **Miner CLI**: Packages submissions, requests vault upload slots, pushes artifacts directly to the backend-controlled storage, and notarizes submissions on-chain.

**Real-time Clients**
- Subscribe to the backend’s public WebSocket endpoint to monitor competitions, validator connectivity, and evaluation progress live.

## Next Steps
- Dive into the [Validator architecture notes](orchestrator.md) to see how the queue, database, and message formats interact.
- Review the [Evaluator internals](evaluator.md) for details on Ray workers, RPC bridges, and logging pipelines.
- Check the [Incentive mechanism](incentive.md) to understand how scores flow into weight updates.
