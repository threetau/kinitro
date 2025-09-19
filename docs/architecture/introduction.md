---
section: Core Concepts
---

# Introduction

## How it works

1. **Define**: Competitions are posted on the Kinitro platform, each with their own set of tasks
2. **Compete**: Miners train and submit agents.
3. **Validate & reward**: Validators evaluate the agents, and the best miners earn rewards.

## Architecture Overview

Kinitro consists of three main components working together to create an incentivized evaluation platform:

```mermaid
graph TB
    %% External Systems
    subgraph "External Systems"
        BC[Bittensor Chain]
        HF[Hugging Face Hub]
        R2[Cloudflare R2/S3 Storage]
    end

    %% Miner
    subgraph "Miner"
        MC[Miner CLI]
        MS[Model Submission]
    end

    %% Backend
    subgraph "Backend Service"
        BE[FastAPI Backend]
        DB[(PostgreSQL)]
        WS[WebSocket Handler]
        API[REST API]
        CM[Chain Monitor]
        JS[Job Scheduler]
    end

    %% Validator
    subgraph "Validator"
        WV[WebSocket Validator]
        PQ[pgqueuer]
        OR[Orchestrator]
        RC[Ray Cluster]
    end

    %% Evaluator
    subgraph "Evaluator (Ray Workers)"
        RW[Rollout Workers]
        EL[Episode Logger]
        ENV[Environment Wrappers]
        RPC[RPC Process]
        SC[Submission Container]
    end

    %% Data Flow - Submission Process
    MC -->|1. Upload Model| HF
    MC -->|2. Commit Info| BC
    
    %% Data Flow - Competition & Job Creation
    CM -->|3. Monitor Commitments| BC
    CM -->|4. Create Jobs| BE
    BE -->|Store| DB
    
    %% Data Flow - Job Distribution
    BE -->|5. Broadcast Jobs| WS
    WS -->|WebSocket| WV
    WV -->|Queue Jobs| PQ
    PQ -->|Process| OR
    
    %% Data Flow - Evaluation Execution
    OR -->|6. Start Evaluation| RC
    RC -->|Create Workers| RW
    RW -->|Initialize| EL
    RW -->|Load Model| SC
    SC -->|Pull from| HF
    
    %% Data Flow - Episode Execution
    RW -->|Run Episodes| ENV
    ENV -->|Capture Observations| EL
    EL -->|Upload Images| R2
    EL -->|Log Episode Data| PQ
    
    %% Data Flow - Results & Status
    PQ -->|Episode Data| WV
    WV -->|Forward| WS
    RW -->|Evaluation Results| WV
    WV -->|Results| WS
    WS -->|Store Results| DB
    WS -->|Update Status| DB
    
    %% API Access
    DB -->|Query| API
    R2 -->|Observation URLs| API

    %% Styling
    classDef external fill:#0277bd,color:#fff,stroke:#01579b,stroke-width:2px
    classDef miner fill:#6a1b9a,color:#fff,stroke:#4a148c,stroke-width:2px
    classDef backend fill:#2e7d32,color:#fff,stroke:#1b5e20,stroke-width:2px
    classDef validator fill:#ef6c00,color:#fff,stroke:#e65100,stroke-width:2px
    classDef evaluator fill:#ad1457,color:#fff,stroke:#880e4f,stroke-width:2px
    
    class BC,HF,R2 external
    class MC,MS miner
    class BE,DB,WS,API,CM,JS backend
    class WV,PQ,OR,RC validator
    class RW,EL,ENV,RPC,SC evaluator
```

### Component Responsibilities

**Backend Service**
- **FastAPI Backend**: REST API endpoints and WebSocket management
- **Chain Monitor**: Monitors Bittensor chain for miner commitments  
- **Job Scheduler**: Creates and distributes evaluation jobs
- **WebSocket Handler**: Real-time communication with validators
- **PostgreSQL**: Stores competitions, jobs, results, episode data

**Validator** 
- **WebSocket Validator**: Connects to backend, receives jobs
- **pgqueuer**: Asynchronous message processing for episode data
- **Orchestrator**: Manages evaluation lifecycle using Ray
- **Ray Cluster**: Distributed computing for parallel evaluations

**Evaluator (Ray Workers)**
- **Rollout Workers**: Execute agent episodes in environments
- **Episode Logger**: Records episode data and observations  
- **Environment Wrappers**: MetaWorld/Gymnasium integration
- **RPC Process**: Communication with agent containers
- **Submission Containers**: Isolated model execution environments
