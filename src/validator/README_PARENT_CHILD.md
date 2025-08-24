# Parent-Child Validator System

This document describes the parent-child validator architecture implemented for distributed job processing.

## Overview

The parent-child validator system allows multiple validator instances to work together:

- **Parent Validator**: Monitors the blockchain, discovers new commitments, and distributes jobs to child validators
- **Child Validators**: Connect to the parent, receive jobs, process them in their own database, and report results back

## Architecture

```
┌─────────────────┐    Job Distribution    ┌───────────────────┐
│                 │◄──────────────────────►│                   │
│ Parent Validator│                        │ Child Validator 1 │
│                 │                        │                   │
│ - Chain Monitor │    RPC Communication   │ - Job Processing  │
│ - Job Queue     │◄──────────────────────►│ - Local DB        │
│ - RPC Server    │                        │ - RPC Client      │
└─────────────────┘                        └───────────────────┘
         │                                           │
         │                                           │
         │              Job Distribution             │
         │◄─────────────────────────────────────────►│
         │                                           │
         │                                  ┌───────────────────┐
         │                                  │                   │
         └─────────────────────────────────►│ Child Validator 2 │
                                            │                   │
                                            │ - Job Processing  │
                                            │ - Local DB        │
                                            │ - RPC Client      │
                                            └───────────────────┘
```

## Configuration

### Parent Validator Configuration

Create a `validator_parent.toml` file:

```toml
version = "0.0.1"
log_level = "INFO"

wallet_path = "~/.bittensor/wallets"
wallet_name = "kinitro-local"
hotkey_name = "parent_vali"

pg_database = "postgresql://postgres@localhost/postgres"

# Parent validator configuration
is_parent_validator = true
parent_port = 8001  # Port for parent RPC server

[subtensor]
network = "local"
address = "ws://127.0.0.1:9944"
netuid = 2

[neuron]
sync_frequency = 300
load_old_nodes = true
min_stake_threshold = 1000
allowed_validators = []
max_commitment_lookback = 3
```

### Child Validator Configuration

Create a `validator_child.toml` file for each child:

```toml
version = "0.0.1"
log_level = "INFO"

wallet_path = "~/.bittensor/wallets"
wallet_name = "kinitro-local" 
hotkey_name = "child_vali_1"

pg_database = "postgresql://postgres@localhost/postgres_child1"

# Child validator configuration
is_parent_validator = false
validator_id = "child_validator_1"  # Unique ID for this child validator
parent_host = "localhost"           # Parent validator host
parent_port = 8001                  # Parent validator RPC port
child_port = 8002                   # This child's RPC port

[subtensor]
network = "local"
address = "ws://127.0.0.1:9944"
netuid = 2

[neuron]
sync_frequency = 300
load_old_nodes = true
min_stake_threshold = 1000
allowed_validators = []
max_commitment_lookback = 3
```

## Running the System

### 1. Start Parent Validator

```bash
python -m validator --config validator_parent.toml
```

The parent validator will:
- Monitor the blockchain for new commitments
- Start an RPC server on port 8001
- Wait for child validators to register
- Distribute jobs to registered children

### 2. Start Child Validators

```bash
# Child 1
python -m validator --config validator_child1.toml

# Child 2 (with different config)
python -m validator --config validator_child2.toml
```

Each child validator will:
- Connect to the parent validator
- Register itself with a unique ID
- Poll for new jobs periodically
- Process jobs in its own database
- Report completion status back to parent

## Communication Protocol

The system uses Cap'n Proto RPC for communication between parent and child validators.

### RPC Interface

```capnp
interface ValidatorJobsService {
  # Child calls this to register with parent
  registerChild @0 (childId: Text, endpoint: Text) -> (success: Bool, message: Text);
  
  # Child calls this to request jobs
  requestJobs @1 (childId: Text, maxJobs: UInt32) -> (jobs: List(EvaluationJobData));
  
  # Child calls this to report job completion
  reportJobCompletion @2 (childId: Text, jobId: UInt64, success: Bool, result: Text) -> (acknowledged: Bool);
  
  # Parent calls this on child to send a job directly
  receiveJob @3 (job: EvaluationJobData) -> (accepted: Bool, message: Text);
  
  # Health check
  ping @4 () -> (pong: Text);
}
```

### Job Flow

1. **Job Discovery**: Parent validator monitors blockchain for new commitments
2. **Job Creation**: Parent creates `EvaluationJob` objects and adds them to pending queue
3. **Job Distribution**: Child validators periodically request jobs from parent
4. **Job Processing**: Child validators queue jobs in their local database using pgqueuer
5. **Job Execution**: Local orchestrator picks up jobs and processes them
6. **Result Reporting**: Child validators report completion status back to parent

## Database Isolation

Each validator maintains its own database:

- **Parent Database**: Stores validator state, commitment fingerprints, and job tracking
- **Child Databases**: Store received jobs and evaluation results independently

This ensures:
- No database contention between validators
- Independent scaling of storage
- Fault tolerance - if one database fails, others continue working

## Monitoring and Management

### Parent Validator Statistics

The parent validator tracks statistics for each child:

```python
{
    "total_children": 2,
    "active_children": 2,
    "pending_jobs": 5,
    "children": [
        {
            "child_id": "child_validator_1",
            "endpoint": "localhost:8002",
            "active": True,
            "last_seen": 1629123456.0,
            "active_jobs": 3,
            "total_jobs_sent": 25,
            "total_jobs_completed": 22
        },
        ...
    ]
}
```

### Health Checks

- Parent and child validators implement ping/pong health checks
- Child validators are marked inactive if not seen for 5 minutes
- Automatic reconnection logic handles temporary network issues

## Command Line Options

### Parent Validator Options

- `--is-parent-validator`: Run as parent validator (default: false)
- `--parent-port PORT`: RPC server port for parent (default: 8001)

### Child Validator Options

- `--validator-id ID`: Unique identifier for child validator
- `--parent-host HOST`: Parent validator hostname (default: localhost)
- `--parent-port PORT`: Parent validator RPC port (default: 8001)  
- `--child-port PORT`: Child validator RPC port (default: 8002)

## Testing

Run the test suite:

```bash
python -m pytest src/validator/tests/test_parent_child_system.py -v
```

The tests verify:
- Parent-child communication protocols
- Job distribution mechanisms
- Configuration handling
- Error scenarios

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure parent validator is running before starting children
2. **Database Errors**: Verify each validator has access to its configured database
3. **Port Conflicts**: Ensure each child uses a unique port number
4. **Registration Failed**: Check network connectivity and firewall settings

### Logs

Enable DEBUG logging to see detailed communication:

```toml
log_level = "DEBUG"
```

Look for these log messages:
- `"Successfully registered with parent validator as {child_id}"`
- `"Sending {count} jobs to child {child_id}"`
- `"Child {child_id} completed job {job_id}"`

## Future Enhancements

Potential improvements to the system:

1. **Load Balancing**: Distribute jobs based on child validator capacity
2. **Failover**: Reassign jobs from failed child validators
3. **Job Priorities**: Support different priority levels for jobs
4. **Metrics**: Detailed performance monitoring and metrics collection
5. **Security**: Add authentication and encryption for RPC communication