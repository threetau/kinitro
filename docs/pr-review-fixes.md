# PR Review Fixes - Split Architecture (PR #1)

This document tracks the issues identified in the CodeRabbit review and our analysis/fixes.

---

## Critical Issues

### 1. Non-deterministic Task IDs (task_generator.py)

**The Problem:**
```python
task_id = hash(f"{block_number}:{env_id}:{miner.uid}:{i}") % (2**31)
```

Python's built-in `hash()` function is **randomized on each process start** (via `PYTHONHASHSEED`). This means:
- Scheduler starts, creates tasks with IDs like `[123, 456, 789]`
- Scheduler restarts (crash, deployment, etc.)
- Same inputs now generate different IDs like `[987, 654, 321]`

**Why it matters:**
- Executors fetch tasks by ID from the database
- If scheduler restarts mid-cycle, new task IDs won't match existing DB records
- Results can't be submitted because task IDs don't exist
- Could cause duplicate tasks or orphaned results

**Additional Concern (from user):**
Using `block_number` as part of the hash is **predictable** - miners could potentially sample-ahead and overfit their models to future evaluation data.

**How affine-cortex handles this:**

Affine-cortex uses a **two-tier ID system**:
1. `task_id` = Dataset index being evaluated (deterministic, from a `sampling_list`)
2. `task_uuid` = Random UUID generated at task creation time (used for execution tracking)

```python
# In affine-cortex task_pool.py:
task_uuid = str(uuid.uuid4())  # Random UUID, unpredictable
```

They prevent prediction through:
1. **Sampling list rotation** - Lists are rotated periodically with random additions
2. **Random task selection** - `random.shuffle(pending_tasks)` before assignment
3. **FIFO rotation** - Remove oldest, add random new tasks

**Recommended Fix:**
- Separate "what to evaluate" (task_id/seed) from "execution tracking" (use UUID)
- Consider rotation-based sampling lists instead of block-number-based seeds
- Use cryptographic randomness for unpredictable task assignment

**Status:** Needs design discussion - may require architectural changes

---

## Major Issues

### 2. Wrong Task Status Logic (storage.py, lines 449-452)

**The Problem:**
```python
task.status = TaskStatus.COMPLETED.value if success or error is None else TaskStatus.FAILED.value
```

The logic `success or error is None` is flawed:
- If `success=False` and `error=None` → evaluates to `False or True` → `True` → COMPLETED ❌
- A failed task with no error message gets marked as completed

**Why it matters:**
- Corrupts scoring data - failed evaluations counted as successes
- Pareto weights become incorrect
- Miners could get undeserved rewards

**Note:** Need to verify what `TaskStatus` states actually indicate and how they're used downstream.

**The Fix:**
```python
task.status = TaskStatus.COMPLETED.value if success else TaskStatus.FAILED.value
```

**Status:** TO FIX - straightforward logic fix

---

### 3. Missing Session Rollback (tasks.py, lines 100-102)

**The Problem:**
```python
try:
    success = await storage.submit_task_result(...)
except Exception as e:
    rejected += 1  # No rollback!
    errors.append(f"Task {result.task_id}: {str(e)}")
```

When a database operation fails, SQLAlchemy leaves the session in a "failed transaction" state. Without calling `rollback()`, all subsequent operations in the same request will fail.

**Why it matters:**
- Executor submits batch of 10 results
- Task 3 fails with an exception
- Tasks 4-10 all fail silently because session is corrupted
- Results are lost

**The Fix:**
```python
except Exception as e:
    await session.rollback()  # Clear the failed state
    rejected += 1
    errors.append(f"Task {result.task_id}: {str(e)}")
```

**Note:** This may be affected by architectural decisions from issue #1.

**Status:** TO FIX

---

### 4. Timeout Kwarg Name (worker.py, line 139)

**The Problem:**
```python
result = await env.evaluate(
    ...
    _timeout=self.config.eval_timeout,
)
```

**Analysis of affinetes behavior:**

Looking at `affinetes/core/wrapper.py`, the `_timeout` parameter IS handled correctly:

```python
async def method_caller(*args, _timeout: Optional[int] = None, **kwargs):
    """
    Args:
        _timeout: Optional call-level timeout in seconds (not passed to remote)
    """
    if _timeout is not None:
        result = await asyncio.wait_for(call_coro, timeout=_timeout)
    else:
        result = await call_coro
```

The `_timeout` parameter:
- Is intercepted by the wrapper's `__getattr__` method
- Is **NOT** passed to the remote method (only `**kwargs` are passed)
- Is used to wrap the call with `asyncio.wait_for()`

**Conclusion:** The current implementation using `_timeout` IS CORRECT. The CodeRabbit review was wrong about this.

**Status:** NO FIX NEEDED - current implementation is correct

---

### 5. Cycle Timeout Publishes Incomplete Weights (main.py, lines 236-244)

**The Problem:**
```python
async def _wait_for_cycle_completion(self, cycle_id: int) -> None:
    while True:
        if await self.storage.is_cycle_complete(session, cycle_id):
            return
        
        if elapsed > timeout:
            logger.warning("cycle_timeout", ...)
            break  # Just breaks out, doesn't indicate failure!

# After calling:
await self._wait_for_cycle_completion(cycle_id)
# Code continues to compute and publish weights regardless of timeout!
```

When the cycle times out:
1. The function just `break`s out of the loop
2. Caller has no idea it timed out
3. Weights are computed from partial results (only completed tasks)
4. Incomplete weights get published to the chain
5. PENDING/ASSIGNED tasks are orphaned forever

**Why it matters:**
- Validators submit incorrect weights to Bittensor
- Some miners evaluated, others not - unfair scoring
- Orphaned tasks waste database space

**How affine-cortex handles this:**
- **No cycle concept** - tasks are independent units
- **Continuous recovery** - missing tasks regenerated every 10 seconds
- **Startup cleanup** - deletes ALL assigned tasks on startup (orphan recovery)

**The Fix:**
Return a boolean and handle timeout as failure:
```python
async def _wait_for_cycle_completion(self, cycle_id: int) -> bool:
    ...
    if elapsed > timeout:
        return False  # Indicate timeout
    return True

# Caller:
if not await self._wait_for_cycle_completion(cycle_id):
    await self.storage.fail_cycle(session, cycle_id, "Timeout")
    return  # Don't publish weights
```

**Status:** TO FIX

---

### 6. No HTTP Timeouts (api_client.py, lines 19-23)

**The Problem:**
```python
async def _get_session(self) -> aiohttp.ClientSession:
    if self._session is None or self._session.closed:
        self._session = aiohttp.ClientSession()  # No timeout!
    return self._session
```

Without a timeout, HTTP requests can hang indefinitely if:
- API server is down but connection isn't refused
- Network is partitioned
- Server accepts connection but never responds

**Why it matters:**
- Executor hangs forever waiting for API
- No tasks get processed
- No error is raised - it just freezes

**The Fix:**
```python
timeout = aiohttp.ClientTimeout(total=30)
self._session = aiohttp.ClientSession(timeout=timeout)
```

**Status:** TO FIX

---

### 7. Health Check Always Returns "Connected" (health.py, lines 18-26)

**The Problem:**
```python
@router.get("/health")
async def health_check(storage: Storage = Depends(get_storage)):
    db_status = "connected" if storage is not None else "disconnected"
    return HealthResponse(database=db_status)
```

`get_storage()` either returns the Storage instance OR raises HTTP 503. It never returns `None`. So `storage is not None` is always `True`.

**Why it matters:**
- Health checks used by load balancers, k8s, monitoring
- Reports "healthy" even when database is down
- Traffic routed to broken instances
- No alerts triggered

**The Fix:**
Actually query the database:
```python
async def health_check(session: AsyncSession = Depends(get_session)):
    try:
        await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    return HealthResponse(database=db_status)
```

**Status:** TO FIX

---

### 8. No Validation of pareto_temperature (scoring.py)

**The Problem:**
```python
def compute_weights(
    miner_scores: dict[int, dict[str, float]],
    pareto_temperature: float = 1.0,  # What if this is 0 or negative?
) -> ...:
    # Later used in softmax: exp(score / temperature)
    weights = scores_to_weights(subset_scores, temperature=pareto_temperature)
```

If `pareto_temperature <= 0`:
- Division by zero in softmax
- NaN/Inf values in weights
- Chain submission fails or corrupts data

**Why it matters:**
- Config could accidentally be set to 0
- Silent corruption of weights
- Difficult to debug

**The Fix:**
```python
def compute_weights(..., pareto_temperature: float = 1.0):
    if pareto_temperature <= 0:
        raise ValueError("pareto_temperature must be > 0")
    ...
```

**Status:** TO FIX

---

## Minor Issues (Address Later)

### 9. Unsorted `__all__` Lists (Ruff RUF022)
- `kinitro/api/__init__.py`
- `kinitro/executor/__init__.py`

### 10. Missing Language in README Code Fence (line 268)
Change ` ``` ` to ` ```text `

### 11. Add Validation Constraints to Config Fields
- Add `ge=1` or `gt=0` to scheduler/executor configs

### 12. Refactor Duplicate Code in scores.py
Extract helper function for building scores response

### 13. Fix O(n²) Loop in miners.py
Accumulate totals in one pass instead of rescanning

### 14. Use `asyncio.to_thread` for Subtensor Calls
Avoid blocking event loop with network I/O

---

## Affine-Cortex Architecture Comparison

### Key Differences

| Aspect | Affine-Cortex | Kinitro |
|--------|---------------|---------|
| **Task Model** | Single task pool, no cycles | Cycle-based with batches |
| **ID System** | task_id (dataset) + task_uuid (execution) | Complex composite keys |
| **Scheduling** | Per-miner slot allocation | Batch-based scheduling |
| **State Machine** | 3 states: pending, assigned, paused | Multiple cycle states |
| **Concurrency** | Dynamic slots per miner (3-10) | Fixed batch sizes |
| **Timeout Handling** | Tasks reset to pending | Cycle marked complete with partial data |

### Affine-Cortex Task Statuses
```
pending → assigned → (success: DELETE, error: pending/paused)
```
- **pending**: Ready for execution
- **assigned**: Being executed
- **paused**: Max retries exceeded (TTL: 4 hours, then deleted)
- Note: NO "completed" status - successful tasks are DELETED

### Key Patterns to Consider

1. **Separation of concerns**: task_id (what) vs task_uuid (tracking)
2. **Sampling list rotation**: Unpredictable task selection
3. **No cycle concept**: Eliminates complex state synchronization
4. **Startup cleanup**: Delete all assigned tasks on restart
5. **Dynamic concurrency**: Adjust slots based on success rate

---

## Action Items

### Immediate Fixes (Issues 5-8) ✅ DONE
- [x] Fix cycle timeout to return boolean and fail cycle
- [x] Add HTTP timeouts to api_client (30s default)
- [x] Fix health check to actually probe database with SELECT 1
- [x] Add pareto_temperature validation (must be > 0)

### Needs Design Discussion (Issues 1-3)
- [ ] Task ID generation and unpredictability
- [ ] Consider simpler architecture without cycles
- [ ] TaskStatus semantics and state machine
- [ ] Issue 2: Wrong task status logic (success=False, error=None → COMPLETED)
- [ ] Issue 3: Missing session rollback in exception handler

### Deferred (Issues 9-14)
- [ ] Linting fixes
- [ ] Code quality improvements
