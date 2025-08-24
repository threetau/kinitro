#!/usr/bin/env python3
"""
Parent Validator RPC Server for job distribution
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

import capnp
from snowflake import SnowflakeGenerator

from core.db.models import EvaluationJob, EvaluationStatus
from core.schemas import ChainCommitmentResponse

schema_file = os.path.join(os.path.dirname(__file__), "validator_jobs.capnp")
validator_jobs_capnp = capnp.load(schema_file)

logger = logging.getLogger("validator.rpc.parent_server")


@dataclass
class ChildValidator:
    """Information about a registered child validator"""

    child_id: str
    endpoint: str  # host:port format
    last_seen: float
    active_jobs: Set[int]
    total_jobs_sent: int = 0
    total_jobs_completed: int = 0


class ParentValidatorServer(validator_jobs_capnp.ValidatorJobsService.Server):
    """Parent validator server that distributes jobs to child validators"""

    def __init__(self, parent_validator):
        self.parent_validator = parent_validator
        self.children: Dict[str, ChildValidator] = {}
        self.pending_jobs: List[EvaluationJob] = []
        self.job_assignment_lock = asyncio.Lock()
        self.snowflake_gen = SnowflakeGenerator(42)
        logger.info("Parent validator server initialized")

    async def registerChild(self, childId, endpoint, **kwargs):
        """Register a child validator"""
        try:
            child_id = str(childId)
            endpoint_str = str(endpoint)

            logger.info(
                f"Child validator {child_id} registering with endpoint {endpoint_str}"
            )

            # Register the child (skip connectivity test for now to avoid KJ loop issues)
            self.children[child_id] = ChildValidator(
                child_id=child_id,
                endpoint=endpoint_str,
                last_seen=time.time(),
                active_jobs=set(),
            )

            logger.info(f"Successfully registered child validator {child_id}")
            # Create proper Cap'n Proto response using schema
            response = validator_jobs_capnp.RegisterChildResponse.new_message()
            response.success = True
            response.message = "Registration successful"
            logger.info(f"Response to child {child_id}: {response}")
            return response

        except Exception as e:
            logger.error(f"Error registering child {childId}: {e}")
            # Create proper Cap'n Proto response using schema
            response = validator_jobs_capnp.RegisterChildResponse.new_message()
            response.success = False
            response.message = f"Registration failed: {str(e)}"
            return response

    async def requestJobs(self, childId, maxJobs, **kwargs):
        """Child validator requests jobs"""
        try:
            child_id = str(childId)
            max_jobs = int(maxJobs)

            if child_id not in self.children:
                logger.warning(f"Unknown child validator {child_id} requesting jobs")
                return {"jobs": []}

            # Update last seen time
            self.children[child_id].last_seen = time.time()

            async with self.job_assignment_lock:
                # Get available jobs for this child
                available_jobs = []
                jobs_to_assign = min(max_jobs, len(self.pending_jobs))

                for i in range(jobs_to_assign):
                    job = self.pending_jobs.pop(0)
                    available_jobs.append(job)
                    self.children[child_id].active_jobs.add(job.id)
                    self.children[child_id].total_jobs_sent += 1

                # Convert jobs to RPC format
                job_data_list = []
                for job in available_jobs:
                    job_data = {
                        "jobId": job.id,
                        "submissionId": job.submission_id,
                        "minerHotkey": job.miner_hotkey,
                        "hfRepoId": job.hf_repo_id,
                        "hfRepoCommit": job.hf_repo_commit or "",
                        "envProvider": job.env_provider,
                        "envName": job.env_name,
                        "logsPath": job.logs_path,
                        "randomSeed": job.random_seed or 0,
                        "maxRetries": job.max_retries,
                        "retryCount": job.retry_count,
                        "createdAt": int(job.created_at.timestamp() * 1000),
                    }
                    job_data_list.append(job_data)

                logger.info(f"Sending {len(job_data_list)} jobs to child {child_id}")
                return {"jobs": job_data_list}

        except Exception as e:
            logger.error(f"Error processing job request from {childId}: {e}")
            return {"jobs": []}

    async def reportJobCompletion(self, childId, jobId, success, result, **kwargs):
        """Child validator reports job completion"""
        try:
            child_id = str(childId)
            job_id = int(jobId)
            success_bool = bool(success)
            result_str = str(result)

            if child_id not in self.children:
                logger.warning(
                    f"Unknown child validator {child_id} reporting completion"
                )
                return {"acknowledged": False}

            # Update child state
            child = self.children[child_id]
            child.last_seen = time.time()

            if job_id in child.active_jobs:
                child.active_jobs.remove(job_id)
                child.total_jobs_completed += 1

                # Update job status in database if we have db_manager
                if self.parent_validator.db_manager:
                    try:
                        status = (
                            EvaluationStatus.COMPLETED
                            if success_bool
                            else EvaluationStatus.FAILED
                        )
                        self.parent_validator.db_manager.update_evaluation_job(
                            job_id,
                            {
                                "status": status,
                                "eval_end": datetime.now(),
                                "updated_at": datetime.now(),
                            },
                        )
                        logger.info(f"Updated job {job_id} status to {status}")
                    except Exception as e:
                        logger.error(f"Failed to update job {job_id} in database: {e}")

                logger.info(
                    f"Child {child_id} completed job {job_id} with success={success_bool}"
                )
                return {"acknowledged": True}
            else:
                logger.warning(
                    f"Child {child_id} reported completion for unknown job {job_id}"
                )
                return {"acknowledged": False}

        except Exception as e:
            logger.error(f"Error processing completion report from {childId}: {e}")
            return {"acknowledged": False}

    async def receiveJob(self, job, **kwargs):
        """Direct job push to child (not implemented in parent, only in child)"""
        # Parent validators don't receive jobs, so we return rejection
        return {"accepted": False, "message": "Parent validators don't receive jobs"}

    async def ping(self, **kwargs):
        """Health check endpoint"""
        return {
            "pong": f"Parent validator pong - {len(self.children)} children registered"
        }

    async def _test_child_connectivity(self, endpoint: str):
        """Test if we can connect to a child validator"""
        host, port = endpoint.split(":")
        port = int(port)

        # Try to create a connection (KJ loop should already be running)
        stream = await capnp.AsyncIoStream.create_connection(host=host, port=port)
        client = capnp.TwoPartyClient(stream)
        child_service = client.bootstrap().cast_as(
            validator_jobs_capnp.ValidatorJobsService
        )

        # Test with a ping
        await child_service.ping()

        # Close connection
        await stream.close()

    def add_job_to_queue(self, commitment: ChainCommitmentResponse):
        """Add a job from chain commitment to the pending queue"""
        try:
            job_id = next(self.snowflake_gen)
            submission_id = next(self.snowflake_gen)

            job = EvaluationJob(
                id=job_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=EvaluationStatus.QUEUED,
                submission_id=submission_id,
                miner_hotkey=commitment.hotkey,  # Fixed: use hotkey instead of miner_hotkey
                hf_repo_id=commitment.data.repo_id,
                hf_repo_commit=commitment.data.version,  # Using version as commit
                env_provider=str(
                    commitment.data.provider.value
                ),  # Convert enum to string
                env_name="default",  # TODO: extract from commitment if available
                container_id=None,
                ray_worker_id=None,
                retry_count=0,
                max_retries=3,
                logs_path="./data/logs",
                random_seed=None,
                eval_start=None,
                eval_end=None,
            )

            # Store in database if available
            if self.parent_validator.db_manager:
                try:
                    self.parent_validator.db_manager.create_evaluation_job(job)
                    logger.debug(f"Stored job {job_id} in database")
                except Exception as e:
                    logger.error(f"Failed to store job {job_id} in database: {e}")

            # Add to pending queue
            self.pending_jobs.append(job)
            logger.info(f"Added job {job_id} for miner {commitment.hotkey} to queue")

        except Exception as e:
            logger.error(f"Failed to create job from commitment: {e}")

    def get_child_stats(self) -> Dict:
        """Get statistics about child validators"""
        stats = {
            "total_children": len(self.children),
            "active_children": 0,
            "pending_jobs": len(self.pending_jobs),
            "children": [],
        }

        current_time = time.time()
        for child_id, child in self.children.items():
            is_active = (current_time - child.last_seen) < 300  # 5 minutes
            if is_active:
                stats["active_children"] += 1

            stats["children"].append(
                {
                    "child_id": child_id,
                    "endpoint": child.endpoint,
                    "active": is_active,
                    "last_seen": child.last_seen,
                    "active_jobs": len(child.active_jobs),
                    "total_jobs_sent": child.total_jobs_sent,
                    "total_jobs_completed": child.total_jobs_completed,
                }
            )

        return stats


async def serve_parent_validator(parent_validator, address="127.0.0.1", port=8001):
    """Serve the parent validator RPC server"""

    async def new_connection(stream):
        """Handler for each new client connection"""
        try:
            server = capnp.TwoPartyServer(
                stream, bootstrap=ParentValidatorServer(parent_validator)
            )
            await server.on_disconnect()
        except Exception as e:
            logger.error(f"Error handling parent server connection: {e}", exc_info=True)

    # Create the server
    logger.info("Starting parent validator RPC server...")
    server = await capnp.AsyncIoStream.create_server(new_connection, address, port)
    logger.info(f"Parent validator RPC server listening on {address}:{port}")

    try:
        async with server:
            await server.serve_forever()
    except Exception as e:
        logger.error(f"Parent server error: {e}", exc_info=True)
    finally:
        logger.info("Parent validator server shutting down")


def start_parent_server(parent_validator, address="127.0.0.1", port=8001):
    """Start parent server with proper asyncio event loop handling"""

    async def run_server_with_kj():
        async with capnp.kj_loop():
            await serve_parent_validator(parent_validator, address, port)

    try:
        asyncio.run(run_server_with_kj())
    except KeyboardInterrupt:
        logger.info("Parent server stopped by user")
