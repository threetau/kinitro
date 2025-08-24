#!/usr/bin/env python3
"""
Child Validator RPC Server and Client for receiving jobs from parent
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Optional
from threading import Thread

import capnp
import asyncpg
from pgqueuer.db import AsyncpgDriver
from pgqueuer.queries import Queries

from core.db.models import EvaluationJob, EvaluationStatus

schema_file = os.path.join(os.path.dirname(__file__), "validator_jobs.capnp")
validator_jobs_capnp = capnp.load(schema_file)

logger = logging.getLogger("validator.rpc.child_server")


class ChildValidatorServer(validator_jobs_capnp.ValidatorJobsService.Server):
    """Child validator server that receives jobs from parent"""

    def __init__(self, child_validator):
        self.child_validator = child_validator
        self.received_jobs = []
        logger.info("Child validator server initialized")

    async def registerChild(self, childId, endpoint, **kwargs):
        """Not implemented for child validators"""
        return {
            "success": False,
            "message": "Child validators don't register other children",
        }

    async def requestJobs(self, childId, maxJobs, **kwargs):
        """Not implemented for child validators"""
        return {"jobs": []}

    async def reportJobCompletion(self, childId, jobId, success, result, **kwargs):
        """Not implemented for child validators"""
        return {"acknowledged": False}

    async def receiveJob(self, job, **kwargs):
        """Receive a job from parent validator"""
        try:
            # Create a complete EvaluationJob object for queuing
            evaluation_job = EvaluationJob(
                id=int(job.jobId),
                created_at=datetime.fromtimestamp(int(job.createdAt) / 1000),
                updated_at=datetime.now(),
                status=EvaluationStatus.QUEUED,
                submission_id=int(job.submissionId),
                miner_hotkey=str(job.minerHotkey),
                hf_repo_id=str(job.hfRepoId),
                hf_repo_commit=str(job.hfRepoCommit) if job.hfRepoCommit else None,
                env_provider=str(job.envProvider),
                env_name=str(job.envName),
                container_id=None,
                ray_worker_id=None,
                retry_count=int(job.retryCount),
                max_retries=int(job.maxRetries),
                logs_path=str(job.logsPath),
                random_seed=int(job.randomSeed) if job.randomSeed else None,
                eval_start=None,
                eval_end=None,
            )

            # Store in local database if available
            if self.child_validator.db_manager:
                try:
                    # For now, skip database storage due to computed field issues
                    # TODO: Fix database storage to handle computed fields properly
                    logger.info(
                        f"Skipping database storage for job {evaluation_job.id} (will be implemented later)"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to store job {evaluation_job.id} in local database: {e}"
                    )
                    # Create proper Cap'n Proto response using JobResponse
                    response = validator_jobs_capnp.JobResponse.new_message()
                    response.accepted = False
                    response.message = f"Database error: {str(e)}"
                    return response

            # Queue the job for processing using pgqueuer
            try:
                await self._queue_job_for_processing(evaluation_job)
                logger.info(
                    f"Successfully queued job {evaluation_job.id} for processing"
                )
                # Create proper Cap'n Proto response using JobResponse
                response = validator_jobs_capnp.JobResponse.new_message()
                response.accepted = True
                response.message = "Job accepted and queued"
                return response

            except Exception as e:
                logger.error(f"Failed to queue job {evaluation_job.id}: {e}")
                # Create proper Cap'n Proto response using JobResponse
                response = validator_jobs_capnp.JobResponse.new_message()
                response.accepted = False
                response.message = f"Queue error: {str(e)}"
                return response

        except Exception as e:
            logger.error(f"Error receiving job: {e}")
            # Create proper Cap'n Proto response using JobResponse
            response = validator_jobs_capnp.JobResponse.new_message()
            response.accepted = False
            response.message = f"Processing error: {str(e)}"
            return response

    async def ping(self, **kwargs):
        """Health check endpoint"""
        return {
            "pong": f"Child validator pong - {len(self.received_jobs)} jobs received"
        }

    async def _queue_job_for_processing(self, job: EvaluationJob):
        """Queue job using pgqueuer for processing by orchestrator"""
        try:
            # Connect to PostgreSQL and queue the job
            pg_url = self.child_validator.config.settings.get("pg_database")
            if not pg_url:
                raise ValueError("No PostgreSQL database URL configured")

            conn = await asyncpg.connect(pg_url)
            driver = AsyncpgDriver(conn)
            queries = Queries(driver)

            # Serialize job as bytes
            job_bytes = job.to_bytes()

            # Enqueue with appropriate priority and delay
            await queries.enqueue(["add_job"], [job_bytes], [0])

            await conn.close()
            logger.debug(f"Successfully enqueued job {job.id} using pgqueuer")

        except Exception as e:
            logger.error(f"Failed to enqueue job {job.id}: {e}")
            raise


class ParentValidatorClient:
    """Client for child validator to communicate with parent validator"""

    def __init__(self, parent_host="localhost", parent_port=8001):
        self.parent_host = parent_host
        self.parent_port = parent_port
        self.client = None
        self.parent_service = None
        self.stream = None
        self._child_id = None
        self._child_endpoint = None

    async def connect(self, child_id: str, child_endpoint: str):
        """Connect to parent validator and register as child"""
        try:
            logger.info(
                f"Connecting to parent at {self.parent_host}:{self.parent_port}"
            )
            self._child_id = child_id
            self._child_endpoint = child_endpoint

            # Create connection to parent (assumes KJ loop is already running)
            self.stream = await capnp.AsyncIoStream.create_connection(
                host=self.parent_host, port=self.parent_port
            )
            self.client = capnp.TwoPartyClient(self.stream)
            self.parent_service = self.client.bootstrap().cast_as(
                validator_jobs_capnp.ValidatorJobsService
            )

            # Register with parent
            result = await self.parent_service.registerChild(child_id, child_endpoint)

            # Extract response fields from the RegisterChildResponse struct
            logger.info(f"Received response from parent: {result}")
            # The response is wrapped in a 'response' field according to schema
            response = result.response if hasattr(result, "response") else result
            success = bool(response.success) if hasattr(response, "success") else False
            message = (
                str(response.message)
                if hasattr(response, "message")
                else "Unknown error"
            )

            if success:
                logger.info(
                    f"Successfully registered with parent validator as {child_id}"
                )
                return True
            else:
                logger.error(f"Failed to register with parent: {message}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to parent validator: {e}", exc_info=True)
            return False

    async def request_jobs(self, max_jobs: int = 5) -> list:
        """Request jobs from parent validator"""
        if not self.parent_service or not self._child_id:
            raise ValueError("Not connected to parent validator")

        try:
            result = await self.parent_service.requestJobs(self._child_id, max_jobs)

            # Extract jobs list properly from Cap'n Proto response
            jobs = list(result.jobs) if hasattr(result, "jobs") else []
            logger.info(f"Received {len(jobs)} jobs from parent validator")
            return jobs

        except Exception as e:
            logger.error(f"Error requesting jobs from parent: {e}")
            return []

    async def report_job_completion(
        self, job_id: int, success: bool, result_message: str = ""
    ):
        """Report job completion to parent validator"""
        if not self.parent_service or not self._child_id:
            logger.warning(
                "Not connected to parent validator, cannot report completion"
            )
            return False

        try:
            result = await self.parent_service.reportJobCompletion(
                self._child_id, job_id, success, result_message
            )

            # Extract acknowledgment properly from Cap'n Proto response
            acknowledged = (
                bool(result.acknowledged) if hasattr(result, "acknowledged") else False
            )

            if acknowledged:
                logger.info(f"Parent acknowledged completion of job {job_id}")
                return True
            else:
                logger.warning(f"Parent did not acknowledge completion of job {job_id}")
                return False

        except Exception as e:
            logger.error(f"Error reporting job completion to parent: {e}")
            return False

    async def ping_parent(self) -> bool:
        """Ping parent validator to check connectivity"""
        if not self.parent_service:
            return False

        try:
            result = await self.parent_service.ping()
            pong_response = (
                str(result.pong) if hasattr(result, "pong") else "No response"
            )
            logger.debug(f"Parent ping response: {pong_response}")
            return True
        except Exception as e:
            logger.error(f"Failed to ping parent validator: {e}")
            return False

    async def close(self):
        """Close connection to parent validator"""
        try:
            if self.stream:
                await self.stream.close()
            self.client = None
            self.parent_service = None
            self.stream = None
            self._child_id = None
            self._child_endpoint = None
        except Exception as e:
            logger.warning(f"Error during close: {e}")


class ChildValidatorManager:
    """Manages child validator operations - both serving and client communication"""

    def __init__(
        self,
        child_validator,
        child_id: str,
        child_port: int = 8002,
        parent_host: str = "localhost",
        parent_port: int = 8001,
    ):
        self.child_validator = child_validator
        self.child_id = child_id
        self.child_port = child_port
        self.parent_host = parent_host
        self.parent_port = parent_port

        self.server_running = False
        self.server_task = None
        self.client = ParentValidatorClient(parent_host, parent_port)

        # Job polling configuration
        self.job_polling_enabled = True
        self.job_polling_interval = 30  # seconds
        self.max_jobs_per_request = 5

    async def start(self):
        """Start child validator server and connect to parent"""
        try:
            # Start the child validator server in a separate task
            self.server_task = asyncio.create_task(self._run_server())

            # Give server time to start
            await asyncio.sleep(2)

            # Connect to parent validator within the same KJ loop context
            child_endpoint = f"localhost:{self.child_port}"
            connection_success = await self.client.connect(
                self.child_id, child_endpoint
            )

            if connection_success:
                logger.info(
                    f"Child validator {self.child_id} started and connected successfully"
                )
            else:
                logger.warning(
                    f"Child validator {self.child_id} started but failed to connect to parent"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to start child validator manager: {e}")
            return False

    async def stop(self):
        """Stop child validator operations"""
        try:
            self.job_polling_enabled = False

            if self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass

            # Close client connection (avoid KJ loop issues)
            try:
                if self.client and hasattr(self.client, "close"):
                    await self.client.close()
            except Exception as e:
                logger.warning(
                    f"Error closing client connection (ignoring KJ loop issues): {e}"
                )

            logger.info(f"Child validator {self.child_id} stopped")

        except Exception as e:
            logger.error(f"Error stopping child validator manager: {e}")

    async def _run_server(self):
        """Run the child validator server"""
        async with capnp.kj_loop():

            async def new_connection(stream):
                try:
                    server = capnp.TwoPartyServer(
                        stream, bootstrap=ChildValidatorServer(self.child_validator)
                    )
                    await server.on_disconnect()
                except Exception as e:
                    logger.error(
                        f"Error handling child server connection: {e}", exc_info=True
                    )

            server = await capnp.AsyncIoStream.create_server(
                new_connection, "127.0.0.1", self.child_port
            )
            logger.info(
                f"Child validator server listening on 127.0.0.1:{self.child_port}"
            )

            try:
                self.server_running = True
                async with server:
                    await server.serve_forever()
            except Exception as e:
                logger.error(f"Child server error: {e}", exc_info=True)
            finally:
                self.server_running = False
                logger.info("Child validator server shut down")

    async def _job_polling_loop(self):
        """Periodically poll parent for new jobs"""
        logger.info(
            f"Starting job polling loop with {self.job_polling_interval}s interval"
        )

        while self.job_polling_enabled:
            try:
                # Request jobs from parent (assumes KJ loop is running)
                if self.client.parent_service:
                    jobs = await self.client.request_jobs(self.max_jobs_per_request)
                    if jobs:
                        logger.info(
                            f"Received {len(jobs)} jobs from parent, processing..."
                        )
                        for job in jobs:
                            await self._process_received_job(job)
                else:
                    logger.debug("Not connected to parent, skipping job request")

                await asyncio.sleep(self.job_polling_interval)

            except Exception as e:
                logger.error(f"Error in job polling loop: {e}")
                await asyncio.sleep(self.job_polling_interval)

    async def _process_received_job(self, job_data):
        """Process a job received from parent"""
        try:
            # Convert to EvaluationJob and queue for processing
            child_server = ChildValidatorServer(self.child_validator)
            result = await child_server.receiveJob(job_data)

            if result["accepted"]:
                logger.info(f"Successfully processed job {job_data.jobId}")
            else:
                logger.error(
                    f"Failed to process job {job_data.jobId}: {result['message']}"
                )

        except Exception as e:
            logger.error(f"Error processing received job: {e}")


async def serve_child_validator(child_validator, address="127.0.0.1", port=8002):
    """Serve the child validator RPC server"""

    async def new_connection(stream):
        """Handler for each new client connection"""
        try:
            server = capnp.TwoPartyServer(
                stream, bootstrap=ChildValidatorServer(child_validator)
            )
            await server.on_disconnect()
        except Exception as e:
            logger.error(f"Error handling child server connection: {e}", exc_info=True)

    # Create the server
    server = await capnp.AsyncIoStream.create_server(new_connection, address, port)
    logger.info(f"Child validator RPC server listening on {address}:{port}")

    try:
        async with server:
            await server.serve_forever()
    except Exception as e:
        logger.error(f"Child server error: {e}", exc_info=True)
    finally:
        logger.info("Child validator server shutting down")


def start_child_server(child_validator, address="127.0.0.1", port=8002):
    """Start child server with proper asyncio event loop handling"""

    async def run_server_with_kj():
        async with capnp.kj_loop():
            await serve_child_validator(child_validator, address, port)

    try:
        asyncio.run(run_server_with_kj())
    except KeyboardInterrupt:
        logger.info("Child server stopped by user")
