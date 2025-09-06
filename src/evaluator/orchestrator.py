import asyncio
import threading
from datetime import datetime, timezone

import asyncpg
import ray
from kubernetes import client, config
from pgqueuer import PgQueuer
from pgqueuer.db import AsyncpgDriver
from pgqueuer.models import Job
from ray.util.queue import Queue

from core.log import get_logger
from core.messages import EvalJobMessage
from evaluator.config import EvaluatorConfig
from evaluator.containers import Containers
from evaluator.rollout import BenchmarkSpec, RolloutCluster
from evaluator.rpc.rpc_process import RPCProcess
from validator.db.db_manager import create_database_manager
from validator.db.models import EvaluationJob

logger = get_logger(__name__)

WAIT_TIME = 5
PROCESS_JOB_WAIT_TIME = 1
QUEUE_MAXSIZE = 100
# TODO: this might be way too long
EVAL_TIMEOUT = 3600


class Orchestrator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        logger.info(f"Orchestrator initialized with db: {self.config.pg_database}")  # pyright: ignore[reportAttributeAccessIssue]
        self.db = create_database_manager(self.config.pg_database, self.config.duck_db)  # pyright: ignore[reportAttributeAccessIssue]
        logger.info(f"Orchestrator initialized with config: {self.config}")

    async def process_job(self, job: Job):
        logger.info(f"Processing job: {job.id}")
        if job.payload:
            # TODO: there is probably a much better way to do this
            eval_job_msg = EvalJobMessage.from_bytes(job.payload)
            evaluation_job = EvaluationJob(
                id=eval_job_msg.job_id,
                competition_id=eval_job_msg.competition_id,
                submission_id=eval_job_msg.submission_id,
                miner_hotkey=eval_job_msg.miner_hotkey,
                hf_repo_id=eval_job_msg.hf_repo_id,
                env_provider=eval_job_msg.env_provider,
                benchmark_name=eval_job_msg.benchmark_name,
                config=eval_job_msg.config,
                # TODO: where should this be set?
                logs_path="./data/logs",
                # created at will be the current datetime
                created_at=datetime.now(timezone.utc),
            )  # type: ignore

            # evaluation_job = EvaluationJob.from_bytes(job.payload)
            # start a container for this evaluation job
            repo = "https://huggingface.co/" + evaluation_job.hf_repo_id
            logger.info(
                f"Creating container for job {evaluation_job.id} with repo {repo}"
            )

            containers = Containers()
            pod = containers.create_container(repo, evaluation_job.submission_id)
            logger.info(f"Created pod: {pod}")

            # Get NodePort and Node IP for direct TCP connection
            config.load_kube_config()
            k8v1api = client.CoreV1Api()
            v1 = client.CoreV1Api()
            service_name = f"submission-{evaluation_job.submission_id}"
            svc = k8v1api.read_namespaced_service(service_name, "default")
            node_port = None
            for port in svc.spec.ports:
                if port.node_port:
                    node_port = port.node_port
                    break
            if not node_port:
                raise RuntimeError(f"No nodePort found for service {service_name}")

            # Get the first node's external IP (or internal if not available)
            nodes = v1.list_node().items
            node_ip = None
            for node in nodes:
                for addr in node.status.addresses:
                    if addr.type == "ExternalIP":
                        node_ip = addr.address
                        break
                if not node_ip:
                    for addr in node.status.addresses:
                        if addr.type == "InternalIP":
                            node_ip = addr.address
                            break
                if node_ip:
                    break
            if not node_ip:
                raise RuntimeError("No node IP found in cluster")

            # wait for some time
            # TODO: why exactly do we wait here though? just going to keep a WAIT_TIME here for now
            await asyncio.sleep(WAIT_TIME)

            # Create a benchmark spec for the job (example: MT1, can be customized)
            benchmark_spec = BenchmarkSpec(
                provider=evaluation_job.env_provider,
                benchmark_name=evaluation_job.benchmark_name,
                config=evaluation_job.config,
                enable_image_obs=True,
                render_mode="rgb_array",  # No rendering
            )

            worker_to_rpc_queue = Queue(
                maxsize=QUEUE_MAXSIZE
            )  # Worker sends TO rpc process
            rpc_to_worker_queue = Queue(
                maxsize=QUEUE_MAXSIZE
            )  # RPC process sends TO worker

            cluster = RolloutCluster("eval-cluster")
            worker = cluster.create_worker(
                evaluation_job.id,
                [benchmark_spec],
                node_ip,
                node_port,
                evaluation_job.submission_id,
            )

            rpc_thread = threading.Thread(
                target=RPCProcess,
                args=(node_ip, node_port, rpc_to_worker_queue, worker_to_rpc_queue),
                daemon=True,
            )
            rpc_thread.start()

            # TODO: is this necessary? going to set job processing wait time to 1 for now
            await asyncio.sleep(PROCESS_JOB_WAIT_TIME)

            # Worker: sends to worker_to_rpc_queue, receives from rpc_to_worker_queue
            res = await worker.test_rpc.remote(worker_to_rpc_queue, rpc_to_worker_queue)
            print(f"rpc test result: {res}")

            # Start the actual evaluation
            print("Starting evaluation...")
            try:
                # Run all benchmark tasks
                evaluation_future = worker.run_all_benchmark_tasks.remote(
                    worker_to_rpc_queue, rpc_to_worker_queue
                )
                results = ray.get(
                    evaluation_future, timeout=EVAL_TIMEOUT
                )  # 1 hour timeout for evaluation

                print(f"Evaluation completed successfully with {len(results)} results")
                print(f"Summary: {len(results)} tasks completed")

                # Log success metrics
                if results:
                    total_episodes = sum(len(result.episodes) for result in results)
                    avg_success_rate = sum(
                        result.success_rate for result in results
                    ) / len(results)
                    avg_reward = sum(result.mean_reward for result in results) / len(
                        results
                    )

                    print(f"Total episodes: {total_episodes}")
                    print(f"Average success rate: {avg_success_rate:.3f}")
                    print(f"Average reward: {avg_reward:.3f}")

            except Exception as e:
                print(f"Evaluation failed: {e}")
                logger.error(f"Evaluation failed for job {evaluation_job.id}: {e}")
                raise

            logger.info(f"Processed: {evaluation_job!r}")

    async def start(self):
        logger.info("Starting orchestrator...")
        conn = await asyncpg.connect(dsn=self.config.pg_database)

        driver = AsyncpgDriver(conn)
        pgq = PgQueuer(driver)

        @pgq.entrypoint("add_job")
        async def process(job: Job) -> None:
            asyncio.get_event_loop().create_task(self.process_job(job))
            logger.info(f"Job {job.id} added to processing queue.")

        logger.info("Orchestrator is now listening for jobs...")
        await pgq.run()
        await asyncio.Future()

    def stop(self):
        logger.info("Stopping orchestrator...")
        # Add cleanup logic here
        pass


if __name__ == "__main__":
    orc = Orchestrator(EvaluatorConfig())
    asyncio.run(orc.start())
