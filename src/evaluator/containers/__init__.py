import os
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Iterable, Optional

import dotenv
import yaml
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.stream import stream

from core.db.models import SnowflakeId
from evaluator.constants import POD_LOG_TAIL_LINES

dotenv.load_dotenv()


class PodSchedulingError(RuntimeError):
    """Raised when a pod cannot be scheduled due to cluster resource limits."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


RUNNER_CONTAINER_NAME = "runner"


class Containers:
    DELETE_TIMEOUT_SECONDS = 60
    DELETE_RETRY_INTERVAL = 5
    CONTAINER_RETRY_COUNT = 2
    # TODO: use proper duration types rather than ints
    POD_READY_TIMEOUT_SECONDS = 600
    POD_POLL_INTERVAL_SECONDS = 2.0
    POD_UNSCHEDULABLE_GRACE_SECONDS = 10
    FAILED_POD_LOG_CACHE_TTL_SECONDS = 600

    _failed_pod_log_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
    _log_cache_lock: Lock = Lock()

    @staticmethod
    def build_resource_name(submission_id: SnowflakeId, job_id: SnowflakeId) -> str:
        """Construct a unique resource name for a submission/job combination."""

        return f"submission-{submission_id}-job-{job_id}"

    @staticmethod
    def _set_env(container_spec: dict, name: str, value: str | None) -> None:
        """Ensure an environment variable is set on a container spec."""
        if value is None:
            return

        env_list = container_spec.setdefault("env", [])
        for env in env_list:
            if env.get("name") == name:
                env["value"] = value
                return

        env_list.append({"name": name, "value": value})

    @staticmethod
    def _log_pod_events(core_api: client.CoreV1Api, pod_name: str) -> None:
        """List recent Kubernetes events for the given pod."""

        try:
            field_selector = f"involvedObject.name={pod_name}"
            events = core_api.list_namespaced_event(
                namespace="default", field_selector=field_selector
            )
            if events.items:
                print(f"Events for pod {pod_name}:")
                for event in events.items:
                    timestamp = getattr(event, "last_timestamp", None)
                    print(f"  {timestamp}: {event.reason} - {event.message}")
            else:
                print(f"No events found for pod {pod_name}")
        except Exception as e:
            print(f"Error retrieving pod events for {pod_name}: {e}")

    @staticmethod
    def _collect_pod_container_logs(
        core_api: client.CoreV1Api,
        pod_name: str,
        *,
        namespace: str = "default",
        tail_lines: int = 200,
        container_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Return raw log text for all containers on a pod."""

        result: Dict[str, Any] = {
            "pod_name": pod_name,
            "namespace": namespace,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "containers": {},
        }

        try:
            pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
        except ApiException as exc:
            result["error"] = f"{exc.status}: {exc.reason}"
            return result
        except Exception as exc:
            result["error"] = str(exc)
            return result

        container_specs = []
        if pod.spec:
            if pod.spec.init_containers:
                container_specs.extend(pod.spec.init_containers)
            if pod.spec.containers:
                container_specs.extend(pod.spec.containers)

        if not container_specs:
            result["warning"] = "Pod spec contained no containers"
            return result

        container_filter = None
        if container_names:
            container_filter = {name for name in container_names if name}

        matched_any = False
        for container in container_specs:
            container_name = getattr(container, "name", None)
            if not container_name:
                continue

            if container_filter is not None and container_name not in container_filter:
                continue

            matched_any = True
            entry: Dict[str, Optional[str]] = {}
            try:
                log_text = core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    container=container_name,
                    tail_lines=tail_lines,
                )
                entry["log"] = log_text
            except ApiException as exc:
                entry["error"] = f"{exc.status}: {exc.reason}"
            except Exception as exc:
                entry["error"] = str(exc)

            result["containers"][container_name] = entry

        if container_filter is not None and not matched_any:
            result["warning"] = (
                f"No containers matched filter: {sorted(container_filter)}"
            )

        return result

    @staticmethod
    def _print_pod_logs(core_api: client.CoreV1Api, pod_name: str) -> None:
        """Print logs for init and main containers."""

        try:
            logs = Containers._collect_pod_container_logs(
                core_api, pod_name, tail_lines=100
            )
        except Exception as exc:
            print(f"Error retrieving container logs for {pod_name}: {exc}")
            return

        if logs.get("error"):
            print(
                f"Failed to retrieve pod logs for {pod_name}: {logs['error']}",
            )
            return

        containers = logs.get("containers", {})
        if not containers:
            print(f"No container logs available for {pod_name}")
            return

        for container_name, entry in containers.items():
            print(f"Logs for {pod_name}/{container_name}:")
            log_text = entry.get("log")
            if log_text:
                print(log_text)
            else:
                print(f"No logs available ({entry.get('error', 'unknown error')})")

    def _handle_failed_pod(
        self,
        core_api: client.CoreV1Api,
        pod_name: str,
        submission_id: SnowflakeId,
        job_id: SnowflakeId,
    ) -> None:
        """Capture diagnostics and tear down a failed pod."""
        cached_runner_logs: Optional[Dict[str, Any]] = None
        try:
            cached_runner_logs = self._collect_pod_container_logs(
                core_api,
                pod_name,
                tail_lines=POD_LOG_TAIL_LINES,
                container_names=[RUNNER_CONTAINER_NAME],
            )
        except Exception as exc:  # pragma: no cover - cache fetch failure
            print(f"Failed to pre-collect runner logs for {pod_name}: {exc}")

        if cached_runner_logs:
            cached_runner_logs["cached_before_deletion"] = True
            self._cache_failed_pod_logs(pod_name, cached_runner_logs)

        self._print_pod_logs(core_api, pod_name)
        self._log_pod_events(core_api, pod_name)

        try:
            self.cleanup_container(submission_id, job_id, wait=False)
        except Exception as exc:
            print(f"Error cleaning up failed pod {pod_name}: {exc}")

    def create_container(
        self,
        submission_id: SnowflakeId,
        job_id: SnowflakeId,
        *,
        archive_url: str,
        archive_sha256: str | None = None,
        port: int = 8000,
    ) -> str:
        print(f"Creating container for submission {submission_id} (job {job_id})")
        print(f"Fetching artifact from {archive_url}")
        config.load_kube_config()
        k8v1api = client.CoreV1Api()

        container_name = self.build_resource_name(submission_id, job_id)

        # Load podspec from YAML template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        podspec_path = os.path.join(current_dir, "podspec.yaml")

        with open(podspec_path, "r") as f:
            pod_template = yaml.safe_load(f)

        # Update the pod configuration
        pod_template["metadata"]["name"] = container_name

        # Set the runtime to kata containers
        # pod_template["spec"]["runtimeClassName"] = "kata"

        # Set the submission archive URL as an annotation
        if not pod_template["metadata"].get("annotations"):
            pod_template["metadata"]["annotations"] = {}
        pod_template["metadata"]["annotations"]["submission-archive-url"] = archive_url

        # Override to use local miner agent image if OVERRIDE_IMAGE env var is set
        if os.getenv("OVERRIDE_IMAGE"):
            print(f"Overriding image to {os.getenv('OVERRIDE_IMAGE')}")
            print(f"Pod template:\n {pod_template}")
            for container in pod_template["spec"]["containers"]:
                if container["name"] == "runner":
                    container["image"] = "miner-agent"
                    # Only for dev
                    container["imagePullPolicy"] = "Never"

            for container in pod_template["spec"]["initContainers"]:
                if container["name"] in {"fetch-submission", "restrict-egress"}:
                    container["image"] = "miner-agent"
                    # Only for dev
                    container["imagePullPolicy"] = "Never"

        for container in pod_template["spec"]["initContainers"]:
            if container["name"] == "fetch-submission":
                self._set_env(container, "SUBMISSION_ARCHIVE_URL", archive_url)
                self._set_env(container, "SUBMISSION_ARCHIVE_SHA256", archive_sha256)
        for container in pod_template["spec"]["containers"]:
            if container["name"] == "runner":
                self._set_env(container, "SUBMISSION_ARCHIVE_URL", archive_url)
                self._set_env(container, "SUBMISSION_ARCHIVE_SHA256", archive_sha256)

        # Create the pod from the template
        print(f"Creating pod from template for {container_name}")
        # Extract only the metadata and spec from the template, excluding apiVersion and kind
        pod_config = {
            "metadata": pod_template["metadata"],
            "spec": pod_template["spec"],
        }
        pod = client.V1Pod(**pod_config)

        pod_created = False
        pod_ready = False

        try:
            # Create the pod in the default namespace
            print("Submitting pod to Kubernetes...")
            for attempt in range(self.CONTAINER_RETRY_COUNT):
                try:
                    k8v1api.create_namespaced_pod(namespace="default", body=pod)
                    pod_created = True
                    break
                except ApiException as api_exc:
                    if (
                        api_exc.status == 409
                        and "AlreadyExists" in api_exc.reason
                        and attempt == 0
                    ):
                        print(
                            f"Pod {container_name} already exists; waiting for prior instance to terminate"
                        )
                        self._wait_for_pod_absence(k8v1api, container_name)
                        continue
                    raise
            print(f"Pod {container_name} created, waiting for it to start...")

            # Wait for the pod to be running before attempting to connect
            start_time = time.time()
            last_phase = None
            last_unsched_message = None
            unsched_since: Optional[float] = None

            while True:
                try:
                    pod_info = k8v1api.read_namespaced_pod(container_name, "default")
                    phase = pod_info.status.phase  # type: ignore
                    if phase != last_phase:
                        print(f"Pod {container_name} status changed: {phase}")
                        last_phase = phase

                    if phase == "Running":
                        print(f"Pod {container_name} is now running!")
                        pod_ready = True
                        break
                    elif phase in ("Failed", "Succeeded"):
                        print(f"Pod {container_name} reached terminal state: {phase}")
                        self._log_pod_events(k8v1api, container_name)
                        raise RuntimeError(
                            f"Pod {container_name} terminated unexpectedly with phase {phase}"
                        )
                    else:
                        conditions = pod_info.status.conditions or []  # type: ignore
                        unsched_messages = []
                        for condition in conditions:
                            if (
                                condition.type == "PodScheduled"
                                and condition.status == "False"
                            ):
                                detail = (
                                    condition.message
                                    or condition.reason
                                    or "unspecified"
                                )
                                unsched_messages.append(detail)
                        if unsched_messages:
                            condensed = "; ".join(sorted(set(unsched_messages)))
                            if condensed != last_unsched_message:
                                print(
                                    f"Pod {container_name} pending scheduling: {condensed}"
                                )
                                last_unsched_message = condensed
                                unsched_since = time.time()
                            elif unsched_since is None:
                                unsched_since = time.time()

                            if (
                                unsched_since is not None
                                and time.time() - unsched_since
                                >= self.POD_UNSCHEDULABLE_GRACE_SECONDS
                            ):
                                raise PodSchedulingError(
                                    f"Pod {container_name} unschedulable: {condensed}"
                                )
                        else:
                            unsched_since = None

                    if (time.time() - start_time) >= self.POD_READY_TIMEOUT_SECONDS:
                        self._log_pod_events(k8v1api, container_name)
                        raise RuntimeError(
                            f"Pod {container_name} did not become ready within {self.POD_READY_TIMEOUT_SECONDS}s"
                        )

                    time.sleep(self.POD_POLL_INTERVAL_SECONDS)
                except Exception as exc:
                    if isinstance(exc, RuntimeError):
                        raise
                    print(
                        f"Waiting for pod to start (elapsed {int(time.time() - start_time)}s): {exc}"
                    )
                    time.sleep(self.POD_POLL_INTERVAL_SECONDS)

        except Exception:
            if pod_created:
                self._handle_failed_pod(k8v1api, container_name, submission_id, job_id)
            raise

        if pod_ready:
            self._print_pod_logs(k8v1api, container_name)
            self._log_pod_events(k8v1api, container_name)

        # Connect to the pod to check if it's working correctly
        resp = stream(
            k8v1api.connect_get_namespaced_pod_exec,
            container_name,
            "default",
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            command=["echo", "Container is running"],
        )

        print(f"Container response: {resp}")

        # After pod is running, create a Service to expose it on the specified port
        service_name = container_name  # Use same name for service
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": service_name},
            "spec": {
                "selector": {"app": container_name},
                "ports": [{"protocol": "TCP", "port": port, "targetPort": port}],
                "type": "NodePort",
            },
        }

        # Patch the pod to add the label for service selector
        try:
            k8v1api.patch_namespaced_pod(
                name=container_name,
                namespace="default",
                body={"metadata": {"labels": {"app": container_name}}},
            )
            print(f"Patched pod {container_name} with label app={container_name}")
        except Exception as e:
            print(f"Error patching pod with label: {e}")

        # Create the service
        try:
            k8v1api.create_namespaced_service(
                namespace="default", body=service_manifest
            )
            print(f"Service {service_name} created to expose pod on port {port}")
        except ApiException as api_exc:
            if api_exc.status == 409 and "AlreadyExists" in api_exc.reason:
                print(
                    f"Service {service_name} already exists; waiting for prior service to be removed"
                )
                self._wait_for_service_absence(k8v1api, service_name)
                k8v1api.create_namespaced_service(
                    namespace="default", body=service_manifest
                )
                print(f"Service {service_name} recreated after cleanup")
            else:
                print(f"Error creating service: {api_exc}")
        except Exception as e:
            print(f"Error creating service: {e}")

        return container_name

    def destroy_container(
        self, submission_id: SnowflakeId, job_id: SnowflakeId
    ) -> None:
        """Delete the pod associated with a submission/job pair."""
        config.load_kube_config()
        v1 = client.CoreV1Api()

        container_name = self.build_resource_name(submission_id, job_id)
        try:
            v1.delete_namespaced_pod(name=container_name, namespace="default")
            print(f"Pod {container_name} deleted successfully")
        except Exception as e:
            print(f"Error deleting pod {container_name}: {e}")

    def cleanup_container(
        self,
        submission_id: SnowflakeId,
        job_id: SnowflakeId,
        *,
        wait: bool = False,
    ) -> None:
        """Clean up both the pod and service for a submission.

        Args:
            submission_id: The ID of the submission whose resources to clean up
            wait: Block until resources are fully removed
        """
        config.load_kube_config()
        v1 = client.CoreV1Api()

        container_name = self.build_resource_name(submission_id, job_id)
        service_name = container_name  # Service has the same name as the pod

        # Delete the service first
        try:
            v1.delete_namespaced_service(name=service_name, namespace="default")
            print(f"Service {service_name} deleted successfully")
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if service doesn't exist
                print(f"Error deleting service {service_name}: {e}")

        # Delete the pod
        try:
            v1.delete_namespaced_pod(name=container_name, namespace="default")
            print(f"Pod {container_name} deleted successfully")
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if pod doesn't exist
                print(f"Error deleting pod {container_name}: {e}")

        if wait:
            self._wait_for_service_absence(v1, service_name)
            self._wait_for_pod_absence(v1, container_name)

    def _wait_for_pod_absence(
        self, core_api: client.CoreV1Api, pod_name: str, namespace: str = "default"
    ) -> None:
        deadline = time.time() + self.DELETE_TIMEOUT_SECONDS
        while True:
            try:
                pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
                if pod.metadata and pod.metadata.deletion_timestamp:
                    print(
                        f"Pod {pod_name} still terminating; waiting for cleanup to finish"
                    )
                else:
                    print(f"Pod {pod_name} still present; deleting existing instance")
                    try:
                        core_api.delete_namespaced_pod(
                            name=pod_name, namespace=namespace
                        )
                    except ApiException as delete_exc:
                        if delete_exc.status != 404:
                            raise
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for pod {pod_name} to be deleted"
                    )
                time.sleep(self.DELETE_RETRY_INTERVAL)
            except ApiException as api_exc:
                if api_exc.status == 404:
                    return
                raise

    def collect_container_logs(
        self,
        submission_id: SnowflakeId,
        job_id: SnowflakeId,
        *,
        namespace: str = "default",
        tail_lines: int = 200,
    ) -> Dict[str, Any]:
        """Collect logs for all containers associated with a submission pod."""

        config.load_kube_config()
        core_api = client.CoreV1Api()
        pod_name = self.build_resource_name(submission_id, job_id)
        live_logs = self._collect_pod_container_logs(
            core_api,
            pod_name,
            namespace=namespace,
            tail_lines=tail_lines,
            # NOTE: we only upload logs from the "runner" container
            container_names=[RUNNER_CONTAINER_NAME],
        )

        if self._should_use_cached_logs(live_logs):
            cached_logs = self._pop_cached_pod_logs(pod_name)
            if cached_logs is not None:
                return cached_logs

        return live_logs

    @classmethod
    def _cache_failed_pod_logs(cls, pod_name: str, logs: Dict[str, Any]) -> None:
        """Store runner logs for pods that are about to be torn down."""

        expiry = time.time() + cls.FAILED_POD_LOG_CACHE_TTL_SECONDS
        with cls._log_cache_lock:
            cls._prune_failed_log_cache_locked(time.time())
            cls._failed_pod_log_cache[pod_name] = (expiry, logs)

    @classmethod
    def _pop_cached_pod_logs(cls, pod_name: str) -> Optional[Dict[str, Any]]:
        """Return cached pod logs if they were collected before deletion."""

        with cls._log_cache_lock:
            cls._prune_failed_log_cache_locked(time.time())
            cached = cls._failed_pod_log_cache.pop(pod_name, None)

        if cached is None:
            return None

        _, logs = cached
        return logs

    @classmethod
    def _prune_failed_log_cache_locked(cls, now: float) -> None:
        expired = [
            pod_name
            for pod_name, (expiry, _) in cls._failed_pod_log_cache.items()
            if expiry <= now
        ]
        for pod_name in expired:
            cls._failed_pod_log_cache.pop(pod_name, None)

    @staticmethod
    def _should_use_cached_logs(log_data: Dict[str, Any]) -> bool:
        """Determine if cached logs should be preferred over live fetch."""

        if not log_data:
            return True
        if log_data.get("containers"):
            return False
        return bool(log_data.get("error") or log_data.get("warning"))

    def _wait_for_service_absence(
        self, core_api: client.CoreV1Api, service_name: str, namespace: str = "default"
    ) -> None:
        deadline = time.time() + self.DELETE_TIMEOUT_SECONDS
        while True:
            try:
                core_api.read_namespaced_service(name=service_name, namespace=namespace)
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for service {service_name} to be deleted"
                    )
                try:
                    core_api.delete_namespaced_service(
                        name=service_name, namespace=namespace
                    )
                except ApiException as delete_exc:
                    if delete_exc.status != 404:
                        raise
                print(
                    f"Service {service_name} still present; waiting for deletion to complete"
                )
                time.sleep(self.DELETE_RETRY_INTERVAL)
            except ApiException as api_exc:
                if api_exc.status == 404:
                    return
                raise
