import os
import time

import yaml
from kubernetes import client, config
from kubernetes.stream import stream

from core.db.models import SnowflakeId


class Containers:
    def create_container(
        self, submission_repo: str, submission_id: SnowflakeId, port: int = 8000
    ) -> str:
        print(
            f"Creating container for submission {submission_id} from repo {submission_repo}"
        )
        config.load_kube_config()
        k8v1api = client.CoreV1Api()

        container_name = f"submission-{submission_id}"

        # Load podspec from YAML template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        podspec_path = os.path.join(current_dir, "podspec.yaml")

        with open(podspec_path, "r") as f:
            pod_template = yaml.safe_load(f)

        # Update the pod configuration
        pod_template["metadata"]["name"] = container_name

        # Set the runtime to kata containers
        # pod_template["spec"]["runtimeClassName"] = "kata"

        # Set the submission repo URL as an annotation
        if not pod_template["metadata"].get("annotations"):
            pod_template["metadata"]["annotations"] = {}
        pod_template["metadata"]["annotations"]["submission-url"] = submission_repo

        # TODO: some of these are forced for dev env
        # Update the image to use threetau/miner-agent
        # print the pod template
        print(f"Pod template:\n {pod_template}")
        for container in pod_template["spec"]["containers"]:
            if container["name"] == "runner":
                container["image"] = "miner-agent"
                # Only for dev
                container["imagePullPolicy"] = "Never"

        for container in pod_template["spec"]["initContainers"]:
            if container["name"] == "fetch-submission":
                container["image"] = "miner-agent"
                # Only for dev
                container["imagePullPolicy"] = "Never"

        # Create the pod from the template
        print(f"Creating pod from template for {container_name}")
        # Extract only the metadata and spec from the template, excluding apiVersion and kind
        pod_config = {
            "metadata": pod_template["metadata"],
            "spec": pod_template["spec"],
        }
        pod = client.V1Pod(**pod_config)

        # Create the pod in the default namespace
        print("Submitting pod to Kubernetes...")
        k8v1api.create_namespaced_pod(namespace="default", body=pod)
        print(f"Pod {container_name} created, waiting for it to start...")

        # Wait for the pod to be running before attempting to connect
        for attempt in range(30):  # Wait up to 30 seconds
            try:
                pod_info = k8v1api.read_namespaced_pod(container_name, "default")
                phase = pod_info.status.phase  # type: ignore
                print(f"Attempt {attempt + 1}/30: Pod status - {phase}")

                if phase == "Running":
                    print(f"Pod {container_name} is now running!")
                    break
                elif phase == "Pending":
                    # Get more details on why it's still pending
                    conditions = pod_info.status.conditions  # type: ignore
                    container_statuses = pod_info.status.container_statuses  # type: ignore

                    if conditions:
                        print(
                            f"Pod conditions: {[f'{c.type}: {c.status}, reason: {c.reason}' for c in conditions]}"
                        )

                    if container_statuses:
                        for container in container_statuses:
                            print(
                                f"Container {container.name} state: {container.state}"
                            )

                time.sleep(1)
            except Exception as e:
                print(f"Waiting for pod to start (attempt {attempt + 1}/30): {e}")
                time.sleep(1)

        # Get and print pod logs for both containers
        try:
            print(f"Getting init container logs for {container_name}...")
            try:
                init_logs = k8v1api.read_namespaced_pod_log(
                    name=container_name,
                    namespace="default",
                    container="fetch-submission",  # Init container name from podspec.yaml
                    tail_lines=100,
                )
                print(f"Init container logs for {container_name}:\n{init_logs}")
            except Exception as e:
                print(f"Error retrieving init container logs: {e}")

            print(f"Getting runner container logs for {container_name}...")
            try:
                runner_logs = k8v1api.read_namespaced_pod_log(
                    name=container_name,
                    namespace="default",
                    container="runner",
                    tail_lines=100,
                )
                print(f"Runner container logs for {container_name}:\n{runner_logs}")
            except Exception as e:
                print(f"Error retrieving runner container logs: {e}")

        except Exception as e:
            print(f"Error retrieving container logs: {e}")

        # Get pod events for troubleshooting
        try:
            field_selector = f"involvedObject.name={container_name}"
            events = k8v1api.list_namespaced_event(
                namespace="default", field_selector=field_selector
            )
            if events.items:
                print(f"Events for pod {container_name}:")
                for event in events.items:
                    print(f"  {event.last_timestamp}: {event.reason} - {event.message}")
            else:
                print(f"No events found for pod {container_name}")
        except Exception as e:
            print(f"Error retrieving pod events: {e}")

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
        except Exception as e:
            print(f"Error creating service: {e}")

        return container_name

    def destroy_container(self, container_id: SnowflakeId) -> None:
        """Delete the pod with the given container ID.

        Args:
            container_id: The ID of the container to destroy
        """
        config.load_kube_config()
        v1 = client.CoreV1Api()

        container_name = f"submission-{container_id}"
        try:
            v1.delete_namespaced_pod(name=container_name, namespace="default")
            print(f"Pod {container_name} deleted successfully")
        except Exception as e:
            print(f"Error deleting pod {container_name}: {e}")

    def cleanup_container(self, submission_id: SnowflakeId) -> None:
        """Clean up both the pod and service for a submission.

        Args:
            submission_id: The ID of the submission whose resources to clean up
        """
        config.load_kube_config()
        v1 = client.CoreV1Api()

        container_name = f"submission-{submission_id}"
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
