#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 <command> <target> [docker-compose args...]

Commands:
  up       Start services for the given target
  down     Stop services for the given target
  config   Show the rendered compose configuration

Targets:
  validator   Postgres + validator/evaluator (compose.base)
  backend     Backend API only (compose.backend)
  local       Full local stack (base + backend + local-dev overlays)
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

COMMAND=$1
TARGET=$2
shift 2

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMPOSE_DIR="$ROOT_DIR/deploy/docker"
BASE_COMPOSE=(-f "$COMPOSE_DIR/compose.base.yaml")
BACKEND_COMPOSE=(-f "$COMPOSE_DIR/compose.backend.yaml")
LOCAL_COMPOSE=(-f "$COMPOSE_DIR/compose.local-dev.yaml")

case "$TARGET" in
  validator)
    COMPOSE_ARGS=(${BASE_COMPOSE[@]})
    : "${ENV_FILE:=env/base.env}"
    ;;
  backend)
    COMPOSE_ARGS=(${BACKEND_COMPOSE[@]})
    : "${ENV_FILE:=env/base.env}"
    ;;
  local)
    COMPOSE_ARGS=(${BASE_COMPOSE[@]} ${BACKEND_COMPOSE[@]} ${LOCAL_COMPOSE[@]})
    : "${ENV_FILE:=env/local.env}"
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    usage
    exit 1
    ;;
esac

export ENV_FILE

# Ensure Minikube kubeconfig paths are available when running the evaluator.
if [[ -z "${HOST_KUBECONFIG:-}" ]]; then
  if [[ -f "${HOME}/.kube/config" ]]; then
    HOST_KUBECONFIG="${HOME}/.kube/config"
  elif [[ -f "${ROOT_DIR}/deploy/docker/config/local/kubeconfig.yaml" ]]; then
    HOST_KUBECONFIG="${ROOT_DIR}/deploy/docker/config/local/kubeconfig.yaml"
  fi
  export HOST_KUBECONFIG
fi

if [[ -z "${HOST_MINIKUBE_DIR:-}" ]]; then
  if [[ -d "${HOME}/.minikube" ]]; then
    HOST_MINIKUBE_DIR="${HOME}/.minikube"
  elif [[ -d "${ROOT_DIR}/deploy/docker/config/local/minikube" ]]; then
    HOST_MINIKUBE_DIR="${ROOT_DIR}/deploy/docker/config/local/minikube"
  fi
  export HOST_MINIKUBE_DIR
fi

GLOBAL_ARGS=()
CMD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--project-name|--env-file|--ansi|--log-level|--progress|--file|-f)
      GLOBAL_ARGS+=("$1")
      if [[ $# -lt 2 ]]; then
        echo "Option $1 requires a value" >&2
        exit 1
      fi
      shift
      GLOBAL_ARGS+=("$1")
      ;;
    --compatibility|--parallel|--verbose|-v)
      GLOBAL_ARGS+=("$1")
      ;;
    --)
      shift
      CMD_ARGS+=("$@")
      break
      ;;
    *)
      CMD_ARGS+=("$1")
      ;;
  esac
  shift
done

case "$COMMAND" in
  up|down|config|run|stop|logs|restart|ps)
    docker compose "${GLOBAL_ARGS[@]}" "${COMPOSE_ARGS[@]}" "$COMMAND" "${CMD_ARGS[@]}"
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage
    exit 1
    ;;
esac
