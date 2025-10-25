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

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\\/&]/\\&/g'
}

generate_embedded_kubeconfig() {
  local source=$1
  local target=$2
  local tmp

  if [[ ! -f "$source" ]]; then
    return 1
  fi

  tmp=$(mktemp "${TMPDIR:-/tmp}/kubeconfig.XXXXXX")
  if command -v kubectl >/dev/null 2>&1; then
    if kubectl --kubeconfig "$source" config view --raw --flatten >"$tmp" 2>/dev/null; then
      mv "$tmp" "$target"
      chmod 644 "$target"
      return 0
    fi
  fi
  rm -f "$tmp"

  if command -v python3 >/dev/null 2>&1; then
    if python3 - "$source" "$target" <<'PY'
import base64
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    sys.exit(1)

source = Path(sys.argv[1]).expanduser()
target = Path(sys.argv[2]).expanduser()

if not source.exists():
    sys.exit(1)

with source.open("r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}

pending = []
missing = False

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (source.parent / path).resolve()
    return path

def queue_embed(obj, key, dest):
    global missing
    value = obj.get(key)
    if not value:
        return
    resolved = resolve_path(value)
    if not resolved.exists():
        missing = True
        return
    encoded = base64.b64encode(resolved.read_bytes()).decode()
    pending.append((obj, key, dest, encoded))

for entry in data.get("clusters", []):
    cluster = entry.get("cluster") or {}
    queue_embed(cluster, "certificate-authority", "certificate-authority-data")

for entry in data.get("users", []):
    user = entry.get("user") or {}
    queue_embed(user, "client-certificate", "client-certificate-data")
    queue_embed(user, "client-key", "client-key-data")

if missing and pending:
    sys.exit(1)

for obj, key, dest, encoded in pending:
    obj.pop(key, None)
    obj[dest] = encoded

with target.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(data, fh, default_flow_style=False)
PY
    then
      chmod 644 "$target"
      return 0
    fi
  fi

  return 1
}

prepare_minikube_assets() {
  local source_kubeconfig=$1
  local generated_root="${COMPOSE_DIR}/config/local/_generated"
  local generated_config="${generated_root}/kubeconfig.yaml"
  local generated_minikube="${generated_root}/minikube"
  local container_minikube="/var/lib/kinitro/.minikube"
  local host_minikube="${HOST_MINIKUBE_DIR:-}"

  mkdir -p "$generated_root"
  rm -rf "$generated_minikube"
  mkdir -p "$generated_minikube"

  if [[ -z "$source_kubeconfig" ]]; then
    HOST_MINIKUBE_DIR="$generated_minikube"
    export HOST_MINIKUBE_DIR
    return
  fi

  if [[ "$source_kubeconfig" == "$generated_config" ]]; then
    HOST_MINIKUBE_DIR="$generated_minikube"
    export HOST_MINIKUBE_DIR
    HOST_KUBECONFIG="$generated_config"
    export HOST_KUBECONFIG
    return
  fi

  if [[ ! -f "$source_kubeconfig" ]]; then
    echo "kubeconfig not found: $source_kubeconfig" >&2
    exit 1
  fi

  if generate_embedded_kubeconfig "$source_kubeconfig" "$generated_config"; then
    HOST_MINIKUBE_DIR="$generated_minikube"
    export HOST_MINIKUBE_DIR
    HOST_KUBECONFIG="$generated_config"
    export HOST_KUBECONFIG
    return
  fi

  echo "Unable to inline kubeconfig certificates (missing kubectl/PyYAML?); falling back to bind-mounted Minikube assets." >&2

  cp "$source_kubeconfig" "$generated_config"
  chmod 644 "$generated_config"

  declare -a host_paths=()
  while IFS= read -r line; do
    local path
    path=$(echo "$line" | sed 's/^[^:]*:[[:space:]]*//')
    if [[ -n "$path" ]]; then
      host_paths+=("$path")
    fi
  done < <(grep -E '^[[:space:]]*(client-certificate|client-key|certificate-authority):' "$source_kubeconfig")

  if [[ -n "$host_minikube" && -d "$host_minikube" ]]; then
    for path in "${host_paths[@]}"; do
      [[ "$path" == "$container_minikube"* ]] && continue
      if [[ "$path" == "$host_minikube"* ]]; then
        local rel="${path#${host_minikube}/}"
        if [[ "$rel" == "$path" || -z "$rel" ]]; then
          continue
        fi
        local target="$generated_minikube/$rel"
        mkdir -p "$(dirname "$target")"
        if [[ -f "$path" ]]; then
          cp "$path" "$target"
          chmod 644 "$target"
        fi
        local escaped_src
        escaped_src=$(escape_sed "$path")
        local replacement="$container_minikube/$rel"
        local escaped_dst
        escaped_dst=$(escape_sed "$replacement")
        sed -i "s|$escaped_src|$escaped_dst|g" "$generated_config"
      fi
    done
  fi

  find "$generated_minikube" -type d -exec chmod 755 {} + >/dev/null 2>&1 || true

  HOST_MINIKUBE_DIR="$generated_minikube"
  export HOST_MINIKUBE_DIR
  HOST_KUBECONFIG="$generated_config"
  export HOST_KUBECONFIG
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

if [[ -n "${HOST_KUBECONFIG:-}" && -z "${HOST_KUBECONFIG_SOURCE:-}" ]]; then
  HOST_KUBECONFIG_SOURCE="${HOST_KUBECONFIG}"
  export HOST_KUBECONFIG_SOURCE
fi

if [[ -z "${HOST_MINIKUBE_DIR:-}" ]]; then
  if [[ -d "${HOME}/.minikube" ]]; then
    HOST_MINIKUBE_DIR="${HOME}/.minikube"
  elif [[ -d "${ROOT_DIR}/deploy/docker/config/local/minikube" ]]; then
    HOST_MINIKUBE_DIR="${ROOT_DIR}/deploy/docker/config/local/minikube"
  fi
  export HOST_MINIKUBE_DIR
fi

if [[ -n "${HOST_KUBECONFIG_SOURCE:-}" ]]; then
  prepare_minikube_assets "$HOST_KUBECONFIG_SOURCE"
elif [[ -n "${HOST_KUBECONFIG:-}" ]]; then
  prepare_minikube_assets "$HOST_KUBECONFIG"
elif [[ -n "${HOST_MINIKUBE_DIR:-}" ]]; then
  prepare_minikube_assets ""
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
