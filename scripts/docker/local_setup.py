#!/usr/bin/env python3
"""
Utilities for preparing a local validator/evaluator stack.

This script can:
  * Create admin API keys
  * Create validator API keys (optionally associated with a hotkey)
  * Bootstrap both keys and write them to a .env file
  * Seed a demo competition via the backend admin API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from backend.auth import UserRole
from backend.cli import ApiKeyManager

DEFAULT_DB_URL = "postgresql+asyncpg://validator:CHANGEME@localhost:5432/kinitrodb"
DEFAULT_BACKEND_URL = "http://localhost:8080"
DEFAULT_ENV_PATH = Path("deploy/docker/env/local.env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_api_key(
    database_url: str,
    *,
    name: str,
    role: UserRole,
    description: Optional[str] = None,
    hotkey: Optional[str] = None,
) -> tuple[str, str]:
    """Return (api_key_name, secret)."""
    manager = ApiKeyManager(database_url)
    try:
        api_key, secret = await manager.create_api_key(
            name=name,
            role=role.value,
            description=description,
            associated_hotkey=hotkey,
        )
        return api_key.name, secret
    finally:
        await manager.cleanup()


def _load_env(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _write_env(path: Path, updates: Dict[str, str]) -> None:
    existing = _load_env(path)
    existing.update(updates)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        for key, value in sorted(existing.items()):
            fh.write(f"{key}={value}\n")


def _load_competition_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Competition config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid competition config JSON: {exc}") from exc
    return data


def _create_competition(
    backend_url: str,
    admin_key: str,
    payload: Dict[str, Any],
) -> None:
    response = requests.post(
        f"{backend_url.rstrip('/')}/competitions",
        headers={"X-API-Key": admin_key, "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )

    if response.status_code == 409:
        print("Competition already exists; skipping.")
        return

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"Competition creation failed: {response.text}")
        raise exc

    print("Competition created successfully.")


def _resolve_admin_key(args) -> str:
    if args.api_key:
        return args.api_key

    env_values = _load_env(args.env_file)
    admin_key = env_values.get("ADMIN_API_KEY")
    if not admin_key:
        raise SystemExit(
            "No admin API key provided. Use --api-key or ensure ADMIN_API_KEY "
            f"is present in {args.env_file}"
        )
    return admin_key


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_create_admin(args) -> None:
    name = args.name
    description = args.description
    database_url = args.database_url

    api_key_name, secret = asyncio.run(
        _create_api_key(
            database_url,
            name=name,
            role=UserRole.ADMIN,
            description=description,
        )
    )

    print(f"Admin key created (name={api_key_name}): {secret}")

    if not args.no_env_update:
        _write_env(args.env_file, {"ADMIN_API_KEY": secret})
        print(f"ADMIN_API_KEY written to {args.env_file}")


def cmd_create_validator(args) -> None:
    database_url = args.database_url
    name = args.name
    hotkey = args.hotkey
    description = args.description

    api_key_name, secret = asyncio.run(
        _create_api_key(
            database_url,
            name=name,
            role=UserRole.VALIDATOR,
            description=description,
            hotkey=hotkey,
        )
    )

    print(f"Validator key created (name={api_key_name}): {secret}")

    if not args.no_env_update:
        _write_env(args.env_file, {"KINITRO_API_KEY": secret})
        print(f"KINITRO_API_KEY written to {args.env_file}")


def cmd_create_competition(args) -> None:
    admin_key = _resolve_admin_key(args)

    payload = _load_competition_config(args.config_file)
    _create_competition(
        args.backend_url,
        admin_key,
        payload=payload,
    )


def cmd_bootstrap(args) -> None:
    print("Creating admin API key...")
    _, admin_secret = asyncio.run(
        _create_api_key(
            args.database_url,
            name=args.admin_name,
            role=UserRole.ADMIN,
            description="Local bootstrap admin API key",
        )
    )

    print("Creating validator API key...")
    _, validator_secret = asyncio.run(
        _create_api_key(
            args.database_url,
            name=args.validator_name,
            role=UserRole.VALIDATOR,
            description="Local bootstrap validator API key",
            hotkey=args.validator_hotkey,
        )
    )

    updates = {
        "ADMIN_API_KEY": admin_secret,
        "KINITRO_API_KEY": validator_secret,
    }

    if not args.no_env_update:
        _write_env(args.env_file, updates)
        print(f"Wrote ADMIN_API_KEY and KINITRO_API_KEY to {args.env_file}")

    if args.create_competition:
        print("Creating demo competition...")
        payload = _load_competition_config(args.competition_file)
        _create_competition(
            args.backend_url,
            admin_secret,
            payload=payload,
        )

    print("Bootstrap complete.")


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local stack helper scripts")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DB_URL),
        help=f"Backend database URL (default: {DEFAULT_DB_URL})",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL),
        help=f"Backend HTTP URL (default: {DEFAULT_BACKEND_URL})",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f".env file to update (default: {DEFAULT_ENV_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    admin_parser = subparsers.add_parser(
        "create-admin-key", help="Create an admin API key"
    )
    admin_parser.add_argument("--name", default="local-admin", help="Key name")
    admin_parser.add_argument("--description", default="Local admin API key")
    admin_parser.add_argument(
        "--no-env-update",
        action="store_true",
        help="Skip writing ADMIN_API_KEY to the env file",
    )
    admin_parser.set_defaults(func=cmd_create_admin)

    validator_parser = subparsers.add_parser(
        "create-validator-key", help="Create a validator API key"
    )
    validator_parser.add_argument("--name", default="local-validator", help="Key name")
    validator_parser.add_argument("--description", default="Local validator API key")
    validator_parser.add_argument(
        "--hotkey",
        default="local-hotkey",
        help="Associated hotkey (optional, defaults to local-hotkey)",
    )
    validator_parser.add_argument(
        "--no-env-update",
        action="store_true",
        help="Skip writing KINITRO_API_KEY to the env file",
    )
    validator_parser.set_defaults(func=cmd_create_validator)

    competition_parser = subparsers.add_parser(
        "create-competition", help="Create a demo competition via the backend"
    )
    competition_parser.add_argument(
        "--api-key",
        help="Admin API key to use (default: read ADMIN_API_KEY from env file)",
    )
    competition_parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("deploy/docker/local/competition.json"),
        help="Path to competition JSON file",
    )
    competition_parser.set_defaults(func=cmd_create_competition)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap-demo", help="Create admin + validator keys and seed a competition"
    )
    bootstrap_parser.add_argument("--admin-name", default="local-admin")
    bootstrap_parser.add_argument("--validator-name", default="local-validator")
    bootstrap_parser.add_argument("--validator-hotkey", default="local-hotkey")
    bootstrap_parser.add_argument(
        "--no-env-update",
        action="store_true",
        help="Skip writing secrets to the env file",
    )
    bootstrap_parser.add_argument(
        "--no-competition",
        dest="create_competition",
        action="store_false",
        help="Skip the competition creation step",
    )
    bootstrap_parser.add_argument(
        "--competition-file",
        type=Path,
        default=Path("deploy/docker/local/competition.json"),
        help="Path to competition JSON file",
    )
    bootstrap_parser.set_defaults(func=cmd_bootstrap, create_competition=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
