#!/usr/bin/env python3
"""
PolicyServer for SmolVLA async inference.

This wrapper starts a LeRobot PolicyServer that serves SmolVLA model inference.
Based on the LeRobot async inference tutorial.
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

def start_policy_server(config_path: Path, host: str = "localhost", port: int = 8080) -> Optional[threading.Thread]:
    """
    Start a PolicyServer in a background thread.
    
    Args:
        config_path: Path to config.json containing async_config
        host: Server host address
        port: Server port
        
    Returns:
        Thread running the server, or None if failed to start
    """
    try:
        # Import LeRobot server components
        from lerobot.scripts.server.configs import PolicyServerConfig
        from lerobot.scripts.server.policy_server import serve
        
        print(f"🚀 Starting PolicyServer on {host}:{port}...", flush=True)
        
        server_config = PolicyServerConfig(
            host=host,
            port=port,
        )
        
        def run_server():
            """Server thread function"""
            try:
                serve(server_config)
            except Exception as e:
                print(f"❌ PolicyServer error: {e}", flush=True)
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Give server time to start
        time.sleep(2.0)
        print("✅ PolicyServer started successfully!", flush=True)
        return server_thread
        
    except ImportError as e:
        print(f"❌ Failed to import LeRobot async components: {e}", flush=True)
        print("💡 Make sure lerobot[async] is installed", flush=True)
        return None
    except Exception as e:
        print(f"❌ Failed to start PolicyServer: {e}", flush=True)
        return None


def stop_policy_server(server_thread: Optional[threading.Thread]):
    """Stop the PolicyServer thread"""
    if server_thread and server_thread.is_alive():
        print("🛑 Stopping PolicyServer...", flush=True)
        # Note: gRPC server shutdown is handled by the serve() function internally
        # The daemon thread will clean up automatically


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start SmolVLA PolicyServer")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        exit(1)
    
    server_thread = start_policy_server(config_path, args.host, args.port)
    if server_thread:
        try:
            print("PolicyServer running. Press Ctrl+C to stop...")
            while server_thread.is_alive():
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            stop_policy_server(server_thread)
    else:
        print("❌ Failed to start PolicyServer")
        exit(1)