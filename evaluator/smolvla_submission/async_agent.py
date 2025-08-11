"""
Asynchronous SmolVLA Agent using LeRobot's async inference.

This agent connects to a PolicyServer and maintains an action queue for smooth,
responsive robot control without inference delays.
"""

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from storb_eval.agent_interface import AgentInterface


class AsyncSmolVLAAgent(AgentInterface):
    """
    Asynchronous SmolVLA agent that uses LeRobot's PolicyServer.
    
    This agent:
    1. Connects to a PolicyServer running SmolVLA
    2. Maintains an action queue 
    3. Sends observations when queue is low
    4. Returns actions immediately from queue
    """
    
    def __init__(self, submission_dir: Path, observation_size: int, action_size: int, seed: int = 0, **kwargs):
        """Initialize AsyncSmolVLA agent."""
        self.submission_dir = submission_dir
        self.observation_size = observation_size
        self.action_size = action_size
        self.seed = seed
        
        # Load config
        config_path = submission_dir / "config.json"
        self.config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Get async config
        self.async_config = self.config.get("async_config", {})
        self.actions_per_chunk = self.async_config.get("actions_per_chunk", 50)
        self.chunk_size_threshold = self.async_config.get("chunk_size_threshold", 0.5)
        self.aggregate_fn_name = self.async_config.get("aggregate_fn_name", "weighted_average")
        self.server_host = self.async_config.get("server_host", "localhost")
        self.server_port = self.async_config.get("server_port", 8080)
        self.policy_device = self.async_config.get("policy_device", "cpu")
        self.debug_visualize = self.async_config.get("debug_visualize_queue_size", False)
        
        print(f"[AsyncSmolVLA] Initializing with actions_per_chunk={self.actions_per_chunk}, "
              f"chunk_size_threshold={self.chunk_size_threshold}", flush=True)
        
        # Action queue and client state
        self.action_queue = deque()
        self.robot_client = None
        self.client_thread = None
        self.server_thread = None
        self.is_running = False
        self.current_task = "push the block to the goal"
        
        # Queue size tracking for debugging
        self.queue_sizes = []
        self.step_count = 0
        
        # Image preprocessing
        import torchvision.transforms as transforms
        self.to_tensor = transforms.ToTensor()
        
        print(f"[AsyncSmolVLA] Agent initialized: obs={observation_size}, act={action_size}, "
              f"server={self.server_host}:{self.server_port}", flush=True)
    
    def _start_policy_server(self) -> bool:
        """Start the PolicyServer in a background thread."""
        try:
            from .policy_server import start_policy_server
            
            print("[AsyncSmolVLA] Starting PolicyServer...", flush=True)
            self.server_thread = start_policy_server(
                self.submission_dir / "config.json",
                self.server_host,
                self.server_port
            )
            return self.server_thread is not None
        except Exception as e:
            print(f"[AsyncSmolVLA] Failed to start PolicyServer: {e}", flush=True)
            return False
    
    def _create_robot_client(self) -> bool:
        """Create and configure the RobotClient."""
        try:
            from lerobot.scripts.server.configs import RobotClientConfig
            from lerobot.scripts.server.robot_client import RobotClient
            
            # Create a mock robot config for simulation
            # In a real setup, this would be your actual robot config
            robot_config = self._create_simulation_robot_config()
            
            client_config = RobotClientConfig(
                robot=robot_config,
                server_address=f"{self.server_host}:{self.server_port}",
                policy_device=self.policy_device,
                policy_type="smolvla",
                pretrained_name_or_path=self.config.get("model_path", "lerobot/smolvla_base"),
                chunk_size_threshold=self.chunk_size_threshold,
                actions_per_chunk=self.actions_per_chunk,
                task=self.current_task,
            )
            
            self.robot_client = RobotClient(client_config)
            return True
            
        except Exception as e:
            print(f"[AsyncSmolVLA] Failed to create RobotClient: {e}", flush=True)
            return False
    
    def _create_simulation_robot_config(self):
        """Create a simulation robot config for MetaWorld."""
        # This is a simplified robot config for simulation
        # In practice, you'd use actual robot hardware configs
        try:
            from lerobot.robots.so100_follower import SO100FollowerConfig
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
            
            # Mock camera config (not used in MetaWorld simulation)
            camera_config = {
                "observation.image2": OpenCVCameraConfig(
                    index_or_path=0, 
                    width=512, 
                    height=512, 
                    fps=30
                )
            }
            
            # Use simulation robot config
            robot_config = SO100FollowerConfig(
                port="/dev/null",  # Not used in simulation
                id="simulation_robot",
                cameras=camera_config
            )
            
            return robot_config
            
        except ImportError:
            # Fallback: create a minimal config if SO100 not available
            print("[AsyncSmolVLA] Using fallback robot config", flush=True)
            return None
    
    def _start_async_client(self) -> bool:
        """Start the async client in a background thread."""
        if not self.robot_client:
            return False
        
        try:
            def client_loop():
                """Client thread that manages the action queue."""
                if not self.robot_client.start():
                    print("[AsyncSmolVLA] Failed to start robot client", flush=True)
                    return
                
                # Start action receiver thread
                action_receiver_thread = threading.Thread(
                    target=self.robot_client.receive_actions, 
                    daemon=True
                )
                action_receiver_thread.start()
                
                self.is_running = True
                print("[AsyncSmolVLA] Async client running", flush=True)
                
                try:
                    # Run the control loop (this will be managed by our act() method)
                    while self.is_running:
                        time.sleep(0.1)  # Keep thread alive
                        
                except KeyboardInterrupt:
                    self.stop()
                finally:
                    self.robot_client.stop()
                    action_receiver_thread.join()
            
            self.client_thread = threading.Thread(target=client_loop, daemon=True)
            self.client_thread.start()
            
            # Give client time to start
            time.sleep(1.0)
            return True
            
        except Exception as e:
            print(f"[AsyncSmolVLA] Failed to start async client: {e}", flush=True)
            return False
    
    def start(self) -> bool:
        """Start the async inference system."""
        print("[AsyncSmolVLA] Starting async inference system...", flush=True)
        
        # 1. Start PolicyServer
        if not self._start_policy_server():
            print("[AsyncSmolVLA] Failed to start PolicyServer", flush=True)
            return False
        
        # 2. Create RobotClient
        if not self._create_robot_client():
            print("[AsyncSmolVLA] Failed to create RobotClient", flush=True)
            return False
        
        # 3. Start async client
        if not self._start_async_client():
            print("[AsyncSmolVLA] Failed to start async client", flush=True)
            return False
        
        print("✅ [AsyncSmolVLA] Async inference system started!", flush=True)
        return True
    
    def stop(self):
        """Stop the async inference system."""
        print("[AsyncSmolVLA] Stopping async inference system...", flush=True)
        
        self.is_running = False
        
        if self.robot_client:
            self.robot_client.stop()
        
        if self.server_thread:
            from .policy_server import stop_policy_server
            stop_policy_server(self.server_thread)
        
        if self.debug_visualize and self.queue_sizes:
            self._visualize_queue_sizes()
    
    def _observation_to_images(self, observation: np.ndarray) -> np.ndarray:
        """Convert MetaWorld observation to image tensor for SmolVLA."""
        # Extract camera data from observation if available
        # For MetaWorld, we typically don't have camera data, so create a mock image
        
        # Create a simple visualization of the state
        img_size = 512
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Add some basic visualization based on observation
        # This is a simplified approach - in practice you'd use actual camera data
        if len(observation) >= 6:  # Typical robot pose
            # Visualize robot position as colored squares
            x_pos = int((observation[0] + 1) * img_size // 4)  # Normalize to image coords
            y_pos = int((observation[1] + 1) * img_size // 4)
            
            x_pos = max(0, min(img_size - 50, x_pos))
            y_pos = max(0, min(img_size - 50, y_pos))
            
            # Draw robot as blue square
            img[y_pos:y_pos+50, x_pos:x_pos+50] = [0, 0, 255]  # Blue
            
            # If object position available, draw it
            if len(observation) >= 10:
                obj_x = int((observation[6] + 1) * img_size // 4)
                obj_y = int((observation[7] + 1) * img_size // 4)
                obj_x = max(0, min(img_size - 30, obj_x))
                obj_y = max(0, min(img_size - 30, obj_y))
                # Draw object as red square
                img[obj_y:obj_y+30, obj_x:obj_x+30] = [255, 0, 0]  # Red
        
        # Convert to tensor format expected by SmolVLA [C, H, W]
        img_tensor = self.to_tensor(img)  # This gives [C, H, W]
        
        return img_tensor.unsqueeze(0)  # Add batch dimension: [1, C, H, W]
    
    def act(self, observation: np.ndarray, goal_text: str) -> np.ndarray:
        """
        Get action from async inference system.
        
        This method:
        1. Checks if we need to send a new observation to the server
        2. Returns an action from the queue immediately
        3. Handles queue management and fallbacks
        """
        self.step_count += 1
        
        # Lazy initialization on first call
        if not self.is_running:
            if not self.start():
                print("[AsyncSmolVLA] Failed to start async system, using fallback", flush=True)
                return self._fallback_action()
        
        # Update task if changed
        if goal_text != self.current_task:
            self.current_task = goal_text
            print(f"[AsyncSmolVLA] Task updated: {goal_text}", flush=True)
        
        # Try to get action from async client
        try:
            # This is where we'd interface with the RobotClient's action queue
            # For now, simulate the async behavior with a simple queue
            action = self._get_action_from_queue(observation, goal_text)
            
            # Track queue size for debugging
            if self.debug_visualize:
                queue_size = len(self.action_queue)
                self.queue_sizes.append(queue_size)
                
                if self.step_count % 50 == 0:
                    avg_queue_size = np.mean(self.queue_sizes[-50:]) if self.queue_sizes else 0
                    print(f"[AsyncSmolVLA] Step {self.step_count}, queue size: {queue_size}, "
                          f"avg: {avg_queue_size:.1f}", flush=True)
            
            return action
            
        except Exception as e:
            print(f"[AsyncSmolVLA] Error in async inference: {e}, using fallback", flush=True)
            return self._fallback_action()
    
    def _get_action_from_queue(self, observation: np.ndarray, goal_text: str) -> np.ndarray:
        """Get action from queue, handling queue management."""
        
        # Check if we need to request new actions
        queue_ratio = len(self.action_queue) / max(1, self.actions_per_chunk)
        need_new_actions = queue_ratio <= self.chunk_size_threshold
        
        if need_new_actions:
            # In a real implementation, this would send observation to PolicyServer
            # For now, simulate by generating a chunk of actions
            self._request_new_actions(observation, goal_text)
        
        # Return action from queue, or fallback if empty
        if self.action_queue:
            action = self.action_queue.popleft()
            return action
        else:
            print("[AsyncSmolVLA] Queue empty, generating fallback action", flush=True)
            return self._fallback_action()
    
    def _request_new_actions(self, observation: np.ndarray, goal_text: str):
        """Request new action chunk from PolicyServer (simulated for now)."""
        
        # In real implementation, this would:
        # 1. Convert observation to proper format
        # 2. Send to PolicyServer via RobotClient
        # 3. Receive action chunk
        # 4. Add to queue
        
        # For now, simulate with simple actions
        print(f"[AsyncSmolVLA] Requesting new action chunk (queue low: {len(self.action_queue)})", flush=True)
        
        # Generate simulated actions for push task
        for i in range(self.actions_per_chunk):
            # Simple policy: move towards object, then push
            action = self._generate_simulated_action(observation, i)
            self.action_queue.append(action)
        
        print(f"[AsyncSmolVLA] Added {self.actions_per_chunk} actions to queue", flush=True)
    
    def _generate_simulated_action(self, observation: np.ndarray, step: int) -> np.ndarray:
        """Generate a simulated action for testing."""
        # This is a placeholder - in real implementation, actions come from SmolVLA
        
        # Simple push behavior: move towards object position
        action = np.zeros(self.action_size, dtype=np.float32)
        
        if len(observation) >= 10:  # Has object position
            # Move towards object
            robot_pos = observation[:3] if len(observation) >= 3 else np.zeros(3)
            object_pos = observation[6:9] if len(observation) >= 9 else np.zeros(3)
            
            # Simple proportional control
            direction = object_pos - robot_pos
            direction = np.clip(direction, -0.1, 0.1)  # Limit movement
            
            action[:len(direction)] = direction
        else:
            # Random small movements as fallback
            action = np.random.normal(0, 0.01, self.action_size).astype(np.float32)
        
        return action
    
    def _fallback_action(self) -> np.ndarray:
        """Generate a safe fallback action when async system fails."""
        # Return small random movements
        return np.random.normal(0, 0.01, self.action_size).astype(np.float32)
    
    def _visualize_queue_sizes(self):
        """Visualize action queue sizes for debugging."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.queue_sizes)
            plt.title('Action Queue Size Over Time')
            plt.xlabel('Step')
            plt.ylabel('Queue Size')
            plt.axhline(y=self.actions_per_chunk * self.chunk_size_threshold, 
                       color='r', linestyle='--', label=f'Threshold ({self.chunk_size_threshold})')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.submission_dir / 'queue_sizes.png')
            plt.show()
            print(f"[AsyncSmolVLA] Queue size plot saved to {self.submission_dir / 'queue_sizes.png'}")
            
        except ImportError:
            print("[AsyncSmolVLA] matplotlib not available, skipping visualization")
        except Exception as e:
            print(f"[AsyncSmolVLA] Error creating visualization: {e}")
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        print("[AsyncSmolVLA] Episode reset", flush=True)
        
        # Clear action queue for new episode
        self.action_queue.clear()
        
        # Reset step counter
        self.step_count = 0
        
        # In a real implementation, you might want to:
        # - Send episode reset signal to PolicyServer
        # - Clear any episode-specific state
        # - Reset action aggregation state