"""
Miner Policy Implementation

This is where you implement your robotics policy.
The RobotPolicy class is imported by server.py and called during evaluation.

Your policy receives:
- Proprioceptive observations: end-effector position + gripper state
- Camera images (optional): RGB images from multiple viewpoints

Your policy returns:
- Action: joint commands as a numpy array

IMPORTANT: Object positions are NOT provided in observations!
You must learn to infer object locations from camera images.
"""

import os
import uuid
from typing import Optional

import numpy as np


class RobotPolicy:
    """
    Your robotics policy implementation.
    
    This class is instantiated once when the server starts.
    Implement the following methods:
    - reset(): Called at the start of each episode
    - act(): Called every timestep to get action
    - cleanup(): Called on shutdown
    """
    
    def __init__(self):
        """
        Initialize your policy.
        
        Load model weights, set up preprocessing, etc.
        This is called once when the server starts.
        """
        # Load your trained model here
        # Example:
        # self.model = torch.load("model.pt")
        # self.model.eval()
        
        self._model_loaded = False
        self._current_task = None
        self._episode_id = None
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        model_path = os.environ.get("MODEL_PATH", "model.pt")
        
        if os.path.exists(model_path):
            try:
                import torch
                self.model = torch.load(model_path, map_location="cpu")
                self.model.eval()
                self._model_loaded = True
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None
        else:
            print(f"Model not found at {model_path}, using random policy")
            self.model = None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    async def reset(self, task_config: dict) -> str:
        """
        Reset policy for a new episode.
        
        Called by the validator at the start of each evaluation episode.
        Use this to reset any internal state.
        
        Args:
            task_config: Dict with env_id, task_name, seed, etc.
            
        Returns:
            Episode ID string
        """
        self._current_task = task_config
        self._episode_id = uuid.uuid4().hex
        
        # Reset any episode-specific state here
        # Example: reset hidden state for RNN policies
        # self.hidden_state = None
        
        return self._episode_id
    
    async def act(
        self,
        observation: np.ndarray,
        images: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Get action for current observation.
        
        This is called every timestep during evaluation.
        You have ~500ms to respond.
        
        Args:
            observation: Proprioceptive state as numpy array
                        [ee_x, ee_y, ee_z, gripper_state]
            images: Optional dict of camera images
                   {"corner": (84, 84, 3), "gripper": (84, 84, 3)}
        
        Returns:
            Action as numpy array (typically 4D for MetaWorld)
        """
        # =====================================================
        # REPLACE THIS WITH YOUR POLICY INFERENCE
        # =====================================================
        
        if self.model is not None:
            # Example: PyTorch inference
            import torch
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                
                # If using images
                if images is not None and hasattr(self.model, 'forward_with_images'):
                    img_tensors = {
                        k: torch.FloatTensor(v).permute(2, 0, 1).unsqueeze(0) / 255.0
                        for k, v in images.items()
                    }
                    action = self.model.forward_with_images(obs_tensor, img_tensors)
                else:
                    action = self.model(obs_tensor)
                
                return action.squeeze(0).numpy()
        
        # Fallback: random action
        return np.random.uniform(-1, 1, size=4).astype(np.float32)
    
    async def cleanup(self):
        """
        Cleanup resources on shutdown.
        
        Called when the server is shutting down.
        """
        # Free GPU memory, close connections, etc.
        self.model = None


# =============================================================================
# Example: Vision-Language-Action Policy
# =============================================================================


class VLAPolicy(RobotPolicy):
    """
    Example Vision-Language-Action policy.
    
    This shows how to implement a policy that:
    1. Encodes task description with language model
    2. Encodes images with vision encoder
    3. Predicts actions with a policy head
    """
    
    def __init__(self):
        super().__init__()
        self.task_embedding = None
    
    async def reset(self, task_config: dict) -> str:
        episode_id = await super().reset(task_config)
        
        # Encode task description
        task_name = task_config.get("task_name", "manipulation")
        # self.task_embedding = self.encode_task(task_name)
        
        return episode_id
    
    async def act(
        self,
        observation: np.ndarray,
        images: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        # Encode images
        # image_features = self.vision_encoder(images)
        
        # Combine with task embedding and proprioception
        # features = concat(observation, image_features, self.task_embedding)
        
        # Predict action
        # action = self.policy_head(features)
        
        return np.random.uniform(-1, 1, size=4).astype(np.float32)


# =============================================================================
# Example: Diffusion Policy
# =============================================================================


class DiffusionPolicy(RobotPolicy):
    """
    Example Diffusion Policy implementation.
    
    Diffusion policies generate action sequences through
    iterative denoising, which can capture multi-modal
    action distributions.
    """
    
    def __init__(self):
        super().__init__()
        self.action_horizon = 8  # Predict 8 actions at once
        self.action_buffer = []
    
    async def reset(self, task_config: dict) -> str:
        episode_id = await super().reset(task_config)
        self.action_buffer = []
        return episode_id
    
    async def act(
        self,
        observation: np.ndarray,
        images: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        # If we have buffered actions, use them
        if self.action_buffer:
            return self.action_buffer.pop(0)
        
        # Otherwise, generate new action sequence
        # action_sequence = self.diffusion_model.sample(observation, images)
        # self.action_buffer = list(action_sequence[1:])  # Buffer future actions
        # return action_sequence[0]
        
        return np.random.uniform(-1, 1, size=4).astype(np.float32)
