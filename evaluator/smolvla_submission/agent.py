"""
SmolVLA agent implementation for the Storb RL evaluator.
Uses the pretrained SmolVLA model from lerobot for robotics control.
"""

import json
import numpy as np
from pathlib import Path
from storb_eval import AgentInterface

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError as e:
    print(f"[Agent] Import error: {e}", flush=True)
    print("[Agent] If running in Ray worker, ensure requirements.txt is properly installed", flush=True)
    raise ImportError(f"Could not import lerobot. Please ensure requirements.txt is correctly installed.") from e


class Agent(AgentInterface):
    """
    SmolVLA agent for robotics control.
    
    Uses the pretrained SmolVLA-450M model from lerobot, which combines:
    1. SmolVLM2 as the vision-language backbone
    2. Flow matching transformer for action generation
    
    Optimized for M1 Pro with MPS acceleration.
    """

    def __init__(
        self,
        submission_dir: Path,
        observation_size: int,
        action_size: int,
        seed: int = 0,
        **kwargs,
    ):
        """Initialize SmolVLA agent."""
        self.submission_dir = submission_dir
        self.observation_size = observation_size
        self.action_size = action_size
        self.seed = seed
        
        # Load config to check for async mode
        config_path = submission_dir / "config.json"
        self.config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Check if async mode is enabled
        inference_mode = self.config.get("inference_mode", "synchronous")
        if inference_mode == "asynchronous":
            print("[Agent] Using async inference mode, delegating to AsyncSmolVLAAgent", flush=True)
            # Import and delegate to async agent
            from .async_agent import AsyncSmolVLAAgent
            self._async_agent = AsyncSmolVLAAgent(submission_dir, observation_size, action_size, seed, **kwargs)
            self._is_async = True
            return
        else:
            print("[Agent] Using synchronous inference mode", flush=True)
            self._is_async = False
        
        # Synchronous mode initialization
        self._img_debug_count = 0
        self._act_call_count = 0
        self._debug_every = int(kwargs.get("debug_every", 25))
        
        # Action queue for better chunking
        self._action_queue = []
        self._last_observation = None
        
        # Set up device
        import os
        import torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Optional CPU threading controls to reduce stalls
        try:
            torch.set_num_interop_threads(1)
            num_threads = int(self.config.get("torch_num_threads", 0))
            print(f"[SmolVLA] Setting torch num threads to {num_threads}", flush=True)
            if num_threads > 0:
                torch.set_num_threads(num_threads)
        except Exception:
            pass
        
        # Load config
        config_path = submission_dir / "config.json"
        self.config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        print(
            f"[SmolVLA] Agent init: obs={observation_size}, act={action_size}, seed={seed}, device={self.device}",
            flush=True,
        )

        # Initialize SmolVLA model with timing
        import time as _time
        _t0 = _time.perf_counter()
        self._init_smolvla_model()
        _t1 = _time.perf_counter()
        print(f"[SmolVLA] Model ready in {(_t1 - _t0):.2f}s", flush=True)

    def _init_smolvla_model(self):
        """Initialize SmolVLA model with safe overrides."""
        model_path = self.config.get("model_path", "lerobot/smolvla_base")
        print(f"[SmolVLA] Loading model: {model_path}", flush=True)

        # 1) Load base policy with default config to match checkpoint exactly
        self.policy = SmolVLAPolicy.from_pretrained(model_path)
        cfg = self.policy.config

        # 2) Build neutral stats and override normalization modules
        from lerobot.configs.types import FeatureType  # type: ignore
        from lerobot.policies.normalize import Normalize, Unnormalize  # type: ignore
        import torch

        def _find_norm_mode(norm_map, ft_type) -> object | None:
            if ft_type in norm_map:
                return norm_map[ft_type]
            name = getattr(ft_type, "name", None)
            if name and name in norm_map:
                return norm_map[name]
            s = str(ft_type)
            if s in norm_map:
                return norm_map[s]
            return None

        def _neutral_stats(features: dict) -> dict:
            stats: dict[str, dict[str, torch.Tensor]] = {}
            norm_map = getattr(cfg, "normalization_mapping", {})
            for key, ft in features.items():
                ft_type = getattr(ft, "type", None)
                raw_shape = tuple(getattr(ft, "shape", ()))
                if ft_type == FeatureType.VISUAL or getattr(ft_type, "name", "") == "VISUAL":
                    shape = (int(raw_shape[0]), 1, 1)
                else:
                    shape = tuple(int(x) for x in raw_shape)
                mode = _find_norm_mode(norm_map, ft_type)
                if mode is None:
                    continue
                if str(mode).endswith("MEAN_STD"):
                    stats[key] = {
                        "mean": torch.zeros(shape, dtype=torch.float32),
                        "std": torch.ones(shape, dtype=torch.float32),
                    }
                elif str(mode).endswith("MIN_MAX"):
                    stats[key] = {
                        "min": torch.zeros(shape, dtype=torch.float32),
                        "max": torch.ones(shape, dtype=torch.float32),
                    }
            return stats

        input_stats = _neutral_stats(getattr(cfg, "input_features", {}))
        output_stats = _neutral_stats(getattr(cfg, "output_features", {}))
        dataset_stats = {**input_stats, **output_stats}

        self.policy.normalize_inputs = Normalize(cfg.input_features, cfg.normalization_mapping, dataset_stats)
        self.policy.normalize_targets = Normalize(cfg.output_features, cfg.normalization_mapping, dataset_stats)
        self.policy.unnormalize_outputs = Unnormalize(cfg.output_features, cfg.normalization_mapping, dataset_stats)
        print("[SmolVLA] Applied neutral normalization stats", flush=True)

        # 3) Apply runtime-only speed tweaks (do not change architecture like num_vlm_layers)
        try:
            if "num_steps" in self.config:
                cfg.num_steps = int(self.config["num_steps"])  # type: ignore[attr-defined]
                print(f"[SmolVLA] Using num_steps={cfg.num_steps}", flush=True)
            if "resize" in self.config:
                r = self.config["resize"]
                if isinstance(r, (list, tuple)) and len(r) == 2:
                    cfg.resize_imgs_with_padding = (int(r[0]), int(r[1]))  # type: ignore[attr-defined]
                    print(f"[SmolVLA] Using resize_imgs_with_padding={cfg.resize_imgs_with_padding}", flush=True)
            if "chunk_size" in self.config:
                cfg.chunk_size = int(self.config["chunk_size"])  # type: ignore[attr-defined]
                print(f"[SmolVLA] Using chunk_size={cfg.chunk_size}", flush=True)
            if "n_action_steps" in self.config:
                cfg.n_action_steps = int(self.config["n_action_steps"])  # type: ignore[attr-defined]
                print(f"[SmolVLA] Using n_action_steps={cfg.n_action_steps}", flush=True)
        except Exception as _ovr_exc:
            print(f"[SmolVLA] Speed tweaks skipped: {_ovr_exc}", flush=True)

        # 4) Finalize
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        print("[SmolVLA] Model loaded and moved to device", flush=True)

    def _observation_to_images(self, observation: np.ndarray):
        """
        Convert MetaWorld observation to image format expected by SmolVLA.
        
        SmolVLA expects RGB images from multiple cameras. Since MetaWorld provides
        1D observations, we create a simple visual representation.
        """
        import torch
        from PIL import Image
        import time as _time

        _t0 = _time.perf_counter()
        # Target size 512x512 to match SmolVLA default and avoid runtime resizing cost
        target_size = 512
        h = w = target_size
        img_array = np.zeros((h, w, 3), dtype=np.uint8)

        # Print observation structure for debugging first few times
        if self._img_debug_count <= 3:
            print(f"[SmolVLA] Observation shape: {observation.shape}, sample values: {observation[:10]}", flush=True)
        
        # Create a more realistic top-down view using position information
        obs = observation.flatten()
        
        if len(obs) >= 12:  # We have robot, object, and goal positions
            # MetaWorld push-v3 observations typically contain:
            # [0:4] - robot gripper position (x, y, z) + gripper state
            # [4:7] - robot gripper velocity 
            # [7:10] - object position (x, y, z)
            # [10:13] - goal position (x, y, z)
            # Let's also check other possible layouts
            
            if self._img_debug_count <= 3:
                print(f"[SmolVLA] Full observation: {obs[:min(20, len(obs))]}", flush=True)
            
            # Try different observation layouts for MetaWorld push task
            if len(obs) >= 39:  # Full MetaWorld observation
                robot_pos = obs[0:3]
                object_pos = obs[4:7]  # Object often starts at index 4
                goal_pos = obs[36:39]  # Goal is typically at the end
            else:
                # Simplified layout
                robot_pos = obs[0:3]
                object_pos = obs[6:9] if len(obs) > 8 else robot_pos
                goal_pos = obs[9:12] if len(obs) > 11 else object_pos
            
            # Convert world coordinates to image coordinates
            def world_to_img(x, y):
                # Map [-0.5, 0.5] world space to [50, 462] image space (with margins)
                img_x = int((x + 0.5) * (target_size - 100) + 50)
                img_y = int((0.5 - y) * (target_size - 100) + 50)  # Flip Y for image coordinates
                return max(10, min(target_size-10, img_x)), max(10, min(target_size-10, img_y))
            
            robot_img = world_to_img(robot_pos[0], robot_pos[1])
            object_img = world_to_img(object_pos[0], object_pos[1])
            goal_img = world_to_img(goal_pos[0], goal_pos[1])
            
            # Create a more realistic table-top view
            # Start with wooden table background
            img_array[:] = [205, 175, 149]  # Wood brown color
            
            # Add some texture/grain to make it look more realistic
            y, x = np.ogrid[:target_size, :target_size]
            texture = ((x + y) % 20 < 2).astype(np.uint8) * 10
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] - texture, 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] - texture, 0, 255)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] - texture, 0, 255)
            
            # Draw goal area as a target zone (more realistic)
            goal_radius = 40
            goal_mask = (x - goal_img[0])**2 + (y - goal_img[1])**2 <= goal_radius**2
            goal_outer_mask = (x - goal_img[0])**2 + (y - goal_img[1])**2 <= (goal_radius-5)**2
            # Green target zone with white center
            img_array[goal_mask] = [100, 200, 100]
            img_array[goal_outer_mask] = [255, 255, 255]
            
            # Draw object as a 3D-looking cube (red/orange gradient)
            obj_size = 25
            obj_x1, obj_y1 = max(0, object_img[0]-obj_size//2), max(0, object_img[1]-obj_size//2)
            obj_x2, obj_y2 = min(target_size, object_img[0]+obj_size//2), min(target_size, object_img[1]+obj_size//2)
            
            # Main cube face (red)
            if obj_x2 > obj_x1 and obj_y2 > obj_y1:
                img_array[obj_y1:obj_y2, obj_x1:obj_x2] = [220, 80, 80]
                # Add 3D shading effect
                if obj_y2 - obj_y1 > 5 and obj_x2 - obj_x1 > 5:
                    # Top edge (lighter)
                    img_array[obj_y1:obj_y1+3, obj_x1:obj_x2] = [255, 120, 120]
                    # Left edge (lighter)
                    img_array[obj_y1:obj_y2, obj_x1:obj_x1+3] = [255, 120, 120]
                    # Bottom edge (darker)
                    img_array[obj_y2-3:obj_y2, obj_x1:obj_x2] = [180, 60, 60]
                    # Right edge (darker)
                    img_array[obj_y1:obj_y2, obj_x2-3:obj_x2] = [180, 60, 60]
            
            # Draw robot gripper as a more realistic end-effector
            robot_radius = 18
            robot_mask = (x - robot_img[0])**2 + (y - robot_img[1])**2 <= robot_radius**2
            robot_inner_mask = (x - robot_img[0])**2 + (y - robot_img[1])**2 <= (robot_radius-5)**2
            # Gray metallic gripper
            img_array[robot_mask] = [120, 120, 140]
            img_array[robot_inner_mask] = [160, 160, 180]
            
            # Add gripper "fingers" as small rectangles
            finger_size = 8
            # Left finger
            img_array[robot_img[1]-2:robot_img[1]+2, robot_img[0]-robot_radius:robot_img[0]-robot_radius+finger_size] = [100, 100, 120]
            # Right finger  
            img_array[robot_img[1]-2:robot_img[1]+2, robot_img[0]+robot_radius-finger_size:robot_img[0]+robot_radius] = [100, 100, 120]
            
            if self._img_debug_count <= 3:
                print(f"[SmolVLA] Robot: {robot_pos[:2]} -> {robot_img}, "
                      f"Object: {object_pos[:2]} -> {object_img}, "
                      f"Goal: {goal_pos[:2]} -> {goal_img}", flush=True)
        
        else:
            # Fallback: use original abstract visualization
            obs = observation.astype(np.float32).reshape(-1)
            if obs.size == 0:
                obs = np.zeros(1, dtype=np.float32)
            min_v = float(obs.min())
            max_v = float(obs.max())
            denom = (max_v - min_v) if (max_v - min_v) > 1e-8 else 1.0
            obs_norm = (obs - min_v) / denom

            g = int(np.sqrt(obs_norm.size))
            g = max(1, g)
            vals = obs_norm[: g * g]
            if vals.size < g * g:
                vals = np.pad(vals, (0, g * g - vals.size), mode="constant")
            grid = vals.reshape(g, g)

            block = max(1, target_size // g)
            big = np.kron(grid, np.ones((block, block), dtype=np.float32))
            big = (big * 255.0).clip(0, 255).astype(np.uint8)

            big_h, big_w = big.shape
            out = np.zeros((target_size, target_size), dtype=np.uint8)
            out[: min(target_size, big_h), : min(target_size, big_w)] = big[:target_size, :target_size]

            img_array[..., 0] = out
            img_array[..., 1] = (out // 2).astype(np.uint8)
            img_array[..., 2] = (out // 3).astype(np.uint8)

        pil_img = Image.fromarray(img_array)

        # Save debug images for the first few steps
        if self._img_debug_count < 10:  # Save first 10 images
            try:
                # Use absolute path accessible from host
                debug_dir = Path("/Users/rishiadhikari/devs/storb-tech/storb-rl/evaluator/debug_images")
                debug_dir.mkdir(exist_ok=True)
                debug_path = debug_dir / f"step_{self._img_debug_count:03d}.png"
                pil_img.save(debug_path)
                if self._img_debug_count <= 3:
                    print(f"[SmolVLA] Saved debug image: {debug_path}", flush=True)
            except Exception as e:
                if self._img_debug_count <= 3:
                    print(f"[SmolVLA] Failed to save debug image: {e}", flush=True)

        # Reuse a single ToTensor transform to reduce overhead
        try:
            import torchvision.transforms as transforms
            if not hasattr(self, "_to_tensor"):
                self._to_tensor = transforms.ToTensor()
            img_tensor = self._to_tensor(pil_img).unsqueeze(0)
        except Exception:
            # Minimal fallback
            arr = np.asarray(pil_img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
            img_tensor = torch.from_numpy(arr)

        img_tensor = img_tensor.to(self.device)
        self._img_debug_count += 1
        if self._img_debug_count <= 3:
            _t1 = _time.perf_counter()
            print(
                f"[SmolVLA] Built image tensor {tuple(img_tensor.shape)} on {self.device} in {(_t1 - _t0):.3f}s",
                flush=True,
            )
        return img_tensor

    def act(self, observation: np.ndarray, goal_text: str) -> np.ndarray:
        """Generate action using SmolVLA."""
        # Delegate to async agent if configured
        if hasattr(self, '_is_async') and self._is_async:
            return self._async_agent.act(observation, goal_text)
        
        return self._act_with_smolvla(observation, goal_text)

    def _act_with_smolvla(self, observation: np.ndarray, goal_text: str) -> np.ndarray:
        """Generate action using the SmolVLA model."""
        import torch
        
        # Convert observation to images
        images = self._observation_to_images(observation)

        # Prepare batch for SmolVLA
        with torch.no_grad():
            # Minimal robot state placeholder
            robot_state = torch.zeros(1, 6, device=self.device)

            # Determine an acceptable image feature key from the loaded config
            image_keys = []
            try:
                cfg = getattr(self.policy, "config", None)
                if cfg is not None and hasattr(cfg, "image_features"):
                    image_keys = list(getattr(cfg, "image_features"))
            except Exception:
                image_keys = []

            image_key = image_keys[0] if image_keys else "observation.images"
            # Only print on the first few calls to reduce spam
            if self._act_call_count < 3:
                print(
                    f"[SmolVLA] Using image key '{image_key}'. Building batch...",
                    flush=True,
                )

            batch = {
                image_key: images,            # [B, C, H, W]
                "task": goal_text,           # text instruction
                "observation.state": robot_state,  # OBS_STATE key
            }

            # Check if this is a cache hit (action chunking) or new computation
            action_queue_len = len(getattr(self.policy, '_queues', {}).get('action', []))
            will_compute = action_queue_len == 0
            
            if will_compute and self._act_call_count > 0:
                print(f"[SmolVLA] Computing new action chunk (queue empty)...", flush=True)
            elif self._act_call_count <= 3:
                print(f"[SmolVLA] Using cached action (queue has {action_queue_len} actions)", flush=True)
            
            # Call policy with batch dict per SmolVLA API
            import time as _time
            _t0 = _time.perf_counter()
            result = self.policy.select_action(batch)
            _t1 = _time.perf_counter()
            
            self._act_call_count += 1
            if will_compute or self._act_call_count <= 3:
                print(
                    f"[SmolVLA] select_action call #{self._act_call_count} took {(_t1 - _t0):.3f}s {'(computed new chunk)' if will_compute else '(cache hit)'}",
                    flush=True,
                )

            # Extract action from possible return shapes/types
            action = result
            if isinstance(result, dict):
                for key in ("actions", "action", "pred_actions", "outputs", "act"):
                    if key in result:
                        action = result[key]
                        break

            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            # Only print on the first few calls
            if self._act_call_count <= 3:
                print(
                    f"[SmolVLA] Action type={type(action)} shape={getattr(action, 'shape', None)}",
                    flush=True,
                )
            
            # Ensure action has correct size
            if action.shape[-1] != self.action_size:
                # Pad or truncate to match expected action size
                if action.shape[-1] < self.action_size:
                    action = np.pad(action, (0, self.action_size - action.shape[-1]))
                else:
                    action = action[:self.action_size]
                # Only print on the first few calls
                if self._act_call_count <= 3:
                    print(
                        f"[SmolVLA] Adjusted action to size {self.action_size}",
                        flush=True,
                    )
            
            # Process and debug actions
            action_flat = action.flatten()[:self.action_size]
            
            # Debug action values
            if self._act_call_count <= 3:
                print(f"[SmolVLA] Raw action values: {action_flat}", flush=True)
                print(f"[SmolVLA] Action range: [{action_flat.min():.4f}, {action_flat.max():.4f}]", flush=True)
            
            # Scale up actions if they seem too small (more conservative scaling)
            max_abs_action = np.abs(action_flat).max()
            if max_abs_action < 0.01:
                scale_factor = 5.0  # More conservative scaling
                action_flat = action_flat * scale_factor
                if self._act_call_count <= 3:
                    print(f"[SmolVLA] Scaled up small actions by {scale_factor}x", flush=True)
            elif max_abs_action > 1.0:
                # Clip very large actions
                action_flat = np.clip(action_flat, -1.0, 1.0)
                if self._act_call_count <= 3:
                    print(f"[SmolVLA] Clipped large actions to [-1, 1]", flush=True)
            
            if self._act_call_count <= 3:
                print(f"[SmolVLA] Final action: {action_flat}", flush=True)
            
            return action_flat

    def reset(self) -> None:
        """Reset agent state for new episode."""
        # Delegate to async agent if configured
        if hasattr(self, '_is_async') and self._is_async:
            return self._async_agent.reset()
        
        # SmolVLA is typically stateless, but we can reset any internal buffers
        pass