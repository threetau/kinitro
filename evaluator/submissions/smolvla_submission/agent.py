"""
SmolVLA agent implementation for the Storb RL evaluator.
Uses the pretrained SmolVLA model from lerobot for robotics control.
"""

import json
import time as _time
from pathlib import Path

import numpy as np
import torch
import torchvision
from lerobot.configs.types import FeatureType  # type: ignore
from lerobot.policies.normalize import Normalize, Unnormalize  # type: ignore
from storb_eval import AgentInterface

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError as e:
    print(f"[Agent] Import error: {e}", flush=True)
    print(
        "[Agent] If running in Ray worker, ensure requirements.txt is properly installed",
        flush=True,
    )
    raise ImportError(
        "Could not import lerobot. Please ensure requirements.txt is correctly installed."
    ) from e


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

        # Load config and force synchronous mode
        config_path = submission_dir / "config.json"
        self.config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
        print("[Agent] Using synchronous inference mode (async disabled)", flush=True)

        # Synchronous mode initialization
        self._act_call_count = 0
        self._debug_every = int(kwargs.get("debug_every", 25))

        # Action queue for better chunking
        self._action_queue = []
        self._last_observation = None

        # Set up device
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

        # Config already loaded above

        print(
            f"[SmolVLA] Agent init: obs={observation_size}, act={action_size}, seed={seed}, device={self.device}",
            flush=True,
        )

        # Initialize SmolVLA model with timing
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
                if (
                    ft_type == FeatureType.VISUAL
                    or getattr(ft_type, "name", "") == "VISUAL"
                ):
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

        self.policy.normalize_inputs = Normalize(
            cfg.input_features, cfg.normalization_mapping, dataset_stats
        )
        self.policy.normalize_targets = Normalize(
            cfg.output_features, cfg.normalization_mapping, dataset_stats
        )
        self.policy.unnormalize_outputs = Unnormalize(
            cfg.output_features, cfg.normalization_mapping, dataset_stats
        )
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
                    print(
                        f"[SmolVLA] Using resize_imgs_with_padding={cfg.resize_imgs_with_padding}",
                        flush=True,
                    )
            if "chunk_size" in self.config:
                cfg.chunk_size = int(self.config["chunk_size"])  # type: ignore[attr-defined]
                print(f"[SmolVLA] Using chunk_size={cfg.chunk_size}", flush=True)
            if "n_action_steps" in self.config:
                cfg.n_action_steps = int(self.config["n_action_steps"])  # type: ignore[attr-defined]
                print(
                    f"[SmolVLA] Using n_action_steps={cfg.n_action_steps}", flush=True
                )
        except Exception as _ovr_exc:
            print(f"[SmolVLA] Speed tweaks skipped: {_ovr_exc}", flush=True)

        # 4) Finalize
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        print("[SmolVLA] Model loaded and moved to device", flush=True)

    # Image encoding aligned with Meta-World state layout
    # See: end-effector XYZ [0:3], gripper [3], object1 XYZ [4:7], object1 quat [7:11],
    #       object2 XYZ [11:14], object2 quat [14:18].
    def _observation_to_debug_images(self, observation: np.ndarray) -> torch.Tensor:
        """Render a simple RGB image from Meta-World style observations.

        - End-effector (EE) XY is drawn as a blue circle
        - Object #1 XY as a red circle (if present)
        - Object #2 XY as a green circle (if present)
        - Gripper openness modulates EE circle radius

        Returns a float tensor of shape [1, 3, H, W] on the configured device.
        """
        img_size = 256
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        def to_px(value: float, size: int = img_size) -> int:
            # Map assumed [-1, 1] to [0, size-1] and clamp
            px = int(((float(value) + 1.0) * 0.5) * (size - 1))
            return max(0, min(size - 1, px))

        def draw_circle(
            image: np.ndarray,
            cx: int,
            cy: int,
            radius: int,
            color: tuple[int, int, int],
        ) -> None:
            x0 = max(0, cx - radius)
            x1 = min(image.shape[1] - 1, cx + radius)
            y0 = max(0, cy - radius)
            y1 = min(image.shape[0] - 1, cy + radius)
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius:
                        image[y, x] = color

        # Parse observation per Meta-World doc
        ee_xy = None
        grip = None
        obj1_xy = None
        obj2_xy = None

        if observation is not None and observation.size >= 2:
            # EE XYZ at [0:3] → use XY for drawing
            ee_x = observation[0] if observation.size >= 1 else 0.0
            ee_y = observation[1] if observation.size >= 2 else 0.0
            ee_xy = (to_px(ee_x), to_px(ee_y))

        if observation is not None and observation.size >= 4:
            grip = float(observation[3])

        if observation is not None and observation.size >= 7:
            # Object #1 XYZ at [4:7]
            obj1_x = observation[4]
            obj1_y = observation[5]
            obj1_xy = (to_px(obj1_x), to_px(obj1_y))

        if observation is not None and observation.size >= 14:
            # Object #2 XYZ at [11:14]
            obj2_x = observation[11]
            obj2_y = observation[12]
            obj2_xy = (to_px(obj2_x), to_px(obj2_y))

        # Draw markers
        if ee_xy is not None:
            # Gripper openness in [-1,1] or [0,1]; map to radius [4, 10]
            grip_val = 0.5 if grip is None else float(grip)
            # Normalize to [0,1]
            grip_norm = max(0.0, min(1.0, (grip_val + 1.0) * 0.5))
            ee_radius = int(4 + (10 - 4) * grip_norm)
            draw_circle(img, ee_xy[0], ee_xy[1], ee_radius, (0, 0, 255))  # Blue

        if obj1_xy is not None:
            draw_circle(img, obj1_xy[0], obj1_xy[1], 6, (255, 0, 0))  # Red

        if obj2_xy is not None:
            draw_circle(img, obj2_xy[0], obj2_xy[1], 6, (0, 255, 0))  # Green

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    def act(self, observation: np.ndarray, goal_text: str) -> np.ndarray:
        """Generate action using SmolVLA."""
        return self._act_with_smolvla(observation, goal_text)

    def _act_with_smolvla(self, observation: np.ndarray, goal_text: str) -> np.ndarray:
        """Generate action using the SmolVLA model."""

        # Convert observation to images
        debug_images = self._observation_to_debug_images(observation)
        # log shape of images and observation
        print(f"[SmolVLA] Image shape: {debug_images.shape}", flush=True)
        print(f"[SmolVLA] Observation shape: {observation.shape}", flush=True)

        # store image at debug_images absolute path
        debug_images_dir = Path("debug_images")
        debug_images_dir.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(
            debug_images, debug_images_dir / f"image_{self._act_call_count}.png"
        )

        # log the absolute path of the saved image
        abs_path = (debug_images_dir / f"image_{self._act_call_count}.png").resolve()
        print(f"[SmolVLA] debug image saved at: {abs_path}")

        # Prepare batch for SmolVLA, following the HuggingFace example here: https://github.com/huggingface/lerobot/blob/0878c6880fa4fbadf0742751cf7b015f2d63a769/examples/2_evaluate_pretrained_policy.py#L82-L120
        with torch.no_grad():
            # Minimal robot state placeholder (float32, shape [1, 6])
            robot_state = torch.zeros(1, 6, dtype=torch.float32, device=self.device)

            # Determine image feature key from policy config, fallback to "observation.images"
            image_keys = []
            try:
                cfg = getattr(self.policy, "config", None)
                if cfg is not None and hasattr(cfg, "image_features"):
                    image_keys = list(getattr(cfg, "image_features"))
            except Exception as e:
                print(
                    f"[SmolVLA][WARN] Exception while getting image_features: {e}",
                    flush=True,
                )
                image_keys = []
            image_key = image_keys[0] if image_keys else "observation.images"

            # Compose batch as in the example
            batch = {
                image_key: debug_images.to(self.device),  # [B, C, H, W]
                "task": goal_text,
                "observation.state": robot_state,
            }

            # Call policy with batch dict per SmolVLA API
            result = self.policy.select_action(batch)

            # Extract action from possible return shapes/types (see example)
            action = result
            if isinstance(result, dict):
                for key in ("actions", "action", "pred_actions", "outputs", "act"):
                    if key in result:
                        action = result[key]
                        print(f"[SmolVLA] Selected action from key: {key}", flush=True)
                        break

            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()

            # If action is batched [T, D] or [B, D], pick first row
            if isinstance(action, np.ndarray) and action.ndim >= 2:
                action = action[0]

            # Map policy action to environment action space
            action_flat = self._map_policy_action_to_env(action)
            print(f"[SmolVLA] Final env action: {action_flat}", flush=True)
            self._act_call_count += 1
            return action_flat

    def _map_policy_action_to_env(self, raw_action: np.ndarray) -> np.ndarray:
        """Map policy output to the environment's expected action size.

        Heuristics:
        - If sizes match, clip to [-1, 1]
        - If policy outputs 6 dims and env expects 4 (Meta-World), use:
          [dx, dy, dz, gripper] where gripper = last dim
        - If fewer dims, pad zeros; if more dims, truncate after heuristic mapping
        - Always return float32, shape (self.action_size,)
        """
        try:
            arr = np.asarray(raw_action, dtype=np.float32).flatten()
            policy_dim = int(arr.shape[0])
            target_dim = int(self.action_size)

            if policy_dim == target_dim:
                mapped = arr
            elif policy_dim == 6 and target_dim == 4:
                # [dx, dy, dz] + use last value as gripper openness
                mapped = np.array([arr[0], arr[1], arr[2], arr[-1]], dtype=np.float32)
            elif policy_dim > target_dim:
                mapped = arr[:target_dim]
            else:
                mapped = np.pad(arr, (0, target_dim - policy_dim)).astype(np.float32)

            # Clip to valid range for Meta-World control
            mapped = np.clip(mapped, -1.0, 1.0).astype(np.float32)
            return mapped
        except Exception as e:
            print(
                f"[SmolVLA][WARN] Failed to map action ({e}); falling back to zeros",
                flush=True,
            )
            return np.zeros(self.action_size, dtype=np.float32)

    def reset(self) -> None:
        """Reset agent state for new episode."""
        # SmolVLA is typically stateless, but we can reset any internal buffers
        pass
