"""
Default SimpleVLA agent implementation as a submission.

This provides the same lightweight VLA-style agent that was previously built-in,
but now follows the standard submission format.
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Tuple as TupleSpace
from gymnasium.spaces.utils import flatten_space
from storb_eval import AgentInterface


def _is_image_space(space: Box) -> bool:
    if not isinstance(space, Box) or space.dtype not in (
        np.uint8,
        np.float32,
        np.float64,
    ):
        return False
    shape = space.shape
    if len(shape) == 3:
        h, w, c = shape
        if c in (1, 3, 4):  # HWC
            return True
        # CHW
        if shape[0] in (1, 3, 4):
            return True
    return False


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256, hidden=(256, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    # Handles CHW or HWC automatically (permutes if needed)
    def __init__(self, in_shape, out_dim=256):
        super().__init__()
        c, h, w = (
            in_shape
            if in_shape[0] in (1, 3, 4)
            else (in_shape[2], in_shape[0], in_shape[1])
        )
        self.hwc = in_shape[0] not in (1, 3, 4)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            nflat = self.conv(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(nn.Linear(nflat, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B,H,W,C) or (B,C,H,W) or (H,W,C)/(C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if self.hwc:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        # Assume uint8 images → scale to [0,1]
        if x.dtype == torch.uint8:
            x = x / 255.0
        return self.head(self.conv(x).flatten(1))


class SpaceEncoder(nn.Module):
    """
    Recursively builds an encoder for any Gymnasium observation space.
    forward(obs) → (B, feat_dim) tensor
    """

    def __init__(self, space: gym.Space, out_dim=256):
        super().__init__()
        self.space = space
        self.encoder, self.feat_dim = self._build(space, out_dim)

    def _build(self, space, out_dim):
        if isinstance(space, Box):
            if _is_image_space(space):
                enc = SmallCNN(space.shape, out_dim)
                return enc, enc.out_dim
            # non-image → flatten + MLP
            flat = flatten_space(space)
            enc = MLP(int(np.prod(flat.shape)), out_dim)
            return enc, enc.out_dim

        if isinstance(space, Discrete):
            emb = nn.Embedding(space.n, out_dim)
            return nn.Sequential(emb, nn.Flatten()), out_dim

        if isinstance(space, MultiDiscrete):
            # one-hot per dimension, then MLP
            total = int(sum(space.nvec))
            proj = nn.Linear(total, out_dim)
            return nn.Sequential(_MultiDiscreteOneHot(space), proj, nn.ReLU()), out_dim

        if isinstance(space, MultiBinary):
            # treat as binary vector → MLP
            enc = MLP(int(np.prod(space.shape)), out_dim)
            return enc, enc.out_dim

        if isinstance(space, TupleSpace):
            subs = nn.ModuleList([SpaceEncoder(s, out_dim) for s in space.spaces])
            feat = sum(m.feat_dim for m in subs)
            return _Concat(subs), feat

        if isinstance(space, DictSpace):
            keys = list(space.spaces.keys())
            subs = nn.ModuleDict(
                {k: SpaceEncoder(space.spaces[k], out_dim) for k in keys}
            )
            feat = sum(subs[k].feat_dim for k in keys)
            return _ConcatDict(keys, subs), feat

        # Fallback: use Gymnasium flatten to a Box and MLP
        flat = flatten_space(space)
        enc = MLP(int(np.prod(flat.shape)), out_dim)
        return enc, enc.out_dim

    def forward(self, obs):
        return self.encoder(obs)


class _Concat(nn.Module):
    def __init__(self, subs):
        super().__init__()
        self.subs = subs

    @property
    def out_dim(self):
        return sum(m.feat_dim for m in self.subs)

    def forward(self, obs_tuple):
        feats = [m(o) for m, o in zip(self.subs, obs_tuple)]
        return torch.cat([f if f.dim() > 1 else f.unsqueeze(0) for f in feats], dim=-1)


class _ConcatDict(nn.Module):
    def __init__(self, keys, subs):
        super().__init__()
        self.keys = keys
        self.subs = subs

    @property
    def out_dim(self):
        return sum(self.subs[k].feat_dim for k in self.keys)

    def forward(self, obs_dict: dict[str, torch.Tensor]):
        feats = [self.subs[k](obs_dict[k]) for k in self.keys]
        return torch.cat([f if f.dim() > 1 else f.unsqueeze(0) for f in feats], dim=-1)


class _MultiDiscreteOneHot(nn.Module):
    def __init__(self, space: MultiDiscrete):
        super().__init__()
        self.nvec = torch.tensor(space.nvec, dtype=torch.long)

    def forward(self, x):
        # x: (B, D) or (D,)
        t = torch.as_tensor(x, dtype=torch.long)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        onehots = []
        for i, n in enumerate(self.nvec):
            oh = torch.zeros(t.shape[0], n, device=t.device)
            oh.scatter_(1, t[:, i : i + 1], 1.0)
            onehots.append(oh)
        return torch.cat(onehots, dim=1)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _hashing_tokenizer(text: str, vocab_size: int, seed: int) -> np.ndarray:
    """
    Turns input text into a bag-of-hashes one-hot vector of length ``vocab_size``.
    This is a tiny stand-in for a language encoder.
    """
    rng = np.random.default_rng(seed)
    # Split on whitespace; keep short for speed.
    tokens = text.lower().strip().split()
    indices = [abs(hash(t)) % vocab_size for t in tokens]
    vec = np.zeros(vocab_size, dtype=np.float32)
    if indices:
        vec[np.array(indices)] = 1.0
    else:
        # If the text is empty, put mass randomly on a single index to avoid
        # the zero vector which can lead to degenerate behaviors.
        vec[rng.integers(low=0, high=vocab_size)] = 1.0
    return vec


class Agent(AgentInterface):
    """
    A minimal VLA-style policy composed of two stages:

    1) "Planner": Encodes the goal text using a hashing tokenizer and projects
       it into a plan embedding of size ``plan_hidden_size``.
    2) "Controller": Given observation and plan embedding, outputs a bounded
       continuous action using an affine transform followed by tanh.
    """

    def __init__(
        self,
        submission_dir: Path,
        observation_space: gym.Space,
        action_space: gym.Space,
        seed: int = 0,
        **kwargs,
    ):
        """
        Initialize the SimpleVLA agent.

        Args:
            submission_dir: Path to submission directory (contains config.json)
            observation_size: Expected observation dimension
            action_size: Expected action dimension
            seed: Random seed for reproducibility
        """
        super().__init__(observation_space, action_space, seed, **kwargs)

        self.submission_dir = submission_dir
        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed

        # Load configuration
        config_path = submission_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        # Set default configuration
        self.language_vocab_size = config.get("language_vocab_size", 64)
        self.plan_hidden_size = config.get("plan_hidden_size", 32)
        self.control_hidden_size = config.get("control_hidden_size", 64)
        self.nonlinearity = config.get("nonlinearity", "tanh")

        if self.nonlinearity != "tanh":
            raise ValueError("Only 'tanh' nonlinearity is supported in SimpleVLA")

        # Initialize weights
        rng = np.random.default_rng(seed)

        # Initialize the space encoder for generic spaces (including Dict)
        self.encoder_out_dim = int(config.get("encoder_out_dim", 128))
        self.space_encoder = SpaceEncoder(
            self.observation_space, out_dim=self.encoder_out_dim
        )

        # Planner weights: language_vocab_size -> plan_hidden_size
        self.W_plan = rng.normal(
            scale=0.1, size=(self.language_vocab_size, self.plan_hidden_size)
        ).astype(np.float32)
        self.b_plan = np.zeros(self.plan_hidden_size, dtype=np.float32)

        # Controller weights: (encoder_feat_dim + plan_hidden_size) -> action_size
        control_in = int(self.space_encoder.feat_dim) + self.plan_hidden_size
        self.W_control = rng.normal(
            scale=0.1, size=(control_in, self.control_hidden_size)
        ).astype(np.float32)
        self.b_control = np.zeros(self.control_hidden_size, dtype=np.float32)

        self.W_out = rng.normal(
            scale=0.1, size=(self.control_hidden_size, self.action_space.shape[0])
        ).astype(np.float32)
        self.b_out = np.zeros(self.action_space.shape[0], dtype=np.float32)

        # Try to load weights if they exist
        weights_path = submission_dir / "model.npz"
        if weights_path.exists():
            self._load_weights(weights_path)

    def _load_weights(self, path: Path) -> None:
        """Load weights from .npz file if available."""
        data = np.load(path, allow_pickle=False)
        if "W_plan" in data:
            self.W_plan = data["W_plan"]
            self.b_plan = data["b_plan"]
            self.W_control = data["W_control"]
            self.b_control = data["b_control"]
            self.W_out = data["W_out"]
            self.b_out = data["b_out"]

    def save_weights(self, path: Path) -> None:
        """Save current weights to .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W_plan=self.W_plan,
            b_plan=self.b_plan,
            W_control=self.W_control,
            b_control=self.b_control,
            W_out=self.W_out,
            b_out=self.b_out,
        )

    def plan(self, goal_text: str) -> np.ndarray:
        """Encode goal text into plan embedding."""
        goal_vec = _hashing_tokenizer(goal_text, self.language_vocab_size, self.seed)
        plan = goal_vec @ self.W_plan + self.b_plan
        return _tanh(plan)

    def act(self, observation, **kwargs) -> torch.Tensor | np.ndarray:
        """Take action given observation and goal text."""
        goal_text = kwargs["goal_text"]
        # Convert observation (possibly dict/numpy) to torch structure expected by encoders
        obs_t = self._to_torch_obs(observation)
        observation_vec_t = self.space_encoder(obs_t)
        if observation_vec_t.dim() == 1:
            observation_vec_t = observation_vec_t.unsqueeze(0)
        observation_vec = observation_vec_t.squeeze(0).detach().cpu().numpy()
        print(f"[DefaultVLA] Observation vector shape: {observation_vec.shape}")

        plan_vec = self.plan(goal_text)
        controller_in = np.concatenate([observation_vec, plan_vec], axis=0)
        hidden = _tanh(controller_in @ self.W_control + self.b_control)
        action = _tanh(hidden @ self.W_out + self.b_out)
        return action.astype(np.float32)

    def _to_torch_obs(self, obs):
        """Recursively convert observation into torch tensors compatible with encoders."""
        if isinstance(obs, torch.Tensor):
            return obs
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs)
        if isinstance(obs, dict):
            return {k: self._to_torch_obs(v) for k, v in obs.items()}
        if isinstance(obs, (list, tuple)):
            return type(obs)(self._to_torch_obs(x) for x in obs)
        # Fallback: wrap scalar
        return torch.as_tensor(obs)

    def reset(self) -> None:
        """Reset agent state. SimpleVLA is stateless, so this is a no-op."""
        pass
