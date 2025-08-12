"""
Default SimpleVLA agent implementation as a submission.

This provides the same lightweight VLA-style agent that was previously built-in,
but now follows the standard submission format.
"""

import json
import numpy as np
from pathlib import Path
from storb_eval import AgentInterface


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
        observation_size: int,
        action_size: int,
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
        self.submission_dir = submission_dir
        self.observation_size = observation_size
        self.action_size = action_size
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

        # Planner weights: language_vocab_size -> plan_hidden_size
        self.W_plan = rng.normal(
            scale=0.1, size=(self.language_vocab_size, self.plan_hidden_size)
        ).astype(np.float32)
        self.b_plan = np.zeros(self.plan_hidden_size, dtype=np.float32)

        # Controller weights: (observation_size + plan_hidden_size) -> action_size
        control_in = observation_size + self.plan_hidden_size
        self.W_control = rng.normal(
            scale=0.1, size=(control_in, self.control_hidden_size)
        ).astype(np.float32)
        self.b_control = np.zeros(self.control_hidden_size, dtype=np.float32)

        self.W_out = rng.normal(
            scale=0.1, size=(self.control_hidden_size, action_size)
        ).astype(np.float32)
        self.b_out = np.zeros(action_size, dtype=np.float32)

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

    def act(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Take action given observation and goal text."""
        goal_text = kwargs["goal_text"]

        if observation.ndim != 1:
            observation = observation.reshape(-1)
        plan_vec = self.plan(goal_text)
        controller_in = np.concatenate(
            [observation.astype(np.float32), plan_vec], axis=0
        )
        hidden = _tanh(controller_in @ self.W_control + self.b_control)
        action = _tanh(hidden @ self.W_out + self.b_out)
        return action.astype(np.float32)

    def reset(self) -> None:
        """Reset agent state. SimpleVLA is stateless, so this is a no-op."""
        pass
