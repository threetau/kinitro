"""
Lightweight VLA-style agent.

This module provides a very small, NumPy-only agent intended for local
development and CI. It mimics a "Vision-Language-Action" (VLA) pipeline by
turning a goal text into a fixed-size embedding, combining it with the current
observation, and producing a continuous control action via a linear layer +
nonlinearity.

It is deliberately simple so it runs anywhere without heavyweight ML
dependencies. It also provides simple save/load helpers so that validators can
load miner-submitted agents from disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


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


@dataclass
class SimpleVLAPolicyConfig:
    observation_size: int
    action_size: int
    language_vocab_size: int = 64
    plan_hidden_size: int = 32
    control_hidden_size: int = 64
    nonlinearity: str = "tanh"
    seed: int = 0


class SimpleVLAPolicy:
    """
    A minimal VLA-style policy composed of two stages:

    1) "Planner": Encodes the goal text using a hashing tokenizer and projects
       it into a plan embedding of size ``plan_hidden_size``.
    2) "Controller": Given observation and plan embedding, outputs a bounded
       continuous action using an affine transform followed by tanh.
    """

    def __init__(self, config: SimpleVLAPolicyConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)

        # Planner weights: language_vocab_size -> plan_hidden_size
        self.W_plan = rng.normal(scale=0.1, size=(config.language_vocab_size, config.plan_hidden_size)).astype(
            np.float32
        )
        self.b_plan = np.zeros(config.plan_hidden_size, dtype=np.float32)

        # Controller weights: (observation_size + plan_hidden_size) -> action_size
        control_in = config.observation_size + config.plan_hidden_size
        self.W_control = rng.normal(scale=0.1, size=(control_in, config.control_hidden_size)).astype(np.float32)
        self.b_control = np.zeros(config.control_hidden_size, dtype=np.float32)

        self.W_out = rng.normal(scale=0.1, size=(config.control_hidden_size, config.action_size)).astype(np.float32)
        self.b_out = np.zeros(config.action_size, dtype=np.float32)

        if config.nonlinearity != "tanh":
            raise ValueError("Only 'tanh' nonlinearity is supported in the simple policy")

    # -----------------------
    # Serialization helpers
    # -----------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W_plan=self.W_plan,
            b_plan=self.b_plan,
            W_control=self.W_control,
            b_control=self.b_control,
            W_out=self.W_out,
            b_out=self.b_out,
            config=np.array([self.config.observation_size, self.config.action_size, self.config.language_vocab_size,
                             self.config.plan_hidden_size, self.config.control_hidden_size, self.config.seed],
                            dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path, observation_size: Optional[int] = None, action_size: Optional[int] = None) -> "SimpleVLAPolicy":
        data = np.load(path, allow_pickle=False)
        cfg_arr = data["config"].astype(np.int64)
        cfg = SimpleVLAPolicyConfig(
            observation_size=int(observation_size or cfg_arr[0]),
            action_size=int(action_size or cfg_arr[1]),
            language_vocab_size=int(cfg_arr[2]),
            plan_hidden_size=int(cfg_arr[3]),
            control_hidden_size=int(cfg_arr[4]),
            seed=int(cfg_arr[5]),
        )

        policy = cls(cfg)
        policy.W_plan = data["W_plan"]
        policy.b_plan = data["b_plan"]
        policy.W_control = data["W_control"]
        policy.b_control = data["b_control"]
        policy.W_out = data["W_out"]
        policy.b_out = data["b_out"]
        return policy

    # -----------------------
    # Inference
    # -----------------------
    def plan(self, goal_text: str) -> np.ndarray:
        goal_vec = _hashing_tokenizer(goal_text, self.config.language_vocab_size, self.config.seed)
        plan = goal_vec @ self.W_plan + self.b_plan
        return _tanh(plan)

    def act(self, observation: np.ndarray, goal_text: str) -> np.ndarray:
        if observation.ndim != 1:
            observation = observation.reshape(-1)
        plan_vec = self.plan(goal_text)
        controller_in = np.concatenate([observation.astype(np.float32), plan_vec], axis=0)
        hidden = _tanh(controller_in @ self.W_control + self.b_control)
        action = _tanh(hidden @ self.W_out + self.b_out)
        return action.astype(np.float32)


def create_default_agent(observation_size: int, action_size: int, seed: int = 0) -> SimpleVLAPolicy:
    config = SimpleVLAPolicyConfig(
        observation_size=observation_size,
        action_size=action_size,
        seed=seed,
    )
    return SimpleVLAPolicy(config)


def load_agent_from_path(
    agent_path: Optional[str | Path], observation_size: int, action_size: int, seed: int = 0
) -> SimpleVLAPolicy:
    """
    Loads an agent from ``agent_path`` if provided; otherwise creates a default
    randomly initialized agent. This is convenient for validator workflows,
    allowing miners to submit a small ``.npz`` file with weights.
    """
    if agent_path is None:
        return create_default_agent(observation_size, action_size, seed)
    return SimpleVLAPolicy.load(agent_path, observation_size=observation_size, action_size=action_size)


