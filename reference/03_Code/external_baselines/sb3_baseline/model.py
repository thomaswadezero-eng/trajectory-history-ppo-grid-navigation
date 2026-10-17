from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class V7FeatureExtractor(BaseFeaturesExtractor):
    """Reproduce the active V7 spatial/history/fusion encoder."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        spatial_space = observation_space.spaces["spatial"]
        history_space = observation_space.spaces["history"]
        if tuple(spatial_space.shape) != (5, 30, 30):
            raise ValueError(f"unexpected spatial shape: {spatial_space.shape}")
        if tuple(history_space.shape) != (10,):
            raise ValueError(f"unexpected history shape: {history_space.shape}")
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.history = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        spatial = observations["spatial"].float()
        history = observations["history"].float()
        return self.fusion(torch.cat((self.cnn(spatial), self.history(history)), dim=-1))


def maskable_ppo_kwargs(seed: int, device: str) -> dict[str, Any]:
    return {
        "policy": "MultiInputPolicy",
        "learning_rate": 0.0002,
        "n_steps": 2048,
        "batch_size": 2048,
        "n_epochs": 3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "seed": int(seed),
        "device": device,
        "verbose": 1,
        "policy_kwargs": {
            "features_extractor_class": V7FeatureExtractor,
            "features_extractor_kwargs": {},
            "net_arch": {"pi": [128], "vf": [128]},
            "activation_fn": nn.ReLU,
            "ortho_init": False,
            "normalize_images": False,
            "share_features_extractor": True,
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {
                "weight_decay": 0.00001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            },
        },
    }


def entropy_coefficient(update_count: int) -> float:
    return max(0.002, 0.02 * (0.995 ** int(update_count)))
