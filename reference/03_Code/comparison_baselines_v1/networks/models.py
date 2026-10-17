from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn

from common.errors import IntegrityError


def validate_masks(masks: torch.Tensor) -> torch.Tensor:
    masks = masks.bool()
    if masks.ndim != 2 or masks.shape[-1] != 4:
        raise IntegrityError(f"action masks must have shape [batch,4], got {tuple(masks.shape)}")
    if torch.any(~masks.any(dim=-1)):
        raise IntegrityError("all actions are invalid")
    return masks


def mask_action_values(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masks = validate_masks(masks)
    if values.shape != masks.shape:
        raise IntegrityError("action values and masks have different shapes")
    return values.masked_fill(~masks, torch.finfo(values.dtype).min)


def masked_argmax(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return torch.argmax(mask_action_values(values, masks), dim=-1)


class SharedEncoder(nn.Module):
    """The frozen V7 observation encoder, without any planning head."""

    def __init__(self, spec: Mapping[str, Any]):
        super().__init__()
        convolutions = spec["convolutions"]
        layers = []
        for item in convolutions:
            layers.extend(
                [
                    nn.Conv2d(
                        int(item["in_channels"]),
                        int(item["out_channels"]),
                        int(item["kernel_size"]),
                        int(item["stride"]),
                        int(item["padding"]),
                    ),
                    nn.ReLU(),
                ]
            )
        spatial_projection = [int(x) for x in spec["spatial_projection"]]
        history = [int(x) for x in spec["history_mlp"]]
        fusion = [int(x) for x in spec["fusion"]]
        self.cnn = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(tuple(int(x) for x in spec["adaptive_pool"])),
            nn.Flatten(),
            nn.Linear(spatial_projection[0], spatial_projection[1]),
            nn.LayerNorm(spatial_projection[1]),
            nn.ReLU(),
        )
        self.history = nn.Sequential(
            nn.Linear(history[0], history[1]),
            nn.ReLU(),
            nn.Linear(history[1], history[2]),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion[0], fusion[1]),
            nn.LayerNorm(fusion[1]),
            nn.ReLU(),
        )
        self.output_size = fusion[-1]

    def forward(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or tuple(spatial.shape[1:]) != (5, 30, 30):
            raise IntegrityError(f"spatial observation shape mismatch: {tuple(spatial.shape)}")
        if history.ndim != 2 or history.shape[-1] != 10:
            raise IntegrityError(f"history observation shape mismatch: {tuple(history.shape)}")
        return self.fusion(torch.cat((self.cnn(spatial), self.history(history)), dim=-1))


class QNetwork(nn.Module):
    def __init__(self, encoder_spec: Mapping[str, Any]):
        super().__init__()
        self.encoder = SharedEncoder(encoder_spec)
        self.q_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encoder(spatial, history))


class DuelingQNetwork(nn.Module):
    """Mean-centered dueling head from Wang et al. (ICML 2016, Eq. 9)."""

    def __init__(self, encoder_spec: Mapping[str, Any]):
        super().__init__()
        self.encoder = SharedEncoder(encoder_spec)
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

    @staticmethod
    def aggregate(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[-1] != 1:
            raise IntegrityError("dueling value stream must have shape [batch,1]")
        if advantage.ndim != 2 or advantage.shape[-1] != 4:
            raise IntegrityError("dueling advantage stream must have shape [batch,4]")
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def forward(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        features = self.encoder(spatial, history)
        return self.aggregate(self.value(features), self.advantage(features))


class ActorCriticNetwork(nn.Module):
    def __init__(self, encoder_spec: Mapping[str, Any]):
        super().__init__()
        self.encoder = SharedEncoder(encoder_spec)
        self.actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))
        self.critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, spatial: torch.Tensor, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(spatial, history)
        return self.actor(features), self.critic(features).squeeze(-1)

