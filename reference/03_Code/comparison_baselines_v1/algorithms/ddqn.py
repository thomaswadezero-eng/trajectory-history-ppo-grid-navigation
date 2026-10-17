from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.errors import IntegrityError
from environment.navigation import Observation
from networks.models import QNetwork, mask_action_values, masked_argmax


def epsilon_by_interaction(
    interaction: int,
    budget: int,
    initial: float,
    final: float,
    exploration_fraction: float,
) -> float:
    decay_interactions = max(1, int(round(budget * exploration_fraction)))
    progress = min(max(interaction, 0) / float(decay_interactions), 1.0)
    return float(initial + progress * (final - initial))


def select_epsilon_greedy(
    q_values: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
    rng: np.random.RandomState,
) -> int:
    values = np.asarray(q_values, dtype=np.float64)
    mask = np.asarray(legal_mask, dtype=np.bool_)
    if values.shape != (4,) or mask.shape != (4,) or not mask.any():
        raise IntegrityError("epsilon-greedy requires four Q-values and at least one legal action")
    legal = np.flatnonzero(mask)
    if rng.random_sample() < epsilon:
        return int(legal[int(rng.randint(0, len(legal)))])
    masked = np.full(4, -np.inf, dtype=np.float64)
    masked[mask] = values[mask]
    return int(np.argmax(masked))


def double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_online_q: torch.Tensor,
    next_target_q: torch.Tensor,
    next_masks: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if next_online_q.shape != next_target_q.shape or next_online_q.shape[-1] != 4:
        raise IntegrityError("Double-DQN next-Q tensors must match and end in four actions")
    masks = next_masks.bool().clone()
    no_legal = ~masks.any(dim=-1)
    nonterminal_without_action = no_legal & (dones < 0.5)
    if torch.any(nonterminal_without_action):
        raise IntegrityError("nonterminal Double-DQN target has no legal next action")
    masks[no_legal] = True
    selected_actions = masked_argmax(next_online_q, masks)
    selected_target_q = next_target_q.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
    return rewards + float(gamma) * (1.0 - dones) * selected_target_q


class DDQNAgent:
    def __init__(
        self,
        encoder_spec: Mapping[str, Any],
        config: Mapping[str, Any],
        device: torch.device,
        network_class: Type[nn.Module] = QNetwork,
    ):
        self.config = config
        self.device = device
        self.online = network_class(encoder_spec).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()
        algorithm = config["algorithm"]
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(algorithm["learning_rate"]),
            eps=float(algorithm["adam_eps"]),
        )
        self.update_count = 0
        self.target_update_count = 0

    def _observation_tensors(self, observation: Observation):
        spatial = torch.as_tensor(observation.spatial[None], dtype=torch.float32, device=self.device)
        history = torch.as_tensor(observation.history[None], dtype=torch.float32, device=self.device)
        return spatial, history

    def q_values(self, observation: Observation) -> np.ndarray:
        spatial, history = self._observation_tensors(observation)
        with torch.no_grad():
            values = self.online(spatial, history)
        return values[0].cpu().numpy()

    def select_action(
        self,
        observation: Observation,
        legal_mask: np.ndarray,
        epsilon: float,
        rng: np.random.RandomState,
    ) -> int:
        return select_epsilon_greedy(self.q_values(observation), legal_mask, epsilon, rng)

    def deterministic_action(self, observation: Observation, legal_mask: np.ndarray) -> int:
        return self.select_action(observation, legal_mask, 0.0, np.random.RandomState(0))

    def update(self, batch: Mapping[str, np.ndarray]) -> Dict[str, float]:
        spatial = torch.as_tensor(batch["spatial"], dtype=torch.float32, device=self.device)
        history = torch.as_tensor(batch["history"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_spatial = torch.as_tensor(batch["next_spatial"], dtype=torch.float32, device=self.device)
        next_history = torch.as_tensor(batch["next_history"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)
        predicted = self.online(spatial, history).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.online(next_spatial, next_history)
            next_target = self.target(next_spatial, next_history)
            targets = double_dqn_targets(
                rewards,
                dones,
                next_online,
                next_target,
                next_masks,
                float(self.config["algorithm"]["gamma"]),
            )
        loss = F.smooth_l1_loss(predicted, targets, beta=float(self.config["algorithm"]["huber_delta"]))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), float(self.config["algorithm"]["max_gradient_norm"])
        )
        self.optimizer.step()
        self.update_count += 1
        return {
            "loss": float(loss.item()),
            "mean_q": float(predicted.detach().mean().item()),
            "mean_target": float(targets.detach().mean().item()),
            "gradient_norm": float(gradient_norm),
        }

    def hard_update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict(), strict=True)
        self.target_update_count += 1

    def set_evaluation_mode(self, enabled: bool) -> bool:
        previous = self.online.training
        self.online.train(not enabled)
        return previous

    def restore_mode(self, training: bool) -> None:
        self.online.train(training)

    def checkpoint_state(self) -> Dict[str, Any]:
        return {
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "target_update_count": self.target_update_count,
        }

    def load_checkpoint_state(self, payload: Mapping[str, Any], load_optimizer: bool = False) -> None:
        self.online.load_state_dict(payload["online_state_dict"], strict=True)
        self.target.load_state_dict(payload["target_state_dict"], strict=True)
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.update_count = int(payload.get("update_count", 0))
        self.target_update_count = int(payload.get("target_update_count", 0))

