from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from common.errors import IntegrityError
from environment.navigation import Observation
from networks.models import ActorCriticNetwork, mask_action_values


def masked_categorical(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    return Categorical(logits=mask_action_values(logits, masks))


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    final_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if rewards.shape != dones.shape or rewards.shape != values.shape or rewards.ndim != 2:
        raise IntegrityError("A2C rollout arrays must share [steps,envs] shape")
    if final_values.shape != (rewards.shape[1],):
        raise IntegrityError("A2C final values must have shape [envs]")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    accumulator = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_values = final_values if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + float(gamma) * next_values * nonterminal - values[step]
        accumulator = delta + float(gamma) * float(gae_lambda) * nonterminal * accumulator
        advantages[step] = accumulator
    return advantages, advantages + values


class A2CAgent:
    def __init__(
        self,
        encoder_spec: Mapping[str, Any],
        config: Mapping[str, Any],
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.model = ActorCriticNetwork(encoder_spec).to(device)
        algorithm = config["algorithm"]
        self.initial_learning_rate = float(algorithm["learning_rate"])
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.initial_learning_rate,
            alpha=float(algorithm["rmsprop_alpha"]),
            eps=float(algorithm["rmsprop_eps"]),
        )
        self.update_count = 0

    def _batch_tensors(self, observations):
        spatial = torch.as_tensor(
            np.stack([item.spatial for item in observations]), dtype=torch.float32, device=self.device
        )
        history = torch.as_tensor(
            np.stack([item.history for item in observations]), dtype=torch.float32, device=self.device
        )
        return spatial, history

    def act_batch(self, observations, masks: np.ndarray):
        spatial, history = self._batch_tensors(observations)
        masks_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, values = self.model(spatial, history)
            distribution = masked_categorical(logits, masks_t)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def values(self, observations) -> np.ndarray:
        spatial, history = self._batch_tensors(observations)
        with torch.no_grad():
            _, values = self.model(spatial, history)
        return values.cpu().numpy()

    def deterministic_action(self, observation: Observation, legal_mask: np.ndarray) -> int:
        spatial, history = self._batch_tensors([observation])
        masks = torch.as_tensor(legal_mask[None], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(spatial, history)
            masked = mask_action_values(logits, masks)
        return int(torch.argmax(masked, dim=-1).item())

    def update(self, rollout: Mapping[str, np.ndarray], interaction: int, budget: int) -> Dict[str, float]:
        algorithm = self.config["algorithm"]
        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["dones"],
            rollout["values"],
            rollout["final_values"],
            float(algorithm["gamma"]),
            float(algorithm["gae_lambda"]),
        )
        spatial = torch.as_tensor(rollout["spatial"].reshape(-1, 5, 30, 30), dtype=torch.float32, device=self.device)
        history = torch.as_tensor(rollout["history"].reshape(-1, 10), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(rollout["masks"].reshape(-1, 4), dtype=torch.bool, device=self.device)
        actions = torch.as_tensor(rollout["actions"].reshape(-1), dtype=torch.long, device=self.device)
        advantages_t = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=self.device)
        rollout_interactions = int(rollout["rewards"].size)
        progress_remaining = max(0.0, 1.0 - (interaction - rollout_interactions) / float(budget))
        learning_rate = self.initial_learning_rate * progress_remaining
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        logits, predicted_values = self.model(spatial, history)
        distribution = masked_categorical(logits, masks)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        policy_loss = -(advantages_t.detach() * log_probs).mean()
        value_loss = F.mse_loss(predicted_values, returns_t)
        total_loss = (
            policy_loss
            + float(algorithm["value_loss_coefficient"]) * value_loss
            - float(algorithm["entropy_coefficient"]) * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), float(algorithm["max_gradient_norm"])
        )
        self.optimizer.step()
        self.update_count += 1
        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(learning_rate),
            "mean_advantage": float(advantages.mean()),
            "mean_return": float(returns.mean()),
        }

    def set_evaluation_mode(self, enabled: bool) -> bool:
        previous = self.model.training
        self.model.train(not enabled)
        return previous

    def restore_mode(self, training: bool) -> None:
        self.model.train(training)

    def checkpoint_state(self) -> Dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_checkpoint_state(self, payload: Mapping[str, Any], load_optimizer: bool = False) -> None:
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.update_count = int(payload.get("update_count", 0))
