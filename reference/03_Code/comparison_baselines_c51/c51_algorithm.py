from __future__ import annotations

import copy
import hashlib
import io
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from barn_protocol import IntegrityError, Observation


def module_sha256(module: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest().upper()


def validate_masks(masks: torch.Tensor, *, allow_empty: bool = False) -> torch.Tensor:
    masks = masks.bool()
    if masks.ndim != 2 or masks.shape[-1] != 4:
        raise IntegrityError(f"action masks must have shape [batch,4], got {tuple(masks.shape)}")
    if not allow_empty and torch.any(~masks.any(dim=-1)):
        raise IntegrityError("a nonterminal state has no legal action")
    return masks


def masked_argmax(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masks = validate_masks(masks)
    if values.shape != masks.shape:
        raise IntegrityError("expected-Q tensor and action mask shape mismatch")
    masked = values.masked_fill(~masks, torch.finfo(values.dtype).min)
    return torch.argmax(masked, dim=-1)


def epsilon_by_interaction(
    interaction: int,
    budget: int,
    initial: float,
    final: float,
    exploration_fraction: float,
) -> float:
    decay_interactions = max(1, int(round(float(budget) * float(exploration_fraction))))
    progress = min(max(int(interaction), 0) / float(decay_interactions), 1.0)
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
        raise IntegrityError("epsilon-greedy requires four expected Q-values and a nonempty legal set")
    if not np.isfinite(values).all() or not 0.0 <= float(epsilon) <= 1.0:
        raise IntegrityError("non-finite expected Q or invalid epsilon")
    legal = np.flatnonzero(mask)
    if rng.random_sample() < float(epsilon):
        return int(legal[int(rng.randint(0, len(legal)))])
    masked = np.full(4, -np.inf, dtype=np.float64)
    masked[mask] = values[mask]
    return int(np.argmax(masked))


class SharedEncoder(nn.Module):
    """Frozen V7/DDQN encoder, with no planning or auxiliary head."""

    def __init__(self, spec: Mapping[str, Any]):
        super().__init__()
        layers: list[nn.Module] = []
        for item in spec["convolutions"]:
            layers.extend(
                [
                    nn.Conv2d(
                        int(item["in_channels"]), int(item["out_channels"]),
                        int(item["kernel_size"]), int(item["stride"]), int(item["padding"]),
                    ),
                    nn.ReLU(),
                ]
            )
        spatial_projection = [int(value) for value in spec["spatial_projection"]]
        history = [int(value) for value in spec["history_mlp"]]
        fusion = [int(value) for value in spec["fusion"]]
        self.cnn = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(tuple(int(value) for value in spec["adaptive_pool"])),
            nn.Flatten(),
            nn.Linear(spatial_projection[0], spatial_projection[1]),
            nn.LayerNorm(spatial_projection[1]),
            nn.ReLU(),
        )
        self.history = nn.Sequential(
            nn.Linear(history[0], history[1]), nn.ReLU(),
            nn.Linear(history[1], history[2]), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion[0], fusion[1]), nn.LayerNorm(fusion[1]), nn.ReLU()
        )
        self.output_size = fusion[-1]

    def forward(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or tuple(spatial.shape[1:]) != (5, 30, 30):
            raise IntegrityError(f"spatial observation shape mismatch: {tuple(spatial.shape)}")
        if history.ndim != 2 or history.shape[-1] != 10:
            raise IntegrityError(f"history observation shape mismatch: {tuple(history.shape)}")
        return self.fusion(torch.cat((self.cnn(spatial), self.history(history)), dim=-1))


class C51Network(nn.Module):
    """Canonical non-dueling categorical Q network: [batch, 4, 51] logits."""

    def __init__(self, encoder_spec: Mapping[str, Any], n_atoms: int, v_min: float, v_max: float):
        super().__init__()
        if int(n_atoms) != 51:
            raise IntegrityError("classic C51 requires exactly 51 atoms")
        if not float(v_min) < float(v_max):
            raise IntegrityError("return support must be strictly increasing")
        self.encoder = SharedEncoder(encoder_spec)
        self.n_atoms = int(n_atoms)
        self.action_count = 4
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.action_count * self.n_atoms)
        )
        self.register_buffer("support", torch.linspace(float(v_min), float(v_max), self.n_atoms))

    def logits(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(spatial, history)).view(-1, self.action_count, self.n_atoms)

    def distribution(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(self.logits(spatial, history), dim=-1)
        if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise IntegrityError("C51 produced an invalid probability mass function")
        return probabilities

    def log_distribution(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        values = F.log_softmax(self.logits(spatial, history), dim=-1)
        if not torch.isfinite(values).all():
            raise IntegrityError("C51 produced non-finite log probabilities")
        return values

    def expected_q_from_pmf(self, pmf: torch.Tensor) -> torch.Tensor:
        if pmf.ndim != 3 or tuple(pmf.shape[1:]) != (4, self.n_atoms):
            raise IntegrityError("C51 PMF must have shape [batch,4,51]")
        return (pmf * self.support.view(1, 1, -1)).sum(dim=-1)

    def expected_q(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.expected_q_from_pmf(self.distribution(spatial, history))


def categorical_projection(
    next_pmf: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bellemare et al. categorical projection, including exact-atom mass retention."""
    if next_pmf.ndim != 2 or support.ndim != 1 or next_pmf.shape[1] != support.numel():
        raise IntegrityError("projection shape mismatch")
    if rewards.shape != dones.shape or rewards.ndim != 1 or rewards.shape[0] != next_pmf.shape[0]:
        raise IntegrityError("projection reward/done shape mismatch")
    if not torch.isfinite(next_pmf).all() or torch.any(next_pmf < 0):
        raise IntegrityError("projection input PMF is invalid")
    input_error = torch.max(torch.abs(next_pmf.sum(dim=-1) - 1.0))
    if float(input_error.item()) > 1e-5:
        raise IntegrityError(f"projection input PMF mass error: {float(input_error.item())}")
    delta_z = support[1] - support[0]
    raw_tz = rewards[:, None] + float(gamma) * (1.0 - dones[:, None]) * support[None, :]
    lower_fraction = float((raw_tz < support[0]).float().mean().item())
    upper_fraction = float((raw_tz > support[-1]).float().mean().item())
    tz = raw_tz.clamp(float(support[0].item()), float(support[-1].item()))
    b = (tz - support[0]) / delta_z
    # Pin clamped endpoints exactly. Floating linspace spacing can otherwise
    # yield b=49.99999 at V_max and leak tiny mass to the penultimate atom.
    b = torch.where(tz <= support[0], torch.zeros_like(b), b)
    b = torch.where(tz >= support[-1], torch.full_like(b, support.numel() - 1), b)
    lower = torch.floor(b).long().clamp(0, support.numel() - 1)
    upper = torch.ceil(b).long().clamp(0, support.numel() - 1)
    exact = lower == upper
    lower_weight = torch.where(exact, torch.ones_like(b), upper.to(b.dtype) - b)
    upper_weight = torch.where(exact, torch.zeros_like(b), b - lower.to(b.dtype))
    projected = torch.zeros_like(next_pmf)
    projected.scatter_add_(1, lower, next_pmf * lower_weight)
    projected.scatter_add_(1, upper, next_pmf * upper_weight)
    mass_error = torch.max(torch.abs(projected.sum(dim=-1) - 1.0))
    if not torch.isfinite(projected).all() or torch.any(projected < -1e-7):
        raise IntegrityError("projected target PMF is invalid")
    if float(mass_error.item()) > 1e-5:
        raise IntegrityError(f"projected target PMF mass error: {float(mass_error.item())}")
    diagnostics = {
        "projected_mass_sum_error_max": float(mass_error.item()),
        "projected_mass_min": float(projected.min().item()),
        "projected_mass_max": float(projected.max().item()),
        "target_support_clamp_lower_fraction": lower_fraction,
        "target_support_clamp_upper_fraction": upper_fraction,
        "exact_atom_fraction": float(exact.float().mean().item()),
    }
    return projected, diagnostics


class C51Agent:
    def __init__(self, config: Mapping[str, Any], device: torch.device, seed: int):
        self.config = config
        self.device = device
        algorithm = config["c51"]
        support = algorithm["support"]
        self.online = C51Network(
            config["network"]["shared_encoder"], int(support["n_atoms"]),
            float(support["v_min"]), float(support["v_max"]),
        ).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=float(algorithm["learning_rate"]),
            eps=float(algorithm["adam_eps"]),
        )
        offsets = config["random_streams"]
        self.exploration_rng = np.random.RandomState((int(seed) + int(offsets["exploration_seed_offset"])) % (2**32))
        self.replay_rng = np.random.RandomState((int(seed) + int(offsets["replay_seed_offset"])) % (2**32))
        self.optimizer_step_count = 0
        self.target_update_count = 0
        self.illegal_executed_action_count = 0
        self.illegal_greedy_action_count = 0
        self.illegal_target_argmax_count = 0

    def _observation_tensors(self, observation: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = torch.as_tensor(observation.spatial[None], dtype=torch.float32, device=self.device)
        history = torch.as_tensor(observation.history[None], dtype=torch.float32, device=self.device)
        return spatial, history

    def q_values(self, observation: Observation) -> np.ndarray:
        spatial, history = self._observation_tensors(observation)
        with torch.no_grad():
            q_values = self.online.expected_q(spatial, history)
        return q_values[0].cpu().numpy()

    def act(self, observation: Observation, legal_mask: np.ndarray, epsilon: float) -> int:
        action = select_epsilon_greedy(
            self.q_values(observation), legal_mask, float(epsilon), self.exploration_rng
        )
        if not bool(np.asarray(legal_mask, dtype=np.bool_)[action]):
            self.illegal_executed_action_count += 1
            raise IntegrityError("C51 attempted to execute an illegal action")
        return action

    def deterministic_action(self, observation: Observation, legal_mask: np.ndarray) -> int:
        action = select_epsilon_greedy(
            self.q_values(observation), legal_mask, 0.0, np.random.RandomState(0)
        )
        if not bool(np.asarray(legal_mask, dtype=np.bool_)[action]):
            self.illegal_greedy_action_count += 1
            raise IntegrityError("C51 deterministic policy selected an illegal action")
        return action

    def update(self, batch: Mapping[str, np.ndarray]) -> dict[str, Any]:
        spatial = torch.as_tensor(batch["spatial"], dtype=torch.float32, device=self.device)
        history = torch.as_tensor(batch["history"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_spatial = torch.as_tensor(batch["next_spatial"], dtype=torch.float32, device=self.device)
        next_history = torch.as_tensor(batch["next_history"], dtype=torch.float32, device=self.device)
        current_masks = validate_masks(torch.as_tensor(batch["current_masks"], dtype=torch.bool, device=self.device))
        next_masks = validate_masks(
            torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device), allow_empty=True
        )
        if torch.any(~current_masks.gather(1, actions[:, None]).squeeze(1)):
            raise IntegrityError("replay batch contains an action illegal under its saved current mask")
        nonterminal_no_legal = (dones < 0.5) & (~next_masks.any(dim=-1))
        if torch.any(nonterminal_no_legal):
            raise IntegrityError("nonterminal target state has no legal action")
        masks_for_argmax = next_masks.clone()
        masks_for_argmax[~masks_for_argmax.any(dim=-1)] = True
        with torch.no_grad():
            next_all_pmf = self.target.distribution(next_spatial, next_history)
            next_q = self.target.expected_q_from_pmf(next_all_pmf)
            target_actions = masked_argmax(next_q, masks_for_argmax)
            legal_target = masks_for_argmax.gather(1, target_actions[:, None]).squeeze(1)
            if torch.any(~legal_target):
                self.illegal_target_argmax_count += int((~legal_target).sum().item())
                raise IntegrityError("target network selected an illegal action")
            next_pmf = next_all_pmf[torch.arange(actions.shape[0], device=self.device), target_actions]
            target_pmf, projection = categorical_projection(
                next_pmf, rewards, dones, self.target.support, float(self.config["c51"]["gamma"])
            )
        logits = self.online.logits(spatial, history)
        chosen_logits = logits[torch.arange(actions.shape[0], device=self.device), actions]
        chosen_log_pmf = F.log_softmax(chosen_logits, dim=-1)
        loss = -(target_pmf.detach() * chosen_log_pmf).sum(dim=-1).mean()
        if not torch.isfinite(loss):
            raise IntegrityError("C51 cross-entropy loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), float(self.config["c51"]["max_gradient_norm"])
        )
        self.optimizer.step()
        self.optimizer_step_count += 1
        with torch.no_grad():
            all_pmf = torch.softmax(logits.detach(), dim=-1)
            all_q = self.online.expected_q_from_pmf(all_pmf)
            entropy = -(all_pmf * torch.log(all_pmf.clamp_min(1e-12))).sum(dim=-1)
            pmf_error = torch.max(torch.abs(all_pmf.sum(dim=-1) - 1.0))
        return {
            "loss": float(loss.item()),
            "q_mean": float(all_q.mean().item()),
            "q_std": float(all_q.std(unbiased=False).item()),
            "q_min": float(all_q.min().item()),
            "q_max": float(all_q.max().item()),
            "pmf_sum_error_max": float(pmf_error.item()),
            "pmf_min": float(all_pmf.min().item()),
            "pmf_max": float(all_pmf.max().item()),
            "distribution_entropy_mean": float(entropy.mean().item()),
            "lower_support_mass_mean": float(all_pmf[..., 0].mean().item()),
            "upper_support_mass_mean": float(all_pmf[..., -1].mean().item()),
            "gradient_norm_before_clip": float(gradient_norm),
            "target_action_uses_target_network_only": True,
            **projection,
        }

    def hard_update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict(), strict=True)
        self.target_update_count += 1

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_step_count": self.optimizer_step_count,
            "target_update_count": self.target_update_count,
            "exploration_rng_state": self.exploration_rng.get_state(),
            "replay_rng_state": self.replay_rng.get_state(),
            "illegal_executed_action_count": self.illegal_executed_action_count,
            "illegal_greedy_action_count": self.illegal_greedy_action_count,
            "illegal_target_argmax_count": self.illegal_target_argmax_count,
        }

    def load_checkpoint_state(self, payload: Mapping[str, Any], *, load_optimizer: bool) -> None:
        self.online.load_state_dict(payload["online_state_dict"], strict=True)
        self.target.load_state_dict(payload["target_state_dict"], strict=True)
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.optimizer_step_count = int(payload["optimizer_step_count"])
        self.target_update_count = int(payload["target_update_count"])
        self.exploration_rng.set_state(payload["exploration_rng_state"])
        self.replay_rng.set_state(payload["replay_rng_state"])
        self.illegal_executed_action_count = int(payload["illegal_executed_action_count"])
        self.illegal_greedy_action_count = int(payload["illegal_greedy_action_count"])
        self.illegal_target_argmax_count = int(payload["illegal_target_argmax_count"])
