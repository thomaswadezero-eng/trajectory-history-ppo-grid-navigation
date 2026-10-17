from __future__ import annotations

from typing import Any, Mapping

import torch

from algorithms.ddqn import DDQNAgent
from networks.models import DuelingQNetwork


class D3QNAgent(DDQNAgent):
    """Double DQN with the standard mean-centered dueling architecture."""

    def __init__(self, encoder_spec: Mapping[str, Any], config: Mapping[str, Any], device: torch.device):
        super().__init__(encoder_spec, config, device, network_class=DuelingQNetwork)

