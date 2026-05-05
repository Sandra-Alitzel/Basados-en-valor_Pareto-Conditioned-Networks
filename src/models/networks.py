from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal init for ``Linear`` layers (zero bias)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def mlp(
    sizes: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        is_last = i == len(sizes) - 2
        linear = nn.Linear(sizes[i], sizes[i + 1])
        gain = 0.01 if is_last else 2.0**0.5
        _orthogonal_init(linear, gain=gain)
        layers.append(linear)
        if not is_last:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class ConditionedActor(nn.Module):
    

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        use_horizon: bool = True,
        state_dependent_std: bool = True,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.use_horizon = use_horizon
        self.state_dependent_std = state_dependent_std

        # input = state + return-target (+ horizon scalar)
        input_dim = obs_dim + reward_dim + (1 if use_horizon else 0)
        self._input_dim = input_dim

        # Shared trunk that mixes the state with the desired return command.
        trunk_sizes = [input_dim, *hidden_sizes]
        trunk_layers: list[nn.Module] = []
        for i in range(len(trunk_sizes) - 1):
            linear = nn.Linear(trunk_sizes[i], trunk_sizes[i + 1])
            _orthogonal_init(linear, gain=2.0**0.5)
            trunk_layers.append(linear)
            trunk_layers.append(activation())
        self.trunk = nn.Sequential(*trunk_layers)

        # Mean head: small init gain so initial actions are ~0.
        self.mu_head = nn.Linear(hidden_sizes[-1], action_dim)
        _orthogonal_init(self.mu_head, gain=0.01)

        # Std head: either state-dependent or a learned global parameter.
        if state_dependent_std:
            self.log_std_head: nn.Module = nn.Linear(
                hidden_sizes[-1], action_dim
            )
            _orthogonal_init(self.log_std_head, gain=0.01)
        else:
            # Initial log_std = 0  ->  std = 1
            self.log_std_param = nn.Parameter(torch.zeros(action_dim))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_input(
        self,
        state: torch.Tensor,
        target_return: torch.Tensor,
        target_horizon: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenate ``[state, target_return, (horizon)]`` along dim -1."""
        if state.shape[-1] != self.obs_dim:
            raise ValueError(
                f"state has last-dim {state.shape[-1]}, expected {self.obs_dim}"
            )
        if target_return.shape[-1] != self.reward_dim:
            raise ValueError(
                f"target_return has last-dim {target_return.shape[-1]}, "
                f"expected {self.reward_dim}"
            )

        parts = [state, target_return]
        if self.use_horizon:
            if target_horizon is None:
                raise ValueError(
                    "use_horizon=True but target_horizon was not provided"
                )
            if target_horizon.dim() == state.dim() - 1:
                target_horizon = target_horizon.unsqueeze(-1)
            parts.append(target_horizon.to(state.dtype))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def forward(
        self,
        state: torch.Tensor,
        target_return: torch.Tensor,
        target_horizon: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        x = self._build_input(state, target_return, target_horizon)
        h = self.trunk(x)
        mu = self.mu_head(h)

        if self.state_dependent_std:
            log_std = self.log_std_head(h)
        else:
            log_std = self.log_std_param.expand_as(mu)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mu, std

    def distribution(
        self,
        state: torch.Tensor,
        target_return: torch.Tensor,
        target_horizon: torch.Tensor | None = None,
    ) -> Independent:
        
        mu, std = self.forward(state, target_return, target_horizon)
        return Independent(Normal(mu, std), 1)
