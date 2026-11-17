"""Adapt foaming's parameterized actions to the RL pipeline's dictionary format."""

from collections.abc import Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HybridActionWrapper(gym.ActionWrapper):
    """Expose Dict(action_type=Discrete(K), parameters=Box(P)).

    Parameters concatenate one component per discrete action. Full vectors and
    compact vectors for the selected component are accepted. The metadata fields
    match those used by the existing RL pipeline's parameter-bound extractors.
    Observations and the Gymnasium reset/step contract pass through unchanged.
    """

    def __init__(self, env):
        super().__init__(env)
        discrete, components = env.action_space.spaces
        self._discrete_n = discrete.n
        self._param_components = list(components.spaces)
        self._component_sizes = [int(np.prod(space.shape)) for space in self._param_components]
        self._per_action_components = [[i] for i in range(self._discrete_n)]
        self._offsets = np.cumsum([0] + self._component_sizes)
        self._total_param_dim = int(self._offsets[-1])
        self.action_space = spaces.Dict({
            "action_type": discrete,
            "parameters": spaces.Box(
                low=np.concatenate([space.low for space in self._param_components]),
                high=np.concatenate([space.high for space in self._param_components]),
                dtype=np.float32,
            ),
        })

    def get_param_space(self, action_type_idx):
        if not self.action_space["action_type"].contains(action_type_idx):
            raise IndexError("Action index is outside the discrete action space")
        return self._param_components[int(action_type_idx)]

    def action(self, action):
        if not isinstance(action, Mapping) or not {"action_type", "parameters"} <= action.keys():
            raise ValueError("Expected a dictionary with action_type and parameters")
        index = action["action_type"]
        if not self.action_space["action_type"].contains(index):
            raise ValueError("action_type must be an integer in the discrete action space")
        index = int(index)
        params = np.asarray(action["parameters"], dtype=np.float32)
        size = self._component_sizes[index]
        if params.shape == (self._total_param_dim,):
            selected = params[self._offsets[index]:self._offsets[index + 1]]
        elif params.shape == (size,):
            selected = params
        else:
            raise ValueError("Expected a full flat parameter vector or the selected action's component")
        components = [np.zeros(space.shape, dtype=space.dtype) for space in self._param_components]
        components[index] = selected.copy()
        return index, tuple(components)
