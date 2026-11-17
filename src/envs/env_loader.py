"""Construction and action metadata for the foaming RL environments."""

import numpy as np

from .env_wrappers import HybridActionWrapper
from .hybrid_registry import make_hybrid_env


def build_env(env_name: str = "Foaming-v0", **kwargs):
    """Create the dictionary-action interface consumed by the RL pipeline."""
    return HybridActionWrapper(make_hybrid_env(env_name, **kwargs))


def infer_dims(env):
    """Return (observation dimension, discrete action count, flat parameter dimension)."""
    return (int(np.prod(env.observation_space.shape)),
            int(env.action_space["action_type"].n),
            int(env.action_space["parameters"].shape[0]))


def hybrid_parameter_bounds(env, action_bound=10.0):
    """Per-action normalized bounds and masks, refreshed after curriculum changes.

    action_bound is retained for pipeline call compatibility; foaming's active
    parameters always use [-1, 1]. Disabled or unused fields have zero bounds.
    """
    _, action_dim, param_dim = infer_dims(env)
    masks = env.unwrapped.parameter_mask.copy()
    if masks.shape != (action_dim, param_dim):
        raise ValueError("Environment parameter_mask has an incompatible shape")
    return (action_dim, param_dim, list(env.unwrapped.action_names),
            -masks, masks.copy(), masks)
