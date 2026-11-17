"""Environment names supported by this repository."""

from .foaming import FOAMING_ENVS, FoamingEnv


def make_hybrid_env(env_name: str, **kwargs):
    """Construct a raw foaming environment with a parameterized tuple action space."""
    if env_name == "Foaming-v0":
        return FoamingEnv(**kwargs)
    if env_name not in FOAMING_ENVS:
        raise ValueError(f"Unknown environment {env_name!r}; choose Foaming-v0 or {list(FOAMING_ENVS)}")
    variant = FOAMING_ENVS[env_name]
    if "variant" in kwargs and kwargs["variant"] != variant:
        raise ValueError(f"{env_name} requires variant={variant!r}")
    kwargs["variant"] = variant
    return FoamingEnv(**kwargs)
