# SPDX-License-Identifier: GPL-3.0-or-later
"""Sequential foaming PAMDP with three curriculum stages and fixed-size spaces."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .foaming_model import (
    GLASS_BOOLS, GLASS_NUMBERS, GLOBAL_NUMBERS, GLOBAL_OPERATIONS, OPERATIONS,
    QUALITY, RANGES, TOOL_BOOLS, TOOL_NUMBERS, VARIANTS, FoamingSimulator,
    GroundedAction, InvalidAction, _specs, generate_instance,
)


FOAMING_ENVS = {
    "FoamingSmall-v0": "small",
    "FoamingMain-v0": "main",
    "FoamingHighDep-v0": "high_dep",
}


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _variant(stage, variant):
    if stage is not None:
        stage = _positive_int(stage, "stage")
        if stage > len(VARIANTS):
            raise ValueError("stage must be 1, 2, or 3")
        if variant is not None and variant != VARIANTS[stage - 1]:
            raise ValueError("stage and variant disagree")
        return VARIANTS[stage - 1]
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return variant or "small"


class FoamingEnv(gym.Env):
    """Choose a grounded operation and its bounded continuous controls.

    Stage 1/2/3 selects small/main/high_dep. All stages reserve the high_dep
    operation/parameter schema and max_glasses slots. Disabled actions are rejected;
    unused parameters are ignored. The same policy can therefore retain its shape.

    Raw actions follow Tuple(Discrete(K), Tuple(Box(...), ...)), with one component
    per grounded action. Each control is normalized to [-1, 1]. build_env() converts
    this to the pipeline Dict(action_type, parameters) interface.

    Invalid domain actions receive invalid_action_penalty without changing physical
    state or elapsed time, but consume a decision. Success and domain failures are
    terminated; reaching max_steps without either is truncated. No change occurs
    between decisions. Only equation-based sequential dynamics are supported.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, *, stage: Optional[int] = None, variant: Optional[str] = None,
                 n_glasses: int = 1, max_glasses: int = 3, slack_pct: float = 20,
                 max_steps: int = 256, seed: Optional[int] = None,
                 goal_reward: float = 10.0, invalid_action_penalty: float = -0.1,
                 time_penalty: float = 0.0, render_mode: Optional[str] = None,
                 model: str = "equation"):
        super().__init__()
        self.variant = _variant(stage, variant)
        self.n_glasses = _positive_int(n_glasses, "n_glasses")
        self.max_glasses = _positive_int(max_glasses, "max_glasses")
        self.max_steps = _positive_int(max_steps, "max_steps")
        if self.n_glasses > self.max_glasses:
            raise ValueError("n_glasses must not exceed max_glasses")
        if not np.isfinite(slack_pct) or slack_pct < 0:
            raise ValueError("slack_pct must be finite and nonnegative")
        if not all(np.isfinite(v) for v in (goal_reward, invalid_action_penalty, time_penalty)):
            raise ValueError("reward settings must be finite")
        if goal_reward <= 0 or invalid_action_penalty > 0 or time_penalty < 0:
            raise ValueError("Require goal_reward > 0, invalid_action_penalty <= 0, time_penalty >= 0")
        if model != "equation":
            raise ValueError("FoamingEnv currently supports model='equation' only")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.slack_pct = float(slack_pct)
        self.goal_reward = float(goal_reward)
        self.invalid_action_penalty = float(invalid_action_penalty)
        # A coefficient per fraction of the episode's processing budget consumed.
        self.time_penalty = float(time_penalty)
        self.render_mode = render_mode
        self._initial_seed = seed
        self._needs_reset = True
        self._steps = 0
        self._invalid_actions = 0
        self._sim = None
        self._instance = None

        self._layout_specs = _specs("high_dep")
        self.actions = tuple(
            GroundedAction(name, g)
            for name in OPERATIONS
            for g in ([None] if name in GLOBAL_OPERATIONS else range(self.max_glasses))
        )
        self.action_names = [action.label for action in self.actions]
        self._action_indices = {action: i for i, action in enumerate(self.actions)}
        self.parameter_names = tuple(
            tuple(self._layout_specs[a.name].controls) if a.name in self._layout_specs else ()
            for a in self.actions
        )
        self._offsets = np.cumsum([0] + [len(names) for names in self.parameter_names])
        self.action_space = spaces.Tuple((spaces.Discrete(len(self.actions)), spaces.Tuple(tuple(
            spaces.Box(-1.0, 1.0, shape=(len(names),), dtype=np.float32)
            for names in self.parameter_names
        ))))
        self.observation_names = self._observation_names()
        observation_high = np.ones(len(self.observation_names), dtype=np.float32)
        observation_high[self.observation_names.index("budget_hours")] = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            np.zeros_like(observation_high), observation_high, dtype=np.float32)
        self._update_parameter_mask()

    @property
    def stage(self):
        return VARIANTS.index(self.variant) + 1

    def set_stage(self, stage: int):
        """Select the next episode's stage; space shapes never change."""
        if not self._needs_reset:
            raise RuntimeError("Change stages between episodes, or use reset(options={'stage': ...})")
        self.variant = _variant(stage, None)
        self._update_parameter_mask()

    def _update_parameter_mask(self):
        sim = FoamingSimulator(self.variant, self.n_glasses)
        self.parameter_mask = np.zeros(
            (len(self.actions), int(self._offsets[-1])), dtype=np.float32)
        for i, action in enumerate(self.actions):
            if (action.name not in sim.operations or
                    (action.glass is not None and action.glass >= self.n_glasses)):
                continue
            spec = sim.specs.get(action.name)
            for j, name in enumerate(self.parameter_names[i]):
                if spec is not None and name in spec.controls:
                    self.parameter_mask[i, self._offsets[i] + j] = 1.0

    def _observation_names(self):
        names = [f"stage_{v}" for v in VARIANTS]
        names += ["active_glasses", "budget_hours", "time_remaining", "decisions_remaining"]
        names += list(TOOL_BOOLS) + list(TOOL_NUMBERS) + list(GLOBAL_NUMBERS)
        for g in range(self.max_glasses):
            names += [f"g{g}.present"]
            names += [f"g{g}.{n}" for n in (*GLASS_BOOLS, *GLASS_NUMBERS)]
            for n in QUALITY["high_dep"]:
                names += [f"g{g}.{n}.target_active", f"g{g}.{n}.target_low",
                          f"g{g}.{n}.target_high"]
        return tuple(names)

    def reset(self, *, seed=None, options=None):
        if seed is None and self._initial_seed is not None:
            seed = self._initial_seed
        self._initial_seed = None
        super().reset(seed=seed)
        options = options or {}
        unknown = set(options) - {"stage", "n_glasses", "slack_pct"}
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        variant = _variant(options["stage"], None) if "stage" in options else self.variant
        n_glasses = _positive_int(options.get("n_glasses", self.n_glasses), "n_glasses")
        slack_pct = options.get("slack_pct", self.slack_pct)
        if n_glasses > self.max_glasses:
            raise ValueError("n_glasses must not exceed max_glasses")
        if not np.isfinite(slack_pct) or slack_pct < 0:
            raise ValueError("slack_pct must be finite and nonnegative")
        instance_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31))
        instance = generate_instance(variant, n_glasses, instance_seed, slack_pct)
        if len(instance.witness) > self.max_steps:
            raise ValueError("max_steps is shorter than the generated witness; increase it")
        self.variant, self.n_glasses, self.slack_pct = variant, n_glasses, float(slack_pct)
        self.instance_seed = instance_seed
        self._instance = instance
        self._sim = FoamingSimulator(self.variant, self.n_glasses)
        self._sim.state = dict(instance.initial_state)
        self._steps = self._invalid_actions = 0
        self._needs_reset = False
        self._update_parameter_mask()
        return self._obs(), self._info()

    def _obs(self):
        sim, instance = self._sim, self._instance

        def normalized(name, glass=None):
            value = float(sim.get(name, glass))
            lo, hi = RANGES.get(name, (0, 100) if name == "humidity_level" else (0, 80))
            return max(0.0, (value - lo) / (hi - lo))

        values = [float(self.variant == v) for v in VARIANTS]
        values += [self.n_glasses / self.max_glasses, float(instance.budget) / 3600,
                   max(0.0, 1 - float(sim.get("process_time") / instance.budget)),
                   max(0.0, 1 - self._steps / self.max_steps)]
        values += [float(sim.get(n)) for n in TOOL_BOOLS]
        values += [normalized(n) for n in (*TOOL_NUMBERS, *GLOBAL_NUMBERS)]
        for g in range(self.max_glasses):
            values += [float(g < self.n_glasses)]
            values += [float(sim.get(n, g)) for n in GLASS_BOOLS]
            values += [normalized(n, g) for n in GLASS_NUMBERS]
            for n in QUALITY["high_dep"]:
                bounds = instance.quality_bounds.get((n, g))
                if bounds is None:
                    values += [0.0, 0.0, 0.0]
                else:
                    lo, hi = RANGES[n]
                    values += [1.0, float((bounds[0] - lo) / (hi - lo)),
                               float((bounds[1] - lo) / (hi - lo))]
        return np.asarray(values, dtype=np.float32)

    def action_index(self, name, glass=None):
        """Return the stable discrete index of a grounded operation."""
        try:
            return self._action_indices[GroundedAction(name, glass)]
        except KeyError as exc:
            raise ValueError(f"No action {name} for glass {glass}") from exc

    def encode_action(self, name, glass=None, *, wrapped=False, **controls):
        """Encode physical controls as a raw action or build_env-compatible dictionary."""
        index = self.action_index(name, glass)
        spec = _specs(self.variant).get(name)
        expected = set(spec.controls) if spec else set()
        if set(controls) != expected:
            raise ValueError(f"Expected controls {sorted(expected)}")
        parameters = np.zeros(int(self._offsets[-1]), dtype=np.float32)
        for j, key in enumerate(self.parameter_names[index]):
            if key in controls:
                lo, hi = spec.controls[key]
                value = float(controls[key])
                if not np.isfinite(value) or not lo <= value <= hi:
                    raise ValueError(f"Control outside physical range: {key}")
                parameters[self._offsets[index] + j] = 2 * (value - float(lo)) / float(hi - lo) - 1
        if wrapped:
            return {"action_type": index, "parameters": parameters}
        return index, tuple(parameters[self._offsets[i]:self._offsets[i + 1]].copy()
                            for i in range(len(self.actions)))

    def witness_actions(self, *, wrapped=False):
        """Baseline/debug actions for the current instance; never included in observations."""
        if self._instance is None:
            raise gym.error.ResetNeeded("Call reset before requesting a witness")
        return [self.encode_action(s.action.name, s.action.glass, wrapped=wrapped, **s.controls)
                for s in self._instance.witness]

    def _quality_failures(self):
        return [f"g{g}.{name}" for (name, g), (lo, hi) in self._instance.quality_bounds.items()
                if self._sim.get("foamed", g) and not lo <= self._sim.get(name, g) <= hi]

    def _info(self, *, reason=None, invalid_reason=None, duration=0.0, success=False):
        return {
            "is_success": bool(success), "termination_reason": reason,
            "invalid_action": invalid_reason is not None, "invalid_reason": invalid_reason,
            "invalid_actions": self._invalid_actions, "episode_steps": self._steps,
            "process_time": float(self._sim.get("process_time")),
            "time_budget": float(self._instance.budget), "action_duration": float(duration),
            "stage": self.stage, "variant": self.variant, "instance_seed": self.instance_seed,
            "n_glasses": self.n_glasses,
            # A true entry permits trying controls; it does not certify every control value.
            "action_mask": np.asarray([not self._sim.state_violations(a) for a in self.actions],
                                      dtype=np.int8),
            "parameter_mask": self.parameter_mask.copy(),
            "quality_failures": self._quality_failures(),
        }

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset before stepping or after an episode ends")
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("Expected (action_type, parameter_components)")
        index, components = action
        if not self.action_space[0].contains(index):
            raise ValueError("action_type must be an integer in the discrete action space")
        index = int(index)
        if len(components) != len(self.actions):
            raise ValueError("Expected one parameter component per discrete action")
        selected = np.asarray(components[index], dtype=float)
        if selected.shape != (len(self.parameter_names[index]),):
            raise ValueError("Incorrect selected parameter shape")
        action = self.actions[index]
        spec = self._sim.specs.get(action.name)
        controls = {}
        for key, value in zip(self.parameter_names[index], selected):
            if spec is not None and key in spec.controls:
                if not np.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError("Active parameters must be finite and in [-1, 1]")
                lo, hi = spec.controls[key]
                controls[key] = float(lo) + (float(value) + 1) * 0.5 * float(hi - lo)
        self._steps += 1
        duration, invalid_reason = 0.0, None
        try:
            duration = float(self._sim.apply(action, controls))
        except InvalidAction as exc:
            invalid_reason = str(exc)
            self._invalid_actions += 1
        reward = self.invalid_action_penalty if invalid_reason is not None else 0.0
        reward -= self.time_penalty * duration / float(self._instance.budget)
        reason = None
        # Normalized controls pass through float32 agents/wrappers. Allow a tiny
        # clock round-trip error, including when slack_pct=0, without relaxing
        # the exact simulator's physical and action constraints.
        clock_tolerance = 1e-7 * max(1.0, float(self._instance.budget))
        if float(self._sim.get("process_time") - self._instance.budget) > clock_tolerance:
            reason = "budget_exceeded"
        elif self._quality_failures():
            # Each glass is foamed once; failed quality cannot subsequently be repaired.
            reason = "quality_failure"
        elif all(self._sim.get("trimmed", g) for g in range(self.n_glasses)):
            reason = "goal"
            reward += self.goal_reward
        terminated = reason is not None
        truncated = not terminated and self._steps >= self.max_steps
        if truncated:
            reason = "decision_limit"
        self._needs_reset = terminated or truncated
        return (self._obs(), float(reward), bool(terminated), bool(truncated),
                self._info(reason=reason, invalid_reason=invalid_reason, duration=duration,
                           success=reason == "goal"))

    def render(self):
        if self.render_mode != "ansi" or self._sim is None:
            return None
        lines = [f"Foaming stage {self.stage} ({self.variant}), decision {self._steps}/{self.max_steps}",
                 f"Time: {float(self._sim.get('process_time')):.1f}/{float(self._instance.budget):.1f} s; "
                 f"tool: {float(self._sim.get('tool_temperature')):.2f} C"]
        for g in range(self.n_glasses):
            flags = [n for n in GLASS_BOOLS if self._sim.get(n, g)]
            lines.append(f"g{g}: {', '.join(flags)}")
        return "\n".join(lines)
