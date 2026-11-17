# Foaming reinforcement learning environment

A Gymnasium environment for manufacturing foamed glass units with a shared tool.
An agent chooses manufacturing operations and their continuous control settings,
finishes every glass within a processing-time budget, and meets per-glass foam
quality targets.

The planning domain is based on the [source thesis on openHSU, publication
24927](https://openhsu.ub.hsu-hh.de/entities/publication/24927). The transition
equations were adapted from the OCTANE foaming domain. This repository provides
the sequential RL adaptation with synthetic reference dynamics.


## Installation and quick start

Requires Python 3.9+, Gymnasium, and NumPy. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

```python
from src.envs import build_env, hybrid_parameter_bounds, infer_dims

env = build_env("Foaming-v0", stage=1, n_glasses=1, max_glasses=3)
obs, info = env.reset(seed=42)
state_dim, action_dim, param_dim = infer_dims(env)
_, _, names, lows, highs, parameter_masks = hybrid_parameter_bounds(env)


# Replay the feasible baseline to check the environment end to end.
for action in env.unwrapped.witness_actions(wrapped=True):
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
assert info["is_success"]
```

A learned policy supplies the same dictionary action:

```python
# action = {"action_type": operation_index, "parameters": parameter_vector}
# obs, reward, terminated, truncated, info = env.step(action)
```

## Three curriculum stages

| Stage | Variant / environment name | Features |
| --- | --- | --- |
| 1 | `small` / `FoamingSmall-v0` | Primer, temperature control, inserts, injection, curing; density target |
| 2 | `main` / `FoamingMain-v0` | Adds flash-off, clamping, insert types A/B/C, mold release, chemical quality |
| 3 | `high_dep` / `FoamingHighDep-v0` | Adds conditioning, humidity, extra controls, cell uniformity and skin quality |

`Foaming-v0` defaults to stage 1. Stages are selected explicitly; there is no
automatic promotion. Increase the number of glasses to introduce shared-tool
coordination. Capacity is fixed at construction, while active glass count and stage
can change at reset:

```python
obs, info = env.reset(seed=43, options={"stage": 2, "n_glasses": 2})
_, _, names, lows, highs, parameter_masks = hybrid_parameter_bounds(env)
env.close()
```

Re-query parameter masks after changing the stage or active glass count. With the
default `max_glasses=3`, all stages use **131 observations, 35 discrete choices,
and 67 flat control parameters**. Disabled choices and unused fields are masked.
`reset(options=...)` accepts `stage`, `n_glasses`, and `slack_pct`; these settings
persist for subsequent resets. `set_stage(stage)` is also available between episodes.

## RL interface

`build_env()` exposes the pipeline's existing interface: a flat float32 observation
and `Dict(action_type=Discrete(K), parameters=Box(P))`. A discrete choice identifies
an operation and, where applicable, its target glass. The full control vector
concatenates one component per choice. Only the selected component is used.

All active controls lie in `[-1, 1]` and map linearly to their physical ranges.
The wrapper also accepts a compact vector for the selected choice, including an
empty vector for a symbolic operation. `get_param_space(index)` returns that
component's space. Fields unused in a stage are ignored.

`FoamingEnv` itself exposes the raw tuple action format for direct use:

```python
from src.envs import FoamingEnv

env = FoamingEnv(stage=2, render_mode="ansi")
obs, info = env.reset(seed=7)
env.step(env.encode_action("pickup_from_glassrack", 0))
obs, reward, terminated, truncated, info = env.step(env.encode_action(
    "apply_primer", 0, primer_flow=1, spray_pressure=60, primer_dwell_time=15,
))
print(env.render())
env.close()
```

`encode_action(..., wrapped=True)` converts named physical controls to the loader's
dictionary format. `action_names`, `parameter_names`, and `observation_names`
provide the stable layouts.

Observations contain the stage, active glass count, total budget in hours, remaining
time and decision fractions, tool state, shop climate, each glass's progress and
physical properties, and its quality target intervals. Physical properties are
normalized using their domain ranges; flags and remaining fractions lie in `[0, 1]`.
The budget in hours may exceed 1. Unused glass slots are zero. Targets and budget
are visible to the policy when they vary between episodes.

`info` reports:

- `is_success`, `termination_reason`, and `quality_failures`.
- `invalid_action`, `invalid_reason`, `invalid_actions`, and `episode_steps`.
- `process_time`, `time_budget`, and `action_duration`, in seconds.
- `action_mask`: state-only applicability. A true entry permits trying controls;
  it does not guarantee that every parameter combination is feasible.
- `parameter_mask`: a `(K, P)` matrix identifying active control fields.
- `stage`, `variant`, `n_glasses`, and `instance_seed`.

## Dynamics, rewards, and episodes

One RL decision executes a complete operation and advances simulated time by its
duration. The foaming tool holds one glass at a time. Ejection frees it dirty and
unclamped; its temperature persists for the next glass. Type-B inserts require
25–30°C at placement, followed by at least 38°C for foaming in stages 2 and 3.
The agent must arrange heating and cooling explicitly. Waiting causes no autonomous
physical change, and operations execute sequentially.

The model uses exact rational arithmetic and checks all action preconditions and
physical output bounds before committing effects. An invalid domain action leaves
the physical state and processing clock unchanged, receives `-0.1` by default,
and consumes one decision. Malformed actions raise `ValueError` without consuming
a decision. Active parameters must be finite and within bounds.

Default rewards are `10` on success and `0` for other valid actions. Optional
`time_penalty >= 0` subtracts `time_penalty * duration / budget` per valid action;
it defaults to zero. `goal_reward` and `invalid_action_penalty` are configurable.

Episodes terminate on `goal`, `budget_exceeded`, or irreversible `quality_failure`.
Success requires every active glass to be trimmed and meet all its quality targets
within budget. The external `max_steps` limit truncates with `decision_limit`.
Reset rejects a decision limit shorter than the generated feasible baseline.
The clock comparison allows a relative `1e-7` tolerance for float32 control round
trips; physical simulator checks remain exact.

Seeded instances are built around a feasible witness. The budget is its duration
plus `slack_pct` percent, and quality windows are centered on its outputs with a
half-width of `(2 + slack_pct)` percent of each property's physical range.
The default slack is 20 percent. Thus increasing slack relaxes both time and quality.
`reset(seed=...)` recreates the same task; unseeded resets advance the random stream.
Use disjoint training and evaluation seeds. Witness controls are available explicitly
for debugging and baselines and are excluded from observations and `info`.

## Repository layout

```text
src/envs/
    foaming.py          Gymnasium environment, curriculum, observations, rewards
    foaming_model.py    Equations, exact transitions, feasible instance generation
    env_wrappers.py     Dictionary-action interface for the RL pipeline
    env_loader.py       Construction, dimensions, bounds and masks
    hybrid_registry.py  Foaming environment names
    __init__.py         Public API
tests/test_foaming.py    Environment and transition regression tests
```
