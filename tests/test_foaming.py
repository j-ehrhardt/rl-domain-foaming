"""Run with python -m unittest discover -s tests -v; no planning stack needed."""

import random
import unittest
from fractions import Fraction

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from src.envs import build_env
from src.envs.env_loader import hybrid_parameter_bounds, infer_dims
from src.envs.foaming import FoamingEnv
from src.envs.foaming_model import (
    RANGES, TOOL_BOOLS, FoamingSimulator, GroundedAction, InvalidAction, generate_instance,
)


class SimulatorTests(unittest.TestCase):
    def test_reference_type_b_sequence(self):
        # Independently calculated reference: insert at 30 C, inject at 40 C.
        sim = FoamingSimulator("main", 1)
        sim.state = sim.seed_state(random.Random(0))
        sim.state.update({("tool_temperature", None): Fraction(20),
                          ("tool_alignment", None): Fraction(85),
                          ("humidity_level", None): Fraction(40),
                          ("surface_adhesion", 0): Fraction(3, 10),
                          ("glass_temperature", 0): Fraction(25)})
        for kind in "ABC":
            sim.state[(f"insert_type_{kind}", 0)] = kind == "B"

        def apply(name, glass=0, **controls):
            return sim.apply(GroundedAction(name, glass), controls)

        apply("pickup_from_glassrack")
        apply("apply_primer", primer_flow=1, spray_pressure=45, primer_dwell_time=20)
        self.assertEqual(sim.get("primer_thickness", 0), 2)
        apply("flash_off", flash_duration=30, airflow=1)
        self.assertEqual(sim.get("flashoff_time", 0), 2180)
        self.assertEqual(sim.get("primer_viscosity", 0), Fraction(152, 7))
        apply("clean_tool", None)
        apply("put_in_foaming_tool")
        apply("clamp_tool", None, clamp_pressure=90, clamp_duration=5)
        self.assertEqual(sim.get("tool_pressure"), 75)
        apply("heat_tool", None, heat_duration=5, heater_power=Fraction(1, 2))
        self.assertEqual(sim.get("tool_temperature"), 30)
        apply("place_inserts", placement_pressure=70, alignment_correction=10, seating_time=5)
        self.assertEqual(sim.get("tool_alignment"), 94)
        apply("heat_tool", None, heat_duration=10, heater_power=Fraction(1, 2))
        self.assertEqual(sim.get("tool_temperature"), 40)
        apply("foaming", injection_time=10, injection_pressure=90, mix_ratio=1)
        self.assertEqual(sim.get("foam_density", 0), Fraction(1000, 51))
        self.assertEqual(sim.get("chemical_concentration", 0), 2)
        apply("eject", cure_duration=8)
        self.assertEqual(sim.get("curing_time", 0), 576)
        apply("trimming")
        self.assertTrue(sim.get("trimmed", 0))
        self.assertEqual(sim.get("process_time"), 3580)

    def test_effect_dependent_conditions_are_atomic(self):
        instance = generate_instance("main", 1, 0)
        sim = FoamingSimulator("main", 1)
        sim.state = dict(instance.initial_state)
        for step in instance.witness:
            if step.action.name in ("place_inserts", "eject"):
                controls = dict(step.controls)
                if step.action.name == "place_inserts":
                    controls["alignment_correction"] = Fraction(0)
                else:
                    controls["cure_duration"] = Fraction(2)
                before = dict(sim.state)
                with self.assertRaises(InvalidAction):
                    sim.apply(step.action, controls)
                self.assertEqual(before, sim.state)
            sim.apply(step.action, step.controls)

    def test_ejection_resets_shared_tool(self):
        instance = generate_instance("high_dep", 2, 0)
        sim = FoamingSimulator("high_dep", 2)
        sim.state = dict(instance.initial_state)
        for step in instance.witness:
            temperature = sim.get("tool_temperature")
            sim.apply(step.action, step.controls)
            if step.action.name == "eject":
                for name in TOOL_BOOLS:
                    self.assertFalse(sim.get(name))
                self.assertEqual(sim.get("tool_alignment"), 85)
                self.assertEqual(sim.get("tool_temperature"), temperature)
                self.assertGreaterEqual(sim.get("curing_time", step.action.glass), 480)


class EnvironmentTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        for stage in (1, 2, 3):
            check_env(FoamingEnv(stage=stage), skip_render_check=True)

    def test_witnesses_through_wrappers(self):
        observed_operations = set()
        observed_types = set()
        for stage in (1, 2, 3):
            env = build_env("Foaming-v0", stage=stage, n_glasses=3)
            for slack_pct in (0, 20):
                for seed in range(8):
                    with self.subTest(stage=stage, slack=slack_pct, seed=seed):
                        obs, info = env.reset(seed=seed, options={"slack_pct": slack_pct})
                        self.assertTrue(env.observation_space.contains(obs))
                        for g in range(3):
                            observed_types.update(k for k in "ABC" if env.unwrapped._sim.get(f"insert_type_{k}", g))
                        for step, action in zip(env.unwrapped._instance.witness,
                                                env.unwrapped.witness_actions(wrapped=True)):
                            observed_operations.add(step.action.name)
                            self.assertTrue(env.action_space.contains(action))
                            self.assertTrue(info["action_mask"][action["action_type"]])
                            obs, reward, terminated, truncated, info = env.step(action)
                            self.assertFalse(info["invalid_action"], info)
                            self.assertTrue(env.observation_space.contains(obs))
                            if terminated or truncated:
                                break
                        self.assertTrue(info["is_success"], info)
                        self.assertTrue(terminated)
                        self.assertFalse(truncated)
                        self.assertEqual(reward, 10.0)
                        self.assertAlmostEqual(info["process_time"],
                                               float(env.unwrapped._instance.witness_time), places=3)
            env.close()
        self.assertIn("cool_tool", observed_operations)
        self.assertEqual(observed_types, set("ABC"))

    def test_stages_and_glass_counts_keep_fixed_spaces(self):
        env = build_env("Foaming-v0")
        dims = infer_dims(env)
        active_counts = []
        for stage in (1, 2, 3):
            obs, info = env.reset(seed=11, options={"stage": stage, "n_glasses": stage})
            self.assertEqual(infer_dims(env), dims)
            _, _, names, lows, highs, masks = hybrid_parameter_bounds(env)
            self.assertEqual(names, env.unwrapped.action_names)
            np.testing.assert_array_equal(masks, info["parameter_mask"])
            self.assertTrue(np.all(lows[masks == 0] == 0))
            self.assertTrue(np.all(highs[masks == 0] == 0))
            active_counts.append(masks.sum())
            if stage < 3:
                self.assertFalse(info["action_mask"][env.unwrapped.action_index("apply_primer", 2)])
        self.assertEqual(active_counts, sorted(active_counts))
        with self.assertRaises(RuntimeError):
            env.unwrapped.set_stage(1)

    def test_invalid_actions_have_no_physical_effect(self):
        env = FoamingEnv(stage=2)
        obs, info = env.reset(seed=1)
        before = dict(env._sim.state)
        invalid = env.encode_action("apply_primer", 0, primer_flow=1, spray_pressure=60,
                                    primer_dwell_time=10)
        self.assertFalse(info["action_mask"][invalid[0]])
        after_obs, reward, terminated, truncated, info = env.step(invalid)
        self.assertEqual(env._sim.state, before)
        self.assertEqual(reward, -0.1)
        self.assertFalse(terminated or truncated)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["process_time"], 0)
        self.assertEqual(info["episode_steps"], 1)
        self.assertNotEqual(obs[env.observation_names.index("decisions_remaining")],
                            after_obs[env.observation_names.index("decisions_remaining")])

    def test_invalid_loop_truncates(self):
        env = FoamingEnv(max_steps=32)
        env.reset(seed=0)
        action = env.encode_action("apply_primer", 0, primer_flow=1, primer_dwell_time=10)
        for _ in range(32):
            _, _, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "decision_limit")
        self.assertEqual(info["process_time"], 0)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(action)

    def test_budget_exhaustion_is_domain_failure(self):
        env = FoamingEnv(slack_pct=0)
        env.reset(seed=0)
        clean = env.encode_action("clean_tool")
        for _ in range(env.max_steps):
            _, _, terminated, truncated, info = env.step(clean)
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["is_success"])
        self.assertEqual(info["termination_reason"], "budget_exceeded")

    def test_bad_foam_quality_ends_episode(self):
        env = FoamingEnv(slack_pct=0)
        env.reset(seed=3)
        for step in env._instance.witness:
            if step.action.name == "foaming":
                controls = dict(step.controls)
                for injection_time in (2, 20):
                    controls["injection_time"] = injection_time
                    new, _ = env._sim.propose(step.action, controls)
                    lo, hi = env._instance.quality_bounds[("foam_density", 0)]
                    if not lo <= new[("foam_density", 0)] <= hi:
                        break
                else:
                    self.fail("Expected at least one deliberately off-target injection")
                action = env.encode_action("foaming", 0, **controls)
                _, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertFalse(info["is_success"])
                self.assertEqual(info["termination_reason"], "quality_failure")
                self.assertLess(reward, env.goal_reward)
                break
            env.step(env.encode_action(step.action.name, step.action.glass, **step.controls))

    def test_targets_and_budget_are_observable(self):
        env = FoamingEnv(stage=3, n_glasses=2)
        obs, _ = env.reset(seed=42)
        for (name, g), (lo, hi) in env._instance.quality_bounds.items():
            a, b = RANGES[name]
            index = env.observation_names.index(f"g{g}.{name}.target_low")
            self.assertAlmostEqual(float(obs[index]), float((lo - a) / (b - a)), places=6)
            self.assertAlmostEqual(float(obs[index + 1]), float((hi - a) / (b - a)), places=6)
        self.assertAlmostEqual(float(obs[env.observation_names.index("budget_hours")]),
                               float(env._instance.budget) / 3600, places=5)

    def test_reproducible_resets_and_constructor_seed(self):
        a, b = FoamingEnv(stage=3, seed=9), FoamingEnv(stage=3)
        obs_a, info_a = a.reset()
        obs_b, info_b = b.reset(seed=9)
        np.testing.assert_array_equal(obs_a, obs_b)
        self.assertEqual(info_a["time_budget"], info_b["time_budget"])
        np.testing.assert_array_equal(a.reset()[0], b.reset()[0])

    def test_disabled_actions_and_unused_parameters(self):
        env = FoamingEnv(stage=1, max_glasses=3)
        _, info = env.reset(seed=0)
        for name, glass in (("flash_off", 0), ("condition_inserts", 0), ("pickup_from_glassrack", 2)):
            index = env.action_index(name, glass)
            self.assertFalse(info["action_mask"][index])
            components = tuple(np.zeros(len(p), dtype=np.float32) for p in env.parameter_names)
            _, reward, terminated, truncated, info = env.step((index, components))
            self.assertEqual(reward, env.invalid_action_penalty)
            self.assertFalse(terminated or truncated)
            self.assertTrue(info["invalid_action"])
        env.step(env.encode_action("pickup_from_glassrack", 0))
        index, components = env.encode_action("apply_primer", 0, primer_flow=1, primer_dwell_time=10)
        components[index][1:3] = np.nan  # Spray pressure and primer temperature are unused in small.
        _, _, _, _, info = env.step((index, components))
        self.assertFalse(info["invalid_action"])

    def test_invalid_configuration_and_malformed_action(self):
        for kwargs in ({"stage": 4}, {"stage": 2, "variant": "small"},
                       {"n_glasses": 0}, {"max_glasses": 1, "n_glasses": 2},
                       {"slack_pct": -1}, {"slack_pct": np.nan}, {"max_steps": 0},
                       {"model": "mlp"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FoamingEnv(**kwargs)
        env = FoamingEnv()
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(None)
        env.reset(seed=0)
        before = dict(env._sim.state)
        index, components = env.encode_action("apply_primer", 0, primer_flow=1, primer_dwell_time=10)
        components[index][0] = np.nan
        with self.assertRaises(ValueError):
            env.step((index, components))
        self.assertEqual(env._sim.state, before)
        self.assertEqual(env._steps, 0)
        with self.assertRaises(ValueError):
            env.reset(options={"unexpected": True})

    def test_registry_names(self):
        for name, stage in (("FoamingSmall-v0", 1), ("FoamingMain-v0", 2),
                            ("FoamingHighDep-v0", 3)):
            env = build_env(name)
            _, info = env.reset(seed=1)
            self.assertEqual(info["stage"], stage)
            env.close()
        with self.assertRaises(ValueError):
            build_env("NavigationBarrier-v0")

    def test_wrapper_accepts_compact_and_full_parameters(self):
        env = build_env("FoamingMain-v0")
        env.reset(seed=0)
        raw = env.unwrapped
        env.step({"action_type": raw.action_index("pickup_from_glassrack", 0),
                  "parameters": np.zeros(0, dtype=np.float32)})
        action = raw.encode_action("apply_primer", 0, wrapped=True,
                                   primer_flow=1, spray_pressure=60, primer_dwell_time=10)
        index = action["action_type"]
        compact = action["parameters"][raw._offsets[index]:raw._offsets[index + 1]]
        self.assertEqual(env.get_param_space(index).shape, compact.shape)
        _, _, terminated, truncated, info = env.step({"action_type": index, "parameters": compact})
        self.assertFalse(terminated or truncated)
        self.assertFalse(info["invalid_action"])
        with self.assertRaises(ValueError):
            env.step({"action_type": index, "parameters": np.zeros(1000)})



if __name__ == "__main__":
    unittest.main()
