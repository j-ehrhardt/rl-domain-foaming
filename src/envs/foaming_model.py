# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 PSO Unit, Fondazione Bruno Kessler
#
# This file is part of OCTANE.
#
# OCTANE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OCTANE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Foaming equations and sequential simulation adapted from the OCTANE planning domain.

Source thesis: https://openhsu.ub.hsu-hh.de/entities/publication/24927
The environment retains the small/main/high_dep variants and uses synthetic
reference equations, a shared foaming tool, and a processing-time budget.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

F = Fraction
VARIANTS = ["small", "main", "high_dep"]

# Seconds each symbolic action takes.
SYMBOLIC_TIME = {
    "pickup_from_glassrack": 30,
    "clean_tool": 180,
    "put_in_foaming_tool": 30,
    "apply_mold_release": 60,
    "place_inserts": 60,  # symbolic only in small
    "trimming": 120,
}

# The physical range of every numeric fluent an opaque effect assigns, from the source's bounds.
RANGES = {
    "flashoff_time": (0, 7200),
    "primer_thickness": (0, 5),
    "surface_adhesion": (0, 30),
    "primer_viscosity": (0, 40),
    "glass_temperature": (0, 80),
    "insert_hardness": (0, 100),
    "insert_flexibility": (0, 100),
    "foam_density": (0, 80),
    "chemical_concentration": (0, 20),
    "foam_expansion_rate": (0, 120),
    "cell_uniformity": (0, 100),
    "skin_quality": (0, 100),
    "curing_time": (0, 7200),
    "tool_temperature": (15, 80),
    "tool_pressure": (0, 120),
    "tool_alignment": (0, 100),
    "tool_humidity": (0, 100),
}
QUALITY = {
    "small": ["foam_density"],
    "main": ["foam_density", "chemical_concentration"],
    "high_dep": [
        "foam_density",
        "chemical_concentration",
        "cell_uniformity",
        "skin_quality",
    ],
}

MIN_CURE = 480
UNCLAMPED_ALIGNMENT = 85  # ``eject`` opens the clamp, so the next glass re-aligns
AMBIENT = 20  # the heater's zero-power target
COOLANT = 15
# Insert type -> (minimum flash-off seconds, tool temperature window) for placing inserts.
INSERT_WINDOWS = {
    "A": (2400, (41, 50)),
    "B": (1500, (25, 30)),
    "C": (1800, (31, 40)),
}
SMALL_WINDOW = (10, (30, 50))
FOAMING_MIN_T = {"small": 30, "main": 38, "high_dep": 38}

def lag(x: Fraction, target, d, tau) -> Fraction:
    """A first-order approach of ``x`` towards ``target`` over ``d`` with time constant ``tau``."""
    return x + (target - x) * d / (d + tau)


def _shot(u) -> Fraction:
    """The injected amount, growing with injection time and pressure."""
    return u["injection_time"] * (1 + (u["injection_pressure"] - 70) / 80)


def _expansion(s, u) -> Fraction:
    """How far the foam expands: more on a hotter tool, less off the stoichiometric ratios."""
    x = 1 + (s["tool_temperature"] - 38) / 100 - 2 * (u["mix_ratio"] - 1) ** 2
    if "catalyst_ratio" in u:
        cat = u["catalyst_ratio"] - 1
        x += (u["shot_temperature"] - 25) / 100 + cat / 2 - 3 * cat**2
    return x


@dataclass(frozen=True)
class Effect:
    """An opaque effect: ``fluent := body(s, u)``, ``s`` the fluents read, ``u`` the controls."""

    fluent: str
    reads: tuple[str, ...]
    body: Callable[[dict, dict], Fraction]


@dataclass
class Spec:
    """A control action: its controls' physical ranges, its clock control and its effects."""

    controls: dict[str, tuple]
    clock: str
    clock_unit: int = 1  # seconds per unit of the clock control
    effects: list[Effect] = field(default_factory=list)
    copies: list[tuple[str, str]] = field(default_factory=list)  # (fluent, control)


def _specs(variant: str) -> dict[str, Spec]:
    small, high = variant == "small", variant == "high_dep"
    specs: dict[str, Spec] = {}

    primer: dict[str, tuple] = {"primer_flow": (F(1, 5), 2)}
    if not small:
        primer["spray_pressure"] = (45, 80)
    if high:
        primer["primer_temperature"] = (20, 35)
    primer["primer_dwell_time"] = (5, 20)
    if small:
        thickness = Effect(
            "primer_thickness",
            ("primer_thickness",),
            lambda s, u: (
                s["primer_thickness"] + u["primer_flow"] * u["primer_dwell_time"] / 20
            ),
        )
        adhesion = Effect(
            "surface_adhesion",
            ("surface_adhesion",),
            lambda s, u: (
                s["surface_adhesion"]
                + u["primer_flow"] / 5
                + 3 * u["primer_dwell_time"] / 100
            ),
        )
    else:
        thickness = Effect(
            "primer_thickness",
            ("primer_thickness",),
            lambda s, u: (
                s["primer_thickness"]
                + u["primer_flow"]
                * u["primer_dwell_time"]
                / 10
                * (1 - (u["spray_pressure"] - 45) / 175)
            ),
        )
        adhesion = Effect(
            "surface_adhesion",
            ("surface_adhesion",),
            lambda s, u: (
                s["surface_adhesion"]
                + 3 * u["primer_flow"] / 2
                + u["primer_dwell_time"] / 10
            ),
        )
    specs["apply_primer"] = Spec(
        primer,
        "primer_dwell_time",
        effects=[
            thickness,
            Effect(
                "flashoff_time",
                ("flashoff_time",),
                lambda s, u: s["flashoff_time"] + u["primer_dwell_time"],
            ),
            adhesion,
        ],
    )
    if not small:
        specs["apply_primer"].effects.append(
            Effect(
                "primer_viscosity",
                (),
                (
                    lambda s, u: (
                        6 + 2 * u["primer_flow"] - (u["primer_temperature"] - 20) / 5
                    )
                )
                if high
                else (lambda s, u: 6 + 2 * u["primer_flow"]),
            )
        )
        specs["flash_off"] = Spec(
            {"flash_duration": (5, 40), "airflow": (0, 1)},
            "flash_duration",
            clock_unit=60,
            effects=[
                Effect(
                    "flashoff_time",
                    ("flashoff_time",),
                    lambda s, u: (
                        s["flashoff_time"]
                        + 60 * u["flash_duration"] * (F(4, 5) + 2 * u["airflow"] / 5)
                    ),
                ),
                Effect(
                    "primer_viscosity",
                    ("primer_viscosity",),
                    lambda s, u: lag(
                        s["primer_viscosity"],
                        40,
                        u["flash_duration"] * (F(1, 2) + u["airflow"]),
                        60,
                    ),
                ),
                Effect(
                    "surface_adhesion",
                    ("surface_adhesion",),
                    lambda s, u: lag(
                        s["surface_adhesion"], 12, u["flash_duration"], 30
                    ),
                ),
            ],
        )

    heat = [
        Effect(
            "tool_temperature",
            ("tool_temperature",),
            lambda s, u: lag(
                s["tool_temperature"],
                AMBIENT + 60 * u["heater_power"],
                u["heat_duration"],
                10,
            ),
        )
    ]
    if high:
        heat.append(
            Effect(
                "tool_humidity",
                ("tool_humidity",),
                lambda s, u: lag(
                    s["tool_humidity"], 0, u["heat_duration"] * u["heater_power"], 30
                ),
            )
        )
    specs["heat_tool"] = Spec(
        {"heat_duration": (1, 30), "heater_power": (0, 1)},
        "heat_duration",
        clock_unit=60,
        effects=heat,
    )
    specs["cool_tool"] = Spec(
        {"cool_duration": (1, 30), "coolant_flow": (0, 1)},
        "cool_duration",
        clock_unit=60,
        effects=[
            Effect(
                "tool_temperature",
                ("tool_temperature",),
                lambda s, u: lag(
                    s["tool_temperature"],
                    COOLANT,
                    u["cool_duration"],
                    10 / (F(1, 5) + u["coolant_flow"]),
                ),
            )
        ],
    )

    if high:
        specs["condition_environment"] = Spec(
            {"air_temperature": (18, 28), "conditioning_time": (1, 20)},
            "conditioning_time",
            clock_unit=60,
            effects=[
                Effect(
                    "glass_temperature",
                    ("glass_temperature",),
                    lambda s, u: lag(
                        s["glass_temperature"],
                        u["air_temperature"],
                        u["conditioning_time"],
                        5,
                    ),
                )
            ],
        )
        specs["condition_inserts"] = Spec(
            {
                "insert_preheat_time": (1, 20),
                "conditioning_load": (0, 1),
                "insert_relaxation": (0, 1),
            },
            "insert_preheat_time",
            clock_unit=60,
            effects=[
                Effect(
                    "insert_hardness",
                    ("insert_hardness",),
                    lambda s, u: (
                        s["insert_hardness"]
                        + 40 * u["conditioning_load"]
                        - u["insert_preheat_time"]
                    ),
                ),
                Effect(
                    "insert_flexibility",
                    ("insert_flexibility",),
                    lambda s, u: (
                        s["insert_flexibility"]
                        + 30 * u["insert_relaxation"]
                        + u["insert_preheat_time"]
                        - 15 * u["conditioning_load"]
                    ),
                ),
            ],
        )

    if not small:
        clamp: dict[str, tuple] = {
            "clamp_pressure": (70, 110),
            "clamp_duration": (1, 10),
        }
        clamp_effects = [
            Effect(
                "tool_pressure",
                (),
                lambda s, u: (
                    u["clamp_pressure"]
                    * u["clamp_duration"]
                    / (u["clamp_duration"] + 1)
                ),
            )
        ]
        if high:
            clamp["vacuum_level"] = (0, 1)
            clamp_effects.append(
                Effect(
                    "tool_alignment",
                    ("tool_alignment",),
                    lambda s, u: s["tool_alignment"] + 5 * u["vacuum_level"],
                )
            )
        specs["clamp_tool"] = Spec(
            clamp, "clamp_duration", effects=clamp_effects
        )
        specs["place_inserts"] = Spec(
            {
                "placement_pressure": (50, 90),
                "alignment_correction": (0, 10),
                "seating_time": (1, 10),
            },
            "seating_time",
            effects=[
                Effect(
                    "tool_alignment",
                    ("tool_alignment",),
                    lambda s, u: (
                        s["tool_alignment"] + 9 * u["alignment_correction"] / 10
                    ),
                ),
                Effect(
                    "surface_adhesion",
                    ("surface_adhesion",),
                    lambda s, u: (
                        s["surface_adhesion"]
                        + u["placement_pressure"] / 100
                        + u["seating_time"] / 20
                    ),
                ),
            ],
        )

    foam: dict[str, tuple] = {
        "injection_time": (2, 20),
        "injection_pressure": (70, 110),
    }
    if small:
        foam_effects = [
            Effect(
                "foam_density",
                ("tool_temperature",),
                lambda s, u: (
                    5 * u["injection_time"] / 2
                    + (u["injection_pressure"] - 70) / 20
                    - (s["tool_temperature"] - 30) / 20
                ),
            )
        ]
    else:
        foam["mix_ratio"] = (F(4, 5), F(6, 5))
        if high:
            foam["shot_temperature"] = (25, 45)
            foam["catalyst_ratio"] = (F(9, 10), F(11, 10))

        def residue(s, u):
            r = 2 + 100 * (u["mix_ratio"] - 1) ** 2
            if high:
                r += 150 * (u["catalyst_ratio"] - 1) ** 2
            return r

        foam_effects = [
            Effect(
                "foam_density",
                ("tool_temperature",),
                lambda s, u: 8 * _shot(u) / (5 * _expansion(s, u)),
            ),
            Effect("chemical_concentration", (), residue),
        ]
        if high:
            foam_effects += [
                Effect(
                    "foam_expansion_rate",
                    ("tool_temperature",),
                    lambda s, u: 60 * _expansion(s, u),
                ),
                Effect(
                    "cell_uniformity",
                    ("tool_alignment",),
                    lambda s, u: (
                        90
                        - 300 * (u["mix_ratio"] - 1) ** 2
                        - 2 * abs(s["tool_alignment"] - 95)
                    ),
                ),
                Effect(
                    "skin_quality",
                    ("tool_temperature",),
                    lambda s, u: (
                        60
                        + (s["tool_temperature"] - 38) / 2
                        + 3 * (u["shot_temperature"] - 25) / 10
                        - 400 * (u["catalyst_ratio"] - 1) ** 2
                    ),
                ),
            ]
    specs["foaming"] = Spec(
        foam,
        "injection_time",
        effects=foam_effects,
        copies=[] if small else [("tool_pressure", "injection_pressure")],
    )
    specs["eject"] = Spec(
        {"cure_duration": (2, 30)},
        "cure_duration",
        clock_unit=60,
        effects=[
            Effect(
                "curing_time",
                ("tool_temperature",),
                lambda s, u: (
                    60 * u["cure_duration"] * (1 + (s["tool_temperature"] - 30) / 50)
                ),
            )
        ],
    )
    return specs


# A common schema gives every curriculum stage the same observation and action layout.
GLASS_BOOLS = (
    "in_rack", "centered", "primer_applied", "in_foaming_tool", "inserts_placed",
    "foamed", "trimmed", "insert_type_A", "insert_type_B", "insert_type_C",
    "environment_conditioned", "inserts_conditioned",
)
TOOL_BOOLS = ("tool_clean", "tool_occupied", "tool_clamped", "mold_release_applied")
TOOL_NUMBERS = ("tool_temperature", "tool_pressure", "tool_alignment", "tool_humidity")
GLASS_NUMBERS = tuple(n for n in RANGES if n not in TOOL_NUMBERS)
GLOBAL_NUMBERS = ("ambient_temperature", "humidity_level")
OPERATIONS = (
    "pickup_from_glassrack", "apply_primer", "flash_off", "condition_environment",
    "condition_inserts", "clean_tool", "put_in_foaming_tool", "clamp_tool",
    "apply_mold_release", "heat_tool", "cool_tool", "place_inserts", "foaming",
    "eject", "trimming",
)
GLOBAL_OPERATIONS = frozenset((
    "clean_tool", "clamp_tool", "apply_mold_release", "heat_tool", "cool_tool",
))


def exact(value) -> Fraction:
    """Interpret decimal inputs without introducing a binary-float denominator."""
    return value if isinstance(value, Fraction) else F(str(value))


@dataclass(frozen=True)
class GroundedAction:
    name: str
    glass: Optional[int] = None

    @property
    def label(self) -> str:
        return self.name if self.glass is None else f"{self.name}(g{self.glass})"


@dataclass(frozen=True)
class PlanStep:
    action: GroundedAction
    controls: dict[str, Fraction]


@dataclass
class FoamingInstance:
    initial_state: dict
    quality_bounds: dict
    budget: Fraction
    witness: tuple[PlanStep, ...]
    witness_time: Fraction


class InvalidAction(ValueError):
    """An action that fails a domain precondition or a physical output bound."""


class FoamingSimulator:
    """Exact, atomic sequential transitions with the planning domain's preconditions.

    State keys are (fluent_name, glass_index), or (fluent_name, None) for shared fluents.
    Invalid actions leave both the physical state and process_time unchanged.
    """

    def __init__(self, variant: str, n_glasses: int):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        if isinstance(n_glasses, bool) or not isinstance(n_glasses, int) or n_glasses < 1:
            raise ValueError("n_glasses must be a positive integer")
        self.variant, self.n_glasses = variant, n_glasses
        self.small, self.high = variant == "small", variant == "high_dep"
        self.specs = _specs(variant)
        self.operations = set(self.specs) | {
            "pickup_from_glassrack", "clean_tool", "put_in_foaming_tool", "trimming",
        }
        self.operations.add("place_inserts" if self.small else "apply_mold_release")
        self.state: dict = {}

    def get(self, name: str, glass: Optional[int] = None):
        return self.state.get((name, glass), F(0))

    def seed_state(self, rng: random.Random) -> dict:
        def draw(lo, hi):
            return exact(round(rng.uniform(lo, hi), 1))

        state = {("process_time", None): F(0), ("tool_temperature", None): draw(18, 25)}
        for name in TOOL_BOOLS:
            state[(name, None)] = False
        for g in range(self.n_glasses):
            for name in GLASS_BOOLS:
                state[(name, g)] = name == "in_rack"
            for name in GLASS_NUMBERS:
                state[(name, g)] = F(0)
            state[("surface_adhesion", g)] = draw(0, F(1, 2))
            if not self.small:
                state[("glass_temperature", g)] = draw(10, 18) if self.high else draw(21, 35)
            if self.high:
                state[("insert_hardness", g)] = draw(45, 55)
                state[("insert_flexibility", g)] = draw(10, 20)
        if not self.small:
            state[("tool_pressure", None)] = F(0)
            state[("tool_alignment", None)] = draw(83, 88)
            state[("humidity_level", None)] = draw(25, 55)
        if self.high:
            state[("tool_humidity", None)] = draw(56, 70)
            state[("ambient_temperature", None)] = draw(15, 22)
        for g in range(self.n_glasses):
            insert_type = rng.choice("ABC")
            if not self.small:
                state[(f"insert_type_{insert_type}", g)] = True
        return state

    def state_violations(self, action: GroundedAction) -> list[str]:
        """Check state-only conditions; effect-dependent conditions need controls too."""
        n, g = action.name, action.glass
        if n not in self.operations:
            return ["operation_disabled"]
        if n in GLOBAL_OPERATIONS:
            if g is not None:
                return ["unexpected_glass"]
        elif isinstance(g, bool) or not isinstance(g, int) or not 0 <= g < self.n_glasses:
            return ["glass_disabled"]
        s = lambda name: self.get(name, g)
        t = lambda name: self.get(name)
        checks = []

        def need(condition, reason):
            if not condition:
                checks.append(reason)

        if n == "pickup_from_glassrack":
            need(s("in_rack"), "glass_not_in_rack")
        elif n == "clean_tool":
            need(not t("tool_occupied"), "tool_occupied")
        elif n == "put_in_foaming_tool":
            need(s("primer_applied"), "primer_missing")
            need(not s("in_foaming_tool"), "glass_already_in_tool")
            need(not s("foamed"), "glass_already_foamed")
            need(not t("tool_occupied"), "tool_occupied")
        elif n == "trimming":
            need(s("foamed"), "glass_not_foamed")
            need(not s("in_foaming_tool"), "glass_still_in_tool")
        elif n == "apply_primer":
            need(s("centered"), "glass_not_centered")
            need(not s("primer_applied"), "primer_already_applied")
        elif n in ("heat_tool", "cool_tool"):
            need(t("tool_clean"), "tool_dirty")
        elif n == "flash_off":
            need(s("primer_applied"), "primer_missing")
        elif n == "apply_mold_release":
            need(t("tool_occupied"), "tool_empty")
        elif n == "clamp_tool":
            need(t("tool_occupied"), "tool_empty")
            need(t("tool_clean"), "tool_dirty")
            need(not t("tool_clamped"), "tool_already_clamped")
        elif n == "place_inserts":
            need(s("in_foaming_tool"), "glass_not_in_tool")
            need(t("tool_clean"), "tool_dirty")
            need(not s("inserts_placed"), "inserts_already_placed")
            if self.small:
                min_flash, window = SMALL_WINDOW
                need(F(1, 2) <= s("primer_thickness") <= 2, "primer_thickness")
                need(s("surface_adhesion") >= F(1, 2), "surface_adhesion")
            else:
                types = [k for k in "ABC" if s(f"insert_type_{k}")]
                if len(types) != 1:
                    return checks + ["invalid_insert_type"]
                min_flash, window = INSERT_WINDOWS[types[0]]
                if types[0] == "C":
                    need(t("mold_release_applied"), "mold_release_missing")
                need(20 <= t("humidity_level") <= 60, "ambient_humidity")
                need(s("glass_temperature") >= 20, "glass_temperature")
                need(s("surface_adhesion") >= 5, "surface_adhesion")
                need(t("tool_pressure") >= 70, "tool_pressure")
                need(s("primer_viscosity") >= 15, "primer_viscosity")
                need(1 <= s("primer_thickness") <= 3, "primer_thickness")
                need(t("tool_clamped"), "tool_unclamped")
                if self.high:
                    need(s("environment_conditioned"), "environment_not_conditioned")
                    need(s("inserts_conditioned"), "inserts_not_conditioned")
                    need(t("ambient_temperature") <= 25, "ambient_temperature")
                    need(s("insert_hardness") >= 60, "insert_hardness")
                    need(s("insert_flexibility") >= 30, "insert_flexibility")
                    need(t("tool_humidity") <= 55, "tool_humidity")
            need(s("flashoff_time") >= min_flash, "flashoff_time")
            need(window[0] <= t("tool_temperature") <= window[1], "placement_temperature")
        elif n == "condition_environment":
            need(not s("environment_conditioned"), "environment_already_conditioned")
        elif n == "condition_inserts":
            need(not s("inserts_conditioned"), "inserts_already_conditioned")
        elif n == "foaming":
            need(s("inserts_placed"), "inserts_missing")
            need(not s("foamed"), "glass_already_foamed")
            need(t("tool_temperature") >= FOAMING_MIN_T[self.variant], "foaming_temperature")
        elif n == "eject":
            need(s("foamed"), "glass_not_foamed")
            need(s("in_foaming_tool"), "glass_not_in_tool")
        return checks

    def propose(self, action: GroundedAction, controls: dict) -> tuple[dict, Fraction]:
        """Evaluate a complete action without mutating the current state."""
        violations = self.state_violations(action)
        if violations:
            raise InvalidAction(", ".join(violations))
        n, g = action.name, action.glass
        spec = self.specs.get(n)
        expected = set(spec.controls) if spec else set()
        if set(controls) != expected:
            raise InvalidAction("incorrect_control_names")
        try:
            u = {k: exact(v) for k, v in controls.items()}
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
            raise InvalidAction("non_finite_control") from exc
        new = dict(self.state)
        if spec:
            if any(not lo <= u[k] <= hi for k, (lo, hi) in spec.controls.items()):
                raise InvalidAction("control_out_of_bounds")
            for e in spec.effects:
                reads = {r: self.get(r, None if r in TOOL_NUMBERS else g) for r in e.reads}
                value = e.body(reads, u)
                lo, hi = RANGES[e.fluent]
                if not lo <= value <= hi:
                    raise InvalidAction(f"output_out_of_bounds:{e.fluent}")
                new[(e.fluent, None if e.fluent in TOOL_NUMBERS else g)] = value
            for fluent, control in spec.copies:
                new[(fluent, None)] = u[control]
            duration = spec.clock_unit * u[spec.clock]
        else:
            duration = F(SYMBOLIC_TIME[n])
        if n == "place_inserts" and not self.small and new[("tool_alignment", None)] < 90:
            raise InvalidAction("resulting_alignment_below_90")
        if n == "eject" and new[("curing_time", g)] < MIN_CURE:
            raise InvalidAction("insufficient_curing")

        def put(name, value, glass=g):
            new[(name, glass)] = value

        if n == "pickup_from_glassrack":
            put("in_rack", False)
            put("centered", True)
        elif n == "clean_tool":
            put("tool_clean", True, None)
        elif n == "put_in_foaming_tool":
            put("in_foaming_tool", True)
            put("tool_occupied", True, None)
        elif n in ("trimming", "apply_primer", "place_inserts", "foaming",
                   "condition_environment", "condition_inserts"):
            flag = {
                "trimming": "trimmed", "apply_primer": "primer_applied",
                "place_inserts": "inserts_placed", "foaming": "foamed",
                "condition_environment": "environment_conditioned",
                "condition_inserts": "inserts_conditioned",
            }[n]
            put(flag, True)
        elif n == "clamp_tool":
            put("tool_clamped", True, None)
        elif n == "apply_mold_release":
            put("mold_release_applied", True, None)
        elif n == "eject":
            put("in_foaming_tool", False)
            put("tool_occupied", False, None)
            put("tool_clean", False, None)
            if not self.small:
                put("tool_clamped", False, None)
                put("tool_alignment", F(UNCLAMPED_ALIGNMENT), None)
                put("mold_release_applied", False, None)
        new[("process_time", None)] = self.get("process_time") + duration
        return new, duration

    def apply(self, action: GroundedAction, controls: dict) -> Fraction:
        new, duration = self.propose(action, controls)
        self.state = new
        return duration


def generate_instance(variant: str, n_glasses: int, seed: int, slack_pct=20) -> FoamingInstance:
    """Construct targets around a validated witness using the source's sampling recipe."""
    slack = exact(slack_pct) / 100
    if slack < 0:
        raise ValueError("slack_pct must be nonnegative")
    rng = random.Random(seed)
    sim = FoamingSimulator(variant, n_glasses)
    initial = sim.seed_state(rng)
    sim.state = dict(initial)
    steps = []

    def step(name, glass=None, pred=lambda state: True):
        action = GroundedAction(name, glass)
        spec = sim.specs.get(name)
        for attempt in range(600 if spec else 1):
            shrink = F(1, 5) if attempt < 300 else F(0)
            controls = {
                key: exact(round(rng.uniform(float(lo + (hi - lo) * shrink),
                                             float(hi - (hi - lo) * shrink)), 2))
                for key, (lo, hi) in (spec.controls.items() if spec else ())
            }
            try:
                new, _ = sim.propose(action, controls)
            except InvalidAction:
                continue
            if pred(lambda n: new.get((n, None if n in TOOL_NUMBERS else glass), F(0))):
                sim.state = new
                steps.append(PlanStep(action, controls))
                return True
        return False

    def must(name, glass=None, pred=lambda state: True):
        if not step(name, glass, pred):
            raise RuntimeError(f"Cannot construct witness: {variant}, seed={seed}, action={name}")

    def tool_to(lo, hi):
        temperature = sim.get("tool_temperature")
        if not lo <= temperature <= hi:
            must("heat_tool" if temperature < lo else "cool_tool",
                 pred=lambda s: lo <= s("tool_temperature") <= hi)

    order = list(range(n_glasses))
    rng.shuffle(order)
    for g in order:
        must("pickup_from_glassrack", g)
        if sim.small:
            min_flash, (lo_t, hi_t) = SMALL_WINDOW
            must("apply_primer", g, lambda s: (
                s("flashoff_time") >= min_flash and F(1, 2) <= s("primer_thickness") <= 2
                and s("surface_adhesion") >= F(1, 2)))
        else:
            insert_type = next(k for k in "ABC" if sim.get(f"insert_type_{k}", g))
            min_flash, (lo_t, hi_t) = INSERT_WINDOWS[insert_type]
            if sim.high:
                must("condition_environment", g, lambda s: s("glass_temperature") >= F(41, 2))
                must("condition_inserts", g, lambda s: (
                    s("insert_hardness") >= 61 and s("insert_flexibility") >= 31))
            must("apply_primer", g, lambda s: 1 <= s("primer_thickness") <= 3)

            def primed(s):
                return (s("flashoff_time") >= min_flash and s("primer_viscosity") >= 15
                        and s("surface_adhesion") >= 5)

            if not step("flash_off", g, primed):
                must("flash_off", g, lambda s: s("flashoff_time") <= 4800)
                must("flash_off", g, primed)
        must("clean_tool")
        must("put_in_foaming_tool", g)
        if not sim.small:
            must("clamp_tool", pred=lambda s: s("tool_pressure") >= 71)
            if insert_type == "C":
                must("apply_mold_release")
            if sim.high and sim.get("tool_humidity") > 54:
                must("heat_tool", pred=lambda s: s("tool_humidity") <= 54)
        tool_to(lo_t + F(1, 2), hi_t - F(1, 2))
        must("place_inserts", g, lambda s: sim.small or s("tool_alignment") >= 91)
        if sim.get("tool_temperature") < FOAMING_MIN_T[variant]:
            tool_to(FOAMING_MIN_T[variant] + F(1, 2), 60)
        must("foaming", g)
        must("eject", g, lambda s: s("curing_time") >= MIN_CURE + F(1, 2))
        must("trimming", g)

    quality_bounds = {}
    for g in range(n_glasses):
        for name in QUALITY[variant]:
            lo, hi = RANGES[name]
            width = (hi - lo) * (F(2, 100) + slack)
            value = sim.get(name, g)
            quality_bounds[(name, g)] = (max(lo, value - width), min(hi, value + width))
    witness_time = sim.get("process_time")
    return FoamingInstance(initial, quality_bounds, witness_time * (1 + slack),
                           tuple(steps), witness_time)

