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

"""Foaming: glass units are primed, fitted with inserts and foamed in one shared foaming tool.

Adapted from the BBG ANML domains ``foaming_small``, ``foaming_main`` and ``foaming_high_dep``
(the citation follows once the paper is published), selected by the domain parameter ``variant``.
Their numeric effects are reference equations standing in for learned transition models, so here
every numeric effect of a control action is an opaque function ``<fluent>_after_<action>`` of the
fluents it reads and all of the action's controls, and the source's ``fluent + delta <= max``
preconditions become bounds on that same application, ``lo <= f(...) <= hi``. Copying a setpoint
into a fluent (``tool_pressure := injection_pressure``) stays transparent.

Each glass is picked up, primed, put into the tool, fitted with inserts, foamed, ejected and
trimmed. The tool is shared: ``put_in_foaming_tool`` needs it free, and ``eject`` frees it dirty,
unclamped and without mold release, so every glass re-cleans (and re-clamps) it, which needs the
tool empty. The tool keeps its temperature, so a type-B glass (inserts placed at 25 to 30 C) after
one foamed at 38 C or more needs ``cool_tool``. Every action adds its duration in seconds to the
global ``process_time``: control actions their duration control, symbolic ones a fixed time
(``SYMBOLIC_TIME``). The goal is every glass trimmed, its foam inside quality windows, and
``process_time <= budget``.

Time is in seconds. Short operations take their duration control in seconds (primer dwell,
clamping, insert seating, injection), thermal and chemical ones in minutes (heating, cooling,
flash-off, conditioning, insert preheating, curing), which their clock converts with a factor
of 60. Each opaque function computes in its controls' own units, so a lag's time constant is in
minutes when its duration is. The two elapsed-time fluents, ``flashoff_time`` and
``curing_time``, are in seconds.

- ``small``: primer, heating and foaming controls, inserts placed symbolically, a density window.
- ``main``: adds flash-off, clamping, insert placement controls, seeded insert types A/B/C with
  type-dependent flash-off and tool-temperature windows (C also needs mold release), and a
  chemical-concentration window.
- ``high_dep``: adds environment and insert conditioning, primer temperature, clamp vacuum,
  shot temperature and catalyst ratio, tool humidity, and cell-uniformity and skin-quality windows.

Deviations from the source, beyond the shared tool, ``eject``, ``cool_tool`` and the clock:

- Heating is a first-order lag towards ``20 + 60 * power`` (the source heats at zero power and
  sums power and duration), cooling one towards 15 C with a rate that grows with coolant flow.
  Lags use the rational factor ``d / (d + tau)`` instead of ``1 - exp(-d / tau)``, so the witness
  stays exact.
- ``condition_environment(g, air_temperature, conditioning_time)`` is a conditioning chamber for
  one glass: it moves that glass's temperature towards the chamber's air setpoint (the source heats
  it above the air), and inserts need 20 C instead of 30. The shop-floor climate,
  ``ambient_temperature`` and ``humidity_level``, is seeded and constant. The source's action writes
  it from its own setpoints (and has a humidity setpoint), so conditioning one glass would change
  the air every other glass's insert placement sees. In ``main``, which has no conditioning, the
  glass temperature is a seeded initial value.
- Curing happens after foaming: ``eject(g, cure_duration)`` needs ``curing_time >= 480`` seconds,
  faster on a hotter tool. The source adds curing in ``clamp_tool``, ``place_inserts`` and
  ``foaming``.
- Clamping builds tool pressure over its duration, ``pressure * d / (d + 1)``, instead of setting
  it, so ``clamp_duration`` matters.
- ``eject`` resets ``tool_alignment`` to 85, since it opens the clamp, so every glass aligns the tool
  again. Otherwise only insert placement and the clamp vacuum would move it, always upwards, and
  later glasses would be left with no correction to make.
- Mix and catalyst ratio are scored by their distance from 1: off-ratio foam expands less (is
  denser), leaves more unreacted residue and is less uniform. The source is monotone in both.
- Preheating inserts softens them, trading hardness against flexibility; the source raises both.
- Primer thickness grows with ``flow * dwell`` and falls slightly with spray pressure, the fresh
  film's viscosity is set by flow (and temperature), and flash-off thickens it towards 40 and
  cures adhesion towards 12, so primer and flash-off reach the insert thresholds from a bare glass.
- ``drying_time``, a second elapsed-time clock whose only check the dwell bound already implies,
  is dropped, as are the checks that cannot fail: foam density and chemical concentration before
  any foaming, the constant insert and glass thickness, and checks of a setpoint against its own
  type (application pressure, primer temperature, placement pressure). Fluents left unread go too,
  except ``curing_time`` and ``foam_expansion_rate``: nothing reads them, but the bounds on their
  applications constrain the controls.
- Quality levels are assigned once per glass instead of accumulated.

The variants have 7, 15 and 23 interpreted functions, of which 3, 5 and 8 are linear: the effects of
``apply_primer`` except the thickness, of insert placement and conditioning, the clamp's
alignment, and ``small``'s foam density. The rest are non-linear in four
ways:

- Lags (heating, cooling, flash-off viscosity and adhesion, glass conditioning, humidity) and the
  clamp's ``pressure * d / (d + 1)`` are rational in the duration.
- Primer thickness multiplies flow and dwell (and spray pressure), flash-off time duration and
  airflow, and ``curing_time`` the cure duration and the tool temperature, which is a fluent.
- Residue, expansion and the quality scores are quadratic in the mix and catalyst ratios, and
  ``cell_uniformity`` also has an ``abs`` kink.
- Foam density outside ``small`` is the bilinear shot over the quadratic expansion.

The domain parameter ``model: equation | mlp`` selects how the effects are computed. With ``mlp``
an effect that has a trained network in ``benchmarks.mlp_weights`` is computed by it instead of its
equation, in exact arithmetic (``benchmarks.mlp``); the witness calls the same body, so it stays
valid. Only ``small``'s ``foam_density_after_foaming`` has one: inputs ``tool_temperature``,
``injection_time`` and ``injection_pressure``, three hidden ReLU layers of 6, weights in multiples
of ``2**-16``, the output rounded to a multiple of ``2**-10``. It is trained by
``benchmarks.train_mlp`` on samples uniform over ``RANGES`` and the control bounds, which the bounds
preconditions keep every argument in. The network is the ground truth of the variant, not an
approximation checked against the equation. Other variants raise with ``mlp``.

The witness takes the glasses in a seeded order and draws each control from the middle 60 percent
of its range (the full range if that misses), redrawing until the step's target holds: a
temperature window, a threshold, the clamp pressure. The budget is the witness's time plus
``slack_pct`` percent, and each quality window is the witness's value plus or minus ``2 +
slack_pct`` percent of the fluent's range. With ``control: int`` a coarse grid can miss those
targets, so ``int_range`` must be at least ``MIN_INT_RANGE``.

``foaming_temporal_opaque`` (``FoamingTemporalGenerator``) is the same domain with durative actions,
so that what the plant does at the same time can overlap: one glass is primed or trimmed while
another is in the foaming tool. Each action lasts the time it adds to ``process_time`` above, and
``process_time`` goes: the goals must hold by a deadline, the witness's makespan plus ``slack_pct``
percent. Conditions are checked at an action's start, its effects happen at its end, and it holds
its resources in between, each a unary lock (``<resource>_free``):

- the glass, for every action on it except ``condition_inserts``, which works on its inserts;
- the station: ``apply_primer`` the primer station, ``trimming`` the trimming station,
  ``condition_environment`` the climate chamber, ``condition_inserts`` the insert oven, and the
  nine actions that act on the tool (from ``clean_tool`` to ``eject``) the foaming tool.

Picking up and flash-off occupy no station; flash-off dries on a rack. The locks keep every
fluent an opaque effect reads unchanged until the effect, so evaluating it at the end equals
evaluating it at the start. Deviations the planner forces on the temporal version:

- Nothing changes while waiting. A primed glass flashes off only during ``apply_primer`` and
  ``flash_off``, and an idle tool keeps its temperature, since OCTANE has no continuous change.
- The tool heater cannot run during a shot or during insert placement: ``heat_tool`` and
  ``cool_tool`` hold the tool like every other tool action.

The witness is the sequential one, retimed to earliest start times: each step starts ``EPS`` after
the earlier steps that share its glass or its station. The next glass is then picked up and primed
while the previous one flashes off or is in the tool.
"""

import random
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cache

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Integer
from unified_planning.model import Action, FNode, Object, Problem
from unified_planning.plans import ActionInstance, SequentialPlan, TimeTriggeredPlan
from unified_planning.shortcuts import (
    GE,
    LE,
    And,
    BoolType,
    DurativeAction,
    EndTiming,
    Fluent,
    GlobalStartTiming,
    InstantaneousAction,
    InterpretedFunction,
    Not,
    Or,
    RealType,
    StartTiming,
    Times,
    UserType,
)
from upbm.generator import Generator
from upbm.utils import MAX_INT

from benchmarks.bakery import _key
from benchmarks.common import EPS, Control, domain_parameter_space, exact
from benchmarks.mlp import MLP

F = Fraction
VARIANTS = ["small", "main", "high_dep"]
MODELS = ["equation", "mlp"]

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
MIN_INT_RANGE = 4  # coarser grids miss the witness targets of ``main``
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

# Temporal only: the station each action occupies. Each station is unary, as is every glass, which
# every glass action occupies except ``condition_inserts``: that one works on the glass's inserts.
TOOL_ACTIONS = [
    "clean_tool",
    "heat_tool",
    "cool_tool",
    "apply_mold_release",
    "clamp_tool",
    "put_in_foaming_tool",
    "place_inserts",
    "foaming",
    "eject",
]
STATIONS = {
    "apply_primer": "primer_station",
    "trimming": "trimming_station",
    "condition_environment": "climate_chamber",
    "condition_inserts": "insert_oven",
    **dict.fromkeys(TOOL_ACTIONS, "foaming_tool"),
}


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
    glass: bool = True
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
        glass=False,
        effects=heat,
    )
    specs["cool_tool"] = Spec(
        {"cool_duration": (1, 30), "coolant_flow": (0, 1)},
        "cool_duration",
        clock_unit=60,
        glass=False,
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
            clamp, "clamp_duration", glass=False, effects=clamp_effects
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


def _with_networks(specs: dict[str, Spec], variant: str) -> dict[str, Spec]:
    """``specs`` with every effect that has a trained network in ``NETWORKS`` computed by it."""
    # Imported here so that train_mlp, which imports this module, can regenerate a missing file.
    from benchmarks.mlp_weights import NETWORKS

    trained = {fn: net for (v, fn), net in NETWORKS.items() if v == variant}
    if not trained:
        raise ValueError(f"model: mlp has no trained network for variant {variant}")
    for action, spec in specs.items():
        for i, e in enumerate(spec.effects):
            net = trained.pop(f"{e.fluent}_after_{action}", None)
            if net is None:
                continue
            assert net["inputs"] == [*e.reads, *spec.controls], (action, e.fluent)
            mlp = MLP(
                net["layers"],
                [RANGES[r] for r in e.reads] + list(spec.controls.values()),
                RANGES[e.fluent],
            )
            spec.effects[i] = replace(e, body=_network_body(mlp, e, spec))
    if trained:
        raise ValueError(
            f"model: mlp has networks for no effect of {variant}: {sorted(trained)}"
        )
    return specs


def _network_body(mlp: MLP, e: Effect, spec: Spec) -> Callable[[dict, dict], Fraction]:
    def body(s, u):
        return mlp([*(s[r] for r in e.reads), *(u[c] for c in spec.controls)])

    return body


class FoamingGenerator(Generator):
    temporal = False

    def __init__(self, domain_params: Configuration):
        super().__init__(domain_params)
        self.variant = domain_params["variant"]
        self.model = domain_params["model"]
        self.specs = _specs(self.variant)
        if self.model == "mlp":
            self.specs = _with_networks(self.specs, self.variant)
        control, int_range = domain_params["control"], domain_params["int_range"]
        if control == "int" and int_range < MIN_INT_RANGE:
            raise ValueError(f"foaming needs int_range >= {MIN_INT_RANGE}")
        self.ctls = {
            name: {
                c: Control(lo, hi, control, int_range)
                for c, (lo, hi) in spec.controls.items()
            }
            for name, spec in self.specs.items()
        }

    @staticmethod
    def get_domain_parameter_space() -> ConfigurationSpace:
        space = domain_parameter_space()
        space.add(Categorical("variant", VARIANTS, default="main"))
        space.add(Categorical("model", MODELS, default="equation"))
        return space

    @property
    def instance_parameter_space(self) -> ConfigurationSpace:
        return ConfigurationSpace(
            name={
                "n_glasses": Integer("n_glasses", (1, MAX_INT), default=1),
                "slack_pct": Integer("slack_pct", (0, MAX_INT), default=20),
                "seed": Integer("seed", (0, MAX_INT), default=0),
            }
        )

    @property
    def name(self) -> str:
        d = self.domain_params
        kind = "temporal-opaque" if self.temporal else "opaque"
        model = ", mlp" if self.model == "mlp" else ""
        return f"Foaming-{kind} ({d['variant']}, {d['control']}{model})"

    @property
    def domain(self) -> Problem:
        return self.get_instance(
            self.instance_parameter_space.get_default_configuration()
        )

    @property
    def pddl_expressible(self) -> bool:
        return False

    @property
    def requires_per_instance_domain(self) -> bool:
        return True

    def get_instance(self, params: Configuration) -> Problem:
        return self._build(_key(params))[0]

    def witness_plan(self, params: Configuration) -> SequentialPlan | TimeTriggeredPlan:
        """A valid plan for the instance, planted by construction."""
        return self._build(_key(params))[1]

    @cache  # noqa: B019 -- generators are few and long-lived, and instances must be shared
    def _build(self, key: tuple) -> tuple[Problem, SequentialPlan | TimeTriggeredPlan]:
        p = dict(key)
        return _Instance(self, p).build()

    def get_objects(self, params: Configuration) -> list[Object]:
        return list(self.get_instance(params).all_objects)

    def get_goal(self, params: Configuration) -> list[FNode]:
        return list(self.get_instance(params).goals)

    def get_initial_state(self, params: Configuration) -> dict[FNode, FNode]:
        return dict(self.get_instance(params).explicit_initial_values)


class FoamingTemporalGenerator(FoamingGenerator):
    """``foaming_opaque`` with durative actions, so glasses overlap where the plant allows it."""

    temporal = True


class _Instance:
    """One instance: the UP problem, and the witness simulated on the same reference functions."""

    def __init__(self, gen: FoamingGenerator, p: dict):
        self.gen, self.p = gen, p
        self.variant = gen.variant
        self.small = self.variant == "small"
        self.high = self.variant == "high_dep"
        self.temporal = gen.temporal
        self.rng = random.Random(p["seed"])
        self.Glass = UserType("Glass")
        self.fluents: dict[str, Fluent] = {}
        self.actions: dict[str, Action] = {}
        self.locks: list[
            str
        ] = []  # temporal only: fluents true while the resource is free

    # -- the domain -------------------------------------------------------------------------------

    def fl(self, name: str, g=None) -> FNode:
        f = self.fluents[name]
        return f(g) if f.arity else f()

    def pre(self, a: InstantaneousAction | DurativeAction, cond):
        """A precondition, or a condition at the start of a durative action."""
        if isinstance(a, DurativeAction):
            a.add_condition(StartTiming(), cond)
        else:
            a.add_precondition(cond)

    def eff(self, a: InstantaneousAction | DurativeAction, fluent, value):
        """An effect, at the end of a durative action."""
        if isinstance(a, DurativeAction):
            a.add_effect(EndTiming(), fluent, value)
        else:
            a.add_effect(fluent, value)

    def _new(self, name: str, glass: bool, params: OrderedDict, seconds) -> tuple:
        """The action ``name``, taking ``seconds``, and its glass parameter if it has one.

        Sequentially it adds ``seconds`` to ``process_time``. Durative, it lasts ``seconds`` and
        holds its glass and station from start to end, so nothing changes what its effects read.
        """
        if glass:
            params = OrderedDict([("g", self.Glass), *params.items()])
        if not self.temporal:
            ia = InstantaneousAction(name, **params)
            g = ia.parameter("g") if glass else None
            ia.add_increase_effect(self.fl("process_time"), seconds(ia))
            self.actions[name] = ia
            return ia, g
        a = DurativeAction(name, **params)
        g = a.parameter("g") if glass else None
        a.set_fixed_duration(seconds(a))
        held = (
            [self.fl("glass_free", g)] if glass and name != "condition_inserts" else []
        )
        if name in STATIONS:
            held.append(self.fl(f"{STATIONS[name]}_free"))
        for lock in held:
            a.add_condition(StartTiming(), lock)
            a.add_effect(StartTiming(), lock, False)
            a.add_effect(EndTiming(), lock, True)
        self.actions[name] = a
        return a, g

    def _declare(self):
        glass_bool = [
            "in_rack",
            "centered",
            "primer_applied",
            "in_foaming_tool",
            "inserts_placed",
            "foamed",
            "trimmed",
        ]
        tool_bool = ["tool_clean", "tool_occupied"]
        glass_num = [
            "flashoff_time",
            "primer_thickness",
            "surface_adhesion",
            "foam_density",
            "curing_time",
        ]
        tool_num = ["tool_temperature"]
        global_num = [] if self.temporal else ["process_time"]
        if not self.small:
            glass_bool += ["insert_type_A", "insert_type_B", "insert_type_C"]
            tool_bool += ["tool_clamped", "mold_release_applied"]
            glass_num += [
                "primer_viscosity",
                "glass_temperature",
                "chemical_concentration",
            ]
            tool_num += ["tool_pressure", "tool_alignment"]
            global_num += ["humidity_level"]
        if self.high:
            glass_bool += ["environment_conditioned", "inserts_conditioned"]
            glass_num += [
                "insert_hardness",
                "insert_flexibility",
                "foam_expansion_rate",
                "cell_uniformity",
                "skin_quality",
            ]
            tool_num += ["tool_humidity"]
            global_num += ["ambient_temperature"]
        for n in glass_bool:
            self.fluents[n] = Fluent(n, BoolType(), g=self.Glass)
        for n in tool_bool:
            self.fluents[n] = Fluent(n, BoolType())
        for n in glass_num:
            self.fluents[n] = Fluent(n, RealType(), g=self.Glass)
        for n in tool_num + global_num:
            self.fluents[n] = Fluent(n, RealType())
        if self.temporal:
            self.fluents["glass_free"] = Fluent("glass_free", BoolType(), g=self.Glass)
            names = self.gen.specs.keys() | SYMBOLIC_TIME.keys()
            stations = sorted({STATIONS[n] for n in names if n in STATIONS})
            for n in stations:
                self.fluents[f"{n}_free"] = Fluent(f"{n}_free", BoolType())
            self.locks = ["glass_free", *(f"{n}_free" for n in stations)]

    def _symbolic(self, name: str, glass: bool = True):
        return self._new(name, glass, OrderedDict(), lambda a: SYMBOLIC_TIME[name])

    def _control(self, name: str):
        """The control action ``name`` with its opaque effects, their bounds and its clock."""
        spec, ctls = self.gen.specs[name], self.gen.ctls[name]
        a, g = self._new(
            name,
            spec.glass,
            OrderedDict((c, k.type) for c, k in ctls.items()),
            lambda a: Times(
                spec.clock_unit, ctls[spec.clock].expr(a.parameter(spec.clock))
            ),
        )
        apps = {}
        for e in spec.effects:
            fn = InterpretedFunction(
                f"{e.fluent}_after_{name}",
                RealType(),
                OrderedDict(
                    [(r, RealType()) for r in e.reads]
                    + [(c, k.type) for c, k in ctls.items()]
                ),
                _callable(e, ctls),
            )
            app = fn(*[self.fl(r, g) for r in e.reads], *[a.parameter(c) for c in ctls])
            lo, hi = RANGES[e.fluent]
            self.pre(a, LE(lo, app))
            self.pre(a, LE(app, hi))
            self.eff(a, self.fl(e.fluent, g), app)
            apps[e.fluent] = app
        for f, c in spec.copies:
            self.eff(a, self.fl(f, g), ctls[c].expr(a.parameter(c)))
        return a, g, apps

    def _domain(self):
        fl = self.fl
        a, g = self._symbolic("pickup_from_glassrack")
        self.pre(a, fl("in_rack", g))
        self.eff(a, fl("in_rack", g), False)
        self.eff(a, fl("centered", g), True)

        a, _ = self._symbolic("clean_tool", glass=False)
        self.pre(a, Not(fl("tool_occupied")))
        self.eff(a, fl("tool_clean"), True)

        a, g = self._symbolic("put_in_foaming_tool")
        self.pre(a, fl("primer_applied", g))
        self.pre(a, Not(fl("in_foaming_tool", g)))
        self.pre(a, Not(fl("foamed", g)))
        self.pre(a, Not(fl("tool_occupied")))
        self.eff(a, fl("in_foaming_tool", g), True)
        self.eff(a, fl("tool_occupied"), True)

        a, g = self._symbolic("trimming")
        self.pre(a, fl("foamed", g))
        self.pre(a, Not(fl("in_foaming_tool", g)))
        self.eff(a, fl("trimmed", g), True)

        a, g, _ = self._control("apply_primer")
        self.pre(a, fl("centered", g))
        self.pre(a, Not(fl("primer_applied", g)))
        self.eff(a, fl("primer_applied", g), True)

        for name in ["heat_tool", "cool_tool"]:
            a, _, _ = self._control(name)
            self.pre(a, fl("tool_clean"))

        if self.small:
            a, g = self._symbolic("place_inserts")
            lo_fo, (lo_t, hi_t) = SMALL_WINDOW
            self.pre(a, GE(fl("flashoff_time", g), lo_fo))
            self.pre(a, GE(fl("tool_temperature"), lo_t))
            self.pre(a, LE(fl("tool_temperature"), hi_t))
            self.pre(a, GE(fl("primer_thickness", g), F(1, 2)))
            self.pre(a, LE(fl("primer_thickness", g), 2))
            self.pre(a, GE(fl("surface_adhesion", g), F(1, 2)))
        else:
            a, g, _ = self._control("flash_off")
            self.pre(a, fl("primer_applied", g))

            a, _ = self._symbolic("apply_mold_release", glass=False)
            self.pre(a, fl("tool_occupied"))
            self.eff(a, fl("mold_release_applied"), True)

            a, _, _ = self._control("clamp_tool")
            self.pre(a, fl("tool_occupied"))
            self.pre(a, fl("tool_clean"))
            self.pre(a, Not(fl("tool_clamped")))
            self.eff(a, fl("tool_clamped"), True)

            a, g, apps = self._control("place_inserts")
            self.pre(a, GE(apps["tool_alignment"], 90))
            self.pre(a, GE(fl("humidity_level"), 20))
            self.pre(a, LE(fl("humidity_level"), 60))
            self.pre(a, GE(fl("glass_temperature", g), 20))
            self.pre(a, GE(fl("surface_adhesion", g), 5))
            self.pre(a, GE(fl("tool_pressure"), 70))
            self.pre(a, GE(fl("primer_viscosity", g), 15))
            self.pre(a, GE(fl("primer_thickness", g), 1))
            self.pre(a, LE(fl("primer_thickness", g), 3))
            types = ["A", "B", "C"]
            branches = []
            for t in types:
                lo_fo, (lo_t, hi_t) = INSERT_WINDOWS[t]
                conds = [
                    fl(f"insert_type_{o}", g)
                    if o == t
                    else Not(fl(f"insert_type_{o}", g))
                    for o in types
                ]
                if t == "C":
                    conds.append(fl("mold_release_applied"))
                conds += [
                    GE(fl("flashoff_time", g), lo_fo),
                    GE(fl("tool_temperature"), lo_t),
                    LE(fl("tool_temperature"), hi_t),
                ]
                branches.append(And(conds))
            self.pre(a, Or(branches))
            if self.high:
                self.pre(a, fl("environment_conditioned", g))
                self.pre(a, fl("inserts_conditioned", g))
                self.pre(a, LE(fl("ambient_temperature"), 25))
                self.pre(a, GE(fl("insert_hardness", g), 60))
                self.pre(a, GE(fl("insert_flexibility", g), 30))
                self.pre(a, LE(fl("tool_humidity"), 55))
            self.pre(a, fl("tool_clamped"))
        self.pre(a, fl("in_foaming_tool", g))
        self.pre(a, fl("tool_clean"))
        self.pre(a, Not(fl("inserts_placed", g)))
        self.eff(a, fl("inserts_placed", g), True)

        if self.high:
            a, g, _ = self._control("condition_environment")
            self.pre(a, Not(fl("environment_conditioned", g)))
            self.eff(a, fl("environment_conditioned", g), True)
            a, g, _ = self._control("condition_inserts")
            self.pre(a, Not(fl("inserts_conditioned", g)))
            self.eff(a, fl("inserts_conditioned", g), True)

        a, g, _ = self._control("foaming")
        self.pre(a, fl("inserts_placed", g))
        self.pre(a, Not(fl("foamed", g)))
        self.pre(a, GE(fl("tool_temperature"), FOAMING_MIN_T[self.variant]))
        self.eff(a, fl("foamed", g), True)

        a, g, apps = self._control("eject")
        self.pre(a, fl("foamed", g))
        self.pre(a, fl("in_foaming_tool", g))
        self.pre(a, GE(apps["curing_time"], MIN_CURE))
        self.eff(a, fl("in_foaming_tool", g), False)
        self.eff(a, fl("tool_occupied"), False)
        self.eff(a, fl("tool_clean"), False)
        if not self.small:
            self.eff(a, fl("tool_clamped"), False)
            self.eff(a, fl("tool_alignment"), UNCLAMPED_ALIGNMENT)
            self.eff(a, fl("mold_release_applied"), False)

    # -- the witness ------------------------------------------------------------------------------

    def _seed_state(self, glasses):
        rng = self.rng

        def draw(lo, hi):
            return exact(round(rng.uniform(lo, hi), 1))

        s = {("tool_temperature", None): draw(18, 25)}
        for g in glasses:
            for n in [
                "flashoff_time",
                "primer_thickness",
                "foam_density",
                "curing_time",
            ]:
                s[(n, g)] = F(0)
            s[("surface_adhesion", g)] = draw(0, F(1, 2))
            if not self.small:
                s[("primer_viscosity", g)] = F(0)
                s[("chemical_concentration", g)] = F(0)
                s[("glass_temperature", g)] = (
                    draw(10, 18) if self.high else draw(21, 35)
                )
            if self.high:
                s[("insert_hardness", g)] = draw(45, 55)
                s[("insert_flexibility", g)] = draw(10, 20)
                for n in ["foam_expansion_rate", "cell_uniformity", "skin_quality"]:
                    s[(n, g)] = F(0)
        if not self.small:
            s[("tool_pressure", None)] = F(0)
            s[("tool_alignment", None)] = draw(83, 88)
            s[("humidity_level", None)] = draw(25, 55)
        if self.high:
            s[("tool_humidity", None)] = draw(56, 70)
            s[("ambient_temperature", None)] = draw(15, 22)
        return s

    def _loc(self, name: str, g):
        return (name, g if self.fluents[name].arity else None)

    def _simulate(self, name: str, g, u: dict) -> dict | None:
        """The state after ``name`` with physical controls ``u``, or None if a bound fails."""
        spec = self.gen.specs[name]
        new = dict(self.state)
        for e in spec.effects:
            v = e.body({r: self.state[self._loc(r, g)] for r in e.reads}, u)
            lo, hi = RANGES[e.fluent]
            if not lo <= v <= hi:
                return None
            new[self._loc(e.fluent, g)] = v
        for f, c in spec.copies:
            new[(f, None)] = u[c]
        return new

    def _step(self, name: str, g=None, pred=lambda s, u: True) -> bool:
        """Append ``name`` with controls drawn until ``pred`` holds on the state it reaches."""
        spec, ctls = self.gen.specs[name], self.gen.ctls[name]
        for attempt in range(600):
            shrink = F(1, 5) if attempt < 300 else F(0)
            values, u = [], {}
            for c, k in ctls.items():
                lo, hi = k.lo + (k.hi - k.lo) * shrink, k.hi - (k.hi - k.lo) * shrink
                v = k.snap(round(self.rng.uniform(float(lo), float(hi)), 2))
                values.append(v)
                u[c] = k.exact(v)
            new = self._simulate(name, g, u)
            if new is not None and pred(lambda n, new=new: new[self._loc(n, g)], u):
                self.state = new
                args = ([g] if g is not None else []) + values
                ai = ActionInstance(self.actions[name], tuple(args))
                self.steps.append((ai, spec.clock_unit * u[spec.clock], name, g))
                return True
        return False

    def _must(self, name: str, g=None, pred=lambda s, u: True):
        if not self._step(name, g, pred):
            raise RuntimeError(f"foaming witness: no controls for {name} ({self.p})")

    def _symbolic_step(self, name: str, g=None):
        args = (g,) if g is not None else ()
        ai = ActionInstance(self.actions[name], args)
        self.steps.append((ai, F(SYMBOLIC_TIME[name]), name, g))

    def _tool_to(self, lo, hi):
        """Heat or cool the tool into ``[lo, hi]`` unless it already is."""
        t = self.state[("tool_temperature", None)]
        if lo <= t <= hi:
            return
        self._must(
            "heat_tool" if t < lo else "cool_tool",
            pred=lambda s, u: lo <= s("tool_temperature") <= hi,
        )

    def _glass(self, g, insert_type):
        sym = self._symbolic_step
        sym("pickup_from_glassrack", g)
        if self.small:
            min_fo, (lo_t, hi_t) = SMALL_WINDOW
            self._must(
                "apply_primer",
                g,
                lambda s, u: (
                    s("flashoff_time") >= min_fo
                    and F(1, 2) <= s("primer_thickness") <= 2
                    and s("surface_adhesion") >= F(1, 2)
                ),
            )
        else:
            min_fo, (lo_t, hi_t) = INSERT_WINDOWS[insert_type]
            if self.high:
                self._must(
                    "condition_environment",
                    g,
                    lambda s, u: s("glass_temperature") >= F(41, 2),
                )
                self._must(
                    "condition_inserts",
                    g,
                    lambda s, u: (
                        s("insert_hardness") >= 61 and s("insert_flexibility") >= 31
                    ),
                )
            self._must("apply_primer", g, lambda s, u: 1 <= s("primer_thickness") <= 3)

            def primed(s, u):
                return (
                    s("flashoff_time") >= min_fo
                    and s("primer_viscosity") >= 15
                    and s("surface_adhesion") >= 5
                )

            if not self._step("flash_off", g, primed):
                # One flash-off falls short; a first, then one more that completes it.
                self._must("flash_off", g, lambda s, u: s("flashoff_time") <= 4800)
                self._must("flash_off", g, primed)
        sym("clean_tool")
        sym("put_in_foaming_tool", g)
        if not self.small:
            self._must(
                "clamp_tool",
                pred=lambda s, u: s("tool_pressure") >= 71,
            )
            if insert_type == "C":
                sym("apply_mold_release")
            if self.high and self.state[("tool_humidity", None)] > 54:
                self._must(
                    "heat_tool",
                    pred=lambda s, u: s("tool_humidity") <= 54,
                )
        self._tool_to(lo_t + F(1, 2), hi_t - F(1, 2))
        if self.small:
            sym("place_inserts", g)
        else:
            self._must(
                "place_inserts",
                g,
                lambda s, u: s("tool_alignment") >= 91,
            )
        foam_min = FOAMING_MIN_T[self.variant]
        if self.state[("tool_temperature", None)] < foam_min:
            self._tool_to(foam_min + F(1, 2), 60)
        self._must("foaming", g)
        self._must("eject", g, lambda s, u: s("curing_time") >= MIN_CURE + F(1, 2))
        if not self.small:
            self.state[("tool_alignment", None)] = F(UNCLAMPED_ALIGNMENT)
        sym("trimming", g)

    def _schedule(self) -> tuple[TimeTriggeredPlan, Fraction]:
        """The witness at earliest start times, and its makespan.

        Each step starts ``EPS`` after the earlier ones it shares its glass or station with.

        Every fluent is read and written only by the actions of one glass or only by tool actions,
        so these are all its dependences, and steps that share neither commute. The witness also
        keeps ``condition_inserts`` after the earlier steps of its glass, which the domain does
        not ask for.
        """
        free: dict = {}  # glass or station -> when its last step so far ends, plus EPS
        plan: list[tuple[Fraction, ActionInstance, Fraction | None]] = []
        makespan = F(0)
        for ai, d, name, g in self.steps:
            held = ([g] if g is not None else []) + (
                [STATIONS[name]] if name in STATIONS else []
            )
            start = max((free[k] for k in held if k in free), default=F(0))
            plan.append((start, ai, d))
            makespan = max(makespan, start + d)
            for k in held:
                free[k] = start + d + EPS
        return TimeTriggeredPlan(plan), makespan

    def build(self) -> tuple[Problem, SequentialPlan | TimeTriggeredPlan]:
        p = self.p
        self._declare()
        self._domain()
        kind = "foaming_temporal" if self.temporal else "foaming"
        params = "_".join(f"{k}{v}" for k, v in sorted(p.items()))
        problem = Problem(f"{kind}_{self.variant}_{self.gen.model}_{params}")
        for n, f in self.fluents.items():
            default = (n in self.locks) if f.type.is_bool_type() else 0
            problem.add_fluent(f, default_initial_value=default)
        problem.add_actions(list(self.actions.values()))
        glasses = [Object(f"g{i}", self.Glass) for i in range(p["n_glasses"])]
        problem.add_objects(glasses)

        self.state = self._seed_state(glasses)
        for (n, g), v in self.state.items():
            problem.set_initial_value(self.fl(n, g), v)
        types = {g: self.rng.choice("ABC") for g in glasses}
        for g in glasses:
            problem.set_initial_value(self.fl("in_rack", g), True)
            if not self.small:
                problem.set_initial_value(self.fl(f"insert_type_{types[g]}", g), True)

        # (instance, seconds, action name, glass)
        self.steps: list[tuple[ActionInstance, Fraction, str, Object | None]] = []
        order = list(glasses)
        self.rng.shuffle(order)
        for g in order:
            self._glass(g, types[g])

        slack = F(p["slack_pct"], 100)
        if not self.temporal:
            budget = sum(d for _, d, _, _ in self.steps) * (1 + slack)
            problem.add_goal(LE(self.fl("process_time"), budget))
        for g in glasses:
            problem.add_goal(self.fl("trimmed", g))
            for n in QUALITY[self.variant]:
                lo, hi = RANGES[n]
                width = (hi - lo) * (F(2, 100) + slack)
                v = self.state[(n, g)]
                problem.add_goal(GE(self.fl(n, g), max(lo, v - width)))
                problem.add_goal(LE(self.fl(n, g), min(hi, v + width)))
        if not self.temporal:
            return problem, SequentialPlan([ai for ai, _, _, _ in self.steps])
        plan, makespan = self._schedule()
        goal = And(*problem.goals)
        problem.clear_goals()
        problem.add_timed_goal(GlobalStartTiming(makespan * (1 + slack) + EPS), goal)
        return problem, plan


def _callable(e: Effect, ctls: dict[str, Control]):
    """The Python callable behind an opaque effect: fluents first, then the physical controls."""
    n = len(e.reads)

    def call(*args):
        s = {r: F(v) for r, v in zip(e.reads, args[:n], strict=True)}
        u = {c: k.exact(v) for (c, k), v in zip(ctls.items(), args[n:], strict=True)}
        return e.body(s, u)

    return call
