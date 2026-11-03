"""Validated state and compact API contracts for the FEM workflow."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .runtime_files import current_environment

_LOG = logging.getLogger(__name__)

UnitSystem = Literal[
    "SI (m-kg-s)",
    "SI-mm (mm-tonne-s)",
    "US-inch (in-lbf-s)",
]
UNIT_SYSTEMS: tuple[UnitSystem, ...] = (
    "SI (m-kg-s)",
    "SI-mm (mm-tonne-s)",
    "US-inch (in-lbf-s)",
)
UNIT_SYSTEM_DETAILS: dict[UnitSystem, dict[str, str]] = {
    "SI (m-kg-s)": {
        "length": "m", "displacement": "m", "force": "N", "mass": "kg",
        "time": "s", "stress_modulus_traction": "Pa (N/m²)",
        "body_force": "N/m³", "acceleration": "m/s²", "density": "kg/m³", "energy": "J (N·m)",
    },
    "SI-mm (mm-tonne-s)": {
        "length": "mm", "displacement": "mm", "force": "N", "mass": "tonne",
        "time": "s", "stress_modulus_traction": "MPa (N/mm²)",
        "body_force": "N/mm³", "acceleration": "mm/s²", "density": "tonne/mm³", "energy": "mJ (N·mm)",
    },
    "US-inch (in-lbf-s)": {
        "length": "in", "displacement": "in", "force": "lbf",
        "mass": "lbf·s²/in", "time": "s",
        "stress_modulus_traction": "psi (lbf/in²)",
        "body_force": "lbf/in³", "acceleration": "in/s²", "density": "lbf·s²/in⁴",
        "energy": "in·lbf",
    },
}
_LEGACY_UNIT_SYSTEMS = {
    "SI (m, N, Pa)": "SI (m-kg-s)",
    "SI: m, N, Pa": "SI (m-kg-s)",
    "SI (mm, N, MPa)": "SI-mm (mm-tonne-s)",
    "US customary (in, lbf, psi)": "US-inch (in-lbf-s)",
}


def _current_unit_system(value):
    return _LEGACY_UNIT_SYSTEMS.get(value, value)


def write_text_atomically(path: Path, text: str) -> None:
    """Replace a file's contents in one step, never leaving it empty.

    ``Path.write_text`` truncates the file and only then writes it, so anyone
    who opens it in between reads nothing at all.  The UI polls these snapshots
    from the script thread -- once per rerun, for every session in the sidebar
    -- while the workflow rewrites the running one from a worker thread, and a
    long step was wide enough to be caught in that gap: the page went down with
    "Expecting value: line 1 column 1".  Writing beside the target and renaming
    is atomic on POSIX, so a reader gets the whole old file or the whole new one.
    """
    target = Path(path)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class Stage(str, Enum):
    NEW = "new"
    GEOMETRY_INTAKE = "geometry_intake"
    GEOMETRY_REVIEW = "geometry_review"
    GEOMETRY_GENERATION = "geometry_generation"
    GEOMETRY_PHYSICS = "geometry_physics"
    GEOMETRY_READY = "geometry_ready"
    BC_INTAKE = "bc_intake"
    BC_SEGMENTATION = "bc_segmentation"
    BC_REVIEW = "bc_review"
    BC_VALUES = "bc_values"
    MATERIAL_INTAKE = "material_intake"
    MATERIAL_INFERENCE = "material_inference"
    SHAPE_OPT_SETTINGS = "shape_opt_settings"
    CASE_READY = "case_ready"
    FORWARD_PENDING = "forward_pending"
    COMPLETE = "complete"
    FAILED = "failed"


class AssetRecord(BaseModel):
    source_name: str
    path: str
    sha256: str
    mime_type: str | None = None
    derived_from: str | None = None


class APIReceipt(BaseModel):
    purpose: Literal[
        "image_description", "geometry_refinement", "cad_geometry_refinement",
        "cadcoder_geometry_refinement",
        "physical_scale_inference",
        "bc_prompt_refinement", "bc_synthesis", "bc_value_inference",
        "material_interpretation", "part_materials",
    ]
    provider: str
    model: str
    cache_key: str
    cached: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ToolReceipt(BaseModel):
    tool: str
    arguments: dict[str, Any]
    returncode: int
    stdout_path: str
    stderr_path: str
    artifacts: list[str] = Field(default_factory=list)
    # sha256 of each artifact as this run left it.  Tools write to stable paths
    # (``case/case.json`` and friends), so a later run with different settings
    # overwrites them; without these digests a cache hit would hand back a
    # receipt whose artifacts on disk are somebody else's.  Empty on receipts
    # written before digests were recorded.
    artifact_digests: dict[str, str] = Field(default_factory=dict)


class GeometryRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refined_prompt: str = Field(min_length=1, max_length=600)


class CADDescription(BaseModel):
    """An expert-level construction description, longer than a visual prompt."""

    model_config = ConfigDict(extra="forbid")

    refined_prompt: str = Field(min_length=1, max_length=8000)


class BCPromptRefinement(BaseModel):
    """Revised supports-and-loads description, ready for selector synthesis."""

    model_config = ConfigDict(extra="forbid")

    # Room to name every segment a region spans (a handle is many CAD faces).
    refined_prompt: str = Field(min_length=1, max_length=1500)


# A strong text model need not be multimodal, so each role has its own default.
DEFAULT_TEXT_MODELS = {
    "deepseek": "deepseek-flash",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash",
}
DEFAULT_VISION_MODELS = {
    "deepseek": "deepseek-flash",
    "groq": "qwen/qwen3.8-27b",
    "gemini": "gemini-2.5-flash",
}


class ImageDescription(BaseModel):
    """What the vision stage saw, written for a text-only stage to act on."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=2400)


class PartMaterial(BaseModel):
    """Isotropic parameters for one part, in the workflow's units."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0)
    youngs_modulus: float = Field(gt=0)
    poisson_ratio: float = Field(gt=-1, lt=0.5)
    density: float = Field(gt=0)


class PartMaterials(BaseModel):
    """One named material per part of the surface."""

    model_config = ConfigDict(extra="forbid")

    parts: list[PartMaterial] = Field(min_length=1)

    @model_validator(mode="after")
    def one_material_per_part(self):
        ids = [part.id for part in self.parts]
        if len(ids) != len(set(ids)):
            raise ValueError("each part id must be named once")
        return self


class MaterialInference(BaseModel):
    """Isotropic parameters for the object; the material itself stays implicit."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["uniform", "vomp"] = "uniform"
    youngs_modulus: float | None = Field(default=None, gt=0)
    poisson_ratio: float | None = Field(default=None, gt=-1, lt=0.5)
    density: float | None = Field(default=None, gt=0)
    # Legacy inference caches may contain this field; the graph ignores it.
    yield_strength: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def complete_uniform_material(self):
        if self.mode == "uniform" and (
            self.youngs_modulus is None or self.poisson_ratio is None
        ):
            raise ValueError("uniform material inference requires E and Poisson ratio")
        if self.mode == "vomp" and any(
            value is not None for value in (
                self.youngs_modulus, self.poisson_ratio, self.density, self.yield_strength
            )
        ):
            raise ValueError("VoMP selection must not invent scalar material values")
        return self


class PhysicalScaleInference(BaseModel):
    """Reference axis and real-world size inferred from the geometry request.

    The axis is what the stated size is measured along, not which way is up.
    Named ``up_axis`` on the wire it invited the model to answer "up is y" and
    then hang the dimension on y, so a request for a metre in z scaled y.
    """

    model_config = ConfigDict(extra="forbid")

    reference_axis: Literal["x", "y", "z"]
    physical_extent: float = Field(gt=0)


SELECTOR_TYPES = {
    "bounding_box", "sphere", "cylinder", "half_space", "slab", "lowest",
    "highest", "percentile", "furthest_from_center", "normal",
    "aligned_normal", "curvature", "flat_region", "sharp_feature",
    "dihedral_angle", "surface_segment", "shape_diameter_segment", "flood_fill", "grow_selection",
    "preview_segments", "boundary_edge", "connected_component", "and", "or",
    "subtract", "not",
}

SELECTOR_PARAMETERS: dict[str, tuple[set[str], set[str]]] = {
    "bounding_box": ({"min_pt", "max_pt"}, {"min_pt", "max_pt"}),
    "sphere": ({"center", "radius"}, {"center", "radius"}),
    "cylinder": ({"p1", "p2", "radius"}, {"p1", "p2", "radius"}),
    "half_space": ({"point", "normal"}, {"point", "normal"}),
    "slab": ({"axis", "min_val", "max_val"}, {"axis", "min_val", "max_val"}),
    "lowest": ({"axis", "fraction"}, set()),
    "highest": ({"axis", "fraction"}, set()),
    "percentile": ({"axis", "lo_pct", "hi_pct"}, {"axis"}),
    "furthest_from_center": ({"fraction"}, set()),
    "normal": ({"target_vector", "max_angle_deg"}, {"target_vector", "max_angle_deg"}),
    "aligned_normal": ({"target_vector", "max_angle_deg", "oriented"}, {"target_vector", "max_angle_deg"}),
    "curvature": ({"min_kappa", "max_kappa", "curvature_type"}, {"min_kappa", "max_kappa"}),
    "flat_region": ({"max_curvature"}, set()),
    "sharp_feature": ({"min_curvature"}, set()),
    "dihedral_angle": ({"min_angle_deg", "max_angle_deg"}, set()),
    "surface_segment": ({"segment_ids"}, {"segment_ids"}),
    "shape_diameter_segment": ({"segment_ids", "num_clusters", "smoothing", "num_rays", "cone_angle_deg", "preserve_seed", "max_segment_area_fraction", "adjacent_segments", "adjacent_direction"}, set()),
    "preview_segments": ({"segment_ids"}, {"segment_ids"}),
    "flood_fill": ({"max_feature_angle_deg"}, set()),
    "grow_selection": ({"num_rings"}, set()),
    "boundary_edge": (set(), set()),
    "connected_component": ({"component_index", "sort_by"}, set()),
    "and": (set(), set()), "or": (set(), set()), "subtract": (set(), set()), "not": (set(), set()),
}


class SelectorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    children: list["SelectorSpec"] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def nest_long_unions(cls, value):
        """Read an ``and``/``or`` of three or more children as nested pairs.

        The selectors combine two at a time, but asked for "the ends of all
        four legs" the model writes one ``or`` with four children, and that
        answer is right: both are associative, so ``or(a, b, c, d)`` is
        ``or(a, or(b, or(c, d)))``.  Rejecting it failed a whole synthesis.
        """
        if not isinstance(value, dict) or value.get("type") not in ("and", "or"):
            return value
        children = value.get("children")
        if not isinstance(children, list) or len(children) <= 2:
            return value
        return {**value, "children": [
            children[0], {"type": value["type"], "params": {}, "children": children[1:]},
        ]}

    @model_validator(mode="after")
    def validate_shape(self):
        if self.type not in SELECTOR_TYPES:
            raise ValueError(f"unsupported selector type: {self.type}")
        if self.type == "slab":
            # Bounds the wrong way round name an empty band rather than an
            # inverted one, and the intent is unambiguous.
            low, high = self.params.get("min_val"), self.params.get("max_val")
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                if low > high:
                    self.params["min_val"], self.params["max_val"] = high, low
        allowed, required = SELECTOR_PARAMETERS[self.type]
        unknown = set(self.params) - allowed
        missing = required - set(self.params)
        if unknown:
            raise ValueError(f"unsupported parameters for {self.type}: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing parameters for {self.type}: {sorted(missing)}")
        arity = {
            "not": 1, "and": 2, "or": 2, "subtract": 2,
            "flood_fill": 1,
            "grow_selection": 1, "boundary_edge": 1,
        }.get(self.type)
        if self.type == "surface_segment":
            if self.children:
                raise ValueError("surface_segment does not accept children")
            segment_ids = self.params.get("segment_ids")
            if not isinstance(segment_ids, list) or not segment_ids:
                raise ValueError("surface_segment requires at least one segment id")
            return self
        if self.type in ("shape_diameter_segment", "preview_segments"):
            # One selector, two ways in: name reviewed segments by id, or seed a
            # fresh one. Naming ids reuses the segmentation already computed and
            # shown, so it needs no seed child.
            if "segment_ids" in self.params:
                if self.children:
                    raise ValueError(
                        "shape_diameter_segment takes segment_ids or one seed "
                        "child, not both"
                    )
            elif len(self.children) != 1:
                raise ValueError(
                    "shape_diameter_segment requires segment_ids or one seed child"
                )
            return self
        if self.type == "connected_component":
            if len(self.children) > 1:
                raise ValueError("connected_component accepts at most one child")
            return self
        if arity is None and self.children:
            raise ValueError(f"{self.type} cannot have children")
        if arity is not None and len(self.children) != arity:
            raise ValueError(f"{self.type} requires {arity} children")
        return self


class DirichletBC(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    selector: SelectorSpec
    value: float | tuple[float, float, float] = 0.0
    components: list[Literal[0, 1, 2]] | None = None

    @field_validator("components", mode="before")
    @classmethod
    def normalize_component_axes(cls, value):
        if value is None:
            return None
        axis = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
        return [axis.get(str(item).lower(), item) for item in value]


class NeumannBC(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    selector: SelectorSpec
    traction: tuple[float, float, float] | None = None


class BCProposal(BaseModel):
    dirichlet: list[DirichletBC] = Field(default_factory=list)
    neumann: list[NeumannBC] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonempty(self):
        # A held region is a well-posedness requirement: with nothing fixed the
        # stiffness matrix is singular and the body floats away on its rigid
        # modes.  A load is not.  Requiring one too was making every request
        # that named only supports -- "fix the bottom" -- illegal to answer, and
        # the model duly invented a traction nobody asked for to get past the
        # validator.  An unloaded case solves to zero displacement, which is the
        # honest answer to an unloaded question.
        if not self.dirichlet:
            raise ValueError("at least one Dirichlet BC is required")
        if len({item.name for item in self.dirichlet}) != len(self.dirichlet):
            raise ValueError("Dirichlet BC names must be unique")
        if len({item.name for item in self.neumann}) != len(self.neumann):
            raise ValueError("Neumann BC names must be unique")
        return self


class DirichletValueInput(BaseModel):
    components: list[Literal[0, 1, 2]] = Field(min_length=1)
    value: float = 0.0

    @field_validator("components")
    @classmethod
    def unique_components(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Dirichlet components must be unique")
        return sorted(value)


class BCValuesInput(BaseModel):
    dirichlet_values: dict[str, DirichletValueInput]
    neumann_values: dict[str, tuple[float, float, float]]
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)


class NeumannForce(BaseModel):
    """A total force on a loaded region, which the controller spreads over its area.

    Dividing by the area was left to the model, and it got the arithmetic wrong
    -- 98 N over 0.0012 m^2 came back as 82.1 Pa, not 82,100 -- so the model now
    states the force and the division is done here, exactly.
    """

    model_config = ConfigDict(extra="forbid")

    force: tuple[float, float, float]


NeumannEntry = tuple[float, float, float] | NeumannForce


class BCValueInference(BaseModel):
    """Tolerant API response; approved names are reconciled by the controller."""

    # Empty collections must be rejected, not tolerated: the controller falls back
    # to the current value for any region the response omits, so an empty answer
    # validates, caches, and silently leaves every value unchanged.
    dirichlet_values: (
        list[DirichletValueInput] | dict[str, DirichletValueInput]
    ) = Field(min_length=1)
    neumann_values: list[NeumannEntry] | dict[str, NeumannEntry] = Field(min_length=1)
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @field_validator("dirichlet_values", "neumann_values", mode="before")
    @classmethod
    def merge_named_entries(cls, value):
        """Accept one object per region as well as one object for all of them.

        Asked for entries keyed by region name, a model reasonably reads "one
        entry per region" as a list of one-key objects.  The values in it are
        named and correct; only the wrapping differs, and rejecting it threw
        away a right answer.  A positional list is left alone -- a Dirichlet
        entry is itself an object, so it is told apart by carrying its own
        fields rather than a region's name.
        """
        if not isinstance(value, list) or not value:
            return value
        if not all(isinstance(item, dict) for item in value):
            return value
        if any({"components", "value", "force"} & set(item) for item in value):
            return value
        merged = {}
        for item in value:
            merged.update(item)
        return merged or value


class UnloadedBCValueInference(BCValueInference):
    """The same answer, for a case that has no loaded region at all.

    A proposal may hold no Neumann region -- "fix the bottom" names a support
    and no load -- and then an empty ``neumann_values`` is not the evasion the
    guard above is for, but the only answer there is.  Asked under the loaded
    schema it was rejected, repaired, and rejected again, which is the model
    being told to invent a traction for a region that does not exist.

    Dirichlet stays required: a proposal always holds at least one held region,
    because a body with nothing held has no solution.
    """

    neumann_values: list[NeumannEntry] | dict[str, NeumannEntry] = Field(
        default_factory=dict
    )


# Material modes that read a spatial field artifact rather than scalar E and nu:
# VoMP's inferred field, or one made from a material per part.
FIELD_MODES = ("vomp", "parts")


class MaterialParameters(BaseModel):
    unit_system: UnitSystem
    mode: Literal["uniform", "vomp", "parts"] = "uniform"
    youngs_modulus: float | None = Field(default=None, gt=0)
    poisson_ratio: float | None = Field(default=None, gt=-1, lt=0.5)
    density: float | None = Field(default=None, gt=0)
    yield_strength: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    artifact: str | None = None
    interpolation: Literal["nearest"] = "nearest"
    approved: bool = False
    # Kept only so version-1 material.json files continue to validate. New case
    # files carry loads separately.
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @field_validator("unit_system", mode="before")
    @classmethod
    def migrate_unit_system(cls, value):
        return _current_unit_system(value)

    @model_validator(mode="after")
    def validate_mode(self):
        if self.mode == "uniform":
            if self.youngs_modulus is None or self.poisson_ratio is None:
                raise ValueError("uniform material requires E and Poisson ratio")
            if self.artifact is not None:
                raise ValueError("uniform material cannot reference a material field")
        else:
            if not self.artifact or not self.approved:
                raise ValueError("a spatial material requires an approved field artifact")
            if self.youngs_modulus is not None or self.poisson_ratio is not None:
                raise ValueError("a spatial material uses its field artifact, not scalar E/nu")
        return self


# The BC preservation methods the workflow offers (the settings card lists
# exactly these).  The solver core also has "graft", which the workflow retired.
BC_FIX_METHODS: tuple[str, ...] = ("zip", "mask")
# The settings card's linear-solver choices, first the default, and the label
# each is shown with.  run_forward maps them to the solver's cg_backend_assembled.
LINEAR_SOLVERS: tuple[str, ...] = ("jaxamg", "pyamgx")
LINEAR_SOLVER_LABELS: dict[str, str] = {"jaxamg": "JAX-AMG (default)", "pyamgx": "pyamgx"}


# The marker optimization_failure_advice() puts in a message, so a caller that
# might append it a second time (run_forward's error, then the graph's) can see
# that it is already there.
OPTIMIZATION_FAILURE_ADVICE_MARKER = "Recommended next:"
# The recommended learning rate moves the surface by at most this fraction of
# the phase-field width epsilon per step.  Half the diffuse band is a step the
# band can follow: the interface stays inside the elements that resolved it
# the iteration before.  (The geometry clean-up sized its first move to 0.5
# h_min, which at the default epsilon = h/2 is a whole epsilon: twice this.)
OPTIMIZATION_STEP_FRACTION = 0.5


def _finite_positive(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def optimization_step_record(gd_lr: float, steps: list[dict] | None,
                             epsilon: float | None, h_min: float | None,
                             fraction: float = OPTIMIZATION_STEP_FRACTION) -> dict:
    """How large a failed optimization's steps were, and the learning rate that
    would have kept each under ``fraction`` of epsilon.

    ``steps`` is the solver's per-iteration record (``_optimize_steps``: the
    largest |dJ/ds| over the free design nodes, and gd_lr times it, the largest
    nodal SDF move).  The recommendation is read from the last step taken, the
    one that moved the design onto the shape that failed:
    gd_lr_new = fraction * epsilon / max|dJ/ds|.  None of the derived numbers
    is given when no step was taken or the gradient was not finite.
    """
    steps = [dict(step) for step in (steps or []) if isinstance(step, dict)]
    record = {"gd_lr": float(gd_lr), "phase_field_epsilon": epsilon, "h_min": h_min,
              "step_fraction_of_epsilon": fraction, "steps": steps,
              "last_step": None, "recommended_gd_lr": None}
    last = steps[-1] if steps else None
    gradient = _finite_positive(last.get("max_abs_grad")) if last else None
    eps = _finite_positive(epsilon)
    if gradient is None or eps is None:
        return record
    moves = [m for m in (_finite_positive(step.get("max_move")) for step in steps) if m]
    record["last_step"] = {
        "iteration": last.get("iteration"), "max_abs_grad": gradient,
        "max_move": float(gd_lr) * gradient,
        "max_move_over_epsilon": float(gd_lr) * gradient / eps,
    }
    record["largest_move_over_epsilon"] = max(moves) / eps if moves else None
    record["recommended_gd_lr"] = fraction * eps / gradient
    return record


def optimization_failure_advice(gd_lr: float, record: dict | None = None) -> str:
    """What to try after a shape optimization fails, for its learning rate.

    A forward solve of the same setup says whether the case itself solves;
    one that does points at the step size, since an SDF step too large for the
    diffuse interface tears the design apart within a few iterations.  With
    ``record`` (optimization_step_record) holding a step, the smaller rate is
    sized to epsilon; without one (it failed before its first backward pass)
    the advice falls back to a tenth.  Only for a failed optimization: a failed
    forward solve has no learning rate to reduce.
    """
    rate = float(gd_lr)
    head = (f"{OPTIMIZATION_FAILURE_ADVICE_MARKER} first run a forward solve with the same "
            "setup (run mode: forward) to check that the case solves; if it does, retry the "
            "optimization with a smaller learning rate")
    last = (record or {}).get("last_step")
    recommended = (record or {}).get("recommended_gd_lr")
    if not last or not recommended:
        return f"{head} (gd_lr is {rate:g}; try {rate / 10:g})."
    eps = float(record["phase_field_epsilon"])
    h_min = _finite_positive(record.get("h_min"))
    fraction = float(record.get("step_fraction_of_epsilon", OPTIMIZATION_STEP_FRACTION))
    widths = f"epsilon = {eps:.3g}" + (f", h_min = {h_min:.3g}" if h_min else "")
    auto = record.get("gd_lr_auto") if isinstance(record.get("gd_lr_auto"), dict) else None
    if auto and auto.get("applied"):
        # gd_lr_auto sized every step to fraction * epsilon at its largest node,
        # so the step's only knob is the fraction: halve it.
        auto_fraction = float(auto["fraction"])
        return (f"{head}: gd_lr_auto moved the surface by up to {auto_fraction:g} epsilon "
                f"({widths}) at every step, the last (iteration {last['iteration']}) at "
                f"gd_lr {rate:.2g}; if the forward solve points at the step, lower "
                f"gd_lr_auto_fraction to {auto_fraction / 2:g}.")
    if recommended >= rate:
        # Steps already inside the rule: a smaller rate is not what it needs.
        return (f"{head} only if the forward solve points at the step: the last step "
                f"(iteration {last['iteration']}) moved the surface by up to "
                f"{last['max_move_over_epsilon']:.2g} epsilon ({widths}) at gd_lr {rate:g}, "
                f"already under {fraction:g} epsilon, so the step size is an unlikely cause.")
    # gd_lr_auto applies the same rule by itself, at every update.
    return (f"{head}: the last step (iteration {last['iteration']}) moved the surface by up "
            f"to {last['max_move_over_epsilon']:.2g} epsilon ({widths}) at gd_lr {rate:g}; "
            f"gd_lr = {recommended:.2g} keeps each step under {fraction:g} epsilon "
            "(or turn on gd_lr_auto, which sizes every step to a fraction of epsilon).")


class ShapeOptSettings(BaseModel):
    """Editable ShapeOptTrimmed settings with the examples' optimization preset."""

    preset_version: Literal[2] = 2
    run_mode: Literal["forward", "shape_optimization"] = "forward"
    relative_tol: float = Field(default=1.0e-2, gt=0)
    absolute_tol: float = Field(default=1.0e-5, gt=0)
    max_newton_iters: int = Field(default=20, ge=1, le=500)
    # A forward solve ramps its load over these steps.  A shape optimization
    # ramps its first solve only; every later iteration starts from the last
    # one's displacement, at full load.
    load_steps: int = Field(default=1, ge=1, le=100)
    remesh_init_octree_depth: int = Field(default=4, ge=2, le=9)
    remesh_octree_depth: int = Field(default=7, ge=3, le=10)
    # No quadrature degree: the Nitsche path integrates with 2x2x2 Gauss points
    # and adaptive subcells, and never read the one this card used to offer.
    # Older settings that carry one still load; it is ignored, as it always was.
    # No BC box size either: selectors cull with one box around the selected
    # patch and decide with a BVH distance test, so there is nothing to size.
    # Older settings that carry one still load; it is ignored.
    phase_field_epsilon_factor: float = Field(default=0.5, gt=0, le=4)
    line_search_enabled: bool = False
    # float64 assembly and solve by default.  The float32 path stalls at a
    # residual floor once the strain falls below about 10 * 1.19e-7 / relative_tol
    # (graph.py, FLOAT32_RESIDUAL_FACTOR), which stiff materials under everyday
    # loads routinely do; the operator's runner already sent this on for every
    # solve.  The solver core's own default (core/default_config.py) stays float32:
    # run_forward always passes this field, so the core default never applies here.
    use_mixed_precision: bool = True
    # Rescale the assembled system to a unit diagonal before the Krylov solve, so
    # a Nitsche penalty thousands of times larger than the elastic terms does not
    # dominate the conditioning.  Exact, not an approximation, but off by default:
    # it helps a stiff case whose strain single precision can still resolve, and
    # does nothing for one below that limit.
    scale_system_to_unit_diagonal: bool = False
    # Nitsche penalty ceiling, as a multiple of Young's modulus.  Sliver boundary
    # elements otherwise drive beta thousands of times above E, and the condition
    # number follows.  None leaves it uncapped.
    beta_max_over_e: float | None = Field(default=None, gt=0)
    # The assembled system's AMG backend.  "jaxamg" is the solve this workflow
    # has always run (cg_backend_assembled="amgx": JAX-AMG, or pyamgx only when
    # JAX-AMG cannot be imported), so settings stored without the key keep it.
    # "pyamgx" runs the same AMGX AMG but keeps its last iterate whatever AMGX
    # reports, so Newton's residual check is what validates the solve; it helps
    # an ill-conditioned case where JAX-AMG raises at Newton iteration 0.
    linear_solver: Literal["jaxamg", "pyamgx"] = "jaxamg"
    # How BC patches survive a design update.  "mask" freezes the SDF on the
    # BC region; "zip" (the exact iteration-0 patch stitched back every remesh)
    # is the alternative.  The solver core's "graft" (snap BC vertices to the
    # initial surface) is not offered by the workflow: a stored one is read as
    # "mask" by ``retire_graft`` below.
    sdf_bc_fix_method: Literal["zip", "mask"] = "mask"
    # The SDF gradient step, in units of length^2; the solver core's own default
    # is not read, since run_forward always passes this field.
    gd_lr: float = Field(default=0.01, gt=0)
    # Size gd_lr automatically instead (off by default): every update steps at
    # gd_lr = gd_lr_auto_fraction * epsilon / max|dJ/ds| over the free design
    # nodes at that iteration, the same rule the failure advice sizes its
    # recommendation by.  With it on, gd_lr above is not used for the SDF step
    # (only kept when the first gradient is zero).  The fraction is every step's
    # largest SDF move in units of epsilon: normalized gradient descent, whose
    # steps do not shrink near an optimum, so the run ends at max_optimize_iters
    # rather than at a relative tolerance (core default_config).
    gd_lr_auto: bool = False
    gd_lr_auto_fraction: float = Field(default=OPTIMIZATION_STEP_FRACTION, gt=0, le=2)
    max_optimize_iters: int = Field(default=101, ge=1, le=100000)
    output_per_optimize_step: int = Field(default=1, ge=0)
    optimize_relative_tol: float = Field(default=1.0e-9, gt=0)
    # A cap at this fraction of the STARTING volume.  Above 1 the shape may grow up
    # to it (the solver normalises by |1 - f|, so the hinge stays an upper limit).
    target_volume_fraction: float = Field(default=0.5, gt=0, le=2)
    # Penalise growth from wherever the shape started instead of driving it to
    # a fraction of its bounding volume.  The two are alternatives, not a pair:
    # with this on, ``target_volume_fraction`` is not read at all.
    volume_as_is: bool = False
    shape_opt_w_vol: float = Field(default=5.0, ge=0)
    reinit_sdf: bool = True
    reinit_every: int = Field(default=1, ge=1)
    # No SDF reinitialization iteration count: it is the Sussman reinit's, and
    # run_forward always selects the marching-cubes reinit on a remeshed octree,
    # which never reads it.  Older settings that carry one still load; it is
    # ignored, as it always was.

    @model_validator(mode="before")
    @classmethod
    def migrate_old_workflow_defaults(cls, value):
        """Update serialized pre-capture defaults while preserving custom values."""
        if not isinstance(value, dict) or "preset_version" in value:
            return value
        migrated = dict(value)
        replacements = {
            "relative_tol": (1.0e-6, 1.0e-2),
            "absolute_tol": (1.0e-8, 1.0e-5),
            "max_newton_iters": (50, 20),
            # Mapped to the current defaults (gd_lr 0.01, float64), which are
            # what an untouched pre-capture card should come up with now.
            "use_mixed_precision": (True, True),
            "gd_lr": (1.0, 0.01),
            "max_optimize_iters": (2000, 101),
            "output_per_optimize_step": (10, 1),
            "optimize_relative_tol": (2.0e-3, 1.0e-9),
        }
        matching_defaults = [
            key for key, (old, _) in replacements.items()
            if key in migrated and migrated[key] == old
        ]
        if len(matching_defaults) >= 4:
            for key in matching_defaults:
                migrated[key] = replacements[key][1]
        migrated["preset_version"] = 2
        return migrated

    @field_validator("sdf_bc_fix_method", mode="before")
    @classmethod
    def retire_graft(cls, value):
        """Read a stored "graft" as the workflow default "mask".

        Migration (2026-10): "graft" was dropped from the workflow's choices
        (it stays in the solver core for examples and figures).  Sessions,
        operator plans and case.json files written before then may carry it,
        and they must still load and run, so the value is mapped here rather
        than rejected.  New input cannot reach it from the UI, which no longer
        lists it, and an answer or plan that still names it gets the same
        mapping, with a logged note, instead of a failed card.
        """
        if isinstance(value, str) and value.strip().lower() == "graft":
            _LOG.info(
                'sdf_bc_fix_method "graft" is no longer a workflow option; '
                'using "mask" instead'
            )
            return "mask"
        return value

    @model_validator(mode="after")
    def valid_octree_depths(self):
        if self.remesh_init_octree_depth > self.remesh_octree_depth:
            raise ValueError("initial octree depth cannot exceed final octree depth")
        return self


class GeometryGenerationSettings(BaseModel):
    """Generator choice and controls, using the reviewed UI's practical defaults.

    The CAD generators (text only) read ``seed`` and ``faces``; Hunyuan3D-2.1
    (image only) reads those, ``inference_steps`` and its own two; the rest are
    TripoSG's.
    """

    generator: Literal["triposg", "text2cad", "cadcoder", "hunyuan3d"] = "triposg"
    faces: int = -1
    # TripoSG's hierarchical extraction: the coarse grid the surface is first found on, then the
    # finest it is refined to.  A coarse first grid misses whole thin parts (a slat, a leg, a rim
    # it never sampled are never refined), a coarse final one loses thin detail.  4 / 7: 6 / 8
    # gave about 4x the faces (744k against 191k), and 72 s more per run in segmentation,
    # painting and the operator's checks (2026-10-09).
    init_octree_depth: int = Field(default=4, ge=1, le=8)
    octree_depth: int = Field(default=7, ge=2, le=9)
    seed: int = Field(default=42, ge=0, le=100)
    inference_steps: int = Field(default=16, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=1, le=15)
    prompt_confidence: float = Field(default=1.0, ge=0, le=1)
    scribble_confidence: float = Field(default=0.4, ge=0, le=1)
    # TripoSG keeps one body of what it generates: its largest-component filter
    # erodes the interior by this many voxels, keeps one connected core and
    # grows it back.  Erosion parts pieces joined only by something thinner than
    # twice the radius, so a positive radius drops thin members -- the slats of
    # a bench, the spindles of a chair -- with everything that hangs from them.
    # 0 keeps every part joined to the body; floating fragments still go.
    erosion_radius: int = Field(default=0, ge=0, le=8)
    # Hunyuan3D-2.1: marching-cubes grid of the shape stage, whether the paint
    # stage colours the surface, and three speed/quality switches -- upstream's
    # faster FlashVDM shape decoding, the baked texture size, and RealESRGAN
    # upscaling of the painted views.  The surface keeps one colour per vertex,
    # so a larger texture or upscaling barely changes what the workflow sees.
    hunyuan3d_octree_resolution: Literal[256, 384] = 384
    texture: bool = True
    hunyuan3d_flashvdm: bool = True
    hunyuan3d_texture_size: Literal[1024, 2048, 4096] = 1024
    hunyuan3d_upscale: bool = False

    @field_validator("faces")
    @classmethod
    def faces_are_finite_or_unlimited(cls, value: int):
        # Up to 300k: a TripoSG body extracted at depth 8 or 9 has 0.7-1.4 M faces, which ran PartField
        # out of GPU memory three times in one run (2026-10-11); capped there it keeps its thin parts.
        if value != -1 and not 1000 <= value <= 300000:
            raise ValueError("faces must be -1 (unlimited) or between 1000 and 300000")
        return value

    @model_validator(mode="after")
    def depths_are_ordered(self):
        if self.init_octree_depth > self.octree_depth:
            raise ValueError("initial octree depth cannot exceed final octree depth")
        return self


class SegmentationSettings(BaseModel):
    method: Literal["shape_diameter", "partfield", "cad_faces"] = "shape_diameter"
    num_clusters: int = Field(default=5, ge=2, le=20)
    num_rays: int = Field(default=25, ge=1, le=100)
    smoothing: float = Field(default=0.25, ge=0, le=1)
    cone_angle_deg: float = Field(default=120.0, gt=0, le=180)


class WorkflowState(BaseModel):
    version: int = 1
    session_id: str
    session_dir: str
    stage: Stage = Stage.NEW
    provider: str = "deepseek"
    # The image role may run on another service entirely; unset means it shares
    # the text role's.
    vision_provider: str | None = None
    model: str | None = None
    vision_model: str | None = None
    compute_env: str = "mechfoundry-warp117"
    octree_depth: int = Field(default=7, ge=4, le=10)
    segmentation_num_clusters: int = Field(default=5, ge=2, le=20)
    segmentation_method: Literal["shape_diameter", "partfield", "cad_faces"] = "shape_diameter"
    segmentation_num_rays: int = Field(default=25, ge=1)
    segmentation_smoothing: float = Field(default=0.25, ge=0, le=1)
    segmentation_cone_angle_deg: float = Field(default=120.0, gt=0, le=180)
    geometry_description: str | None = None
    geometry_mode: Literal["text", "image", "scribble", "text_scribble", "surface"] | None = None
    refined_prompt: str | None = None
    # What the mesh on disk was actually built from.  `refined_prompt` follows
    # the box the user is typing in and is overwritten the moment they submit a
    # revision, so it stops describing the geometry that exists; this does not
    # move until a new one is generated, which is what lets every stage that
    # looks at the render agree on what it was asked to be.
    generation_prompt: str = ""
    # The last description written by a CAD generator's own revision, and for
    # which generator.  Text2CAD and CAD-Coder do not know object names, so a
    # prompt only works for them once revised; comparing the box against this
    # is how the card tells a revised prompt from one typed since.
    cad_revision_generator: str | None = None
    cad_revision_prompt: str | None = None
    # What the vision stage made of the six-view render, read once as soon as
    # there is geometry to read.  Scaling, material and a later revision all
    # work from this one reading rather than each asking for their own.
    geometry_overview: str = ""
    generation: GeometryGenerationSettings = Field(default_factory=GeometryGenerationSettings)
    input_asset: AssetRecord | None = None
    # The newest image SDXL drew for the geometry card, cut out from its
    # background, the drawing as drawn, and the description and seed it was
    # drawn from.  It becomes the input only when an answer picks it, like an
    # upload.
    drawn_image: str | None = None
    drawn_image_raw: str | None = None
    drawn_image_prompt: str = ""
    drawn_image_seed: int = 0
    # What of the drawing's prompt SDXL never read (its text encoder takes 77 tokens) and whether
    # that reached into the description itself rather than only the appended style.
    drawn_image_dropped: str = ""
    drawn_image_description_cut: bool = False
    # What the geometry card held when the image was drawn -- the input kind,
    # the generator and its settings -- so it comes back as it was, with the
    # drawing beside it.
    geometry_draft: dict[str, Any] = Field(default_factory=dict)
    surface_path: str | None = None
    # The B-rep face of each triangle, when the surface came from a CAD model.
    # Scaling keeps the triangle order, so it labels the scaled surface too.
    cad_faces_path: str | None = None
    up_axis: Literal["x", "y", "z"] | None = None
    physical_extent: float | None = Field(default=None, gt=0)
    unit_system: UnitSystem | None = None
    scale_description: str = ""
    material_description: str = ""
    material_inferred: bool = False
    bc_description: str | None = None
    bc_value_description: str = ""
    segmentation_metadata_path: str | None = None
    segmentation_montage_path: str | None = None
    # What the vision stage made of the segment montage, read once the preview
    # renders so the card can show it and the later synthesis reuses the cache.
    segment_description: str = ""
    # The sha256 of each file under inputs/, geometry/, boundary_conditions/,
    # materials/ and case/ as this branch left it, by path in the session.
    # They sit at fixed paths, so every branch of a rewound session overwrote
    # the others'; entering a node puts this branch's back from versions/.
    # None on a checkpoint recorded before files were tracked: left as found.
    session_files: dict[str, str] | None = None
    bc_proposal: BCProposal | None = None
    # What the synthesis call returned and the wording it was made from, kept
    # even when it fails to resolve: the one proposal a revision may edit.  A
    # revision changes selectors only, so the regions stay the synthesized ones.
    bc_synthesized: BCProposal | None = None
    bc_synthesized_description: str | None = None
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: MaterialParameters | None = None
    material_artifact: str | None = None
    # The per-part field: the parts' values as the part stage (or the user)
    # gave them, in the workflow's units, and the artifact made from them.
    material_parts: list[dict[str, Any]] | None = None
    parts_artifact: str | None = None
    parts_montage: str | None = None
    # The content digest of the surface the parts were valued on: the surface
    # keeps its path (geometry/physical.glb) when it is regenerated or
    # rescaled, so only its contents tell the parts are still its own.
    parts_surface_digest: str | None = None
    # The content digest of the part labels (segments.npz) they were valued on:
    # re-segmenting the same surface rewrites the labels at the same path, and
    # values for the old parts must not be shown or approved for the new ones.
    parts_segments_digest: str | None = None
    # A coloured copy of the analysis surface for VoMP, painted by Hunyuan3D-2.1
    # from the prompt or an uploaded reference image; same vertices, plus
    # colour.  ``textured_from`` is the surface it was painted from, and
    # ``textured_from_digest`` its contents, so a rescaled or replaced surface
    # is never paired with a stale painting, even at the same path.
    texture_prompt: str = ""
    texture_seed: int = 0
    texture_reference: str | None = None
    textured_surface: str | None = None
    textured_from: str | None = None
    textured_from_digest: str | None = None
    # Set when the painter crashed on the surface (graph._paint_surface): the painting was made
    # with another seed ("reseeded"), or the surface was given one flat colour read from the
    # words ("plain_colour"); with the seeds that crashed and a note to report.
    texture_fallback: dict[str, Any] | None = None
    # Material inference chose VoMP for a bare surface, which must be painted
    # first; the card then opens in VoMP mode at that step.
    vomp_suggested: bool = False
    shape_opt: ShapeOptSettings = Field(default_factory=ShapeOptSettings)
    case_path: str | None = None
    api_error: str | None = None
    api_error_source: str | None = None
    api_receipts: list[APIReceipt] = Field(default_factory=list)
    tool_receipts: list[ToolReceipt] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("unit_system", mode="before")
    @classmethod
    def migrate_unit_system(cls, value):
        return _current_unit_system(value)

    @field_validator("compute_env", mode="before")
    @classmethod
    def migrate_compute_env(cls, value):
        # A checkpoint from before the MechFoundry rename names an ``implicit-*``
        # environment; once that is uninstalled the session runs in its renamed one.
        return current_environment(value) if isinstance(value, str) else value

    @property
    def root(self) -> Path:
        return Path(self.session_dir)
