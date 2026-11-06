import hashlib
import json
import shutil
from datetime import datetime
import time
import signal
import math
import sys
from types import SimpleNamespace
import os
from pathlib import Path

import pytest
from PIL import Image

from src.agents_wrapper.fem_workflow.models import (
    BCProposal,
    BCValueInference,
    BCValuesInput,
    GeometryGenerationSettings,
    GeometryRefinement,
    MaterialParameters,
    PartMaterials,
    SegmentationSettings,
    SelectorSpec,
    ShapeOptSettings,
    UNIT_SYSTEM_DETAILS,
)
from src.agents_wrapper.fem_workflow.monitor import ToolMonitor, ToolRejected
from src.agents_wrapper.fem_workflow.scripts import workflow_tool
from src.agents_wrapper.fem_workflow import run_workflow
from src.agents_wrapper.fem_workflow.specialists import SpecialistCalls
from src.ui.controller import (
    WORKER_RELAUNCH_BACKOFF,
    WorkflowController, WorkflowEvent, faces_slider_options, interrupt_marker, restored_interrupt,
    segmentation_method_choices,
)
from src.agents_wrapper.fem_workflow.assets import (
    field_asset_request, load_step_surfaces, plan_solution_prefetch,
)
from src.agents_wrapper import GenerationResponse, TokenUsage
from src.agents_wrapper import APIResponseError


def test_geometry_refinement_requires_exactly_one_result():
    assert GeometryRefinement(refined_prompt="bracket").refined_prompt == "bracket"
    detailed = "x" * 500
    assert GeometryRefinement(refined_prompt=detailed).refined_prompt == detailed
    with pytest.raises(ValueError):
        GeometryRefinement()
    with pytest.raises(ValueError):
        GeometryRefinement(refined_prompt="a", clarification="b")


def test_ui_prefers_ready_partfield_and_falls_back_when_unavailable():
    assert segmentation_method_choices(True, "partfield") == (
        ["partfield", "shape_diameter"], "partfield"
    )
    # An existing reviewed choice is not silently replaced.
    assert segmentation_method_choices(True, "shape_diameter")[1] == "shape_diameter"
    assert segmentation_method_choices(False, "partfield") == (
        ["shape_diameter"], "shape_diameter"
    )


def test_face_slider_offers_a_resumed_sessions_face_target():
    # the operator skill caps a fine TripoSG extraction at 300,000 faces; a resumed
    # session showed that value to a slider that did not offer it (ValueError)
    options, value = faces_slider_options(300000)
    assert value == 300000 and value in options and options[-1] == -1
    options, value = faces_slider_options(123456)
    assert value == 123456 and value in options
    assert faces_slider_options(-1)[1] == -1 and faces_slider_options(None)[1] == -1
    assert faces_slider_options(20000)[1] in faces_slider_options(20000)[0]


def test_geometry_generation_uses_marco_defaults_and_validates_depths():
    settings = GeometryGenerationSettings()
    assert settings.model_dump() == {
        "generator": "triposg",
        "faces": -1,
        "init_octree_depth": 4,
        "octree_depth": 7,
        "seed": 42,
        "inference_steps": 16,
        "guidance_scale": 7.0,
        "prompt_confidence": 1.0,
        "scribble_confidence": 0.4,
        "erosion_radius": 0,
        "hunyuan3d_octree_resolution": 384,
        "texture": True,
        "hunyuan3d_flashvdm": True,
        "hunyuan3d_texture_size": 1024,
        "hunyuan3d_upscale": False,
    }
    with pytest.raises(ValueError, match="initial octree depth"):
        GeometryGenerationSettings(init_octree_depth=6, octree_depth=5)
    with pytest.raises(ValueError):
        GeometryGenerationSettings(prompt_confidence=1.01)
    with pytest.raises(ValueError):
        GeometryGenerationSettings(scribble_confidence=-0.01)
    assert GeometryGenerationSettings(faces=-1).faces == -1
    with pytest.raises(ValueError, match="unlimited"):
        GeometryGenerationSettings(faces=0)
    with pytest.raises(ValueError):
        GeometryGenerationSettings(seed=101)


def test_segmentation_settings_are_validated():
    assert SegmentationSettings().model_dump() == {
        "method": "shape_diameter",
        "num_clusters": 5, "num_rays": 25,
        "smoothing": 0.25, "cone_angle_deg": 120.0,
    }
    with pytest.raises(ValueError):
        SegmentationSettings(num_clusters=1)
    with pytest.raises(ValueError):
        SegmentationSettings(cone_angle_deg=181)


def test_recursive_selector_allowlist_and_arity():
    selector = SelectorSpec(type="and", children=[
        SelectorSpec(type="lowest", params={"axis": "z", "fraction": 0.1}),
        SelectorSpec(type="aligned_normal", params={"target_vector": [0, 0, -1], "max_angle_deg": 30}),
    ])
    assert len(selector.children) == 2
    with pytest.raises(ValueError, match="unsupported"):
        SelectorSpec(type="python_eval")
    with pytest.raises(ValueError, match="requires 2"):
        SelectorSpec(type="and", children=[])
    direct = SelectorSpec(type="preview_segments", params={"segment_ids": [1, 3]})
    assert direct.params["segment_ids"] == [1, 3]


def test_preview_segment_selector_reuses_saved_face_labels():
    import numpy as np

    torch = pytest.importorskip("torch")

    labels = np.array([0, 1, 2, 1], dtype=np.int64)
    selector = workflow_tool._selector({
        "type": "preview_segments", "params": {"segment_ids": [1]}, "children": [],
    }, labels)
    faces = torch.zeros((4, 3), dtype=torch.int64)
    dummy_vec = torch.zeros((4, 3))
    mask = selector.select(
        torch.zeros((1, 3)), faces, dummy_vec, dummy_vec, torch.ones(4)
    )
    assert mask.tolist() == [False, True, False, True]
    with pytest.raises(ValueError, match="invalid preview segment IDs"):
        workflow_tool._selector({
            "type": "preview_segments", "params": {"segment_ids": [9]}, "children": [],
        }, labels)


def test_bc_and_material_validation():
    low = {"type": "lowest", "params": {"axis": "z", "fraction": 0.1}}
    high = {"type": "highest", "params": {"axis": "z", "fraction": 0.1}}
    proposal = BCProposal.model_validate({
        "dirichlet": [{"name": "base", "selector": low, "components": ["z"], "value": 0}],
        "neumann": [{"name": "top", "selector": high, "traction": [0, 0, -1]}],
    })
    assert proposal.dirichlet[0].components == [2]
    with pytest.raises(ValueError):
        MaterialParameters(unit_system="SI (m-kg-s)", youngs_modulus=1, poisson_ratio=0.5)
    with pytest.raises(ValueError):
        MaterialParameters(unit_system="SI", youngs_modulus=1, poisson_ratio=0.3)
    assert MaterialParameters(
        unit_system="SI (m, N, Pa)", youngs_modulus=1, poisson_ratio=0.3
    ).unit_system == "SI (m-kg-s)"
    assert UNIT_SYSTEM_DETAILS["SI-mm (mm-tonne-s)"]["density"] == "tonne/mm³"


def test_adaptive_material_and_acceleration_contracts_are_backward_compatible():
    from pydantic import ValidationError
    from src.agents_wrapper.fem_workflow.models import (
        BCValueInference, MaterialInference, MaterialParameters,
    )

    legacy = MaterialInference(youngs_modulus=1.0e9, poisson_ratio=0.3)
    assert legacy.mode == "uniform" and legacy.density is None
    assert MaterialInference(mode="vomp").mode == "vomp"
    with pytest.raises(ValidationError):
        MaterialInference(mode="vomp", youngs_modulus=1.0)
    adaptive = MaterialParameters(
        unit_system="SI (m-kg-s)", mode="vomp", artifact="materials/vomp",
        approved=True,
    )
    assert adaptive.interpolation == "nearest"
    values = BCValueInference(
        dirichlet_values={"held": {"components": [0, 1, 2]}},
        neumann_values={"load": [0, 0, 0]}, acceleration=[0, -9.81, 0],
    )
    assert values.body_force == (0, 0, 0)
    assert values.acceleration == (0, -9.81, 0)
    legacy_values = BCValueInference(
        dirichlet_values={"held": {"components": [0, 1, 2]}},
        neumann_values={"load": [0, 0, 0]},
    )
    assert legacy_values.acceleration == (0, 0, 0)
    assert UNIT_SYSTEM_DETAILS["US-inch (in-lbf-s)"]["mass"] == "lbf·s²/in"


def test_bc_values_validate_components_loads_and_body_force():
    values = BCValuesInput.model_validate({
        "dirichlet_values": {"fixed": {"components": [2, 0], "value": 0.01}},
        "neumann_values": {"load": [0, -5, 0]},
        "body_force": [0, -9.81, 0],
    })
    assert values.dirichlet_values["fixed"].components == [0, 2]
    assert values.neumann_values["load"] == (0.0, -5.0, 0.0)
    with pytest.raises(ValueError):
        BCValuesInput.model_validate({
            "dirichlet_values": {"fixed": {"components": [], "value": 0}},
            "neumann_values": {"load": [0, -5, 0]},
        })
    # Tolerance covers shape and length, but not emptiness: the controller
    # back-fills omitted regions from the current values, so an all-empty
    # response would validate, cache, and leave every value unchanged.
    with pytest.raises(ValueError):
        BCValueInference.model_validate({
            "dirichlet_values": [], "neumann_values": [],
            "body_force": [0, 0, 0],
        })
    short = BCValueInference.model_validate({
        "dirichlet_values": [{"components": [0, 1, 2], "value": 0}],
        "neumann_values": [[0, -100, 0]], "body_force": [0, 0, 0],
    })
    assert len(short.dirichlet_values) == 1
    tolerant = BCValueInference.model_validate({
        "dirichlet_values": {"fixed": {"components": [0, 1, 2], "value": 0}},
        "neumann_values": {"load": [0, -100, 0]},
        "body_force": [0, 0, 0],
    })
    assert tolerant.neumann_values["load"] == (0.0, -100.0, 0.0)


def test_identical_payloads_at_different_steps_are_not_the_same_arrival():
    """A rebuilt step must not be mistaken for one already on screen.

    The scaling step is rebuilt from workflow state, so re-deciding the geometry
    before it hands back a byte-identical payload.  Treating that as a repeat
    skipped recording it, and the geometry card was left drawing the scaling
    step -- it disappeared.
    """
    payload = {"kind": "geometry_physics", "up_axis": "y", "physical_extent": 0.1}
    first = WorkflowEvent("s1", payload, {}, "checkpoint-1")
    regenerated = WorkflowEvent("s1", dict(payload), {}, "checkpoint-2")
    assert interrupt_marker(first) != interrupt_marker(regenerated)
    # The same step seen twice in one rerun is still one arrival.
    assert interrupt_marker(first) == interrupt_marker(
        WorkflowEvent("s1", dict(payload), {}, "checkpoint-1")
    )


def test_resume_rewinds_to_the_step_it_names(monkeypatch, tmp_path):
    """Re-deciding an earlier step must reach that step, not the pending one.

    Pinning a checkpoint on the invocation hands the answer to whatever task is
    waiting now, so editing an earlier card used to answer a later question with
    its payload.  The rewind has to roll the thread back first.
    """
    from typing import TypedDict
    from langgraph.graph import StateGraph, END
    from langgraph.types import interrupt

    class State(TypedDict, total=False):
        workflow: dict

    def ask_size(state):
        answer = interrupt({"kind": "size"})
        return {"workflow": {**state.get("workflow", {}), "size": answer}}

    def ask_material(state):
        answer = interrupt({"kind": "material"})
        return {"workflow": {**state.get("workflow", {}), "material": answer}}

    def build():
        graph = StateGraph(State)
        graph.add_node("size", ask_size)
        graph.add_node("material", ask_material)
        graph.set_entry_point("size")
        graph.add_edge("size", "material")
        graph.add_edge("material", END)
        return graph

    monkeypatch.setattr("src.ui.controller.build_graph", build)
    controller = WorkflowController(tmp_path)
    (tmp_path / "s1").mkdir()

    start = controller._invoke("s1", {"workflow": {}})
    assert start.interrupt == {"kind": "size"}
    size_step = start.checkpoint_id

    after_size = controller.resume("s1", "10 cm")
    assert after_size.interrupt == {"kind": "material"}
    done = controller.resume("s1", "steel")
    assert done.workflow == {"size": "10 cm", "material": "steel"}

    revised = controller.resume("s1", "20 cm", size_step)
    # The size question was asked again and took the new answer; everything the
    # old one led to is gone, so the material question is pending once more.
    assert revised.interrupt == {"kind": "material"}
    assert revised.workflow == {"size": "20 cm"}
    assert controller.resume("s1", "oak").workflow == {"size": "20 cm", "material": "oak"}


def test_history_keeps_card_payloads_and_collapses_repeated_steps(tmp_path):
    """Steps carry the payload that redraws them; repeated ones reuse one entry."""
    controller = WorkflowController(tmp_path)
    (tmp_path / "s1").mkdir()
    controller.append_history("s1", "assistant", "Describe the geometry", "cp1",
                              interrupt={"kind": "geometry_input"})
    controller.append_history("s1", "assistant", "Review settings", "cp2",
                              interrupt={"kind": "shape_opt_settings"})
    # A second solve refreshes the settings card rather than stacking a new one.
    controller.append_history("s1", "assistant", "The FEA solve failed", "cp3",
                              interrupt={"kind": "shape_opt_settings"},
                              replace_last=True)
    history = controller.history("s1")
    assert [entry["content"] for entry in history] == [
        "Describe the geometry", "The FEA solve failed",
    ]
    assert history[-1]["checkpoint_id"] == "cp3"
    controller.update_history_entry("s1", 0, answer={"description": "bracket"})
    assert controller.history("s1")[0]["answer"] == {"description": "bracket"}


def test_restored_interrupt_redraws_a_card_with_the_answer_it_was_sent(tmp_path):
    entry = {
        "interrupt": {
            "kind": "bc_values",
            "value_description": "",
            "body_force": [0.0, 0.0, 0.0],
            "dirichlet": [{"name": "fix", "components": [0], "value": 0.0}],
            "neumann": [{"name": "load", "traction": [0.0, 0.0, 0.0], "area": 2.0}],
            "defaults": {"relative_tol": 1e-6},
        },
        "answer": {
            "action": "apply_values",
            "value_description": "2 MPa downward",
            "body_force": [0.0, -9.81, 0.0],
            "dirichlet_values": {"fix": {"components": [0, 1, 2], "value": 0.5}},
            "neumann_values": {"load": [0.0, -2.0e6, 0.0]},
            "relative_tol": 1e-2,
        },
    }
    restored = restored_interrupt(entry)
    assert restored["value_description"] == "2 MPa downward"
    assert restored["body_force"] == [0.0, -9.81, 0.0]
    assert restored["dirichlet"] == [{"name": "fix", "components": [0, 1, 2], "value": 0.5}]
    # The region keeps the geometry the payload carried and takes the sent load.
    assert restored["neumann"] == [
        {"name": "load", "traction": [0.0, -2.0e6, 0.0], "area": 2.0},
    ]
    assert restored["defaults"]["relative_tol"] == 1e-2
    assert "action" not in restored
    assert restored_interrupt({"content": "legacy step"}) is None


def test_solver_progress_reads_load_steps_and_residuals():
    """The solver tab's numbers come from parsing the log the solve writes."""
    from src.ui.telemetry import read_solver_progress

    log = (
        "[09-16] Finish initialization\n"
        "\n===== Load step 1/3: scale=0.333333 =====\n"
        "  scale=0.3333 Iter 0: ||R|| = 7.734e-01, rel = 1.000e+00\n"
        "  scale=0.3333 Iter 1: ||R|| = 8.923e-03, rel = 1.154e-02\n"
        "\n===== Load step 2/3: scale=0.666667 =====\n"
        "  scale=0.6667 Iter 0: ||R|| = 5.409e-07, rel = 6.994e-07\n"
    )
    progress = read_solver_progress(log)
    assert (progress.load_step, progress.load_steps) == (2, 3)
    assert progress.fraction == pytest.approx(2 / 3)
    assert not progress.finished
    assert [row["relative"] for row in progress.iterations] == [1.0, 1.154e-2, 6.994e-7]
    # Each iteration carries the step it belongs to, so the chart can break the
    # line there instead of joining a residual that restarted.
    assert [row["load_step"] for row in progress.iterations] == [1.0, 1.0, 2.0]

    # A single-step solve logs no banner at all, which means one step, not none.
    single = read_solver_progress(
        "  scale=1.0000 Iter 0: ||R|| = 1.0e+00, rel = 1.0e+00\n"
        "Simulation finalized. Open /tmp/solution.pvd in ParaView to view results.\n"
    )
    assert (single.load_step, single.load_steps) == (1, 1)
    assert single.finished and single.fraction == 1.0

    empty = read_solver_progress("")
    assert (empty.load_step, empty.fraction, empty.iterations) == (0, 0.0, [])


def test_von_mises_matches_its_closed_form_cases():
    """The equivalent scalars, checked where the answer is known by hand.

    The Voigt shear slots hold tensor components -- half the engineering shear
    -- so the off-diagonals enter the double contraction doubled.  Treating them
    as engineering shear would inflate pure shear by sqrt(3).
    """
    import importlib.util

    import numpy as np

    path = (
        Path(__file__).parents[1] / "fem_workflow" / "scripts" / "workflow_tool.py"
    )
    spec = importlib.util.spec_from_file_location("workflow_tool_vm", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    def scalar(voigt, field):
        return float(tool._von_mises(np.asarray([voigt], dtype=float), field)[0])

    stress = 250.0
    # Uniaxial stress returns the stress itself.
    assert scalar([stress, 0, 0, 0, 0, 0], "stress") == pytest.approx(stress)
    # Pure shear returns sqrt(3) times the shear.
    assert scalar([0, 0, 0, stress, 0, 0], "stress") == pytest.approx(
        math.sqrt(3.0) * stress
    )
    # Hydrostatic pressure is entirely volumetric, so it has no deviator.
    assert scalar([stress, stress, stress, 0, 0, 0], "stress") == pytest.approx(0.0)
    # Uniaxial strain at nu = 0.5 is incompressible: equivalent strain is the strain.
    strain = 1.0e-3
    assert scalar(
        [strain, -0.5 * strain, -0.5 * strain, 0, 0, 0], "strain"
    ) == pytest.approx(strain)
    # Never negative, whatever the sign of the components.
    mixed = np.random.default_rng(0).normal(size=(64, 6))
    assert (tool._von_mises(mixed, "stress") >= 0).all()


def test_voigt_component_order_agrees_across_the_viewer_and_the_exporter():
    """Both sides must name Voigt slots the same way, or components mislabel.

    compute_strain_tensor stacks the gradient terms as
    (00, 11, 22, 01, 12, 02) -- that is xx, yy, zz, xy, yz, xz -- and the
    exporter indexes a column by that name while the viewer offers the names.
    A silent disagreement would colour one component and title it another.
    """
    source = (
        Path(__file__).parents[1] / "fem_workflow" / "assets.py"
    ).read_text(encoding="utf-8")
    tool = (
        Path(__file__).parents[1]
        / "fem_workflow" / "scripts" / "workflow_tool.py"
    ).read_text(encoding="utf-8")
    expected = '("xx", "yy", "zz", "xy", "yz", "xz")'
    assert f"TENSOR_COMPONENTS = {expected}" in source
    assert f"TENSOR_COMPONENTS = {expected}" in tool


def test_segment_legend_names_colours_and_keeps_them_distinct():
    """The vision stage is told the colours in words, not as hex codes.

    A hex triplet cannot be matched against a rendered fill, and two segments
    sharing one colour name would be worse than sending no legend at all.
    """
    from src.agents_wrapper.fem_workflow.specialists import (
        TAB20_NAMES, SpecialistCalls, color_name,
    )

    names = [color_name(f"#{value}") for value in TAB20_NAMES]
    assert len(set(names)) == len(names)
    assert color_name("#aec7e8") == color_name("AEC7E8") == "light blue"
    # Off-palette colours still get a usable word rather than nothing.
    assert color_name("#3366cc") == "blue"
    assert color_name("not a colour") == ""

    legend = SpecialistCalls._segment_legend(
        {
            "segments": [
                {"id": 0, "color": "#1f77b4"},
                {"id": 1, "color": "#aec7e8"},
                {"id": 2, "color": ""},
                {"no id": True},
            ]
        }
    )
    assert legend == {"0": "blue", "1": "light blue"}
    assert SpecialistCalls._segment_legend({}) == {}

    # Given the object's box, each id is anchored by where it actually sits --
    # measured, because judging it from the render is what the model gets wrong.
    anchored = SpecialistCalls._segment_legend(
        {
            "bounds": [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
            "segments": [
                {"id": 0, "color": "#1f77b4", "centroid": [0.2, 1.0, 1.9]},
                {"id": 1, "color": "#d62728", "centroid": [1.0, 0.2, 1.0]},
            ],
        }
    )
    # Height and depth are worded, not spelled as axes: +y is up and the
    # generator faces its output along +z, so "at the top" and "at the front"
    # are the facts the stage needs without translating a label first.  Width
    # keeps the axis form, because left and right depend on whose they are.
    assert anchored == {
        "0": "blue, low x, at mid height, at the front",
        "1": "red, mid x, at the bottom, mid depth",
    }

    # Thirds of a box gave a foot and the panel above it the same address, and
    # the stage, with nothing to separate them, invented a difference.  Five
    # steps of height and the measured size of each segment separate them.
    crowded = SpecialistCalls._segment_legend(
        {
            "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "segments": [
                {"id": 0, "color": "#1f77b4", "centroid": [0.13, 0.02, 0.88],
                 "area_fraction": 0.0069},
                {"id": 1, "color": "#9467bd", "centroid": [0.08, 0.23, 0.88],
                 "area_fraction": 0.0343},
            ],
        }
    )
    assert crowded["0"] == "blue, 0.7% of the surface, low x, at the bottom, at the front"
    assert crowded["1"] == "purple, 3.4% of the surface, low x, low, at the front"
    assert len(set(crowded.values())) == 2


def test_the_viewer_stands_the_model_on_a_ground_grid():
    """A grid on the x-z plane through the model's lowest point.

    Not through y=0.  A placed model does rest on the ground -- every solved
    surface in this repository starts within a centimetre of zero -- but the
    geometry card shows TripoSG's own output, which is centred on the origin, so
    a grid at y=0 saws exactly the model being iterated on in half.  The lowest
    point is the floor for both.  The plane stays horizontal because that y is
    constant across every vertex, which is what keeps the grid in x-z.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    grid = source[source.index("function placeGround"):]
    grid = grid[: grid.index("\n      }}")]
    # One y for the whole grid, taken from the model rather than from the world.
    assert "positions.push(originX + x, floor, originZ + z)" in grid
    assert "placeGround(center, box.min.y, span)" in source
    # Sized and placed from the model that framed the camera, so a 0.1 m bracket
    # and a 2 m table each get a grid they can be read against.
    assert "function placeGround(center, floor, span)" in source
    # Lines only, and they write no depth: orbiting under the floor looks up
    # through it at the model rather than at a lid.
    assert "LineSegments" in grid and "depthWrite: false" in grid
    # Snapped to whole cells, so the lines land on round world coordinates and
    # the spacing is a scale rather than a subdivision of this model.
    assert "Math.round(center.x / cell) * cell" in grid
    # Placed again -- another model, another frame -- the old grid goes away
    # rather than accumulating in the scene.
    assert "scene.remove(ground)" in grid and "ground.geometry.dispose()" in grid


def test_a_case_with_no_load_may_answer_with_no_tractions():
    """An empty load set is a refusal to answer, except when it is the answer.

    The value inference rejects empty collections on purpose: the controller
    back-fills whatever the response omits, so an empty answer would validate,
    cache, and silently leave every value as it was.  But a proposal may hold no
    Neumann region at all -- "fix the bottom" names a support and no load -- and
    then there is nothing to value and no other answer to give.  Asked under the
    loaded schema, that case was rejected, repaired and rejected again.
    """
    from pydantic import ValidationError

    from src.agents_wrapper.fem_workflow.models import (
        BCValueInference, UnloadedBCValueInference,
    )

    held = {"dirichlet_values": {"bottom": {"components": [0, 1, 2], "value": 0.0}}}
    unloaded = UnloadedBCValueInference.model_validate(
        {**held, "body_force": [0.0, -9810.0, 0.0]}
    )
    assert unloaded.neumann_values == {}
    # Gravity is where an unloaded case puts its load, and it still comes through.
    assert unloaded.body_force == (0.0, -9810.0, 0.0)
    # Dirichlet stays required either way: a body with nothing held has no
    # solution, so an empty answer there is never the answer.
    with pytest.raises(ValidationError):
        UnloadedBCValueInference.model_validate({"dirichlet_values": {}})
    # And where loads do exist the guard is untouched.
    with pytest.raises(ValidationError):
        BCValueInference.model_validate({**held, "neumann_values": {}})

    # Matching values onto regions must survive having no regions to match.  The
    # "named none of the approved regions" guard is for a mis-keyed answer, and
    # with no regions there is nothing to key against and nothing to mis-key.
    from src.agents_wrapper.fem_workflow.graph import _matched_values

    assert _matched_values({}, [], {}, "Neumann") == {}
    with pytest.raises(ValueError, match="name none of the approved regions"):
        _matched_values(
            {"wrong_name": [0.0, -1.0, 0.0]},
            [SimpleNamespace(name="top")],
            {"top": [0.0, 0.0, 0.0]},
            "Neumann",
        )

    # The caller says which case it is from the proposal it already holds.
    source = (
        Path(__file__).parents[1] / "fem_workflow" / "graph.py"
    ).read_text(encoding="utf-8")
    assert "bool(state.bc_proposal.neumann)," in source
    # The instructions carry the same exception, or the model answers the schema
    # it is validated against by inventing a region.
    values_reference = (
        Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
        / "boundary_condition_values.md"
    ).read_text(encoding="utf-8")
    assert '"neumann_values":{}' in values_reference


def test_the_grid_quotes_its_square_in_the_session_s_own_unit():
    """What one square measures, in the units everything else is quoted in.

    TripoSG returns a normalised box about two units across, and nothing gives
    it a size until scale_surface writes physical.glb -- so the geometry card is
    showing a shape whose "0.5" is not half a metre or half of anything else.
    That case gets a grid with no figure on it rather than an invented one.

    Nor is the figure converted into a tidier unit: a millimetre session's cell
    of 50 measures 50 mm, and calling it 5 cm would make the grid the one thing
    on screen not quoted the way the loads, the stresses and the extents are.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    label = source[source.index("function cellLabel"):]
    label = label[: label.index("\n      }}")]
    # The unit comes from the session; nothing here invents cm or mm.
    assert "groundUnit" in label
    assert "'cm'" not in label and "'mm'" not in label
    # No unit, no figure: the box stays hidden rather than showing a bare number.
    assert "if (scaleBox && groundUnit)" in source
    # And the unit is withheld for exactly the asset that has no size yet.
    assert 'surface.name != "physical.glb"' in source
    assert "length_unit=None if unscaled else UNIT_SYSTEM_DETAILS.get(" in source


def test_viewer_script_declares_bindings_before_using_them():
    """`let` bindings in the viewer must be declared above every use.

    The viewer is JavaScript inside a Python string, so nothing type-checks it
    and no test here can run it -- there is no WebGL in this environment.  A
    `let` used above its declaration parses cleanly and then throws "Cannot
    access X before initialization" at load, taking the whole viewer down.
    """
    import re

    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    start = source.index('<script type="module">')
    script = source[start : source.index("</script>", start)]
    # Comments name these bindings before declaring them, which is fine.
    script = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", script, flags=re.S))
    # Only the module's own level: a regex cannot model scope, and a name can be
    # a parameter in one function and a const in another quite legally.  The
    # dead-zone bug this guards against was a top-level binding reached by a
    # top-level statement, which is exactly what stays in range here.  let and
    # const share the dead zone, and the viewer is mostly const.
    top_level = r"^      (?:let|const) (\w+) ="
    declared = re.findall(top_level, script, re.MULTILINE)
    assert len(declared) > 5, "expected the viewer to still declare its bindings"
    for name in declared:
        declaration = re.search(rf"^      (?:let|const) {name} =", script, re.MULTILINE)
        first_use = re.search(rf"\b{name}\b", script)
        assert first_use.start() >= declaration.start(), (
            f"{name} is used before it is declared"
        )


def test_measured_extents_only_speak_for_a_supplied_mesh(tmp_path):
    """An uploaded mesh's own size is known; a generated one's is arbitrary.

    inspect_surface has always recorded the extents and nothing read them back,
    so a mesh modelled at its true size was rescaled to a placeholder unless the
    number was retyped.
    """
    from src.agents_wrapper.fem_workflow.graph import measured_extents

    def state(name, mode, payload=None):
        root = tmp_path / name
        (root / "geometry").mkdir(parents=True, exist_ok=True)
        if payload is not None:
            (root / "geometry" / "inspection.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return SimpleNamespace(geometry_mode=mode, root=root)

    assert measured_extents(
        state("uploaded", "surface", {"extents": [0.42, 0.75, 0.6]})
    ) == {"x": 0.42, "y": 0.75, "z": 0.6}
    # A generated mesh lives in the generator's [-1, 1] domain: its extents say
    # nothing physical, so they must not become a default.
    assert measured_extents(state("generated", "text", {"extents": [2.0, 2.0, 2.0]})) == {}
    # Missing, malformed or degenerate readings simply offer nothing.
    assert measured_extents(state("absent", "surface")) == {}
    assert measured_extents(state("short", "surface", {"extents": [1.0, 2.0]})) == {}
    assert measured_extents(state("flat", "surface", {"extents": [1.0, 0.0, 2.0]})) == {
        "x": 1.0,
        "z": 2.0,
    }


def test_residual_chart_marks_points_and_breaks_between_load_steps():
    """One line per load step, with a marker on every iteration.

    A residual restarts when the load steps up, so one line through the whole
    history draws a climb at the boundary that never happened.
    """
    from src.ui.telemetry import residual_chart

    spec = residual_chart(
        [
            {"load_step": 1.0, "residual": 7.7e-1, "relative": 1.0},
            {"load_step": 1.0, "residual": 8.9e-3, "relative": 1.2e-2},
            {"load_step": 2.0, "residual": 4.2e-1, "relative": 1.0},
        ]
    ).to_dict()
    assert spec["mark"] == {"type": "line", "point": True}
    # Both loops break the line, because either can be the one that restarted
    # the residual: load steps in a forward ramp, iterations in an optimization.
    assert [item["field"] for item in spec["encoding"]["detail"]] == [
        "load step", "opt iteration",
    ]
    rows = spec["datasets"][next(iter(spec["datasets"]))]
    # Two series over three iterations, each row tagged with its load step.
    assert len(rows) == 6
    assert {row["load step"] for row in rows} == {1, 2}
    absolute = [row for row in rows if row["series"] == "absolute"]
    assert [row["load step"] for row in absolute] == [1, 1, 2]
    # Log scale, so the orders of magnitude are what the axis shows.
    assert absolute[1]["log residual"] == pytest.approx(math.log10(8.9e-3))

    assert residual_chart([]).to_dict()["mark"]["point"] is True


def test_the_residual_chart_is_plotted_against_the_wallclock():
    """Each iteration sits at the second it was reached.

    Time comes from the stamps the solver writes on every line, measured from
    its first, so a log read after the fact plots what a log watched live did.
    The horizontal distance between two points is then the time between them,
    and the gap between one load step's line and the next is the mesh rebuild
    that no iteration count shows.
    """
    from src.ui.telemetry import read_solver_progress, residual_chart

    log = (
        "[09-17-2026,10:00:00.000] Finish initialization\n"
        "\n===== Load step 1/2: scale=0.5 =====\n"
        "[09-17-2026,10:00:02.500]   scale=0.5000 Iter 0: ||R|| = 5.4e-01, rel = 1.0e+00\n"
        "[09-17-2026,10:00:04.000]   scale=0.5000 Iter 1: ||R|| = 1.2e-02, rel = 2.3e-02\n"
        "\n===== Load step 2/2: scale=1.0 =====\n"
        "[09-17-2026,10:00:07.250]   scale=1.0000 Iter 0: ||R|| = 5.4e-01, rel = 1.0e+00\n"
    )
    progress = read_solver_progress(log)
    # Measured from the solver's first line, not its first iteration: meshing
    # and BC resolution are most of a short solve and the clock should say so.
    assert [row["elapsed"] for row in progress.iterations] == [2.5, 4.0, 7.25]

    spec = residual_chart(progress.iterations).to_dict()
    assert spec["encoding"]["x"]["field"] == "seconds"
    assert spec["encoding"]["x"]["scale"]["domain"] == [0, 7.25]
    rows = spec["datasets"][next(iter(spec["datasets"]))]
    absolute = [row for row in rows if row["series"] == "absolute"]
    assert [row["seconds"] for row in absolute] == [2.5, 4.0, 7.25]
    # The residual restarts when the load steps up, so its line still breaks
    # there -- now leaving a gap as wide as the rebuild took.
    assert [item["field"] for item in spec["encoding"]["detail"]] == [
        "load step", "opt iteration",
    ]

    # A solve that finishes inside five seconds still opens on five, rather than
    # spreading three iterations across an axis reading in hundredths.  A run
    # that outgrows the floor is plotted to its own last second instead.
    brief = read_solver_progress(
        "[09-17-2026,10:00:00.000] Finish initialization\n"
        "[09-17-2026,10:00:00.180]   scale=1.0000 Iter 0: ||R|| = 1.0e+00, rel = 1.0e+00\n"
        "[09-17-2026,10:00:00.400]   scale=1.0000 Iter 1: ||R|| = 1.0e-03, rel = 1.0e-03\n"
    )
    assert residual_chart(brief.iterations).to_dict()["encoding"]["x"]["scale"][
        "domain"
    ] == [0, 5.0]

    # An unstamped log -- or a mock one -- is plotted against the ordinal.
    plain = read_solver_progress(
        "  scale=1.0000 Iter 0: ||R|| = 1.0e+00, rel = 1.0e+00\n"
    )
    assert "elapsed" not in plain.iterations[0]
    assert residual_chart(plain.iterations).to_dict()["encoding"]["x"]["field"] == "step"


def test_an_optimization_is_read_by_its_iterations_not_its_load_steps():
    """A shape optimization's outer loop is what the panel has to report.

    It runs a whole Newton solve per iteration on a shape that has just moved,
    so the residual restarts at every one of them exactly as it does at a load
    step -- and its load step is always one of one, which says nothing about a
    run a hundred iterations long.
    """
    from src.ui.telemetry import read_solver_progress

    log = (
        "[09-17-2026,10:00:00.000] Workflow effective config: "
        "use_mixed_precision=False, run_mode=shape_optimization, "
        "max_optimize_iters=40, volume_as_is=True\n"
        "[09-17-2026,10:00:01.000] ===== Optimization Iteration 0 =====\n"
        "[09-17-2026,10:00:02.000]   scale=1.0000 Iter 0: ||R|| = 5.0e-01, rel = 1.0e+00\n"
        "[09-17-2026,10:00:03.000]   scale=1.0000 Iter 1: ||R|| = 5.0e-04, rel = 1.0e-03\n"
        "[09-17-2026,10:00:04.000]   Opt iter 0: obj=1.250000e+00, "
        "constraints=[none (gd)], rel_red=inf\n"
        "[09-17-2026,10:00:05.000] ===== Optimization Iteration 1 =====\n"
        "[09-17-2026,10:00:06.000]   scale=1.0000 Iter 0: ||R|| = 4.0e-01, rel = 1.0e+00\n"
        "[09-17-2026,10:00:08.000]   Opt iter 1: obj=1.100000e+00, "
        "constraints=[none (gd)], rel_red=1.2e-01\n"
    )
    progress = read_solver_progress(log)
    assert progress.optimizing
    # The residual restarts at each iteration, so each row says which it belongs
    # to -- and the Newton count inside one iteration is not the iteration.
    assert [row["opt_iteration"] for row in progress.iterations] == [0.0, 0.0, 1.0]
    assert [row["iteration"] for row in progress.iterations] == [0.0, 1.0, 0.0]
    # Counted against what the run was asked for, which the solve logs before
    # its first iteration; iterations are numbered from zero, so the second one
    # under way is two of forty.
    assert (progress.opt_iteration, progress.opt_iterations) == (1, 40)
    assert progress.fraction == pytest.approx(2 / 40)
    assert [row["value"] for row in progress.objective] == [1.25, 1.1]
    assert [row["elapsed"] for row in progress.objective] == [4.0, 8.0]

    # A forward solve carries the same config line -- one tool writes both -- and
    # must not be read as an optimization on the strength of it.
    forward = read_solver_progress(
        "[09-17-2026,10:00:00.000] Workflow effective config: "
        "run_mode=forward, max_optimize_iters=40\n"
        "\n===== Load step 1/2: scale=0.5 =====\n"
        "[09-17-2026,10:00:02.000]   scale=0.5000 Iter 0: ||R|| = 1.0e+00, rel = 1.0e+00\n"
    )
    assert not forward.optimizing
    assert forward.opt_iterations == 0
    assert forward.fraction == pytest.approx(0.5)


def test_the_objective_is_plotted_against_the_right_hand_axis():
    """What the optimizer is minimising, beside the residuals it took to get it.

    It shares the clock and nothing else: it is not a residual, is not measured
    in orders of magnitude, and there is one of it per iteration rather than one
    per Newton step.  So it is layered on a scale of its own.
    """
    from src.ui.telemetry import residual_chart

    iterations = [
        {"load_step": 1.0, "opt_iteration": 0.0, "residual": 5.0e-1,
         "relative": 1.0, "elapsed": 2.0},
        {"load_step": 1.0, "opt_iteration": 1.0, "residual": 4.0e-1,
         "relative": 1.0, "elapsed": 6.0},
    ]
    objective = [
        {"opt_iteration": 0.0, "value": 1.25, "elapsed": 4.0},
        {"opt_iteration": 1.0, "value": 1.10, "elapsed": 8.0},
    ]
    spec = residual_chart(iterations, objective).to_dict()
    residuals, values = spec["layer"]
    assert spec["resolve"]["scale"]["y"] == "independent"
    assert residuals["encoding"]["y"]["field"] == "log residual"
    assert values["encoding"]["y"]["field"] == "objective"
    assert values["encoding"]["y"]["axis"]["orient"] == "right"
    # Not anchored at zero: the reading is how far the objective has come down,
    # and two values a percent apart would otherwise share one pixel.
    assert values["encoding"]["y"]["scale"]["zero"] is False
    # One clock for both, so a point's horizontal place means the same thing in
    # either layer.
    assert values["encoding"]["x"]["field"] == "seconds"
    # Wide enough for both: the objective is logged after its iteration's last
    # residual, so an axis measured off the residuals alone would cut off the
    # final reading of the very thing being minimised.
    assert residuals["encoding"]["x"]["scale"]["domain"] == [0, 8.0]
    assert values["encoding"]["x"]["scale"]["domain"] == [0, 8.0]
    # Its own entry in the one legend the two layers share.
    assert values["encoding"]["color"]["field"] == "series"
    rows = spec["datasets"][values["data"]["name"]]
    assert [row["objective"] for row in rows] == [1.25, 1.1]

    # A log with no timestamps has no shared ordinal to plot it against -- the
    # residuals are counted per Newton step and the objective per iteration --
    # so the chart is the residuals alone rather than two scales out of step.
    untimed = residual_chart(
        [{"load_step": 1.0, "opt_iteration": 0.0, "residual": 1.0, "relative": 1.0}],
        [{"opt_iteration": 0.0, "value": 1.25}],
    ).to_dict()
    assert "layer" not in untimed
    assert untimed["encoding"]["x"]["field"] == "step"


def test_which_of_a_run_s_shapes_the_viewer_can_hold():
    """Every shape carried into the frame is a megabyte and a mesh.

    A hundred-iteration run cannot be handed over whole, so a long one is
    thinned -- evenly, and keeping both ends, so the slider still spans the
    optimization rather than covering its first dozen iterations and stopping.
    """
    from src.ui.telemetry import VIEWER_FRAMES, chosen_iterations, iteration_numbers

    # Short enough to carry whole.
    assert chosen_iterations(1) == [0]
    assert chosen_iterations(VIEWER_FRAMES) == list(range(VIEWER_FRAMES))
    assert chosen_iterations(0) == []

    spread = chosen_iterations(101, cap=6)
    assert len(spread) == 6
    # Both ends, so the shape it started from and the one it reached are always
    # among them; the rest evenly spaced between.
    assert spread[0] == 0 and spread[-1] == 100
    assert spread == [0, 20, 40, 60, 80, 100]
    assert chosen_iterations(7, cap=4) == [0, 2, 4, 6]

    # Labelled with the iteration each surface came from, not its position: a
    # run saving every tenth iteration wrote its fourth file at iteration 30.
    assert iteration_numbers(4, 10) == [0, 10, 20, 30]
    assert iteration_numbers(3, None) == [0, 1, 2]
    assert iteration_numbers(0, 1) == []


def test_a_shape_dropped_to_make_room_is_not_sent_again():
    """The viewer is fed from a high-water mark, not from what it still holds.

    Those part company the moment it is full and starts discarding, and reading
    the second as the first made a running optimization jump backwards: a shape
    dropped to make room read as one never sent, so it was sent again, and the
    picture fell ten iterations behind before the next one pulled it forward.
    """
    from src.ui.telemetry import next_live_frame

    # Nothing written since the viewer was last brought up to date.
    assert next_live_frame(seen=3, count=4, held=4, cap=16) is None
    assert next_live_frame(seen=0, count=1, held=1, cap=16) is None
    # The newest, and no discard while there is room.
    assert next_live_frame(seen=3, count=5, held=4, cap=16) == (4, None)
    # Only the newest: building one takes longer than an iteration, so the
    # three that landed while the last was being built are stepped over.
    assert next_live_frame(seen=3, count=8, held=4, cap=16) == (7, None)
    # Full, so the second shape makes way -- never the first, which is what the
    # newest one is worth comparing against.
    assert next_live_frame(seen=15, count=17, held=16, cap=16) == (16, 1)
    # And the one just discarded does not come back as though it were new: the
    # mark has moved past it even though the viewer no longer holds it.
    assert next_live_frame(seen=16, count=17, held=16, cap=16) is None


def test_what_kind_of_solve_wrote_these_surfaces_is_read_from_its_log(tmp_path):
    """The viewer asks the run, not the card.

    The settings in the workflow are the ones that would be used next; the
    surfaces on disk were written by the ones used last, and re-deciding an
    earlier card rewinds the first past the second.  A slider that walked
    iterations only while the card still said so would vanish from a result it
    is the only way to read.
    """
    from src.ui.telemetry import optimization_saves

    log = tmp_path / "log.txt"
    log.write_text(
        "[09-17-2026,10:00:00.000] Workflow effective config: "
        "run_mode=shape_optimization, max_optimize_iters=40, "
        "output_per_optimize_step=10, volume_as_is=False\n"
        "[09-17-2026,10:00:01.000] ===== Optimization Iteration 0 =====\n",
        encoding="utf-8",
    )
    assert optimization_saves(log) == 10

    forward = tmp_path / "forward.txt"
    forward.write_text(
        "[09-17-2026,10:00:00.000] Workflow effective config: run_mode=forward, "
        "max_optimize_iters=40, output_per_optimize_step=10\n"
        "\n===== Load step 1/2: scale=0.5 =====\n",
        encoding="utf-8",
    )
    assert optimization_saves(forward) is None
    assert optimization_saves(None) is None
    assert optimization_saves(tmp_path / "nothing.txt") is None

    # A log from before the interval was recorded still says plainly what it
    # was, and every such run saved every iteration.
    older = tmp_path / "older.txt"
    older.write_text("===== Optimization Iteration 0 =====\n", encoding="utf-8")
    assert optimization_saves(older) == 1


def test_the_iteration_slider_is_in_the_viewer_beside_the_other_two(tmp_path):
    """All three controls in one strip, and none of them costing a rerun.

    The iteration slider walks shapes already in the frame, exactly as the load
    step slider does, because a control that rebuilt the picture on every pick
    would not belong beside two that do not.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    # One control, named for whichever loop the run actually walked, drawn in
    # the same strip as the deformation slider.
    assert 'step_label = "Iteration" if optimizing else "Load step"' in source
    assert "saves_every = optimization_saves(newest_solve_log(root))" in source
    assert '{html.escape(step_label)} ' in source
    assert "'<div class=\"viewer-controls\">{step_markup}{deform_markup}{safety_markup}</div>'" in source
    # Stamped with its iteration from the first surface, so iteration zero is
    # not written unstamped and then rewritten the moment a second one lands.
    assert "max(len(steps), 2) if optimizing" in source
    # Each iteration is coloured over its own extremes, so the bar travels with
    # the shape rather than describing only the one the viewer opened on.
    assert "function showRange(range)" in source


def test_the_viewer_is_fed_while_it_runs_rather_than_rebuilt(tmp_path):
    """Why the running solve reaches the viewer through a script, not a redraw.

    The viewer takes its height from a flex chain that runs from the panel down
    to the frame; a fragment wraps what it draws in a block that is not part of
    that chain, and the frame collapsed to nothing -- a viewport that was simply
    not there for the whole of a run.  So the viewer stays in the page and the
    shapes are posted into it.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    assert "def live_frame_bridge() -> None:" in source
    assert "@st.fragment(run_every=VIEWER_REFRESH)" in source
    # Posted into the frame that is already there, not drawn over it.
    assert "frame.contentWindow.postMessage({payload}, '*')" in source
    # The frame the viewer actually lives in; a selector that missed it pushed
    # every shape into nothing and the slider never grew.
    assert """'[class*="st-key-field_view"] iframe'""" in source
    assert "offer = next_live_frame(plan[\"seen\"], len(steps), len(carried))" in source
    assert "function discardFrame(index)" in source
    assert "window.addEventListener('message', (event) => {{" in source
    # And the viewer must not be inside the fragment that feeds it.
    panel = source.index('key="visualization_panel"')
    assert "render_visualization(event)" in source[panel:panel + 600]
    assert "render_visualization_live" not in source
    # Which shape to send, and what makes way for it, is one decision and it is
    # tested on its own -- see the high-water-mark test above.
    assert "index, drop = offer" in source
    # The page is frozen for the length of a step, so the one redraw that puts a
    # result in front of the bridge has to be asked for.
    assert "def first_surface_landed(pending: dict) -> bool:" in source
    assert "shown_first_surface" in source
    # Redrawing must not re-encode megabytes of surface each time.
    assert "@st.cache_data(show_spinner=False, max_entries=ENCODED_ASSETS)" in source


def test_volume_as_is_reaches_the_solver_from_the_card():
    """Penalising growth from the starting volume, rather than to a fraction.

    ShapeOpt has read this parameter all along; nothing in the workflow set it,
    so a run could only ever chase a target fraction.
    """
    import inspect

    from src.agents_wrapper.fem_workflow.models import ShapeOptSettings
    from src.agents_wrapper.fem_workflow.scripts import workflow_tool

    assert ShapeOptSettings().volume_as_is is False
    assert ShapeOptSettings.model_validate(
        {"preset_version": 2, "volume_as_is": True}
    ).volume_as_is is True

    # Into the params the solver reads, under the name ShapeOpt already uses.
    source = inspect.getsource(workflow_tool.run_forward)
    assert '"volume_as_is": settings.volume_as_is,' in source
    solver = (
        Path(__file__).parents[2] / "warp_wrapper" / "core" / "shape_opt.py"
    ).read_text(encoding="utf-8")
    assert 'volume_as_is = bool(self.params.get("volume_as_is", False))' in solver

    app = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    assert '"Keep the starting volume",' in app
    assert '"volume_as_is": volume_as_is,' in app


def test_machine_load_readings_are_percentages():
    from src.ui.telemetry import read_cpu_totals, read_gpu, read_memory

    assert read_memory("MemTotal: 1000 kB\nMemAvailable: 250 kB\n") == pytest.approx(75.0)
    assert read_memory("MemTotal: 1000 kB\n") is None
    # busy excludes idle and iowait, the 4th and 5th fields.
    assert read_cpu_totals("cpu  10 0 10 80 0 0 0\ncpu0 1 1 1 1\n") == (20, 100)
    assert read_cpu_totals("intr 1 2 3\n") is None
    # Several cards report one line each and stay separate: a solve pins one of
    # them, and pooling would average that away.
    cards = read_gpu("29, 15254, 49140\n71, 4386, 49140")
    assert len(cards) == 2
    assert cards[0] == (pytest.approx(29.0), pytest.approx(100 * 15254 / 49140))
    assert cards[1] == (pytest.approx(71.0), pytest.approx(100 * 4386 / 49140))
    assert read_gpu("") == ()
    assert read_gpu("garbage\n29, 15254, 49140") == (
        (pytest.approx(29.0), pytest.approx(100 * 15254 / 49140)),
    )


def test_backend_events_merge_api_calls_and_tool_runs(tmp_path):
    import json as _json
    from src.ui.telemetry import read_backend_events

    (tmp_path / "api_log.jsonl").write_text(_json.dumps({
        "timestamp": 100.0, "purpose": "material_interpretation", "model": "m",
        "cached": True, "total_tokens": 42, "prompt": "ask", "response": "answer",
    }) + "\nnot json\n", encoding="utf-8")
    receipts = tmp_path / "tool_receipts"
    receipts.mkdir()
    stdout = tmp_path / "out.log"
    stdout.write_text("built the case", encoding="utf-8")
    (receipts / "abc.json").write_text(_json.dumps({
        "tool": "build_case", "returncode": 0, "stdout_path": str(stdout),
        "stderr_path": str(tmp_path / "missing.log"), "artifacts": [str(tmp_path / "case.json")],
    }), encoding="utf-8")

    events = read_backend_events(tmp_path)
    assert [event["kind"] for event in events] == ["api", "tool"]
    assert "ask" in events[0]["body"] and "answer" in events[0]["body"]
    assert events[0]["detail"] == "42 tokens"
    assert "cached" in events[0]["title"]
    assert "built the case" in events[1]["body"]
    assert events[1]["detail"] == "case.json"


def test_monitor_rejects_unregistered_and_external_paths(tmp_path):
    monitor = ToolMonitor(tmp_path)
    with pytest.raises(ToolRejected, match="not registered"):
        monitor.execute("shell", {})
    with pytest.raises(ToolRejected, match="inside"):
        monitor.execute("render_surface", {"surface": "/tmp/outside.glb", "output": str(tmp_path / "a.png")})
    # The optional loads file is a session path like the rest of the case inputs.
    inside = {key: str(tmp_path / f"{key}.json") for key in (
        "surface", "bc_json", "segments_npz", "material_json", "settings_json",
        "transform_json", "output", "solve_output")}
    with pytest.raises(ToolRejected, match="loads_json must be inside"):
        monitor.execute("build_case", {**inside, "octree_depth": 7,
                                       "loads_json": "/tmp/outside/loads.json"})


def test_partfield_cold_start_uses_its_interpreter_without_conda_run(
        monkeypatch, tmp_path):
    import subprocess
    from types import SimpleNamespace

    surface = tmp_path / "surface.glb"
    surface.write_bytes(b"mesh")
    outputs = [tmp_path / "segments.npz", tmp_path / "segments.json",
               tmp_path / "segments.glb"]
    commands = []

    monkeypatch.setattr(
        "src.agents_wrapper.fem_workflow.monitor.partfield_status",
        lambda: SimpleNamespace(
            ready=True, interpreter="/opt/conda/envs/mechfoundry-partfield/bin/python",
            environment="mechfoundry-partfield", summary="ready",
        ),
    )
    monkeypatch.setattr(
        "src.agents_wrapper.fem_workflow.monitor._direct_interpreter",
        lambda _environment: Path("/opt/conda/envs/mechfoundry-warp117/bin/python"),
    )
    monkeypatch.setattr(ToolMonitor, "_shutdown_geometry_worker", lambda self: None)
    monkeypatch.setattr(ToolMonitor, "_run_worker", lambda self, *args, **kwargs: None)

    def fake_run(command, **kwargs):
        commands.append(command)
        for output in outputs:
            output.write_bytes(b"artifact")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(
        "src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run
    )
    ToolMonitor(tmp_path).execute("segment_surface", {
        "method": "partfield", "surface": str(surface),
        "output_npz": str(outputs[0]), "output_json": str(outputs[1]),
        "output_glb": str(outputs[2]),
        "feature_cache": str(tmp_path / "partfield_features"), "clusters": 4,
    })
    assert commands[0][0] == "/opt/conda/envs/mechfoundry-partfield/bin/python"
    assert "conda" not in commands[0]
    assert commands[0][1].endswith("src/partfield_wrapper/segment.py")
    cluster_flag = commands[0].index("--cluster-interpreter")
    assert commands[0][cluster_flag + 1] == "/opt/conda/envs/mechfoundry-warp117/bin/python"


def test_entering_or_restoring_a_partfield_card_ensures_its_worker(
        monkeypatch, tmp_path):
    from src.ui.controller import WorkflowController

    controller = WorkflowController(tmp_path)
    launched = []
    monkeypatch.setattr(
        controller, "preload_partfield_worker", launched.append
    )
    workflow = {"compute_env": "mechfoundry-warp117"}
    controller._preload_for_interrupt("session", {
        "kind": "bc_input",
        "partfield": {"ready": True},
        "segmentation": {"method": "partfield"},
    }, workflow)
    assert launched == ["mechfoundry-warp117"]

    # A redraw or restore is idempotent: the ensure call finds the same worker.
    controller._preload_for_interrupt("session", {
        "kind": "bc_input",
        "partfield": {"ready": True},
        "segmentation": {"method": "partfield"},
    }, workflow)
    assert launched == ["mechfoundry-warp117", "mechfoundry-warp117"]

    controller._preload_for_interrupt("session", {
        "kind": "bc_input",
        "partfield": {"ready": True},
        "segmentation": {"method": "shape_diameter"},
    }, workflow)
    assert len(launched) == 2


def test_monitor_cache_tracks_input_file_contents(monkeypatch, tmp_path):
    import subprocess

    paths = {
        name: tmp_path / filename for name, filename in {
            "surface": "surface.glb", "bc_json": "bc.json",
            "segments_npz": "segments.npz", "material_json": "material.json",
            "settings_json": "settings.json", "transform_json": "transform.json",
        }.items()
    }
    for path in paths.values():
        path.write_text("initial", encoding="utf-8")
    output = tmp_path / "case.json"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output.write_text("case", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    arguments = {
        **{name: str(path) for name, path in paths.items()},
        "output": str(output), "solve_output": str(tmp_path / "solve"),
        "octree_depth": 7,
    }
    monitor = ToolMonitor(tmp_path)
    monitor.execute("build_case", arguments)
    paths["settings_json"].write_text("edited", encoding="utf-8")
    monitor.execute("build_case", arguments)
    assert len(calls) == 2


def test_file_digest_is_reused_until_the_file_changes(monkeypatch, tmp_path):
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    target = tmp_path / "large.vtu"
    target.write_bytes(b"a" * 4096)
    monitor_module._digest_cache.clear()
    original_open = Path.open
    reads = []

    def counted_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == target and mode == "rb":
            reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    first = monitor_module._digest(target)
    assert monitor_module._digest(target) == first
    assert len(reads) == 1

    target.write_bytes(b"b" * 4096)
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert monitor_module._digest(target) != first
    assert len(reads) == 2


def test_monitor_reruns_when_a_later_call_overwrote_the_cached_artifact(monkeypatch, tmp_path):
    """Reverting a setting must not reuse a receipt whose output was overwritten.

    Every build_case writes the same case.json, so settings A -> B leaves B's
    case on disk while A's receipt still matches its own fingerprint.  Going
    back to A has to rebuild the case rather than solve with B's.
    """
    import subprocess

    paths = {
        name: tmp_path / filename for name, filename in {
            "surface": "surface.glb", "bc_json": "bc.json",
            "segments_npz": "segments.npz", "material_json": "material.json",
            "settings_json": "settings.json", "transform_json": "transform.json",
        }.items()
    }
    for path in paths.values():
        path.write_text("initial", encoding="utf-8")
    output = tmp_path / "case.json"

    def fake_run(command, **kwargs):
        output.write_text(paths["settings_json"].read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    arguments = {
        **{name: str(path) for name, path in paths.items()},
        "output": str(output), "solve_output": str(tmp_path / "solve"),
        "octree_depth": 7,
    }
    monitor = ToolMonitor(tmp_path)
    paths["settings_json"].write_text("mixed=False", encoding="utf-8")
    monitor.execute("build_case", arguments)
    paths["settings_json"].write_text("mixed=True", encoding="utf-8")
    monitor.execute("build_case", arguments)
    assert output.read_text(encoding="utf-8") == "mixed=True"
    paths["settings_json"].write_text("mixed=False", encoding="utf-8")
    monitor.execute("build_case", arguments)
    assert output.read_text(encoding="utf-8") == "mixed=False"


def test_monitor_puts_a_cached_artifact_back_from_the_session_versions(monkeypatch, tmp_path):
    """A rewind restores its checkpoint's files over a later call's output (a painting made
    after the checkpoint, at the same path); the same call again is served from its receipt with
    its own output copied back from versions/, not run again (repaints of unchanged words)."""
    import subprocess

    from src.agents_wrapper.fem_workflow.monitor import _digest

    paths = {
        name: tmp_path / filename for name, filename in {
            "surface": "surface.glb", "bc_json": "bc.json",
            "segments_npz": "segments.npz", "material_json": "material.json",
            "settings_json": "settings.json", "transform_json": "transform.json",
        }.items()
    }
    for path in paths.values():
        path.write_text("initial", encoding="utf-8")
    output = tmp_path / "case" / "case.json"
    output.parent.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output.write_text(f"made from {paths['settings_json'].read_text(encoding='utf-8')}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    arguments = {**{name: str(path) for name, path in paths.items()},
                 "output": str(output), "solve_output": str(tmp_path / "solve"), "octree_depth": 7}
    monitor = ToolMonitor(tmp_path)
    monitor.execute("build_case", arguments)
    made = output.read_text(encoding="utf-8")
    # what the graph's checkpoint record keeps: every branch file, by content
    (tmp_path / "versions").mkdir()
    (tmp_path / "versions" / f"{_digest(output)}.json").write_text(made, encoding="utf-8")
    # the rewind puts the checkpoint's earlier file back at the same path
    output.write_text("an earlier checkpoint's file", encoding="utf-8")
    receipt = monitor.execute("build_case", arguments)
    assert len(calls) == 1
    assert output.read_text(encoding="utf-8") == made
    assert receipt.returncode == 0
    # the file it displaced is kept among the versions, as the graph's rewind keeps its own
    displaced = [p.read_text(encoding="utf-8") for p in (tmp_path / "versions").iterdir()]
    assert "an earlier checkpoint's file" in displaced
    # without a kept version the call runs again, as before
    output.write_text("an earlier checkpoint's file", encoding="utf-8")
    (tmp_path / "versions" / f"{receipt.artifact_digests[str(output)]}.json").unlink()
    monitor.execute("build_case", arguments)
    assert len(calls) == 2


def test_monitor_restores_artifacts_only_inside_the_session(tmp_path):
    """A receipt naming a file outside the session, or a symbolic link out of it, is never
    written through: the tool runs again instead."""
    from src.agents_wrapper.fem_workflow.models import ToolReceipt
    from src.agents_wrapper.fem_workflow.monitor import _restore_artifacts

    session, outside = tmp_path / "session", tmp_path / "elsewhere"
    (session / "versions").mkdir(parents=True)
    outside.mkdir()
    kept = "a" * 64
    (session / "versions" / f"{kept}.glb").write_text("kept", encoding="utf-8")

    def receipt(path):
        return ToolReceipt(tool="paint_hunyuan3d_surface", arguments={}, returncode=0,
                           stdout_path="", stderr_path="", artifacts=[str(path)],
                           artifact_digests={str(path): kept})

    target = outside / "textured.glb"
    target.write_text("theirs", encoding="utf-8")
    assert _restore_artifacts(receipt(target), session) is False
    assert target.read_text(encoding="utf-8") == "theirs"
    link = session / "textured.glb"
    link.symlink_to(target)
    assert _restore_artifacts(receipt(link), session) is False
    assert target.read_text(encoding="utf-8") == "theirs" and link.is_symlink()
    inside_dir = session / "materials"
    inside_dir.mkdir()
    (inside_dir / "out").symlink_to(outside, target_is_directory=True)
    assert _restore_artifacts(receipt(inside_dir / "out" / "textured.glb"), session) is False
    assert target.read_text(encoding="utf-8") == "theirs"


def test_forward_solve_never_reuses_a_cached_receipt(monkeypatch, tmp_path):
    """Run FEA means solve, even when nothing about the case changed."""
    import subprocess

    solve_output = tmp_path / "solve"
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"output": str(solve_output)}), encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        solve_output.mkdir(exist_ok=True)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    monitor = ToolMonitor(tmp_path)
    monitor.execute("run_forward", {"case": str(case)})
    monitor.execute("run_forward", {"case": str(case)})
    assert len(calls) == 2


def test_forward_solve_disables_xla_gpu_preallocation(monkeypatch, tmp_path):
    import subprocess

    solve_output = tmp_path / "solve"
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"output": str(solve_output)}), encoding="utf-8")
    environments = []

    def fake_run(command, **kwargs):
        environments.append(kwargs["env"])
        solve_output.mkdir()
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    ToolMonitor(tmp_path).execute("run_forward", {"case": str(case)})
    assert environments[0]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert environments[0]["XDG_CACHE_HOME"] == str(tmp_path.parent / ".runtime_cache")


def test_stage_asset_preserves_bytes_and_hash(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"unchanged")
    output = tmp_path / "session" / "original.png"
    args = type("Args", (), {"source": source, "output": output})()
    artifacts = workflow_tool.stage_asset(args)
    assert output.read_bytes() == b"unchanged"
    metadata = json.loads(Path(artifacts[1]).read_text())
    assert metadata["sha256"] == hashlib.sha256(b"unchanged").hexdigest()


def test_bc_geometry_reuses_mesh_arrays_and_tensors_until_surface_changes(tmp_path):
    pytest.importorskip("torch")
    import trimesh

    source = tmp_path / "surface.glb"
    trimesh.creation.box().export(source)
    workflow_tool._mesh_cache.clear()
    workflow_tool._bc_geometry_cache.clear()

    first = workflow_tool._bc_geometry(source, "cpu")
    second = workflow_tool._bc_geometry(source, "cpu")
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert second[2] is first[2]
    assert second[3] is first[3]

    trimesh.creation.icosphere(subdivisions=1).export(source)
    changed = workflow_tool._bc_geometry(source, "cpu")
    assert changed[0] is not first[0]
    assert len(changed[0].faces) != len(first[0].faces)


def test_scale_surface_preserves_reference_embedding_transform(tmp_path):
    import trimesh

    source = tmp_path / "source.glb"
    output = tmp_path / "physical.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(source)
    args = type("Args", (), {
        "surface": source, "output": output, "axis": "y", "extent": 10.0,
        "units": "SI (m-kg-s)", "octree_depth": 7,
    })()
    artifacts = workflow_tool.scale_surface(args)
    transform = json.loads(Path(artifacts[1]).read_text())
    # h_norm snaps the source bbox to 0.045 planes, so the 2.0-unit up axis
    # becomes 1.98 and the requested 10.0 extent comes out of a 5.0505 scale.
    cube_norm = workflow_tool.OCTREE_CUBE_PAD * 3.0
    assert transform["h_norm"] == pytest.approx(cube_norm / 128.0)
    assert transform["uniform_scale"] == pytest.approx(10.0 / 1.98)
    assert transform["mesh_center"] == pytest.approx([2.5, 5.0, 7.5])
    assert transform["h_target"] == pytest.approx(
        (cube_norm / 128.0) * transform["uniform_scale"]
    )


def test_scale_surface_octree_cube_tracks_physical_extent(tmp_path):
    """A given depth must resolve the part the same way the examples do."""
    import trimesh

    def transform_for(extents, extent, depth=7):
        source = tmp_path / f"src-{extents}-{extent}-{depth}.glb"
        output = tmp_path / f"out-{extents}-{extent}-{depth}.glb"
        trimesh.creation.box(extents=extents).export(source)
        args = type("Args", (), {
            "surface": source, "output": output, "axis": "y", "extent": extent,
            "units": "SI (m-kg-s)", "octree_depth": depth,
        })()
        return json.loads(Path(workflow_tool.scale_surface(args)[1]).read_text())

    # The cube side is the pad factor times the largest physical extent, so the
    # leaf size is independent of how the source mesh happened to be normalized.
    coarse = transform_for((1.0, 2.0, 3.0), 10.0)
    fine = transform_for((0.01, 0.02, 0.03), 10.0)
    assert fine["h_target"] == pytest.approx(coarse["h_target"], rel=1e-6)
    physical_max_extent = 3.0 * coarse["uniform_scale"]
    assert coarse["octree_cube_side"] == pytest.approx(
        workflow_tool.OCTREE_CUBE_PAD * physical_max_extent, rel=1e-6
    )
    assert coarse["octree_cube_side"] == pytest.approx(
        coarse["h_target"] * 128.0, rel=1e-6
    )
    # This is the shape_opt_capture recipe: 3.84 / 2**depth for a 2.0-unit part.
    capture = transform_for((2.0, 2.0, 1.22), 2.0)
    assert capture["h_target"] == pytest.approx(3.84 / 128.0, rel=0.02)


class FakeStructuredAgent:
    model = "fake-model"

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def generate_structured(self, *args, **kwargs):
        self.calls += 1
        return self.value, GenerationResponse("{}", TokenUsage(10, 4, 14))


def test_specialist_cache_avoids_second_api_call(tmp_path):
    calls = object.__new__(SpecialistCalls)
    calls.session_dir = tmp_path
    calls.cache_dir = tmp_path / "api_cache"
    calls.cache_dir.mkdir()
    calls.provider = "groq"
    calls.agent = FakeStructuredAgent({"refined_prompt": "a compact bracket"})
    first, receipt1 = calls.refine_geometry("bracket")
    second, receipt2 = calls.refine_geometry("bracket")
    assert first == second
    assert calls.agent.calls == 1
    assert receipt1.total_tokens == 14
    assert receipt2.cached is True


def test_specialist_repairs_one_malformed_response(tmp_path):
    class RepairingAgent:
        model = "fake-model"

        def __init__(self):
            self.calls = 0

        def generate_structured(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise APIResponseError("bad JSON", raw_text="not json")
            return {"refined_prompt": "a compact bracket"}, GenerationResponse(
                "{}", TokenUsage(8, 3, 11)
            )

    calls = object.__new__(SpecialistCalls)
    calls.session_dir = tmp_path
    calls.cache_dir = tmp_path / "api_cache"
    calls.cache_dir.mkdir()
    calls.provider = "groq"
    calls.agent = RepairingAgent()
    result, receipt = calls.refine_geometry("bracket")
    assert result.refined_prompt == "a compact bracket"
    assert calls.agent.calls == 2
    assert receipt.total_tokens == 11


def test_corrupt_specialist_cache_is_replaced(tmp_path):
    calls = object.__new__(SpecialistCalls)
    calls.session_dir = tmp_path
    calls.cache_dir = tmp_path / "api_cache"
    calls.cache_dir.mkdir()
    calls.provider = "groq"
    calls.agent = FakeStructuredAgent({"refined_prompt": "recovered"})
    # The key is deterministic, so create a valid cache and then corrupt it.
    _, receipt = calls.refine_geometry("bracket")
    cache = calls.cache_dir / f"{receipt.cache_key}.json"
    cache.write_text("broken", encoding="utf-8")
    result, new_receipt = calls.refine_geometry("bracket")
    assert result.refined_prompt == "recovered"
    assert new_receipt.cached is False
    assert calls.agent.calls == 2


def test_api_preview_is_bounded_and_original_untouched(tmp_path):
    original = tmp_path / "large.png"
    Image.new("RGB", (2000, 1500), "white").save(original)
    before = original.read_bytes()
    calls = object.__new__(SpecialistCalls)
    calls.session_dir = tmp_path
    preview = calls._preview(original)
    with Image.open(preview) as image:
        assert max(image.size) <= 768
    assert original.read_bytes() == before


def test_bc_metadata_is_compact_but_keeps_selector_evidence():
    metadata = {
        "bounds": [[0.00000123, 0, 0], [1, 2, 3]],
        "num_clusters": 5, "num_rays": 25, "smoothing": 0.25,
        "cone_angle_deg": 120.0,
        "segments": [
            {"id": 0, "color": "#fff", "faces": 60,
             "area_fraction": 0.6, "bounds": [[0, 0, 0], [1, 2, 3]],
             "centroid": [0.5, 1, 1.5],
             "facing_fraction_45deg": {"+y": 0.75, "-y": 0.0}},
            {"id": 1, "color": "#000", "faces": 40,
             "area_fraction": 0.4, "bounds": [[0, 0, 0], [1, 1, 2]],
             "centroid": [0.5, 0.5, 1.0],
             "facing_fraction_45deg": {"+y": 0.0, "-y": 0.8}},
        ],
    }
    compact = SpecialistCalls._compact_geometry(metadata)
    assert compact["bounds"][0][0] == pytest.approx(0.0)
    assert compact["segmentation_informative"] is True
    assert compact["segments"][1]["facing_fraction_45deg"]["+y"] == 0.0
    assert compact["segmentation"]["num_rays"] == 25


def test_skill_demands_minimal_non_executable_output():
    skill = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "SKILL.md").read_text()
    assert "exactly one JSON object" in skill
    assert "Never request or invent a command" in skill


def test_skill_rates_the_vision_stage_as_intermediate_evidence():
    """The text stage never sees the image, so it must not treat the words as fact."""
    skill = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "SKILL.md").read_text()
    assert "credibility is intermediate" in skill
    assert "weigh it against everything that does" in skill
    # Conflicts resolve against the description, not in its favour.
    assert "follow those and set it aside" in skill
    # Still short enough to ride on every call.
    assert len(skill.split()) < 400


def test_saved_segment_labels_only_reach_selectors_resolved_on_that_surface():
    """Saved ids index the source surface; the solver rebuilds a coarser one."""
    from src.agents_wrapper.fem_workflow.scripts.workflow_tool import _uses_saved_segments

    ids = {"type": "surface_segment", "params": {"segment_ids": [3]}, "children": []}
    seed = {"type": "shape_diameter_segment", "params": {}, "children": [
        {"type": "sphere", "params": {"center": [0, 0, 0], "radius": 0.1}, "children": []}]}
    normal = {"type": "aligned_normal", "children": [],
              "params": {"target_vector": [0, 1, 0], "max_angle_deg": 45}}

    # Both spellings of the id form count, at any depth.
    assert _uses_saved_segments(ids)
    assert _uses_saved_segments(
        {"type": "preview_segments", "params": {"segment_ids": [3]}, "children": []})
    assert _uses_saved_segments({"type": "and", "params": {}, "children": [ids, normal]})
    # A seeded selector is resolved later against the solver's own surface, so it
    # must segment that one afresh rather than carry stale labels.
    assert not _uses_saved_segments(seed)
    assert not _uses_saved_segments({"type": "and", "params": {}, "children": [seed, normal]})
    assert not _uses_saved_segments(
        {"type": "lowest", "params": {"axis": "y", "fraction": 0.1}, "children": []})


def test_reviewed_and_seeded_segment_selectors_have_distinct_contracts():
    import json as json_module
    import re as re_module

    from src.agents_wrapper.fem_workflow.models import SelectorSpec

    reference = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
                 / "boundary_conditions_api.md").read_text()
    assert "`surface_segment` names parts" in reference
    assert "seed is the single child, never a parameter" in reference.replace("\n", " ")

    # Every skeleton the skill teaches must pass schema validation.
    forms = [
        SelectorSpec.model_validate(json_module.loads(
            skeleton.replace("[x,y,z]", "[0.5,0.45,0.5]").replace('"radius":r', '"radius":0.08')
        ))
        for skeleton in re_module.findall(
            r'`(\{"type":"(?:surface_segment|shape_diameter_segment)".*?\})`', reference, re_module.S)
    ]
    assert {"segment_ids"} in [set(form.params) for form in forms]
    assert ["sphere"] in [[c.type for c in form.children] for form in forms]

    for invalid in (
        {"type": "surface_segment", "params": {"segment_ids": [2]},
         "children": [{"type": "sphere", "params": {"center": [0, 0, 0], "radius": 0.1}}]},
        {"type": "shape_diameter_segment", "params": {}, "children": []},
    ):
        with pytest.raises(ValueError):
            SelectorSpec.model_validate(invalid)

    # Sessions saved before the merge still load.
    legacy = SelectorSpec.model_validate(
        {"type": "preview_segments", "params": {"segment_ids": [2, 4]}, "children": []})
    assert legacy.params["segment_ids"] == [2, 4]


def test_bc_prompt_skill_separates_region_limits_from_load_direction():
    """A revision that reads "only the upward portion" as a direction changes the case."""
    reference = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
                 / "boundary_condition_prompt.md").read_text()
    assert "is not a load direction" in reference
    assert "only the upward portion" in reference
    assert "when none is given, give none" in reference
    # A hard word budget is what pressures the model into dropping the qualifier.
    assert "never drop a" in reference


def test_bc_prompt_skill_expands_as_well_as_trims():
    """Revision must complete a terse prompt, not only delete from a verbose one."""
    reference = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
                 / "boundary_condition_prompt.md").read_text()
    assert "Complete what the user implied" in reference
    # It binds each region to the segments it covers, all of them, by name.
    assert "Bind every region to the segments it covers" in reference
    assert "names all of them" in reference
    # A direction given in words is made explicit; none is invented.
    assert "downward (-y)" in reference
    # Expansion is bounded: locations the selector stage can resolve, nothing invented.
    assert "Never add a region, magnitude, unit, direction, component subset" in reference


def test_bc_skill_explains_segmentation_seed_contract():
    reference = (
        Path(__file__).parents[1]
        / "fem_workflow"
        / "skill"
        / "references"
        / "boundary_conditions.md"
    ).read_text()
    assert "does **not** accept `segment_id`" in reference
    assert "does **not** accept `segment_id`, `seed`, or children" in reference
    assert "legacy\n`shape_diameter_segment` expansion selector" in reference
    assert "never recomputes" in reference
    assert '"type": "shape_diameter_segment"' in reference
    assert "locally rendered and resolved preview" in reference
    assert "is authoritative" in reference
    assert '"segment 2 is fixed"' in reference
    assert "geometry.segments[].id" in reference
    assert "All coordinates and distances use the supplied **physical**" in reference
    assert "Spatial selectors test triangle centroids" in reference
    for selector in (
        "bounding_box", "sphere", "cylinder", "half_space", "slab",
        "lowest", "highest", "percentile", "normal", "aligned_normal",
        "curvature", "flat_region", "sharp_feature", "dihedral_angle",
        "flood_fill", "grow_selection", "boundary_edge", "connected_component",
        "and", "or", "subtract", "not",
    ):
        assert f"`{selector}`" in reference
    for parameter in ("num_clusters", "num_rays", "smoothing", "cone_angle_deg"):
        assert parameter in reference

    api_reference = (
        Path(__file__).parents[1]
        / "fem_workflow"
        / "skill"
        / "references"
        / "boundary_conditions_api.md"
    ).read_text()
    # The ceiling is a guard against this file drifting into an essay, not a
    # target: it went from 600 to 700 when "fix the bottom" was coming back as
    # the segment the vision stage had called "front base" -- 2.9% of it faces
    # downward and it spans 63% of the object's height.  Naming the failure
    # concretely in guidance 4 is what fixes it; the same rule stated in the
    # abstract, and the same rule repeated in two other paragraphs, both left
    # the wrong answer in place when measured against the live model.
    assert len(api_reference.split()) < 700
    assert "JSON" in api_reference
    # preview_segments merged into shape_diameter_segment; the API reference
    # teaches only the surviving spelling, in both of its forms.
    assert "preview_segments" not in api_reference
    assert '"segment_ids":[2,4]' in api_reference
    assert "Upward=`+y`, downward=`-y`" in api_reference


def test_bc_value_skill_preserves_values_and_direction_convention():
    reference = (
        Path(__file__).parents[1]
        / "fem_workflow"
        / "skill"
        / "references"
        / "boundary_condition_values.md"
    ).read_text()
    assert "upward is `+y`" in reference
    assert "downward is `-y`" in reference
    # The prompt no longer carries the current values at all: with them present
    # the model echoed them back, making the call a silent no-op.
    assert "current_values" not in reference
    assert "Derive every value from the description" in reference
    assert "Never invent a nonzero magnitude" in reference
    # A total force goes back as a force; the division by area is done in code.
    assert '{"force":[x,y,z]}' in reference
    assert "controller divides it by the region's" in reference
    assert "never divide that by area" in reference
    assert "JSON" in reference


def test_a_total_force_is_spread_over_the_region_area_exactly():
    """98.07 N over 0.0011944 m^2 is 82,110 Pa; the model once wrote 82.1."""
    from src.agents_wrapper.fem_workflow.graph import _as_traction
    from src.agents_wrapper.fem_workflow.models import BCValueInference, NeumannForce

    parsed = BCValueInference.model_validate({
        "dirichlet_values": {"handle": {"components": [0, 1, 2], "value": 0.0}},
        "neumann_values": {"rim": {"force": [0, -98.0665, 0]}, "side": [-2000, 0, 0]},
    })
    assert isinstance(parsed.neumann_values["rim"], NeumannForce)
    traction = _as_traction(parsed.neumann_values["rim"], 0.0011944, "rim")
    assert traction[1] == pytest.approx(-98.0665 / 0.0011944)
    assert _as_traction(parsed.neumann_values["side"], 0.5, "side") == (-2000, 0, 0)
    with pytest.raises(ValueError, match="area is unknown"):
        _as_traction(parsed.neumann_values["rim"], None, "rim")
    # A positional answer of forces is kept in order, not merged into one entry.
    positional = BCValueInference.model_validate({
        "dirichlet_values": [{"components": [0], "value": 0.0}],
        "neumann_values": [{"force": [0, -1, 0]}, {"force": [0, -2, 0]}],
    })
    assert [entry.force[1] for entry in positional.neumann_values] == [-1, -2]


def test_old_region_preview_is_upgraded_with_physical_area(tmp_path):
    import numpy as np
    import trimesh
    from src.agents_wrapper.fem_workflow.graph import _region_statistics
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    surface = tmp_path / "physical.glb"
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.export(surface)
    root = tmp_path / "boundary_conditions"
    root.mkdir()
    (root / "resolved.json").write_text(json.dumps({
        "regions": [{"key": "dirichlet_0", "category": "dirichlet",
                     "name": "fixed", "faces": len(mesh.faces)}],
        "legend": [],
    }), encoding="utf-8")
    np.savez_compressed(root / "resolved.npz",
                        dirichlet_0=np.ones(len(mesh.faces), dtype=bool))
    state = WorkflowState(session_id="test", session_dir=str(tmp_path),
                          surface_path=str(surface))
    statistics = _region_statistics(state)
    assert statistics[0]["area"] == pytest.approx(6.0)
    assert statistics[0]["area_fraction"] == pytest.approx(1.0)
    assert statistics[0]["centroid"] == pytest.approx([0.0, 0.0, 0.0])


def test_geometry_skill_requires_constraint_preserving_prompt():
    reference = (
        Path(__file__).parents[1]
        / "fem_workflow"
        / "skill"
        / "references"
        / "geometry.md"
    ).read_text()
    assert "This is an edit" in reference
    assert "retain every explicit geometric requirement" in reference
    assert "short or generic label" in reference
    assert "already detailed or long, compact it" in reference
    assert "5--15 words" in reference
    for detail in ("part counts", "dimensions", "topology", "holes", "relative placement"):
        assert detail in reference


def test_empty_value_inference_is_rejected_not_silently_preserved():
    """An empty answer used to validate, cache, and change nothing."""
    from pydantic import ValidationError

    from src.agents_wrapper.fem_workflow.models import BCValueInference

    for empty in (
        {"dirichlet_values": [], "neumann_values": [], "body_force": [0.0, 0.0, 0.0]},
        {"dirichlet_values": {}, "neumann_values": {}, "body_force": [0.0, 0.0, 0.0]},
    ):
        with pytest.raises(ValidationError):
            BCValueInference.model_validate(empty)

    # A genuine answer, and a short one the controller still back-fills, both pass.
    assert BCValueInference.model_validate({
        "dirichlet_values": [{"components": [0, 1, 2], "value": 0.0}],
        "neumann_values": [[0.0, -158.9, 0.0]],
        "body_force": [0.0, 0.0, 0.0],
    }).neumann_values == [(0.0, -158.9, 0.0)]

    reference = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
                 / "boundary_condition_values.md").read_text()
    assert "Never omit an entry" in reference
    assert "never return an empty list where regions of that kind were supplied" in reference
    # The one exception, for a proposal that holds no loaded region at all, is
    # pinned by test_a_case_with_no_load_may_answer_with_no_tractions.


def test_api_error_detail_names_the_offending_field():
    """"1 validation error for BCProposal" alone gives the user nothing to act on."""
    from pydantic import ValidationError

    from src.agents_wrapper.fem_workflow.graph import _api_error_detail
    from src.agents_wrapper.fem_workflow.models import BCProposal

    payload = {
        "dirichlet": [{"name": "b", "value": 0.0, "selector": {
            "type": "lowest", "params": {"axis": "y", "fraction": 0.1}}}],
        "neumann": [{"name": "t", "traction": None, "selector": {
            "type": "and", "params": {}, "children": [
                {"type": "highest", "params": {"axis": "y", "fraction": 0.1}}]}}],
    }
    try:
        BCProposal.model_validate(payload)
    except ValidationError as exc:
        detail = _api_error_detail(exc)
    assert "neumann.0.selector" in detail
    assert "and requires 2 children" in detail
    # Non-pydantic failures keep their single-line summary.
    assert _api_error_detail(RuntimeError("boom\nstack")) == "boom"


def test_api_failure_reports_on_the_same_card_and_clears_on_retry(monkeypatch, tmp_path):
    """A failed call must not replace the card the user was filling in."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper import APIRequestError
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, GeometryRefinement, WorkflowState,
    )

    attempts = {"count": 0}

    class FlakySpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def refine_geometry(self, description, overview=""):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise APIRequestError("Groq request failed (RateLimitError)")
            return GeometryRefinement(refined_prompt="compact solid bracket"), APIReceipt(
                purpose="geometry_refinement", provider="groq", model="fake",
                cache_key="test-retry",
            )

    monkeypatch.setattr(graph_module, "SpecialistCalls", FlakySpecialist)
    workflow = WorkflowState(session_id="flaky", session_dir=str(tmp_path), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "flaky"}}
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)

    failed = graph.invoke(Command(resume={
        "description": "bracket", "revise_prompt": True,
    }), config=config)["__interrupt__"][0].value
    # Same card, not a recovery screen, and the typed description survives.
    assert failed["kind"] == "geometry_input"
    assert failed["description"] == "bracket"
    assert "RateLimitError" in failed["api_error"]
    assert failed["api_error_source"] == "revise_geometry_prompt"

    # Clicking again retries with whatever the user edited, and clears the error.
    retried = graph.invoke(Command(resume={
        "description": "bracket with two feet", "revise_prompt": True,
    }), config=config)["__interrupt__"][0].value
    assert retried["kind"] == "geometry_input"
    assert retried["description"] == "compact solid bracket"
    assert retried["api_error"] is None
    assert attempts["count"] == 2


def test_material_inference_round_trips_and_then_advances(monkeypatch, tmp_path):
    """Inference fills the fields in place; applying still moves on to shape-opt."""
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCProposal, DirichletBC, MaterialInference, NeumannBC,
        SelectorSpec, ToolReceipt, WorkflowState,
    )

    seen = {}

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            for key, value in arguments.items():
                if not key.startswith("output"):
                    continue
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}") if path.suffix == ".json" else path.write_bytes(b"stub")
                if name == "stage_asset":
                    path.with_suffix(path.suffix + ".json").write_text(json_module.dumps(
                        {"source_name": path.name, "path": str(path), "sha256": "0" * 64}))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    proposal = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, selector=SelectorSpec(
            type="lowest", params={"axis": "y", "fraction": 0.1}))],
        neumann=[NeumannBC(name="top", traction=(0.0, -1.0, 0.0), selector=SelectorSpec(
            type="highest", params={"axis": "y", "fraction": 0.1}))],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

        def synthesize_bcs(self, description, metadata, montage):
            return proposal, APIReceipt(purpose="bc_synthesis", provider="groq",
                                        model="fake", cache_key="k1")

        def infer_material(self, description, request, modulus_unit, image=None):
            seen.update(description=description, request=request, unit=modulus_unit)
            return MaterialInference(
                youngs_modulus=1.1e10, poisson_ratio=0.35, yield_strength=60e6,
            ), APIReceipt(purpose="material_interpretation", provider="groq",
                          model="fake", cache_key="k2")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    source = tmp_path / "chair.glb"
    source.write_bytes(b"glb")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "material"}}
    workflow = WorkflowState(session_id="material", session_dir=str(tmp_path), model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "an armchair", "asset_path": str(source),
                                 "asset_kind": "surface"}), config=config)
    graph.invoke(Command(resume={"action": "apply_scaling", "up_axis": "y",
                                 "physical_extent": 1.0,
                                 "unit_system": "SI (m-kg-s)"}), config=config)
    settings = {"num_clusters": 5, "num_rays": 25, "smoothing": 0.25, "cone_angle_deg": 120.0}
    graph.invoke(Command(resume={"action": "preview_regions", "description": "fix feet, load top",
                                 "segmentation": settings}), config=config)
    graph.invoke(Command(resume={"action": "approve_regions"}), config=config)
    at_material = graph.invoke(Command(resume={
        "action": "apply_values",
        "dirichlet_values": {"feet": {"components": [0, 1, 2], "value": 0.0}},
        "neumann_values": {"top": [0.0, -1.0, 0.0]}, "body_force": [0.0, 0.0, 0.0],
    }), config=config)
    assert at_material["__interrupt__"][0].value["kind"] == "material_input"

    inferred = graph.invoke(Command(resume={
        "action": "infer_material", "material_description": "oak frame",
    }), config=config)["__interrupt__"][0].value
    # Same card, values filled, and the assumed material named for review.
    assert inferred["kind"] == "material_input"
    assert inferred["youngs_modulus"] == 1.1e10
    assert inferred["poisson_ratio"] == 0.35
    assert inferred["yield_strength"] is None  # Legacy AI/cache strengths are ignored.
    assert inferred["material_inferred"] is True
    assert inferred["material_description"] == "oak frame"   # user's text untouched
    # Sent the physical-scaling call's inputs, in the selected system's unit.
    assert seen == {"description": "an armchair", "request": "oak frame",
                    "unit": "Pa (N/m²)"}

    # The conditional edge must still lead onward once applied.
    applied = graph.invoke(Command(resume={
        "action": "apply_material", "youngs_modulus": 1.1e10, "poisson_ratio": 0.35,
    }), config=config)
    assert applied["__interrupt__"][0].value["kind"] == "shape_opt_settings"
    # A new session's settings card comes up with the current defaults (2026-10-20:
    # gd_lr 8 -> 0.01, zip -> mask, float64), and without the retired Sussman count.
    defaults = applied["__interrupt__"][0].value["defaults"]
    assert defaults == ShapeOptSettings().model_dump(mode="json")
    assert (defaults["gd_lr"], defaults["sdf_bc_fix_method"], defaults["use_mixed_precision"]) \
        == (0.01, "mask", True)
    assert "reinit_n_iters" not in defaults
    material = json.loads((tmp_path / "case" / "material.json").read_text())
    assert material["youngs_modulus"] == 1.1e10 and material["poisson_ratio"] == 0.35


def _rewind_session(monkeypatch, tmp_path):
    """A session driven to its region card, with tools that write numbered stubs.

    Each tool writes files that name the call that made them, so a branch's
    files can be told from another's.  Returns the graph, an ``answer`` that
    resumes the pending card, ``card`` for the current checkpoint, the scaling
    card's checkpoint, the tool calls, and what each ``build_case`` was handed.
    """
    pytest.importorskip("langgraph")

    from langgraph.checkpoint.memory import MemorySaver

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.graph import resume_command
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCProposal, DirichletBC, NeumannBC, SelectorSpec, ToolReceipt,
        WorkflowState,
    )

    calls, built = [], []
    bc = tmp_path / "boundary_conditions"

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append(name)
            if name == "build_case":
                built.append({
                    "traction": json.loads(Path(arguments["bc_json"]).read_text())[
                        "neumann"][0]["traction"],
                    "body_force": json.loads(Path(arguments["loads_json"]).read_text())[
                        "body_force"],
                    "surface": Path(arguments["surface"]).read_text(),
                    "transform": Path(arguments["transform_json"]).read_text(),
                    "segments": Path(arguments["segments_npz"]).read_text(),
                    "resolved": (bc / "resolved.json").read_text(),
                })
            tagged = name in ("segment_surface", "resolve_bcs", "render_segmentation")
            for key, value in arguments.items():
                if not key.startswith("output"):
                    continue
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".json":
                    path.write_text(json.dumps({"call": len(calls), "regions": []})
                                    if tagged else "{}")
                else:
                    path.write_text(f"{name} call {len(calls)}")
                if name == "stage_asset":
                    path.with_suffix(path.suffix + ".json").write_text(json.dumps(
                        {"source_name": path.name, "path": str(path), "sha256": "0" * 64}))
                if name == "scale_surface":
                    # The scale the case is built at, written beside the surface.
                    path.with_suffix(path.suffix + ".transform.json").write_text(
                        json.dumps({"extent": arguments["extent"], "octree_depth": 7}))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    proposal = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, selector=SelectorSpec(
            type="lowest", params={"axis": "y", "fraction": 0.1}))],
        neumann=[NeumannBC(name="top", traction=(0.0, -1.0, 0.0), selector=SelectorSpec(
            type="highest", params={"axis": "y", "fraction": 0.1}))],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

        def synthesize_bcs(self, description, metadata, montage):
            return proposal.model_copy(deep=True), APIReceipt(
                purpose="bc_synthesis", provider="groq", model="fake", cache_key="k1")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    source = tmp_path / "stool.glb"
    source.write_bytes(b"glb")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "rewind"}}
    workflow = WorkflowState(session_id="rewind", session_dir=str(tmp_path), model="fake")

    def answer(value):
        return graph.invoke(resume_command(graph, config, value), config=config)

    def card():
        return graph.get_state(config).config["configurable"]["checkpoint_id"]

    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    answer({"description": "a stool", "asset_path": str(source), "asset_kind": "surface"})
    scaling_card = card()
    answer({"action": "apply_scaling", "up_axis": "y", "physical_extent": 1.0,
            "unit_system": "SI (m-kg-s)"})
    return SimpleNamespace(graph=graph, config=config, answer=answer, card=card,
                           scaling_card=scaling_card, calls=calls, built=built, bc=bc)


def _solve_load_case(session, traction, body_force):
    """Apply values, a material and the default settings; the settings card's checkpoint."""
    session.answer({"action": "apply_values",
                    "dirichlet_values": {"feet": {"components": [0, 1, 2], "value": 0.0}},
                    "neumann_values": {"top": traction}, "body_force": body_force})
    at_settings = session.answer({"action": "apply_material", "youngs_modulus": 1.1e10,
                                  "poisson_ratio": 0.35, "density": 700.0})
    assert at_settings["__interrupt__"][0].value["kind"] == "shape_opt_settings"
    settings_card = session.card()
    solved = session.answer({})
    assert solved["__interrupt__"][0].value["solve_complete"]
    return settings_card


def _preview_regions(session, clusters=5):
    """Preview and approve the regions; the values card's checkpoint."""
    session.answer({"action": "preview_regions", "description": "fix feet, load top",
                    "segmentation": {"num_clusters": clusters}})
    at_values = session.answer({"action": "approve_regions"})
    assert at_values["__interrupt__"][0].value["kind"] == "bc_values"
    return session.card()


LIGHT_LOAD = ([0.0, -784.5, 0.0], [0.0, 0.0, 0.0])
REGION_FILES = ("segments.npz", "segments.json", "segments.glb", "segments.png",
                "segments.source.json", "resolved.npz", "resolved.json", "resolved.glb")


def test_rewind_to_an_earlier_load_case_builds_that_case(monkeypatch, tmp_path):
    """A settings card of an earlier load case solves with that case's values.

    The values and material cards were the only writers of ``proposal.json``
    and ``loads.json``, and a rewind restores the state, not the files: the
    stool 7cf4533ae018, rewound to a settings card of its 784.5 N branch,
    solved the 2353.6 N applied since.
    """
    from src.agents_wrapper.fem_workflow.graph import reopen_card

    session = _rewind_session(monkeypatch, tmp_path)
    values_card = _preview_regions(session)
    heavy = ([0.0, -2353.6, 0.0], [0.0, -9.81, 0.0])
    light_settings = _solve_load_case(session, *LIGHT_LOAD)
    reopen_card(session.graph, session.config, values_card)
    _solve_load_case(session, *heavy)
    cases = [(case["traction"], case["body_force"]) for case in session.built]
    assert cases == [LIGHT_LOAD, heavy]

    # Back to the light case's settings card, and run it again as it was.
    reopened = reopen_card(session.graph, session.config, light_settings)
    assert reopened["__interrupt__"][0].value["kind"] == "shape_opt_settings"
    session.answer({})
    assert (session.built[-1]["traction"], session.built[-1]["body_force"]) == LIGHT_LOAD
    proposal = json.loads((session.bc / "proposal.json").read_text())
    assert proposal["neumann"][0]["traction"] == LIGHT_LOAD[0]


def test_rewind_puts_back_the_branch_region_files(monkeypatch, tmp_path):
    """Segments and resolved regions follow a rewind, as the values do.

    They sit at fixed paths in boundary_conditions/, so a branch segmented
    again left its labels for every other: an earlier branch's settings card
    built its case on the newer segments.npz, and its values card read the
    newer resolved.json's loaded areas.
    """
    from src.agents_wrapper.fem_workflow.graph import reopen_card

    session = _rewind_session(monkeypatch, tmp_path)
    regions_card = session.card()

    def files():
        return {name: (session.bc / name).read_bytes() for name in REGION_FILES}

    first_values = _preview_regions(session)
    first_files = files()
    first_settings = _solve_load_case(session, *LIGHT_LOAD)

    # Before any preview this branch had no segments.  The first branch's stay
    # on disk for the viewer, but are neither offered as this branch's nor
    # taken for them: the same settings segment again.
    reopened = reopen_card(session.graph, session.config, regions_card)
    assert reopened["__interrupt__"][0].value["has_segmentation"] is False
    assert files() == first_files
    _preview_regions(session)
    assert session.calls.count("segment_surface") == 2
    second_files = files()
    # Every file differs but the stamp, which names only the surface.
    assert [name for name in REGION_FILES
            if first_files[name] == second_files[name]] == ["segments.source.json"]
    _solve_load_case(session, *LIGHT_LOAD)
    assert session.built[0]["segments"] != session.built[1]["segments"]

    reopen_card(session.graph, session.config, first_settings)
    assert files() == first_files
    session.answer({})
    assert session.built[-1] == session.built[0]
    # Put back from the versions kept, not made again.
    assert session.calls.count("segment_surface") == 2

    reopen_card(session.graph, session.config, first_values)
    assert files() == first_files


def test_rewind_puts_back_the_branch_surface_and_scale(monkeypatch, tmp_path):
    """The scaled surface and its transform follow a rewind across a rescale.

    Both sit at geometry/physical.glb(.transform.json) whatever the extent, so
    a settings card from before a rescale built its case on the rescaled part.
    """
    from src.agents_wrapper.fem_workflow.graph import reopen_card

    session = _rewind_session(monkeypatch, tmp_path)
    _preview_regions(session)
    first_settings = _solve_load_case(session, *LIGHT_LOAD)

    reopen_card(session.graph, session.config, session.scaling_card)
    session.answer({"action": "apply_scaling", "up_axis": "y", "physical_extent": 2.0,
                    "unit_system": "SI (m-kg-s)"})
    _preview_regions(session)
    _solve_load_case(session, *LIGHT_LOAD)
    first, second = session.built
    assert json.loads(first["transform"])["extent"] == 1.0
    assert json.loads(second["transform"])["extent"] == 2.0
    assert first["surface"] != second["surface"]

    reopen_card(session.graph, session.config, first_settings)
    session.answer({})
    assert session.built[-1] == first


def test_branch_files_follow_the_checkpoint_not_the_disk(tmp_path):
    """Recording and restoring, on the material and case folders a card graph reaches last."""
    from src.agents_wrapper.fem_workflow.graph import (
        _record_branch_files, _restore_branch_files,
    )
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    def write(files):
        for relative, text in files.items():
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / relative).write_text(text)

    def read(names):
        return {name: (tmp_path / name).read_text() for name in names}

    first = {"materials/vomp/manifest.json": "vomp A", "materials/vomp/material_grid.npy": "grid A",
             "case/case.json": "case A", "inputs/original.png": "photo A"}
    write(first)
    # Made an hour before the second branch's files: the viewer opens on the newest.
    written = time.time_ns() - 3600 * 10**9
    os.utime(tmp_path / "materials/vomp/manifest.json", ns=(written, written))
    # Neither a cache keyed by its own inputs nor a write in progress is a branch's.
    write({"boundary_conditions/partfield_features/features.npy": "features",
           "geometry/.physical.glb.123.tmp": "partial"})
    state = WorkflowState(session_id="s", session_dir=str(tmp_path), model="fake")
    _record_branch_files(state)
    recorded = dict(state.session_files)
    assert set(recorded) == set(first)

    second = {"materials/vomp/manifest.json": "vomp B", "materials/vomp/material_grid.npy": "grid B",
              "case/case.json": "case B", "materials/parts/field/manifest.json": "parts B"}
    write(second)
    _record_branch_files(state)
    second_record = dict(state.session_files)

    # The first checkpoint's files come back; one it never had is left alone.
    state.session_files = recorded
    _restore_branch_files(state)
    assert read(first) == first
    # As old as when its branch wrote it, not as new as the rewind.
    assert (tmp_path / "materials/vomp/manifest.json").stat().st_mtime_ns == written
    assert read(["materials/parts/field/manifest.json"]) == {
        "materials/parts/field/manifest.json": "parts B"}
    # And the second's come back from the versions kept of them.
    state.session_files = second_record
    _restore_branch_files(state)
    assert read(second) == second

    # A checkpoint from before files were tracked leaves the disk as it is.
    state.session_files = None
    write({"case/case.json": "case C"})
    _restore_branch_files(state)
    assert read(["case/case.json"]) == {"case/case.json": "case C"}
    # A path outside the session is never written.
    state.session_files = {"../outside.txt": recorded["case/case.json"]}
    _restore_branch_files(state)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_image_calls_split_into_a_vision_then_a_text_stage(monkeypatch, tmp_path):
    """Only the vision stage sees the picture; the text stage gets its words."""
    from src.agents_wrapper.fem_workflow import specialists as specialists_module
    from src.agents_wrapper.fem_workflow.models import ImageDescription

    calls = []

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            self.model = model or "fake"

        def generate_structured(self, prompt, image=None, **kwargs):
            calls.append({"model": self.model, "image": image, "prompt": prompt})
            if "describe_image" in prompt:
                value = {"description": "a wooden armchair standing up the y axis"}
            else:
                value = {"reference_axis": "y", "physical_extent": 0.9}
            return value, type("R", (), {"usage": None})()

    monkeypatch.setattr(specialists_module, "AgentsWrapper", FakeAgent)
    monkeypatch.setitem(specialists_module.PROVIDERS, "groq", lambda: object())
    image = tmp_path / "views.png"
    image.write_bytes(b"png")
    monkeypatch.setattr(specialists_module.SpecialistCalls, "_preview",
                        lambda self, path, max_size=768: path)

    specialist = specialists_module.SpecialistCalls(
        tmp_path, "groq", "text-model", "vision-model")
    # The workflow reads the render once and carries the words on; here that is
    # the two halves of it, run back to back.
    overview = specialist.describe_image("geometry_overview", image)
    inferred, receipt = specialist.infer_physical_scale(
        "an armchair", "about a metre tall", "m", overview)

    assert len(calls) == 2
    vision, text = calls
    # Stage one: the vision model, holding the image, briefed on this case.
    assert vision["model"] == "vision-model" and vision["image"] == image
    assert '"subject"' in vision["prompt"] and '"describe"' in vision["prompt"]
    assert "stands up along" in vision["prompt"]
    # Stage two: the text model, no image, carrying the description forward.
    assert text["model"] == "text-model" and text["image"] is None
    assert "a wooden armchair standing up the y axis" in text["prompt"]
    assert "about a metre tall" in text["prompt"]
    assert inferred.reference_axis == "y"
    # Both calls are billed, not just the one returned.
    assert [r.purpose for r in specialist.receipts] == ["image_description"]
    assert receipt.purpose == "physical_scale_inference"


def test_refinement_sends_no_picture_and_takes_the_reading_it_is_given(
    monkeypatch, tmp_path
):
    """Prompt refinement never looks at anything itself.

    The reference the user attached goes to the generator as conditioning, so
    describing it here paid a vision call to say what the generator could
    already see.  What helps on a revision is what came out last time, and that
    reading was taken once when the geometry appeared.
    """
    from src.agents_wrapper.fem_workflow import specialists as specialists_module

    calls = []

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            self.model = model or "m"

        def generate_structured(self, prompt, **kwargs):
            calls.append({"prompt": prompt, "image": kwargs.get("image"),
                          "model": self.model})
            return {"refined_prompt": "a thicker bracket"}, type("R", (), {"usage": None})()

    monkeypatch.setattr(specialists_module, "AgentsWrapper", FakeAgent)
    monkeypatch.setitem(specialists_module.PROVIDERS, "groq", lambda: object())
    specialist = specialists_module.SpecialistCalls(
        tmp_path, "groq", "text-model", "vision-model")

    specialist.refine_geometry("a bracket")
    assert len(calls) == 1 and calls[0]["image"] is None
    assert calls[0]["model"] == "text-model"
    assert "image_description" not in calls[0]["prompt"]

    specialist.refine_geometry("a bracket", "a thin L-bracket, one arm bent")
    assert len(calls) == 2 and calls[1]["image"] is None
    assert "a thin L-bracket, one arm bent" in calls[1]["prompt"]


def test_the_render_is_read_knowing_what_it_was_asked_to_be(monkeypatch, tmp_path):
    """A crude mesh is ambiguous from some angles; the prompt settles it.

    A blocky armchair came back from the vision stage as "a single L-shaped
    bracket" -- true of two of the six views, where the armrest hides the seat.
    Told what was asked for, the same model and the same render read as a sofa.
    The field is guidance, not evidence: the skill tells the stage to describe
    what is actually there and to say so when it differs.
    """
    from src.agents_wrapper.fem_workflow import specialists as specialists_module

    seen = []

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            self.model = model or "m"

        def generate_structured(self, prompt, **kwargs):
            seen.append(prompt)
            return {"description": "a sofa"}, type("R", (), {"usage": None})()

    monkeypatch.setattr(specialists_module, "AgentsWrapper", FakeAgent)
    monkeypatch.setitem(specialists_module.PROVIDERS, "groq", lambda: object())
    monkeypatch.setattr(specialists_module.SpecialistCalls, "_preview",
                        lambda self, path, max_size=768: path)
    views = tmp_path / "views.png"
    views.write_bytes(b"png")

    calls = specialists_module.SpecialistCalls(
        tmp_path, "groq", "text-model", "vision-model", requested="a single-seat sofa")
    calls.describe_image("geometry_overview", views)
    assert '"requested":"a single-seat sofa"' in seen[0]

    # With nothing recorded the key is not padded with an empty field.
    bare = specialists_module.SpecialistCalls(
        tmp_path / "bare", "groq", "text-model", "vision-model")
    bare.describe_image("geometry_overview", views)
    assert "requested" not in seen[1]

    # The montage hears the request too, but never without the reading beside
    # it: told only what was wanted, it named a seat and a backrest on a
    # C-shaped block that had neither, and those names place supports and loads.
    montage = tmp_path / "segments.png"
    montage.write_bytes(b"png")
    both = specialists_module.SpecialistCalls(
        tmp_path / "both", "groq", "text-model", "vision-model",
        requested="a single-seat sofa", overview="a blocky C-shaped solid")
    both.describe_image("bc_synthesis", montage, {"0": "blue"})
    assert '"object":"a blocky C-shaped solid"' in seen[-1]
    assert '"requested":"a single-seat sofa"' in seen[-1]
    assert specialists_module.CONTEXT_KEYS["bc_synthesis"] == ("object", "requested")
    assert specialists_module.CONTEXT_KEYS["geometry_overview"] == ("requested",)

    reference = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
                 / "image_description.md").read_text()
    # The request supplies the vocabulary and the picture supplies the shape;
    # a purely geometric account of slabs and tabs leaves the stage that places
    # supports and loads with no "seat" or "leg" to bind its words to.
    # Wrapping is not the contract; the sentences are.
    flowed = " ".join(reference.split())
    assert "Read the picture as that object" in flowed
    assert "Its vocabulary is the vocabulary of your answer" in flowed
    assert "not licence to report what is not there" in flowed
    assert "say that part is absent" in flowed
    assert "it settles what is there and what is not; `requested` settles what to call it" in flowed


def test_the_generating_prompt_survives_the_user_retyping_the_box(
    monkeypatch, tmp_path
):
    """Every stage reading one render must send one payload, or each pays.

    `refined_prompt` follows the text box and is overwritten the moment a
    revision is submitted, so keying the reading on it would make a revision
    miss the cache that scaling had just filled.
    """
    pytest.importorskip("langgraph")

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            output = Path(arguments.get("output") or arguments["output_npz"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"stub")
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    workflow = WorkflowState(session_id="p", session_dir=str(tmp_path), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "p"}}
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    after = graph.invoke(
        Command(resume={"description": "a single-seat sofa"}), config=config)

    state = WorkflowState.model_validate(after["workflow"])
    assert state.generation_prompt == "a single-seat sofa"
    # And the render was read on the way through, without anyone pressing Infer.
    assert state.geometry_overview == "a stub reading of the render"
    # Typing something new moves refined_prompt but not what the mesh is.
    state.refined_prompt = "a chunkier sofa"
    assert state.generation_prompt == "a single-seat sofa"


def test_a_snapshot_is_never_readable_half_written(tmp_path):
    """The UI reads these while the workflow rewrites them, from two threads.

    `Path.write_text` truncates first and writes second, so a reader landing in
    between got an empty file and the page died with "Expecting value: line 1
    column 1".  The sidebar reads every session on every rerun, so the window
    was hit often enough to matter during a long step.
    """
    import threading

    from src.agents_wrapper.fem_workflow.models import write_text_atomically

    target = tmp_path / "session.json"
    payloads = [json.dumps({"stage": "bc_segmentation", "pad": "x" * size})
                for size in (2000, 40000)]
    write_text_atomically(target, payloads[0])

    stop = threading.Event()
    failures = []

    def writer():
        index = 0
        while not stop.is_set():
            write_text_atomically(target, payloads[index % 2])
            index += 1

    def reader():
        while not stop.is_set():
            try:
                json.loads(target.read_text(encoding="utf-8"))
            except FileNotFoundError:
                failures.append("missing")
            except json.JSONDecodeError:
                failures.append("truncated")

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    time.sleep(0.5)
    stop.set()
    for thread in threads:
        thread.join()

    assert not failures, f"reader saw {len(failures)} damaged reads: {set(failures)}"
    # And nothing is left lying beside it.
    assert [p.name for p in tmp_path.iterdir()] == ["session.json"]


def test_an_unreadable_snapshot_costs_one_title_not_the_page(tmp_path):
    """One damaged file must not take down a sidebar that lists them all."""
    from src.ui.controller import WorkflowController

    controller = WorkflowController(tmp_path)
    session = tmp_path / "broken"
    session.mkdir()
    (session / "session.json").write_text("", encoding="utf-8")
    assert controller.read_workflow("broken") == {}
    (session / "session.json").write_text("[1, 2]", encoding="utf-8")
    assert controller.read_workflow("broken") == {}


def test_an_uploaded_mesh_may_still_be_named(monkeypatch, tmp_path):
    """An upload needs no prompt to exist, but saying what it is still pays.

    The card offered no description field for a surface, so the vision stage
    read the render with nothing to go on -- and that reading is what sets the
    scale, picks the material, and decides whether a segment is a seat or a
    slab.  Nothing is generated from these words; they are context.
    """
    pytest.importorskip("langgraph")

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    told = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            output = Path(arguments.get("output") or arguments["output_npz"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"stub")
            if name == "stage_asset":
                output.with_suffix(output.suffix + ".json").write_text(
                    json.dumps({"source_name": output.name, "path": str(output),
                                "sha256": "0" * 64})
                )
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            told.append(kwargs.get("requested"))

        def describe_image(self, purpose, image, legend=None):
            return "a reading of the render"

    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    source = tmp_path / "chair.glb"
    source.write_bytes(b"glb")

    def upload(session, description):
        root = tmp_path / session
        root.mkdir()
        workflow = WorkflowState(session_id=session, session_dir=str(root), model="fake")
        graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": session}}
        graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
        done = graph.invoke(Command(resume={
            "description": description, "asset_path": str(source),
            "asset_kind": "surface",
        }), config=config)
        return WorkflowState.model_validate(done["workflow"])

    named = upload("named", "a single-seat sofa")
    assert named.generation_prompt == "a single-seat sofa"
    assert named.geometry_description == "a single-seat sofa"
    assert told[-1] == "a single-seat sofa", "the vision stage is told what it is"
    assert named.geometry_overview == "a reading of the render"

    # Still optional: an upload with no words is not blocked, and no stand-in
    # phrase is invented for it -- one would be prefilled back into the box on
    # the next redraw and would reach the vision stage as the user's own claim.
    bare = upload("bare", "")
    assert bare.generation_prompt == ""
    assert bare.geometry_description == ""
    assert told[-1] == ""


def test_conversations_are_grouped_by_the_day_they_were_worked_on():
    """The sidebar's dividers and its order must come from one timestamp.

    Grouping on a different clock than the sort would scatter a day's
    conversations under several headings of the same name.
    """
    import os
    import tempfile
    import time as clock

    from src.ui.controller import WorkflowController, day_heading

    now = clock.time()
    assert day_heading(now) == "Today"
    assert day_heading(now - 86400) == "Yesterday"
    older = datetime.fromtimestamp(now - 86400 * 4).date()
    assert day_heading(now - 86400 * 4) == f"{older:%b} {older.day}"
    # A year is only worth printing when it is not the current one.
    assert str(datetime.fromtimestamp(now).year) not in day_heading(now - 86400 * 4)
    assert day_heading(now - 86400 * 400).endswith(
        str(datetime.fromtimestamp(now - 86400 * 400).year)
    )

    with tempfile.TemporaryDirectory() as root:
        controller = WorkflowController(Path(root))
        for name, age in (("newest", 0), ("middle", 3600), ("oldest", 86400 * 3)):
            session = Path(root) / name
            session.mkdir()
            (session / "checkpoints.sqlite").write_text("x", encoding="utf-8")
            os.utime(session, (now - age, now - age))
        # A directory with no checkpoint file is not a conversation.
        (Path(root) / "scratch").mkdir()

        dated = controller.dated_sessions()
        assert [name for name, _ in dated] == ["newest", "middle", "oldest"]
        assert [name for name, _ in dated] == controller.sessions()
        assert [day_heading(when) for _, when in dated] == [
            "Today", "Today", day_heading(now - 86400 * 3)
        ]

        # Deleting the only conversation of a day takes its heading with it,
        # because the headings are drawn from whatever is left.
        controller.delete_session("oldest")
        assert controller.sessions() == ["newest", "middle"]
        assert day_heading(now - 86400 * 3) not in [
            day_heading(when) for _, when in controller.dated_sessions()
        ]
        # Deleting what is already gone is not an error.
        controller.delete_session("oldest")
        assert not (Path(root) / "oldest").exists()
        with pytest.raises(ValueError):
            controller.delete_session("../elsewhere")


def test_opening_a_conversation_does_not_reorder_the_list():
    """The list is ordered by work done, not by what was last looked at.

    Resuming a session re-invokes the graph, which rewrites the checkpoint and
    the transcript and touches the directory -- so ordering on any of those put
    whatever had just been read at the top, however old it was.  Only the state
    snapshot is written when a node actually changes something.
    """
    import os
    import tempfile
    import time as clock

    from src.ui.controller import WorkflowController

    now = clock.time()
    with tempfile.TemporaryDirectory() as root:
        controller = WorkflowController(Path(root))
        for name, worked in (("recent", now - 3600), ("stale", now - 86400 * 2)):
            session = Path(root) / name
            session.mkdir()
            for leaf in ("checkpoints.sqlite", "session.json", "ui_history.json"):
                (session / leaf).write_text("{}", encoding="utf-8")
                os.utime(session / leaf, (worked, worked))
            os.utime(session, (worked, worked))
        assert controller.sessions() == ["recent", "stale"]

        # Open the older one: everything a resume writes is touched, except the
        # snapshot, which only a step that changes the state rewrites.
        stale = Path(root) / "stale"
        for leaf in ("checkpoints.sqlite", "ui_history.json"):
            os.utime(stale / leaf, (now, now))
        os.utime(stale, (now, now))
        assert controller.sessions() == ["recent", "stale"], "reading is not working"

        # Doing something in it does move it.
        os.utime(stale / "session.json", (now, now))
        assert controller.sessions() == ["stale", "recent"]

        # A conversation that never got past its first card has no snapshot and
        # is placed by its directory rather than dropped.
        fresh = Path(root) / "fresh"
        fresh.mkdir()
        (fresh / "checkpoints.sqlite").write_text("{}", encoding="utf-8")
        os.utime(fresh, (now + 60, now + 60))
        assert controller.sessions()[0] == "fresh"


def test_the_detailed_settings_are_folded_away():
    """Three blocks of sliders that most runs never touch, behind one line each.

    They are keyed per card, because a conversation keeps its earlier cards on
    screen and two open at once would otherwise share one expander's state.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    for label, key in (
        ("Advanced geometry settings", "advanced_geometry_{card.key}"),
        ("Advanced segmentation settings", "advanced_segmentation_{card.key}"),
        ("Advanced solve settings", "advanced_solve_{card.key}_{run_mode}"),
    ):
        assert f'"{label}",' in source, label
        assert f'key=f"{key}"' in source, key
    # The headings they replaced are gone rather than repeated inside them.
    for heading in ("**Geometry generation settings**", "**Shape-diameter segmentation**",
                    "**Discretization**"):
        assert heading not in source, heading


def test_the_delete_prompt_is_asked_outside_the_list():
    """Built inside the list, the prompt could not be made to look right.

    The list flattens what it holds -- gap zero, every margin stripped, buttons
    clamped to the height of one line, backgrounds forced transparent with
    !important -- so the prompt came out with its buttons on its text and its
    confirm button invisible.  A dialog inherits none of that.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    assert '@st.dialog("Delete conversation")' in source
    assert 'key="dialog_confirm"' in source and 'key="dialog_keep"' in source
    # Nothing is built inside the list to ask the question any more.
    for dead in ("confirm_row_", "confirm_delete_", "st-key-keep_"):
        assert dead not in source, dead
    # Deleting is irreversible, and says so in its own colour.
    red = source.index('[class*="st-key-dialog_confirm"] button {')
    assert "#d32f2f" in source[red : source.index("}", red)]
    # The trash overlaps the title instead of shortening it, so it has to mask
    # what it covers -- all of it.  What matters is that the masking element
    # spans the row top to bottom and is opaque; which element carries the
    # height is an implementation detail and has changed more than once.
    box = source.index('[class*="st-key-history_row_"] [class*="st-key-delete_"] {')
    frame = source[box : source.index("}", box)]
    assert "position: absolute !important" in frame
    assert "top: 0" in frame and "bottom: 0" in frame, "covers the whole row"
    assert "--secondary-background-color" in frame, "opaque, not see-through"
    # And it is hidden and inert until the row is pointed at.
    hidden = source.index(
        '[class*="st-key-history_row_"] [class*="st-key-delete_"] {\n    opacity: 0;'
    )
    assert "pointer-events: none" in source[hidden : source.index("}", hidden)]

    # Nothing is reserved for the trash: the title spans the full width.
    assert "padding-right: 2.2rem" not in source


def test_split_handles_drive_panels_that_exist():
    """The drag handles are JavaScript in a Python string; nothing checks them.

    They find the panels by Streamlit's generated key classes, so renaming a
    panel would leave the handles silently pointing at nothing.  The custom
    properties they write are likewise only useful if the stylesheet reads
    them.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    start = source.index("SPLIT_HANDLES = ")
    opening = source.index('"""', start) + 3
    script = source[opening : source.index('"""', opening)]

    for panel in ("workflow_panel", "visualization_panel", "tool_panel"):
        assert panel in script, panel
        assert f'key="{panel}"' in source, panel
    for prop in ("--workspace-split", "--tools-height"):
        assert f"'{prop}'" in script, prop
        assert f"var({prop}," in source, prop
    # The sidebar's width is set as an inline style, never in the stylesheet:
    # Streamlit's own resize handle writes one too, and an !important rule
    # would win against it and silently stop the sidebar being resizable.
    assert "stSidebarResizeHandle" not in source
    assert '[data-testid="stSidebar"]' in script
    assert '[data-testid="stSidebar"] {' not in source
    # Both gutters come from one symmetric padding rule.
    assert "padding-left: 0.6rem !important" in source
    assert "padding-right: 0.6rem !important" in source
    assert "scrollbar-gutter: stable both-edges" in source
    # The frame has to be rendered to run, so it is moved out of the flow
    # rather than hidden.
    assert 'key="split_handles"' in source
    assert "display: none" not in source[
        source.index('[class*="st-key-split_handles"]') :
        source.index('.workspace-split {')
    ]
    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        assert script.count(opener) == script.count(closer), opener


def test_prompt_revision_speaks_the_segmentation_s_language(monkeypatch, tmp_path):
    """The next stage can only select regions that exist, so name them as found.

    Revision used to edit the supports-and-loads text against a one-line
    description of what the object was meant to be, with no idea what the
    surface had actually been divided into.  "hold the feet" came back as "the
    four legs at the feet"; with the segments in hand it comes back naming them
    one by one, which is what the selector stage can bind.
    """
    from src.agents_wrapper.fem_workflow import specialists as specialists_module

    sent = []

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            self.model = model or "m"

        def generate_structured(self, prompt, **kwargs):
            sent.append(prompt)
            return {"refined_prompt": "fix the legs"}, type("R", (), {"usage": None})()

    monkeypatch.setattr(specialists_module, "AgentsWrapper", FakeAgent)
    monkeypatch.setitem(specialists_module.PROVIDERS, "groq", lambda: object())
    calls = specialists_module.SpecialistCalls(tmp_path, "groq", "text-model")

    calls.refine_bc_prompt("a sofa", "hold the feet", "0: front left leg; 6: seat")
    assert '"segments":"0: front left leg; 6: seat"' in sent[0]
    # Before anything has been segmented there is nothing to say, and the field
    # is left out rather than sent empty.
    calls.refine_bc_prompt("a sofa", "hold the feet")
    assert "segments" not in sent[1]

    reference = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
                 / "boundary_condition_prompt.md").read_text()
    flowed = " ".join(reference.split())
    assert "to locate the user's regions, never to choose new ones" in flowed
    assert "do not add a region because a segment exists" in flowed.lower()
    # Ids change with a re-segmentation; the names outlive it.
    assert "ids change whenever the segmentation is re-run" in flowed
    # A region with no matching segment keeps the user's own words.
    assert "no counterpart among the segments keeps the user's words" in flowed


def test_every_boundary_region_gets_its_own_colour():
    """One blue for all supports hid which of four feet a selector had caught.

    The two roles still read apart -- held is cool, loaded is warm -- so being
    able to tell the regions apart does not cost the distinction that matters
    at a glance.
    """
    from src.agents_wrapper.fem_workflow.scripts.workflow_tool import (
        BC_COLORS, BC_ROLE,
    )

    assert set(BC_COLORS) == {"dirichlet", "neumann"}
    assert BC_ROLE == {"dirichlet": "held", "neumann": "load"}
    for family, swatches in BC_COLORS.items():
        assert len(set(swatches)) == len(swatches), family
        assert len(swatches) >= 6, family
        assert all(s.startswith("#") and len(s) == 7 for s in swatches), family
    # The families must not share a colour, or a support and a load would draw
    # the same on a surface where only one of them is visible.
    assert not set(BC_COLORS["dirichlet"]) & set(BC_COLORS["neumann"])
    # Cool against warm: red dominates every load colour, blue or green every
    # held one.
    def channels(swatch):
        return [int(swatch[offset:offset + 2], 16) for offset in (1, 3, 5)]

    for swatch in BC_COLORS["neumann"]:
        red, green, blue = channels(swatch)
        assert red > green and red > blue, swatch
    for swatch in BC_COLORS["dirichlet"]:
        red, green, blue = channels(swatch)
        assert max(green, blue) > red, swatch
    # And the overlap colour is not one of them, so it stays legible as its own
    # thing when two regions land on the same face.
    assert "#991ab3" not in set(BC_COLORS["dirichlet"]) | set(BC_COLORS["neumann"])


def test_inferred_values_land_on_the_region_they_were_worked_out_for():
    """A positional answer put the right loads on the wrong regions, silently.

    Asked for bare arrays, the model had to map regions named in prose onto
    unnamed slots, and when the description mentioned them in a different order
    than they were sent it got the mapping wrong: 10 kg downwards on the seat
    came back as a backwards load on the backrest, with nothing in the result
    to say so.  Values are now keyed by region name, and a named answer is
    matched by name alone.
    """
    from src.agents_wrapper.fem_workflow.graph import _matched_values
    from src.agents_wrapper.fem_workflow.models import BCValueInference

    class Region:
        def __init__(self, name):
            self.name = name

    regions = [Region("seat_traction"), Region("backrest_plus_z")]
    current = {"seat_traction": (0.0, 0.0, 0.0), "backrest_plus_z": (0.0, 0.0, 0.0)}

    # Named out of order: the names decide, not the order they arrive in.
    matched = _matched_values(
        {"backrest_plus_z": (0.0, 0.0, -121.9), "seat_traction": (0.0, -228.5, 0.0)},
        regions, current, "Neumann",
    )
    assert matched["seat_traction"] == (0.0, -228.5, 0.0)
    assert matched["backrest_plus_z"] == (0.0, 0.0, -121.9)

    # A region left out keeps what it had; it does not inherit a neighbour's.
    partial = _matched_values(
        {"seat_traction": (0.0, -228.5, 0.0)}, regions, current, "Neumann"
    )
    assert partial["backrest_plus_z"] == (0.0, 0.0, 0.0)

    # Names that match nothing are refused rather than quietly used by position,
    # which is what made a mis-keyed answer indistinguishable from a right one.
    with pytest.raises(ValueError, match="name none of the approved regions"):
        _matched_values({"nonsense": (1.0, 2.0, 3.0)}, regions, current, "Neumann")

    # A bare array still works, for a provider that cannot emit an object.
    positional = _matched_values(
        [(0.0, -228.5, 0.0), (0.0, 0.0, -121.9)], regions, current, "Neumann"
    )
    assert positional["seat_traction"] == (0.0, -228.5, 0.0)

    # "One entry per region" is read as a list of one-key objects; the values in
    # it are named and right, and only the wrapping differs.
    wrapped = BCValueInference.model_validate({
        "dirichlet_values": [{"legs": {"components": [0, 1, 2], "value": 0.0}}],
        "neumann_values": [{"seat_traction": [0.0, -228.5, 0.0]},
                           {"backrest_plus_z": [0.0, 0.0, -121.9]}],
    })
    assert wrapped.neumann_values["seat_traction"] == (0.0, -228.5, 0.0)
    assert set(wrapped.dirichlet_values) == {"legs"}
    # A positional Dirichlet list is still a list: its entries carry their own
    # fields rather than a region's name.
    plain = BCValueInference.model_validate({
        "dirichlet_values": [{"components": [0, 1, 2], "value": 0.0}],
        "neumann_values": [[0.0, -228.5, 0.0]],
    })
    assert isinstance(plain.dirichlet_values, list)


def test_the_deformation_slider_is_sized_from_the_solve(tmp_path):
    """True-scale deflection is invisible, so the slider is sized in shape terms.

    The top of it moves the surface by a fixed share of the object however small
    the real displacement was, so the end of the travel means the same thing on
    every part instead of being a number the user has to find by trial.
    """
    import json as json_module

    from src.ui.telemetry import (
        DEFORMATION_REACH, DEFORMATION_STEPS, deformation_scales,
    )

    solve = tmp_path / "solve"
    solve.mkdir()
    largest = 7.1e-4
    (solve / "displacement_magnitude.glb.json").write_text(
        json_module.dumps({"maximum": largest, "minimum": 0.0}), encoding="utf-8"
    )
    scales = deformation_scales(tmp_path, {"physical_extent": 1.0})
    assert len(scales) == DEFORMATION_STEPS
    assert scales[0] == 0.0, "the default is the undeformed shape"
    assert scales == sorted(scales)
    # The far end moves the surface by the reach, whatever the solve produced.
    assert abs(scales[-1] * largest - DEFORMATION_REACH * 1.0) < 1e-9
    # Twice the displacement, half the scale: the picture is the same.
    (solve / "displacement_magnitude.glb.json").write_text(
        json_module.dumps({"maximum": 2 * largest}), encoding="utf-8"
    )
    assert abs(deformation_scales(tmp_path, {"physical_extent": 1.0})[-1]
               - scales[-1] / 2) < 1e-9
    # Nothing to size it from means no slider rather than a broken one.
    assert deformation_scales(tmp_path, {}) == []
    assert deformation_scales(tmp_path, {"physical_extent": 0}) == []
    (solve / "displacement_magnitude.glb.json").write_text(
        json_module.dumps({"maximum": 0.0}), encoding="utf-8"
    )
    assert deformation_scales(tmp_path, {"physical_extent": 1.0}) == []
    assert deformation_scales(tmp_path / "nowhere", {"physical_extent": 1.0}) == []


def test_load_steps_are_listed_in_order_from_the_newest_solve(tmp_path):
    """A ramp leaves one surface per step; the viewer walks them in order."""
    import os
    import time as clock

    from src.agents_wrapper.fem_workflow.assets import newest_surface_vtu

    solve = tmp_path / "solve"
    old_run = solve / "run-old" / "vtu_surface"
    new_run = solve / "run-new" / "vtu_surface"
    for run in (old_run, new_run):
        run.mkdir(parents=True)
    written = "<VTKFile>...</VTKFile>\n"
    (old_run / "surface_t0000.vtu").write_text(written, encoding="utf-8")
    # Written out of order on purpose: the sequence comes from the names, which
    # the writer numbers, not from when the files happened to land.
    for name in ("surface_t0002.vtu", "surface_t0000.vtu", "surface_t0001.vtu"):
        (new_run / name).write_text(written, encoding="utf-8")
    now = clock.time()
    os.utime(old_run, (now - 3600, now - 3600))
    os.utime(new_run, (now, now))

    steps = load_step_surfaces(solve)
    assert [p.name for p in steps] == [
        "surface_t0000.vtu", "surface_t0001.vtu", "surface_t0002.vtu"
    ]
    assert all(p.parent == new_run for p in steps), "only the newest solve"
    assert load_step_surfaces(tmp_path / "nowhere") == []

    # The viewer now follows a run as it writes, and the writer streams a
    # surface into place rather than renaming one in: a file caught mid-save is
    # not a step yet, and offering it would reach the user as an export failure
    # rather than as a picture.
    (new_run / "surface_t0003.vtu").write_text(
        "<VTKFile>\n  <UnstructuredGrid>\n", encoding="utf-8"
    )
    assert [p.name for p in load_step_surfaces(solve)] == [
        "surface_t0000.vtu", "surface_t0001.vtu", "surface_t0002.vtu"
    ]
    assert newest_surface_vtu(solve).name == "surface_t0002.vtu"


def test_surface_export_takes_a_deformation_scale():
    """The scale has to reach the exporter, and the tool has to accept it."""
    from src.agents_wrapper.fem_workflow.monitor import ToolMonitor
    import inspect

    from src.agents_wrapper.fem_workflow.scripts import workflow_tool

    signature = inspect.signature(workflow_tool._export_surface_field)
    assert signature.parameters["deform_scale"].default == 0.0
    source = inspect.getsource(workflow_tool._export_surface_field)
    # Vertices move; colours do not.  They are the field's values, not the
    # positions', and a slider that changed them would be a different picture.
    assert "points = points + displacement * float(deform_scale)" in source
    assert "max_displacement" in source
    monitor_source = inspect.getsource(ToolMonitor.__init__)
    assert '"deform_scale"' in monitor_source


def test_the_deformation_slider_lives_in_the_viewer(tmp_path):
    """Dragging has to be immediate, which rules out a widget on the page.

    A Streamlit slider costs a rerun and a rebuilt asset per step -- export,
    re-encode, reload -- before anything moves.  The displacements ride along
    with the asset instead, and the viewer moves the vertices itself.
    """
    import numpy as np

    from src.ui.telemetry import surface_deformation

    asset = tmp_path / "stress_von_mises.glb"
    asset.write_bytes(b"glb")
    scales = [0.0, 50.0, 100.0]
    assert surface_deformation(asset, scales) is None, "no displacements, no slider"

    payload = np.asarray([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype="<f4")
    asset.with_suffix(asset.suffix + ".disp.bin").write_bytes(payload.tobytes())
    handed = surface_deformation(asset, scales)
    assert handed["max_scale"] == scales[-1]
    assert handed["stops"] == len(scales)
    import base64 as b64
    assert np.frombuffer(
        b64.b64decode(handed["data"]), dtype="<f4"
    ).reshape(-1, 3).tolist() == payload.tolist()
    # A solve with nothing to scale by gets no slider rather than a broken one.
    assert surface_deformation(asset, []) is None

    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    # The control is markup inside the frame, and the page no longer carries one.
    assert 'id="deform-range"' in source
    assert '"Deformation scale"' not in source
    # Moving the vertices is not enough: lighting and culling follow the shape.
    assert "computeVertexNormals()" in source
    assert "computeBoundingSphere()" in source
    # The surface claimed is the one the displacements fit.
    assert "positions.count * 3 === deltas.length" in source
    # Load steps ride along too, so stepping a ramp swaps which frame is shown
    # rather than rebuilding, re-encoding and reloading one per step.
    assert 'id="step-range"' in source
    assert "function showFrame(index)" in source
    assert "loadedFrames" in source
    # Each frame brings its own displacements: a ramp's last step moves furthest,
    # so one shared set would deform every step by the same amount.
    assert "deformDeltas = entry.deltas;" in source


def test_six_view_cameras_use_honest_y_up_bases():
    """Panel names denote camera axes, not whichever axis points screen-right."""
    import numpy as np

    expected_axes = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    views = {label: (np.asarray(eye), np.asarray(up))
             for eye, up, label in workflow_tool._VIEWS}
    assert set(views) == set(expected_axes)

    for label, axis in expected_axes.items():
        eye, up = views[label]
        right = np.cross(up, eye)
        assert np.linalg.norm(eye) == pytest.approx(1.0)
        assert np.linalg.norm(up) == pytest.approx(1.0)
        assert np.linalg.norm(right) == pytest.approx(1.0)
        assert eye @ up == pytest.approx(0.0, abs=1e-12)
        assert eye @ axis > 0.9, f"{label} camera is not nearest {label}"

    elevation = math.radians(20.0)
    for label in ("+x", "-x", "+z", "-z"):
        eye, up = views[label]
        assert eye[1] == pytest.approx(math.sin(elevation))
        assert up @ np.array([0.0, 1.0, 0.0]) == pytest.approx(math.cos(elevation))
    for label in ("+y", "-y"):
        eye, up = views[label]
        assert eye == pytest.approx(expected_axes[label])
        assert up == pytest.approx([0.0, 0.0, 1.0])


def test_vision_briefs_cover_every_image_bearing_call():
    """A missing brief would raise KeyError only once a user supplied an image."""
    from src.agents_wrapper.fem_workflow.specialists import VISION_BRIEFS

    # Prompt refinement is not on this list: the reference the user attached is
    # handed to the generator as conditioning, so describing it in words bought
    # a vision call to tell the generator what it could already see.
    assert set(VISION_BRIEFS) == {"geometry_overview", "bc_synthesis"}
    # Scaling and material once briefed the vision model separately, on whatever
    # the user had supplied.  They now share one reading of the rendered mesh,
    # so the brief has to carry what both of them used to ask for.
    subject, wanted, _ = VISION_BRIEFS["geometry_overview"]
    assert "+x, -x, +y, -y, +z, -z" in subject
    assert "elevated 20 degrees toward +y" in subject
    assert "+/-y axial views instead use +z as screen-up" in subject
    assert "which axis it stands up along" in wanted
    assert "structural bulk as opposed to padding or covering" in wanted
    assert "hints at its real size" in wanted
    for purpose, (subject, wanted, limit) in VISION_BRIEFS.items():
        assert subject and wanted, purpose
        assert 40 <= limit <= 400, purpose
    # The segment montage has to enumerate every id, so it needs the widest budget.
    subject, wanted, limit = VISION_BRIEFS["bc_synthesis"]
    assert limit == max(item[2] for item in VISION_BRIEFS.values())
    assert "+x, -x, +y, -y, +z, -z" in subject
    assert "exact +/-y views use +z as screen-up" in subject
    assert "every segment id" in wanted
    # It asks what each segment is, not where.  Position is measured from the
    # segmentation and handed over in the legend; asking the image for it
    # produced confident, wrong answers -- a backrest read as a base, both
    # armrests placed on the same side.  Relative wording is allowed only to
    # separate two segments that would otherwise read alike.
    assert "what that segment is or what it covers" in wanted
    # A pose on every clause is not a description of what a part is.  Told to
    # tell lookalikes apart, the stage appended one to all twelve -- "left
    # armrest, upright", "seat cushion, flat" -- so the qualifier is now the
    # exception it was meant to be, and a pose is never reportable at all.
    assert "report the part and nothing else" in wanted
    assert "Add a distinguishing word only where two ids would otherwise" in wanted
    assert "Never report an axis, a coordinate, a measurement or a pose" in wanted
    assert "upper or lower" in wanted
    # The measured position is what settles a small segment low in the object.
    assert "let the position overrule what a view seems to show" in wanted
    assert "legend" in subject and "legend" in wanted
    reference = (Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
                 / "image_description.md").read_text()
    assert "You are the only stage that sees the" in reference
    assert "Do not state a numeric dimension" in reference
    assert "one clause per labelled item" in reference


def test_vision_stages_read_the_built_mesh_not_the_users_reference(monkeypatch, tmp_path):
    """A generated shape is judged on itself, not on the photo that asked for it.

    The reference the user uploads is a statement of intent; the generator can
    miss it, and scaling and material then described an object that was never
    built.  Both stages now read the same rendered mesh.
    """
    pytest.importorskip("langgraph")

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        AssetRecord, ToolReceipt, WorkflowState,
    )

    rendered = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            rendered.append((name, arguments["surface"]))
            Path(arguments["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(arguments["output"]).write_bytes(b"png")
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    reference = tmp_path / "inputs" / "original.png"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"png")
    mesh = tmp_path / "geometry" / "generated.glb"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(b"glb")

    state = WorkflowState(
        session_id="views", session_dir=str(tmp_path), geometry_mode="image",
        input_asset=AssetRecord(
            path=str(reference), source_name="chair.png", sha256="0" * 64,
            mime_type="image/png",
        ),
        surface_path=str(mesh),
    )
    views = graph_module._geometry_views(state)
    assert views == tmp_path / "geometry" / "views.png"
    assert rendered == [("render_surface", str(mesh))]

    # Scaling rewrites the mesh; the monitor's fingerprint covers its bytes, so
    # asking again re-runs rather than handing back a view of the old shape.
    state.surface_path = str(tmp_path / "geometry" / "physical.glb")
    Path(state.surface_path).write_bytes(b"scaled")
    graph_module._geometry_views(state)
    assert rendered[-1] == ("render_surface", state.surface_path)

    # Before anything is built there is nothing to look at, and no call is made.
    state.surface_path = None
    assert graph_module._geometry_views(state) is None
    assert len(rendered) == 2


def test_strain_estimate_flags_cases_single_precision_cannot_solve():
    """Calibrated against measured solves, not chosen by eye."""
    from src.agents_wrapper.fem_workflow.graph import (
        FLOAT32_EPSILON, FLOAT32_RESIDUAL_FACTOR, _strain_estimate)
    from src.agents_wrapper.fem_workflow.models import (
        BCProposal, DirichletBC, MaterialParameters, NeumannBC, SelectorSpec,
        WorkflowState)

    def state_for(modulus, traction, body=(0.0, 0.0, 0.0),
                  density=None, acceleration=(0.0, 0.0, 0.0)):
        return WorkflowState(
            session_id="s", session_dir="/tmp/s", unit_system="SI (m-kg-s)",
            physical_extent=1.0, body_force=body, acceleration=acceleration,
            material=MaterialParameters(
                unit_system="SI (m-kg-s)", youngs_modulus=modulus,
                poisson_ratio=0.3, density=density),
            bc_proposal=BCProposal(
                dirichlet=[DirichletBC(name="d", value=0.0, selector=SelectorSpec(
                    type="lowest", params={"axis": "y", "fraction": 0.1}))],
                neumann=[NeumannBC(name="n", traction=traction, selector=SelectorSpec(
                    type="highest", params={"axis": "y", "fraction": 0.1}))]),
        )

    floor = FLOAT32_RESIDUAL_FACTOR * FLOAT32_EPSILON / 1.0e-2
    # The case that stalled at rel 3.7e+01 in float32.
    assert _strain_estimate(state_for(1e9, (0.0, -91.86, 0.0))) < floor
    # The case that converged at rel 4.0e-03 in float32.
    assert _strain_estimate(state_for(1e6, (0.0, -257.21, 0.0))) > floor
    # A body force over the part counts toward the strain as well.
    assert _strain_estimate(state_for(1e9, (0.0, 0.0, 0.0), body=(0.0, -1e6, 0.0))) == 1e-3
    # Self-weight enters through rho*a, while direct body force remains additive.
    assert _strain_estimate(state_for(
        1e6, (0.0, 0.0, 0.0), body=(0.0, -1.0, 0.0),
        density=1000.0, acceleration=(0.0, -9.999, 0.0),
    )) == pytest.approx(1.0e-2)
    # No material or no load yet: nothing to estimate from, so no claim is made.
    assert _strain_estimate(WorkflowState(session_id="s", session_dir="/tmp/s")) is None
    assert _strain_estimate(state_for(1e9, (0.0, 0.0, 0.0))) is None


def test_material_inference_returns_parameters_only():
    """Parameters only: the material stays implicit and is never returned."""
    from pydantic import ValidationError

    from src.agents_wrapper.fem_workflow.models import MaterialInference

    assert MaterialInference(youngs_modulus=1.1e10, poisson_ratio=0.35).youngs_modulus
    # A name, or any other field, is not part of the contract.
    with pytest.raises(ValidationError):
        MaterialInference(youngs_modulus=1.1e10, poisson_ratio=0.35,
                          material="oak, along grain")
    # 0.5 is incompressible; the solver cannot use it.
    with pytest.raises(ValidationError):
        MaterialInference(youngs_modulus=1e8, poisson_ratio=0.5)
    with pytest.raises(ValidationError):
        MaterialInference(youngs_modulus=0.0, poisson_ratio=0.3)


def test_parts_take_the_uniform_stages_values_and_answer_in_its_fields():
    """A part is valued by the same rules and table as a whole object."""
    from pydantic import ValidationError

    values = SpecialistCalls._instructions("material_values.md", shared_preamble=False)
    uniform = SpecialistCalls._instructions("material.md", "material_values.md")
    parts = SpecialistCalls._instructions(
        "part_material.md", "material_values.md", shared_preamble=False
    )
    assert uniform.endswith(values) and parts.endswith(values)
    assert "Return no material name" in parts
    answer = PartMaterials.model_validate({"parts": [
        {"id": 0, "youngs_modulus": 1.5e8, "poisson_ratio": 0.4, "density": 900.0},
        {"id": 1, "youngs_modulus": 2.0e11, "poisson_ratio": 0.29, "density": 7850.0},
    ]})
    assert [part.id for part in answer.parts] == [0, 1]
    with pytest.raises(ValidationError):
        PartMaterials.model_validate({"parts": [
            {"id": 0, "youngs_modulus": 1e8, "poisson_ratio": 0.5, "density": 900.0}]})
    with pytest.raises(ValidationError, match="named once"):
        PartMaterials.model_validate({"parts": [
            {"id": 0, "youngs_modulus": 1e8, "poisson_ratio": 0.3, "density": 900.0}] * 2})


def test_material_skill_carries_a_usable_reference_table():
    """The table is the point of the skill; check it parses and is sane.

    Every row is a band rather than a value.  Point values read as a menu: the
    model returned them verbatim, so a sofa came back as the foam row's 1e6 Pa.
    A band has no number to copy, which forces a value chosen for the object.
    """
    reference = SpecialistCalls._instructions(
        "material.md", "material_values.md", shared_preamble=False
    )
    assert "1 psi = 6895 Pa" in reference
    assert "strictly below 0.5" in reference
    assert "not a value to copy" in reference
    assert "Return no material name" in reference
    rows = [
        line.split("|")[1:4] for line in reference.splitlines()
        if line.startswith("| ") and "---" not in line and "material |" not in line
    ]
    assert len(rows) >= 20

    def band(cell):
        low, high = (float(part) for part in cell.split("\u2013"))
        assert low < high, cell
        return low, high

    for name, modulus, ratio in rows:
        assert name.strip()
        low, high = band(modulus)
        assert 1.0e4 < low and high < 1.0e12, name
        low, high = band(ratio)
        assert 0.0 < low and high < 0.5, name
    names = [name.strip() for name, _, _ in rows]
    assert "structural steel" in names and "polyurethane foam" in names


def test_preview_segmentation_describes_the_segments_for_the_card(monkeypatch, tmp_path):
    """Previewing segments reads the montage straight away.

    The description is what the card shows, and having it cached is what leaves
    the later region synthesis a single text call.
    """
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    rendered = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            rendered.append(name)
            for key, value in arguments.items():
                if key.startswith("output"):
                    path = Path(value)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix == ".json":
                        path.write_text(json_module.dumps({"segments": []}))
                    else:
                        path.write_bytes(b"stub")
                    if name == "stage_asset":
                        path.with_suffix(path.suffix + ".json").write_text(
                            json_module.dumps(
                                {"source_name": path.name, "path": str(path),
                                 "sha256": "0" * 64}
                            )
                        )
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    asked = {}

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

        @staticmethod
        def _segment_legend(metadata):
            return {"0": "blue"}

        def describe_image(self, purpose, image, legend=None):
            asked["purpose"] = purpose
            asked["legend"] = legend
            return "0: seat. 1: leg."

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    source = tmp_path / "chair.glb"
    source.write_bytes(b"glb")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "segdesc"}}
    ready_partfield = SimpleNamespace(
        ready=True,
        model_dump=lambda: {
            "ready": True, "environment": "mechfoundry-partfield", "summary": "ready"
        },
    )
    monkeypatch.setattr(graph_module, "partfield_status", lambda: ready_partfield)
    workflow = WorkflowState(session_id="segdesc", session_dir=str(tmp_path), model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "an armchair", "asset_path": str(source),
                                 "asset_kind": "surface"}), config=config)
    first_bc_card = graph.invoke(Command(resume={
        "action": "apply_scaling", "up_axis": "y", "physical_extent": 1.0,
        "unit_system": "SI (m-kg-s)",
    }), config=config)
    assert first_bc_card["__interrupt__"][0].value["segmentation"]["method"] == "partfield"

    previewed = graph.invoke(Command(resume={
        "action": "preview_segmentation",
        "description": "",
        "segmentation": {"num_clusters": 5, "num_rays": 25,
                         "smoothing": 0.25, "cone_angle_deg": 120.0},
    }), config=config)
    card = previewed["__interrupt__"][0].value
    assert card["kind"] == "bc_input"
    assert card["segment_description"] == "0: seat. 1: leg."
    # Read with the montage's own brief, and told which colour each id is.
    assert asked["purpose"] == "bc_synthesis"
    assert asked["legend"] == {"0": "blue"}
    # The montage has to exist before it can be read.
    assert "render_segmentation" in rendered


def test_revise_bc_prompt_returns_to_bc_input_and_invalidates_regions(monkeypatch, tmp_path):
    """The optional call rewrites the text and drops regions built from the old one."""
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCPromptRefinement, BCProposal, DirichletBC, NeumannBC,
        SelectorSpec, ToolReceipt, WorkflowState,
    )

    seen = {}

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            for key, value in arguments.items():
                if not key.startswith("output"):
                    continue
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}") if path.suffix == ".json" else path.write_bytes(b"stub")
                if name == "stage_asset":
                    path.with_suffix(path.suffix + ".json").write_text(json_module.dumps(
                        {"source_name": path.name, "path": str(path), "sha256": "0" * 64}
                    ))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    proposal = BCProposal(
        dirichlet=[DirichletBC(
            name="feet",
            selector=SelectorSpec(type="lowest", params={"axis": "y", "fraction": 0.1}),
            value=0.0,
        )],
        neumann=[NeumannBC(
            name="top",
            selector=SelectorSpec(type="highest", params={"axis": "y", "fraction": 0.1}),
            traction=(0.0, -1.0, 0.0),
        )],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

        def synthesize_bcs(self, description, metadata, montage):
            return proposal, APIReceipt(purpose="bc_synthesis", provider="groq",
                                        model="fake", cache_key="test-bc-synth")

        def refine_bc_prompt(self, geometry, description, segments=""):
            seen["geometry"] = geometry
            seen["description"] = description
            seen["segments"] = segments
            return BCPromptRefinement(
                refined_prompt="fix four feet fully; 2 MPa downward traction on top face"
            ), APIReceipt(purpose="bc_prompt_refinement", provider="groq",
                          model="fake", cache_key="test-bc-prompt")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    source = tmp_path / "chair.glb"
    source.write_bytes(b"glb")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "bcprompt"}}
    workflow = WorkflowState(session_id="bcprompt", session_dir=str(tmp_path), model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "an armchair", "asset_path": str(source),
                                 "asset_kind": "surface"}), config=config)
    at_bc = graph.invoke(Command(resume={
        "action": "apply_scaling", "up_axis": "y", "physical_extent": 1.0,
        "unit_system": "SI (m-kg-s)",
    }), config=config)
    assert at_bc["__interrupt__"][0].value["kind"] == "bc_input"

    original = "um so hold the feet and push down on top"
    settings = {"num_clusters": 7, "num_rays": 25,
                "smoothing": 0.25, "cone_angle_deg": 120.0}
    previewed = graph.invoke(Command(resume={
        "action": "preview_regions", "description": original, "segmentation": settings,
    }), config=config)
    assert previewed["__interrupt__"][0].value["has_regions"] is True

    revised = graph.invoke(Command(resume={
        "action": "revise_prompt", "description": original, "segmentation": settings,
    }), config=config)
    prompt = revised["__interrupt__"][0].value
    assert prompt["kind"] == "bc_input"
    assert prompt["description"] == (
        "fix four feet fully; 2 MPa downward traction on top face"
    )
    # The object gives region words their meaning, so what the render was read
    # as is sent with the text -- not the description box, which for a CAD
    # generator holds construction steps rather than what the object is.
    assert seen["geometry"] == "a stub reading of the render"
    assert seen["description"] == original
    # And what the segments were found to be, so the wording it produces is
    # wording the stage that resolves regions can actually bind to.
    assert "segments" in seen
    # Slider edits submitted with the revision survive the round trip.
    assert prompt["segmentation"]["num_clusters"] == 7
    # Regions synthesized from the old wording must not stay approvable.
    assert prompt["has_regions"] is False


def test_revise_regions_edits_synthesized_selectors_and_only_those(monkeypatch, tmp_path):
    """A revision corrects what the synthesis made; it never stands in for the synthesis."""
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCProposal, DirichletBC, NeumannBC, SelectorSpec, ToolReceipt, WorkflowState,
    )

    resolved = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            if name == "resolve_bcs":
                resolved.append(json_module.loads(Path(arguments["bc_json"]).read_text()))
            for key, value in arguments.items():
                if not key.startswith("output"):
                    continue
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}") if path.suffix == ".json" else path.write_bytes(b"stub")
                if name == "stage_asset":
                    path.with_suffix(path.suffix + ".json").write_text(json_module.dumps(
                        {"source_name": path.name, "path": str(path), "sha256": "0" * 64}
                    ))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    synthesized = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, selector=SelectorSpec(
            type="lowest", params={"axis": "y", "fraction": 0.1}))],
        neumann=[NeumannBC(name="top", traction=(0.0, -1.0, 0.0), selector=SelectorSpec(
            type="highest", params={"axis": "y", "fraction": 0.1}))],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

        def _segment_legend(self, metadata):
            return ""

        def synthesize_bcs(self, description, metadata, montage):
            return synthesized, APIReceipt(purpose="bc_synthesis", provider="groq",
                                           model="fake", cache_key="test-bc-revise")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    source = tmp_path / "block.glb"
    source.write_bytes(b"glb")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "bcrevise"}}
    workflow = WorkflowState(session_id="bcrevise", session_dir=str(tmp_path), model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "a block", "asset_path": str(source),
                                 "asset_kind": "surface"}), config=config)
    graph.invoke(Command(resume={"action": "apply_scaling", "up_axis": "y", "physical_extent": 1.0,
                                 "unit_system": "SI (m-kg-s)"}), config=config)

    def fork():
        """Drop a refused answer and return to the card, as the app does."""
        graph.invoke(None, config=graph.update_state({"configurable": {
            **graph.get_state(config).config["configurable"], "checkpoint_ns": ""}}, None))

    narrower = SelectorSpec(type="lowest", params={"axis": "y", "fraction": 0.02})
    revision = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, selector=narrower)],
        neumann=[NeumannBC(name="top", traction=(0.0, -1.0, 0.0),
                           selector=synthesized.neumann[0].selector)],
    ).model_dump(mode="json")
    # Before any synthesis there is nothing to revise: the call is not skippable.
    with pytest.raises(Exception, match="preview_regions first"):
        graph.invoke(Command(resume={"action": "revise_regions", "proposal": revision}), config=config)
    fork()
    card = graph.invoke(Command(resume={"action": "preview_regions", "description": "hold the feet, push the top"}),
                        config=config)["__interrupt__"][0].value
    assert card["has_regions"] is True and card["can_revise"] is True and len(resolved) == 1
    # It keeps the synthesized regions: no renamed, added or dropped region.
    renamed = {**revision, "dirichlet": [{**revision["dirichlet"][0], "name": "base"}]}
    with pytest.raises(Exception, match="keeps the synthesized support regions"):
        graph.invoke(Command(resume={"action": "revise_regions", "proposal": renamed}), config=config)
    fork()
    added = {**revision, "neumann": [*revision["neumann"], {**revision["neumann"][0], "name": "side"}]}
    with pytest.raises(Exception, match="keeps the synthesized load regions"):
        graph.invoke(Command(resume={"action": "revise_regions", "proposal": added}), config=config)
    fork()
    # A selector edit is re-resolved by the same tool and recorded beside the proposal.
    card = graph.invoke(Command(resume={"action": "revise_regions", "proposal": revision}),
                        config=config)["__interrupt__"][0].value
    assert card["kind"] == "bc_input" and card["has_regions"] is True
    assert len(resolved) == 2 and resolved[-1]["dirichlet"][0]["selector"]["params"]["fraction"] == 0.02
    root = tmp_path / "boundary_conditions"
    record = json_module.loads((root / "revision.json").read_text())
    assert [r["name"] for r in record["regions"]] == ["feet"]
    assert record["regions"][0]["synthesized"]["params"]["fraction"] == 0.1
    assert record["regions"][0]["revised"]["params"]["fraction"] == 0.02
    assert json_module.loads((root / "synthesized.json").read_text())["dirichlet"][0]["selector"]["params"]["fraction"] == 0.1
    # Only the selectors change: a value or a traction sent with the revision is not taken.
    retraction = {**revision, "neumann": [{**revision["neumann"][0], "traction": [0.0, -9.0, 0.0]}]}
    graph.invoke(Command(resume={"action": "revise_regions", "proposal": retraction}), config=config)
    assert resolved[-1]["neumann"][0]["traction"] == [0.0, -1.0, 0.0]
    # New wording makes the synthesized regions another description's: no revision of them.
    graph.invoke(Command(resume={"action": "preview_segmentation", "description": "something else"}),
                 config=config)
    with pytest.raises(Exception, match="preview_regions first"):
        graph.invoke(Command(resume={"action": "revise_regions", "proposal": revision}), config=config)


def test_infer_scale_returns_to_physics_step_with_inferred_values(monkeypatch, tmp_path):
    """The optional call fills both fields and hands them back for review."""
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, PhysicalScaleInference, ToolReceipt, WorkflowState,
    )

    executed = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            executed.append(name)
            output = Path(arguments.get("output") or arguments["output_npz"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"stub")
            if name == "stage_asset":
                output.with_suffix(output.suffix + ".json").write_text(json_module.dumps(
                    {"source_name": output.name, "path": str(output), "sha256": "0" * 64}
                ))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

        def infer_physical_scale(self, description, request, length_unit, overview=""):
            assert description == "an armchair"
            assert request == "a normal armchair"
            assert length_unit == "m"
            # An uploaded surface is not a picture of the object; it was
            # rendered and read on the way in, and what arrives here is words.
            assert overview == "a stub reading of the render"
            return PhysicalScaleInference(
                reference_axis="y", physical_extent=0.9
            ), APIReceipt(
                purpose="physical_scale_inference", provider="groq", model="fake",
                cache_key="test-scale",
            )

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    source = tmp_path / "chair.glb"
    source.write_bytes(b"glb")
    workflow = WorkflowState(session_id="scale", session_dir=str(tmp_path), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "scale"}}
    started = graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    assert started["__interrupt__"][0].value["kind"] == "geometry_input"

    physics = graph.invoke(Command(resume={
        "description": "an armchair", "asset_path": str(source), "asset_kind": "surface",
    }), config=config)
    assert physics["__interrupt__"][0].value["kind"] == "geometry_physics"

    inferred = graph.invoke(Command(resume={
        "action": "infer_scale", "unit_system": "SI (m-kg-s)",
        "scale_description": "a normal armchair",
    }), config=config)
    review = inferred["__interrupt__"][0].value
    assert review["kind"] == "geometry_physics"
    assert review["up_axis"] == "y"
    assert review["physical_extent"] == 0.9
    assert review["scale_description"] == "a normal armchair"
    assert "render_surface" in executed

    # Applying still scales and advances: the pre-existing contract is unchanged.
    applied = graph.invoke(Command(resume={
        "action": "apply_scaling", "up_axis": "y", "physical_extent": 0.9,
        "unit_system": "SI (m-kg-s)",
    }), config=config)
    assert applied["__interrupt__"][0].value["kind"] == "bc_input"
    assert "scale_surface" in executed


def test_geometry_physics_accepts_answers_without_an_action(monkeypatch, tmp_path):
    """The CLI's existing answer shape must keep working unchanged."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    workflow = WorkflowState(
        session_id="legacy", session_dir=str(tmp_path), model="fake",
        geometry_mode="surface", surface_path=str(tmp_path / "s.glb"),
    )
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "legacy"}}
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "a chair", "revise_prompt": False}),
                 config=config)
    graph.invoke(Command(resume={"up_axis": "z", "physical_extent": 0.25,
                                 "unit_system": "SI (m-kg-s)"}), config=config)
    workflow_state = graph.get_state(config).values["workflow"]
    assert workflow_state["up_axis"] == "z"
    assert workflow_state["physical_extent"] == 0.25


def test_cli_revision_branch_returns_revised_geometry_input(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import APIReceipt, WorkflowState

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

        def refine_geometry(self, description, overview=""):
            assert description == "bracket"
            return GeometryRefinement(refined_prompt="compact solid mounting bracket"), APIReceipt(
                purpose="geometry_refinement", provider="groq", model="fake",
                cache_key="test-revision",
            )

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    workflow = WorkflowState(
        session_id="cli-revision", session_dir=str(tmp_path), model="fake"
    )
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "cli-revision"}}
    first = graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    assert first["__interrupt__"][0].value["kind"] == "geometry_input"

    revised = graph.invoke(Command(resume={
        "description": "bracket", "revise_prompt": True,
    }), config=config)
    prompt = revised["__interrupt__"][0].value
    assert prompt["kind"] == "geometry_input"
    assert prompt["description"] == "compact solid mounting bracket"


def test_workflow_defaults_match_reference_examples():
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    state = WorkflowState(session_id="test", session_dir="/tmp/test")
    assert state.octree_depth == 7
    assert state.segmentation_num_clusters == 5
    assert state.segmentation_method == "shape_diameter"
    assert state.segmentation_num_rays == 25
    assert state.segmentation_smoothing == 0.25
    assert state.segmentation_cone_angle_deg == 120.0
    assert state.shape_opt.run_mode == "forward"
    assert state.shape_opt.gd_lr == 0.01
    assert state.shape_opt.max_optimize_iters == 101
    assert state.shape_opt.output_per_optimize_step == 1
    assert state.shape_opt.optimize_relative_tol == 1.0e-9
    assert state.shape_opt.relative_tol == 1.0e-2
    assert state.shape_opt.absolute_tol == 1.0e-5
    assert state.shape_opt.max_newton_iters == 20
    assert state.shape_opt.use_mixed_precision is True
    assert state.shape_opt.sdf_bc_fix_method == "mask"
    assert state.shape_opt.target_volume_fraction == 0.5
    assert state.shape_opt.shape_opt_w_vol == 5.0
    assert state.shape_opt.remesh_init_octree_depth == 4
    assert state.shape_opt.remesh_octree_depth == 7
    # The solver never read a quadrature degree, so the card no longer offers
    # one; a stored setting that has it still loads.
    assert not hasattr(state.shape_opt, "quadrature_degree")
    assert ShapeOptSettings.model_validate({"quadrature_degree": 3}).remesh_octree_depth == 7
    # Likewise the retired BC box size.
    assert not hasattr(state.shape_opt, "bc_bbox_size_factor")
    assert ShapeOptSettings.model_validate({"bc_bbox_size_factor": 1.5}).remesh_octree_depth == 7
    # And the Sussman iteration count, which the marching-cubes reinit the
    # workflow always selects never read: gone from the card, still loads.
    assert "reinit_n_iters" not in ShapeOptSettings.model_fields
    old = ShapeOptSettings.model_validate({"preset_version": 2, "reinit_n_iters": 40})
    assert "reinit_n_iters" not in old.model_dump()


def test_shape_opt_settings_are_validated():
    settings = ShapeOptSettings(run_mode="shape_optimization", max_optimize_iters=25)
    assert settings.max_optimize_iters == 25
    # above 1 the target is a cap the shape may grow up to (the solver normalises
    # by |1 - f|); the card accepts it up to 2
    assert ShapeOptSettings(target_volume_fraction=1.5).target_volume_fraction == 1.5
    assert ShapeOptSettings(target_volume_fraction=2.0).target_volume_fraction == 2.0
    with pytest.raises(ValueError):
        ShapeOptSettings(target_volume_fraction=2.1)
    with pytest.raises(ValueError):
        ShapeOptSettings(target_volume_fraction=0.0)
    with pytest.raises(ValueError):
        ShapeOptSettings(max_newton_iters=0)
    with pytest.raises(ValueError, match="initial octree depth"):
        ShapeOptSettings(remesh_init_octree_depth=8, remesh_octree_depth=7)
    migrated = ShapeOptSettings.model_validate({
        "relative_tol": 1.0e-6, "absolute_tol": 1.0e-8,
        "max_newton_iters": 50, "use_mixed_precision": True,
        "gd_lr": 1.0, "max_optimize_iters": 2000,
        "output_per_optimize_step": 10, "optimize_relative_tol": 2.0e-3,
    })
    # An untouched pre-capture card comes up with the current defaults.
    assert migrated.gd_lr == 0.01
    assert migrated.use_mixed_precision is True
    assert migrated.max_optimize_iters == 101
    assert migrated.preset_version == 2


def test_build_case_carries_shape_opt_settings(tmp_path):
    import numpy as np

    surface = tmp_path / "surface.glb"
    surface.write_bytes(b"placeholder")
    bc_json = tmp_path / "bc.json"
    bc_json.write_text(json.dumps({
        "dirichlet": [{"name": "fixed", "selector": {
            "type": "lowest", "params": {"axis": "z", "fraction": 0.1}}}],
        "neumann": [{"name": "load", "selector": {
            "type": "highest", "params": {"axis": "z", "fraction": 0.1}},
            "traction": [0, 0, -1]}],
    }))
    material_json = tmp_path / "material.json"
    material_json.write_text(MaterialParameters(
        unit_system="SI (m-kg-s)", youngs_modulus=1e6, poisson_ratio=0.3
    ).model_dump_json())
    settings_json = tmp_path / "shape_opt.json"
    settings_json.write_text(ShapeOptSettings(
        run_mode="shape_optimization", max_optimize_iters=25
    ).model_dump_json())
    transform_json = tmp_path / "transform.json"
    transform_json.write_text(json.dumps({
        "octree_depth": 7, "h_target": 0.01, "mesh_center": [0, 0, 0]}))
    output = tmp_path / "case.json"
    segments_npz = tmp_path / "segments.npz"
    np.savez_compressed(segments_npz, segment_ids=np.array([0], dtype=np.int64))
    args = type("Args", (), {
        "surface": surface, "bc_json": bc_json, "material_json": material_json,
        "segments_npz": segments_npz,
        "settings_json": settings_json, "transform_json": transform_json,
        "output": output, "solve_output": tmp_path / "solve", "octree_depth": 7,
    })()
    workflow_tool.build_case(args)
    case = json.loads(output.read_text())
    assert case["shape_opt"]["run_mode"] == "shape_optimization"
    assert case["shape_opt"]["max_optimize_iters"] == 25
    assert case["segments_npz"] == str(segments_npz.resolve())


def test_build_case_carries_adaptive_material_and_separate_loads(tmp_path):
    surface = tmp_path / "surface.glb"
    surface.write_bytes(b"placeholder")
    artifact = tmp_path / "materials" / "vomp"
    artifact.mkdir(parents=True)
    bc_json = tmp_path / "bc.json"
    bc_json.write_text(json.dumps({
        "dirichlet": [{"name": "fixed", "selector": {
            "type": "lowest", "params": {"axis": "z", "fraction": 0.1}}}],
        "neumann": [],
    }))
    material_json = tmp_path / "material.json"
    material_json.write_text(MaterialParameters(
        unit_system="SI (m-kg-s)", mode="vomp", artifact=str(artifact),
        approved=True,
    ).model_dump_json())
    loads_json = tmp_path / "loads.json"
    loads_json.write_text(json.dumps({
        "body_force": [1.0, 2.0, 3.0], "acceleration": [0.0, -9.81, 0.0],
    }))
    settings_json = tmp_path / "shape_opt.json"
    settings_json.write_text(ShapeOptSettings(run_mode="forward").model_dump_json())
    transform_json = tmp_path / "transform.json"
    transform_json.write_text(json.dumps({
        "octree_depth": 7, "h_target": 0.01, "mesh_center": [0, 0, 0]}))
    output = tmp_path / "case.json"
    args = type("Args", (), {
        "surface": surface, "bc_json": bc_json, "material_json": material_json,
        "loads_json": loads_json, "segments_npz": None,
        "settings_json": settings_json, "transform_json": transform_json,
        "output": output, "solve_output": tmp_path / "solve", "octree_depth": 7,
    })()
    workflow_tool.build_case(args)
    case = json.loads(output.read_text())
    assert case["version"] == 2
    assert case["material"]["mode"] == "vomp"
    assert case["material"]["artifact"] == str(artifact)
    assert case["loads"] == {
        "body_force": [1.0, 2.0, 3.0], "acceleration": [0.0, -9.81, 0.0],
    }


def test_workflow_accepts_direct_surface_mode(tmp_path):
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    state = WorkflowState(session_id="test", session_dir=str(tmp_path),
                          geometry_mode="surface")
    assert state.geometry_mode == "surface"


def test_ui_controller_upload_and_history_are_session_scoped(tmp_path):
    controller = WorkflowController(tmp_path)
    session = tmp_path / "abc123"
    session.mkdir()
    upload = controller.save_upload("abc123", "mesh.glb", b"mesh")
    assert upload.parent == session / "ui_uploads"
    assert upload.suffix == ".glb"
    assert upload.read_bytes() == b"mesh"
    controller.append_history("abc123", "user", "hello")
    assert controller.history("abc123") == [{"role": "user", "content": "hello"}]
    controller.append_history("abc123", "assistant", "next", "checkpoint-1")
    assert controller.history("abc123")[-1]["checkpoint_id"] == "checkpoint-1"
    controller.truncate_history("abc123", 1)
    assert controller.history("abc123") == [{"role": "user", "content": "hello"}]
    with pytest.raises(ValueError):
        controller.session_dir("../outside")


def test_ui_controller_starts_geometry_worker_once(monkeypatch, tmp_path):
    controller = WorkflowController(tmp_path)
    session = tmp_path / "abc123"
    session.mkdir()
    (session / "session.json").write_text(
        json.dumps({"compute_env": "mechfoundry-warp117"}), encoding="utf-8"
    )
    calls = []

    class Process:
        pid = os.getpid()

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.workers.subprocess.Popen", fake_popen)
    assert controller.preload_geometry_model("abc123")
    assert controller.preload_geometry_model("abc123")
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:6] == [
        "conda", "run", "--no-capture-output", "-n", "mechfoundry-warp117", "python"
    ]
    assert "--serve" in command
    assert kwargs["start_new_session"] is True
    # The model weights are cached beside the pid, under the output root.  The
    # socket moved to a short directory under the system temp root -- a Unix
    # socket path cannot be long -- and a cache derived from where the socket
    # sits would follow it there and re-download several gigabytes of weights.
    cache = Path(kwargs["env"]["XDG_CACHE_HOME"])
    assert cache == tmp_path / ".workers" / ".runtime_cache"


def test_physical_scale_skill_asks_for_a_reference_axis_not_an_up_axis():
    """The axis is what the size is measured along, not which way is up.

    Called `up_axis`, and defined as "the axis the object stands up along", the
    field invited the answer "up is y" -- and since the extent was defined as the
    size along that axis, "1 m in z" scaled y to a metre instead of z.
    """
    from src.agents_wrapper.fem_workflow.models import PhysicalScaleInference

    assert "reference_axis" in PhysicalScaleInference.model_fields
    assert "up_axis" not in PhysicalScaleInference.model_fields

    reference = (
        Path(__file__).parents[1]
        / "fem_workflow" / "skill" / "references" / "physical_scale.md"
    ).read_text()
    assert '"reference_axis"' in reference
    assert "not a claim" in reference and "which way is up" in reference
    # A direction stated for the dimension wins over the standing axis.
    assert '"1 m in z"' in reference
    # And with nothing stated it still falls back to the workflow's up axis.
    assert "return `y`" in reference


def test_token_meter_doubles_its_scale_and_keeps_its_length():
    """The bar holds its width; the scale behind it doubles when usage fills it.

    A conversation that outgrew a fixed scale would sit pinned at full and stop
    saying anything, so the budget doubles and the widths are re-read against it.
    """
    import re

    from src.ui.telemetry import session_token_usage, token_budget, token_bar_markup

    def widths(receipts):
        markup = token_bar_markup({"api_receipts": receipts})
        return [float(w) for w in re.findall(r"width:([0-9.]+)%", markup)]

    assert token_budget(0) == 10_000
    assert token_budget(10_000) == 10_000
    assert token_budget(10_001) == 20_000
    assert token_budget(20_001) == 40_000
    assert token_budget(1_280_001) == 2_560_000

    # 9k of the first 10k: the two segments are read against that scale.
    assert widths([{"input_tokens": 7000, "output_tokens": 2000}]) == [70.0, 20.0]
    # 25k spills past 20k, so the scale is 40k and the same spend reads smaller.
    assert widths([{"input_tokens": 20000, "output_tokens": 5000}]) == [50.0, 12.5]
    assert widths([]) == [0.0, 0.0]

    # A cached call cost nothing the second time; its receipt still carries the
    # counts of the call that filled the cache, so it must not be charged twice.
    cached = [{"input_tokens": 900_000, "output_tokens": 900_000, "cached": True}]
    assert session_token_usage({"api_receipts": cached}) == (0, 0)
    assert widths(cached) == [0.0, 0.0]

    markup = token_bar_markup({"api_receipts": [{"input_tokens": 7000, "output_tokens": 2000}]})
    assert "token-in" in markup and "token-out" in markup
    assert "9.0k/10.0k" in markup


def test_gemini_counts_thinking_tokens_as_output():
    """Every provider's receipt should reconcile: input + output == total.

    Gemini reports a thinking model's "thoughts" beside the candidates rather
    than inside them, while total_token_count counts both -- the SDK says so --
    so reading candidates alone undercounts a 2.5-flash answer and leaves the
    receipt not adding up.  The OpenAI-shaped providers fold reasoning into
    their completion count, and this keeps the three comparable.
    """
    from types import SimpleNamespace

    from src.agents_wrapper.gemini import GeminiAPI
    from src.agents_wrapper.interfaces import GenerationRequest

    def client(meta):
        chat = SimpleNamespace(
            send_message=lambda contents: SimpleNamespace(text="OK", usage_metadata=meta)
        )
        return SimpleNamespace(chats=SimpleNamespace(create=lambda **kw: chat))

    def usage(meta):
        api = GeminiAPI(client=client(meta))
        return api.generate_response(
            GenerationRequest(model="gemini-2.5-flash", text="hi")
        ).usage

    thinking = usage(SimpleNamespace(
        prompt_token_count=100, candidates_token_count=20,
        thoughts_token_count=300, total_token_count=420,
    ))
    assert thinking.output_tokens == 320
    assert thinking.input_tokens + thinking.output_tokens == thinking.total_tokens

    plain = usage(SimpleNamespace(
        prompt_token_count=100, candidates_token_count=20,
        thoughts_token_count=None, total_token_count=120,
    ))
    assert plain.output_tokens == 20

    # Absent accounting stays unknown rather than being reported as zero.
    assert usage(None) is None


def test_each_role_can_run_on_its_own_provider(tmp_path):
    """Text and image calls choose their service independently.

    The strongest reasoning model and the one that reads a montage best are not
    always on the same service, and a receipt has to name whichever answered.
    """
    from src.agents_wrapper.fem_workflow.specialists import (
        PROVIDERS, SpecialistCalls, provider_factory,
    )

    assert list(PROVIDERS) == ["deepseek", "groq", "gemini"], "DeepSeek leads"
    with pytest.raises(ValueError, match="deepseek, groq, gemini"):
        provider_factory("nope")

    class FakeAPI:
        default_model = "stub"

    calls = SpecialistCalls.__new__(SpecialistCalls)
    calls.session_dir = tmp_path
    calls.provider, calls.vision_provider = "deepseek", "groq"

    class Agent:
        def __init__(self, model):
            self.model = model

    calls.agent, calls.vision_agent = Agent("deepseek-flash"), Agent("qwen/qwen3.8-27b")
    assert calls._provider_for(calls.agent) == "deepseek"
    assert calls._provider_for(calls.vision_agent) == "groq"
    # An image routes to the vision agent, anything else to the text one.
    assert calls._agent_for(tmp_path / "x.webp") is calls.vision_agent
    assert calls._agent_for(None) is calls.agent


def test_headroom_follows_the_model_not_the_role():
    """A reasoning model needs room to think whichever role it is filling.

    Granting it by role assumed only the text stage reasons; deepseek-flash
    reads pictures and reasons, and without headroom spends the whole budget
    thinking and returns an empty answer.
    """
    from src.agents_wrapper.fem_workflow import specialists

    assert "qwen/qwen3.8-27b" in specialists.NON_REASONING_MODELS
    assert "deepseek-flash" not in specialists.NON_REASONING_MODELS
    assert specialists.REASONING_HEADROOM >= 512


def test_deepseek_sends_the_documented_thinking_parameters():
    """`thinking` travels in extra_body; the level is the top-level parameter.

    Guessed spellings are silently ignored by an OpenAI-compatible endpoint, so
    a wrong one here looks exactly like a model that will not stop thinking.
    """
    from src.agents_wrapper.deepseek import DeepSeekAPI
    from src.agents_wrapper.interfaces import GenerationRequest

    sent = []

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    sent.append(kwargs)
                    message = type("M", (), {"content": '{"ok":true}'})()
                    return type("R", (), {
                        "choices": [type("C", (), {"message": message})()],
                        "usage": None,
                    })()

    api = DeepSeekAPI(client=FakeClient())
    for effort in (None, "none", "low"):
        api.generate_response(GenerationRequest(
            model="deepseek-flash", text="hi", reasoning_effort=effort,
        ))
    default, off, low = sent
    assert "extra_body" not in default and "reasoning_effort" not in default
    assert off["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in off
    assert low["extra_body"] == {"thinking": {"type": "enabled"}}
    assert low["reasoning_effort"] == "low"


def test_thinking_is_bought_per_stage_and_only_where_it_paid(
    monkeypatch, tmp_path
):
    """Measured per stage on deepseek-flash; most stages gained nothing.

    Scale and material returned character-identical answers for 3-4x the output
    tokens, and the image stages returned no answer at all above "none".  Only
    the supports-and-loads rewrite came back better for thinking, so only it
    asks, and only it is budgeted for the tokens thinking costs.
    """
    from src.agents_wrapper.fem_workflow import specialists as specialists_module

    seen = []

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            self.model = model or "m"

        def generate_structured(self, prompt, **kwargs):
            seen.append((kwargs.get("image") is not None,
                         kwargs.get("reasoning_effort"),
                         kwargs["max_output_tokens"]))
            if kwargs.get("image"):
                value = {"description": "a chair"}
            elif "physical_scale" in prompt:
                value = {"reference_axis": "y", "physical_extent": 0.9}
            else:
                value = {"refined_prompt": "fix the feet"}
            return value, type("R", (), {"usage": None})()

    monkeypatch.setattr(specialists_module, "AgentsWrapper", FakeAgent)
    monkeypatch.setitem(specialists_module.PROVIDERS, "deepseek", lambda: object())
    monkeypatch.setattr(specialists_module.SpecialistCalls, "_preview",
                        lambda self, path, max_size=768: path)
    image = tmp_path / "views.png"
    image.write_bytes(b"png")

    calls = specialists_module.SpecialistCalls(
        tmp_path, "deepseek", "deepseek-flash", vision_model="deepseek-flash")
    calls.describe_image("geometry_overview", image)
    calls.infer_physical_scale("a chair", "", "m")
    calls.refine_bc_prompt("a chair", "fix the feet")

    (saw_image, vision_effort, vision_budget), (_, scale_effort, scale_budget), \
        (_, prompt_effort, prompt_budget) = seen
    assert saw_image and vision_effort == "none"
    assert scale_effort == "none", "an identical answer is not worth 3x the output"
    assert prompt_effort == "none"
    # Nothing is going to think, so nothing is budgeted for thinking.
    assert vision_budget == 32 + 2 * specialists_module.VISION_BRIEFS[
        "geometry_overview"][2]
    assert scale_budget == 64
    # Room to name every segment a region spans.
    assert prompt_budget == 600


def test_region_synthesis_thinks_and_is_given_room_to_finish():
    """Selector trees are the longest answer, so their headroom is its own.

    Thinking about one ran to 1569 output tokens in a measured run; the budget
    that suits a one-line rewrite would cut that off and return nothing.
    """
    from src.agents_wrapper.fem_workflow import specialists

    assert specialists.REASONING_HEADROOM_BY_PURPOSE["bc_synthesis"] > 1569
    # Nothing is buying thinking at the moment, so nothing is spending it.
    assert specialists.DEFAULT_REASONING_EFFORT == "none"
    assert specialists.REASONING_EFFORT == {}


def test_an_empty_answer_is_re_asked_rather_than_repaired(monkeypatch, tmp_path):
    """Repairing nothing invents something, and an image call invents worst.

    The repair prompt used to go out without the image, so a vision stage whose
    first attempt came back empty was asked to correct an empty string with no
    picture attached.  It answered, accurately, that it could see no image --
    and that sentence was stored and cached as the description of the geometry.
    """
    from src.agents_wrapper.errors import APIResponseError
    from src.agents_wrapper.fem_workflow import specialists as specialists_module

    seen = []

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            self.model = model or "m"

        def generate_structured(self, prompt, **kwargs):
            seen.append({"prompt": prompt, "image": kwargs.get("image"),
                         "budget": kwargs["max_output_tokens"]})
            if len(seen) == 1:
                raise APIResponseError("DeepSeek returned no text", raw_text="")
            return {"description": "a blocky armchair"}, type("R", (), {"usage": None})()

    monkeypatch.setattr(specialists_module, "AgentsWrapper", FakeAgent)
    monkeypatch.setitem(specialists_module.PROVIDERS, "deepseek", lambda: object())
    monkeypatch.setattr(specialists_module.SpecialistCalls, "_preview",
                        lambda self, path, max_size=768: path)
    image = tmp_path / "views.png"
    image.write_bytes(b"png")

    calls = specialists_module.SpecialistCalls(
        tmp_path, "deepseek", "deepseek-flash", vision_model="deepseek-flash")
    assert calls.describe_image("geometry_overview", image) == "a blocky armchair"

    first, second = seen
    assert "repair_json" not in second["prompt"]
    assert second["prompt"] == first["prompt"], "the real question, asked again"
    assert second["image"] == image, "and still holding the picture"
    assert second["budget"] > first["budget"], "with room for the answer that overran"


def test_a_malformed_answer_is_repaired_with_the_image_still_attached(
    monkeypatch, tmp_path
):
    from src.agents_wrapper.errors import APIResponseError
    from src.agents_wrapper.fem_workflow import specialists as specialists_module

    seen = []

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            self.model = model or "m"

        def generate_structured(self, prompt, **kwargs):
            seen.append({"prompt": prompt, "image": kwargs.get("image")})
            if len(seen) == 1:
                raise APIResponseError("not JSON", raw_text="description: a chair")
            return {"description": "a chair"}, type("R", (), {"usage": None})()

    monkeypatch.setattr(specialists_module, "AgentsWrapper", FakeAgent)
    monkeypatch.setitem(specialists_module.PROVIDERS, "deepseek", lambda: object())
    monkeypatch.setattr(specialists_module.SpecialistCalls, "_preview",
                        lambda self, path, max_size=768: path)
    image = tmp_path / "views.png"
    image.write_bytes(b"png")

    calls = specialists_module.SpecialistCalls(
        tmp_path, "deepseek", "deepseek-flash", vision_model="deepseek-flash")
    calls.describe_image("geometry_overview", image)
    assert "repair_json" in seen[1]["prompt"]
    assert seen[1]["image"] == image


def test_a_reply_one_bracket_off_is_mended_without_asking_again():
    """At temperature 0 a repair request told only "not a single JSON value" sent the same
    broken object back, and every resend of the step failed alike.  A bracket over or short is
    mended locally and validated by the stage's schema; anything else goes to the repair with
    the parser's position."""
    from src.agents_wrapper.errors import APIResponseError
    from src.agents_wrapper.fem_workflow.models import BCProposal
    from src.agents_wrapper.fem_workflow.specialists import _one_bracket_mend, _parse_error

    selector = ('{"type":"and","params":{},"children":[{"type":"lowest","params":{},"children":[]},'
                '{"type":"slab","params":{"axis":"y","min_val":0.0,"max_val":0.01},"children":[]}]}')
    over = ('{"dirichlet":[{"name":"base","selector":' + selector + '],"components":[0,1,2]}],'
            '"neumann":[]}')
    short = ('{"dirichlet":[{"name":"base","components":[0,1,2],"selector":' + selector + '],'
             '"neumann":[]}')
    for broken in (over, short, "```json\n" + over + "\n```"):
        mended = _one_bracket_mend(broken, BCProposal)
        assert mended["dirichlet"][0]["components"] == [0, 1, 2], broken
        assert mended["dirichlet"][0]["selector"]["children"][1]["type"] == "slab"
    assert _one_bracket_mend('{"dirichlet": "base, the lowest faces"', BCProposal) is None
    said = _parse_error(APIResponseError("the API did not return a single JSON value"), over)
    assert "Expecting ',' delimiter at character" in said and '"children":[]}]}' in said


def test_a_mended_reply_spends_no_second_call(monkeypatch, tmp_path):
    from src.agents_wrapper.errors import APIResponseError
    from src.agents_wrapper.fem_workflow import specialists as specialists_module

    seen = []

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            self.model = model or "m"

        def generate_structured(self, prompt, **kwargs):
            seen.append(prompt)
            if len(seen) == 3:
                return {"description": "a block"}, type("R", (), {"usage": None})()
            raise APIResponseError("the API did not return a single JSON value",
                                   raw_text='{"description": "a block"}}', response=reply[0])

    from src.agents_wrapper.interfaces import GenerationResponse
    reply = [GenerationResponse(text="", finish_reason="stop")]
    monkeypatch.setattr(specialists_module, "AgentsWrapper", FakeAgent)
    monkeypatch.setitem(specialists_module.PROVIDERS, "deepseek", lambda: object())
    monkeypatch.setattr(specialists_module.SpecialistCalls, "_preview",
                        lambda self, path, max_size=768: path)
    image = tmp_path / "views.png"
    image.write_bytes(b"png")
    calls = specialists_module.SpecialistCalls(
        tmp_path, "deepseek", "deepseek-flash", vision_model="deepseek-flash")
    assert calls.describe_image("geometry_overview", image) == "a block"
    assert len(seen) == 1
    # a mended reply is not cached as the model's answer: the same question asks again
    reply[0] = GenerationResponse(text="", finish_reason="length")
    calls.describe_image("geometry_overview", image)
    # and a reply cut off at the token budget is never mended, nor sent to a repair held to the same
    # budget: the question is asked again with more room
    assert len(seen) == 3 and "repair_json" not in seen[2] and seen[2] == seen[1]


def test_a_mend_with_two_readings_is_no_mend():
    """A missing brace after the last load could also be "closed" at the end of the reply,
    which moves the acceleration into the loads as a region and drops gravity: two readings,
    so the reply goes to the repair request instead."""
    from src.agents_wrapper.fem_workflow.models import BCValueInference
    from src.agents_wrapper.fem_workflow.specialists import _one_bracket_mend, _truncated
    from src.agents_wrapper.interfaces import GenerationResponse, TokenUsage

    reply = ('{"dirichlet_values":{"base":{"components":[0,1,2]}},'
             '"neumann_values":{"top":{"force":[0,-10,0]},"acceleration":[0,-9.80665,0]}')
    assert _one_bracket_mend(reply, BCValueInference) is None
    assert _truncated(GenerationResponse(text="", finish_reason="MAX_TOKENS"))
    assert _truncated(GenerationResponse(text="", usage=TokenUsage(output_tokens=1024)), budget=1024)
    assert not _truncated(GenerationResponse(text="", finish_reason="stop"), budget=1024)


def test_startup_probe_is_one_shot_and_budgets_for_thinking(monkeypatch):
    observed = {}

    class FakeAPI:
        default_model = "default-model"

    class FakeAgent:
        def __init__(self, api, model=None, **kwargs):
            observed["attempts"] = kwargs["max_request_attempts"]
            self.model = model or api.default_model

        def generate(self, text, **kwargs):
            observed.update(kwargs)
            return "OK"

    monkeypatch.setattr(run_workflow, "provider_factory", lambda name: FakeAPI)
    monkeypatch.setattr(run_workflow, "AgentsWrapper", FakeAgent)
    assert run_workflow.probe_api("groq", None) == "default-model"
    assert observed["attempts"] == 1
    # Still one shot, but no longer tiny: a reasoning model spends output tokens
    # thinking before it answers, and deepseek-flash thinks about even "reply
    # OK", so the probe budgets for the thinking instead of coming back empty.
    assert observed["max_output_tokens"] == run_workflow.REASONING_HEADROOM


def test_failed_startup_probe_accepts_explicit_replacement(monkeypatch):
    selections = []

    def fake_probe(provider, model):
        selections.append((provider, model))
        if len(selections) == 1:
            raise APIResponseError("unavailable")
        return model or "resolved-model"

    monkeypatch.setattr(run_workflow, "probe_api", fake_probe)
    monkeypatch.setattr("builtins.input", lambda _: '{"provider":"gemini","model":"gemini-test"}')
    assert run_workflow.select_available_api("groq", "bad-model") == ("gemini", "gemini-test")
    assert selections == [("groq", "bad-model"), ("gemini", "gemini-test")]


FIELD_COMPONENTS_FIXTURE = {
    "displacement": ("magnitude", "x", "y", "z"),
    "strain": ("von_mises", "xx", "yy", "zz", "xy", "yz", "xz"),
    "stress": ("von_mises", "xx", "yy", "zz", "xy", "yz", "xz"),
}


def _ramp(count: int) -> list[Path]:
    return [Path(f"/session/solve/run/vtu_surface/surface_t{i:04d}.vtu")
            for i in range(count)]


def test_prefetch_and_click_ask_for_an_asset_in_the_same_words():
    """What the prefetch builds is what the click finds, or it built nothing.

    The monitor caches a tool on a fingerprint of its arguments, so an asset
    built ahead of time is only reused if the two routes name it identically.
    One shared builder is what guarantees that, and this pins the guarantee: the
    request the viewer makes for a combination must equal the one the plan made
    for it.
    """
    steps = _ramp(3)
    planned = plan_solution_prefetch(
        Path("/session"), steps, ("displacement", "stress"), FIELD_COMPONENTS_FIXTURE
    )
    # Exactly how the viewport asks, drawing the last step of a three-step ramp
    # -- including which surface the colours are scaled against, which is part
    # of the request and so part of what has to match.
    _, clicked = field_asset_request(
        "/session", str(steps[2]), "stress", "von_mises", 0.0, 2, 3, str(steps[2])
    )
    assert clicked in planned
    # And an earlier step, which is the one the shared range actually changes.
    _, earlier = field_asset_request(
        "/session", str(steps[0]), "stress", "von_mises", 0.0, 0, 3, str(steps[2])
    )
    assert earlier in planned


def test_prefetch_plans_what_the_viewer_opens_on_and_stops_there():
    """Every field's opening component at every step, then displacement's rest.

    Eighteen component pairs across six steps is over a hundred exports nobody
    opens.  What a click actually lands on is the first component of whichever
    field is picked -- and for a ramp, that component at every step at once,
    which is the stall worth removing.
    """
    steps = _ramp(3)
    planned = plan_solution_prefetch(
        Path("/session"), steps,
        ("displacement", "strain", "stress"), FIELD_COMPONENTS_FIXTURE,
    )
    names = [Path(request["output"]).name for request in planned]
    assert names == [
        "displacement_magnitude_s00.glb", "displacement_magnitude_s01.glb",
        "displacement_magnitude_s02.glb",
        "strain_von_mises_s00.glb", "strain_von_mises_s01.glb",
        "strain_von_mises_s02.glb",
        "stress_von_mises_s00.glb", "stress_von_mises_s01.glb",
        "stress_von_mises_s02.glb",
        # The components people reach for next, at the step the viewer starts on.
        "displacement_x_s02.glb", "displacement_y_s02.glb",
        "displacement_z_s02.glb",
    ]
    # Each request colours its own step's surface, not the newest one.
    for request, name in zip(planned, names):
        expected = steps[2] if name.endswith(("_x_s02.glb", "_y_s02.glb", "_z_s02.glb")) \
            else steps[int(name[-6:-4])]
        assert request["vtu"] == str(expected)


def test_prefetch_plans_nothing_it_cannot_see():
    """A field the solve did not write is not colourable, and is not planned."""
    steps = _ramp(2)
    assert plan_solution_prefetch(
        Path("/s"), steps, ("strain",), FIELD_COMPONENTS_FIXTURE
    ) == plan_solution_prefetch(
        Path("/s"), steps, ("strain", "temperature"), FIELD_COMPONENTS_FIXTURE
    )
    assert plan_solution_prefetch(Path("/s"), steps, (), FIELD_COMPONENTS_FIXTURE) == []
    assert plan_solution_prefetch(
        Path("/s"), [], ("displacement",), FIELD_COMPONENTS_FIXTURE
    ) == []


def test_unramped_solve_is_planned_without_a_step_stamp():
    """A run that never ramped and a ramp's first step must not share a name.

    They hold different surfaces, so the stamp appears for every step of a ramp,
    step zero included, and for no step of a run that has only one.
    """
    single = plan_solution_prefetch(
        Path("/s"), _ramp(1), ("displacement",), FIELD_COMPONENTS_FIXTURE
    )
    assert Path(single[0]["output"]).name == "displacement_magnitude.glb"
    ramped = plan_solution_prefetch(
        Path("/s"), _ramp(2), ("displacement",), FIELD_COMPONENTS_FIXTURE
    )
    assert Path(ramped[0]["output"]).name == "displacement_magnitude_s00.glb"


def test_magnitude_is_spelled_as_no_component_for_the_exporter():
    """The viewport's word for it is not the exporter's.

    `displacement` has no column called `magnitude`; asking for one is the error
    that reached a user, and translating at this boundary is what prevents it.
    """
    _, request = field_asset_request("/s", "/s/a.vtu", "displacement", "magnitude")
    assert request["component"] == ""
    _, named = field_asset_request("/s", "/s/a.vtu", "displacement", "y")
    assert named["component"] == "y"


SERVABLE_ARGUMENTS = {
    "stage_asset": {"source": "a.glb", "output": "b.glb"},
    "inspect_surface": {"surface": "s.glb", "output": "i.json"},
    "scale_surface": {"surface": "s.glb", "output": "o.glb", "axis": "y",
                      "extent": 1.0, "units": "SI (m-kg-s)", "octree_depth": 7},
    "render_surface": {"surface": "s.glb", "output": "r.png"},
    "plain_colour_surface": {"surface": "s.glb", "rgb": "200,170,101", "output": "t.glb"},
    "render_segmentation": {"surface": "s.glb", "segments_npz": "s.npz",
                            "segments_json": "s.json", "output": "m.png"},
    "render_part_montage": {"surface": "s.glb", "segments_npz": "s.npz",
                            "segments_json": "s.json", "output": "p.png"},
    "build_part_field": {"surface": "s.glb", "segments_npz": "s.npz",
                         "materials_json": "parts.json", "unit_system": "SI (m-kg-s)",
                         "output": "field"},
    "segment_surface": {"surface": "s.glb", "output_npz": "s.npz",
                        "output_json": "s.json", "output_glb": "s.glb",
                        "clusters": 5, "rays": 25, "smoothing": 0.25,
                        "cone_angle_deg": 120.0},
    "resolve_bcs": {"surface": "s.glb", "bc_json": "bc.json",
                    "segments_npz": "s.npz", "output_npz": "r.npz",
                    "output_json": "r.json", "output_glb": "r.glb"},
    "build_case": {"surface": "s.glb", "bc_json": "bc.json", "segments_npz": "s.npz",
                   "material_json": "m.json", "settings_json": "o.json",
                   "transform_json": "t.json", "output": "case.json",
                   "solve_output": "solve", "octree_depth": 7},
    "export_surface_field": {"vtu": "s.vtu", "output": "o.glb",
                             "field": "displacement", "component": "",
                             "deform_scale": 0.0},
}


def test_every_servable_tool_survives_the_round_trip_through_argv(tmp_path):
    """The worker runs a tool by re-parsing the argv a subprocess would have got.

    That is what keeps the resident route and the one-shot route from drifting:
    there is one parser, and a subcommand added to the tool module is servable
    with no change here.  It only holds while every servable tool is reachable
    through that parser, which is what this checks.
    """
    from src.agents_wrapper.fem_workflow.monitor import WORKER_TOOLS

    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    assert set(WORKER_TOOLS) == set(SERVABLE_ARGUMENTS)
    assert "run_forward" not in WORKER_TOOLS
    for name in sorted(WORKER_TOOLS):
        tail = monitor.registry[name].command(SERVABLE_ARGUMENTS[name])
        parsed = workflow_tool.parser().parse_args([str(item) for item in tail[1:]])
        assert parsed.func.__name__ == name


@pytest.fixture
def socket_home():
    """Take away the short socket directories these tests create.

    The sockets deliberately live outside the session tree, because a Unix
    socket path cannot be long -- so unlike everything else a test writes, they
    are not swept up with `tmp_path`, and a test that makes one leaves nothing
    behind only by removing it itself.
    """
    made = []
    yield made.append
    for directory in made:
        shutil.rmtree(directory, ignore_errors=True)


def _fake_worker(socket_path: Path, behaviour):
    """A stand-in worker speaking the real protocol, on a real socket."""
    import socket as socket_module
    import threading

    server = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    server.bind(os.fspath(socket_path))
    server.listen(4)
    seen = []

    def loop():
        try:
            connection, _ = server.accept()
        except OSError:
            return
        with connection:
            behaviour(connection, seen)
        server.close()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return seen, thread


def _greet_and_answer(response):
    def behaviour(connection, seen):
        connection.sendall(b'{"ready":true}\n')
        seen.append(json.loads(connection.makefile("r", encoding="utf-8").readline()))
        connection.sendall((json.dumps(response) + "\n").encode())
    return behaviour


def test_a_served_tool_reads_exactly_like_the_subprocess_it_replaces(tmp_path, socket_home):
    """Same stdout, same stderr, same returncode -- and the same receipt.

    The backend log the UI shows is read out of those receipts, so a tool that
    ran in the worker has to be indistinguishable from one that ran on its own.
    """
    session = tmp_path / "session"
    (session / "boundary_conditions").mkdir(parents=True)
    monitor = ToolMonitor(session)
    socket_path, _ = monitor._worker_paths("tools")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = session / "out.glb"
    seen, _thread = _fake_worker(socket_path, _greet_and_answer(
        {"ok": True, "stdout": '{"artifacts":["out.glb"]}\n', "stderr": ""}
    ))
    artifact.write_bytes(b"glb")
    receipt = monitor.execute("export_surface_field", {
        "vtu": str(session / "s.vtu"), "output": str(artifact),
        "field": "displacement", "component": "", "deform_scale": 0.0,
    })
    assert receipt.returncode == 0
    assert Path(receipt.stdout_path).read_text() == '{"artifacts":["out.glb"]}\n'
    # The argv handed over is the one the subprocess would have been given,
    # minus the script that would have run it.
    assert seen[0]["argv"][0] == "export-surface-field"
    assert "--deform-scale" in seen[0]["argv"]


def test_a_tool_that_fails_in_the_worker_still_names_its_diagnostic(tmp_path, socket_home):
    """A traceback ends in the line the monitor quotes, so failures read alike."""
    session = tmp_path / "session"
    session.mkdir(parents=True)
    monitor = ToolMonitor(session)
    socket_path, _ = monitor._worker_paths("tools")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_worker(socket_path, _greet_and_answer({
        "ok": False, "stdout": "",
        "stderr": "Traceback (most recent call last):\n  ...\n"
                  "ValueError: 'displacement' has no component 'magnitude'\n",
    }))
    with pytest.raises(RuntimeError) as failure:
        monitor.execute("export_surface_field", {
            "vtu": str(session / "s.vtu"), "output": str(session / "o.glb"),
            "field": "displacement", "component": "magnitude", "deform_scale": 0.0,
        })
    assert "has no component 'magnitude'" in str(failure.value)


def test_a_busy_worker_is_left_alone_and_the_tool_runs_here_instead(
        tmp_path, monkeypatch, socket_home):
    """Silence means busy, and a busy worker must keep its socket.

    A worker in the middle of a segmentation cannot greet, so the caller gives
    up and runs the tool itself -- having sent nothing, so the two cannot both
    run it.  Clearing the socket on that timeout would shut a working worker out
    of the rest of the session.
    """
    session = tmp_path / "session"
    session.mkdir(parents=True)
    monitor = ToolMonitor(session)
    socket_path, pid_path = monitor._worker_paths("tools")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    _fake_worker(socket_path, lambda connection, seen: time.sleep(5))
    monkeypatch.setattr(
        "src.agents_wrapper.fem_workflow.monitor.WORKER_ACK_TIMEOUT", 0.2
    )
    ran = []

    def fake_run(command, **kwargs):
        ran.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"glb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run
    )
    monitor.execute("export_surface_field", {
        "vtu": str(session / "s.vtu"), "output": str(session / "o.glb"),
        "field": "displacement", "component": "", "deform_scale": 0.0,
    })
    assert len(ran) == 1
    assert socket_path.exists(), "a busy worker must not be struck off"
    assert pid_path.exists()


def test_a_dead_worker_s_socket_is_cleared_so_the_next_preload_relaunches(
        tmp_path, monkeypatch, socket_home):
    """A socket left by a killed worker must not be taken for a live one.

    Left in place it answers nothing and is believed by every later check, so
    preloading never starts a replacement and every call quietly runs cold for
    the rest of the session.
    """
    session = tmp_path / "session"
    session.mkdir(parents=True)
    monitor = ToolMonitor(session)
    socket_path, pid_path = monitor._worker_paths("tools")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_bytes(b"")  # a file where a socket was, connecting to nothing
    pid_path.write_text("999999999", encoding="utf-8")

    def fake_run(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_bytes(b"glb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run
    )
    monitor.execute("export_surface_field", {
        "vtu": str(session / "s.vtu"), "output": str(session / "o.glb"),
        "field": "displacement", "component": "", "deform_scale": 0.0,
    })
    assert not socket_path.exists()
    assert not pid_path.exists()


def test_the_solve_is_what_unloads_the_geometry_model(tmp_path, monkeypatch):
    """TripoSG stays resident across regenerations; the solve is what evicts it.

    Unloading after every generation made the card people iterate on most reload
    the model each attempt, which is the slowest thing that card does.  What the
    unload is actually protecting is the solve, so that is where it belongs --
    and in the monitor rather than behind the button, so a resumed graph or a
    second solve on the same card unloads it too.
    """
    session = tmp_path / "session"
    (session / "case").mkdir(parents=True)
    monitor = ToolMonitor(session)
    unloads = []
    monkeypatch.setattr(ToolMonitor, "_shutdown_geometry_worker",
                        lambda self: unloads.append(True))
    monkeypatch.setattr(ToolMonitor, "_run_geometry_worker",
                        lambda self, name, arguments, timeout: None)

    def fake_run(command, **kwargs):
        if "--output" in command:
            Path(command[command.index("--output") + 1]).write_bytes(b"glb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run
    )
    surface = session / "geometry" / "surface.glb"
    surface.parent.mkdir(parents=True)
    monitor.execute("generate_text_surface",
                    {"output": str(surface), "prompt": "a bracket"})
    assert unloads == [], "a finished generation keeps the model"
    # The same generation again: served from its receipt, and still no unload.
    monitor.execute("generate_text_surface",
                    {"output": str(surface), "prompt": "a bracket"})
    assert unloads == [], "a cached generation keeps the model too"

    solve_output = session / "solve"
    solve_output.mkdir()
    case = session / "case" / "case.json"
    case.write_text(json.dumps({"output": str(solve_output)}), encoding="utf-8")
    monitor.execute("run_forward", {"case": str(case)})
    assert unloads == [True], "the solve makes room for itself"


def test_a_tool_worker_starts_once_and_again_only_once_it_is_gone(
        monkeypatch, tmp_path, socket_home):
    """A worker is alive while its process is, not while its socket file is.

    A worker binds its socket only once warm, so a live process with no socket
    is one on its way up and must not be started twice.  A socket with no
    process is the remains of one killed outright -- believed, it wedges
    preloading for the rest of the session while every call runs cold.
    """
    controller = WorkflowController(tmp_path)
    launched = []

    class Process:
        pid = os.getpid()

    def fake_popen(command, **kwargs):
        launched.append(command)
        return Process()

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.workers.subprocess.Popen", fake_popen)
    assert controller.preload_tool_worker("mechfoundry-warp117")
    socket_path, pid_path = controller.worker_paths("tools", "mechfoundry-warp117")
    # Warming, so no socket yet -- and still not a reason to start a second one.
    assert not socket_path.exists()
    assert controller.preload_tool_worker("mechfoundry-warp117")
    assert len(launched) == 1
    assert "--serve" in launched[0] and "--warm" in launched[0]
    assert launched[0][-1] == "light"

    # Killed outright: the socket outlives the process and must not be believed.
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_bytes(b"")
    pid_path.write_text("999999999", encoding="utf-8")
    # Not yet, though.  Dying this soon after being started is what a worker
    # that cannot start looks like, and these preloads run on every rerun, so
    # trying again immediately would mean trying again several times a second.
    assert not controller.preload_tool_worker("mechfoundry-warp117", "cuda")
    assert len(launched) == 1

    attempted = tmp_path / ".workers" / "tools.attempted"
    aged = time.time() - WORKER_RELAUNCH_BACKOFF - 1
    os.utime(attempted, (aged, aged))
    assert controller.preload_tool_worker("mechfoundry-warp117", "cuda")
    assert len(launched) == 2, "a worker gone for a while is replaced"
    assert launched[1][-1] == "cuda"


def test_a_socket_path_stays_short_enough_for_the_system_to_bind(tmp_path):
    """Unix sockets have a path limit, and a silent bind failure is the worst
    shape this can fail in: the worker is detached, so nothing surfaces, and
    every page draw would start another one that dies the same way.

    Keeping the socket out of the session tree is what buys the headroom -- a
    deeply nested checkout is otherwise entirely capable of exceeding it.
    """
    from src.agents_wrapper.fem_workflow.monitor import worker_paths

    deep = tmp_path / ("nested/" * 20) / "output"
    socket_path, pid_path = worker_paths(deep, "tools", "mechfoundry-warp117")
    assert len(os.fspath(socket_path).encode()) < 100
    # The pid, which has no such limit, stays where a person would look for it.
    assert pid_path.parent == deep / ".workers"


def test_the_two_workers_do_not_answer_each_other_s_calls(tmp_path):
    """One environment's worker, one kind of worker, one socket each."""
    controller = WorkflowController(tmp_path)
    tools, _ = controller.worker_paths("tools", "mechfoundry-warp117")
    geometry, _ = controller.worker_paths("triposg", "mechfoundry-warp117")
    other, _ = controller.worker_paths("tools", "mechfoundry-other")
    assert len({tools, geometry, other}) == 3
    # And the monitor looks where the controller puts them.
    session = tmp_path / "session"
    session.mkdir()
    assert ToolMonitor(session)._worker_paths("tools")[0] == tools
    assert ToolMonitor(session)._geometry_worker_paths()[0] == geometry


def test_a_warm_request_is_made_once_per_worker_not_once_per_rerun(monkeypatch,
                                                                   tmp_path,
                                                                   socket_home):
    """Asking again on every page draw would stall the page on a busy worker.

    The warm-up is requested while the page is being drawn, so it cannot wait;
    and a worker in the middle of a segmentation does not answer for as long as
    that takes.  Asking once per worker is what keeps that from being a stall
    repeated several times a second -- and a worker that has been replaced is a
    different worker, which has not been asked.
    """
    import src.agents_wrapper.fem_workflow.workers as workers_module

    controller = WorkflowController(tmp_path)
    asked = []
    monkeypatch.setattr(workers_module, "request_warm",
                        lambda path, tier, serving: asked.append((tier, serving)))

    class Process:
        pid = os.getpid()

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.workers.subprocess.Popen",
                        lambda command, **kwargs: Process())
    controller.preload_tool_worker("mechfoundry-warp117", "light")
    for _ in range(5):
        controller.preload_tool_worker("mechfoundry-warp117", "cuda")
    assert len(asked) == 5, "the controller asks; request_warm is what deduplicates"
    assert {tier for tier, _ in asked} == {"cuda"}
    # And it is told which worker it is talking to, so it can tell one that has
    # been replaced from one it has already asked.
    assert {serving for _, serving in asked} == {str(os.getpid())}


def test_the_real_warm_request_declines_to_repeat_itself(tmp_path, socket_home):
    """The deduplication itself, on the path the controller actually calls."""
    from src.agents_wrapper.fem_workflow.monitor import worker_paths
    from src.agents_wrapper.fem_workflow.workers import _REQUESTED_TIER, request_warm

    socket_path, _ = worker_paths(tmp_path, "tools", "mechfoundry-warp117")
    socket_home(socket_path.parent)
    _REQUESTED_TIER.pop(os.fspath(socket_path), None)
    # No socket: nothing to ask, and nothing remembered, so a later worker is
    # still asked.
    request_warm(socket_path, "cuda", "111")
    assert os.fspath(socket_path) not in _REQUESTED_TIER

    import socket as socket_module
    import threading

    socket_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    server.bind(os.fspath(socket_path))
    server.listen(4)
    served = []

    def loop():
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            with connection:
                connection.sendall(b'{"ready":true}\n')
                served.append(connection.makefile("r", encoding="utf-8").readline())

    threading.Thread(target=loop, daemon=True).start()

    def served_count(expected):
        """The worker records on its own thread, a moment after we let go."""
        deadline = time.monotonic() + 5.0
        while len(served) < expected and time.monotonic() < deadline:
            time.sleep(0.01)
        return len(served)

    try:
        for _ in range(4):
            request_warm(socket_path, "cuda", "111")
        assert served_count(1) == 1
        # A different pid is a different worker, and has not been asked.
        request_warm(socket_path, "cuda", "222")
        assert served_count(2) == 2
    finally:
        server.close()
        _REQUESTED_TIER.pop(os.fspath(socket_path), None)


def test_the_solve_is_launched_directly_but_never_served(tmp_path):
    """Two independent decisions that used to be one set.

    The solve skips `conda run` like every other tool -- the assembled backend
    it links against was the reason to doubt that, and it was checked -- but it
    must not be handed to the resident worker, which serves one call at a time
    and would be held for the length of a solve while the page waited on an
    export.  A solve also wants the GPU to itself, which is why it is the thing
    that unloads the geometry model.
    """
    from src.agents_wrapper.fem_workflow.monitor import DIRECT_LAUNCH, WORKER_TOOLS

    assert "run_forward" in DIRECT_LAUNCH
    assert "run_forward" not in WORKER_TOOLS
    assert WORKER_TOOLS < DIRECT_LAUNCH

    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session, "mechfoundry-warp117")
    interpreter = monitor.interpreter
    if interpreter is None:
        pytest.skip("no discoverable environment interpreter on this machine")
    # The launcher the monitor would build, without running anything.
    assert monitor.registry["run_forward"].command({"case": "c.json"})[0].endswith(
        "workflow_tool.py"
    )


def test_every_step_of_a_ramp_is_coloured_against_the_same_surface():
    """One scale for the ramp, or the ramp cannot be read.

    Each step left to itself stretches the colours over its own largest value,
    so a run that triples its load looks identical at every step -- while the
    one legend on screen, the last step's, claims to speak for all of them.
    Naming the fully loaded step as the reference is what makes an early step
    come out proportionally cooler and the legend true for every frame.
    """
    steps = _ramp(4)
    planned = plan_solution_prefetch(
        Path("/s"), steps, ("displacement",), FIELD_COMPONENTS_FIXTURE
    )
    assert planned, "a ramp should plan something"
    assert {request["value_range_from"] for request in planned} == {str(steps[-1])}


def test_a_single_step_solve_is_coloured_against_itself():
    """Nothing to compare with, so nothing is named, and the tool keeps its own
    range -- the behaviour every non-ramped solve had before ramps existed."""
    planned = plan_solution_prefetch(
        Path("/s"), _ramp(1), ("displacement",), FIELD_COMPONENTS_FIXTURE
    )
    assert {request["value_range_from"] for request in planned} == {""}


def test_the_range_reference_is_optional_to_the_tool(tmp_path):
    """Callers that do not care about a shared range need not say so.

    The cache fingerprint is taken over the arguments, so an argument nobody
    asked for must stay absent rather than be spelled out as a default.
    """
    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    bare = {"vtu": "s.vtu", "output": "o.glb", "field": "displacement",
            "component": "", "deform_scale": 0.0}
    argv = monitor.registry["export_surface_field"].command(bare)
    assert "--value-range-from" not in argv
    argv = monitor.registry["export_surface_field"].command(
        {**bare, "value_range_from": "ref.vtu"}
    )
    assert argv[argv.index("--value-range-from") + 1] == "ref.vtu"
    # And the tool accepts both spellings.
    parsed = workflow_tool.parser().parse_args([str(item) for item in argv[1:]])
    assert parsed.value_range_from == "ref.vtu"


def test_a_worker_that_has_exited_is_not_mistaken_for_a_live_one(tmp_path):
    """An idled-out worker must not wedge preloading for the rest of a session.

    Workers are started with ``start_new_session``, which gives them their own
    session but leaves them children of whoever started them -- the app, which
    keeps running.  A worker that exits on its idle timeout therefore stays in
    the process table until it is collected, and a zombie answers signal 0
    exactly as a running process does.  Believing it, the app never starts
    another, and every tool call for the rest of that session pays the cold
    start the worker exists to avoid.
    """
    import subprocess

    from src.agents_wrapper.fem_workflow.workers import worker_is_serving

    socket_path = tmp_path / "tools.sock"
    socket_path.write_bytes(b"")
    pid_path = tmp_path / "tools.pid"
    # A child that exits at once and is deliberately never waited on.
    child = subprocess.Popen(["true"], start_new_session=True)
    pid_path.write_text(str(child.pid), encoding="utf-8")
    for _ in range(500):
        try:
            line = Path(f"/proc/{child.pid}/stat").read_text(encoding="utf-8")
        except OSError:
            break
        if line[line.rindex(")") + 1:].split()[0] == "Z":
            break
        time.sleep(0.01)
    # This is what the old check asked, and why it was fooled: the pid answers.
    os.kill(child.pid, 0)

    assert worker_is_serving(socket_path, pid_path) is False
    # And the remains are cleared, so the next attempt starts one rather than
    # trusting what is left behind.
    assert not pid_path.exists() and not socket_path.exists()

    # A process that really is running is still reported as serving, including
    # one this process did not start -- which is the command line's case.
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    socket_path.write_bytes(b"")
    assert worker_is_serving(socket_path, pid_path) is True


def test_a_worker_holding_edited_tools_stands_down(tmp_path, socket_home, monkeypatch):
    """An edited tool must not keep being served from the old import.

    A resident worker holds the code it imported for as long as it lives, so an
    edit reaches every fresh subprocess and none of this worker's calls -- which
    surfaces as the tool rejecting an argument the caller can plainly see it
    accepts.  The worker says so and stops; the call runs here instead, and the
    next preload starts a worker with the code that is on disk.
    """
    session = tmp_path / "session"
    session.mkdir(parents=True)
    monitor = ToolMonitor(session)
    socket_path, pid_path = monitor._worker_paths("tools")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_worker(socket_path, _greet_and_answer(
        {"ok": False, "stale": True, "stdout": "",
         "stderr": "tool module changed on disk since this worker loaded it"}
    ))
    ran = []

    def fake_run(command, **kwargs):
        ran.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"glb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run
    )
    receipt = monitor.execute("export_surface_field", {
        "vtu": str(session / "s.vtu"), "output": str(session / "o.glb"),
        "field": "displacement", "component": "", "deform_scale": 0.0,
    })
    assert len(ran) == 1, "a stale worker's answer is not the tool's answer"
    assert receipt.returncode == 0
    # The worker's own complaint must not reach the receipt as a tool failure.
    assert "changed on disk" not in Path(receipt.stderr_path).read_text()


def test_a_served_tool_runs_where_the_subprocess_would_have(tmp_path, socket_home):
    """The worker is handed the working directory, not left in its own.

    A one-shot tool runs with the session directory as its working directory, so
    a relative path in the arguments means something there.  A resident worker
    sits wherever it was started, so without being told it would resolve the
    same argument against a different directory and quietly read or write the
    wrong file.
    """
    session = tmp_path / "session"
    session.mkdir(parents=True)
    monitor = ToolMonitor(session)
    socket_path, _ = monitor._worker_paths("tools")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = session / "out.glb"
    artifact.write_bytes(b"glb")
    seen, _thread = _fake_worker(socket_path, _greet_and_answer(
        {"ok": True, "stdout": "", "stderr": ""}
    ))
    monitor.execute("export_surface_field", {
        "vtu": str(session / "s.vtu"), "output": str(artifact),
        "field": "displacement", "component": "", "deform_scale": 0.0,
    })
    assert seen[0]["cwd"] == str(session.resolve())


def test_the_command_line_preloads_on_the_same_schedule_as_the_app(monkeypatch,
                                                                   tmp_path):
    """Both front ends want the same things warm at the same moments.

    The monitor uses a worker whenever one is listening, so a command-line run
    was already able to benefit -- but nothing in it ever started one, so it got
    the warm path only by accident, when an app session happened to have left a
    worker behind.  These are the calls that close that gap, and they have to
    line up with the app's: a light worker before the first tool, TripoSG while
    the geometry is being described, and the GPU imports bought during
    generation rather than in front of the first call that needs them.
    """
    from src.agents_wrapper.fem_workflow import run_workflow

    started = []
    monkeypatch.setattr(run_workflow, "ensure_tool_worker",
                        lambda root, env, tier="light": started.append(("tools", tier)))
    monkeypatch.setattr(run_workflow, "ensure_geometry_worker",
                        lambda root, session, env: started.append(("triposg", env)))
    monkeypatch.setattr(run_workflow, "ensure_partfield_worker",
                        lambda root, env: started.append(("partfield", env)))

    run_workflow.preload_for(tmp_path, tmp_path / "s", "mechfoundry-warp117",
                             {"kind": "geometry_input"})
    assert ("triposg", "mechfoundry-warp117") in started
    assert ("tools", "light") in started

    started.clear()
    run_workflow.after_answer(tmp_path, "mechfoundry-warp117", {"kind": "geometry_input"})
    assert started == [("tools", "cuda")], "generation is when the GPU imports are free"

    started.clear()
    run_workflow.preload_for(tmp_path, tmp_path / "s", "mechfoundry-warp117",
                             {"kind": "bc_input"})
    assert started == [("tools", "cuda")]

    started.clear()
    run_workflow.preload_for(tmp_path, tmp_path / "s", "mechfoundry-warp117", {
        "kind": "bc_input", "partfield": {"ready": True},
        "segmentation": {"method": "partfield"},
    })
    assert started == [("tools", "cuda"), ("partfield", "mechfoundry-warp117")]

    # A step that is not a card at all must not start anything.
    started.clear()
    run_workflow.preload_for(tmp_path, tmp_path / "s", "mechfoundry-warp117", "not a dict")
    run_workflow.after_answer(tmp_path, "mechfoundry-warp117", {"kind": "bc_input"})
    assert started == []


def test_the_command_line_colours_a_finished_solve_once(monkeypatch, tmp_path):
    """The same prefetch the app runs, on the same solve_complete edge.

    A command-line run has no viewer, but the session it leaves is opened in
    one.  Doing it here is what makes that session open warm -- and doing it
    once is what keeps a repeated card from starting it again.
    """
    from src.agents_wrapper.fem_workflow import run_workflow

    built = []
    monkeypatch.setattr(run_workflow, "solution_prefetch_requests",
                        lambda session: [{"output": "a.glb"}, {"output": "b.glb"}])
    monkeypatch.setattr(run_workflow, "prefetch_solution_assets",
                        lambda session, env, requests: built.append(len(requests)))
    seen = set()
    first = run_workflow.start_prefetch(tmp_path, "mechfoundry-warp117",
                                        {"solve_complete": True}, seen)
    assert first is not None
    first.join(timeout=5)
    again = run_workflow.start_prefetch(tmp_path, "mechfoundry-warp117",
                                        {"solve_complete": True}, seen)
    assert again is None, "the same solve is not coloured twice"
    assert built == [2]
    # A card that is not a finished solve starts nothing.
    assert run_workflow.start_prefetch(tmp_path, "mechfoundry-warp117",
                                       {"kind": "bc_input"}, seen) is None


def test_the_command_line_does_not_need_the_app_to_run():
    """The shared code is shared, not borrowed from the UI.

    Preloading used to live in the Streamlit controller, so giving it to the
    command line by importing from there would have made every command-line run
    depend on Streamlit, altair and pandas being installed and importable.
    """
    source = (
        Path(__file__).parents[1] / "fem_workflow" / "run_workflow.py"
    ).read_text(encoding="utf-8")
    assert "src.ui" not in source
    for module in ("assets", "workers"):
        shared = (
            Path(__file__).parents[1] / "fem_workflow" / f"{module}.py"
        ).read_text(encoding="utf-8")
        assert "streamlit" not in shared
        assert "src.ui" not in shared


def test_the_browser_never_leaves_the_session_s_own_folder(tmp_path):
    """One place decides what the file browser may reach.

    Its position is held as a path relative to the session folder and comes
    back from the page as a string, so the guard has to answer for a ``..``, an
    absolute path and a symlink pointing out of the session alike -- and for a
    folder deleted while somebody was standing in it, which is the same answer.
    """
    from src.ui.browse import breadcrumb, resolve_within

    root = tmp_path / "1d910c9f8f95"
    (root / "solve" / "vtu_surface").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_text("no", encoding="utf-8")
    (root / "escape").symlink_to(outside)

    assert resolve_within(root, "") == root.resolve()
    assert resolve_within(root, "solve/vtu_surface") == (
        root / "solve" / "vtu_surface"
    ).resolve()
    # Every way out is the same answer, and the caller falls back to the root.
    assert resolve_within(root, "..") is None
    assert resolve_within(root, "../elsewhere") is None
    assert resolve_within(root, str(outside)) is None
    assert resolve_within(root, "escape/secret.txt") is None
    # Gone is not a crash either: a solve's folder can be deleted underneath.
    assert resolve_within(root, "solve/never_existed") is None

    assert breadcrumb(root, root) == "1d910c9f8f95"
    assert breadcrumb(root, root / "solve" / "vtu_surface") == (
        "1d910c9f8f95 › solve › vtu_surface"
    )


def test_a_folder_lists_its_folders_before_its_files(tmp_path):
    """Folders first, then names, and no walking to draw a row.

    A solve folder here holds a thousand entries; sizing each folder in the
    listing would read the whole tree to show one screen of it, so folders
    carry no size at all and the icon says what they are.
    """
    from src.ui.browse import listing

    (tmp_path / "solve").mkdir()
    (tmp_path / "Geometry").mkdir()
    (tmp_path / "session.json").write_text("{}", encoding="utf-8")
    (tmp_path / "api_log.jsonl").write_bytes(b"x" * 40)

    rows = listing(tmp_path)
    assert [row.name for row in rows] == [
        "Geometry", "solve", "api_log.jsonl", "session.json",
    ], "folders first, then case-insensitive by name"
    assert [row.is_dir for row in rows] == [True, True, False, False]
    assert rows[0].size is None and rows[1].size is None
    assert rows[2].size == 40
    assert rows[2].mtime > 0
    # A folder that cannot be read is empty, not an exception: the listing is
    # redrawn on every click and one unreadable path must not take the page.
    assert listing(tmp_path / "not_here") == []


def test_one_file_is_sent_as_itself_and_anything_more_as_a_zip(tmp_path):
    """What the download button offers, and what it calls it."""
    from src.ui.browse import archive_name, needs_archive

    (tmp_path / "solve").mkdir()
    log = tmp_path / "log.txt"
    log.write_text("hello", encoding="utf-8")
    case = tmp_path / "case.json"
    case.write_text("{}", encoding="utf-8")

    assert needs_archive([log]) is False
    assert needs_archive([log, case]) is True
    assert needs_archive([tmp_path / "solve"]) is True

    # One thing is named for itself; several are named for where they came
    # from, session included, so a "solve.zip" is not one of four in a
    # downloads folder with nothing to tell them apart.
    here = tmp_path / "solve"
    assert archive_name(tmp_path, tmp_path, [tmp_path / "solve"]) == "solve.zip"
    assert archive_name(tmp_path, tmp_path, [log, case]) == f"{tmp_path.name}.zip"
    assert archive_name(tmp_path, here, [log, case]) == f"{tmp_path.name}-solve.zip"


def test_the_zip_carries_the_paths_the_browser_showed(tmp_path):
    """Members are named from the folder being browsed, and nothing else is.

    A zip taken in the session root holds ``solve/log.txt`` and one taken
    inside ``solve`` holds ``log.txt`` -- what was on screen is what comes out.
    An empty folder keeps an entry of its own rather than disappearing, and a
    symlink out of the session cannot carry a file out through the archive any
    more than the listing can navigate to one.
    """
    import io
    import zipfile

    from src.ui.browse import build_archive

    root = tmp_path / "session"
    (root / "solve" / "vtu_surface").mkdir(parents=True)
    (root / "solve" / "log.txt").write_text("residuals", encoding="utf-8")
    (root / "solve" / "vtu_surface" / "surface_t0000.vtu").write_text(
        "<VTKFile></VTKFile>", encoding="utf-8"
    )
    (root / "solve" / "empty").mkdir()
    (root / "session.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_text("no", encoding="utf-8")
    (root / "solve" / "escape").symlink_to(outside)

    whole = zipfile.ZipFile(io.BytesIO(build_archive(root, [root / "solve"])))
    assert set(whole.namelist()) == {
        "solve/log.txt",
        "solve/vtu_surface/surface_t0000.vtu",
        "solve/empty/",
    }
    assert whole.read("solve/log.txt") == b"residuals"
    # Nothing from beyond the session, by either name.
    assert not [name for name in whole.namelist() if "secret" in name]

    # The same folder browsed from inside it drops the prefix.
    inside = zipfile.ZipFile(
        io.BytesIO(build_archive(root / "solve", [root / "solve" / "log.txt"]))
    )
    assert inside.namelist() == ["log.txt"]

    # And a selection of loose files sits at the top of the archive.
    pair = zipfile.ZipFile(
        io.BytesIO(build_archive(root, [root / "session.json", root / "solve"]))
    )
    assert "session.json" in pair.namelist()


def test_a_download_too_large_to_hold_in_memory_is_refused(tmp_path):
    """The cap is decided on a walk, not on what a folder's inode says.

    Streamlit keeps a finished download in the server's own memory until the
    browser has taken it, and a session here runs to gigabytes -- so the figure
    the button is disabled on has to be the real weight of the selection.
    """
    from src.ui.browse import ARCHIVE_LIMIT, human_size, selection_bytes

    (tmp_path / "solve").mkdir()
    (tmp_path / "solve" / "a.vtu").write_bytes(b"x" * 1000)
    (tmp_path / "solve" / "deeper").mkdir()
    (tmp_path / "solve" / "deeper" / "b.vtu").write_bytes(b"x" * 2000)
    (tmp_path / "log.txt").write_bytes(b"x" * 7)

    assert selection_bytes([tmp_path / "log.txt"]) == 7
    assert selection_bytes([tmp_path / "solve"]) == 3000
    assert selection_bytes([tmp_path / "solve", tmp_path / "log.txt"]) == 3007
    # Something that is no longer there weighs nothing rather than raising.
    assert selection_bytes([tmp_path / "gone.txt"]) == 0

    assert ARCHIVE_LIMIT == 1 << 30
    assert human_size(0) == "0 B"
    assert human_size(3 * 1024 * 1024) == "3.0 MB"
    assert human_size(ARCHIVE_LIMIT) == "1.0 GB"


def test_what_the_preview_window_will_show(tmp_path):
    """Images, text, and an honest nothing for the rest.

    The window shows the head of a long file, not its tail: a surface or a case
    is read from the top, and a solve log's tail is what the Backend section is
    already for.  It says how much of the file it is showing, so the reader
    knows to download it for the rest.
    """
    from src.ui.browse import (
        PREVIEW_TEXT_LIMIT, download_mime, preview_kind, preview_text,
    )

    assert preview_kind(Path("preview.webp")) == "image"
    assert preview_kind(Path("render.PNG")) == "image"
    assert preview_kind(Path("log.txt")) == "text"
    assert preview_kind(Path("surface_t0000.vtu")) == "text"
    assert preview_kind(Path("session.json")) == "text"
    # A GLB, a checkpoint and a displacement buffer have nothing to show.
    for opaque in ("field.glb", "checkpoints.sqlite", "field.glb.disp.bin"):
        assert preview_kind(Path(opaque)) is None, opaque

    long = tmp_path / "log.txt"
    long.write_text("r" * (PREVIEW_TEXT_LIMIT + 500), encoding="utf-8")
    text, size = preview_text(long)
    assert len(text) == PREVIEW_TEXT_LIMIT
    assert size == PREVIEW_TEXT_LIMIT + 500, "the whole length is still reported"
    assert preview_text(tmp_path / "gone.txt")[0].startswith("Could not read")

    assert download_mime(Path("session.json")) == "application/json"
    assert download_mime(Path("field.glb")) == "model/gltf-binary"
    # And anything the table cannot name is bytes, rather than nothing at all.
    assert download_mime(Path("latents.pkl")) == "application/octet-stream"


def test_the_file_browser_is_a_tool_section_beside_the_other_three(tmp_path):
    """A fourth section, and the two layout rules it needs to be usable.

    The table is given the panel's height the same way the charts are, through
    a flex chain that every box down to it has to be told to join; and the path
    is written as its own line rather than a caption, because the container
    query drops every caption in this panel when it gets short and the path is
    the one thing that cannot go.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    # Appended, so an existing conversation still opens on Resources.
    assert 'TOOL_SECTIONS = ("Resources", "Solver", "Backend", "Files")' in source
    assert 'elif section == "Files":\n            render_files_tab(session_dir)' in source
    # Native checkboxes over a thousand rows, reported by position.
    assert "key=rows_key," in source
    assert 'on_select="rerun",' in source
    assert 'selection_mode="multi-row",' in source
    assert 'height="stretch",' in source
    # A real integer and a real timestamp, so the table's own sorting is right.
    assert 'st.column_config.NumberColumn("Size", format="bytes")' in source
    assert 'st.column_config.DatetimeColumn(' in source
    rule = source.index(
        '[data-testid="stElementContainer"]:has([data-testid="stDataFrame"]) {'
    )
    assert "flex: 1 1 auto !important" in source[rule : source.index("}", rule)]
    assert '<div class="browser-path">' in source
    move = source.index("move = st.session_state.pop(BROWSER_MOVE, None)")
    assert move < source.index("event = st.dataframe("), "settled before drawn"
    # Three things to press, on the path's own line, and no fourth: opening is
    # the row itself.  The heading is reserved above the table and filled in
    # afterwards, because what it says about the selection is only known then.
    assert 'heading = st.container(key="browser_heading")' in source
    assert source.index('heading = st.container(') < source.index("event = st.dataframe(")
    assert "where, up, again, take = st.columns(" in source
    for key in ("browser_up", "browser_refresh", "browser_download"):
        assert f'key="{key}"' in source, key
    assert "browser_open" not in source, "the Open button is gone"


def test_a_zip_is_built_when_it_is_clicked_and_not_before():
    """The download is a callable, never bytes handed over in advance.

    Streamlit runs a deferred callable only when the button is clicked, so
    standing in a folder of 3.8 MB surfaces reads none of them, and a rerun --
    of which this page has one every couple of seconds while a solve writes --
    does not pack the same selection again.  Handing it bytes would do both.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    assert "payload = lambda: build_archive(here, chosen)" in source
    assert "payload = lambda: chosen[0].open(\"rb\")" in source
    assert "data=payload," in source
    # Nowhere is the archive built while the page is merely being drawn: the
    # only mention of it anywhere in the page is the one inside that lambda.
    assert source.count("build_archive(") == 1
    assert "data=build_archive(" not in source
    # The preview window's own download is deferred for the same reason: a
    # surface VTU is megabytes, and the window is drawn to be looked at.
    assert 'data=lambda: path.open("rb"), file_name=path.name,' in source


def test_a_row_is_opened_by_double_clicking_it():
    """The row goes in the URL; nothing in the table is touched to say which.

    The listing draws itself on a canvas and reports only which rows are
    ticked, so there is no double-click to read and no way to ask it what the
    pointer was over.  The first attempt at this said which row by ticking it
    and unticking it again a moment later -- and the tick was visible, was
    sometimes still there afterwards, and was a choice nobody had made: ticks
    are what the download button reads.

    So the row is counted off in pixels instead, through however far the
    listing has been scrolled, and put in the query string; pressing refresh
    reruns the page and a rerun brings the URL with it.  Nothing is selected on
    the way, which is why nothing can be left selected.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    script = source[
        source.index("BROWSER_SCRIPT = ")
        : source.index("@st.cache_data(show_spinner=False, max_entries=256)")
    ]
    assert "listen('dblclick'" in script
    # Counted from the top of the rows and through the scroll, which is why a
    # heading may not sort: sorting would put the row under the pointer and the
    # row Python counts to out of step, silently.
    assert "Math.floor((y - HEADER + scrolled) / ROW)" in script
    assert "scroller ? scroller.scrollTop : 0" in script
    # The URL first, the rerun second, or the rerun would not carry it.
    assert script.index("history.replaceState") < script.index("refresh.click()")
    assert "'?%(open)s=' + at.index" in script
    assert 'st-key-browser_refresh' in script, "the rerun is the refresh button"
    # Python and the script have to agree on where a row starts, so the height
    # is pinned rather than left to whatever the theme would choose.
    assert "ROW_HEIGHT = 35" in source
    assert "row_height=ROW_HEIGHT," in source
    assert '"row": ROW_HEIGHT' in source
    # Nothing about the ticks is read or written when something is opened.
    handler = source[source.index("opened = st.query_params.get(BROWSER_OPEN)"):
                     source.index("    chosen = [entries[row]")]
    assert "index = int(opened) if opened.isdigit() else -1" in handler
    assert "if 0 <= index < len(entries):" in handler, "a row that has gone"
    for tick in ("BROWSER_ROWS", "rows_key", "selection", "rows)"):
        assert tick not in handler, tick
    # And the machinery that existed only to undo that tick is gone with it.
    for dead in ("BROWSER_TICKED", "BROWSER_RESTORE", "meant = set("):
        assert dead not in source, dead


def test_a_dismissed_preview_window_does_not_come_back():
    """Closing the window has to be closing it, whichever way it was closed.

    The file being shown is held in session state and the window is drawn from
    it on every run.  Its own Close button let go of that, but the corner cross
    and a click outside did not -- so the window went away, and then the next
    thing pressed anywhere on the page drew it again, and again after that.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    assert '@st.dialog("File preview", width="large", on_dismiss=forget_preview)' in source
    body = source[source.index("def forget_preview()") : source.index("@st.dialog")]
    assert "st.session_state.pop(BROWSER_PREVIEW, None)" in body
    # And the button in the window lets go through the same one place, so the
    # two ways out cannot drift apart again.
    closing = source[source.index("def preview_file("):]
    assert closing.count("forget_preview()") == 2
    assert "st.session_state.pop(BROWSER_PREVIEW" not in closing


def test_only_the_tick_column_picks_a_row_out():
    """A click on a name used to mark the row, in a way Python never saw.

    The grid draws its own outline around whatever cell was last clicked.  That
    outline is painted on the canvas, so no stylesheet can reach it, and it is
    never reported to Python -- but sitting next to a column of ticks that do
    mean "chosen", it read as though the row had been chosen, and then as
    though the choice had survived into the next folder.  Those clicks are
    stopped before the grid sees them.

    Only propagation is stopped and never the default, which is what leaves the
    parts of the table that are not selection working: a heading still sorts,
    the scrollbar still drags, the wheel is untouched, and the tick column --
    the one place a row may be picked out -- is let through.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    script = source[
        source.index("BROWSER_SCRIPT = ")
        : source.index("@st.cache_data(show_spinner=False, max_entries=256)")
    ]
    assert "for (const type of ['pointerdown', 'mousedown', 'click']) {" in script
    assert "listen(type, (event) => { if (spot(event)) event.stopPropagation(); });" in script
    # Stopping the default here would take sorting and the scrollbar with it.
    assert "preventDefault" not in script
    # One idea of what counts as a row, shared by the swallowing and by the
    # double-click, so a click cannot be stopped in a place a double-click is
    # still answered -- or the other way about.
    assert "const at = spot(event);" in script
    # Its own toolbar is its own business, and is hidden rather than swallowed.
    assert "stElementToolbar" in script
    toolbar = source.index('[data-testid="stElementToolbar"] {')
    assert "display: none !important" in source[toolbar : source.index("}", toolbar)]
    # The tick column is let through at the heading too, so "all of them"
    # still works even though the other headings no longer sort.
    assert "if (x < TICKS) return null;" in script and "TICKS = 48" in script


def test_entering_a_directory_starts_with_nothing_ticked():
    """Ticks belong to the folder they were made in.

    They are reported by position, so three rows down in one folder is a
    different file in the next, and carrying them across would hand the
    download button things nobody pointed at.  There is one way to change
    folder, so there is one place this can be got wrong.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    # Both ways of moving -- the button above, and a double-click on a folder
    # row -- go through the one helper.
    assert source.count("def browse_to(") == 1
    assert source.count("browse_to(root, ") == 2
    # Which records the move; and the move is the only thing that writes where
    # the browser is, so nothing can relocate it without clearing the ticks.
    assert source.count("st.session_state[BROWSER_AT] = ") == 1
    settled = source.index("move = st.session_state.pop(BROWSER_MOVE, None)")
    block = source[settled : source.index("entries = listing(here)")]
    assert "st.session_state[BROWSER_AT] = move" in block
    # And the ticks cannot come along, because the table entering a directory
    # is not the same table: it is named after the folder it shows, so there is
    # nothing to clear rather than a clearing that has to be remembered.
    assert 'rows_key = f"{BROWSER_ROWS}_{hashlib.sha1(str(here).encode())' in block
    assert "key != rows_key" in block, "the folders left behind are let go of"
    assert source.index("here = resolve_within(") < source.index("rows_key = ")


def test_the_bridge_replaces_whatever_it_left_behind_last_time():
    """A change here has to reach a page that is already open.

    The page keeps running while the script behind it changes: Streamlit reruns
    and swaps the frame, but nobody reloads the browser.  The first version
    marked the table it had taken up and skipped the work if the mark was
    there, so a page carried across a change found the mark from the version
    before and never wired up the new one -- the change appeared to have been
    made and to have done nothing, which is exactly how it was reported.

    So nothing is marked.  Each copy of the frame takes down what the last one
    registered and registers its own, on the document rather than on the table,
    which also means it does not matter whether the table exists yet.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    script = source[
        source.index("BROWSER_SCRIPT = ")
        : source.index("@st.cache_data(show_spinner=False, max_entries=256)")
    ]
    assert "for (const [type, fn] of (w.__femBrowse || [])) {" in script
    assert "doc.removeEventListener(type, fn, true);" in script
    assert "w.__femBrowse = wired;" in script
    # Taken down before anything of this copy's own goes up, or the two would
    # both be listening and one click would be answered twice.
    assert script.index("removeEventListener") < script.index("doc.addEventListener")
    # Nothing is written on the table itself any more; that mark was the bug.
    assert "dataset.femBrowse" not in script
    # Above whatever the grid registers, so a click reaches this first.
    assert "doc.addEventListener(type, fn, true);" in script


def test_a_path_too_long_for_its_line_loses_its_beginning():
    """The end of a path is the part worth keeping.

    The last name in it is the folder actually being looked at, and what comes
    before is the session it started from, which is the same for everything in
    this conversation and can be guessed.  The sidebar truncates the other way
    round because a conversation is named by its first words; a path is not.

    The buttons beside it are laid out here rather than by column weights,
    because a weight is a share of whatever width the panel happens to have and
    these want to be one size at any width.
    """
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    rule = source.index("  .browser-path {")
    block = source[rule : source.index("}", rule)]
    # Laying the box out right to left is what puts the ellipsis at the front.
    assert "direction: rtl" in block
    assert "text-overflow: ellipsis" in block and "white-space: nowrap" in block
    # The text keeps its own direction, so the separators stay as written.
    inner = source.index("  .browser-path > span {")
    assert "direction: ltr" in source[inner : source.index("}", inner)]
    assert '<div class="browser-path"><span>' in source
    # Same size as the rest of the page's text rather than a caption's.
    sized = source.index("  .browser-path, .browser-note {")
    assert "font-size: 0.875rem" in source[sized : source.index("}", sized)]
    # Square, and the path takes whatever is left.  Anchored to the heading,
    # because ":first-child" on a column is a rule the workspace split uses too.
    heading = source.index('[class*="st-key-browser_heading"] '
                           '[data-testid="stHorizontalBlock"] {')
    layout = source[heading:]
    first = layout.index('[data-testid="stColumn"]:first-child {')
    assert "min-width: 0" in layout[first : layout.index("}", first)], "so it can shrink"
    rest = layout.index('[data-testid="stColumn"]:not(:first-child) {')
    assert "flex: 0 0 2rem" in layout[rest : layout.index("}", rest)]
    square = source.index('[class*="st-key-browser_up"] button,')
    assert "height: 2rem" in source[square : source.index("}", square)]


def _idle_generate_surface(monkeypatch):
    """Import the generator script without the GPU stack it normally needs.

    The script imports torch and the TripoSG wrapper at module scope, neither of
    which is installed in the environment the tests run in -- but the part under
    test here is its socket loop, which touches none of it.  Stubs for the four
    names it imports are enough to reach that loop, and loading the file by path
    keeps this copy out of the way of anything else that imports it.
    """
    import importlib.util
    import sys
    import types

    stubs = {
        "torch": {"float16": object(), "float32": object()},
        "huggingface_hub": {"snapshot_download": lambda **kwargs: None},
        "src.triposg_wrapper.triposg_pipeline_wrapper": {
            "TripoSGPipelineWrapper": object()
        },
        "src.triposg_wrapper.utils": {
            "filter_largest_interior_component": lambda *a, **k: None,
            "sdf2surfacemesh": lambda *a, **k: None,
        },
    }
    for name, attributes in stubs.items():
        stub = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(stub, attribute, value)
        monkeypatch.setitem(sys.modules, name, stub)

    path = (
        Path(__file__).parents[1] / "fem_workflow" / "scripts" / "generate_surface.py"
    )
    spec = importlib.util.spec_from_file_location("generate_surface_idle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The weights are the one thing a stub cannot stand in for, and the loop
    # under test runs after they would have been loaded.
    monkeypatch.setattr(module.SurfaceGenerator, "load", lambda *a, **k: None)
    return module


def test_an_idle_geometry_worker_gives_the_gpu_back(monkeypatch, socket_home):
    """A resident TripoSG that nobody is asking must not hold the card forever.

    It keeps the model on the device between jobs -- 4.4 GB measured on a loaded
    scribble variant -- and the only thing that ever shut it down was a forward
    solve, which a conversation that generated a shape and stopped there never
    reaches.  Two were found alive on this machine at nine and thirty-one hours
    old, both belonging to apps that had already exited.
    """
    import argparse
    import socket as socket_module
    import tempfile
    import threading

    module = _idle_generate_surface(monkeypatch)
    # Short enough for AF_UNIX, which is why these never live under `tmp_path`.
    home = Path(tempfile.mkdtemp(prefix="implicit-fem-idle-"))
    socket_home(home)
    socket_path = home / "triposg.sock"
    job = argparse.Namespace(
        serve=socket_path, device="cpu", preload="scribble",
        weights_dir=None, idle_timeout=0.3,
    )

    stopped = threading.Event()

    def run():
        try:
            module.serve(job)
        finally:
            stopped.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # Asking for something resets the clock, so the worker is still there for
    # the next job rather than counting down from when it started.
    deadline = time.monotonic() + 5.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists(), "the worker never began serving"
    with socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM) as client:
        client.connect(os.fspath(socket_path))
        client.sendall(b'{"mode":"scribble"}\n')
        answer = json.loads(client.makefile("r", encoding="utf-8").readline())
    assert answer["ok"] is False, "a stubbed generator cannot have generated anything"
    assert not stopped.is_set(), "a worker that has just answered is not idle"

    assert stopped.wait(10.0), "an idle worker went on waiting"
    assert not socket_path.exists(), "the socket outlived the worker behind it"


def test_a_worker_is_stopped_without_waiting_out_a_greeting(monkeypatch, socket_home, tmp_path):
    """Both kinds of worker hear the shutdown, and neither is waited on.

    TripoSG's worker reads a request before it says anything, unlike every
    other model server, which greets first.  Stopping one waited for that
    greeting, so each switch away from TripoSG sat out the full 3 s timeout --
    between skipping the texture and the scaling card, where PartField is
    started in its place -- and then killed it rather than asking.
    """
    import argparse
    import threading

    from src.agents_wrapper.fem_workflow import workers

    module = _idle_generate_surface(monkeypatch)
    socket_path, _ = workers.worker_paths(tmp_path, "triposg", "mechfoundry-warp117")
    socket_home(socket_path.parent)
    job = argparse.Namespace(
        serve=socket_path, device="cpu", preload="scribble",
        weights_dir=None, idle_timeout=30.0,
    )
    stopped = threading.Event()

    def run():
        try:
            module.serve(job)
        finally:
            stopped.set()

    threading.Thread(target=run, daemon=True).start()
    deadline = time.monotonic() + 5.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists(), "the worker never began serving"
    started = time.monotonic()
    workers.stop_worker(tmp_path, "triposg", "mechfoundry-warp117")
    assert time.monotonic() - started < 1.0, "the stop waited for a greeting TripoSG never sends"
    assert stopped.wait(2.0), "TripoSG was not asked to shut down"

    socket_path, _ = workers.worker_paths(tmp_path, "partfield", "mechfoundry-partfield")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    seen, thread = _fake_worker(socket_path, _greet_and_answer({"ok": True}))
    workers.stop_worker(tmp_path, "partfield", "mechfoundry-partfield")
    thread.join(2.0)
    assert seen == [{"action": "shutdown"}], "a greeting worker no longer hears the shutdown"


def test_the_geometry_worker_is_launched_without_overriding_its_idle_timeout():
    """Both workers idle out, and neither front end has to remember to say so.

    The launch is a detached process nobody is waiting on, so an idle timeout
    passed at the call site would be one more thing to keep in step between the
    app and the command line; the default in the script is what holds.
    """
    source = (
        Path(__file__).parents[1] / "fem_workflow" / "workers.py"
    ).read_text(encoding="utf-8")
    assert "--idle-timeout" not in source
    generator = (
        Path(__file__).parents[1] / "fem_workflow" / "scripts" / "generate_surface.py"
    ).read_text(encoding="utf-8")
    server = (
        Path(__file__).parents[1] / "fem_workflow" / "scripts" / "tool_server.py"
    ).read_text(encoding="utf-8")
    assert "IDLE_TIMEOUT = 1800.0" in generator and "IDLE_TIMEOUT = 1800.0" in server


def test_a_guidance_scale_is_refused_where_the_model_cannot_use_one(monkeypatch, tmp_path):
    """The two modes on the distilled checkpoint say no rather than drop it.

    `img2latent` was the only branch that ever forwarded the number, so a text
    or scribble generation ran at the scribble pipeline's own default of 0.0
    while its receipt recorded the 7.0 it had been sent -- a parameter accepted,
    transmitted and silently discarded.  Forwarding it on all three branches
    would have been the wrong repair: TripoSG-scribble is CFG-distilled, and
    anything above 1 switches on a second denoising pass per step that the
    weights already account for.
    """
    import sys

    module = _idle_generate_surface(monkeypatch)
    scribble = tmp_path / "scribble.png"

    def parsed(*argv):
        monkeypatch.setattr(sys, "argv", ["generate_surface.py", *argv])
        return module.parse_args()

    common = ("--output", str(tmp_path / "out.glb"), "--device", "cpu",
              "--prompt", "a bracket", "--input", str(scribble))
    for mode in ("text", "scribble", "text_scribble"):
        job = parsed("--mode", mode, "--guidance-scale", "7.0", *common)
        with pytest.raises(ValueError, match="CFG-distilled"):
            module.generate(job)
        # Without the flag the same job gets as far as the model it needs.
        assert parsed("--mode", mode, *common).guidance_scale is None

    # Nothing is put in its place either: unasked, the image branch is left to
    # the pipeline's own default rather than given one to keep in step with it.
    assert parsed("--mode", "image", *common).guidance_scale is None
    assert parsed("--mode", "image", "--guidance-scale", "7.0",
                  *common).guidance_scale == 7.0
    source = (
        Path(__file__).parents[1] / "fem_workflow" / "scripts" / "generate_surface.py"
    ).read_text(encoding="utf-8")
    assert "guidance_scale=args.guidance_scale" not in source


def test_the_geometry_worker_is_handed_a_guidance_scale_only_when_there_is_one(
    tmp_path, socket_home
):
    """The warm path and the command line put the same arguments to the tool.

    The request this builds used to default the scale to 7.0 whatever the
    caller sent, which would now be the one route that still reaches the
    generator with a number it refuses.
    """
    import socket as socket_module

    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    socket_path, _ = monitor._worker_paths("triposg")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    def answer(connection, seen):
        seen.append(json.loads(connection.makefile("r", encoding="utf-8").readline()))
        connection.sendall(b'{"ok":true,"result":{"artifacts":[]}}\n')

    def ask(arguments):
        socket_path.unlink(missing_ok=True)
        seen, _thread = _fake_worker(socket_path, answer)
        result = monitor._run_geometry_worker(
            "generate_text_surface",
            {"output": str(session / "generated.glb"), **arguments},
            30,
        )
        assert result.returncode == 0
        return seen[0]

    assert ask({"prompt": "a bracket"})["guidance_scale"] is None
    assert ask({"prompt": "a bracket", "guidance_scale": 7.0})["guidance_scale"] == 7.0
    # The rest of the defaults are still filled in, so only this one changed.
    assert ask({"prompt": "a bracket"})["steps"] == 16


def test_the_graph_sends_a_guidance_scale_to_the_one_mode_that_has_one():
    """Submitted for every modality, it read as though it had been applied.

    The card carries one set of settings whatever is being generated, so the
    value exists for a text run too; what must not happen is its arriving at a
    tool that cannot use it, and being written into that run's receipt.
    """
    source = (
        Path(__file__).parents[1] / "fem_workflow" / "graph.py"
    ).read_text(encoding="utf-8")
    submission = source.index('args = {\n            "output": str(output)')
    guarded = source.index('if mode == "image":', submission)
    block = source[submission:guarded]
    assert '"guidance_scale"' not in block, "not among the arguments every mode sends"
    assert guarded < source.index("if state.input_asset:", submission)
    assert 'args["guidance_scale"] = state.generation.guidance_scale' in source


def test_a_forward_run_is_not_asked_to_preserve_bc_regions():
    """Preservation exists for design updates, and a forward run makes none.

    The settings card stores its method ("mask" by default) whatever the run mode -- the control is only
    drawn for a shape optimization -- so a forward solve used to reach
    `_init_bc_graft` at its first step and cut the exact BC patch off the
    initial surface, validating boundary loops for a stitch that never runs.
    The background frozen mask beside it is `bg_n**3` wide and read only by the
    reinit that follows a remesh.
    """
    import importlib.util

    from src.agents_wrapper.fem_workflow.models import ShapeOptSettings

    path = (
        Path(__file__).parents[1] / "fem_workflow" / "scripts" / "workflow_tool.py"
    )
    spec = importlib.util.spec_from_file_location("workflow_tool_bc", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    optimizing = ShapeOptSettings(run_mode="shape_optimization")
    assert tool._bc_preservation(optimizing) == {"sdf_bc_fix_method": "mask"}
    # Whatever method was chosen travels, not just the default.
    assert tool._bc_preservation(
        ShapeOptSettings(run_mode="shape_optimization", sdf_bc_fix_method="zip")
    ) == {"sdf_bc_fix_method": "zip"}

    forward = ShapeOptSettings(run_mode="forward", sdf_bc_fix_method="zip")
    assert forward.sdf_bc_fix_method == "zip", "the stored setting is untouched"
    # It is the solver that is told not to build any of it, rather than the
    # method being swapped for a cheaper one that still builds a mask.
    assert tool._bc_preservation(forward) == {"preserve_bc_regions": False}

    source = path.read_text(encoding="utf-8")
    assert '"sdf_bc_fix_method": settings.sdf_bc_fix_method,' not in source
    assert "**_bc_preservation(settings)," in source


def test_graft_is_retired_from_the_workflow_and_old_sessions_read_it_as_mask(
    tmp_path, caplog,
):
    """The workflow offers "zip" and "mask" only; a stored "graft" loads as "mask".

    "graft" stays in the solver core for the examples and figures, but the
    settings card no longer lists it.  Sessions, plans and case.json files
    written before that still carry it and must load and run, so the value is
    mapped on load (with a logged note) rather than rejected -- and the case
    run_forward reads therefore cannot hand "graft" to the solver.
    """
    import importlib.util
    import logging

    from pydantic import ValidationError

    from src.agents_wrapper.fem_workflow.models import (
        BC_FIX_METHODS, ShapeOptSettings, WorkflowState,
    )

    assert BC_FIX_METHODS == ("zip", "mask")
    assert ShapeOptSettings().sdf_bc_fix_method == "mask"

    # An old session.json, as stored, with an optimization set to graft.
    old = WorkflowState(session_id="s", session_dir=str(tmp_path)).model_dump(mode="json")
    old["shape_opt"].update(run_mode="shape_optimization", sdf_bc_fix_method="graft")
    with caplog.at_level(logging.INFO, logger="src.agents_wrapper.fem_workflow.models"):
        state = WorkflowState.model_validate_json(json.dumps(old))
    assert state.shape_opt.sdf_bc_fix_method == "mask"
    assert any("graft" in record.getMessage() for record in caplog.records)

    # An old case.json's shape_opt goes through the same model in run_forward,
    # so what reaches the solver is "mask".
    path = Path(__file__).parents[1] / "fem_workflow" / "scripts" / "workflow_tool.py"
    spec = importlib.util.spec_from_file_location("workflow_tool_graft", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    case_settings = ShapeOptSettings.model_validate(old["shape_opt"])
    assert tool._bc_preservation(case_settings) == {"sdf_bc_fix_method": "mask"}
    assert 'ShapeOptSettings.model_validate(case["shape_opt"])' in path.read_text(
        encoding="utf-8"
    )

    # Any other method outside the offered pair is still refused.
    with pytest.raises(ValidationError):
        ShapeOptSettings(sdf_bc_fix_method="snap")

    # The settings card lists exactly the offered methods.
    app = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    card = app[app.index('"BC preservation"'):]
    card = card[:card.index(")", card.index("index="))]
    assert "BC_FIX_METHODS" in card and "graft" not in card
    assert '"graft"' not in app


def test_the_legend_says_which_way_a_segment_points():
    """Position alone puts an underside and a seat at the same address.

    "mid x, at the bottom, mid z" is true of the flat face across the bottom of
    a sofa and would be true of a low seat, and the stage called the underside
    a seat.  Which way the area points is already measured per segment and
    separates them outright.  Up and down get words, because they mean the same
    for every object; x and z keep the axis form, because which end of z is the
    front is a fact about the object and not one to assert here.
    """
    from src.agents_wrapper.fem_workflow.specialists import (
        PREVIEW_SIZE, SpecialistCalls, _dominant_facing,
    )

    assert _dominant_facing({"-y": 0.97, "+z": 0.03}) == "faces downward"
    assert _dominant_facing({"+y": 0.5, "-x": 0.19}) == "faces upward"
    assert _dominant_facing({"-z": 1.0}) == "faces the back"
    assert _dominant_facing({"+z": 0.65, "+y": 0.27}) == "faces the front"
    # The two sides of a body share its area, so neither reaches the bar alone.
    assert _dominant_facing({"+x": 0.36, "-x": 0.36}) == "faces both x sides"
    # A shape whose area agrees on nothing says nothing rather than guessing.
    assert _dominant_facing({"+x": 0.2, "-y": 0.2, "+z": 0.2}) == ""
    assert _dominant_facing(None) == "" and _dominant_facing({}) == ""

    legend = SpecialistCalls._segment_legend(
        {
            "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "segments": [
                {"id": 0, "color": "#1f77b4", "centroid": [0.5, 0.0, 0.5],
                 "area_fraction": 0.18,
                 "facing_fraction_45deg": {"-y": 0.97, "+z": 0.03}},
                {"id": 1, "color": "#d62728", "centroid": [0.5, 0.5, 0.7],
                 "area_fraction": 0.24,
                 "facing_fraction_45deg": {"+y": 0.5, "-x": 0.19}},
            ],
        }
    )
    assert legend["0"].endswith("at the bottom, mid depth, faces downward")
    assert legend["1"].endswith("at mid height, at the front, faces upward")
    # A segmentation that never measured facing still gets its old legend.
    assert SpecialistCalls._segment_legend(
        {"bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
         "segments": [{"id": 0, "color": "#1f77b4", "centroid": [0.5, 0.0, 0.5]}]}
    ) == {"0": "blue, mid x, at the bottom, mid depth"}

    # Six panels divide the montage's long edge by three before anything is
    # read, and the parts a PartField segmentation splits into are a finer
    # judgment than the blocks that preceded it.
    assert PREVIEW_SIZE["bc_synthesis"] == 1024


def test_which_end_of_the_object_is_its_front_is_stated_rather_than_guessed():
    """Asserted, because leaving it open cost more than getting it wrong can.

    The generator builds its output facing +z, so the montage's +z panel is the
    front view.  Left to infer that, the stage named the back face of a sofa
    the "front panel" in the same breath as calling the segment facing the
    other way the backrest; a rule saying the two cannot both be the front, and
    the measured relation between them in the legend, each moved it not at all.
    Naming the ends outright did, and the same call then read every segment
    correctly.  The brief has to carry the convention too, or the panel titles
    stay unreadable in exactly the same way.
    """
    from src.agents_wrapper.fem_workflow.specialists import (
        FRONT_AXIS, VISION_BRIEFS, _bounded_position, _dominant_facing,
    )

    assert FRONT_AXIS == "+z"
    box = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert _bounded_position([0.5, 0.5, 0.9], box).endswith("at the front")
    assert _bounded_position([0.5, 0.5, 0.1], box).endswith("at the back")
    assert _bounded_position([0.5, 0.5, 0.5], box).endswith("mid depth")
    # The two ends read the same way whether they are a position or a facing.
    assert _dominant_facing({"+z": 0.9}) == "faces the front"
    assert _dominant_facing({"-z": 0.9}) == "faces the back"
    # No axis letter survives into the words for the two settled directions.
    for text in (_bounded_position([0.5, 0.9, 0.9], box), _dominant_facing({"+z": 0.9})):
        assert " z" not in text and "+z" not in text and "-z" not in text
    subject = VISION_BRIEFS["bc_synthesis"][0]
    assert "front faces +z" in subject


def test_every_stage_that_resolves_a_direction_is_told_where_the_front_is():
    """One convention, stated wherever a direction becomes a number.

    The vision stage named the back of a sofa its front until the montage brief
    said which end of z the object faces.  The text stages turn the same words
    into vectors -- a traction, a `half_space` normal, an `aligned_normal`
    target -- and were told only about y, so "push on the front" left them the
    same guess.  These references are prose and cannot import the constant, so
    the tie is made here: change `FRONT_AXIS` and the documents that contradict
    it fail rather than quietly disagreeing with the legend.
    """
    from src.agents_wrapper.fem_workflow.specialists import FRONT_AXIS

    back = f"-{FRONT_AXIS[1]}"
    references = (
        Path(__file__).parents[1] / "fem_workflow" / "skill" / "references"
    )
    # Where a direction in words becomes a component of a vector.
    def prose(name):
        """One line of it, so a sentence that wraps still reads as a sentence."""
        return " ".join((references / name).read_text(encoding="utf-8").split())

    for name in ("boundary_conditions.md", "boundary_condition_values.md",
                 "boundary_condition_prompt.md"):
        text = prose(name)
        assert f"forward is `{FRONT_AXIS}` and backward is `{back}`" in text, name
        # The y half was always there and has to stay: a stage given one axis
        # and not the other is the state this came from.
        assert "upward is `+y`" in text and "downward is `-y`" in text, name

    # The scale stage maps a named dimension onto an axis rather than resolving
    # a direction, so it states the same convention in its own terms -- and
    # "deep" belongs to the axis the object faces along, not to height.
    scale = prose("physical_scale.md")
    assert f"front-back is `{FRONT_AXIS[1]}`" in scale
    assert f"depth and deep are `{FRONT_AXIS[1]}`" in scale
    assert "deep are `y`" not in scale


def _ready_text2cad(interpreter="/envs/mechfoundry-text2cad/bin/python"):
    return SimpleNamespace(
        ready=True, interpreter=interpreter, environment="mechfoundry-text2cad",
        summary="Text2CAD ready", model_dump=lambda: {"ready": True},
    )


@pytest.mark.parametrize("kind", ["text2cad", "cadcoder"])
def test_cad_generators_are_text_only_choices(monkeypatch, tmp_path, kind):
    """Choosing a CAD generator sends the text to its own tool with only its settings."""
    pytest.importorskip("langgraph")

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    calls = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, arguments))
            output = Path(arguments.get("output") or arguments["output_npz"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"stub")
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a plate"

    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, f"{kind}_status", _ready_text2cad)
    workflow = WorkflowState(session_id="c", session_dir=str(tmp_path), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "c"}}
    first = graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    card = first["__interrupt__"][0].value
    assert "generator" in card["fields"] and card[kind] == {"ready": True}
    assert card["generator"] == "triposg"

    after = graph.invoke(Command(resume={
        "description": "a plate with four holes", "generator": kind,
        "generation_settings": {"seed": 7, "faces": 5000},
    }), config=config)
    name, arguments = next(call for call in calls if call[0].startswith("generate_"))
    assert name == f"generate_{kind}_text_surface"
    assert arguments == {"output": str(tmp_path / "geometry" / "generated.glb"),
                         "prompt": "a plate with four holes", "seed": 7, "faces": 5000}
    state = WorkflowState.model_validate(after["workflow"])
    assert state.generation.generator == kind


@pytest.mark.parametrize("kind", ["text2cad", "cadcoder"])
@pytest.mark.parametrize("answer, message", [
    ({"asset_path": "sketch.png", "asset_kind": "scribble"}, "text only"),
    ({}, "not ready"),
])
def test_cad_generators_refuse_images_and_an_uninstalled_model(
        monkeypatch, tmp_path, answer, message, kind):
    pytest.importorskip("langgraph")

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    status = SimpleNamespace(ready=False, summary="not installed",
                             model_dump=lambda: {"ready": False})
    monkeypatch.setattr(graph_module, f"{kind}_status", lambda: status)
    workflow = WorkflowState(session_id="r", session_dir=str(tmp_path), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "r"}}
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    with pytest.raises(ValueError, match=message):
        graph.invoke(Command(resume={
            "description": "a bracket", "generator": kind, **answer,
        }), config=config)


def _ready_cad(kind):
    """A CAD generator table entry that reports ready, in its own environment."""
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    status_of, settings_of, label = monitor_module.CAD_GENERATORS[kind]
    ready = SimpleNamespace(
        ready=True, interpreter=f"/envs/mechfoundry-{kind}/bin/python",
        environment=f"mechfoundry-{kind}", summary="ready", model_dump=lambda: {"ready": True},
    )
    return (lambda: ready, lambda: (f"mechfoundry-{kind}", None, None), label)


@pytest.mark.parametrize("kind", ["text2cad", "cadcoder"])
def test_only_one_generator_model_is_resident(tmp_path, monkeypatch, kind):
    """A CAD model evicts every other model, and the others (and a solve) evict it."""
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    session = tmp_path / "session"
    (session / "case").mkdir(parents=True)
    monitor = ToolMonitor(session)
    unloads, launches = [], []
    for name in ("geometry", "partfield", "vomp"):
        monkeypatch.setattr(ToolMonitor, f"_shutdown_{name}_worker",
                            lambda self, name=name: unloads.append(name))
    monkeypatch.setattr(ToolMonitor, "_shutdown_cad_workers",
                        lambda self, keep=None: unloads.append(f"cad except {keep}"))
    monkeypatch.setattr(ToolMonitor, "_shutdown_hunyuan3d_worker",
                        lambda self: unloads.append("hunyuan3d"))
    monkeypatch.setattr(ToolMonitor, "_run_geometry_worker",
                        lambda self, name, arguments, timeout: None)
    monkeypatch.setattr(ToolMonitor, "_run_worker",
                        lambda self, worker, env, tail, timeout, **kw: launches.append(
                            (worker, env, kw)) or None)
    monkeypatch.setitem(monitor_module.CAD_GENERATORS, kind, _ready_cad(kind))
    monkeypatch.setattr("src.agents_wrapper.fem_workflow.workers.ensure_cad_worker",
                        lambda root, worker: launches.append(("ensure", root, worker)))
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--output" in command:
            Path(command[command.index("--output") + 1]).write_bytes(b"glb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(monitor_module.subprocess, "run", fake_run)
    surface = session / "geometry" / "cad.glb"
    surface.parent.mkdir(parents=True)
    monitor.execute(f"generate_{kind}_text_surface",
                    {"output": str(surface), "prompt": "a plate", "seed": 3})
    assert unloads == ["geometry", "partfield", f"cad except {kind}", "hunyuan3d", "vomp"]
    assert launches[0] == ("ensure", session.parent, kind)
    assert launches[1] == (kind, f"mechfoundry-{kind}", {"wait_for_start": True})
    # No worker answered, so it ran once through the generator's interpreter itself.
    assert commands[0][0] == f"/envs/mechfoundry-{kind}/bin/python"
    assert commands[0][1].endswith(f"{kind}_wrapper/generate.py")
    assert commands[0][-4:] == ["--prompt", "a plate", "--seed", "3"]

    unloads.clear()
    monitor.execute("generate_text_surface",
                    {"output": str(session / "geometry" / "t.glb"), "prompt": "a sofa"})
    assert unloads == ["partfield", "cad except None", "hunyuan3d", "vomp"]

    unloads.clear()
    solve_output = session / "solve"
    solve_output.mkdir()
    case = session / "case" / "case.json"
    case.write_text(json.dumps({"output": str(solve_output)}), encoding="utf-8")
    monitor.execute("run_forward", {"case": str(case)})
    assert unloads == ["geometry", "partfield", "cad except None", "hunyuan3d", "vomp"]


def test_solve_starts_only_after_stopped_model_workers_have_ended(tmp_path, monkeypatch):
    """A resident paint worker (here one that ignores the polite stop) is gone -- its process
    ended, so its memory is free -- before the solve runs, not merely asked to leave."""
    import subprocess as real_subprocess
    import sys
    from src.agents_wrapper.fem_workflow import monitor as monitor_module
    from src.agents_wrapper.fem_workflow.runtime_files import worker_paths
    from src.hunyuan3d_wrapper.availability import settings as hunyuan3d_settings

    session = tmp_path / "session"
    (session / "case").mkdir(parents=True)
    worker = real_subprocess.Popen(
        [sys.executable, "-c",
         "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
        start_new_session=True)
    try:
        time.sleep(0.3)         # the interpreter has installed its handler
        _, pid_path = worker_paths(session.parent, "hunyuan3d", hunyuan3d_settings()[0])
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(worker.pid), encoding="utf-8")
        monkeypatch.setattr(monitor_module, "MODEL_EXIT_WAIT", 0.5)
        alive_at_solve = []

        def fake_run(command, **kwargs):
            alive_at_solve.append(worker.poll() is None)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(monitor_module.subprocess, "run", fake_run)
        solve_output = session / "solve"
        solve_output.mkdir()
        case = session / "case" / "case.json"
        case.write_text(json.dumps({"output": str(solve_output)}), encoding="utf-8")
        ToolMonitor(session).execute("run_forward", {"case": str(case)})
        assert alive_at_solve == [False]
        assert not pid_path.exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_waiting_out_a_worker_kills_only_the_process_it_recorded():
    """A worker still alive past the wait is killed -- by itself when it leads no process group --
    but a pid that now runs another command (reused) is left alone."""
    import subprocess as real_subprocess
    import sys
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    member = real_subprocess.Popen([sys.executable, "-c", code])          # not a group leader
    other = real_subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        time.sleep(0.3)
        assert os.getpgid(member.pid) != member.pid
        monitor_module._wait_for_exit(
            [(member.pid, monitor_module._cmdline(member.pid)), (other.pid, b"another command\0")], 0.3)
        member.wait(timeout=5)                           # killed (the wait reaped it), this test still runs
        assert member.poll() is not None
        assert other.poll() is None                      # not the recorded command: left alone
    finally:
        for process in (member, other):
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize("kind", ["text2cad", "cadcoder"])
def test_cad_workers_launch_through_their_interpreters(monkeypatch, tmp_path, kind):
    from src.agents_wrapper.fem_workflow import monitor as monitor_module
    from src.agents_wrapper.fem_workflow import workers

    stopped, calls = [], []

    class Process:
        pid = os.getpid()

    for name in monitor_module.CAD_GENERATORS:
        monkeypatch.setitem(monitor_module.CAD_GENERATORS, name, _ready_cad(name))
    monkeypatch.setattr(workers, "stop_worker",
                        lambda root, worker, env: stopped.append(worker))
    monkeypatch.setattr(workers.subprocess, "Popen",
                        lambda command, **kwargs: calls.append(command) or Process())
    assert workers.ensure_cad_worker(tmp_path, kind)
    assert workers.ensure_cad_worker(tmp_path, kind)
    assert len(calls) == 1
    assert calls[0][0] == f"/envs/mechfoundry-{kind}/bin/python"
    assert calls[0][2].endswith(f"{kind}_wrapper/server.py") and "--serve" in calls[0]
    # Every other model is stopped, and this one is left alone.
    assert set(stopped) == ({"triposg", "partfield", "hunyuan3d", "vomp"}
                            | (set(monitor_module.CAD_GENERATORS) - {kind}))


def test_both_front_ends_preload_the_chosen_generator(monkeypatch, tmp_path):
    from src.ui import controller as controller_module

    started = []
    monkeypatch.setattr(run_workflow, "ensure_tool_worker", lambda *a, **k: None)
    monkeypatch.setattr(run_workflow, "ensure_geometry_worker",
                        lambda root, session, env: started.append("triposg"))
    monkeypatch.setattr(run_workflow, "ensure_cad_worker",
                        lambda root, kind, env: started.append(kind))
    monkeypatch.setattr(run_workflow, "ensure_hunyuan3d_worker",
                        lambda root, env: started.append("hunyuan3d"))
    for generator in ("text2cad", "cadcoder", "hunyuan3d", "triposg"):
        run_workflow.preload_for(tmp_path, tmp_path / "s", "mechfoundry-warp117",
                                 {"kind": "geometry_input", "generator": generator})
    assert started == ["text2cad", "cadcoder", "hunyuan3d", "triposg"]

    started.clear()
    monkeypatch.setattr(controller_module, "ensure_geometry_worker",
                        lambda root, session, env: started.append("triposg"))
    monkeypatch.setattr(controller_module, "ensure_cad_worker",
                        lambda root, kind, env: started.append(kind))
    monkeypatch.setattr(controller_module, "ensure_hunyuan3d_worker",
                        lambda root, env: started.append("hunyuan3d"))
    controller = WorkflowController(tmp_path)
    session = tmp_path / "abc"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({
        "compute_env": "mechfoundry-warp117", "generation": {"generator": "cadcoder"},
    }), encoding="utf-8")
    controller.preload_geometry_model("abc")  # the session's last choice
    controller.preload_geometry_model("abc", generator="triposg")
    controller.preload_geometry_model("abc", generator="hunyuan3d")
    assert started == ["cadcoder", "triposg", "hunyuan3d"]


def _ready_hunyuan3d():
    return SimpleNamespace(
        ready=True, interpreter="/envs/mechfoundry-hunyuan3d/bin/python",
        environment="mechfoundry-hunyuan3d", summary="ready", model_dump=lambda: {"ready": True},
    )


def _image_graph(monkeypatch, tmp_path, status):
    """A graph at the geometry card, with tools and the vision stage faked."""
    from langgraph.checkpoint.memory import MemorySaver

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    calls = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, arguments))
            output = Path(arguments.get("output") or arguments["output_npz"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"stub")
            if name == "stage_asset":
                output.with_suffix(output.suffix + ".json").write_text(json.dumps(
                    {"source_name": "photo.png", "sha256": "0" * 64}))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a mug"

    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "hunyuan3d_status", lambda: status)
    workflow = WorkflowState(session_id="h", session_dir=str(tmp_path), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "h"}}
    first = graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    return graph, config, first["__interrupt__"][0].value, calls


def test_hunyuan3d_generates_from_the_image_with_its_own_settings(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow.models import WorkflowState

    graph, config, card, calls = _image_graph(monkeypatch, tmp_path, _ready_hunyuan3d())
    assert card["hunyuan3d"] == {"ready": True}
    photo = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "red").save(photo)
    after = graph.invoke(Command(resume={
        "description": "a red mug", "asset_path": str(photo), "asset_kind": "image",
        "generator": "hunyuan3d",
        "generation_settings": {"seed": 5, "faces": 8000, "inference_steps": 20,
                                "hunyuan3d_octree_resolution": 256, "texture": False,
                                "hunyuan3d_flashvdm": False, "hunyuan3d_texture_size": 2048,
                                "hunyuan3d_upscale": True},
    }), config=config)
    name, arguments = next(call for call in calls if call[0].startswith("generate_"))
    assert name == "generate_hunyuan3d_image_surface"
    assert arguments == {
        "output": str(tmp_path / "geometry" / "generated.glb"),
        "input": str(tmp_path / "inputs" / "original.png"),
        "seed": 5, "faces": 8000, "steps": 20, "octree_resolution": 256, "texture": False,
        "flashvdm": False, "texture_size": 2048, "upscale": True,
    }
    state = WorkflowState.model_validate(after["workflow"])
    assert state.generation.generator == "hunyuan3d" and state.cad_faces_path is None


def test_hunyuan3d_defaults_to_30_steps_and_the_fast_settings(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    graph, config, _, calls = _image_graph(monkeypatch, tmp_path, _ready_hunyuan3d())
    photo = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "red").save(photo)
    graph.invoke(Command(resume={
        "description": "a red mug", "asset_path": str(photo), "asset_kind": "image",
        "generator": "hunyuan3d",
    }), config=config)
    _, arguments = next(call for call in calls if call[0].startswith("generate_"))
    assert arguments["steps"] == 30
    assert (arguments["flashvdm"], arguments["texture_size"], arguments["upscale"]) == (True, 1024, False)


@pytest.mark.parametrize("generator, settings, steps", [
    (None, {"seed": 0}, 50),                        # an image keeps its 50 beside a seed
    ("hunyuan3d", {"seed": 0}, 30),
    (None, {"seed": 0, "inference_steps": 20}, 20),  # given, kept
])
def test_settings_without_steps_keep_the_route_s_default(monkeypatch, tmp_path, generator, settings, steps):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    graph, config, _, calls = _image_graph(monkeypatch, tmp_path, _ready_hunyuan3d())
    photo = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "red").save(photo)
    graph.invoke(Command(resume={
        "description": "a red mug", "asset_path": str(photo), "asset_kind": "image",
        **({"generator": generator} if generator else {}), "generation_settings": settings,
    }), config=config)
    _, arguments = next(call for call in calls if call[0].startswith("generate_"))
    assert arguments["steps"] == steps


@pytest.mark.parametrize("answer, ready, message", [
    ({}, True, "from an image"),
    ({"asset_path": "sketch.png", "asset_kind": "scribble"}, True, "from an image"),
    ({"asset_path": "photo.png", "asset_kind": "image"}, False, "not ready"),
])
def test_hunyuan3d_needs_an_image_and_an_installed_model(
        monkeypatch, tmp_path, answer, ready, message):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    status = (_ready_hunyuan3d() if ready else
              SimpleNamespace(ready=False, summary="not installed",
                              model_dump=lambda: {"ready": False}))
    graph, config, _, calls = _image_graph(monkeypatch, tmp_path, status)
    with pytest.raises(ValueError, match=message):
        graph.invoke(Command(resume={
            "description": "a mug", "generator": "hunyuan3d", **answer,
        }), config=config)
    assert not any(name.startswith("generate_") for name, _ in calls)


def test_hunyuan3d_is_resident_alone(tmp_path, monkeypatch):
    """Hunyuan3D-2.1 evicts every other model, and each of them evicts it."""
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    unloads, launches = [], []
    for name in ("geometry", "partfield", "hunyuan3d", "vomp"):
        monkeypatch.setattr(ToolMonitor, f"_shutdown_{name}_worker",
                            lambda self, name=name: unloads.append(name))
    monkeypatch.setattr(ToolMonitor, "_shutdown_cad_workers",
                        lambda self, keep=None: unloads.append(f"cad except {keep}"))
    monkeypatch.setattr(ToolMonitor, "_run_worker",
                        lambda self, worker, env, tail, timeout, **kw: launches.append(
                            (worker, env, kw)) or None)
    monkeypatch.setattr(monitor_module, "hunyuan3d_status", _ready_hunyuan3d)
    monkeypatch.setattr("src.agents_wrapper.fem_workflow.workers.ensure_hunyuan3d_worker",
                        lambda root, paint=False, **kw: launches.append(("ensure", root)))
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"glb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(monitor_module.subprocess, "run", fake_run)
    photo = session / "inputs" / "original.png"
    photo.parent.mkdir()
    photo.write_bytes(b"png")
    (session / "geometry").mkdir()
    monitor.execute("generate_hunyuan3d_image_surface", {
        "output": str(session / "geometry" / "generated.glb"), "input": str(photo),
        "seed": 3, "octree_resolution": 256, "texture": False,
        "flashvdm": False, "texture_size": 2048, "upscale": True,
    })
    assert unloads == ["geometry", "partfield", "cad except None", "vomp"]
    assert launches == [("ensure", session.parent),
                        ("hunyuan3d", "mechfoundry-hunyuan3d", {"wait_for_start": True})]
    # No worker answered, so it ran once through its own interpreter.
    assert commands[0][0] == "/envs/mechfoundry-hunyuan3d/bin/python"
    assert commands[0][1].endswith("hunyuan3d_wrapper/generate.py")
    assert commands[0][-9:] == ["--seed", "3", "--octree-resolution", "256", "--texture-size",
                                "2048", "--no-texture", "--no-flashvdm", "--upscale"]

    monitor_module_status = SimpleNamespace(ready=False, summary="gone", interpreter=None)
    monkeypatch.setattr(monitor_module, "hunyuan3d_status", lambda: monitor_module_status)
    with pytest.raises(ToolRejected, match="Hunyuan3D-2.1 is not ready"):
        monitor.execute("generate_hunyuan3d_image_surface", {
            "output": str(session / "geometry" / "other.glb"), "input": str(photo),
        })


def test_hunyuan3d_worker_launches_through_its_interpreter(monkeypatch, tmp_path):
    from src.agents_wrapper.fem_workflow import monitor as monitor_module
    from src.agents_wrapper.fem_workflow import workers

    stopped, calls = [], []

    class Process:
        pid = os.getpid()

    monkeypatch.setattr(workers, "hunyuan3d_status", _ready_hunyuan3d)
    monkeypatch.setattr(workers, "stop_worker",
                        lambda root, worker, env: stopped.append(worker))
    monkeypatch.setattr(workers.subprocess, "Popen",
                        lambda command, **kwargs: calls.append(command) or Process())
    assert workers.ensure_hunyuan3d_worker(tmp_path)
    assert workers.ensure_hunyuan3d_worker(tmp_path)
    assert len(calls) == 1
    assert calls[0][0] == "/envs/mechfoundry-hunyuan3d/bin/python"
    assert calls[0][2].endswith("hunyuan3d_wrapper/server.py") and "--serve" in calls[0]
    assert set(stopped) == {"triposg", "partfield", "vomp", *monitor_module.CAD_GENERATORS}

    # Started for painting it loads only the paint stage, and leaves VoMP be:
    # the two share the material card and fit on one GPU together.
    monkeypatch.setattr(workers, "worker_is_serving", lambda socket_path, pid_path: False)
    for marker in tmp_path.rglob("hunyuan3d.attempted"):
        marker.unlink()  # as a stop would, so the relaunch is not held back
    stopped.clear(), calls.clear()
    assert workers.ensure_hunyuan3d_worker(tmp_path, paint=True)
    assert calls[0][-1] == "--paint"
    assert "vomp" not in stopped

    # Every other model's start stops it in turn.
    stopped.clear()
    monkeypatch.setattr(workers, "_launch", lambda *args, **kwargs: True)
    monkeypatch.setattr(workers, "partfield_status", _ready_hunyuan3d)
    workers.ensure_geometry_worker(tmp_path, tmp_path)
    workers.ensure_partfield_worker(tmp_path)
    assert stopped.count("hunyuan3d") == 2


def test_surface_renders_carry_vertex_colours(tmp_path):
    import numpy as np
    import trimesh

    box = trimesh.creation.box()
    assert workflow_tool._vertex_face_colors(box) is None
    colours = np.tile([255, 0, 0, 255], (len(box.vertices), 1)).astype(np.uint8)
    colours[box.vertices[:, 0] > 0] = (0, 0, 255, 255)
    box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=colours)
    surface = tmp_path / "coloured.glb"
    box.export(surface)
    loaded = trimesh.load(surface, force="mesh", process=False)
    faces = workflow_tool._vertex_face_colors(loaded)
    assert faces.shape == (len(loaded.faces), 4)
    right = loaded.triangles_center[:, 0] > 0.49
    assert np.allclose(faces[right, :3], (0, 0, 1))


def test_an_empty_answer_resumes_the_waiting_step():
    """`{}` means "keep the defaults", not "an empty map of interrupt ids"."""
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    from src.agents_wrapper.fem_workflow.graph import resume_command

    class State(TypedDict, total=False):
        answer: object

    def ask(state):
        return {"answer": interrupt({"kind": "settings"})}

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    graph = builder.compile(checkpointer=MemorySaver())
    for answer in ({}, {"run_mode": "forward"}, "yes"):
        config = {"configurable": {"thread_id": repr(answer)}}
        graph.invoke({}, config=config)
        result = graph.invoke(resume_command(graph, config, answer), config=config)
        assert "__interrupt__" not in result and result["answer"] == answer


def _cad_box_surface(tmp_path):
    """A box whose six sides are six labelled CAD faces, the top a 'cylinder'."""
    import numpy as np
    import trimesh

    from src.cad_wrapper.text2cad_wrapper.availability import face_labels_path

    box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    _, labels = np.unique(box.face_normals.round(6), axis=0, return_inverse=True)
    labels = labels.reshape(-1) * 3  # B-rep face ids need not be contiguous
    surface = tmp_path / "geometry" / "generated.glb"
    surface.parent.mkdir(parents=True, exist_ok=True)
    box.export(surface)
    kinds = ["plane"] * 16
    kinds[int(labels[np.argmax(box.face_normals[:, 2])])] = "cylinder"
    np.savez_compressed(face_labels_path(surface), face_ids=labels.astype(np.int32),
                        surface_types=np.asarray(kinds))
    return surface, labels


def test_cad_faces_become_one_segment_each(tmp_path):
    from argparse import Namespace

    import numpy as np

    surface, labels = _cad_box_surface(tmp_path)
    out = tmp_path / "boundary_conditions"
    args = Namespace(
        method="cad_faces", surface=surface, cad_faces=surface.with_suffix(".glb.faces.npz"),
        output_npz=out / "segments.npz", output_json=out / "segments.json",
        output_glb=out / "segments.glb", clusters=5,
    )
    workflow_tool.segment_surface(args)
    metadata = json.loads(args.output_json.read_text())
    assert metadata["method"] == "cad_faces" and metadata["num_clusters"] == 6
    ids = np.load(args.output_npz)["segment_ids"]
    # Renumbered from zero, but the partition is exactly the CAD one.
    assert len(set(zip(ids.tolist(), labels.tolist()))) == 6
    top = [s for s in metadata["segments"] if s["cad_surface"] == "cylinder"]
    assert len(top) == 1 and top[0]["facing_fraction_45deg"]["+z"] == 1.0

    import trimesh
    trimesh.creation.icosphere().export(surface)  # a different surface
    with pytest.raises(ValueError, match="different surface"):
        workflow_tool.segment_surface(args)


def test_segment_legend_names_the_cad_surface_type():
    legend = SpecialistCalls._segment_legend({
        "bounds": [[0, 0, 0], [1, 1, 1]],
        "segments": [{"id": 0, "color": "#1f77b4", "area_fraction": 0.1,
                      "centroid": [0.5, 0.5, 0.5], "cad_surface": "cylinder"}],
    })
    assert "cylinder face" in legend["0"]


def test_ui_offers_cad_faces_first_only_when_the_surface_has_them():
    assert segmentation_method_choices(True, None, cad_faces=True) == (
        ["cad_faces", "partfield", "shape_diameter"], "cad_faces"
    )
    assert segmentation_method_choices(False, "shape_diameter", cad_faces=True)[1] == "shape_diameter"
    assert segmentation_method_choices(True, "cad_faces") == (
        ["partfield", "shape_diameter"], "partfield"
    )


def test_a_text2cad_part_is_segmented_by_its_own_faces_by_default(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    calls = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, arguments))
            if name == "generate_text2cad_text_surface":
                _cad_box_surface(tmp_path)
            elif name == "segment_surface":
                Path(arguments["output_npz"]).parent.mkdir(parents=True, exist_ok=True)
                for key in ("output_npz", "output_glb"):
                    Path(arguments[key]).write_bytes(b"stub")
                Path(arguments["output_json"]).write_text('{"segments": []}')
            else:
                output = Path(arguments.get("output") or arguments["output_npz"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"stub")
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a block"

        _segment_legend = staticmethod(SpecialistCalls._segment_legend)

    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "text2cad_status", _ready_text2cad)
    # No painter installed, so no optional paint card for the bare CAD part.
    monkeypatch.setattr(graph_module, "hunyuan3d_status", lambda: SimpleNamespace(
        ready=False, model_dump=lambda: {"ready": False}))
    monkeypatch.setattr(graph_module, "partfield_status", lambda: SimpleNamespace(
        ready=True, model_dump=lambda: {"ready": True}))
    workflow = WorkflowState(session_id="f", session_dir=str(tmp_path), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "f"}}
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    physics = graph.invoke(Command(resume={"description": "a block", "generator": "text2cad"}),
                           config=config)["__interrupt__"][0].value
    assert physics["kind"] == "geometry_physics" and physics["cad_faces"] is True
    card = graph.invoke(Command(resume={
        "up_axis": "z", "physical_extent": 0.2, "unit_system": "SI (m-kg-s)",
    }), config=config)["__interrupt__"][0].value
    assert card["kind"] == "bc_input" and card["cad_faces"] is True
    assert card["segmentation"]["method"] == "cad_faces", "ahead of a ready PartField"
    graph.invoke(Command(resume={"action": "preview_segmentation",
                                 "segmentation": card["segmentation"]}), config=config)
    segment = next(arguments for name, arguments in calls if name == "segment_surface")
    assert segment["method"] == "cad_faces"
    assert segment["cad_faces"] == str(tmp_path / "geometry" / "generated.glb.faces.npz")


@pytest.mark.parametrize("generator, expected", [
    ("text2cad", "cad_geometry_refinement"), ("cadcoder", "cadcoder_geometry_refinement"),
    ("triposg", "geometry_refinement"),
])
def test_each_generator_revises_the_prompt_its_own_way(monkeypatch, tmp_path, generator, expected):
    """Text2CAD reads construction, TripoSG a visual phrase: one revision each."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import APIReceipt, WorkflowState

    calls = []

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def _revise(self, purpose, text):
            calls.append(purpose)
            return GeometryRefinement(refined_prompt=text), APIReceipt(
                purpose=purpose, provider="groq", model="fake", cache_key=purpose,
            )

        def refine_geometry(self, description, overview=""):
            return self._revise("geometry_refinement", "a compact hexagonal nut")

        def refine_cad_geometry(self, description):
            assert description == "a hex nut"
            return self._revise("cad_geometry_refinement",
                                "A hexagonal prism with a circular hole through its center.")

        def refine_cadcoder_geometry(self, description):
            assert description == "a hex nut"
            return self._revise("cadcoder_geometry_refinement",
                                "Start by creating a new coordinate system ...")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "text2cad_status", _ready_text2cad)
    monkeypatch.setattr(graph_module, "cadcoder_status", _ready_text2cad)
    workflow = WorkflowState(session_id="r", session_dir=str(tmp_path), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "r"}}
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    revised = graph.invoke(Command(resume={
        "description": "a hex nut", "generator": generator, "revise_prompt": True,
    }), config=config)["__interrupt__"][0].value
    assert calls == [expected]
    assert revised["kind"] == "geometry_input"
    assert revised["generator"] == generator
    # The card says which generator the text in the box was revised for, which
    # is what tells the front ends a CAD prompt is ready to send.
    assert revised["revise_required"] == ["cadcoder", "text2cad"]
    if generator == "triposg":
        assert revised["cad_revision"]["generator"] is None
    else:
        assert revised["cad_revision"] == {"generator": generator,
                                           "prompt": revised["description"]}


def test_a_long_prompt_is_not_taken_for_a_file(tmp_path, monkeypatch):
    """An expert-level description runs to hundreds of characters; checking it
    as a path raised "File name too long" before the tool could run."""
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    monkeypatch.setattr(ToolMonitor, "_shutdown_partfield_worker", lambda self: None)
    monkeypatch.setattr(ToolMonitor, "_shutdown_cad_workers", lambda self, keep=None: None)
    monkeypatch.setattr(ToolMonitor, "_run_geometry_worker",
                        lambda self, name, arguments, timeout: None)

    def fake_run(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_bytes(b"glb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(monitor_module.subprocess, "run", fake_run)
    output = session / "geometry" / "generated.glb"
    output.parent.mkdir()
    receipt = monitor.execute("generate_text_surface",
                              {"output": str(output), "prompt": "a sketch " * 100})
    assert receipt.returncode == 0


def test_an_unrevised_cad_prompt_is_flagged():
    from src.agents_wrapper.fem_workflow.graph import needs_cad_revision

    revision = {"generator": "cadcoder", "prompt": "Start by creating ..."}
    assert needs_cad_revision("cadcoder", "a hex nut", revision)
    assert needs_cad_revision("text2cad", "Start by creating ...", revision), "revised for another"
    assert not needs_cad_revision("cadcoder", "  Start by creating ...\n", revision)
    assert not needs_cad_revision("triposg", "a hex nut", None)

    card = {"kind": "geometry_input", "cad_revision": revision}
    warning = run_workflow.unrevised_cad_warning(
        card, {"description": "a hex nut", "generator": "cadcoder"})
    assert warning and "CAD-Coder" in warning and "revise_prompt" in warning
    assert run_workflow.unrevised_cad_warning(
        card, {"description": "a hex nut", "generator": "cadcoder", "revise_prompt": True}) is None
    assert run_workflow.unrevised_cad_warning(
        card, {"description": "Start by creating ...", "generator": "cadcoder"}) is None
    assert run_workflow.unrevised_cad_warning(
        card, {"description": "a sofa", "generator": "triposg"}) is None


def _forward_case(session: Path) -> Path:
    """A case file whose solve output already exists, as run_forward expects."""
    (session / "case").mkdir(parents=True, exist_ok=True)
    (session / "solve").mkdir(exist_ok=True)
    case = session / "case" / "case.json"
    case.write_text(json.dumps({"output": str(session / "solve")}), encoding="utf-8")
    return case


def test_a_solve_runs_in_its_spare_and_leaves_room_for_the_next(
        tmp_path, monkeypatch, socket_home):
    """The spare serves the solve; the next one is not mistaken for it or held off.

    It exits after this solve on its own, so its pid must not keep the next
    preload from starting another, and neither must the launch backoff meant
    for spares that fail to start.
    """
    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    socket_path, pid_path = monitor._worker_paths("solver")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    attempted = pid_path.parent / "solver.attempted"
    attempted.touch()
    seen, _thread = _fake_worker(socket_path, _greet_and_answer(
        {"ok": True, "stdout": '{"artifacts":[]}\n', "stderr": ""}
    ))
    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run",
                        lambda *a, **k: pytest.fail("the spare was there to run it"))
    receipt = monitor.execute("run_forward", {"case": str(_forward_case(session))})
    assert receipt.returncode == 0
    assert seen[0]["argv"][0] == "run-forward"
    assert not pid_path.exists() and not attempted.exists()


def test_a_spare_stuck_warming_is_stopped_and_the_solve_runs_itself(
        tmp_path, monkeypatch, socket_home):
    from src.agents_wrapper.fem_workflow import monitor as monitor_module
    from src.agents_wrapper.fem_workflow import workers

    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    socket_path, pid_path = monitor._worker_paths("solver")
    socket_home(socket_path.parent)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")  # alive, never binds
    monkeypatch.setattr(monitor_module, "SOLVER_WARM_WAIT", 0.2)
    stopped, ran = [], []
    monkeypatch.setattr(workers, "stop_solver_worker",
                        lambda root, env=None: stopped.append(root))
    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run",
                        lambda command, **k: ran.append(command) or SimpleNamespace(
                            returncode=0, stdout="", stderr=""))
    monitor.execute("run_forward", {"case": str(_forward_case(session))})
    assert stopped == [session.parent], "a stuck spare must not share the GPU with the solve"
    assert len(ran) == 1 and "run-forward" in ran[0]


def test_the_solver_spare_starts_one_shot_in_the_solve_s_environment(
        monkeypatch, tmp_path):
    from src.agents_wrapper.fem_workflow import workers

    launched, stopped = [], []

    class Process:
        pid = os.getpid()

    monkeypatch.setattr(workers.subprocess, "Popen", lambda command, **kwargs: launched.append(
        (command, kwargs["env"])) or Process())
    monkeypatch.setattr(workers, "stop_worker", lambda root, kind, env: stopped.append(kind))
    assert workers.ensure_solver_worker(tmp_path)
    command, environment = launched[0]
    assert "--one-shot" in command and command[command.index("--warm") + 1] == "solver"
    assert command[2].endswith("tool_server.py")
    # What ToolMonitor gives a solve it launches itself.
    assert environment["XDG_CACHE_HOME"] == str(tmp_path / ".runtime_cache")
    assert environment["MPLCONFIGDIR"] == str(tmp_path / ".runtime_cache" / "matplotlib")
    assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

    # Warming: trusted, not relaunched.
    assert workers.ensure_solver_worker(tmp_path) and len(launched) == 1
    # Alive but long past a warm-up without serving: replaced.
    _, pid_path = workers.worker_paths(tmp_path, "solver", workers.DEFAULT_COMPUTE_ENV)
    old = time.time() - 2 * workers.SOLVER_WARM_WAIT
    os.utime(pid_path, (old, old))
    (pid_path.parent / "solver.attempted").unlink()
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    os.utime(pid_path, (old, old))
    assert workers.ensure_solver_worker(tmp_path)
    assert stopped == ["solver"] and len(launched) == 2


def test_the_one_shot_server_exits_after_its_one_tool(tmp_path, monkeypatch, socket_home):
    """A warm request keeps it; the tool it was started for is its last."""
    import socket as socket_module
    import sys
    import threading

    sys.path.insert(0, str(Path(workflow_tool.__file__).parent))
    import tool_server

    monkeypatch.setattr(tool_server, "warm", lambda tier: None)
    directory = Path("/tmp") / f"oneshot-{os.getpid()}"
    directory.mkdir(exist_ok=True)
    socket_home(directory)
    socket_path = directory / "s.sock"
    thread = threading.Thread(target=tool_server.serve, args=(socket_path, "solver", 30.0, True),
                              daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not socket_path.exists() and time.time() < deadline:
        time.sleep(0.01)

    def ask(request):
        with socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM) as client:
            client.connect(os.fspath(socket_path))
            stream = client.makefile("r", encoding="utf-8")
            assert json.loads(stream.readline())["ready"]
            client.sendall((json.dumps(request) + "\n").encode())
            return json.loads(stream.readline())

    assert ask({"action": "warm", "tier": "light"})["ok"]
    assert thread.is_alive(), "warming is not the tool it serves"
    assert not ask({"argv": ["no-such-subcommand"], "cwd": str(tmp_path)})["ok"]
    thread.join(timeout=5)
    assert not thread.is_alive() and not socket_path.exists()


def test_the_settings_card_starts_the_solve_s_process(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(run_workflow, "ensure_tool_worker", lambda *a, **k: True)
    monkeypatch.setattr(run_workflow, "ensure_solver_worker",
                        lambda root, env: started.append((root, env)))
    run_workflow.preload_for(tmp_path, tmp_path / "s", "env", {"kind": "bc_input"})
    assert started == []
    run_workflow.preload_for(tmp_path, tmp_path / "s", "env", {"kind": "shape_opt_settings"})
    assert started == [(tmp_path, "env")]


def test_a_build_lock_left_by_a_killed_worker_does_not_wedge_the_next(monkeypatch, tmp_path):
    """A VoMP worker killed while compiling nvdiffrast left torch's build lock
    behind, and every later worker in that output root waited on it forever."""
    from src.agents_wrapper.fem_workflow import workers

    class Process:
        pid = os.getpid()

    monkeypatch.setattr(workers.subprocess, "Popen", lambda command, **kwargs: Process())

    def lock(cache, age=0.0):
        path = cache / "torch_extensions" / "py310_cu118" / "nvdiffrast_plugin" / "lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.utime(path, (time.time() - age, time.time() - age))
        return path

    # A worker's own cache: no worker of its kind is alive, so the lock is stale.
    owned = lock(tmp_path / ".runtime_cache" / "vomp")
    assert workers._launch(tmp_path, "vomp", "env", ["x"], cwd=tmp_path,
                           cache=owned.parents[3], owns_cache=True)
    assert not owned.exists()
    assert "stale build lock" in (tmp_path / ".workers" / "vomp.stderr.log").read_text()

    # The shared cache: a one-shot tool may be compiling right now, so only a
    # lock older than any build goes.
    shared = tmp_path / ".runtime_cache"
    building = lock(shared)
    assert workers._launch(tmp_path, "tools", "env", ["x"], cwd=tmp_path, cache=shared)
    assert building.exists()
    left = lock(shared, age=workers.STALE_BUILD_LOCK_AGE + 60)
    (tmp_path / ".workers" / "tools.attempted").unlink()
    assert workers._launch(tmp_path, "tools", "env", ["x"], cwd=tmp_path, cache=shared)
    assert not left.exists()


def test_a_bare_surface_is_painted_before_vomp_reads_it(monkeypatch, tmp_path):
    """VoMP refuses a grey surface; painting it unlocks VoMP, which then reads the painted copy."""
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCProposal, DirichletBC, MaterialInference, NeumannBC,
        SelectorSpec, ToolReceipt, WorkflowState,
    )

    calls = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, dict(arguments)))
            for key, value in arguments.items():
                if key.startswith("output") or (name == "infer_material_field" and key == "artifact"):
                    path = Path(value)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}") if path.suffix == ".json" else path.write_bytes(b"stub")
                    if name == "stage_asset":
                        path.with_suffix(path.suffix + ".json").write_text(json_module.dumps(
                            {"source_name": path.name, "path": str(path), "sha256": "0" * 64}))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    proposal = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, selector=SelectorSpec(
            type="lowest", params={"axis": "y", "fraction": 0.1}))],
        neumann=[NeumannBC(name="top", traction=(0.0, -1.0, 0.0), selector=SelectorSpec(
            type="highest", params={"axis": "y", "fraction": 0.1}))],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a stub reading of the render"

        def synthesize_bcs(self, description, metadata, montage):
            return proposal, APIReceipt(purpose="bc_synthesis", provider="groq",
                                        model="fake", cache_key="k1")

        def infer_material(self, description, request, modulus_unit, image=None):
            return MaterialInference(mode="vomp"), APIReceipt(
                purpose="material_interpretation", provider="groq", model="fake", cache_key="k2")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    # Only the painted copy has colour.
    monkeypatch.setattr(graph_module, "_surface_has_colour",
                        lambda path: False if not path else path.endswith("textured.glb"))
    monkeypatch.setattr(graph_module, "texture_status", lambda: SimpleNamespace(
        model_dump=lambda: {"ready": True, "summary": "ready"}))
    monkeypatch.setattr(graph_module, "hunyuan3d_status",
                        lambda: SimpleNamespace(ready=True, model_dump=lambda: {"ready": True}))
    source = tmp_path / "chair.glb"
    source.write_bytes(b"glb")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "paint"}}
    workflow = WorkflowState(session_id="paint", session_dir=str(tmp_path), model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "an armchair", "asset_path": str(source),
                                 "asset_kind": "surface"}), config=config)
    # The optional paint card before scaling is skipped: VoMP's own step is under test.
    graph.invoke(Command(resume={"action": "skip_texture"}), config=config)
    graph.invoke(Command(resume={"action": "apply_scaling", "up_axis": "y",
                                 "physical_extent": 1.0,
                                 "unit_system": "SI (m-kg-s)"}), config=config)
    settings = {"num_clusters": 5, "num_rays": 25, "smoothing": 0.25, "cone_angle_deg": 120.0}
    graph.invoke(Command(resume={"action": "preview_regions", "description": "fix feet, load top",
                                 "segmentation": settings}), config=config)
    graph.invoke(Command(resume={"action": "approve_regions"}), config=config)
    card = graph.invoke(Command(resume={
        "action": "apply_values",
        "dirichlet_values": {"feet": {"components": [0, 1, 2], "value": 0.0}},
        "neumann_values": {"top": [0.0, -1.0, 0.0]}, "body_force": [0.0, 0.0, 0.0],
    }), config=config)["__interrupt__"][0].value
    assert card["kind"] == "material_input"
    assert card["surface_has_colour"] is False and card["textured_surface"] is None
    assert card["texture_prompt"] == "an armchair"          # prefilled from the geometry

    # Inference that picks VoMP opens the VoMP card at its paint step instead of running it.
    card = graph.invoke(Command(resume={"action": "infer_material"}), config=config)["__interrupt__"][0].value
    assert card["material_mode"] == "vomp"
    assert not any(name == "infer_material_field" for name, _ in calls)

    # Running VoMP directly on the grey surface is refused; the card is then
    # answered again, as the app does.
    before = graph.get_state(config).config
    with pytest.raises(Exception, match="no colour"):
        graph.invoke(Command(resume={"action": "infer_vomp"}), config=config)
    # Fork the card's checkpoint, dropping the refused answer, and replay to its interrupt.
    graph.invoke(None, config=graph.update_state({"configurable": {
        **before["configurable"], "checkpoint_ns": ""}}, None))
    card = graph.invoke(graph_module.resume_command(graph, config, {
        "action": "texture_surface", "texture_prompt": "black leather, chrome frame", "texture_seed": 2,
    }), config=config)["__interrupt__"][0].value
    paint = next(arguments for name, arguments in calls if name == "paint_hunyuan3d_surface")
    assert paint["prompt"] == "black leather, chrome frame" and paint["seed"] == 2
    assert paint["surface"].endswith("physical.glb") and "input" not in paint
    assert card["textured_surface"] == paint["output"]
    assert card["texture_reference"] == paint["output_reference"]
    assert card["texture_views"].endswith("views.png")

    graph.invoke(Command(resume={"action": "infer_vomp"}), config=config)
    vomp = next(arguments for name, arguments in calls if name == "infer_material_field")
    assert vomp["surface"] == paint["output"]


def test_a_painting_without_colour_is_not_recorded(monkeypatch, tmp_path):
    """A paint whose surface came back all one plain value (an acceptance run, 2026-10-10: every
    vertex (0, 0, 0), from a receipt that said it painted) fails, and is never the textured surface."""
    np = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    surface = tmp_path / "geometry" / "physical.glb"
    surface.parent.mkdir(parents=True)
    trimesh.creation.box().export(surface)
    painted_colour = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            if name == "paint_hunyuan3d_surface":
                box = trimesh.creation.box()
                colours = np.tile(np.array(painted_colour[-1] + (255,), np.uint8), (len(box.vertices), 1))
                colours[0, :3] //= 2  # a painting varies; the failed one is (0, 0, 0) throughout
                box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=colours)
                Path(arguments["output"]).parent.mkdir(parents=True, exist_ok=True)
                box.export(arguments["output"])
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="", stderr_path="", artifacts=[])

    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    state = WorkflowState(session_id="paint", session_dir=str(tmp_path), model="fake",
                          surface_path=str(surface))
    answer = {"texture_prompt": "light wood", "texture_seed": 0}
    painted_colour.append((0, 0, 0))
    with pytest.raises(RuntimeError, match="the painting failed: the painted surface carries no colour"):
        graph_module._paint_surface(state, answer, tmp_path / "materials" / "texture")
    assert state.textured_surface is None
    painted_colour.append((196, 150, 100))
    graph_module._paint_surface(state, answer, tmp_path / "materials" / "texture")
    assert state.textured_surface == str(tmp_path / "materials" / "texture" / "textured.glb")


def _crashing_paint_monitor(np, trimesh, calls, crash_seeds, fail_seeds=(), kill_seeds=()):
    """A tool monitor whose painter crashes (SIGSEGV) at ``crash_seeds``, fails with a Python
    error at ``fail_seeds`` and paints otherwise; the plain colour is the real workflow tool."""
    import argparse

    from src.agents_wrapper.fem_workflow.models import ToolReceipt
    from src.agents_wrapper.fem_workflow.monitor import ToolFailed
    from src.agents_wrapper.fem_workflow.scripts import workflow_tool

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, dict(arguments)))
            code = 0
            if name == "paint_hunyuan3d_surface":
                if arguments["seed"] in crash_seeds:
                    code = -11
                elif arguments["seed"] in fail_seeds:
                    code = 1
                elif arguments["seed"] in kill_seeds:
                    code = -9
                else:
                    box = trimesh.creation.box()
                    colours = np.tile(np.array((196, 150, 100, 255), np.uint8), (len(box.vertices), 1))
                    colours[0, :3] //= 2
                    box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=colours)
                    Path(arguments["output"]).parent.mkdir(parents=True, exist_ok=True)
                    box.export(arguments["output"])
            elif name == "plain_colour_surface":
                workflow_tool.plain_colour_surface(argparse.Namespace(
                    surface=Path(arguments["surface"]), rgb=arguments["rgb"], output=Path(arguments["output"]),
                    **{key: (Path(arguments[key]) if key != "part_rgb" else arguments[key])
                       for key in ("segments_npz", "segments_json", "part_rgb") if key in arguments}))
            receipt = ToolReceipt(tool=name, arguments=arguments, returncode=code,
                                  stdout_path="", stderr_path="", artifacts=[])
            if code == -11:
                raise ToolFailed(f"{name} failed: crashed (SIGSEGV, return code -11) without a Python "
                                 "error; the same inputs crash again; see x.log", receipt)
            if code == -9:
                raise ToolFailed(f"{name} failed: killed by SIGKILL (return code -9): out of memory, "
                                 "or stopped by a user; see x.log", receipt)
            if code:
                raise ToolFailed(f"{name} failed: RuntimeError: the painting failed: no colour; see x.log",
                                 receipt)
            return receipt

    return FakeMonitor


def test_a_crashing_painter_falls_back_to_a_marked_plain_colour(monkeypatch, tmp_path):
    """A painter that crashes on a surface (a held-out run, 2026-10-10: SIGSEGV four times, at
    two seeds) is run at the asked seed and one more, never twice on the same inputs; then the
    surface takes one flat colour read from the words, marked, and the run goes on."""
    np = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    surface = tmp_path / "geometry" / "physical.glb"
    surface.parent.mkdir(parents=True)
    trimesh.creation.box().export(surface)
    calls = []
    monkeypatch.setattr(graph_module, "ToolMonitor", _crashing_paint_monitor(np, trimesh, calls, {0, 1}))
    answer = {"texture_prompt": "light yellow metal", "texture_seed": 0}
    directory = tmp_path / "materials" / "texture"

    def fresh():
        return WorkflowState(session_id="paint", session_dir=str(tmp_path), model="fake",
                             surface_path=str(surface))

    state = fresh()
    graph_module._paint_surface(state, answer, directory)
    assert [(n, a.get("seed")) for n, a in calls] == [
        ("paint_hunyuan3d_surface", 0), ("paint_hunyuan3d_surface", 1),
        ("plain_colour_surface", None), ("render_surface", None)]
    fallback = state.texture_fallback
    assert fallback["kind"] == "plain_colour" and fallback["crashed_seeds"] == [0, 1]
    assert fallback["colour_word"] == "light yellow" and "NOT PAINTED" in fallback["note"]
    assert state.texture_seed == 0 and state.texture_reference is None
    assert state.textured_surface == str(directory / "textured.glb")
    colours = np.asarray(trimesh.load(state.textured_surface, force="mesh").visual.vertex_colors)
    assert (colours[:, :3] == fallback["colour"]).all()
    assert graph_module._texture_fields(state)["texture_fallback"] == fallback

    # The card's own paint after the background one (a throwaway state), and a repaint with other
    # words: neither runs the painter again on this surface.
    for words in ("light yellow metal", "grey steel"):
        calls.clear()
        state = fresh()
        graph_module._paint_surface(state, {**answer, "texture_prompt": words}, directory)
        assert [n for n, _ in calls] == ["plain_colour_surface", "render_surface"]
    assert state.texture_fallback["colour_word"] == "grey"


def test_per_part_colours_give_each_part_one_flat_colour_without_the_painter(monkeypatch, tmp_path):
    """The painter places no colour by part (a held-out run, 2026-10-10, painted seven times without
    the asked colours on the asked parts): per_part_colours colours each part of the per-part
    segmentation flat, through the plain-colour tool extended to colours by part."""
    np = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState
    from src.hunyuan3d_wrapper.surface import has_colour, plain_colour

    surface = tmp_path / "geometry" / "physical.glb"
    surface.parent.mkdir(parents=True)
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0]).subdivide().subdivide().subdivide()
    box.export(surface)
    loaded = trimesh.load(surface, force="mesh", process=False)
    labels = (loaded.triangles_center[:, 1] > 0.49).astype(np.int64)      # the top face: part 1
    parts = tmp_path / "materials" / "parts"
    parts.mkdir(parents=True)
    np.savez(parts / "segments.npz", segment_ids=labels)
    (parts / "segments.json").write_text(json.dumps({"segments": [
        {"id": 0, "color": "#1f77b4"}, {"id": 1, "color": "#ff7f0e"}]}))
    calls = []
    monkeypatch.setattr(graph_module, "ToolMonitor", _crashing_paint_monitor(np, trimesh, calls, set()))
    state = WorkflowState(session_id="parts", session_dir=str(tmp_path), model="fake", surface_path=str(surface))
    directory = tmp_path / "materials" / "texture"
    graph_module._colour_parts(state, {"per_part_colours": {"1": "red"}}, directory,
                               parts / "segments.npz", parts / "segments.json")
    assert [n for n, _ in calls] == ["plain_colour_surface", "render_surface"]
    painted = trimesh.load(state.textured_surface, force="mesh", process=False)
    assert len(painted.vertices) == len(loaded.vertices) and len(painted.faces) == len(loaded.faces)
    colours = np.asarray(painted.visual.vertex_colors)[:, :3]
    # a vertex takes the part with the most face area round it: inside each part, its colour
    top = (painted.vertices[:, 1] > 0.25) & (np.abs(painted.vertices[:, [0, 2]]).max(axis=1) < 0.45)
    rest = painted.vertices[:, 1] < 0.45
    assert (colours[top] == plain_colour("red")[0]).all() and (colours[rest] == (0x1F, 0x77, 0xB4)).all()
    assert has_colour(state.textured_surface)
    fallback = state.texture_fallback
    assert fallback["kind"] == "part_colours" and fallback["asked"] == {"1": "red"} and fallback["parts"] == [0, 1]
    with pytest.raises(ValueError, match="names parts \\[7\\]"):
        graph_module._colour_parts(state, {"per_part_colours": {"7": "red"}}, directory,
                                   parts / "segments.npz", parts / "segments.json")


def test_a_killed_painter_is_reported_and_never_counted_as_a_crash(monkeypatch, tmp_path):
    """SIGKILL (the out-of-memory killer, or a user) says nothing about the surface: the error
    is raised as it is, with no crash record, so the same paint runs again once there is room."""
    np = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    surface = tmp_path / "geometry" / "physical.glb"
    surface.parent.mkdir(parents=True)
    trimesh.creation.box().export(surface)
    calls, killed = [], {0}
    monkeypatch.setattr(graph_module, "ToolMonitor",
                        _crashing_paint_monitor(np, trimesh, calls, set(), kill_seeds=killed))
    state = WorkflowState(session_id="paint", session_dir=str(tmp_path), model="fake",
                          surface_path=str(surface))
    answer = {"texture_prompt": "red", "texture_seed": 0}
    with pytest.raises(RuntimeError, match="killed by SIGKILL"):
        graph_module._paint_surface(state, answer, tmp_path / "materials" / "texture")
    assert [n for n, _ in calls] == ["paint_hunyuan3d_surface"]
    assert not (tmp_path / "tool_receipts" / graph_module.PAINT_CRASHES).exists()
    killed.clear()
    graph_module._paint_surface(state, answer, tmp_path / "materials" / "texture")
    assert state.texture_fallback is None and calls[1][1]["seed"] == 0


def test_a_python_error_from_the_painter_falls_back_to_the_plain_colour(monkeypatch, tmp_path):
    """A Python exception from the painter (a batch of 2026-10-12: ``RuntimeError: max() ...
    numel() == 0`` in MeshRender.back_project on a long thin body, at every seed) is a paint
    failure: the asked seed and one more, then the marked plain colour, never re-raised (the run
    looped on it three times); the next paint of the same surface runs no painter."""
    np = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    surface = tmp_path / "geometry" / "physical.glb"
    surface.parent.mkdir(parents=True)
    trimesh.creation.box().export(surface)
    calls = []
    monkeypatch.setattr(graph_module, "ToolMonitor",
                        _crashing_paint_monitor(np, trimesh, calls, set(), fail_seeds=set(range(20))))
    state = WorkflowState(session_id="paint", session_dir=str(tmp_path), model="fake",
                          surface_path=str(surface))
    graph_module._paint_surface(state, {"texture_prompt": "light grey steel", "texture_seed": 7},
                                tmp_path / "materials" / "texture")
    assert [(n, a.get("seed")) for n, a in calls] == [
        ("paint_hunyuan3d_surface", 7), ("paint_hunyuan3d_surface", 8),
        ("plain_colour_surface", None), ("render_surface", None)]
    assert state.texture_fallback["kind"] == "plain_colour" and state.texture_fallback["crashed_seeds"] == [7, 8]
    assert "NOT PAINTED" in state.texture_fallback["note"]
    calls.clear()
    graph_module._paint_surface(state, {"texture_prompt": "light grey steel", "texture_seed": 7},
                                tmp_path / "materials" / "texture")
    assert [n for n, _ in calls] == ["plain_colour_surface", "render_surface"]


def test_a_painter_crash_retries_one_other_seed(monkeypatch, tmp_path):
    """The asked seed crashes and the next one paints: kept, with the asked seed still the
    card's (the runner compares it with the plan's), and marked reseeded."""
    np = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    surface = tmp_path / "geometry" / "physical.glb"
    surface.parent.mkdir(parents=True)
    trimesh.creation.box().export(surface)
    calls = []
    monkeypatch.setattr(graph_module, "ToolMonitor", _crashing_paint_monitor(np, trimesh, calls, {3}))
    state = WorkflowState(session_id="paint", session_dir=str(tmp_path), model="fake",
                          surface_path=str(surface))
    graph_module._paint_surface(state, {"texture_prompt": "red", "texture_seed": 3},
                                tmp_path / "materials" / "texture")
    assert [(n, a.get("seed")) for n, a in calls][:2] == [
        ("paint_hunyuan3d_surface", 3), ("paint_hunyuan3d_surface", 4)]
    assert state.texture_seed == 3
    assert state.texture_fallback["kind"] == "reseeded" and state.texture_fallback["seed_used"] == 4


def test_materials_are_assigned_per_part_edited_and_approved(monkeypatch, tmp_path):
    """Per part: PartField parts, the colour montage, one value per part, a field for the case."""
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCProposal, DirichletBC, NeumannBC, PartMaterials,
        SelectorSpec, ToolReceipt, WorkflowState,
    )

    calls, asked = [], []
    segments = {"segments": [
        {"id": 0, "color": "#1f77b4", "area_fraction": 0.8, "centroid": [0, 0.6, 0]},
        {"id": 1, "color": "#aec7e8", "area_fraction": 0.2, "centroid": [0, 0.1, 0]},
    ], "bounds": [[-1, 0, -1], [1, 1, 1]]}

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, dict(arguments)))
            for key, value in arguments.items():
                if key.startswith("output"):
                    path = Path(value)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if name == "segment_surface" and path.suffix == ".json":
                        path.write_text(json_module.dumps(segments))
                    elif name == "build_part_field":
                        path.mkdir(exist_ok=True)
                        (path / "manifest.json").write_text("{}")
                    else:
                        path.write_text("{}") if path.suffix == ".json" else path.write_bytes(b"stub")
                    if name == "stage_asset":
                        path.with_suffix(path.suffix + ".json").write_text(json_module.dumps(
                            {"source_name": path.name, "path": str(path), "sha256": "0" * 64}))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    proposal = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, selector=SelectorSpec(
            type="lowest", params={"axis": "y", "fraction": 0.1}))],
        neumann=[NeumannBC(name="top", traction=(0.0, -1.0, 0.0), selector=SelectorSpec(
            type="highest", params={"axis": "y", "fraction": 0.1}))],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        _segment_legend = staticmethod(SpecialistCalls._segment_legend)

        def describe_image(self, purpose, image, legend=None):
            return "part 0 is the seat, part 1 the legs" if legend else "a chair"

        def synthesize_bcs(self, description, metadata, montage):
            return proposal, APIReceipt(purpose="bc_synthesis", provider="groq",
                                        model="fake", cache_key="k1")

        def assign_part_materials(self, montage, metadata, modulus_unit, request="", segments=""):
            asked.append({"montage": str(montage), "unit": modulus_unit,
                          "request": request, "segments": segments})
            # Asked about "gigapascals", the stand-in model answers in them.
            steel = 200.0 if "gigapascals" in request else 2.0e11
            return PartMaterials.model_validate({"parts": [
                {"id": 0, "youngs_modulus": 1.5e8, "poisson_ratio": 0.4, "density": 900.0},
                {"id": 1, "youngs_modulus": steel, "poisson_ratio": 0.29, "density": 7850.0},
            ]}), APIReceipt(purpose="part_materials", provider="groq", model="fake", cache_key="k3")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "_surface_has_colour", lambda path: True)
    monkeypatch.setattr(graph_module, "partfield_status", lambda: SimpleNamespace(
        ready=True, model_dump=lambda: {"ready": True, "summary": "ready"}))
    source = tmp_path / "chair.glb"
    source.write_bytes(b"glb")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "parts"}}
    workflow = WorkflowState(session_id="parts", session_dir=str(tmp_path), model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "an armchair", "asset_path": str(source),
                                 "asset_kind": "surface"}), config=config)
    graph.invoke(Command(resume={"action": "apply_scaling", "up_axis": "y",
                                 "physical_extent": 1.0,
                                 "unit_system": "SI (m-kg-s)"}), config=config)
    # Named outright: a partial segmentation keeps the card's method, PartField here.
    settings = {"method": "shape_diameter", "num_clusters": 5, "num_rays": 25,
                "smoothing": 0.25, "cone_angle_deg": 120.0}
    graph.invoke(Command(resume={"action": "preview_regions", "description": "fix feet, load top",
                                 "segmentation": settings}), config=config)
    graph.invoke(Command(resume={"action": "approve_regions"}), config=config)
    graph.invoke(Command(resume={
        "action": "apply_values",
        "dirichlet_values": {"feet": {"components": [0, 1, 2], "value": 0.0}},
        "neumann_values": {"top": [0.0, -1.0, 0.0]}, "body_force": [0.0, 0.0, 0.0],
    }), config=config)

    card = graph.invoke(Command(resume={
        "action": "infer_parts", "material_description": "a steel base",
    }), config=config)["__interrupt__"][0].value
    # The boundary conditions used shape-diameter segments, so the parts are PartField's own.
    segment = next(arguments for name, arguments in calls
                   if name == "segment_surface" and arguments["method"] == "partfield")
    assert segment["clusters"] == graph_module.MATERIAL_PART_CLUSTERS
    assert "materials" in segment["output_npz"]
    montage = next(arguments for name, arguments in calls if name == "render_part_montage")
    assert montage["surface"].endswith("physical.glb")
    assert asked == [{"montage": montage["output"], "unit": "Pa (N/m²)",
                      "request": "a steel base", "segments": "part 0 is the seat, part 1 the legs"}]
    field = [arguments for name, arguments in calls if name == "build_part_field"]
    assert len(field) == 1 and field[0]["unit_system"] == "SI (m-kg-s)"
    assert card["material_mode"] == "parts" and card["material_inferred"] is True
    assert card["parts_artifact"] == field[0]["output"]
    assert [part["youngs_modulus"] for part in card["material_parts"]] == [1.5e8, 2.0e11]

    # An edit rewrites the values and the field, without another call.
    edited = [dict(card["material_parts"][0], youngs_modulus=3.0e8), card["material_parts"][1]]
    card = graph.invoke(Command(resume={"action": "apply_parts", "parts": edited}),
                        config=config)["__interrupt__"][0].value
    assert len(asked) == 1 and card["material_inferred"] is False
    written = json_module.loads(Path(field[0]["materials_json"]).read_text())
    assert written["parts"][0]["youngs_modulus"] == 3.0e8
    def fork():
        """Drop a refused answer and return to the card, as the app does."""
        graph.invoke(None, config=graph.update_state({"configurable": {
            **graph.get_state(config).config["configurable"], "checkpoint_ns": ""}}, None))

    with pytest.raises(Exception, match="must match the assigned parts"):
        graph.invoke(Command(resume={"action": "apply_parts", "parts": edited[:1]}), config=config)
    fork()
    # Steel's modulus in MPa, typed into an SI case.
    slipped = [edited[0], dict(edited[1], youngs_modulus=2.0e5)]
    with pytest.raises(Exception, match="check the units"):
        graph.invoke(Command(resume={"action": "apply_parts", "parts": slipped}), config=config)

    # The surface is replaced at the same path: its old parts are neither shown,
    # edited nor approved, and assigning again segments it afresh.  In a session
    # recorded before its files were tracked, that is: a tracked one puts the
    # surface this checkpoint had back, parts and all.
    monkeypatch.setattr(graph_module, "_restore_branch_files", lambda state: None)
    surface = Path(montage["surface"])
    original = surface.read_bytes()
    surface.write_bytes(b"another surface")
    fork()
    card = graph.get_state(config).tasks[0].interrupts[0].value
    assert card["material_parts"] is None and card["parts_artifact"] is None
    assert card["material_mode"] == "uniform"
    with pytest.raises(Exception, match="on this surface"):
        graph.invoke(Command(resume={"action": "apply_parts", "parts": edited}), config=config)
    fork()
    with pytest.raises(Exception, match="on this surface"):
        graph.invoke(Command(resume={"material_mode": "parts"}), config=config)
    fork()
    graph.invoke(Command(resume={"action": "infer_parts"}), config=config)
    assert sum(name == "segment_surface" and arguments["method"] == "partfield"
               for name, arguments in calls) == 2
    surface.write_bytes(original)
    graph.invoke(Command(resume={"action": "infer_parts"}), config=config)
    assert sum(name == "segment_surface" and arguments["method"] == "partfield"
               for name, arguments in calls) == 3
    field = [arguments for name, arguments in calls if name == "build_part_field"][-1:]

    # A model answer off by a unit factor is reported under the button, and builds nothing.
    built = sum(name == "build_part_field" for name, _ in calls)
    card = graph.invoke(Command(resume={
        "action": "infer_parts", "material_description": "a steel base, in gigapascals",
    }), config=config)["__interrupt__"][0].value
    assert "unit slip" in card["api_error"] and card["api_error_source"] == "infer_parts"
    assert sum(name == "build_part_field" for name, _ in calls) == built
    assert card["parts_artifact"] == field[0]["output"]

    card = graph.invoke(Command(resume={"material_mode": "parts"}),
                        config=config)["__interrupt__"][0].value
    assert card["kind"] == "shape_opt_settings"
    material = json_module.loads((tmp_path / "case" / "material.json").read_text())
    assert material["mode"] == "parts" and material["artifact"] == field[0]["output"]


def test_part_values_off_by_a_unit_factor_are_refused():
    """A part's values must be some solid's; a 10⁶ slip is not, in any unit system."""
    from src.agents_wrapper.fem_workflow.graph import _implausible_part

    def part(modulus, density):
        return [{"id": 3, "youngs_modulus": modulus, "poisson_ratio": 0.3, "density": density}]

    for modulus, density in ((2e11, 7850), (1.5e8, 900), (2e4, 40), (1e5, 1100), (7e10, 2500)):
        assert _implausible_part(part(modulus, density), "SI (m-kg-s)") is None
    assert _implausible_part(part(2e5, 7.85e-9), "SI-mm (mm-tonne-s)") is None
    # Steel's modulus given in MPa to an SI case: a plausible foam's E, but not at steel's density.
    assert "part 3" in _implausible_part(part(2e5, 7850), "SI (m-kg-s)")
    # Pascals and kg/m³ given to a millimetre case.
    assert "10⁶" in _implausible_part(part(2e11, 7850), "SI-mm (mm-tonne-s)")
    assert _implausible_part(part(2e5, 7850e-12 * 1e6), "SI-mm (mm-tonne-s)")


def test_a_painting_belongs_to_the_surface_contents_not_its_path(tmp_path):
    """A surface regenerated at the same path does not keep the old surface's painting."""
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    surface, painted = tmp_path / "physical.glb", tmp_path / "textured.glb"
    surface.write_bytes(b"first surface")
    painted.write_bytes(b"painted")
    state = WorkflowState(session_id="s", session_dir=str(tmp_path), model="fake",
                          surface_path=str(surface), textured_surface=str(painted),
                          textured_from=str(surface),
                          textured_from_digest=graph_module._surface_digest(str(surface)))
    assert graph_module._current_texture(state) == str(painted)
    surface.write_bytes(b"second surface")
    assert graph_module._current_texture(state) is None


def test_painting_a_surface_runs_on_the_hunyuan3d_worker(tmp_path, monkeypatch):
    """The paint tool goes to the resident Hunyuan3D worker; words need the reference models, an image does not."""
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    session = tmp_path / "session"
    (session / "geometry").mkdir(parents=True)
    surface = session / "geometry" / "physical.glb"
    surface.write_bytes(b"glb")
    for name in ("geometry", "partfield", "hunyuan3d", "vomp"):
        monkeypatch.setattr(ToolMonitor, f"_shutdown_{name}_worker", lambda self: None)
    monkeypatch.setattr(ToolMonitor, "_shutdown_cad_workers", lambda self, keep=None: None)
    monkeypatch.setattr(ToolMonitor, "_run_worker", lambda self, *args, **kw: None)
    painting = []
    monkeypatch.setattr("src.agents_wrapper.fem_workflow.workers.ensure_hunyuan3d_worker",
                        lambda root, paint=False: painting.append(paint))
    monkeypatch.setattr(monitor_module, "hunyuan3d_status", _ready_hunyuan3d)
    not_ready = SimpleNamespace(ready=False, interpreter=None, summary="SDXL missing")
    monkeypatch.setattr(monitor_module, "texture_status", lambda: not_ready)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        for flag in ("--output", "--reference"):
            Path(command[command.index(flag) + 1]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[command.index(flag) + 1]).write_bytes(b"x")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(monitor_module.subprocess, "run", fake_run)
    texture = session / "materials" / "texture"
    args = {"surface": str(surface), "prompt": "oak", "seed": 4,
            "output": str(texture / "textured.glb"), "output_reference": str(texture / "reference.png")}
    with pytest.raises(ToolRejected, match="SDXL missing"):
        ToolMonitor(session).execute("paint_hunyuan3d_surface", args)
    photo = session / "inputs" / "look.png"
    photo.parent.mkdir()
    photo.write_bytes(b"png")
    ToolMonitor(session).execute("paint_hunyuan3d_surface", {**args, "input": str(photo)})
    command = commands[-1]
    assert command[0] == "/envs/mechfoundry-hunyuan3d/bin/python"
    assert command[1].endswith("hunyuan3d_wrapper/generate.py")
    flag = lambda name: command[command.index(name) + 1]
    assert flag("--surface") == str(surface) and flag("--input") == str(photo)
    assert flag("--seed") == "4" and flag("--reference") == args["output_reference"]
    # A cold start for a paint loads the paint stage, not the shape stage.
    assert painting == [True]


_DYING_PAINT_WORKER = """
import os, signal, sys, faulthandler
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from src.hunyuan3d_wrapper import server
def die(args, runtime):
    if sys.argv[2] == "segv":
        faulthandler._sigsegv()
    os.kill(os.getpid(), signal.SIGKILL)
server.run = die
Path(sys.argv[4]).write_text(str(os.getpid()))
server.serve(Path(sys.argv[1]), None, 60)
"""


@pytest.mark.parametrize("death", ["segv", "kill"])
def test_a_paint_worker_that_dies_mid_request_is_not_run_again_cold(tmp_path, monkeypatch, socket_home, death):
    """The real Hunyuan3D worker loop (server.serve), its paint dying in the worker process.  A
    held-out run (2026-10-10) ran each crashing seed twice: the worker died, the monitor read
    EOF and ran the same paint again from a cold load, which crashed the same way.  Now the
    worker's fault report is the result: a SIGSEGV is a crash, recorded against the inputs; a
    death without a report is a kill (out of memory, a user), said as such and not counted."""
    import subprocess as subprocess_module
    import sys as sys_module

    from src.agents_wrapper.fem_workflow import monitor as monitor_module
    from src.agents_wrapper.fem_workflow.monitor import ToolFailed, crashed, fault_report
    from src.hunyuan3d_wrapper import server as server_module

    session = tmp_path / "session"
    (session / "geometry").mkdir(parents=True)
    surface = session / "geometry" / "physical.glb"
    surface.write_bytes(b"glb")
    for name in ("geometry", "partfield", "hunyuan3d", "vomp"):
        monkeypatch.setattr(ToolMonitor, f"_shutdown_{name}_worker", lambda self: None)
    monkeypatch.setattr(ToolMonitor, "_shutdown_cad_workers", lambda self, keep=None: None)
    monkeypatch.setattr("src.agents_wrapper.fem_workflow.workers.ensure_hunyuan3d_worker",
                        lambda root, paint=False: None)
    monkeypatch.setattr(monitor_module, "texture_status", _ready_hunyuan3d)
    one_shot = []
    monkeypatch.setattr(monitor_module.subprocess, "run", lambda *a, **k: one_shot.append(a) or (_ for _ in ()).throw(
        AssertionError("the paint was run again cold")))
    monitor = ToolMonitor(session)
    socket_path, pid_path = monitor._worker_paths("hunyuan3d", "mechfoundry-hunyuan3d")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    assert server_module.fault_report(socket_path) == fault_report(socket_path)
    worker = subprocess_module.Popen(
        [sys_module.executable, "-c", _DYING_PAINT_WORKER, str(socket_path), death,
         str(Path(__file__).resolve().parents[3]), str(pid_path)],
        stdout=subprocess_module.DEVNULL, stderr=subprocess_module.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while not socket_path.exists():
            assert worker.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        texture = session / "materials" / "texture"
        args = {"surface": str(surface), "prompt": "light grey metal", "seed": 0,
                "output": str(texture / "textured.glb"), "output_reference": str(texture / "reference.png")}
        with pytest.raises(ToolFailed) as failure:
            monitor.execute("paint_hunyuan3d_surface", args)
    finally:
        returncode = worker.wait(timeout=30)
    assert not one_shot
    if death == "segv":
        assert returncode == -signal.SIGSEGV and failure.value.receipt.returncode == -signal.SIGSEGV
        assert crashed(failure.value.receipt) and "crashed (SIGSEGV" in str(failure.value)
        assert "Segmentation fault" in Path(failure.value.receipt.stderr_path).read_text()
    else:
        assert returncode == -signal.SIGKILL and not crashed(failure.value.receipt)
        assert "killed by SIGKILL" in str(failure.value) and "out of memory" in str(failure.value)


def test_only_a_fault_in_native_code_is_a_crash():
    """SIGKILL (the out-of-memory killer, a user), SIGTERM and SIGINT say nothing about the
    inputs: no permanent crash record for them."""
    from src.agents_wrapper.fem_workflow.models import ToolReceipt
    from src.agents_wrapper.fem_workflow.monitor import crashed

    def receipt(code):
        return ToolReceipt(tool="paint_hunyuan3d_surface", arguments={}, returncode=code,
                           stdout_path="", stderr_path="", artifacts=[])

    for number in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT, signal.SIGILL, signal.SIGFPE):
        assert crashed(receipt(-number)) and crashed(receipt(128 + number))
    for number in (signal.SIGKILL, signal.SIGTERM, signal.SIGINT):
        assert not crashed(receipt(-number)) and not crashed(receipt(128 + number))
    assert not crashed(receipt(1)) and not crashed(receipt(0)) and not crashed(None)


def test_an_exit_status_over_128_is_a_signal_only_for_a_fatal_one():
    """A shell's 128 + n is a signal for the signals it reports; 255 (sys.exit(-1), a launcher's
    own failure) is an exit status, and its stderr line stays the diagnostic."""
    from src.agents_wrapper.fem_workflow.monitor import EXIT_SIGNALS, signal_of

    assert signal_of(-signal.SIGSEGV) == signal.SIGSEGV and signal_of(128 + signal.SIGKILL) == signal.SIGKILL
    assert signal_of(255) is None and signal_of(128 + 63) is None and signal_of(1) is None
    assert signal.SIGFPE in EXIT_SIGNALS and signal.SIGHUP not in EXIT_SIGNALS


@pytest.mark.parametrize("code, said, unsaid", [
    (255, "ValueError: no surface", "out of memory"),
    (-int(signal.SIGTERM), "stopped by SIGTERM", "out of memory"),
    (-int(signal.SIGKILL), "out of memory", "stopped by SIGKILL"),
    (-int(signal.SIGFPE), "crashed (SIGFPE", "out of memory")])
def test_a_failed_tool_says_its_own_error_unless_killed(tmp_path, monkeypatch, code, said, unsaid):
    """Only SIGKILL is said as out of memory: a stop keeps what the process printed, and an exit
    status that is no signal keeps its stderr line."""
    from src.agents_wrapper.fem_workflow import monitor as monitor_module
    from src.agents_wrapper.fem_workflow.monitor import ToolFailed

    session = tmp_path / "session"
    (session / "geometry").mkdir(parents=True)
    surface = session / "geometry" / "physical.glb"
    surface.write_bytes(b"glb")
    monkeypatch.setattr(ToolMonitor, "_run_worker", lambda self, *args, **kw: None)
    monkeypatch.setattr(monitor_module.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=code, stdout="", stderr="Traceback (most recent call last):\nValueError: no surface\n"))
    with pytest.raises(ToolFailed) as failure:
        ToolMonitor(session).execute("inspect_surface", {"surface": str(surface),
                                                         "output": str(session / "geometry" / "x.json")})
    assert said in str(failure.value) and unsaid not in str(failure.value)


def test_each_fault_name_is_cpythons_own(tmp_path):
    """The fault report's first line, as CPython's faulthandler writes it for each crash signal."""
    import subprocess as subprocess_module
    import sys as sys_module

    from src.agents_wrapper.fem_workflow.monitor import FAULT_NAMES

    code = ("import faulthandler, os, signal, sys\n"
            "f = open(sys.argv[2], 'w'); faulthandler.enable(file=f)\n"
            "os.kill(os.getpid(), int(sys.argv[1]))\n")
    for name, number in FAULT_NAMES.items():
        report = tmp_path / f"fault_{int(number)}"
        subprocess_module.run([sys_module.executable, "-c", code, str(int(number)), str(report)], check=False)
        assert report.read_text().startswith(f"Fatal Python error: {name}")


def test_the_paint_worker_lives_while_a_painting_card_is_open(monkeypatch, tmp_path):
    """Entering either card that paints starts the paint worker; any other card stops it."""
    from src.agents_wrapper.fem_workflow import workers
    from src.ui import controller as controller_module

    events = []
    monkeypatch.setattr(workers, "ensure_hunyuan3d_worker",
                        lambda root, env, paint=False: events.append(("start", paint)))
    monkeypatch.setattr(workers, "stop_hunyuan3d_worker",
                        lambda root: events.append(("stop",)))
    for kind in workers.PAINT_CARDS:
        workers.hunyuan3d_for_card(tmp_path, {"kind": kind})
    assert events == [("start", True), ("start", True)]
    events.clear()
    for card in ({"kind": "geometry_physics"}, {"kind": "bc_input"},
                 {"kind": "shape_opt_settings"}, {"kind": "geometry_input", "generator": "triposg"}):
        workers.hunyuan3d_for_card(tmp_path, card)
    assert events == [("stop",)] * 4
    events.clear()
    # The geometry card set to Hunyuan3D keeps it for generation.
    workers.hunyuan3d_for_card(tmp_path, {"kind": "geometry_input", "generator": "hunyuan3d"})
    assert events == []

    # Both front ends apply it on every card they enter.
    seen = []
    for module in (run_workflow, controller_module):
        monkeypatch.setattr(module, "hunyuan3d_for_card",
                            lambda root, card, env: seen.append(card["kind"]))
    monkeypatch.setattr(run_workflow, "ensure_tool_worker", lambda *a, **k: None)
    run_workflow.preload_for(tmp_path, tmp_path / "s", "mechfoundry-warp117",
                             {"kind": "material_input"})
    WorkflowController(tmp_path)._preload_for_interrupt(
        "abc", {"kind": "geometry_texture"}, {"compute_env": "mechfoundry-warp117"})
    assert seen == ["material_input", "geometry_texture"]


@pytest.mark.parametrize("free, kept", [(40_000, True), (20_000, False), (None, False)])
def test_vomp_keeps_the_paint_worker_when_it_fits(tmp_path, monkeypatch, free, kept):
    """VoMP runs beside the paint worker when the GPU has room, and unloads it otherwise."""
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    session = tmp_path / "session"
    session.mkdir()
    surface = session / "textured.glb"
    surface.write_bytes(b"glb")
    unloads = []
    for name in ("geometry", "partfield", "hunyuan3d", "vomp"):
        monkeypatch.setattr(ToolMonitor, f"_shutdown_{name}_worker",
                            lambda self, name=name: unloads.append(name))
    monkeypatch.setattr(ToolMonitor, "_shutdown_cad_workers", lambda self, keep=None: None)
    monkeypatch.setattr(ToolMonitor, "_run_worker", lambda self, *args, **kw: None)
    monkeypatch.setattr(monitor_module, "gpu_free_mib", lambda: free)
    monkeypatch.setattr("src.vomp_wrapper.availability.vomp_status", lambda *a: SimpleNamespace(
        ready=True, interpreter="/envs/mechfoundry-vomp/bin/python", environment="mechfoundry-vomp",
        summary="ready"))

    def fake_run(command, **kwargs):
        artifact = Path(command[command.index("--artifact") + 1])
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "manifest.json").write_text("{}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(monitor_module.subprocess, "run", fake_run)
    ToolMonitor(session).execute("infer_material_field", {
        "surface": str(surface), "artifact": str(session / "vomp"),
        "cache": str(tmp_path / "cache"), "unit_system": "SI (m-kg-s)"})
    assert ("hunyuan3d" in unloads) is not kept
    assert "vomp" not in unloads


@pytest.mark.parametrize("paint", [False, True])
def test_a_bare_surface_may_be_painted_before_scaling(monkeypatch, tmp_path, paint):
    """A surface with no colour gets an optional paint card before scaling: skipped, or painted and kept."""
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import ToolReceipt, WorkflowState

    calls = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, dict(arguments)))
            for key, value in arguments.items():
                if key.startswith("output"):
                    path = Path(value)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}") if path.suffix == ".json" else path.write_bytes(b"stub")
                    if name == "stage_asset":
                        path.with_suffix(path.suffix + ".json").write_text(json_module.dumps(
                            {"source_name": path.name, "path": str(path), "sha256": "0" * 64}))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def describe_image(self, purpose, image, legend=None):
            return "a chair"

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "_surface_has_colour",
                        lambda path: None if not path else path.endswith("textured.glb"))
    ready = SimpleNamespace(ready=True, model_dump=lambda: {"ready": True, "summary": "ready"})
    monkeypatch.setattr(graph_module, "hunyuan3d_status", lambda: ready)
    monkeypatch.setattr(graph_module, "texture_status", lambda: ready)
    source = tmp_path / "chair.obj"
    source.write_bytes(b"obj")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    workflow = WorkflowState(session_id="t", session_dir=str(tmp_path), model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    card = graph.invoke(Command(resume={"description": "an armchair", "asset_path": str(source),
                                        "asset_kind": "surface"}), config=config)["__interrupt__"][0].value
    assert card["kind"] == "geometry_texture"
    assert card["surface_has_colour"] is False and card["texture_prompt"] == "an armchair"
    assert card["surface_views"].endswith("views.png")
    uploaded = tmp_path / "geometry" / "original.obj"
    if paint:
        card = graph.invoke(Command(resume={"action": "texture_surface", "texture_prompt": "oak",
                                            "texture_seed": 1}), config=config)["__interrupt__"][0].value
        assert card["kind"] == "geometry_texture"            # back on the card, to review
        painted = card["textured_surface"]
        assert painted.endswith("geometry/texture/textured.glb")
        assert next(a for n, a in calls if n == "paint_hunyuan3d_surface")["surface"] == str(uploaded)
        card = graph.invoke(Command(resume={"action": "use_texture"}), config=config)["__interrupt__"][0].value
    else:
        card = graph.invoke(Command(resume={"action": "skip_texture"}), config=config)["__interrupt__"][0].value
        assert not any(name == "paint_hunyuan3d_surface" for name, _ in calls)
    assert card["kind"] == "geometry_physics"
    # The geometry is read, and scaled, from the painted copy when it was kept.
    surface = painted if paint else str(uploaded)
    read = [a["surface"] for n, a in calls if n == "render_surface" and a["output"].endswith("geometry/views.png")]
    assert read[-1] == surface
    graph.invoke(Command(resume={"action": "apply_scaling", "up_axis": "y", "physical_extent": 1.0,
                                 "unit_system": "SI (m-kg-s)"}), config=config)
    assert next(a for n, a in calls if n == "scale_surface")["surface"] == surface


# --- Fixes for the operator skill's known issues (references/known_issues.md) ---


def _validating_graph():
    """Two cards; the first rejects a size it cannot use, after asking for it."""
    from typing import TypedDict
    from langgraph.graph import END, StateGraph
    from langgraph.types import interrupt

    class State(TypedDict, total=False):
        workflow: dict

    def ask_size(state):
        answer = interrupt({"kind": "size"})
        if not answer:
            raise ValueError("a size is required")
        return {"workflow": {**state.get("workflow", {}), "size": answer}}

    def ask_material(state):
        answer = interrupt({"kind": "material"})
        return {"workflow": {**state.get("workflow", {}), "material": answer}}

    graph = StateGraph(State)
    graph.add_node("size", ask_size)
    graph.add_node("material", ask_material)
    graph.set_entry_point("size")
    graph.add_edge("size", "material")
    graph.add_edge("material", END)
    return graph


def test_a_rejected_answer_leaves_its_card_answerable(monkeypatch, tmp_path):
    """LangGraph replays a rejected answer, even over a new one, unless the card is forked."""
    pytest.importorskip("langgraph")
    monkeypatch.setattr("src.ui.controller.build_graph", _validating_graph)
    controller = WorkflowController(tmp_path)
    (tmp_path / "s1").mkdir()
    start = controller._invoke("s1", {"workflow": {}})
    with pytest.raises(ValueError, match="a size is required"):
        controller.resume("s1", "", start.checkpoint_id)
    # Opening the conversation again shows the card rather than the error ...
    assert controller.resume("s1").interrupt == {"kind": "size"}
    # ... and the corrected answer is the one that lands, from the card as drawn.
    with pytest.raises(ValueError):
        controller.resume("s1", "", start.checkpoint_id)
    assert controller.resume("s1", "10 cm", start.checkpoint_id).interrupt == {"kind": "material"}
    assert controller.resume("s1", "oak").workflow == {"size": "10 cm", "material": "oak"}


def test_the_command_line_reopens_a_rejected_card(tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver

    from src.agents_wrapper.fem_workflow.graph import (
        failed_step, reopen_card, resume_command,
    )

    graph = _validating_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"workflow": {}}, config=config)
    with pytest.raises(ValueError):
        graph.invoke(resume_command(graph, config, ""), config=config)
    assert failed_step(graph, config)
    # Without a checkpoint: the newest one waiting on a card, as `resume` uses it.
    reopened = reopen_card(graph, config)
    assert reopened["__interrupt__"][0].value == {"kind": "size"}
    assert not failed_step(graph, config)
    result = graph.invoke(resume_command(graph, config, "10 cm"), config=config)
    assert result["__interrupt__"][0].value == {"kind": "material"}


def test_the_command_line_prints_how_to_finish_after_a_solve():
    assert "finish" in run_workflow.finish_hint({"kind": "shape_opt_settings",
                                                 "solve_complete": True})
    assert run_workflow.finish_hint({"kind": "shape_opt_settings",
                                     "solve_complete": False}) is None


def test_cards_keep_what_they_show_and_a_failed_solve_hides_older_results(
        monkeypatch, tmp_path):
    """One pass over the cards with answers that leave fields out.

    Each field left out keeps the card's value (scaling axis, extent and
    units; the segmentation method; a Dirichlet region; the solver
    settings), a force is spread over the loaded area, a bad material answer
    costs nothing, and a failed solve stops offering an earlier run's surfaces.
    """
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.assets import SOLVE_FAILURE, newest_surface_vtu
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCProposal, DirichletBC, NeumannBC, ToolReceipt, WorkflowState,
    )

    calls = []
    solves = {"count": 0}
    regions = {"regions": [
        {"key": "dirichlet_0", "category": "dirichlet", "name": "feet",
         "area": 1.0, "loaded_area": 1.0},
        {"key": "neumann_1", "category": "neumann", "name": "top",
         "area": 2.0, "loaded_area": 0.5},
    ], "legend": []}

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, dict(arguments)))
            if name == "run_forward":
                solves["count"] += 1
                if solves["count"] == 1:
                    raise RuntimeError("run_forward failed: AMGX did not converge")
                if solves["count"] == 3:
                    # A shape optimization that failed at iteration 5, as
                    # run_forward leaves it: the run's log and status, and the
                    # displacement asset rebuilt from iteration 4.
                    run = session / "solve" / "api_fem_forward-opt"
                    (run / "vtu_surface").mkdir(parents=True)
                    (run / "log.txt").write_text("===== Optimization Iteration 5 =====")
                    (run / "optimization_status.json").write_text(json.dumps({
                        "status": "failed", "failed_iteration": 5, "shown_iteration": 4}))
                    (session / "solve" / "displacement_magnitude.glb").write_bytes(b"iter 4")
                    raise RuntimeError("run_forward failed: Newton solver: NaN residual")
            for key, value in arguments.items():
                if not key.startswith("output"):
                    continue
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "inspect_surface":
                    path.write_text(json.dumps({"extents": [2.0, 1.0, 0.4]}))
                elif name == "resolve_bcs" and path.suffix == ".json":
                    path.write_text(json.dumps(regions))
                elif name == "segment_surface" and path.suffix == ".json":
                    path.write_text(json.dumps({"segments": [], "bounds": []}))
                elif path.suffix == ".json":
                    path.write_text("{}")
                else:
                    path.write_bytes(b"stub")
                if name == "stage_asset":
                    path.with_suffix(path.suffix + ".json").write_text(json.dumps(
                        {"source_name": path.name, "path": str(path), "sha256": "0" * 64}))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    proposal = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, components=[0, 1, 2],
                               selector=SelectorSpec(type="lowest",
                                                     params={"axis": "z", "fraction": 0.1}))],
        neumann=[NeumannBC(name="top", traction=(0.0, 0.0, -1.0), selector=SelectorSpec(
            type="highest", params={"axis": "z", "fraction": 0.1}))],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        _segment_legend = staticmethod(SpecialistCalls._segment_legend)

        def describe_image(self, purpose, image, legend=None):
            return "a bench"

        def synthesize_bcs(self, description, metadata, montage):
            return proposal, APIReceipt(purpose="bc_synthesis", provider="groq",
                                        model="fake", cache_key="k")

    not_ready = SimpleNamespace(ready=False, model_dump=lambda: {"ready": False, "summary": "off"})
    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "_surface_has_colour", lambda path: True)
    monkeypatch.setattr(graph_module, "hunyuan3d_status", lambda: not_ready)
    monkeypatch.setattr(graph_module, "texture_status", lambda: not_ready)
    monkeypatch.setattr(graph_module, "cad_generator_status", lambda name: not_ready)
    monkeypatch.setattr(graph_module, "partfield_status", lambda: SimpleNamespace(
        ready=True, model_dump=lambda: {"ready": True, "summary": "ready"}))
    source = tmp_path / "bench.glb"
    source.write_bytes(b"glb")
    session = tmp_path / "session"
    session.mkdir()
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "cards"}}

    def card(answer):
        command = graph_module.resume_command(graph, config, answer)
        return graph.invoke(command, config=config)["__interrupt__"][0].value

    def workflow():
        return graph.get_state(config).values["workflow"]

    graph.invoke({"workflow": WorkflowState(
        session_id="cards", session_dir=str(session), model="fake").model_dump(mode="json")},
        config=config)
    scaling = card({"description": "a bench", "asset_path": str(source), "asset_kind": "surface"})
    # The measured extent is offered with the axis it was measured along.
    assert (scaling["up_axis"], scaling["physical_extent"]) == ("z", 0.4)

    bc_card = card({"unit_system": "SI (m-kg-s)"})
    scaled = next(arguments for name, arguments in calls if name == "scale_surface")
    assert (scaled["axis"], scaled["extent"]) == ("z", 0.4)
    assert bc_card["segmentation"]["method"] == "partfield"

    card({"action": "preview_regions", "description": "fix the feet, load the top",
          "segmentation": {"num_clusters": 7}})
    segmented = next(arguments for name, arguments in calls if name == "segment_surface")
    assert (segmented["method"], segmented["clusters"]) == ("partfield", 7)

    values_card = card({"action": "approve_regions"})
    assert values_card["neumann"][0]["loaded_area"] == 0.5
    material_card = card({"neumann_values": {"top": {"force": [0.0, 0.0, -10.0]}}})
    assert material_card["kind"] == "material_input"
    loads = {item["name"]: item for item in workflow()["bc_proposal"]["neumann"]}
    assert loads["top"]["traction"] == [0.0, 0.0, -20.0]
    assert workflow()["bc_proposal"]["dirichlet"][0]["components"] == [0, 1, 2]

    # A uniform material without E is refused by name, and the card survives it.
    with pytest.raises(ValueError, match="youngs_modulus"):
        card({})
    assert graph_module.failed_step(graph, config)
    assert graph_module.reopen_card(graph, config)["__interrupt__"][0].value["kind"] == (
        "material_input")
    settings_card = card({"youngs_modulus": 2.0e11, "poisson_ratio": 0.3, "density": 7850.0})
    assert settings_card["finish_answer"] == {"action": "finish"}
    # The automatic learning rate is offered, and off, on a new card.
    assert (settings_card["defaults"]["gd_lr_auto"],
            settings_card["defaults"]["gd_lr_auto_fraction"]) == (False, 0.5)

    solve_root = session / "solve"
    old_run = solve_root / "api_fem_forward-old"
    (old_run / "vtu_surface").mkdir(parents=True)
    old_surface = old_run / "vtu_surface" / "surface_t0000.vtu"
    old_surface.write_text("<VTKFile></VTKFile>")
    (old_run / "log.txt").write_text("Simulation finalized.")
    for path in (old_surface, old_run / "log.txt"):
        os.utime(path, (time.time() - 100.0, time.time() - 100.0))
    stale = solve_root / "displacement_magnitude.glb"
    stale.write_bytes(b"old")
    assert newest_surface_vtu(solve_root) == old_surface

    failed = card({"relative_tol": 1.0e-3})
    assert failed["solve_error"] and failed["solve_log_path"] is None
    # A forward solve's failure carries no learning-rate advice.
    assert "Recommended next" not in failed["solve_error"]
    assert (solve_root / SOLVE_FAILURE).is_file() and not stale.exists()
    assert newest_surface_vtu(solve_root) is None
    assert workflow()["shape_opt"]["relative_tol"] == 1.0e-3

    done = card({"load_steps": 3})
    assert done["solve_complete"]
    assert workflow()["shape_opt"]["relative_tol"] == 1.0e-3
    assert workflow()["shape_opt"]["load_steps"] == 3
    assert not (solve_root / SOLVE_FAILURE).exists()

    # A failed optimization keeps what it built from its last good iteration,
    # and says which iteration that is; assets from before it are still dropped.
    earlier = solve_root / "stress_von_mises.glb"
    earlier.write_bytes(b"from the forward solve")
    os.utime(earlier, (time.time() - 100.0, time.time() - 100.0))
    failed = card({"run_mode": "shape_optimization"})
    # The monitor's one-line error did not carry run_forward's advice, so the
    # graph adds it: no step record here, so a tenth of gd_lr.
    assert failed["solve_error"].endswith(
        "Newton solver: NaN residual Recommended next: first run a forward solve with the "
        "same setup (run mode: forward) to check that the case solves; if it does, retry "
        "the optimization with a smaller learning rate (gd_lr is 0.01; try 0.001).")
    assert failed["optimization"] == {
        "status": "failed", "failed_iteration": 5, "shown_iteration": 4}
    assert (solve_root / "displacement_magnitude.glb").read_bytes() == b"iter 4"
    assert not earlier.exists()
    record = json.loads((solve_root / SOLVE_FAILURE).read_text())
    assert record["optimization"]["shown_iteration"] == 4


def test_a_moved_session_works_in_the_folder_it_is_opened_from(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.graph import _rebased
    from src.agents_wrapper.fem_workflow.models import APIReceipt, WorkflowState

    assert _rebased({"a": ["/old/s/x.glb", "/old/sx"], "b": "/old/s"}, "/old/s", "/new/s") == {
        "a": ["/new/s/x.glb", "/old/sx"], "b": "/new/s"}

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        def refine_geometry(self, description, overview=""):
            return GeometryRefinement(refined_prompt="a solid bracket"), APIReceipt(
                purpose="geometry_refinement", provider="groq", model="fake", cache_key="k")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    original, moved = tmp_path / "original", tmp_path / "moved"
    original.mkdir()
    moved.mkdir()
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "m", "session_dir": str(moved)}}
    graph.invoke({"workflow": WorkflowState(
        session_id="m", session_dir=str(original), model="fake").model_dump(mode="json")},
        config=config)
    graph.invoke(Command(resume={"description": "bracket", "revise_prompt": True}),
                 config=config)
    assert graph.get_state(config).values["workflow"]["session_dir"] == str(moved)
    assert (moved / "session.json").is_file()
    assert not (original / "session.json").exists()


def test_a_union_of_more_than_two_regions_is_read_as_nested_pairs():
    leg = lambda index: {"type": "sphere", "params": {"center": [index, 0, 0], "radius": 1}}
    spec = SelectorSpec.model_validate({"type": "or", "children": [leg(i) for i in range(4)]})
    assert [child.type for child in spec.children] == ["sphere", "or"]
    assert [child.type for child in spec.children[1].children] == ["sphere", "or"]
    assert len(spec.children[1].children[1].children) == 2
    assert SelectorSpec.model_validate({"type": "and", "children": [leg(0), leg(1)]}).children[1].type == "sphere"
    with pytest.raises(ValueError, match="subtract requires 2 children"):
        SelectorSpec.model_validate({"type": "subtract", "children": [leg(i) for i in range(3)]})


def test_a_failed_call_is_logged_with_what_came_back(tmp_path):
    class WrongAgent:
        model = "fake-model"

        def generate_structured(self, *args, **kwargs):
            return {"prompt": "wrong key"}, GenerationResponse("{}", TokenUsage(8, 3, 11))

    calls = object.__new__(SpecialistCalls)
    calls.session_dir = tmp_path
    calls.cache_dir = tmp_path / "api_cache"
    calls.cache_dir.mkdir()
    calls.provider = "groq"
    calls.agent = WrongAgent()
    with pytest.raises(Exception):
        calls.refine_geometry("bracket")
    entry = json.loads((tmp_path / "api_log.jsonl").read_text().splitlines()[-1])
    assert entry["error"].startswith("ValidationError")
    assert "wrong key" in entry["response"]
    assert not list(calls.cache_dir.glob("*.json"))
    from src.ui.telemetry import read_backend_events
    assert read_backend_events(tmp_path)[-1]["title"].endswith("failed")


def test_a_force_is_divided_by_the_area_it_loads():
    import numpy as np

    from src.agents_wrapper.fem_workflow.graph import _loaded_area

    areas = np.array([1.0, 1.0])
    # A flat patch is loaded in full; a patch folded over an edge loses the cosine.
    assert _loaded_area(areas, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])) == pytest.approx(2.0)
    assert _loaded_area(areas, np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])) == pytest.approx(
        2.0 * math.sqrt(0.5))


def test_an_stl_upload_is_welded_when_staged(tmp_path):
    import trimesh

    source = tmp_path / "part.stl"
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(source)
    assert not trimesh.load(source, force="mesh", process=False).is_watertight
    target = tmp_path / "staged" / "original.glb"
    workflow_tool.stage_asset(SimpleNamespace(source=source, output=target))
    staged = trimesh.load(target, force="mesh", process=False)
    assert staged.is_watertight and len(staged.faces) == 12
    assert json.loads(target.with_suffix(".glb.json").read_text())["source_name"] == "part.stl"


def test_workers_are_keyed_on_the_resolved_output_root(monkeypatch, tmp_path):
    from src.agents_wrapper.fem_workflow.monitor import worker_paths

    monkeypatch.chdir(tmp_path)
    assert worker_paths(Path("roots/a"), "tools", "env") == worker_paths(
        tmp_path / "roots" / "a", "tools", "env")


def test_segment_thickness_is_reported_as_a_length(tmp_path):
    import numpy as np
    import trimesh

    from src.agents_wrapper.fem_workflow.segmentation_artifacts import write_segmentation

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    labels = np.zeros(len(mesh.faces), dtype=int)
    relative = np.full(len(mesh.faces), 0.5)
    write_segmentation(mesh, labels, method="shape_diameter", num_clusters=1,
                       output_npz=tmp_path / "s.npz", output_json=tmp_path / "s.json",
                       output_glb=tmp_path / "s.glb", shape_diameter=relative,
                       shape_diameter_range=(0.02, 0.10))
    segment = json.loads((tmp_path / "s.json").read_text())["segments"][0]
    assert segment["shape_diameter"]["mean"] == pytest.approx(0.06)
    assert segment["shape_diameter_relative"]["mean"] == pytest.approx(0.5)
    # The saved per-face array stays normalized: the selectors read it as such.
    assert np.load(tmp_path / "s.npz")["shape_diameter"][0] == pytest.approx(0.5)


def test_the_viewer_draws_an_uploaded_obj_or_ply_through_a_glb_copy(tmp_path):
    import trimesh

    from src.ui.browse import viewer_surface

    glb = tmp_path / "physical.glb"
    trimesh.creation.box().export(glb)
    assert viewer_surface(glb, tmp_path / "previews") == glb
    ply = tmp_path / "original.ply"
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(ply)
    preview = viewer_surface(ply, tmp_path / "previews")
    assert preview.suffix == ".glb" and preview.parent == tmp_path / "previews"
    assert trimesh.load(preview, force="mesh").extents == pytest.approx([1.0, 2.0, 3.0])
    # A rewind puts back an older upload at the same path: the copy of the
    # newer one is not taken for it, older than the copy though it is.
    newer = trimesh.creation.box(extents=(4.0, 5.0, 6.0))
    older = ply.read_bytes()
    newer.export(ply)
    assert trimesh.load(viewer_surface(ply, tmp_path / "previews"),
                        force="mesh").extents == pytest.approx([4.0, 5.0, 6.0])
    ply.write_bytes(older)
    earlier = preview.stat().st_mtime_ns - 3600 * 10**9
    os.utime(ply, ns=(earlier, earlier))
    assert trimesh.load(viewer_surface(ply, tmp_path / "previews"),
                        force="mesh").extents == pytest.approx([1.0, 2.0, 3.0])
    broken = tmp_path / "broken.obj"
    broken.write_text("not a mesh")
    assert viewer_surface(broken, tmp_path / "previews") is None


def test_the_solver_panel_learns_the_last_solve_failed(tmp_path):
    from src.ui.telemetry import last_solve_failure

    assert last_solve_failure(tmp_path) is None
    (tmp_path / "solve").mkdir()
    (tmp_path / "solve" / "failed.json").write_text(json.dumps(
        {"started": 1.0, "error": "RuntimeError: AMGX", "log": None}))
    record = last_solve_failure(tmp_path)
    assert record["error"].startswith("RuntimeError") and record["written"] > 0


def test_the_solver_panel_shows_no_error_box_for_a_failed_solve(tmp_path):
    """A failed solve is noted in the Solver panel, never quoted in a red box.

    The settings card reports the error (with its advice); the panel says only
    that the run failed and what it left.
    """
    import ast
    import warnings

    from src.ui.telemetry import solver_failure_notes

    assert solver_failure_notes(None) == []
    error = "RuntimeError: Linear solver failed: JAX-AMG stalled"
    notes = solver_failure_notes({"error": error, "log": None})
    assert notes[0] == "The last solve failed; the settings card shows the error."
    assert notes[-1] == "It stopped before the solver wrote a log."
    assert not any("JAX-AMG" in note or "RuntimeError" in note for note in notes)
    log = tmp_path / "log.txt"
    log.write_text("x")
    notes = solver_failure_notes({"error": error, "log": str(log), "optimization": {
        "status": "failed", "failed_iteration": 3, "shown_iteration": 2}})
    assert len(notes) == 2 and "iteration 2, the last that solved" in notes[1]

    # The panel itself draws no st.error/st.exception/st.warning box.
    app = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    with warnings.catch_warnings():     # app.py's regex literals, not this test's business
        warnings.simplefilter("ignore", DeprecationWarning)
        tree = ast.parse(app)
    panel = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "render_solver_tab")
    boxes = {node.func.attr for node in ast.walk(panel)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr in {"error", "exception", "warning"}}
    assert not boxes
    assert "solver_failure_notes(failure)" in ast.get_source_segment(app, panel)


def test_the_linear_solver_setting_defaults_to_jaxamg_and_offers_pyamgx():
    from pydantic import ValidationError

    from src.agents_wrapper.fem_workflow.models import (
        LINEAR_SOLVER_LABELS, LINEAR_SOLVERS, ShapeOptSettings, WorkflowState,
    )

    assert ShapeOptSettings().linear_solver == "jaxamg"
    assert LINEAR_SOLVERS == ("jaxamg", "pyamgx")
    assert LINEAR_SOLVER_LABELS == {"jaxamg": "JAX-AMG (default)", "pyamgx": "pyamgx"}
    assert ShapeOptSettings(linear_solver="pyamgx").linear_solver == "pyamgx"
    for other in ("amgx", "native", "PyAMGX", ""):
        with pytest.raises(ValidationError):
            ShapeOptSettings(linear_solver=other)
    # A session (or a case.json) stored before the setting existed runs JAX-AMG.
    old = WorkflowState(session_id="s", session_dir="/tmp/s").model_dump(mode="json")
    old["shape_opt"].pop("linear_solver")
    assert WorkflowState.model_validate(old).shape_opt.linear_solver == "jaxamg"
    assert ShapeOptSettings.model_validate(
        {"relative_tol": 1e-6, "absolute_tol": 1e-8}).linear_solver == "jaxamg"

    # The settings card offers the two, labelled, and sends the choice.
    app = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    card = app[app.index('"Linear solver",'):]
    card = card[:card.index("help=")]
    assert "LINEAR_SOLVERS" in card and "LINEAR_SOLVER_LABELS.get" in card
    assert '"linear_solver": linear_solver,' in app


def _stub_solver_modules(monkeypatch, built, failure=None):
    """Stand-ins for torch and the solver core, enough for run_forward's setup.

    ``built`` collects the params each ShapeOptTrimmed is constructed with; the
    stand-in's ``optimize`` fails at iteration 2 after two recorded steps, as the
    real loop's bookkeeping leaves them.
    """
    import types

    class Problem:
        def __init__(self, config, override_params=None):
            built.append(dict(override_params))
            self.params = SimpleNamespace(**{
                "use_mixed_precision": True, "relative_tol": 1e-2, "absolute_tol": 1e-5,
                "max_iter": 20, "load_steps": 1, "remesh_octree_depth": 7,
                "remesh_init_octree_depth": 4, "max_optimize_iters": 5,
                "output_per_optimize_step": 1, "volume_as_is": False,
                **override_params})
            self.path = str(Path(override_params["output_root"]) / "run")
            Path(self.path).mkdir(parents=True, exist_ok=True)
            self.logged = []

        def log(self, text):
            built[-1].setdefault("_log", []).append(text)

        def save(self, *args, **kwargs):
            pass

        def finalize(self):
            pass

        def forward(self, on_load_step=None):
            if failure:
                raise RuntimeError(failure)
            raise StopIteration("forward stub")

        def optimize(self):
            self._optimize_iteration = 2
            self._optimize_failed_iteration = 2
            self._optimize_steps = [
                {"iteration": 0, "max_abs_grad": 0.1, "max_move": 0.001},
                {"iteration": 1, "max_abs_grad": 0.4, "max_move": 0.004},
            ]
            raise RuntimeError(failure or "Newton solver: NaN residual")

    def module(name, **attrs):
        made = types.ModuleType(name)
        made.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, made)
        return made

    module("torch", cuda=SimpleNamespace(is_available=lambda: False))
    module("core")
    module("core.nitsche_field", bc_match_tolerance=lambda eps, h: eps)
    module("core.shape_opt", ShapeOptTrimmed=Problem)
    class Selector:      # a dict key in run_forward's BC maps
        def __init__(self, **params):
            self.params = params
    module("core.bc_surface_selectors", LowestSelector=Selector, HighestSelector=Selector)


def _stub_case(tmp_path, **shape_opt):
    import trimesh

    surface = tmp_path / "physical.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(surface)
    case = tmp_path / "case.json"
    case.write_text(json.dumps({
        "surface": str(surface), "output": str(tmp_path / "solve"),
        "h_target": 0.01, "octree_depth": 7, "mesh_center": [0.0, 0.0, 0.0],
        "material": {"mode": "uniform", "youngs_modulus": 1e6, "poisson_ratio": 0.3,
                     "unit_system": "SI (m-kg-s)"},
        "shape_opt": {"preset_version": 2, **shape_opt},
        "boundary_conditions": {
            "dirichlet": [{"name": "feet", "value": 0.0, "components": [0, 1, 2],
                           "selector": {"type": "lowest", "params": {"axis": "z"}}}],
            "neumann": [{"name": "top", "traction": [0.0, 0.0, -1.0],
                         "selector": {"type": "highest", "params": {"axis": "z"}}}]},
    }))
    return case


def test_run_forward_hands_the_solver_the_chosen_linear_backend(tmp_path, monkeypatch):
    """jaxamg is today's cg_backend_assembled "amgx"; pyamgx is pyamgx itself.

    A pyamgx solve refuses a process whose AMGX JAX-AMG already loaded.
    """
    import argparse

    from src.agents_wrapper.fem_workflow.scripts import workflow_tool

    assert workflow_tool.assembled_backend("jaxamg") == "amgx"
    assert workflow_tool.assembled_backend("pyamgx") == "pyamgx"
    built = []
    _stub_solver_modules(monkeypatch, built)
    for name in workflow_tool._JAXAMG_NATIVE:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert "jaxamg._amgx" in workflow_tool._JAXAMG_NATIVE
    for choice, backend in ((None, "amgx"), ("jaxamg", "amgx"), ("pyamgx", "pyamgx")):
        case = _stub_case(tmp_path, **({"linear_solver": choice} if choice else {}))
        with pytest.raises(StopIteration):
            workflow_tool.run_forward(argparse.Namespace(case=case))
        assert built[-1]["cg_backend_assembled"] == backend
        assert any(f"cg_backend_assembled={backend}" in line for line in built[-1]["_log"])

    count = len(built)
    monkeypatch.setitem(sys.modules, "jaxamg._amgx", object())
    with pytest.raises(RuntimeError, match="cannot share a process"):
        workflow_tool.run_forward(argparse.Namespace(
            case=_stub_case(tmp_path, linear_solver="pyamgx")))
    assert len(built) == count
    with pytest.raises(StopIteration):    # JAX-AMG itself is fine there
        workflow_tool.run_forward(argparse.Namespace(case=_stub_case(tmp_path)))


def test_a_failed_optimization_recommends_a_forward_solve_then_an_epsilon_sized_rate():
    from src.agents_wrapper.fem_workflow.models import (
        OPTIMIZATION_STEP_FRACTION, optimization_failure_advice, optimization_step_record,
    )

    assert OPTIMIZATION_STEP_FRACTION == 0.5
    steps = [{"iteration": 0, "max_abs_grad": 0.5, "max_move": 0.005},
             {"iteration": 1, "max_abs_grad": 0.8, "max_move": 0.008}]
    record = optimization_step_record(0.01, steps, epsilon=0.0025, h_min=0.005)
    # gd_lr_new = 0.5 * eps / max|dJ/ds| of the last step taken.
    assert record["recommended_gd_lr"] == pytest.approx(0.5 * 0.0025 / 0.8)
    assert record["last_step"]["iteration"] == 1
    assert record["last_step"]["max_move_over_epsilon"] == pytest.approx(3.2)
    assert record["largest_move_over_epsilon"] == pytest.approx(3.2)
    advice = optimization_failure_advice(0.01, record)
    assert advice.startswith("Recommended next: first run a forward solve with the same setup")
    for said in ("up to 3.2 epsilon", "epsilon = 0.0025", "h_min = 0.005", "gd_lr 0.01",
                 "gd_lr = 0.0016 keeps each step under 0.5 epsilon"):
        assert said in advice

    # No step was taken (it failed before its first backward pass), or the
    # gradient is not finite: a tenth of the current rate.
    for empty in (optimization_step_record(0.01, [], 0.0025, 0.005),
                  optimization_step_record(0.01, [{"iteration": 0, "max_abs_grad": float("nan"),
                                                   "max_move": float("nan")}], 0.0025, 0.005),
                  optimization_step_record(0.01, steps, None, None), None):
        said = optimization_failure_advice(0.01, empty)
        assert said.endswith("smaller learning rate (gd_lr is 0.01; try 0.001).")
        assert "forward solve" in said

    # Steps already under the rule: a smaller rate is not offered as the fix.
    small = optimization_step_record(1e-4, [{"iteration": 4, "max_abs_grad": 1.0,
                                              "max_move": 1e-4}], 0.0025, 0.005)
    said = optimization_failure_advice(1e-4, small)
    assert "unlikely cause" in said and "gd_lr =" not in said


def test_run_forward_s_failed_optimization_carries_the_advice_and_its_numbers(
        tmp_path, monkeypatch):
    import argparse

    from src.agents_wrapper.fem_workflow.scripts import workflow_tool

    built = []
    _stub_solver_modules(monkeypatch, built)
    case = _stub_case(tmp_path, run_mode="shape_optimization", gd_lr=0.05,
                      phase_field_epsilon_factor=0.5)
    with pytest.raises(RuntimeError) as failure:
        workflow_tool.run_forward(argparse.Namespace(case=case))
    message = str(failure.value)
    assert message.startswith("Newton solver: NaN residual; shape optimization failed at "
                              "iteration 2")
    # h_target 0.01 at the case's depth: h_min 0.01, eps 0.005; the last step's
    # |dJ/ds| 0.4 moved 0.02 = 4 eps; 0.5 * 0.005 / 0.4 = 0.00625.
    assert "first run a forward solve" in message
    assert "up to 4 epsilon (epsilon = 0.005, h_min = 0.01) at gd_lr 0.05" in message
    assert "gd_lr = 0.0062 keeps each step under 0.5 epsilon" in message
    status = json.loads((tmp_path / "solve" / "run" / workflow_tool.OPTIMIZATION_STATUS)
                        .read_text())
    step = status["step_size"]
    assert step["recommended_gd_lr"] == pytest.approx(0.00625)
    assert step["phase_field_epsilon"] == pytest.approx(0.005)
    assert step["h_min"] == pytest.approx(0.01)
    assert step["last_step"] == {"iteration": 1, "max_abs_grad": 0.4,
                                 "max_move": pytest.approx(0.02),
                                 "max_move_over_epsilon": pytest.approx(4.0)}
    assert len(step["steps"]) == 2

    # The loop records its steps (bookkeeping beside the GD update).
    loop = (Path(__file__).parents[2] / "warp_wrapper" / "core" / "abstract_field.py").read_text(
        encoding="utf-8")
    assert "self._optimize_steps = []" in loop
    assert '"max_move": float(gd_lr) * max_abs_grad,' in loop


def test_gd_lr_auto_settings_are_off_by_default_and_bounded():
    from pydantic import ValidationError

    settings = ShapeOptSettings()
    assert settings.gd_lr_auto is False and settings.gd_lr_auto_fraction == 0.5
    on = ShapeOptSettings.model_validate({"gd_lr_auto": True, "gd_lr_auto_fraction": 2.0})
    again = ShapeOptSettings.model_validate(on.model_dump(mode="json"))
    assert (again.gd_lr_auto, again.gd_lr_auto_fraction) == (True, 2.0)
    # A stored settings dict from before the option loads with it off.
    old = ShapeOptSettings().model_dump(mode="json")
    old.pop("gd_lr_auto"), old.pop("gd_lr_auto_fraction")
    assert ShapeOptSettings.model_validate(old).gd_lr_auto is False
    for fraction in (0.0, -0.1, 2.5):
        with pytest.raises(ValidationError):
            ShapeOptSettings(gd_lr_auto_fraction=fraction)


def test_run_forward_passes_gd_lr_auto_only_when_on_and_records_every_rate_it_chose(
        tmp_path, monkeypatch):
    import argparse

    from src.agents_wrapper.fem_workflow.scripts import workflow_tool

    built = []
    _stub_solver_modules(monkeypatch, built)
    base = sys.modules["core.shape_opt"].ShapeOptTrimmed

    class AutoProblem(base):
        def optimize(self):
            if not getattr(self.params, "gd_lr_auto", False):
                return super().optimize()
            # What the core leaves with the rate sized at every update:
            # 0.5 * eps 0.005 / 0.1 = 0.025, then / 0.4 = 0.00625; each step's
            # largest move is 0.0025 = 0.5 eps.
            rates = [(0, 0.1, 0.025), (1, 0.4, 0.00625)]
            self._optimize_steps = [
                {"iteration": k, "max_abs_grad": g, "max_move": g * r, "gd_lr": r}
                for k, g, r in rates]
            self._gd_lr_auto = {
                "gd_lr_configured": 0.05, "fraction": 0.5, "length": 0.005,
                "length_name": "epsilon",
                "steps": [{"iteration": k, "max_abs_grad": g, "gd_lr": r, "applied": True}
                          for k, g, r in rates],
                "iteration": 1, "max_abs_grad": 0.4, "gd_lr_effective": 0.00625,
                "applied": True}
            self._optimize_iteration = self._optimize_failed_iteration = 2
            raise RuntimeError("Newton solver: NaN residual")

    monkeypatch.setattr(sys.modules["core.shape_opt"], "ShapeOptTrimmed", AutoProblem)
    # Off (the default): the solver is handed nothing new, and the advice offers it.
    with pytest.raises(RuntimeError) as failure:
        workflow_tool.run_forward(argparse.Namespace(case=_stub_case(
            tmp_path, run_mode="shape_optimization", gd_lr=0.05)))
    assert not any(key.startswith("gd_lr_auto") for key in built[-1])
    assert "or turn on gd_lr_auto, which sizes every step" in str(failure.value)
    status_path = tmp_path / "solve" / "run" / workflow_tool.OPTIMIZATION_STATUS
    assert "gd_lr_auto" not in json.loads(status_path.read_text())

    with pytest.raises(RuntimeError) as failure:
        workflow_tool.run_forward(argparse.Namespace(case=_stub_case(
            tmp_path, run_mode="shape_optimization", gd_lr=0.05, gd_lr_auto=True,
            gd_lr_auto_fraction=0.5)))
    params = built[-1]
    assert (params["gd_lr_auto"], params["gd_lr_auto_fraction"]) == (True, 0.5)
    # Its length is the solver's own epsilon: factor 0.5 x the 0.01 leaf.
    assert params["phase_field_epsilon"] == pytest.approx(0.005)
    assert "gd_lr_auto_length" not in params
    status = json.loads(status_path.read_text())
    # Every update's rate, and the last one.
    assert [s["gd_lr"] for s in status["gd_lr_auto"]["steps"]] == pytest.approx([0.025, 0.00625])
    assert status["gd_lr_auto"]["gd_lr_effective"] == pytest.approx(0.00625)
    step = status["step_size"]
    assert [s["gd_lr"] for s in step["steps"]] == pytest.approx([0.025, 0.00625])
    # The last step was taken at the last rate: |dJ/ds| 0.4 * 0.00625 = 0.5 eps.
    assert step["gd_lr"] == pytest.approx(0.00625)
    assert step["last_step"]["max_move_over_epsilon"] == pytest.approx(0.5)
    assert step["largest_move_over_epsilon"] == pytest.approx(0.5)
    message = str(failure.value)
    # With every step at the fraction, the knob is the fraction: halve it.
    assert "gd_lr_auto moved the surface by up to 0.5 epsilon" in message
    assert "lower gd_lr_auto_fraction to 0.25" in message


def test_the_solver_panel_reads_the_rates_gd_lr_auto_chose():
    from src.ui.telemetry import gd_lr_auto_note, read_solver_progress

    stamp = "[10-20-2026,10:00:00.000000]"
    lines = [f"{stamp} ===== Optimization Iteration 0 =====",
             f"{stamp} gd_lr_auto: iteration 0: gd_lr 2.5000e-02 = 0.5 * epsilon "
             "5.0000e-03 / max|dJ/ds| 1.0000e-01",
             f"{stamp} ===== Optimization Iteration 1 =====",
             f"{stamp} gd_lr_auto: iteration 1: gd_lr 6.2500e-03 = 0.5 * epsilon "
             "5.0000e-03 / max|dJ/ds| 4.0000e-01"]
    progress = read_solver_progress("\n".join(lines[:2]) + "\n")
    assert [e["gd_lr"] for e in progress.gd_lr_auto] == pytest.approx([0.025])
    one = gd_lr_auto_note(progress)
    assert one.startswith("Auto learning rate: gd_lr 0.025 at iteration 0; each step")
    assert "range" not in one and "up to 0.5 ε (ε = 0.005)" in one
    progress = read_solver_progress("\n".join(lines) + "\n")
    assert [e["iteration"] for e in progress.gd_lr_auto] == [0, 1]
    note = gd_lr_auto_note(progress)
    # The latest rate, and the range so far.
    assert note.startswith("Auto learning rate: gd_lr 0.00625 at iteration 1 "
                           "(range 0.00625–0.025 over 2 steps)")
    kept = read_solver_progress(f"{stamp} gd_lr_auto: iteration 0: max|dJ/ds| is 0; "
                                "nothing to scale, keeping gd_lr 0.01\n")
    assert "was kept" in gd_lr_auto_note(kept)
    # Every run without the option: no line, no note.
    assert gd_lr_auto_note(read_solver_progress(f"{stamp} Opt iter 0: obj=1.0\n")) is None


JAXAMG_STALL = (
    "Linear solver failed at Newton iteration 0: JAX-AMG solve failed with status "
    "AMGXStatus.NOT_CONVERGED after 1000 iterations (||b - Kx||/||b||=4.1e-02); it stalled "
    "4e+04x above the 1.0e-06 tolerance, which is the rounding floor of an ill-conditioned "
    "system, not a tolerance set too tight -- try scale_system_to_unit_diagonal=True, then "
    "beta_max_over_e, to cap a Nitsche penalty")


def test_a_jaxamg_linear_failure_recommends_pyamgx_and_a_pyamgx_one_does_not(
        tmp_path, monkeypatch):
    import argparse

    from src.agents_wrapper.fem_workflow.scripts import workflow_tool

    for name in workflow_tool._JAXAMG_NATIVE:
        monkeypatch.delitem(sys.modules, name, raising=False)
    said = {}
    for choice in ("jaxamg", "pyamgx"):
        _stub_solver_modules(monkeypatch, [], failure=JAXAMG_STALL)
        with pytest.raises(RuntimeError) as failure:
            workflow_tool.run_forward(argparse.Namespace(
                case=_stub_case(tmp_path, linear_solver=choice)))
        said[choice] = str(failure.value)
    # The solver's own advice stays; pyamgx is added after it, once.
    assert said["jaxamg"] == JAXAMG_STALL + workflow_tool.PYAMGX_NOTE
    assert 'linear_solver "pyamgx"' in workflow_tool.PYAMGX_NOTE
    assert said["pyamgx"] == JAXAMG_STALL

    # A failure that is not JAX-AMG's linear solve gets no such advice.
    _stub_solver_modules(monkeypatch, [], failure="Newton solver did not converge in 20 iterations")
    with pytest.raises(RuntimeError) as failure:
        workflow_tool.run_forward(argparse.Namespace(case=_stub_case(tmp_path)))
    assert "pyamgx" not in str(failure.value)


def test_a_failed_optimization_carries_the_pyamgx_and_the_step_advice_once_each(
        tmp_path, monkeypatch):
    import argparse

    from src.agents_wrapper.fem_workflow.scripts import workflow_tool

    _stub_solver_modules(monkeypatch, [], failure=JAXAMG_STALL)
    case = _stub_case(tmp_path, run_mode="shape_optimization", gd_lr=0.05)
    with pytest.raises(RuntimeError) as failure:
        workflow_tool.run_forward(argparse.Namespace(case=case))
    message = str(failure.value)
    assert message.startswith(JAXAMG_STALL + workflow_tool.PYAMGX_NOTE
                              + "; shape optimization failed at iteration 2")
    assert message.count('linear_solver "pyamgx"') == 1
    assert message.count("Recommended next:") == 1 and "first run a forward solve" in message


def test_the_page_shows_a_moved_session_s_files_from_where_it_is_now(tmp_path):
    from src.ui.controller import moved_here

    session = tmp_path / "root" / "abc123def456"
    old = "/elsewhere/output/abc123def456"
    payload = {"surface_views": f"{old}/geometry/views.png", "kind": "geometry_physics",
               "list": [f"{old}/solve", "/unrelated/file.png", "abc123def456"],
               "here": str(session / "geometry" / "physical.glb")}
    assert moved_here(payload, session) == {
        "surface_views": str(session / "geometry" / "views.png"),
        "kind": "geometry_physics",
        "list": [str(session / "solve"), "/unrelated/file.png", "abc123def456"],
        "here": str(session / "geometry" / "physical.glb"),
    }
    controller = WorkflowController(tmp_path / "root")
    session.mkdir(parents=True)
    (session / "session.json").write_text(json.dumps(
        {"session_dir": old, "surface_path": f"{old}/geometry/physical.glb"}))
    (session / "ui_history.json").write_text(json.dumps(
        [{"role": "assistant", "interrupt": {"surface_views": f"{old}/geometry/views.png"}}]))
    assert controller.read_workflow("abc123def456")["surface_path"] == str(
        session / "geometry" / "physical.glb")
    assert controller.history("abc123def456")[0]["interrupt"]["surface_views"] == str(
        session / "geometry" / "views.png")


def test_saved_part_labels_survive_the_session_moving(tmp_path):
    """Labels are matched to the surface by its contents, not the path it had."""
    from src.agents_wrapper.fem_workflow.graph import _segments_current, _segments_source
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    old, new = tmp_path / "old" / "s", tmp_path / "new" / "s"
    for root in (old, new):
        (root / "geometry").mkdir(parents=True)
        (root / "geometry" / "physical.glb").write_bytes(b"the same surface")
    npz = old / "segments.npz"
    npz.write_bytes(b"labels")
    state = WorkflowState(session_id="s", session_dir=str(old),
                          surface_path=str(old / "geometry" / "physical.glb"))
    (old / "segments.source.json").write_text(json.dumps(_segments_source(state)))
    for name in ("segments.npz", "segments.source.json"):
        (new / name).write_bytes((old / name).read_bytes())
    moved = state.model_copy(update={"session_dir": str(new),
                                     "surface_path": str(new / "geometry" / "physical.glb")})
    assert _segments_current(moved, new / "segments.npz")
    (new / "geometry" / "physical.glb").write_bytes(b"a rescaled surface")
    assert not _segments_current(moved, new / "segments.npz")


# ---------------------------------------------------------------- SDXL input --

def _ready_sdxl():
    return SimpleNamespace(
        ready=True, interpreter="/envs/mechfoundry-hunyuan3d/bin/python",
        environment="mechfoundry-hunyuan3d", summary="SDXL ready",
        model_dump=lambda: {"ready": True},
    )


def _sdxl_graph(monkeypatch, tmp_path, sdxl=None):
    """The geometry card with SDXL (and Hunyuan3D) ready and every tool faked."""
    from src.agents_wrapper.fem_workflow import graph as graph_module

    monkeypatch.setattr(graph_module, "sdxl_status", lambda: sdxl or _ready_sdxl())
    return _image_graph(monkeypatch, tmp_path, _ready_hunyuan3d())


def test_sdxl_draws_the_description_and_the_card_comes_back_with_it(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    graph, config, card, calls = _sdxl_graph(monkeypatch, tmp_path)
    assert card["sdxl"] == {"ready": True} and card["drawn_image"] is None
    assert "image_prompt" not in card["fields"], "the description is the drawing's prompt"
    settings = {"generator": "hunyuan3d", "seed": 7, "inference_steps": 30}
    after = graph.invoke(Command(resume={
        "action": "draw_image", "description": "a sturdy oak bench, a thick plank seat",
        "image_seed": 3, "asset_kind": "image", "generation_settings": settings,
    }), config=config)
    name, arguments = calls[-1]
    assert name == "draw_sdxl_image"
    assert arguments["prompt"] == "a sturdy oak bench, a thick plank seat"
    assert arguments["seed"] == 3
    drawn = Path(arguments["output"])
    assert drawn.parent == tmp_path / "inputs" / "sdxl" and drawn.name.endswith("_s3.png")
    assert arguments["output_raw"] == str(drawn.with_name(drawn.stem + "_drawn.png"))
    # Back on the same card, showing the drawing and the card as it was sent.
    card = after["__interrupt__"][0].value
    assert card["kind"] == "geometry_input"
    assert card["drawn_image"] == str(drawn)
    assert (card["drawn_image_prompt"], card["image_seed"]) == (
        "a sturdy oak bench, a thick plank seat", 3)
    assert card["description"] == "a sturdy oak bench, a thick plank seat"
    # An answer that leaves the description out draws what the card shows.
    graph.invoke(Command(resume={"action": "draw_image", "image_seed": 4}), config=config)
    assert calls[-1][1]["prompt"] == "a sturdy oak bench, a thick plank seat"
    assert calls[-1][1]["seed"] == 4
    assert card["asset_kind"] == "image" and card["generation_settings"] == settings
    assert card["generator"] == "hunyuan3d"
    assert not any(n.startswith("generate_") for n, _ in calls), "nothing built yet"


def test_the_card_says_what_of_the_prompt_sdxl_never_read(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module

    graph, config, card, calls = _sdxl_graph(monkeypatch, tmp_path)
    assert card["drawn_image_dropped"] == "" and card["drawn_image_description_cut"] is False
    fake = graph_module.ToolMonitor.execute

    def execute(self, name, arguments):
        receipt = fake(self, name, arguments)
        out = tmp_path / f"{name}.stdout.log"
        out.write_text("loading\n" + json.dumps({"prompt_tokens": 101, "token_limit": 77,
                                                 "dropped": "plain white background, product photo",
                                                 "description_cut": False}) + "\n")
        return receipt.model_copy(update={"stdout_path": str(out)})
    monkeypatch.setattr(graph_module.ToolMonitor, "execute", execute)
    after = graph.invoke(Command(resume={"action": "draw_image", "description": "a long clog",
                                         "image_seed": 1}), config=config)
    card = after["__interrupt__"][0].value
    assert card["drawn_image_dropped"] == "plain white background, product photo"
    assert card["drawn_image_description_cut"] is False


def test_the_drawing_feeds_any_image_generator(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    for generator, tool, steps in (("triposg", "generate_image_surface", 50),
                                   ("hunyuan3d", "generate_hunyuan3d_image_surface", 30)):
        root = tmp_path / generator
        graph, config, _, calls = _sdxl_graph(monkeypatch, root)
        graph.invoke(Command(resume={"action": "draw_image", "description": "an oak bench"}),
                     config=config)
        drawn = calls[-1][1]["output"]
        graph.invoke(Command(resume={
            "description": "an oak bench", "use_drawn_image": True, "generator": generator,
        }), config=config)
        stage = next(arguments for name, arguments in calls if name == "stage_asset")
        assert stage["source"] == drawn
        assert stage["output"] == str(root / "inputs" / "original.png")
        name, arguments = next(call for call in calls if call[0].startswith("generate_"))
        assert name == tool
        assert arguments["input"] == str(root / "inputs" / "original.png")
        # An image generation's own default step count, as for an upload.
        assert arguments["steps"] == steps


@pytest.mark.parametrize("answer, sdxl, message", [
    ({"action": "draw_image", "description": "a bench"},
     SimpleNamespace(ready=False, summary="weights missing", model_dump=lambda: {"ready": False}),
     "SDXL is not available: weights missing"),
    ({"action": "draw_image", "description": " "}, None, "SDXL draws the description"),
    ({"action": "draw_image", "description": "a bench", "image_seed": -1}, None, "image_seed"),
    ({"description": "a bench", "use_drawn_image": True}, None, "draw an image before using it"),
    ({"action": "draw", "description": "a bench"}, None, "only action is draw_image"),
])
def test_drawing_needs_sdxl_words_and_a_drawing_to_use(monkeypatch, tmp_path, answer, sdxl,
                                                        message):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    graph, config, _, calls = _sdxl_graph(monkeypatch, tmp_path, sdxl)
    with pytest.raises(ValueError, match=message):
        graph.invoke(Command(resume=answer), config=config)
    assert not calls


def test_the_sdxl_drawing_runs_once_in_its_own_environment(tmp_path, monkeypatch):
    """No resident worker: SDXL needs none of Hunyuan3D's models, so none are loaded."""
    from src.agents_wrapper.fem_workflow import monitor as monitor_module

    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    unloads = []
    for name in ("geometry", "partfield", "hunyuan3d", "vomp"):
        monkeypatch.setattr(ToolMonitor, f"_shutdown_{name}_worker",
                            lambda self, name=name: unloads.append(name))
    monkeypatch.setattr(ToolMonitor, "_shutdown_cad_workers",
                        lambda self, keep=None: unloads.append("cad"))
    monkeypatch.setattr(ToolMonitor, "_run_worker",
                        lambda *a, **k: pytest.fail("the drawing needs no worker"))
    monkeypatch.setattr(monitor_module, "sdxl_status", _ready_sdxl)
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        for flag in ("--output", "--output-raw"):
            Path(command[command.index(flag) + 1]).write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(monitor_module.subprocess, "run", fake_run)
    (session / "inputs" / "sdxl").mkdir(parents=True)
    arguments = {"prompt": "an oak bench", "seed": 3,
                 "output": str(session / "inputs" / "sdxl" / "a_s3.png"),
                 "output_raw": str(session / "inputs" / "sdxl" / "a_s3_drawn.png")}
    receipt = monitor.execute("draw_sdxl_image", arguments)
    # TripoSG and Hunyuan3D, the two that take the drawing next, may stay.
    assert unloads == ["partfield", "cad", "vomp"]
    command, kwargs = commands[0]
    assert command[0] == "/envs/mechfoundry-hunyuan3d/bin/python"
    assert command[1].endswith("hunyuan3d_wrapper/draw_image.py")
    assert command[2:] == ["--prompt", "an oak bench", "--output", arguments["output"],
                           "--output-raw", arguments["output_raw"], "--seed", "3"]
    assert kwargs["cwd"] == session
    assert sorted(Path(a).name for a in receipt.artifacts) == ["a_s3.png", "a_s3_drawn.png"]

    monkeypatch.setattr(monitor_module, "sdxl_status", lambda: SimpleNamespace(
        ready=False, summary="no environment", interpreter=None))
    with pytest.raises(ToolRejected, match="SDXL is not available: no environment"):
        monitor.execute("draw_sdxl_image", {**arguments, "seed": 4})


def test_triposg_erosion_is_a_setting_that_defaults_to_none(monkeypatch, tmp_path, socket_home):
    """The largest-component filter no longer erodes unless asked to.

    Eroding by 2 voxels, as it always did, cut a bench's slats off its frames
    and kept one leg; the radius is now the card's to choose, 0 by default, and
    reaches the generator by both routes.
    """
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    assert GeometryGenerationSettings().erosion_radius == 0
    with pytest.raises(ValueError):
        GeometryGenerationSettings(erosion_radius=9)
    graph, config, _, calls = _sdxl_graph(monkeypatch, tmp_path)
    graph.invoke(Command(resume={
        "description": "a slatted bench", "generation_settings": {"erosion_radius": 2},
    }), config=config)
    name, arguments = next(call for call in calls if call[0].startswith("generate_"))
    assert name == "generate_text_surface" and arguments["erosion_radius"] == 2

    # The one-shot command and the resident worker's request both carry it.
    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    tail = monitor.registry["generate_text_surface"].command(
        {"output": str(session / "g.glb"), "prompt": "a bench", "erosion_radius": 2})
    assert tail[tail.index("--erosion-radius") + 1] == "2"
    socket_path, _ = monitor._worker_paths("triposg")
    socket_home(socket_path.parent)
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    def answer(connection, seen):
        seen.append(json.loads(connection.makefile("r", encoding="utf-8").readline()))
        connection.sendall(b'{"ok":true,"result":{"artifacts":[]}}\n')

    seen, _thread = _fake_worker(socket_path, answer)
    monitor._run_geometry_worker("generate_text_surface",
                                 {"output": str(session / "g.glb"), "prompt": "a bench"}, 30)
    assert seen[0]["erosion_radius"] == 0


def test_the_generator_erodes_as_asked(monkeypatch, tmp_path):
    import sys

    module = _idle_generate_surface(monkeypatch)
    radii = []

    def stop_after_filter(sdf, erosion_radius, domain_length):
        radii.append(erosion_radius)
        raise RuntimeError("stop here")

    monkeypatch.setattr(module, "filter_largest_interior_component", stop_after_filter)
    fake_pipe = SimpleNamespace(text2latent=lambda *a, **k: None,
                                latent2sdf=lambda *a, **k: (None, None))
    monkeypatch.setattr(module.SurfaceGenerator, "load",
                        lambda self, *a, **k: setattr(self, "pipe", fake_pipe))
    monkeypatch.setattr(module.torch, "Generator",
                        lambda device: SimpleNamespace(manual_seed=lambda seed: None),
                        raising=False)
    monkeypatch.setattr(module.torch, "cuda",
                        SimpleNamespace(is_available=lambda: True), raising=False)
    for extra, expected in (((), 0), (("--erosion-radius", "2"), 2)):
        monkeypatch.setattr(sys, "argv", ["generate_surface.py", "--mode", "text",
                                          "--prompt", "a bench", "--device", "cpu",
                                          "--output", str(tmp_path / "g.glb"), *extra])
        with pytest.raises(RuntimeError, match="stop here"):
            module.generate(module.parse_args())
    assert radii == [0, 2]
    monkeypatch.setattr(sys, "argv", ["generate_surface.py", "--mode", "text",
                                      "--prompt", "a bench", "--device", "cpu",
                                      "--output", str(tmp_path / "g.glb"),
                                      "--erosion-radius", "9"])
    with pytest.raises(ValueError, match="erosion-radius"):
        module.generate(module.parse_args())


def test_a_geometry_worker_older_than_its_code_steps_aside(monkeypatch, tmp_path, socket_home):
    """It answers "stale" and exits, and the monitor runs the job itself.

    A worker holds the code it imported for as long as it lives, so one started
    before the erosion setting existed would have served a request for it as if
    it were absent.
    """
    import argparse
    import socket as socket_module
    import tempfile
    import threading

    module = _idle_generate_surface(monkeypatch)
    monkeypatch.setattr(module, "LOADED_DIGEST", "an older copy")
    home = Path(tempfile.mkdtemp(prefix="implicit-fem-stale-"))
    socket_home(home)
    socket_path = home / "triposg.sock"
    stopped = threading.Event()

    def run():
        try:
            module.serve(argparse.Namespace(serve=socket_path, device="cpu", preload="scribble",
                                            weights_dir=None, idle_timeout=30.0))
        finally:
            stopped.set()

    threading.Thread(target=run, daemon=True).start()
    deadline = time.monotonic() + 5.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    with socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM) as client:
        client.connect(os.fspath(socket_path))
        client.sendall(b'{"mode":"text","erosion_radius":0}\n')
        answer = json.loads(client.makefile("r", encoding="utf-8").readline())
    assert answer["ok"] is False and answer["stale"] is True
    assert stopped.wait(10.0), "a stale worker went on serving"

    session = tmp_path / "session"
    session.mkdir()
    monitor = ToolMonitor(session)
    worker_socket, _ = monitor._worker_paths("triposg")
    socket_home(worker_socket.parent)
    worker_socket.parent.mkdir(parents=True, exist_ok=True)

    def stale(connection, seen):
        seen.append(connection.makefile("r", encoding="utf-8").readline())
        connection.sendall(b'{"ok":false,"stale":true,"error":"changed on disk"}\n')

    seen, _thread = _fake_worker(worker_socket, stale)
    assert monitor._run_geometry_worker(
        "generate_text_surface", {"output": str(session / "g.glb"), "prompt": "a bench"},
        30) is None, "run once here instead of failing the generation"


def test_yield_strength_validation_and_asset_cache(tmp_path):
    from pydantic import ValidationError
    from src.agents_wrapper.fem_workflow.models import MaterialInference, MaterialParameters
    from src.agents_wrapper.fem_workflow.assets import field_asset_request
    material = dict(unit_system="SI (m-kg-s)", youngs_modulus=2e11, poisson_ratio=0.3)
    assert MaterialParameters(**material).yield_strength is None
    assert MaterialInference(youngs_modulus=2e11, poisson_ratio=0.3, yield_strength=250e6).yield_strength == 250e6
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            MaterialParameters(**material, yield_strength=value)
    # Strength is supplied in the viewer, so stress exports need no material.
    first, args = field_asset_request(str(tmp_path), "surface.vtu", "stress", "safety_factor")
    assert args["component"] == "von_mises"
    assert "yield_strength" not in args
    assert "safety_factor" in first


def test_safety_factor_export_ratio_zero_stress_and_colors(tmp_path, monkeypatch):
    import numpy as np
    from src.agents_wrapper.fem_workflow.scripts import workflow_tool as tool
    import trimesh
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    def surface(path, field, component):
        assert field == "stress" and component == "von_mises"
        return points, np.array([[0, 1, 2]]), np.zeros_like(points), np.array([500., 125., 0.])
    monkeypatch.setattr(tool, "_read_surface", surface)
    output = tmp_path / "safety.glb"
    tool._export_surface_field(tmp_path / "unused.vtu", output, "stress", "safety_factor", yield_strength=250.)
    metadata = json.loads(output.with_suffix(".glb.json").read_text())
    np.testing.assert_array_equal(np.fromfile(output.with_suffix(".glb.values.bin"), dtype="<f8"), [500., 125., 0.])
    assert metadata["observed_minimum"] == 0.5
    assert metadata["observed_maximum"] is None
    assert metadata["minimum"] == 0 and metadata["maximum"] == 5
    assert metadata["surface_sample_fraction_below_one"] == pytest.approx(1 / 3)
    mesh = trimesh.load(output, force="mesh")
    colors = mesh.visual.vertex_colors
    assert (colors == [205, 35, 45, 255]).all(axis=1).sum() == 1
    assert (colors == [55, 190, 95, 255]).all(axis=1).sum() == 2
    with pytest.raises(ValueError):
        tool._export_surface_field(tmp_path / "unused.vtu", output, "stress", "safety_factor", yield_strength=0.)


def test_hover_samples_preserve_values_and_segment_ids(tmp_path):
    import base64
    import numpy as np
    from src.ui.hover import surface_hover
    asset = tmp_path / "stress.glb"
    asset.write_bytes(b"placeholder")
    assert surface_hover(asset) is None
    values = np.array([0., -12.3456789, 60000000.], dtype="<f8")
    asset.with_suffix(".glb.values.bin").write_bytes(values.tobytes())
    asset.with_suffix(".glb.json").write_text(json.dumps({"field": "stress", "component": "xx"}))
    payload = surface_hover(asset)
    assert payload["component"] == "xx"
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(payload["data"]), dtype="<f8"), values)
    asset.with_suffix(".glb.values.bin").write_bytes(b"bad")
    assert surface_hover(asset) is None
    segments = tmp_path / "segments.glb"
    np.savez_compressed(segments.with_suffix(".npz"), segment_ids=np.array([0, 20, 1]))
    payload = surface_hover(segments)
    assert payload["kind"] == "segment"
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(payload["data"]), dtype="<i4"), [0, 20, 1])


def test_a_ready_compute_environment_is_remembered_for_every_root(monkeypatch, tmp_path):
    from src.agents_wrapper import environment

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    probes = []
    ready = environment.ComputeEnvironmentStatus("env", True, True, True, True, "2", "1.17")
    monkeypatch.setattr(environment, "probe_compute_environment",
                        lambda name: probes.append(name) or ready)
    stamp = [100.0]
    monkeypatch.setattr(environment, "_env_stamp", lambda name: stamp[0])
    assert environment.require_compute_environment("env") == ready
    assert environment.require_compute_environment("env") == ready and probes == ["env"]
    stamp[0] = 200.0                                     # something was installed: check again
    environment.require_compute_environment("env")
    assert probes == ["env", "env"]
    # A failed check is never remembered.
    broken = environment.ComputeEnvironmentStatus("bad", True, True, False, False, error="no CUDA")
    monkeypatch.setattr(environment, "probe_compute_environment", lambda name: probes.append(name) or broken)
    for _ in range(2):
        with pytest.raises(environment.ComputeEnvironmentError):
            environment.require_compute_environment("bad")
    assert probes.count("bad") == 2


def _write_surface_vtu(path, displacement, closed=True):
    """A one-triangle surface VTU in the writer's layout (base64, 8-byte size header)."""
    import base64 as b64

    import numpy as np

    def encoded(array):
        raw = np.ascontiguousarray(array).tobytes()
        return b64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode("ascii")

    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="<f8")
    displacement = np.asarray(displacement, dtype="<f8").reshape(3, 3)
    text = (
        '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0">'
        '<UnstructuredGrid><Piece NumberOfPoints="3" NumberOfCells="1">'
        f'<Points><DataArray type="Float64" NumberOfComponents="3" format="binary">{encoded(points)}</DataArray></Points>'
        f'<PointData><DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="binary">{encoded(displacement)}</DataArray></PointData>'
        f'<Cells><DataArray type="Int32" Name="connectivity" format="binary">{encoded(np.asarray([0, 1, 2], dtype="<i4"))}</DataArray>'
        f'<DataArray type="Int32" Name="offsets" format="binary">{encoded(np.asarray([3], dtype="<i4"))}</DataArray>'
        f'<DataArray type="UInt8" Name="types" format="binary">{encoded(np.asarray([5], dtype="u1"))}</DataArray></Cells>'
        '</Piece></UnstructuredGrid>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("</VTKFile>\n" if closed else ""), encoding="utf-8")


def _optimization_run(solve, saves):
    """A shape-optimization run folder whose save i holds displacement ``saves[i]`` along y."""
    run = solve / "api_fem_forward-opt"
    for index, value in enumerate(saves):
        _write_surface_vtu(run / "vtu_surface" / f"surface_t{index:04d}.vtu",
                           [[0.0, value, 0.0]] * 3)
        for folder, name in (("glb", "surface"), ("vtu", "solution"), ("vtu_sdf", "solution")):
            target = run / folder / f"{name}_t{index:04d}.{'glb' if folder == 'glb' else 'vtu'}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"save {index}", encoding="utf-8")
    (run / "surface.pvd").write_text(
        '<?xml version="1.0"?><VTKFile type="Collection"><Collection>'
        + "".join(f'<DataSet timestep="{i}" file="vtu_surface/surface_t{i:04d}.vtu"/>'
                  for i in range(len(saves)))
        + "</Collection></VTKFile>", encoding="utf-8")
    (run / "log.txt").write_text("run_mode=shape_optimization", encoding="utf-8")
    return run


def test_a_failed_optimization_shows_its_last_iteration_that_solved(tmp_path):
    """The loop saves the state it failed in as its newest surface; the viewer must not follow.

    run_forward sets that save (and anything from it on) aside under failed_iteration/, takes
    the PVD entries out, drops the assets the live viewer built from it, rebuilds the
    displacement asset from the last iteration that solved, and records both iterations.
    """
    import numpy as np

    solve = tmp_path / "solve"
    nan = float("nan")
    # save every 2: saves 0, 1, 2 are iterations 0, 2, 4; save 3 is the failure save of 5
    run = _optimization_run(solve, [1.0, 2.0, 3.0, nan])
    view = workflow_tool._OptimizationView(solve, run, save_every=2)
    live = solve / "strain_von_mises_s03.glb"
    live.write_bytes(b"built from the failed surface")
    kept = solve / "strain_von_mises_s02.glb"
    kept.write_bytes(b"built from iteration 4")
    problem = SimpleNamespace(_optimize_iteration=5, _optimize_failed_iteration=5,
                              finalize=lambda: None)

    status = view.fail(problem, RuntimeError("Newton solver: NaN residual"))

    assert status["status"] == "failed" and status["failed_iteration"] == 5
    assert status["failed_in_solve"] is True
    assert status["shown_iteration"] == 4 and status["shown_surface"] == "surface_t0002.vtu"
    assert status["set_aside"] == ["surface_t0003.vtu"]
    assert json.loads((run / "optimization_status.json").read_text()) == status
    # set aside, not deleted, and out of every folder the viewer reads
    assert not (run / "vtu_surface" / "surface_t0003.vtu").exists()
    assert (run / "failed_iteration" / "vtu_surface" / "surface_t0003.vtu").is_file()
    assert (run / "failed_iteration" / "glb" / "surface_t0003.glb").is_file()
    assert (run / "failed_iteration" / "vtu" / "solution_t0003.vtu").is_file()
    assert "surface_t0003" not in (run / "surface.pvd").read_text()
    assert "surface_t0002" in (run / "surface.pvd").read_text()
    assert load_step_surfaces(solve)[-1].name == "surface_t0002.vtu"
    assert not live.exists() and kept.exists()
    # the displacement asset, with both sidecars, from iteration 4
    contour = solve / "displacement_magnitude.glb"
    meta = json.loads(contour.with_suffix(".glb.json").read_text())
    assert Path(meta["source"]).name == "surface_t0002.vtu"
    assert meta["maximum"] == pytest.approx(3.0)
    disp = np.frombuffer(contour.with_suffix(".glb.disp.bin").read_bytes(), dtype="<f4")
    assert disp.reshape(-1, 3)[:, 1].tolist() == [3.0, 3.0, 3.0]


def test_a_failed_optimization_skips_saves_that_hold_non_finite_values(tmp_path):
    """A save that did not raise but holds NaN is no better than the failed one."""
    solve = tmp_path / "solve"
    nan = float("nan")
    run = _optimization_run(solve, [1.0, nan, nan])
    view = workflow_tool._OptimizationView(solve, run, save_every=1)
    # failed outside the solve (the design update after iteration 2's solve):
    # no failure save, and iteration 2's own save is beyond the loop's reach
    problem = SimpleNamespace(_optimize_iteration=2, _optimize_failed_iteration=None,
                              finalize=lambda: None)
    status = view.fail(problem, ValueError("BC zip init failed"))
    assert status["failed_in_solve"] is False
    assert status["shown_iteration"] == 0
    assert status["set_aside"] == ["surface_t0001.vtu", "surface_t0002.vtu"]

    # nothing solved: nothing is shown, and no displacement asset of this run is left
    solve2 = tmp_path / "solve2"
    run2 = _optimization_run(solve2, [nan])
    view2 = workflow_tool._OptimizationView(solve2, run2, save_every=1)
    (solve2 / "displacement_magnitude.glb").write_bytes(b"refreshed from save 0")
    status = view2.fail(SimpleNamespace(_optimize_iteration=0, _optimize_failed_iteration=0,
                                        finalize=lambda: None), RuntimeError("NaN"))
    assert status["shown_iteration"] is None and status["displacement_asset"] is None
    assert not (solve2 / "displacement_magnitude.glb").exists()
    # a failure before the loop (the first shape's load ramp) has nothing to show
    assert view2.fail(SimpleNamespace(_optimize_iteration=None, finalize=lambda: None),
                      RuntimeError("ramp")) is None


def test_an_optimization_keeps_its_displacement_asset_current(tmp_path):
    """Like a forward solve's, the displacement asset exists while the loop runs, and follows it.

    Saves land asynchronously, so a surface still being written is not taken; the run's end
    exports its last save and records it as the one shown.
    """
    solve = tmp_path / "solve"
    run = _optimization_run(solve, [1.0])
    view = workflow_tool._OptimizationView(solve, run, save_every=1)
    _write_surface_vtu(run / "vtu_surface" / "surface_t0001.vtu", [[0.0, 5.0, 0.0]] * 3,
                       closed=False)
    view.refresh()
    contour = solve / "displacement_magnitude.glb"
    source = lambda: Path(json.loads(contour.with_suffix(".glb.json").read_text())["source"]).name
    assert source() == "surface_t0000.vtu"
    assert contour.with_suffix(".glb.disp.bin").is_file()
    _write_surface_vtu(run / "vtu_surface" / "surface_t0001.vtu", [[0.0, 5.0, 0.0]] * 3)
    status = view.complete()
    assert source() == "surface_t0001.vtu"
    assert status == {"status": "complete", "shown_iteration": 1,
                      "shown_surface": "surface_t0001.vtu"}

    # run_forward wires it into the loop, and the loop says where it failed
    text = Path(workflow_tool.__file__).read_text(encoding="utf-8")
    assert "problem.save = save_and_show" in text
    assert "view.fail(problem, exc, " in text
    loop = (Path(__file__).parents[2] / "warp_wrapper" / "core" / "abstract_field.py").read_text(
        encoding="utf-8")
    failure_branch = loop.split("self.step(tape=tape)\n            except Exception as e:", 1)[1]
    assert failure_branch.index("self._optimize_failed_iteration = it") < failure_branch.index(
        "self.save(time=it)")
    assert workflow_tool.OPTIMIZATION_STATUS == __import__(
        "src.agents_wrapper.fem_workflow.runtime_files", fromlist=["x"]).OPTIMIZATION_STATUS


def test_the_viewer_offers_displacement_during_an_optimization(tmp_path):
    """The displacement entry no longer waits for the solve-time asset, and its slider is sized
    from whatever asset the solve has; the asset counts as current only for its own surface."""
    from src.ui.telemetry import contour_shows, deformation_scales, optimization_shown

    solve = tmp_path / "solve"
    run = _optimization_run(solve, [1.0, 2.0])
    contour = solve / "displacement_magnitude.glb"
    workflow_tool._export_displacement_contour(run / "vtu_surface" / "surface_t0000.vtu", contour)
    newest = run / "vtu_surface" / "surface_t0001.vtu"
    # written after the newer surface, but coloured from the older one
    os.utime(newest, (time.time() - 50.0, time.time() - 50.0))
    assert not contour_shows(contour, newest)
    assert contour_shows(contour, run / "vtu_surface" / "surface_t0000.vtu")
    assert not contour_shows(contour, None)

    workflow = {"physical_extent": 2.0}
    assert deformation_scales(tmp_path, workflow)[-1] == pytest.approx(0.3 * 2.0 / 1.0)
    # no solve-time asset: any of the solve's assets says how far it moved
    for path in solve.glob("displacement_magnitude.glb*"):
        path.unlink()
    assert deformation_scales(tmp_path, workflow) == []
    workflow_tool._export_surface_field(newest, solve / "strain_von_mises_s01.glb")
    assert deformation_scales(tmp_path, workflow)[-1] == pytest.approx(0.3 * 2.0 / 2.0)

    assert optimization_shown(None) is None
    assert optimization_shown({"status": "complete", "shown_iteration": 3}) is None
    said = optimization_shown({"status": "failed", "failed_iteration": 7, "shown_iteration": 6})
    assert "iteration 7" in said and "iteration 6" in said
    assert "no result" in optimization_shown(
        {"status": "failed", "failed_iteration": 0, "shown_iteration": None})

    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    assert 'if displacement_contour.exists() or "displacement" in carried:' in source
    assert "contour_shows(displacement_contour, step_vtu)" in source
    assert 'displacement_contour.with_suffix(".glb.values.bin").exists()' in source


def test_the_viewer_spin_restarts_per_visualization_not_per_rerun():
    """The 3-D view spins until dragged, and again only on a switch of visualization.

    The key is what the frame compares against the host tab's record, so it
    must change with the artifact or the field component and with nothing a
    rerun or a live solve changes (an asset's path or mtime).
    """
    from src.ui.controller import visualization_key

    geometry = visualization_key("s1", "Geometry · interactive surface")
    assert geometry == visualization_key("s1", "Geometry · interactive surface")
    keys = {
        geometry,
        visualization_key("s1", "Boundary conditions · selected regions"),
        visualization_key("s1", "FEM results · displacement", "magnitude"),
        visualization_key("s1", "FEM results · displacement", "x"),
        visualization_key("s2", "Geometry · interactive surface"),
    }
    assert len(keys) == 5
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    # Spin is decided from the record, stopped by any take-hold of the view,
    # and the selected visualization is what keys it.
    assert "controls.autoRotate = !(sameView && lastSpin.stopped === true);" in source
    assert "controls.addEventListener('start'" in source
    assert "view_key=visualization_key(" in source
    # The viewer no longer over-exposes stored colours (ACES + 1.6-1.9x rig).
    assert "THREE.NeutralToneMapping" in source and "ACESFilmicToneMapping" not in source


def test_an_assigned_material_opens_on_youngs_modulus():
    """The three fields are written E first and density last, so the newest was density."""
    from src.ui.controller import preferred_visualization

    material = [("Material field · Young's modulus", 10.0),
                ("Material field · Poisson ratio", 11.0), ("Material field · Density", 12.0)]
    assert preferred_visualization([("Geometry · interactive surface", 1.0), *material]) == (
        "Material field · Young's modulus")
    # Something newer than the material is still what is opened.
    assert preferred_visualization([*material, ("FEM results · displacement", 20.0)]) == (
        "FEM results · displacement")
    # A field set without E opens on its newest, as before.
    assert preferred_visualization(material[1:]) == "Material field · Density"
    assert preferred_visualization([]) is None
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    assert 'label = f"Material field · {title}"' in source
    assert "newest_label = preferred_visualization(" in source


def test_material_fields_follow_the_approved_material_mode(tmp_path):
    """A uniform material lists no field; VoMP and per-part list their own; unapproved, the newest."""
    import time as time_module

    from src.ui.controller import material_field_root

    vomp = tmp_path / "materials" / "vomp"
    parts = tmp_path / "materials" / "parts" / "field"
    for folder in (vomp, parts):
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text("{}")
        (folder / "youngs_modulus.glb").write_bytes(b"glb")
        time_module.sleep(0.02)
    both = {"material_artifact": str(vomp), "parts_artifact": str(parts)}

    # Under review (the material card is up): the newer field, the parts one here.
    assert material_field_root({**both, "material": None}, tmp_path) == parts
    assert material_field_root({"material_artifact": str(vomp)}, tmp_path) == vomp
    # Approved: the mode decides, not which was built last.
    assert material_field_root(
        {**both, "material": {"mode": "vomp", "artifact": str(vomp)}}, tmp_path) == vomp
    assert material_field_root(
        {**both, "material": {"mode": "parts", "artifact": str(parts)}}, tmp_path) == parts
    # Uniform after a field was assigned: the field stays in state and on disk, unlisted --
    # and the disk fallback does not bring it back either.
    uniform = {"mode": "uniform", "youngs_modulus": 1e9, "poisson_ratio": 0.3, "artifact": None}
    assert material_field_root({**both, "material": uniform}, tmp_path) is None
    assert material_field_root({"material": uniform}, tmp_path) is None
    assert (parts / "youngs_modulus.glb").is_file() and (vomp / "manifest.json").is_file()
    # A rewind past the material card clears the state: the newest on disk is shown.
    assert material_field_root({}, tmp_path) == parts
    # An approved field whose files are gone lists nothing rather than failing.
    assert material_field_root(
        {"material": {"mode": "vomp", "artifact": str(tmp_path / "gone")}}, tmp_path) is None
    source = (Path(__file__).parents[2] / "ui" / "app.py").read_text(encoding="utf-8")
    assert "material_root = material_field_root(workflow, root)" in source


def test_uniform_after_parts_hides_the_material_fields_until_parts_come_back(
        monkeypatch, tmp_path):
    """Parts assigned, then uniform chosen: the viewer drops E/nu/rho; a rewind or re-approval
    of the parts brings back the field as built (the files never leave the session)."""
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCProposal, DirichletBC, NeumannBC, PartMaterials,
        SelectorSpec, ToolReceipt, WorkflowState,
    )
    from src.ui.controller import material_field_root

    segments = {"segments": [
        {"id": 0, "color": "#1f77b4", "area_fraction": 0.8, "centroid": [0, 0.6, 0]},
        {"id": 1, "color": "#aec7e8", "area_fraction": 0.2, "centroid": [0, 0.1, 0]},
    ], "bounds": [[-1, 0, -1], [1, 1, 1]]}

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            for key, value in arguments.items():
                if key.startswith("output"):
                    path = Path(value)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if name == "segment_surface" and path.suffix == ".json":
                        path.write_text(json_module.dumps(segments))
                    elif name == "build_part_field":
                        path.mkdir(exist_ok=True)
                        (path / "manifest.json").write_text("{}")
                        (path / "youngs_modulus.glb").write_bytes(b"glb")
                    else:
                        path.write_text("{}") if path.suffix == ".json" else path.write_bytes(b"stub")
                    if name == "stage_asset":
                        path.with_suffix(path.suffix + ".json").write_text(json_module.dumps(
                            {"source_name": path.name, "path": str(path), "sha256": "0" * 64}))
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    proposal = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, selector=SelectorSpec(
            type="lowest", params={"axis": "y", "fraction": 0.1}))],
        neumann=[NeumannBC(name="top", traction=(0.0, -1.0, 0.0), selector=SelectorSpec(
            type="highest", params={"axis": "y", "fraction": 0.1}))],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        _segment_legend = staticmethod(SpecialistCalls._segment_legend)

        def describe_image(self, purpose, image, legend=None):
            return "part 0 is the seat, part 1 the legs" if legend else "a chair"

        def synthesize_bcs(self, description, metadata, montage):
            return proposal, APIReceipt(purpose="bc_synthesis", provider="groq",
                                        model="fake", cache_key="k1")

        def assign_part_materials(self, montage, metadata, modulus_unit, request="", segments=""):
            return PartMaterials.model_validate({"parts": [
                {"id": 0, "youngs_modulus": 1.5e8, "poisson_ratio": 0.4, "density": 900.0},
                {"id": 1, "youngs_modulus": 2.0e11, "poisson_ratio": 0.29, "density": 7850.0},
            ]}), APIReceipt(purpose="part_materials", provider="groq", model="fake",
                            cache_key="k3")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "_surface_has_colour", lambda path: True)
    monkeypatch.setattr(graph_module, "partfield_status", lambda: SimpleNamespace(
        ready=True, model_dump=lambda: {"ready": True, "summary": "ready"}))
    source = tmp_path / "chair.glb"
    source.write_bytes(b"glb")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "uniform-after-parts"}}
    workflow = WorkflowState(session_id="uniform-after-parts", session_dir=str(tmp_path),
                             model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "an armchair", "asset_path": str(source),
                                 "asset_kind": "surface"}), config=config)
    graph.invoke(Command(resume={"action": "apply_scaling", "up_axis": "y",
                                 "physical_extent": 1.0,
                                 "unit_system": "SI (m-kg-s)"}), config=config)
    graph.invoke(Command(resume={"action": "preview_regions", "description": "fix feet, load top",
                                 "segmentation": {"method": "shape_diameter", "num_clusters": 5,
                                                  "num_rays": 25, "smoothing": 0.25,
                                                  "cone_angle_deg": 120.0}}), config=config)
    graph.invoke(Command(resume={"action": "approve_regions"}), config=config)
    graph.invoke(Command(resume={
        "action": "apply_values",
        "dirichlet_values": {"feet": {"components": [0, 1, 2], "value": 0.0}},
        "neumann_values": {"top": [0.0, -1.0, 0.0]}, "body_force": [0.0, 0.0, 0.0],
    }), config=config)

    def viewer_field():
        # What the viewer is handed: the checkpointed workflow, as the controller reads it.
        return material_field_root(graph.get_state(config).values["workflow"], tmp_path)

    def checkpoint():
        return graph.get_state(config).config["configurable"]["checkpoint_id"]

    card = graph.invoke(Command(resume={"action": "infer_parts"}),
                        config=config)["__interrupt__"][0].value
    field = Path(card["parts_artifact"])
    assert viewer_field() == field, "the field under review is listed"
    material_card = checkpoint()

    card = graph.invoke(Command(resume={"material_mode": "parts"}),
                        config=config)["__interrupt__"][0].value
    assert card["kind"] == "shape_opt_settings" and viewer_field() == field
    parts_settings = checkpoint()

    # Back to the material card; uniform is chosen.  The field stays in the state and on
    # disk (versions/ and a rewind need it) but is no longer the case's material.
    graph_module.reopen_card(graph, config, material_card)
    card = graph.invoke(Command(resume={
        "material_mode": "uniform", "youngs_modulus": 7e10, "poisson_ratio": 0.33,
        "density": 2700.0}), config=config)["__interrupt__"][0].value
    assert card["kind"] == "shape_opt_settings"
    state = graph.get_state(config).values["workflow"]
    assert state["material"]["mode"] == "uniform" and state["parts_artifact"] == str(field)
    assert (field / "youngs_modulus.glb").is_file()
    assert viewer_field() is None, "uniform: no material field is listed"

    # Rewound to the per-part branch's settings card: the field is listed again.
    graph_module.reopen_card(graph, config, parts_settings)
    assert viewer_field() == field

    # Or the material card re-decided as per-part: listed again, the field as rebuilt.
    graph_module.reopen_card(graph, config, material_card)
    graph.invoke(Command(resume={"material_mode": "uniform", "youngs_modulus": 7e10,
                                 "poisson_ratio": 0.33}), config=config)
    assert viewer_field() is None
    graph_module.reopen_card(graph, config, material_card)
    graph.invoke(Command(resume={"action": "infer_parts"}), config=config)
    graph.invoke(Command(resume={"material_mode": "parts"}), config=config)
    assert graph.get_state(config).values["workflow"]["material"]["mode"] == "parts"
    assert viewer_field() == field


def test_a_rewound_part_field_is_rebuilt_for_the_values_assigned_again(monkeypatch, tmp_path):
    """Rewind to an earlier part field, assign the later values again: the viewer's GLBs (and
    the solve's grid) must be the later field's, not the earlier one's with a new manifest.

    The receipt named only manifest.json, so the cache hit put the later manifest back and
    left the earlier field's GLBs and grid in place.  Every file of the field is named now.
    """
    import subprocess

    from src.agents_wrapper.fem_workflow.monitor import _digest

    session = tmp_path / "session"
    parts = session / "materials" / "parts"
    parts.mkdir(parents=True)
    surface, segments = session / "physical.glb", session / "segments.npz"
    surface.write_bytes(b"surface")
    segments.write_bytes(b"segments")
    materials = parts / "materials.json"
    field = parts / "field"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        values = materials.read_text(encoding="utf-8")
        field.mkdir(exist_ok=True)
        (field / "manifest.json").write_text(f'{{"from": "{values}"}}', encoding="utf-8")
        for name in ("youngs_modulus.glb", "density.glb", "material_grid.npy"):
            (field / name).write_text(f"{name} of {values}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def keep_versions():
        # what the graph's checkpoint record does: every branch file, by content
        (session / "versions").mkdir(exist_ok=True)
        snapshot = {}
        for path in field.iterdir():
            digest = _digest(path)
            shutil.copy2(path, session / "versions" / f"{digest}{path.suffix}")
            snapshot[path.name] = digest
        return snapshot

    def rewind_to(snapshot):
        for name, digest in snapshot.items():
            suffix = Path(name).suffix
            shutil.copy2(session / "versions" / f"{digest}{suffix}", field / name)

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    monitor = ToolMonitor(session)
    arguments = {"surface": str(surface), "segments_npz": str(segments),
                 "materials_json": str(materials), "unit_system": "SI (m-kg-s)",
                 "output": str(field)}
    materials.write_text("A", encoding="utf-8")
    monitor.execute("build_part_field", arguments)
    first = keep_versions()
    materials.write_text("B", encoding="utf-8")
    receipt = monitor.execute("build_part_field", arguments)
    assert str((field / "youngs_modulus.glb").resolve()) in receipt.artifacts
    keep_versions()
    rewind_to(first)
    assert (field / "youngs_modulus.glb").read_text() == "youngs_modulus.glb of A"

    materials.write_text("B", encoding="utf-8")
    monitor.execute("build_part_field", arguments)
    assert len(calls) == 2, "served from the receipt, every file put back from versions/"
    for name in ("youngs_modulus.glb", "density.glb", "material_grid.npy"):
        assert (field / name).read_text() == f"{name} of B"
    assert "B" in (field / "manifest.json").read_text()

    # without B's GLB among the versions the field is built again
    rewind_to(first)
    for path in (session / "versions").iterdir():
        if path.read_text(errors="replace") == "youngs_modulus.glb of B":
            path.unlink()
    monitor.execute("build_part_field", arguments)
    assert len(calls) == 3
    assert (field / "youngs_modulus.glb").read_text() == "youngs_modulus.glb of B"


def test_materials_by_part_follow_a_change_of_segmentation_to_cad_faces(monkeypatch, tmp_path):
    """Re-segmented with its CAD faces, the same body is valued on those faces, not the old parts.

    525c794ba2ae (2026-10-20): the regions were switched to CAD faces, and the per-part
    material came back unchanged.  The material step took the regions' segments for
    PartField only; with CAD faces it went on reading its own PartField parts -- the same
    labels, montage and (cached) answer as before the change.  And a table valued on one
    segmentation stayed "current" after the labels at the same path changed, since only
    the surface was checked.
    """
    pytest.importorskip("langgraph")
    import json as json_module

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import (
        APIReceipt, BCProposal, DirichletBC, NeumannBC, PartMaterials,
        SelectorSpec, ToolReceipt, WorkflowState,
    )

    calls, asked = [], []
    labelled = {"partfield": 2, "cad_faces": 3}

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, name, arguments):
            calls.append((name, dict(arguments)))
            if name == "generate_text2cad_text_surface":
                _cad_box_surface(tmp_path)
            method = arguments.get("method")
            for key, value in arguments.items():
                if not key.startswith("output"):
                    continue
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "segment_surface" and path.suffix == ".json":
                    path.write_text(json_module.dumps({"method": method, "segments": [
                        {"id": i, "color": "#1f77b4", "area_fraction": 1 / labelled[method],
                         "centroid": [0, i, 0]} for i in range(labelled[method])],
                        "bounds": [[-1, 0, -1], [1, 1, 1]]}))
                elif name == "segment_surface" and path.suffix == ".npz":
                    path.write_bytes(f"labels by {method}".encode())
                elif name == "build_part_field":
                    path.mkdir(exist_ok=True)
                    (path / "manifest.json").write_text("{}")
                else:
                    path.write_text("{}") if path.suffix == ".json" else path.write_bytes(b"stub")
            return ToolReceipt(tool=name, arguments=arguments, returncode=0,
                               stdout_path="out", stderr_path="err")

    proposal = BCProposal(
        dirichlet=[DirichletBC(name="feet", value=0.0, selector=SelectorSpec(
            type="lowest", params={"axis": "z", "fraction": 0.1}))],
        neumann=[NeumannBC(name="top", traction=(0.0, 0.0, -1.0), selector=SelectorSpec(
            type="highest", params={"axis": "z", "fraction": 0.1}))],
    )

    class FakeSpecialist:
        def __init__(self, *args, **kwargs):
            pass

        _segment_legend = staticmethod(SpecialistCalls._segment_legend)

        def describe_image(self, purpose, image, legend=None):
            return "the segments" if legend else "a block"

        def synthesize_bcs(self, description, metadata, montage):
            return proposal, APIReceipt(purpose="bc_synthesis", provider="groq",
                                        model="fake", cache_key="k1")

        def assign_part_materials(self, montage, metadata, modulus_unit, request="", segments=""):
            ids = [int(segment["id"]) for segment in metadata["segments"]]
            asked.append({"method": metadata.get("method"), "ids": ids})
            return PartMaterials.model_validate({"parts": [
                {"id": i, "youngs_modulus": 2.0e11, "poisson_ratio": 0.29, "density": 7850.0}
                for i in ids]}), APIReceipt(purpose="part_materials", provider="groq",
                                            model="fake", cache_key="k3")

    monkeypatch.setattr(graph_module, "SpecialistCalls", FakeSpecialist)
    monkeypatch.setattr(graph_module, "ToolMonitor", FakeMonitor)
    monkeypatch.setattr(graph_module, "text2cad_status", _ready_text2cad)
    monkeypatch.setattr(graph_module, "hunyuan3d_status", lambda: SimpleNamespace(
        ready=False, model_dump=lambda: {"ready": False}))
    monkeypatch.setattr(graph_module, "_surface_has_colour", lambda path: True)
    monkeypatch.setattr(graph_module, "partfield_status", lambda: SimpleNamespace(
        ready=True, model_dump=lambda: {"ready": True, "summary": "ready"}))
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "resegment"}}
    workflow = WorkflowState(session_id="resegment", session_dir=str(tmp_path), model="fake")
    graph.invoke({"workflow": workflow.model_dump(mode="json")}, config=config)
    graph.invoke(Command(resume={"description": "a block", "generator": "text2cad"}),
                 config=config)
    card = graph.invoke(Command(resume={
        "up_axis": "z", "physical_extent": 0.2, "unit_system": "SI (m-kg-s)",
    }), config=config)["__interrupt__"][0].value
    assert card["kind"] == "bc_input"
    regions_card = graph.get_state(config).config["configurable"]["checkpoint_id"]

    def assign_on(method):
        graph.invoke(Command(resume={
            "action": "preview_regions", "description": "fix the feet, load the top",
            "segmentation": {**card["segmentation"], "method": method}}), config=config)
        graph.invoke(Command(resume={"action": "approve_regions"}), config=config)
        graph.invoke(Command(resume={
            "action": "apply_values",
            "dirichlet_values": {"feet": {"components": [0, 1, 2], "value": 0.0}},
            "neumann_values": {"top": [0.0, 0.0, -1.0]}, "body_force": [0.0, 0.0, 0.0],
        }), config=config)
        return graph.invoke(Command(resume={"action": "infer_parts"}),
                            config=config)["__interrupt__"][0].value

    bc_npz = str(tmp_path / "boundary_conditions" / "segments.npz")
    first = assign_on("partfield")
    assert asked[-1] == {"method": "partfield", "ids": [0, 1]}
    assert [part["id"] for part in first["material_parts"]] == [0, 1]

    # Back to the regions card; the same body is segmented by its CAD faces instead.
    graph_module.reopen_card(graph, config, regions_card)
    second = assign_on("cad_faces")
    assert asked[-1] == {"method": "cad_faces", "ids": [0, 1, 2]}, "valued on the CAD faces"
    built = [arguments for name, arguments in calls if name == "build_part_field"]
    assert built[-1]["segments_npz"] == bc_npz
    assert not any(name == "segment_surface" and arguments["output_npz"] != bc_npz
                   for name, arguments in calls), "no PartField parts of the material step's own"
    assert [part["id"] for part in second["material_parts"]] == [0, 1, 2]
    state = WorkflowState.model_validate(graph.get_state(config).values["workflow"])
    assert graph_module._current_parts(state)

    # The labels change under the same path with the parts table still in state: it is
    # neither shown nor approved for the new labels.
    monkeypatch.setattr(graph_module, "_restore_branch_files", lambda state: None)
    Path(bc_npz).write_bytes(b"other faces")
    assert not graph_module._current_parts(state)
    graph.invoke(None, config=graph.update_state({"configurable": {
        **graph.get_state(config).config["configurable"], "checkpoint_ns": ""}}, None))
    stale = graph.get_state(config).tasks[0].interrupts[0].value
    assert stale["kind"] == "material_input"
    assert stale["material_parts"] is None and stale["parts_artifact"] is None
    with pytest.raises(Exception, match="on this surface"):
        graph.invoke(Command(resume={"material_mode": "parts"}), config=config)


def test_a_part_table_from_before_its_labels_were_recorded_keeps_the_old_rule(tmp_path):
    """Sessions valued before parts_segments_digest existed: current exactly when the old
    rule (regions' segments for PartField only) still reads the same labels."""
    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState

    surface = tmp_path / "geometry" / "physical.glb"
    surface.parent.mkdir(parents=True)
    surface.write_bytes(b"surface")
    field = tmp_path / "materials" / "parts" / "field"
    field.mkdir(parents=True)
    (field / "manifest.json").write_text("{}")
    sha = graph_module._surface_digest(str(surface))
    for folder, method in (("boundary_conditions", "cad_faces"), ("materials/parts", "partfield")):
        root = tmp_path / folder
        root.mkdir(parents=True, exist_ok=True)
        (root / "segments.npz").write_bytes(method.encode())
        (root / "segments.json").write_text(json.dumps({"method": method, "segments": []}))
        (root / "segments.source.json").write_text(json.dumps({"sha256": sha}))
    state = WorkflowState(
        session_id="legacy", session_dir=str(tmp_path), model="fake",
        surface_path=str(surface), segmentation_method="shape_diameter",
        segmentation_metadata_path=str(tmp_path / "boundary_conditions" / "segments.json"),
        material_parts=[{"id": 0, "youngs_modulus": 1e9, "poisson_ratio": 0.3, "density": 1e3}],
        parts_artifact=str(field), parts_surface_digest=sha)
    # the regions are CAD faces now: the old table was valued on the material step's own parts
    assert not graph_module._current_parts(state)
    (tmp_path / "boundary_conditions" / "segments.json").write_text(
        json.dumps({"method": "shape_diameter", "segments": []}))
    assert graph_module._current_parts(state)


def test_pre_rename_environment_names_resolve_to_mechfoundry(monkeypatch, tmp_path):
    """Sessions stored before the MechFoundry rename name ``implicit-*`` environments.

    They keep that environment while it is installed and move to its ``mechfoundry-*`` name
    once it is gone: in the lookup, the worker keys, a resumed session and a loaded state.
    """
    from src.agents_wrapper.fem_workflow import WorkflowState, runtime_files as rf
    for name in ("CONDA_EXE", "CONDA_ROOT", "CONDA_PREFIX_1"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rf.sys, "executable", str(tmp_path / "envs" / "other" / "bin" / "python"))

    def install(name):
        bin_dir = tmp_path / "envs" / name / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").write_text("")
        return bin_dir / "python"

    new = install("mechfoundry-warp117")
    old = install("implicit-warp117")
    # Both installed: the session keeps the environment it was made in.
    assert rf.current_environment("implicit-warp117") == "implicit-warp117"
    assert rf.direct_interpreter("implicit-warp117") == old
    shutil.rmtree(tmp_path / "envs" / "implicit-warp117")
    # The old one uninstalled: the renamed one, everywhere a name becomes an interpreter.
    assert rf.current_environment("implicit-warp117") == "mechfoundry-warp117"
    assert rf.direct_interpreter("implicit-warp117") == new
    assert rf.worker_key(tmp_path, "implicit-warp117") == rf.worker_key(tmp_path, "mechfoundry-warp117")
    assert ToolMonitor(tmp_path / "s", "implicit-warp117").compute_env == "mechfoundry-warp117"
    assert WorkflowState(session_id="old", session_dir=str(tmp_path),
                         compute_env="implicit-warp117").compute_env == "mechfoundry-warp117"
    session = tmp_path / "session"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"compute_env": "implicit-warp117"}))
    assert run_workflow.read_compute_env(session) == "mechfoundry-warp117"
    # Neither found: the new name, which is what ``conda run -n`` needs after the rename.
    assert rf.current_environment("implicit-agents") == "mechfoundry-agents"
    # PartField's unprefixed ``partfield`` follows the same rule: kept while installed, renamed
    # once gone (here neither is installed, so the new name).
    assert rf.current_environment("partfield") == "mechfoundry-partfield"
    old_partfield = install("partfield")
    assert rf.current_environment("partfield") == "partfield"
    assert rf.direct_interpreter("partfield") == old_partfield
    shutil.rmtree(tmp_path / "envs" / "partfield")
    new_partfield = install("mechfoundry-partfield")
    assert rf.direct_interpreter("partfield") == new_partfield
    # Names outside the table and the new names themselves are left alone.
    for name in ("mechfoundry-warp117", "mechfoundry-partfield", "custom-env"):
        assert rf.current_environment(name) == name
    assert set(rf.RENAMED_ENVIRONMENTS.values()) == {
        f"mechfoundry-{suffix}" for suffix in
        ("warp117", "agents", "text2cad", "cadcoder", "hunyuan3d", "vomp", "partfield")}


def _solve_case(session):
    """A session folder with a case to solve, as the monitor is handed one."""
    session.mkdir()
    case = session / "case.json"
    case.write_text(json.dumps({"output": str(session / "solve")}), encoding="utf-8")
    return case


def test_a_second_solve_is_refused_before_it_touches_anything(monkeypatch, tmp_path):
    """Eight solves of one case began in four minutes on 2026-11-06 and ran each other out of
    memory.  The refusal comes first, ahead of the model unload that makes room for a solve."""
    import subprocess

    from src.agents_wrapper.fem_workflow.runtime_files import (
        SolveAlreadyRunning, solve_claim, solve_process, solve_running,
    )

    case = _solve_case(tmp_path / "session")
    session = case.parent
    launches = []

    def fake_run(command, **kwargs):
        # Held for the whole subprocess, as it is across the worker's request.
        assert solve_running(session)
        launches.append(command)
        (session / "solve").mkdir(exist_ok=True)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    monitor = ToolMonitor(session)
    monkeypatch.setattr(monitor, "release_models_for_solve", lambda: launches.append("unload"))
    # Once a solve has returned the session is free again, and the next one runs.
    monitor.execute("run_forward", {"case": str(case)})
    assert len(launches) == 2 and not solve_running(session)
    launches.clear()
    receipts = sorted(path.name for path in monitor.receipt_dir.iterdir())

    # Another front end's solve, between threads of one process as much as between processes.
    with solve_claim(session):
        with pytest.raises(SolveAlreadyRunning, match="already running"):
            monitor.execute("run_forward", {"case": str(case)})
    # A solve that outlived the monitor that started it, Streamlit restarted beneath it.
    with solve_process(session):
        with pytest.raises(SolveAlreadyRunning, match="already running"):
            monitor.execute("run_forward", {"case": str(case)})
    assert launches == []
    assert sorted(path.name for path in monitor.receipt_dir.iterdir()) == receipts


def test_the_solve_holds_the_lock_on_the_worker_route_too(monkeypatch, tmp_path):
    from src.agents_wrapper.fem_workflow.runtime_files import solve_running

    case = _solve_case(tmp_path / "session")
    session = case.parent
    seen = []
    monitor = ToolMonitor(session)

    def served(command_tail, timeout):
        seen.append(solve_running(session))
        (session / "solve").mkdir(exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(monitor, "release_models_for_solve", lambda: None)
    monkeypatch.setattr(monitor, "_run_solver_worker", served)
    monitor.execute("run_forward", {"case": str(case)})
    assert seen == [True] and not solve_running(session)


def test_the_solve_marks_itself_running_wherever_it_lands(tmp_path):
    """The lock that outlives the monitor is taken in the solving process, keyed on the session
    that holds the case's output, so the warm worker and the backend run are both covered."""
    from src.agents_wrapper.fem_workflow.runtime_files import (
        SolveAlreadyRunning, solve_process, solve_running,
    )

    case = _solve_case(tmp_path / "session")
    session = case.parent
    seen = []

    @workflow_tool._holding_solve_lock
    def solve(args):
        seen.append(solve_running(session))
        return ["done"]

    assert solve(SimpleNamespace(case=case)) == ["done"]
    assert seen == [True] and not solve_running(session)
    assert hasattr(workflow_tool.run_forward, "__wrapped__")
    # The last line of defence: a second solve that got past every claim still does not run.
    with solve_process(session):
        with pytest.raises(SolveAlreadyRunning):
            solve(SimpleNamespace(case=case))
    assert len(seen) == 1


@pytest.mark.parametrize("sig", ["SIGKILL", "SIGTERM"])
def test_a_killed_solve_leaves_no_lock_behind(tmp_path, sig):
    import subprocess

    from src.agents_wrapper.fem_workflow import runtime_files
    from src.agents_wrapper.fem_workflow.runtime_files import solve_claim, solve_running

    session = tmp_path / "session"
    session.mkdir()
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time\n"
         f"sys.path.insert(0, {str(Path(runtime_files.__file__).parent)!r})\n"
         "import runtime_files\n"
         f"with runtime_files.solve_process({str(session)!r}):\n"
         "    print('held', flush=True)\n"
         "    time.sleep(60)\n"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert solve_running(session)
    finally:
        holder.send_signal(getattr(signal, sig))
        holder.wait(10)
        holder.stdout.close()
    assert not solve_running(session)
    with solve_claim(session):
        assert solve_running(session)
    assert not solve_running(session)


def test_a_session_that_never_solved_reads_as_idle_and_gains_no_files(tmp_path):
    from src.agents_wrapper.fem_workflow.runtime_files import solve_running

    assert not solve_running(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_refused_solve_leaves_the_session_and_its_results_alone(monkeypatch, tmp_path):
    """Refused at the solve itself: the failure path would drop the running solve's assets."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.assets import SOLVE_FAILURE
    from src.agents_wrapper.fem_workflow.models import WorkflowState
    from src.agents_wrapper.fem_workflow.runtime_files import (
        SolveAlreadyRunning, solve_claim,
    )

    case = _solve_case(tmp_path / "session")
    session = case.parent
    asset = session / "solve" / "displacement_magnitude.glb"
    asset.parent.mkdir()
    asset.write_bytes(b"the running solve's")
    state = WorkflowState(session_id="s", session_dir=str(session), model="fake",
                          case_path=str(case)).model_dump(mode="json")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s"}}
    graph.update_state(config, {"workflow": state}, as_node="build_case")
    assert graph.get_state(config).next == ("solve",)

    with solve_claim(session):
        with pytest.raises(SolveAlreadyRunning):
            graph.invoke(None, config=config)
    assert asset.read_bytes() == b"the running solve's"
    assert not (session / "solve" / SOLVE_FAILURE).exists()
    assert graph.get_state(config).values["workflow"] == state


def test_a_settings_answer_during_a_solve_rewrites_nothing(tmp_path):
    """The other front ends' guard, for the CLI and the operator: the answer stops at the card."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.models import WorkflowState
    from src.agents_wrapper.fem_workflow.runtime_files import (
        SolveAlreadyRunning, solve_claim,
    )

    session = tmp_path / "session"
    session.mkdir()
    state = WorkflowState(session_id="s", session_dir=str(session), model="fake")
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s"}}
    graph.update_state(config, {"workflow": state.model_dump(mode="json"),
                                "pending": {}}, as_node="solve")
    assert graph.get_state(config).next == ("shape_opt_input",)
    graph.invoke(None, config=config)
    with solve_claim(session):
        with pytest.raises(SolveAlreadyRunning):
            graph.invoke(graph_module.resume_command(graph, config, {"load_steps": 3}),
                         config=config)
    assert not (session / "case" / "shape_opt.json").exists()


def _solve_graph(launches):
    """A card, then a solve that counts its launches, then the card after it."""
    from typing import TypedDict

    from langgraph.graph import END, StateGraph
    from langgraph.types import interrupt

    class State(TypedDict, total=False):
        workflow: dict

    def settings(state):
        answer = interrupt({"kind": "settings"})
        return {"workflow": {**state.get("workflow", {}), "settings": answer}}

    def solve(state):
        launches.append(state["workflow"].get("settings"))
        return {"workflow": {**state["workflow"], "stage": "complete"}}

    def build():
        graph = StateGraph(State)
        graph.add_node("settings", settings)
        graph.add_node("solve", solve)
        graph.set_entry_point("settings")
        graph.add_edge("settings", "solve")
        graph.add_edge("solve", END)
        return graph

    return build


def _stopped_ahead_of_the_solve(controller, session_id, build):
    """Stop the session where a server dying mid-solve leaves it."""
    from src.agents_wrapper.fem_workflow.graph import open_session_graph, session_thread

    session_dir = controller.session_dir(session_id)
    with open_session_graph(session_dir, build) as graph:
        thread = session_thread(session_id, session_dir)
        graph.update_state(thread, {"workflow": {"settings": "first"}}, as_node="settings")
        assert graph.get_state(thread).next == ("solve",)


def test_opening_a_session_never_runs_its_pending_solve(monkeypatch, tmp_path):
    """Each opening of a session waiting at ``solve`` used to start the solve."""
    pytest.importorskip("langgraph")
    launches = []
    build = _solve_graph(launches)
    monkeypatch.setattr("src.ui.controller.build_graph", build)
    controller = WorkflowController(tmp_path)
    (tmp_path / "s1").mkdir()

    parked = controller._invoke("s1", {"workflow": {}})
    # A card waiting is simply shown, as many times as it is opened.
    for _ in range(2):
        opened = controller.open("s1")
        assert opened.interrupt == {"kind": "settings"}
        assert opened.checkpoint_id == parked.checkpoint_id

    # No card waiting: the newest one comes back, and it is what runs the solve.
    _stopped_ahead_of_the_solve(controller, "s1", build)
    for _ in range(3):
        assert controller.open("s1").interrupt == {"kind": "settings"}
    assert launches == []
    assert controller.resume("s1", "second").complete
    assert launches == ["second"]

    # Finished: the session is shown as it ended, with nothing run again.
    finished = controller.open("s1")
    assert finished.complete and finished.interrupt is None and launches == ["second"]

    with pytest.raises(FileNotFoundError):
        controller.open("absent")


def test_opening_a_session_mid_solve_only_reads_it(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from src.agents_wrapper.fem_workflow.graph import open_session_graph, session_thread
    from src.agents_wrapper.fem_workflow.runtime_files import (
        SolveAlreadyRunning, solve_claim,
    )

    launches = []
    build = _solve_graph(launches)
    monkeypatch.setattr("src.ui.controller.build_graph", build)
    controller = WorkflowController(tmp_path)
    session = tmp_path / "s1"
    session.mkdir()
    controller._invoke("s1", {"workflow": {}})
    _stopped_ahead_of_the_solve(controller, "s1", build)

    def head():
        with open_session_graph(session, build) as graph:
            return graph.get_state(session_thread("s1", session)).config[
                "configurable"]["checkpoint_id"]

    before = head()
    # A step of this tab's own, one the lock does not cover.
    assert controller.open("s1", step_running=True).interrupt is None
    with solve_claim(session):
        assert controller.solve_running("s1")
        opened = controller.open("s1")
        assert opened.interrupt is None and opened.workflow == {"settings": "first"}
        # Stopped ahead of the rewind an answer can ask for, as well as the run.
        with pytest.raises(SolveAlreadyRunning, match="already running"):
            controller.resume("s1", "second", opened.checkpoint_id)
    assert head() == before and launches == []
    assert not controller.solve_running("s1")


def _running_solve(session, *, ignore_term=False, child=False):
    """A real process holding the session's run lock, on its way through a long solve."""
    import subprocess

    from src.agents_wrapper.fem_workflow import runtime_files

    session.mkdir(exist_ok=True)
    body = (
        "import signal, subprocess, sys, time\n"
        f"sys.path.insert(0, {str(Path(runtime_files.__file__).parent)!r})\n"
        "import runtime_files\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"with runtime_files.solve_process({str(session)!r}):\n"
        + ("    below = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
           "    print(below.pid, flush=True)\n" if child else "")
        + "    print('held', flush=True)\n"
        "    time.sleep(60)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", body], stdout=subprocess.PIPE, text=True,
                               start_new_session=True)
    first = process.stdout.readline().strip()
    if child:
        assert process.stdout.readline().strip() == "held"
        return process, int(first)
    assert first == "held"
    return process, None


def test_stop_ends_the_solve_and_frees_the_session(tmp_path):
    from src.agents_wrapper.fem_workflow.runtime_files import (
        SOLVE_RUN_LOCK, solve_claim, solve_running, solve_stop_requested, stop_solve,
    )

    session = tmp_path / "session"
    process, below = _running_solve(session, child=True)
    try:
        assert solve_running(session)
        assert stop_solve(session)
        assert process.wait(10) == -signal.SIGTERM
        assert not solve_running(session)
        assert solve_stop_requested(session)
        # What is below it goes too, and it reads as a zombie or as nothing.
        from src.agents_wrapper.fem_workflow.runtime_files import process_state
        deadline = time.monotonic() + 5
        while process_state(below) not in (None, "Z", "X") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert process_state(below) in (None, "Z", "X")
    finally:
        process.kill()
        process.stdout.close()
    # Free at once, and the next solve starts clean of the request.
    with solve_claim(session):
        assert not solve_stop_requested(session)
    assert (session / SOLVE_RUN_LOCK).read_text() != ""       # a killed holder cannot clear it


def test_a_solve_that_ignores_the_stop_is_killed(tmp_path):
    from src.agents_wrapper.fem_workflow.runtime_files import solve_running, stop_solve

    session = tmp_path / "session"
    process, _ = _running_solve(session, ignore_term=True)
    try:
        assert stop_solve(session, grace=0.2)
        assert process.wait(10) == -signal.SIGKILL
        assert not solve_running(session)
    finally:
        process.kill()
        process.stdout.close()


def test_a_stop_never_signals_a_reused_pid(tmp_path):
    """The record names a pid; the process there now is someone else's."""
    import subprocess

    from src.agents_wrapper.fem_workflow.runtime_files import (
        SOLVE_RUN_LOCK, _process_identity, solve_process, solve_stop_requested, stop_solve,
    )

    session = tmp_path / "session"
    session.mkdir()
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # Stale and unlocked: a holder long dead, nothing to do.
        identity = _process_identity(bystander.pid)
        (session / SOLVE_RUN_LOCK).write_text(json.dumps(identity))
        assert not stop_solve(session)

        with solve_process(session):
            # Locked, with a record of the same pid under another command, then another start.
            for changed in ({"command": b"something else".hex()}, {"started": "1"}):
                (session / SOLVE_RUN_LOCK).write_text(json.dumps({**identity, **changed}))
                assert not stop_solve(session)
            # A holder from before records existed: nothing to verify, so nothing is signalled.
            (session / SOLVE_RUN_LOCK).write_text("")
            assert not stop_solve(session)
        assert bystander.poll() is None
        assert not solve_stop_requested(session)
    finally:
        bystander.kill()
        bystander.wait()


def test_the_stop_button_is_live_only_during_a_solve(monkeypatch, tmp_path):
    from src.agents_wrapper.fem_workflow.runtime_files import solve_claim, solve_process

    controller = WorkflowController(tmp_path)
    session = tmp_path / "s1"
    session.mkdir()
    assert not controller.solve_running("s1")
    with solve_claim(session):
        assert controller.solve_running("s1")
    assert not controller.solve_running("s1")
    with solve_process(session):
        assert controller.solve_running("s1")
    assert not controller.solve_running("s1")
    assert not controller.stop_solve("s1")
    process, _ = _running_solve(session)
    try:
        assert controller.solve_running("s1") and not controller.solve_stopped("s1")
        assert controller.stop_solve("s1")
        process.wait(10)
        assert not controller.solve_running("s1") and controller.solve_stopped("s1")
    finally:
        process.kill()
        process.stdout.close()


@pytest.mark.parametrize("returncode", [-signal.SIGTERM, -signal.SIGKILL])
def test_a_requested_stop_keeps_the_results_and_returns_the_card(
        monkeypatch, tmp_path, returncode):
    pytest.importorskip("langgraph")
    import subprocess

    from langgraph.checkpoint.memory import MemorySaver

    from src.agents_wrapper.fem_workflow import graph as graph_module
    from src.agents_wrapper.fem_workflow.assets import SOLVE_FAILURE
    from src.agents_wrapper.fem_workflow.models import WorkflowState
    from src.agents_wrapper.fem_workflow.runtime_files import (
        SOLVE_STOP_REQUEST, solve_running,
    )

    case = _solve_case(tmp_path / "session")
    session = case.parent
    solve_root = session / "solve"
    solve_root.mkdir()
    asset = solve_root / "displacement_magnitude.glb"
    asset.write_bytes(b"iteration 40")
    (solve_root / SOLVE_FAILURE).write_text("{}")   # an earlier failure no longer stands

    def fake_run(command, **kwargs):
        (session / SOLVE_STOP_REQUEST).write_text("0")
        return subprocess.CompletedProcess(command, returncode, "", "RuntimeError: progress bar")

    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    monkeypatch.setattr(ToolMonitor, "release_models_for_solve", lambda self: None)
    monkeypatch.setattr(ToolMonitor, "_run_solver_worker", lambda self, tail, timeout: None)
    state = WorkflowState(session_id="s", session_dir=str(session), model="fake",
                          case_path=str(case))
    graph = graph_module.build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s"}}
    graph.update_state(config, {"workflow": state.model_dump(mode="json")}, as_node="build_case")

    card = graph.invoke(None, config=config)["__interrupt__"][0].value
    assert card["kind"] == "shape_opt_settings"
    assert card["message"].startswith("Solve stopped")
    assert card["solve_stopped"] and not card["solve_error"] and not card["solve_complete"]
    assert asset.read_bytes() == b"iteration 40"
    assert not (solve_root / SOLVE_FAILURE).exists()
    assert graph.get_state(config).values["workflow"]["errors"] == []
    assert not solve_running(session)


def test_a_stopped_worker_is_not_run_again_here(monkeypatch, tmp_path):
    """The monitor runs a request itself when a worker dies under it, which would undo a stop."""
    import subprocess

    from src.agents_wrapper.fem_workflow.monitor import ToolFailed
    from src.agents_wrapper.fem_workflow.runtime_files import SOLVE_STOP_REQUEST

    case = _solve_case(tmp_path / "session")
    session = case.parent
    launches = []
    monitor = ToolMonitor(session)
    monkeypatch.setattr(monitor, "release_models_for_solve", lambda: None)

    def stopped_under_it(tail, timeout):
        (session / SOLVE_STOP_REQUEST).write_text("0")
        return None

    def fake_run(command, **kwargs):
        launches.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(monitor, "_run_solver_worker", stopped_under_it)
    monkeypatch.setattr("src.agents_wrapper.fem_workflow.monitor.subprocess.run", fake_run)
    with pytest.raises(ToolFailed):
        monitor.execute("run_forward", {"case": str(case)})
    assert launches == []


def test_a_stop_waits_for_a_claimed_solve_to_start(tmp_path):
    # Stop FEA is on offer from the click on Run FEA, seconds before the solve
    # reaches its lock, and a click then must not come back "nothing to stop".
    import subprocess

    from src.agents_wrapper.fem_workflow.runtime_files import solve_claim, solve_running, stop_solve

    session = tmp_path / "s1"
    session.mkdir()
    script = (
        "import sys, time\n"
        "from src.agents_wrapper.fem_workflow.runtime_files import solve_process\n"
        "time.sleep(0.5)\n"
        "with solve_process(sys.argv[1]):\n"
        "    time.sleep(60)\n"
    )
    with solve_claim(session):
        process = subprocess.Popen([sys.executable, "-c", script, str(session)], cwd=Path(__file__).resolve().parents[3])
        try:
            assert not stop_solve(session)                  # no wait: nothing to reach yet
            assert stop_solve(session, start_wait=20.0)
            assert process.wait(timeout=10) < 0
        finally:
            process.kill()
    assert not solve_running(session)


def test_the_solver_panel_reports_a_stop_after_the_log_says_finished():
    # 2026-11-06: stopped while saving its surfaces, six seconds after the log's
    # "Simulation finalized", a forward solve went on reading "complete".
    from src.ui.telemetry import solver_state

    assert solver_state(failed=False, finished=True, running=False, stopped=True) == "stopped"
    assert solver_state(failed=False, finished=False, running=False, stopped=True) == "stopped"
    # The request is written before the signal: not stopped until it has ended.
    assert solver_state(failed=False, finished=False, running=True, stopped=True) == "running"
    assert solver_state(failed=False, finished=True, running=False, stopped=False) == "complete"
    assert solver_state(failed=True, finished=False, running=False, stopped=False) == "failed"
