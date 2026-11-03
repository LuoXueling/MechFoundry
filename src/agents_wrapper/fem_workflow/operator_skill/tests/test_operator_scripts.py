"""Unit tests for the operator skill's scripts: no GPU, no API, no graph run.

Run with the agents environment:
    python -m pytest src/agents_wrapper/fem_workflow/operator_skill/tests

Built bodies are remeshed by an in-process stand-in (conftest.py); the tests of what the real
MeshLab remesh makes are marked slow and skipped by default.  Run them with ``-m slow``.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
from conftest import tabled  # noqa: E402
import inspect_session  # noqa: E402
import sessionlib as lib  # noqa: E402

SI = "SI (m-kg-s)"


# ------------------------------------------------------------ merge / lint --

def settings_card(**extra):
    from src.agents_wrapper.fem_workflow.models import ShapeOptSettings
    return {"kind": "shape_opt_settings", "defaults": ShapeOptSettings().model_dump(mode="json"),
            "strain_estimate": None, "strain_floor": 1.19e-4, **extra}


def test_shape_opt_settings_keep_every_field_the_card_shows():
    card = settings_card()
    card["defaults"]["use_mixed_precision"] = True
    prepared = drive.prepare_answer(card, {}, {"max_newton_iters": 40})
    assert not prepared.errors
    assert prepared.answer["use_mixed_precision"] is True      # kept, not reset
    assert prepared.answer["max_newton_iters"] == 40
    assert prepared.answer["preset_version"] == 2
    assert prepared.gpu and prepared.workers == ["solver"]


def test_plan_settings_drop_keys_the_settings_no_longer_have():
    """An old plan's reinit_n_iters (removed, never read) must not read as a change after every
    solve: the card's defaults never carry it."""
    from src.agents_wrapper.fem_workflow.models import ShapeOptSettings
    settings = drive.plan_settings({"settings": {"reinit_n_iters": 40, "gd_lr": 0.5}})
    assert "reinit_n_iters" not in settings and settings["gd_lr"] == 0.5
    assert settings["use_mixed_precision"] is True and settings["sdf_bc_fix_method"] == "mask"
    assert not drive._settings_changed(
        drive.plan_settings({"settings": {"reinit_n_iters": 40}}),
        ShapeOptSettings().model_dump(mode="json"))


def test_an_old_plan_s_graft_runs_as_mask():
    """graft left the workflow's BC preservation choices; a plan that names it runs as mask,
    and does not read as a change against the card's defaults."""
    from src.agents_wrapper.fem_workflow.models import ShapeOptSettings
    settings = drive.plan_settings({"settings": {"sdf_bc_fix_method": "graft"}})
    assert settings["sdf_bc_fix_method"] == "mask"
    assert not drive._settings_changed(settings, ShapeOptSettings().model_dump(mode="json"))
    assert drive.plan_settings({"settings": {"sdf_bc_fix_method": "zip"}})["sdf_bc_fix_method"] == "zip"


def test_finish_passes_through_untouched():
    prepared = drive.prepare_answer(settings_card(), {}, {"action": "finish"})
    assert prepared.answer == {"action": "finish"} and not prepared.errors and not prepared.gpu


def test_shape_optimization_rejected_for_field_materials():
    workflow = {"material": {"mode": "parts"}}
    prepared = drive.prepare_answer(settings_card(), workflow, {"run_mode": "shape_optimization"})
    assert any("uniform material" in e for e in prepared.errors)


def test_strain_below_float32_floor_warns_about_precision():
    card = settings_card(strain_estimate=1e-6, strain_floor=1.19e-4)
    # float64 is the card's default now, so the default answer is not warned about
    assert not drive.prepare_answer(card, {}, {}).warnings
    prepared = drive.prepare_answer(card, {}, {"use_mixed_precision": False})
    assert any("use_mixed_precision" in w for w in prepared.warnings)
    prepared = drive.prepare_answer(card, {}, {"use_mixed_precision": True})
    assert not prepared.warnings


def test_geometry_physics_fills_axis_and_extent_from_the_card():
    card = {"kind": "geometry_physics", "up_axis": "y", "physical_extent": 0.45,
            "unit_system": SI}
    prepared = drive.prepare_answer(card, {}, {})
    assert not prepared.errors
    assert prepared.answer["up_axis"] == "y" and prepared.answer["physical_extent"] == 0.45
    assert prepared.answer["action"] == "apply_scaling"


def test_geometry_physics_rejects_a_missing_extent():
    card = {"kind": "geometry_physics", "up_axis": None, "physical_extent": None,
            "unit_system": None}
    prepared = drive.prepare_answer(card, {}, {"unit_system": SI, "up_axis": "y"})
    assert any("physical_extent" in e for e in prepared.errors)


def test_partial_segmentation_keeps_the_method():
    card = {"kind": "bc_input", "description": "", "has_regions": False, "cad_faces": False,
            "partfield": {"ready": True},
            "segmentation": {"method": "partfield", "num_clusters": 5, "num_rays": 25,
                             "smoothing": 0.25, "cone_angle_deg": 120.0}}
    prepared = drive.prepare_answer(card, {}, {"action": "preview_segmentation",
                                              "segmentation": {"num_clusters": 7}})
    assert not prepared.errors
    assert prepared.answer["segmentation"]["method"] == "partfield"
    assert prepared.answer["segmentation"]["num_clusters"] == 7
    assert set(prepared.workers) == {"tools", "partfield"}


def test_bc_input_needs_an_action_and_approval_needs_regions():
    card = {"kind": "bc_input", "description": "fix the support", "has_regions": False,
            "segmentation": {}, "partfield": {"ready": False}}
    assert drive.prepare_answer(card, {}, {"description": "x"}).errors
    assert drive.prepare_answer(card, {}, {"action": "approve_regions"}).errors


def bc_values_card():
    return {"kind": "bc_values",
            "dirichlet": [{"name": "support", "components": [0, 1, 2], "value": 0.0}],
            "neumann": [{"name": "top", "traction": [0, 0, 0], "area": 0.2, "loaded_area": 0.16},
                        {"name": "face", "traction": [0, 0, 0], "area": 0.1, "loaded_area": 0.1}],
            "body_force": [0, 0, 0], "acceleration": [0, 0, 0]}


def test_bc_values_merge_by_name_and_leave_a_force_to_the_workflow():
    prepared = drive.prepare_answer(
        bc_values_card(), {"unit_system": SI},
        {"neumann_values": {"top": {"force": [0, -800, 0]}, "face": [0, 0, -100]}})
    assert not prepared.errors, prepared.errors
    assert prepared.answer["neumann_values"]["top"] == {"force": [0.0, -800.0, 0.0]}
    assert prepared.answer["neumann_values"]["face"] == [0.0, 0.0, -100.0]
    assert prepared.answer["dirichlet_values"]["support"]["components"] == [0, 1, 2]
    assert any("loaded area 0.16" in n and "-5000" in n for n in prepared.notes)
    assert any("faces many ways" in w for w in prepared.warnings)   # 0.16 of 0.2 is loaded


def test_bc_values_reject_unknown_names_and_bad_components():
    card = bc_values_card()
    bad = drive.prepare_answer(card, {}, {"neumann_values": {"sit": [0, -1, 0]}})
    assert any("unknown regions" in e for e in bad.errors)
    bad = drive.prepare_answer(card, {}, {"dirichlet_values": {"support": {"components": ["y"], "value": 0}}})
    assert any("components" in e for e in bad.errors)


def test_bc_values_warn_without_any_load():
    prepared = drive.prepare_answer(bc_values_card(), {}, {})
    assert any("no nonzero load" in w for w in prepared.warnings)


def test_a_force_or_the_cards_gravity_counts_as_a_load():
    force_only = drive.prepare_answer(bc_values_card(), {},
                                      {"neumann_values": {"top": {"force": [0, -800, 0]}}})
    assert not any("no nonzero load" in w for w in force_only.warnings)
    card = bc_values_card()
    card["acceleration"] = [0, -9.80665, 0]
    gravity_on_card = drive.prepare_answer(card, {}, {})
    assert not any("no nonzero load" in w for w in gravity_on_card.warnings)


def test_a_region_keyed_at_the_top_level_is_rejected():
    # The graph ignores keys it does not read: the top would keep a zero traction.
    prepared = drive.prepare_answer(bc_values_card(), {},
                                    {"top": {"force": [0, -800, 0]}, "support": {}, "units": "SI",
                                     "acceleration": [0, -9.81, 0]})
    assert "'top' is a load region: put it under neumann_values" in prepared.errors
    assert "'support' is a support region: put it under dirichlet_values" in prepared.errors
    assert any("unknown key 'units'" in e for e in prepared.errors)


def test_a_load_region_left_at_zero_traction_is_flagged():
    prepared = drive.prepare_answer(bc_values_card(), {},
                                    {"neumann_values": {"top": [0, -5000, 0]}})
    assert not prepared.errors
    assert any("'face' has zero traction" in w for w in prepared.warnings)
    assert not any("'top' has zero traction" in w for w in prepared.warnings)


def material_card(**extra):
    return {"kind": "material_input", "material_mode": "uniform", "youngs_modulus": None,
            "poisson_ratio": None, "density": None, "surface_has_colour": False,
            "textured_surface": None, "material_parts": None, **extra}


def test_uniform_material_takes_inferred_values_from_the_card():
    card = material_card(youngs_modulus=1.1e10, poisson_ratio=0.35, density=650.0)
    prepared = drive.prepare_answer(card, {"unit_system": SI}, {})
    assert not prepared.errors
    assert prepared.answer["youngs_modulus"] == 1.1e10 and prepared.answer["material_mode"] == "uniform"


def test_uniform_material_catches_a_unit_slip():
    # 200 "Pa" steel: someone typed GPa into a Pa field.
    prepared = drive.prepare_answer(material_card(), {"unit_system": SI},
                                    {"youngs_modulus": 200, "poisson_ratio": 0.3, "density": 7850})
    assert any("unit slip" in e for e in prepared.errors)


def test_density_required_with_gravity():
    prepared = drive.prepare_answer(material_card(), {"unit_system": SI, "acceleration": [0, -9.81, 0]},
                                    {"youngs_modulus": 1e10, "poisson_ratio": 0.3})
    assert any("density is required" in e for e in prepared.errors)


def test_infer_parts_needs_colour_and_is_a_gpu_step():
    prepared = drive.prepare_answer(material_card(), {}, {"action": "infer_parts"})
    assert any("no colour" in e for e in prepared.errors)
    coloured = drive.prepare_answer(material_card(surface_has_colour=True), {}, {"action": "infer_parts"})
    assert not coloured.errors and coloured.gpu


def test_apply_parts_edits_one_part_and_keeps_the_rest():
    parts = [{"id": 0, "youngs_modulus": 1e10, "poisson_ratio": 0.35, "density": 700.0},
             {"id": 1, "youngs_modulus": 5e6, "poisson_ratio": 0.3, "density": 40.0}]
    card = material_card(material_parts=parts, surface_has_colour=True)
    prepared = drive.prepare_answer(card, {"unit_system": SI},
                                    {"action": "apply_parts", "parts": [{"id": 1, "youngs_modulus": 2e7}]})
    assert not prepared.errors
    assert prepared.answer["parts"][0]["youngs_modulus"] == 1e10
    assert prepared.answer["parts"][1]["youngs_modulus"] == 2e7


def geometry_card(**extra):
    return {"kind": "geometry_input", "cad_revision": {"generator": None, "prompt": None},
            "text2cad": {"ready": True}, "cadcoder": {"ready": True},
            "hunyuan3d": {"ready": True}, **extra}


def test_cad_generators_need_their_revision():
    prepared = drive.prepare_answer(geometry_card(), {}, {"description": "a block with one bore",
                                                          "generator": "text2cad"})
    assert any("revise_prompt" in e for e in prepared.errors)
    card = geometry_card(cad_revision={"generator": "text2cad", "prompt": "A rectangular block."})
    ok = drive.prepare_answer(card, {}, {"description": "A rectangular block.", "generator": "text2cad"})
    assert not ok.errors and ok.workers == ["text2cad"]


def test_revision_is_not_a_gpu_step_and_missing_assets_are_caught(tmp_path):
    prepared = drive.prepare_answer(geometry_card(), {}, {"description": "a solid body",
                                                          "revise_prompt": True})
    assert not prepared.gpu
    missing = drive.prepare_answer(geometry_card(), {}, {
        "description": "a hollow body", "asset_path": str(tmp_path / "none.png"), "asset_kind": "image"})
    assert any("does not exist" in e for e in missing.errors)


def test_triposg_depths_are_the_default_or_a_rung_of_the_image_ladder():
    def errors(settings, image):
        card = geometry_card(drawn_image="d.png") if image else geometry_card()
        answer = {"description": "a body", "generation_settings": settings,
                  **({"use_drawn_image": True} if image else {})}
        prepared = drive.prepare_answer(card, {}, answer)
        return [e for e in prepared.errors if "octree_depth" in e], prepared.need_mib
    assert errors({"init_octree_depth": 3}, False)[0]                    # lower: refused
    assert errors({"init_octree_depth": 6, "octree_depth": 9}, False)[0]  # text stays at 4/7
    assert errors({"init_octree_depth": 4, "octree_depth": 8}, True)[0]   # init rises with the depth
    assert errors({"init_octree_depth": 6, "octree_depth": 9}, True) == ([], 10_000)
    assert errors({"init_octree_depth": 5, "octree_depth": 8}, True) == ([], 5_000)
    assert errors({"seed": 3}, False) == ([], 0)


def test_hunyuan_needs_an_image():
    prepared = drive.prepare_answer(geometry_card(), {}, {"description": "a hollow body",
                                                          "generator": "hunyuan3d"})
    assert any("image" in e for e in prepared.errors)


def test_draw_image_is_a_gpu_step_that_draws_the_description():
    card = geometry_card(sdxl={"ready": True})
    ok = drive.prepare_answer(card, {}, {"action": "draw_image", "description": "a long body",
                                         "image_seed": 3})
    assert not ok.errors and ok.gpu and ok.workers == []
    from_card = drive.prepare_answer(geometry_card(sdxl={"ready": True}, description="a body"),
                                     {}, {"action": "draw_image"})
    assert not from_card.errors
    missing = drive.prepare_answer(geometry_card(sdxl={"ready": False, "summary": "no weights"}),
                                   {}, {"action": "draw_image", "description": "a body",
                                        "image_seed": -1})
    assert any("SDXL is not available here: no weights" in e for e in missing.errors)
    assert any("image_seed" in e for e in missing.errors)
    wordless = drive.prepare_answer(card, {}, {"action": "draw_image"})
    assert any("needs a description" in e for e in wordless.errors)
    duplicated = drive.prepare_answer(card, {}, {"action": "draw_image", "description": "a body",
                                                 "image_prompt": "a body"})
    assert any("there is no image_prompt" in e for e in duplicated.errors)
    other = drive.prepare_answer(card, {}, {"action": "generate", "description": "a body"})
    assert any("only action is draw_image" in e for e in other.errors)


def test_the_drawing_is_used_as_an_image_input():
    card = geometry_card(drawn_image="/s/inputs/sdxl/a_s3.png", description="a long body")
    ok = drive.prepare_answer(card, {}, {"use_drawn_image": True, "generator": "hunyuan3d"})
    assert not ok.errors and ok.gpu and ok.workers == ["hunyuan3d"]
    # The description it was drawn from, merged from the card rather than typed twice.
    assert ok.answer["description"] == "a long body"
    assert any("description taken from the card" in n for n in ok.notes)
    none_drawn = drive.prepare_answer(geometry_card(), {}, {"description": "a body",
                                                            "use_drawn_image": True})
    assert any("draw_image" in e for e in none_drawn.errors)
    both = drive.prepare_answer(card, {}, {"description": "a body", "use_drawn_image": True,
                                           "asset_kind": "scribble"})
    assert any("leave asset_path and asset_kind out" in e for e in both.errors)


def test_erosion_above_zero_is_flagged():
    prepared = drive.prepare_answer(geometry_card(), {}, {
        "description": "a ribbed body", "generation_settings": {"erosion_radius": 2}})
    assert not prepared.errors
    assert any("drops members thinner than 4 voxels" in w for w in prepared.warnings)


# -------------------------------------------------------------------- GPUs --

def gpu(index, used, util=0.0):
    return lib.GPU(index, f"GPU-{index}", "A6000", used, 49140.0, util)


def test_idle_gpu_ignores_our_workers_and_small_foreign_contexts():
    apps = [{"pid": 10, "uuid": "GPU-0", "used_mib": 4362}, {"pid": 10, "uuid": "GPU-1", "used_mib": 264},
            {"pid": 20, "uuid": "GPU-1", "used_mib": 9000}]
    samples = [[gpu(0, 4474), gpu(1, 9287)]]
    verdicts = lib.classify(samples, apps, owners={10: None, 20: "triposg"})
    assert not verdicts[0].idle and "foreign memory" in verdicts[0].reason
    assert verdicts[1].idle                    # 9 GB is our own worker, 264 MiB is foreign
    assert verdicts[1].our_workers and lib.choose_gpu(verdicts).index == 1


def test_busy_utilisation_on_any_sample_is_not_idle():
    samples = [[gpu(0, 100, 0)], [gpu(0, 100, 95)], [gpu(0, 100, 0)]]
    verdicts = lib.classify(samples, [], owners={})
    assert not verdicts[0].idle and "utilization" in verdicts[0].reason
    assert lib.choose_gpu(verdicts) is None


def _guard_args(**extra):
    import argparse
    base = dict(gpu="auto", allow_busy_gpu=False, gpu_wait=0.0, max_foreign_mib=1024.0,
                max_util=10.0)
    base.update(extra)
    return argparse.Namespace(**base)


def _verdicts(idle0, idle1):
    return [lib.GPUVerdict(0, "A6000", 49140, 4474, 0, 4474, 0, [35086], [], idle0,
                           "idle" if idle0 else "foreign memory"),
            lib.GPUVerdict(1, "A6000", 49140, 700, 400, 300, 0, [], [], idle1,
                           "idle" if idle1 else "utilization")]


def test_guard_pins_the_idle_gpu_and_moves_a_worker_parked_elsewhere(monkeypatch):
    stopped = []
    monkeypatch.setattr(lib, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(lib, "gpu_table", lambda *a, **k: _verdicts(False, True))
    monkeypatch.setattr(lib, "worker_files", lambda root: [
        (111, "solver", Path("/x/solver-k.pid")), (222, "tools", Path("/x/tools-k.pid")),
        (333, "partfield", Path("/x/partfield-k.pid"))])
    monkeypatch.setattr(lib, "query_apps", lambda: [])
    monkeypatch.setattr(lib, "query_gpus", lambda: [])
    monkeypatch.setattr(lib, "worker_gpu", lambda pid, apps, gpus: [1] if pid == 222 else [0])
    monkeypatch.setattr(drive, "_stop_worker", lambda pid, kind, pid_file: stopped.append(kind))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    chosen = drive.select_gpu(_guard_args(), Path("/tmp"), True, ["solver", "tools"], "env")
    assert chosen["index"] == 1
    # tools is already on GPU 1; partfield is elsewhere but not needed by this step
    assert stopped == ["solver"]
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_guard_refuses_when_nothing_is_idle(monkeypatch):
    monkeypatch.setattr(lib, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(lib, "gpu_table", lambda *a, **k: _verdicts(False, False))
    with pytest.raises(drive.NoIdleGPU):
        drive.select_gpu(_guard_args(), Path("/tmp"), True, ["solver"], "env")
    # A CPU step is still pinned somewhere, to the least-loaded card, without refusing.
    chosen = drive.select_gpu(_guard_args(), Path("/tmp"), False, [], "env")
    assert chosen["override"].startswith("CPU step")


def test_user_override_is_reported(monkeypatch):
    monkeypatch.setattr(lib, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(lib, "gpu_table", lambda *a, **k: _verdicts(False, True))
    monkeypatch.setattr(lib, "worker_files", lambda root: [])
    monkeypatch.setattr(lib, "query_apps", lambda: [])
    monkeypatch.setattr(lib, "query_gpus", lambda: [])
    chosen = drive.select_gpu(_guard_args(gpu="0"), Path("/tmp"), True, [], "env")
    assert chosen["index"] == 0 and "busy" in chosen["override"]


# -------------------------------------------------------------------- logs --

FORWARD_LOG = """\
[09-26-2026,21:17:59.914] Added flexible Dirichlet BC: 1157 nodes, components=all
[09-26-2026,21:18:00.317] Workflow effective config: use_mixed_precision=True, relative_tol=1.000e-02, remesh_depth=7
[09-26-2026,21:18:02.376] ShapeOptTrimmed: outer 134359 nodes, inner 112990 nodes (trimmed 21369)
[09-26-2026,21:18:02.498]   scale=1.0000 Iter 0: ||R|| = 5.964e+00, rel = 1.000e+00
[09-26-2026,21:18:05.728]   scale=1.0000 Iter 1: ||R|| = 4.065e-03, rel = 6.816e-04
[09-26-2026,21:18:05.928] Simulation finalized. Open x in ParaView to view results.
"""

OPT_LOG = """\
[09-18-2026,12:46:59.300]   SE=1.0000 (raw=6.388635e-01), Vol=0.7296 (raw=7.296200e-01, target=0.7296, norm=0.0000, penalty=0.0014), Total=1.006931e+00
[09-18-2026,12:47:00.989]   Opt iter 0: obj=1.006931e+00, constraints=[none (gd)], rel_red=inf
[09-18-2026,12:47:00.989] ===== Optimization Iteration 1 =====
[09-18-2026,12:47:02.100]   scale=1.0000 Iter 0: ||R|| = 4.370e+00, rel = 1.000e+00
[09-18-2026,12:47:02.900]   scale=1.0000 Iter 1: ||R|| = 5.322e-03, rel = 1.218e-03
[09-18-2026,12:47:03.001]   SE=0.9969 (raw=6.369117e-01), Vol=0.7166 (raw=7.166479e-01, target=0.7296, norm=-0.0178, penalty=0.0000), Total=9.969463e-01
[09-18-2026,12:47:04.606]   Opt iter 1: obj=9.969463e-01, constraints=[none (gd)], rel_red=9.916414e-03
"""

DIVERGING_LOG = """\
[09-20-2026,10:00:00.000]   scale=1.0000 Iter 0: ||R|| = 1.0e+00, rel = 1.000e+00
[09-20-2026,10:00:01.000]   scale=1.0000 Iter 1: ||R|| = 3.7e+00, rel = 3.700e+00
[09-20-2026,10:00:02.000]   scale=1.0000 Iter 2: ||R|| = 5.1e+00, rel = 5.100e+00
[09-20-2026,10:00:03.000] ERROR: Newton solver did not converge in 20 iterations. Final residual: 4.586e-03
"""


def test_forward_log(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text(FORWARD_LOG)
    info = lib.parse_solve_log(path)
    assert info["finalized"] and info["nodes"][-1] == {"outer": 134359, "inner": 112990}
    summary = lib.newton_summary(info["newton"])
    assert summary["final_iterations"] == 1 and summary["final_rel"] == pytest.approx(6.816e-4)
    assert not summary["rel_above_one"]
    assert info["duration_s"] == pytest.approx(6.014, abs=1e-2)
    assert info["effective_config"]["use_mixed_precision"] == "True"


def test_optimization_log(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text(OPT_LOG)
    info = lib.parse_solve_log(path)
    assert [row["opt_iter"] for row in info["trajectory"]] == [0, 1]
    assert info["objective"] == pytest.approx([1.006931, 0.9969463])
    assert "opt iter 1" in lib.progress_line(info)


def test_diverging_log_is_flagged(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text(DIVERGING_LOG)
    info = lib.parse_solve_log(path)
    assert not info["finalized"] and lib.newton_summary(info["newton"])["rel_above_one"]
    assert any("did not converge" in f for f in info["failures"])


# --------------------------------------------------------------- inspector --

def test_segment_names():
    names = inspect_session.segment_names("0: base; 1: underside; 2: pad")
    assert names == {0: "base", 1: "underside", 2: "pad"}
    names = inspect_session.segment_names("0: part a. 1: top. 2: part c. 10: part d 2.5 cm.")
    assert names == {0: "part a", 1: "top", 2: "part c", 10: "part d 2.5 cm"}


def test_face_components_group_faces_by_shared_vertices():
    labels = inspect_session._face_components(np.asarray([[0, 1, 2], [1, 2, 3], [4, 5, 6]], dtype=np.int64))
    assert labels[0] == labels[1] != labels[2]


def test_design_pieces_tell_a_sub_leaf_neck_from_a_detached_solid():
    import trimesh

    def pieces(gap, h):
        # The gap is measured between vertices, so both faces need a fine grid.
        body = trimesh.creation.box(extents=(2.0, 1.0, 1.0)).subdivide_to_size(0.05)
        strut = trimesh.creation.box(extents=(0.5, 0.5, 0.5)).subdivide_to_size(0.05)
        strut.apply_translation((1.25 + gap, 0.0, 0.0))
        design = trimesh.util.concatenate([body, strut])
        return [p["kind"] for p in inspect_session.design_pieces(design, h)]

    assert pieces(0.02, h=0.03) == ["main body", inspect_session.NECK]
    assert pieces(0.10, h=0.03) == ["main body", "detached solid"]
    assert pieces(0.02, h=None) == ["main body", "detached solid"]


def test_region_rows_flag_mixed_facing_loads_and_missing_support():
    resolved = {"regions": [{
        "name": "top", "category": "neumann", "faces": 500, "area": 0.2, "loaded_area": 0.1,
        "area_fraction": 0.1,
        "bounds": [[0, 0, 0], [0.4, 0.05, 0.4]], "centroid": [0.2, 0.4, 0.2],
        "mean_normal": [0.0, 0.5, 0.1], "selector": {"type": "surface_segment"}}], "legend": []}
    proposal = {"dirichlet": [], "neumann": [{"name": "top", "traction": [0, -1000, 0]}]}
    warnings: list[str] = []
    rows, total, peak = inspect_session.region_rows(resolved, proposal, 0.01, warnings)
    assert rows[0]["applied_force"] == pytest.approx([0, -100, 0])
    assert rows[0]["nominal_force"] == pytest.approx([0, -200, 0])
    assert rows[0]["loaded_fraction"] == pytest.approx(0.5)
    assert peak == pytest.approx(1000)
    assert any("facing many ways" in w for w in warnings)
    assert any("no support" in w for w in warnings)


def _vtu_array(name, values, components=1):
    import base64
    raw = np.asarray(values, dtype="<f4").tobytes()
    payload = base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode()
    return (f'<DataArray type="Float32" Name="{name}" NumberOfComponents="{components}" '
            f'format="binary">{payload}</DataArray>')


def test_read_vtu_and_von_mises(tmp_path):
    points = [[0, 0, 0], [1, 0, 0]]
    stress = [[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0]]     # uniaxial 1, pure shear 1
    path = tmp_path / "s.vtu"
    path.write_text('<VTKFile><UnstructuredGrid><Piece>'
                    + _vtu_array("Points", points, 3) + _vtu_array("stress", stress, 6)
                    + _vtu_array("youngs_modulus", [1e9, 3e10])
                    + '</Piece></UnstructuredGrid></VTKFile>')
    data = lib.read_vtu(path, ["stress"])
    assert set(data) == {"Points", "stress"} and data["stress"].shape == (2, 6)
    assert lib.von_mises(data["stress"]) == pytest.approx([1.0, np.sqrt(3.0)])


def test_the_inspector_reads_a_failed_solve_marker(tmp_path):
    session = tmp_path / "abc"
    run = session / "solve" / "api_fem_forward-1"
    run.mkdir(parents=True)
    (run / "log.txt").write_text(FORWARD_LOG)
    (run / "solution.pvd").write_text("")
    (run / "vtu_surface").mkdir()
    (run / "vtu_surface" / "surface_t0000.vtu").write_text("")
    import os, time
    old = time.time() - 100
    os.utime(run / "log.txt", (old, old))
    (session / "solve" / "failed.json").write_text(json.dumps(
        {"started": time.time(), "error": "RuntimeError: boom", "log": None}))
    warnings: list[str] = []
    solve = inspect_session.solve_summary(session, run, [1, 1, 1], 0.0, "uniform", {}, 0.01,
                                          None, warnings)
    assert solve["succeeded"] is False and solve["latest_attempt_failed"]["error"].endswith("boom")
    assert any("failed before writing a run folder" in w for w in warnings)


# ------------------------------------------------------------------ run --

PLAN = {
    "geometry": {"description": "a steel body standing on its flat base"},
    "geometry_routes": ["triposg_text", "drawing", "hunyuan3d"],
    "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "y", "physical_extent": 0.1},
    "regions": "Fix all components of the lowest faces (the lowest 2% of the height). "
               "Apply a downward (-y) traction on the upward-facing top of the face.",
    "values": {"load": {"force": [0, -20, 0]}, "acceleration": [0, -9.80665, 0]},
    "material": {"youngs_modulus": 2.3e9, "poisson_ratio": 0.35, "density": 1050},
    "settings": {},
    "contacts": {"base": {"transmits": "force and moment", "regions": ["support"]}},
    "prediction": {"max_displacement": 1e-4, "max_stress": 2e6},
}


def test_load_plan_names_what_is_missing(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({k: v for k, v in PLAN.items() if k != "values"}))
    with pytest.raises(drive.PlanError, match="values"):
        drive.load_plan(path)
    path.write_text(json.dumps({**PLAN, "load": {}}))
    with pytest.raises(drive.PlanError, match="unknown plan keys"):
        drive.load_plan(path)
    path.write_text(json.dumps(PLAN))
    assert drive.load_plan(path)["regions"].startswith("Fix")


def test_plan_answers_the_cards_in_order():
    answer, stage, stop = drive.plan_answer("geometry_input", {"kind": "geometry_input"}, PLAN)
    assert answer == PLAN["geometry"] and (stage, stop) == ("generation", "geometry")
    assert drive.plan_answer("geometry_texture", {}, PLAN)[0] == {"action": "skip_texture"}
    answer, stage, stop = drive.plan_answer("geometry_physics", {}, PLAN)
    assert answer["action"] == "apply_scaling" and answer["physical_extent"] == 0.1
    # First visit: segment and name the parts; then preview; after the stop, with the
    # card showing these words: approve.
    answer, stage, stop = drive.plan_answer("bc_input", {"description": ""}, PLAN)
    assert answer == {"action": "preview_segmentation"} and stage is None
    segmented = {"description": "", "has_segmentation": True}
    answer, stage, stop = drive.plan_answer("bc_input", segmented, PLAN)
    assert answer["action"] == "preview_regions" and (stage, stop) == ("regions", "regions")
    shown = {"description": PLAN["regions"], "has_regions": True, "has_segmentation": True,
             "segmentation": {"method": "partfield"}}
    assert drive.plan_answer("bc_input", shown, PLAN)[0] == {"action": "approve_regions"}
    # Reworded regions or another segmentation preview again.
    plan = {**PLAN, "segmentation": {"num_clusters": 8}}
    assert drive.plan_answer("bc_input", shown, plan)[0]["action"] == "preview_segmentation"
    with pytest.raises(drive.PlanError, match="reword"):
        drive.plan_answer("bc_input", {**shown, "preview_error": "BC 'x' selected no faces"}, PLAN)
    card = {"dirichlet": [{"name": "support"}], "neumann": [{"name": "face"}, {"name": "end"}]}
    answer, stage, stop = drive.plan_answer("bc_values", card, PLAN)
    assert answer["dirichlet_values"] == {"support": {"components": [0, 1, 2], "value": 0.0}}
    assert answer["neumann_values"] == {"face": {"force": [0, -20, 0]}, "end": {"force": [0, -20, 0]}}
    assert answer["acceleration"] == [0, -9.80665, 0]
    with pytest.raises(drive.PlanError, match="no load"):
        drive.plan_answer("bc_values", card, {**PLAN, "values": {"neumann_values": {"face": [0, -5, 0]}}})
    answer, _, _ = drive.plan_answer("material_input", {}, PLAN)
    assert answer["material_mode"] == "uniform" and answer["density"] == 1050
    answer, stage, stop = drive.plan_answer("shape_opt_settings", {"defaults": {}}, PLAN)
    assert answer["use_mixed_precision"] is True and (stage, stop) == ("solve", "solve")
    # Every field is sent, from the class defaults, so a setting tried once does
    # not outlive its line in the plan.
    assert answer["beta_max_over_e"] is None and answer["remesh_octree_depth"] == 7
    sent = {**answer}
    solved = {"solve_complete": True, "defaults": sent}
    assert drive.plan_answer("shape_opt_settings", solved, PLAN) == (None, None, None)
    changed = {**PLAN, "settings": {"remesh_octree_depth": 8}}
    assert drive.plan_answer("shape_opt_settings", solved, changed)[1] == "solve"
    stale = {"solve_complete": True, "defaults": {**sent, "beta_max_over_e": 500}}
    assert drive.plan_answer("shape_opt_settings", stale, PLAN)[0]["beta_max_over_e"] is None
    failed = {"solve_error": "boom", "defaults": sent}
    with pytest.raises(drive.PlanError, match="failed"):
        drive.plan_answer("shape_opt_settings", failed, PLAN)
    assert drive.plan_answer("shape_opt_settings", failed, changed)[2] == "solve"


def test_plan_draws_first_then_generates_from_the_drawing():
    plan = {**PLAN, "geometry": {**PLAN["geometry"], "draw": True, "image_seed": 3}}
    answer, stage, stop = drive.plan_answer("geometry_input", {"sdxl": {"ready": True}}, plan)
    assert answer == {"action": "draw_image", "description": PLAN["geometry"]["description"],
                      "image_seed": 3} and stage == "drawing"
    card = {"drawn_image": "x.png", "drawn_image_prompt": PLAN["geometry"]["description"], "image_seed": 3}
    answer, stage, stop = drive.plan_answer("geometry_input", card, plan)
    assert answer["use_drawn_image"] is True and "draw" not in answer and stop == "geometry"
    # Other words, or another seed, draw again instead of reusing the drawing.
    other = {**plan, "geometry": {**plan["geometry"], "image_seed": 4}}
    assert drive.plan_answer("geometry_input", card, other)[0]["action"] == "draw_image"
    other = {**plan, "geometry": {**plan["geometry"], "description": "another body"}}
    assert drive.plan_answer("geometry_input", card, other)[0]["action"] == "draw_image"
    assert stage is None          # the drawing was the attempt; no text generation is spent


def test_plan_revises_the_prompt_then_generates_the_revision(tmp_path):
    first = "a block 40 x 20 x 10 mm with one 6 mm bore through its thickness"
    plan = {**PLAN, "geometry": {"description": first, "generator": "text2cad",
                                 "revise_prompt": True}}
    card = {"kind": "geometry_input", "description": first}
    answer, stage, stop = drive.plan_answer("geometry_input", card, plan)
    # The revision of a CAD description is a stop: it is read before generating.
    assert answer["revise_prompt"] is True and stage is None and stop == "revision"
    revised = {"kind": "geometry_input", "description": "A rectangular block with one hole"}
    drive.note_revision(tmp_path, "geometry_input", answer, revised)
    record = lib.read_json(tmp_path / "operator" / "run_revision.json")
    answer, stage, stop = drive.plan_answer("geometry_input", revised, plan, record)
    assert answer["description"] == revised["description"] and answer["revise_prompt"] is False
    # A CAD generation, a built body and a text generation each have their own attempts.
    assert (stage, stop) == ("cad", "geometry")
    # Without the record the card's text is not taken for a revision of these words.
    assert drive.plan_answer("geometry_input", revised, plan)[2] == "revision"
    upload = {**PLAN, "geometry": {"asset_path": "/x/blank.glb", "asset_kind": "surface"}}
    assert drive.plan_answer("geometry_input", {}, upload)[1:] == ("upload", "geometry")
    assert drive.plan_answer("geometry_input", {}, PLAN)[1:] == ("generation", "geometry")
    assert {"cad", "upload"} <= set(drive.RUN_LIMITS)


def test_a_description_edited_at_the_revision_stop_is_revised_again(tmp_path):
    first, second = "a block 40 x 20 x 10 mm", "a block 40 x 20 x 10 mm with two 6 mm bores"
    geometry = {"generator": "cadcoder", "revise_prompt": True}
    sent = {**geometry, "description": first}
    drive.note_revision(tmp_path, "geometry_input", sent,
                        {"kind": "geometry_input", "description": "revised first"})
    record = lib.read_json(tmp_path / "operator" / "run_revision.json")
    edited = {**PLAN, "geometry": {**geometry, "description": second}}
    card = {"kind": "geometry_input", "description": "revised first"}
    answer, stage, stop = drive.plan_answer("geometry_input", card, edited, record)
    # The edit is revised, not replaced by the stale revision of the old words.
    assert answer["description"] == second and answer["revise_prompt"] is True
    assert stage is None and stop == "revision"
    # Another generator's revision of the same words is not this generator's.
    other = {**PLAN, "geometry": {**geometry, "generator": "text2cad", "description": first}}
    assert drive.plan_answer("geometry_input", card, other, record)[2] == "revision"
    # A revision whose call failed leaves no record to generate from.
    drive.note_revision(tmp_path / "x", "geometry_input", {**geometry, "description": second},
                        {"kind": "geometry_input", "description": second, "api_error": "timeout"})
    assert lib.read_json(tmp_path / "x" / "operator" / "run_revision.json") is None


def test_run_revises_an_edited_description_before_generating(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    state = {"card": {"kind": "geometry_input", "description": "first words"}}
    sent, printed = [], []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": state["card"], "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        answer = prepared.answer
        sent.append(answer)
        if answer.get("revise_prompt"):       # the reviser returns to the same card
            state["card"] = {"kind": "geometry_input", "description": "revised " + answer["description"]}
        else:
            state["card"] = {"kind": "geometry_physics"}
        return {"ok": True, "card": state["card"]}
    monkeypatch.setattr(drive, "_run_step", step)
    plan_path = session / "operator" / "plan.json"

    def run(words):
        plan = {**PLAN, "geometry": {"description": words, "generator": "cadcoder", "revise_prompt": True}}
        plan_path.write_text(json.dumps(plan))
        drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                         plan=str(plan_path), full=True))
        return printed[-1]

    assert run("first words")["stop"] == "revision"
    # Edited at the revision stop: the new words are revised, not the old revision generated.
    assert run("second words")["stop"] == "revision"
    assert sent[-1] == {"description": "second words", "generator": "cadcoder", "revise_prompt": True}
    # Accepted: the revision of the current words is generated, and counted once.
    out = run("second words")
    assert out["stop"] == "geometry" and sent[-1]["description"] == "revised second words"
    assert sent[-1]["revise_prompt"] is False and out["attempts"] == {"cad": 1}
    # The generation restored the revision budget: two more revisions, then the third is refused
    # before it is sent, with the way on.
    state["card"] = {"kind": "geometry_input", "description": ""}
    assert run("third words")["attempts_left"]["cad_revisions"] == 1
    assert run("fourth words")["attempts_left"]["cad_revisions"] == 0
    count = len(sent)
    out = run("fifth words")
    assert out["stop"] == "limit" and "generate from the revision the card shows" in out["error"]
    assert len(sent) == count


def test_an_unchanged_resend_of_a_cached_stage_is_not_charged(tmp_path):
    answer = {"description": "words one", "generator": "text2cad"}
    first = drive.charge_attempt(tmp_path, "cad", answer)
    again = drive.charge_attempt(tmp_path, "cad", dict(answer))
    assert first["counted"] and not again["counted"] and again["counts"]["cad"] == 1
    assert drive.charge_attempt(tmp_path, "cad", {**answer, "description": "words two"})["counted"]
    third = drive.charge_attempt(tmp_path, "cad", {**answer, "description": "words three"})
    assert third["counts"]["cad"] == 3 and not third["over"]
    assert drive.charge_attempt(tmp_path, "cad", {**answer, "description": "words four"})["over"]
    # An upload is the same attempt only if the file's bytes are the same.
    body = tmp_path / "body.glb"
    body.write_bytes(b"one")
    upload = {"asset_path": str(body), "asset_kind": "surface"}
    assert drive.charge_attempt(tmp_path, "upload", upload)["counted"]
    assert not drive.charge_attempt(tmp_path, "upload", upload)["counted"]
    body.write_bytes(b"two")
    assert drive.charge_attempt(tmp_path, "upload", upload)["counted"]
    # A solve is never served from a cache: every one counts.
    drive.charge_attempt(tmp_path, "solve", {"remesh_octree_depth": 7})
    assert drive.charge_attempt(tmp_path, "solve", {"remesh_octree_depth": 7})["counts"]["solve"] == 2
    # A step the GPU guard never ran is taken face.
    drive.refund_attempt(tmp_path, "upload", upload)
    assert lib.read_json(tmp_path / "operator" / "run_counts.json")["upload"] == 1
    assert drive.charge_attempt(tmp_path, "upload", upload)["counted"]


def test_attempt_stage_names_the_limit_each_answer_spends():
    stage = drive.attempt_stage
    assert stage("geometry_input", {"description": "a"}) == "generation"
    assert stage("geometry_input", {"description": "a", "generator": "text2cad"}) == "cad"
    assert stage("geometry_input", {"description": "a", "generator": "cadcoder",
                                    "revise_prompt": True}) is None
    assert stage("geometry_input", {"asset_path": "/b.glb", "asset_kind": "surface"}) == "upload"
    assert stage("geometry_input", {"action": "draw_image", "description": "a"}) == "drawing"
    assert stage("geometry_input", {"use_drawn_image": True}) is None
    assert stage("bc_input", {"action": "preview_regions"}) == "regions"
    assert stage("bc_input", {"action": "approve_regions"}) is None
    assert stage("material_input", {"action": "infer_parts"}) == "materials"
    assert stage("shape_opt_settings", {"use_mixed_precision": True}) == "solve"
    assert stage("shape_opt_settings", {"action": "finish"}) is None


def test_answer_counts_and_limits_the_attempts_it_sends(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    card = {"kind": "geometry_input", "description": ""}
    sent, printed = [], []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": card, "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        sent.append(prepared.answer)
        return {"ok": True, "card": card}
    monkeypatch.setattr(drive, "_run_step", step)

    def answer(obj):
        drive.cmd_answer(argparse.Namespace(
            output_root=str(tmp_path), session_id="abc", json=json.dumps(obj), file=None,
            no_merge=False, no_lint=False, dry_run=False, force=False))
    for words in ("words one", "words one"):
        answer({"description": words})
    counts = lib.read_json(session / "operator" / "run_counts.json")
    assert counts == {"generation": 1} and len(sent) == 2     # the repeat was sent, not counted
    answer({"description": "words two"})                       # TripoSG text is tried once
    assert len(sent) == 2 and "TripoSG text is tried once" in printed[-1]["error"]
    answer({"description": "words one", "generation_settings": {"seed": 1}})   # its body was made
    assert len(sent) == 2 and "TripoSG text is tried once" in printed[-1]["error"]
    # again only with a new seed when it returned no watertight surface (no body was made)
    failed = "RuntimeError: TripoSG produced no usable watertight surface component"
    drive.note_no_surface(session, "geometry_input", {"description": "words one"}, failed)
    answer({"description": "words two", "generation_settings": {"seed": 1}})   # new words: refused
    assert len(sent) == 2
    answer({"description": "words one", "generation_settings": {"seed": 1}})
    assert len(sent) == 3 and lib.read_json(session / "operator" / "run_counts.json") == {"generation": 2}
    # that seed change failing too is not changed again
    drive.note_no_surface(session, "geometry_input", {"description": "words one",
                                                      "generation_settings": {"seed": 1}}, failed)
    assert lib.read_json(session / "operator" / "no_surface.json")["retried"] is True
    assert not drive.seed_retry(session, {"description": "words one", "generation_settings": {"seed": 2}})
    answer({"description": "words four"})                      # refused, nothing sent
    assert len(sent) == 3


def test_a_held_support_that_moves_in_its_held_components_is_sent_to_the_region():
    motion = [{"name": "base", "category": "dirichlet", "mean": 0.3, "max": 0.3,
               "held_mean": 0.3, "held_max": 0.3},
              {"name": "side", "category": "dirichlet", "mean": 0.9, "max": 0.9,
               "held_mean": 0.001, "held_max": 0.002},          # moves only where released
              {"name": "edge", "category": "dirichlet", "mean": 0.2, "max": 0.2,
               "held_mean": 0.2, "held_max": 0.2},              # moves where it holds
              {"name": "stray", "category": "dirichlet", "mean": 0.01, "max": 0.78,
               "held_mean": 0.01, "held_max": 0.78},            # a part of it moves
              {"name": "still", "category": "dirichlet", "mean": 0.01, "max": 0.02,
               "held_mean": 0.01, "held_max": 0.02},
              {"name": "top", "category": "neumann", "mean": 1.0, "max": 1.0}]
    held = {"base": [0, 1, 2], "side": [1], "edge": [1], "stray": [0, 1, 2], "still": [0, 1, 2]}
    out = inspect_session.support_motion_warnings(motion, held, 1.0)
    assert len(out) == 3
    assert "'base'" in out[0] and "did not hold in the solve" in out[0] and "before any solver setting" in out[0]
    assert "'edge'" in out[1] and "components it holds [1]" in out[1]
    assert "'stray'" in out[2] and "up to 78%" in out[2]


def test_cad_face_summary_counts_the_brep_faces_by_type(tmp_path):
    import numpy as np
    (tmp_path / "geometry").mkdir()
    np.savez(tmp_path / "geometry" / "generated.glb.faces.npz",
             face_ids=np.zeros(4, dtype=np.int32),
             surface_types=np.array(["plane"] * 6 + ["cylinder"] * 4))
    out = drive.cad_face_summary(tmp_path)
    assert out["faces"] == 10 and out["by_type"] == {"plane": 6, "cylinder": 4}
    assert drive.cad_face_summary(tmp_path / "none") is None


# The real MeshLab remesh (about 10 s a body): this test is about what it makes.
@pytest.mark.slow
@pytest.mark.real_remesh
def test_a_built_body_has_faces_to_select_and_is_not_too_dense():
    box = drive.build_blank("box", size=[0.26, 0.018, 0.024])
    assert 500 < len(box.faces) <= 60_000 and box.is_watertight
    cyl = drive.build_blank("cylinder", radius=0.013, length=0.9, axis="x")
    assert 500 < len(cyl.faces) <= 60_000 and cyl.is_watertight


def test_load_plan_rejects_a_load_that_is_neither_traction_nor_force(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**PLAN, "values": {"load": {"traction": [0, -5000, 0]}}}))
    with pytest.raises(drive.PlanError, match="traction \\[tx, ty, tz\\]"):
        drive.load_plan(path)
    path.write_text(json.dumps({**PLAN, "values": {"load": [0, -5000, 0],
                                                   "neumann_values": {"upper": {"force": [0, -1, 0]}}}}))
    assert drive.load_plan(path)["values"]["load"] == [0, -5000, 0]


def test_changed_sections_name_the_earliest_card_to_rewind_to():
    sent = {s: drive._normalized(PLAN[s]) for s in ("geometry", "scale", "regions", "values", "material")}
    assert drive.changed_sections(PLAN, sent, "shape_opt_settings") == []
    plan = {**PLAN, "regions": "Fix the support. Load the end.", "material": {"youngs_modulus": 1e9}}
    assert drive.changed_sections(plan, sent, "shape_opt_settings")[0] == ("regions", "bc_input")
    # A section whose card is still ahead is not a rewind.
    assert drive.changed_sections(plan, sent, "bc_input") == []
    plan = {**PLAN, "geometry": {"description": "another body"}}
    assert drive.changed_sections(plan, sent, "geometry_texture") == [("geometry", "geometry_input")]


def test_plan_values_are_compared_as_numbers_not_text():
    # statics' reaction as the runner applied it ([-0.0, ..., 0.0]) against the plan's [0, ..., 0]:
    # the same values, so no rewind to the values card (it repainted every run before)
    applied = {**PLAN["values"], "neumann_values": {"rest": {"force": [-0.0, 120.5, 0.0]}}}
    sent = {s: drive._normalized(PLAN[s]) for s in ("geometry", "scale", "regions", "material")}
    sent["values"] = drive._normalized(applied)
    plan = {**PLAN, "values": {**PLAN["values"], "neumann_values": {"rest": {"force": [0, 120.5, 0]}}}}
    assert drive.changed_sections(plan, sent, "shape_opt_settings") == []
    # rounding past the sixth significant digit is no change; a real change still is one
    plan["values"]["neumann_values"]["rest"]["force"] = [0.0, 120.5000001, 0]
    assert drive.changed_sections(plan, sent, "shape_opt_settings") == []
    plan["values"]["neumann_values"]["rest"]["force"] = [0, 121.0, 0]
    assert drive.changed_sections(plan, sent, "shape_opt_settings") == [("values", "bc_values")]
    assert drive.same_plan_value(drive._normalized({"up_axis": "z", "physical_extent": 1}),
                                 {"up_axis": "z", "physical_extent": 1.0})
    assert not drive.same_plan_value(drive._normalized({"up_axis": "z"}), {"up_axis": "y"})
    # integers and booleans exactly; only floats rounded
    assert not drive.same_plan_value(drive._normalized({"flag": True}), {"flag": 1})
    assert not drive.same_plan_value(drive._normalized({"n": 123456789}), {"n": 123456788})
    assert drive.same_plan_value(drive._normalized({"x": 0.1 + 0.2}), {"x": 0.3})
    # an integral float is its integer, past six digits too; a fraction beside it is not
    assert drive.same_plan_value(drive._normalized({"x": [0, 1234567, 0]}), {"x": [-0.0, 1234567.0, 0.0]})
    assert drive.same_plan_value(drive._normalized({"E": 7e10}), {"E": 70000000000})
    assert not drive.same_plan_value(drive._normalized({"n": 1234567}), {"n": 1234568.0})


def test_statics_reactions_the_runner_applied_are_no_plan_change(tmp_path):
    # As an acceptance run (2026-10-10) left it: the plan loads two feet and declares two simple
    # supports, statics' reactions on them were applied by the runner and sent, and the plan never
    # held them.  Each later run read that as a changed plan.values and rewound to the values card.
    session = tmp_path / "s"
    own = {"acceleration": [0, -9.80665, 0], "body_force": [0, 0, 0],
           "neumann_values": {"load_a": {"force": [0, -1569, 0]}, "load_b": {"force": [0, -1569, 0]}}}
    reactions = [{"region": "support_a", "contact": "a", "force_N": [0.0, 1582.308, 0.0]},
                 {"region": "support_b", "contact": "b", "force_N": [-0.0, 1583.769, 0.0]}]
    sent_values = {**own, "neumann_values": {**own["neumann_values"],
                                             "support_a": {"force": [0.0, 1582.308, 0.0]},
                                             "support_b": {"force": [-0.0, 1583.769, 0.0]}}}
    sent = {s: drive._normalized(PLAN[s]) for s in ("geometry", "scale", "regions", "material")}
    sent["values"] = drive._normalized(sent_values)
    plan = {**PLAN, "values": own}
    drive.lib.write_json(drive.lib.operator_dir(session) / "statics.json",
                         {"reactions": reactions, "applied": ["support_a", "support_b"]})
    # the bug: the plan alone against what was sent
    assert drive.changed_sections(plan, sent, "material_input") == [("values", "bc_values")]
    # the runner's own reactions are not a change, whether the plan leaves them out or copies them
    assert drive.changed_sections(plan, sent, "material_input", *drive.statics_filled(session)) == []
    copied = {**own, "neumann_values": {**own["neumann_values"], "support_a": {"force": [0, 1582.308, 0]},
                                        "support_b": {"force": [0, 1583.769, 0]}}}
    assert drive.changed_sections({**plan, "values": copied}, sent, "material_input",
                                  *drive.statics_filled(session)) == []
    # a value the runner applied over the plan's own (statics corrected it) is the runner's too
    typed = {**own, "neumann_values": {**own["neumann_values"], "support_a": {"force": [0, 1500, 0]}}}
    assert drive.changed_sections({**plan, "values": typed}, sent, "material_input",
                                  *drive.statics_filled(session)) == []
    # a load the agent changed is still a change
    moved = {**own, "neumann_values": {**own["neumann_values"], "load_a": {"force": [0, -1600, 0]}}}
    assert drive.changed_sections({**plan, "values": moved}, sent, "material_input",
                                  *drive.statics_filled(session)) == [("values", "bc_values")]
    # a record from before ``applied`` existed: a reaction the plan lacks is still the runner's
    drive.lib.write_json(drive.lib.operator_dir(session) / "statics.json", {"reactions": reactions})
    assert drive.changed_sections(plan, sent, "material_input", *drive.statics_filled(session)) == []
    assert drive.changed_sections({**plan, "values": typed}, sent, "material_input",
                                  *drive.statics_filled(session)) == [("values", "bc_values")]


def test_planned_load_shows_the_newtons_each_region_gets():
    regions = [{"name": "support", "category": "dirichlet", "loaded_area": 0.01},
               {"name": "upper", "category": "neumann", "loaded_area": 17.0}]
    rows = drive.planned_load({"values": {"load": [0, -5000, 0]}}, regions)
    assert rows == [{"region": "upper", "force_N": [0.0, -85000.0, 0.0], "magnitude_N": 85000.0,
                     "from": "traction [0, -5000, 0] x loaded area 17"}]
    rows = drive.planned_load({"values": {"load": {"force": [0, -5000, 0]}}}, regions)
    assert rows[0]["magnitude_N"] == 5000.0 and rows[0]["from"] == "a force total"


def test_judge_replaces_the_report_placeholder(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    lib.write_json(session / "operator" / "prediction_check.json", {"displacement_ratio": 1.1, "stress_ratio": 0.9})
    lib.write_json(session / "operator" / "prediction_history.json", [{"run": "r", "solved": {}}])   # a solve ran
    (session / "operator" / "report.md").write_text(
        "# Session abc\n\n## Warnings\n- none\n\n## Engineer's judgement\n\n_(fill in)_\n")
    judgement = tmp_path / "j.md"
    judgement.write_text("The brief's case was run.\n## Deviations\nnone\nThe body deflects 13 um under 30 N.\n")
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                       file=str(judgement), text=None))
    text = (session / "operator" / "report.md").read_text()
    assert printed[-1]["ok"] and "_(fill in)_" not in text
    assert text.endswith("## Engineer's judgement\n\nThe brief's case was run.\n## Deviations\nnone\nThe body deflects 13 um under 30 N.\n")
    assert "## Warnings\n- none" in text


def test_brief_run_keeps_the_decision_inputs():
    value = {"ok": True, "stop": "regions", "steps": [{"kind": "bc_input", "action": "preview_regions",
                                                       "ok": True, "elapsed_s": 3.0, "notes": [], "gpu": 1}],
             "regions": [{"name": "upper", "category": "neumann", "faces": 10, "centroid": [0, 0, 0],
                          "bbox_size": [1, 1, 1], "normal_coherence": 0.99, "loaded_fraction": 0.99,
                          "selector": "and", "area_fraction": 0.1}],
             "planned_load": [], "look_at": ["a.png"], "warnings": [], "limits": {}}
    brief = drive.brief_run(value)
    assert "limits" not in brief and "gpu" not in brief["steps"][0]
    assert brief["regions"][0] == {"name": "upper", "category": "neumann", "faces": 10,
                                   "area_fraction": 0.1, "normal_coherence": 0.99,
                                   "loaded_fraction": 0.99, "selector": "and"}


def test_plan_material_parts_paints_values_corrects_then_approves():
    plan = {**PLAN, "material": {"mode": "parts", "description": "steel part one, wood part two",
                                 "parts": [{"id": 1, "youngs_modulus": 2e11}]}}
    bare = material_card(surface_has_colour=False)
    answer, stage, stop = drive.plan_answer("material_input", bare, plan)
    assert answer == {"action": "texture_surface", "per_part_colours": True}    # flat by part, no painter
    assert stage is None and stop is None and not drive.plan_paints(plan)
    told = {**plan, "material": {**plan["material"], "texture_prompt": "grey steel, brown wood"}}
    assert drive.plan_answer("material_input", bare, told)[0]["texture_prompt"] == "grey steel, brown wood"
    painted = material_card(surface_has_colour=False, textured_surface="/x/painted.glb")
    answer, stage, stop = drive.plan_answer("material_input", painted, plan)
    assert answer == {"action": "infer_parts", "material_description": "steel part one, wood part two"}
    assert (stage, stop) == ("materials", "materials")
    valued = material_card(surface_has_colour=True, material_description="steel part one, wood part two", material_parts=[
        {"id": 0, "youngs_modulus": 1e10, "poisson_ratio": 0.35, "density": 700.0},
        {"id": 1, "youngs_modulus": 3e9, "poisson_ratio": 0.35, "density": 1200.0}])
    answer, stage, stop = drive.plan_answer("material_input", valued, plan)
    assert answer == {"action": "apply_parts", "parts": [{"id": 1, "youngs_modulus": 2e11}]}
    assert stage is None and stop == "materials"
    corrected = material_card(surface_has_colour=True, material_description="steel part one, wood part two", material_parts=[
        {"id": 0, "youngs_modulus": 1e10, "poisson_ratio": 0.35, "density": 700.0},
        {"id": 1, "youngs_modulus": 2e11, "poisson_ratio": 0.35, "density": 1200.0}])
    assert drive.plan_answer("material_input", corrected, plan) == ({"material_mode": "parts"}, None, None)
    # Without corrections the valued card is approved at once.
    plain = {**plan, "material": {"mode": "parts", "description": "steel part one, wood part two"}}
    assert drive.plan_answer("material_input", valued, plain)[0] == {"material_mode": "parts"}


def test_load_plan_checks_the_parts_material(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**PLAN, "material": {"mode": "vomp"}}))
    with pytest.raises(drive.PlanError, match="VoMP"):
        drive.load_plan(path)
    path.write_text(json.dumps({**PLAN, "material": {"mode": "parts"}}))
    with pytest.raises(drive.PlanError, match="description"):
        drive.load_plan(path)
    path.write_text(json.dumps({**PLAN, "material": {"mode": "parts", "description": "steel, wood",
                                                     "parts": [{"youngs_modulus": 1}]}}))
    with pytest.raises(drive.PlanError, match="id"):
        drive.load_plan(path)
    path.write_text(json.dumps({**PLAN, "material": {"mode": "parts", "description": "steel, wood"}}))
    assert drive.load_plan(path)["material"]["mode"] == "parts"


def test_materials_stop_lists_the_parts_with_their_segments(tmp_path):
    session = tmp_path / "s"
    (session / "boundary_conditions").mkdir(parents=True)
    (session / "boundary_conditions" / "segments.json").write_text(json.dumps({"method": "partfield", "segments": [
        {"id": 0, "area_fraction": 0.7, "centroid": [0, 0.1, 0]},
        {"id": 1, "area_fraction": 0.3, "centroid": [0, 0.5, 0]}]}))
    # stamped with the surface they were computed on, as the workflow stamps them
    from src.agents_wrapper.fem_workflow.monitor import _digest
    surface = session / "physical.glb"
    surface.write_bytes(b"body")
    np.savez(session / "boundary_conditions" / "segments.npz", segment_ids=np.zeros(2))
    lib.write_json(session / "boundary_conditions" / "segments.source.json", {"sha256": _digest(surface)})
    lib.write_json(session / "session.json", {"segmentation_method": "partfield", "surface_path": str(surface)})
    montage = session / "montage.png"
    montage.write_bytes(b"")
    card = {"kind": "material_input", "parts_montage": str(montage), "material_parts": [
        {"id": 0, "youngs_modulus": 1e10, "poisson_ratio": 0.35, "density": 700.0},
        {"id": 1, "youngs_modulus": 5e7, "poisson_ratio": 0.4, "density": 1100.0}]}
    payload = drive._stop_payload("materials", session, tmp_path, card,
                                  {**PLAN, "material": {"mode": "parts", "description": "two parts"}})
    assert payload["look_at"] == [str(montage)]
    assert payload["parts"][1]["area_fraction"] == 0.3 and payload["parts"][1]["centroid"] == [0, 0.5, 0]
    assert payload["soft_parts"] == [1]
    brief = drive.brief_run({"ok": True, **payload})
    assert brief["parts"] and brief["soft_parts"] == [1]


def test_parts_mean_density_reads_the_field(tmp_path):
    import numpy as np
    root = tmp_path / "materials" / "parts" / "field"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"channels": ["youngs_modulus", "poisson_ratio", "density"]}))
    grid = np.zeros((2, 2, 2, 3)); grid[..., 2] = 800.0; grid[0, 0, 0, 2] = 7850.0
    mask = np.zeros((2, 2, 2), dtype=bool); mask[0, 0, 0] = True; mask[1, 1, 1] = True
    np.save(root / "material_grid.npy", grid); np.save(root / "valid_mask.npy", mask)
    assert drive.parts_mean_density(tmp_path) == 4325.0
    assert drive.parts_mean_density(tmp_path / "missing") is None


def test_body_and_loads_estimates_the_mass_of_a_parts_material(tmp_path):
    import numpy as np
    root = tmp_path / "materials" / "parts" / "field"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"channels": ["youngs_modulus", "poisson_ratio", "density"]}))
    grid = np.zeros((1, 1, 2, 3)); grid[..., 2] = 1000.0
    np.save(root / "material_grid.npy", grid); np.save(root / "valid_mask.npy", np.ones((1, 1, 2), dtype=bool))
    summary = {"geometry": {"physical_volume": 0.5, "physical_extents": [1, 1, 1], "leaf_size_h": 0.01},
               "material": {"mode": "parts", "density": None}, "loads": {"applied": 1}}
    out = drive.body_and_loads(tmp_path, summary)
    assert out["body"]["mass"] == 500.0 and "mean density" in out["body"]["mass_note"]
    assert out["loads"]["self_weight_estimate"] == round(500 * 9.80665, 4) and out["loads"]["applied"] == 1
    uniform = drive.body_and_loads(tmp_path, {**summary, "material": {"mode": "uniform", "density": 700}})
    assert uniform["body"]["mass"] == 350.0 and "mass_note" not in uniform["body"]


def test_run_needs_a_case_brief_with_the_five_headings(tmp_path):
    path = tmp_path / "brief.md"
    assert drive.brief_missing(path) == "missing"
    path.write_text("# Use\nA load rests on its top for an hour.\n# Contacts\ntwo supports on the ground, held fully")
    assert "Load" in drive.brief_missing(path)
    path.write_text(tabled("## Use\nA load rests on its top. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
                    "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
                    "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    assert drive.brief_missing(path) is None
    path.write_text("## Use\nx\n## Contacts\ntwo supports on the ground, held fully\n## Load\n785 N on the top top\n"
                    "## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m, wood\n## Prediction\nmembers in compression, microns")
    assert "empty under 'Use'" == drive.brief_missing(path)


def test_brief_is_snapshotted_and_edits_are_reported(tmp_path):
    op = tmp_path / "operator"; op.mkdir()
    brief = ("## Use\nA load rests on its top. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports, held in all components\n"
             "## Load\n785 N on the top\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m, wood 1.2e10\n"
             "## Prediction\nmembers in compression, 0.1 MPa, 4 um")
    (op / "brief.md").write_text(brief)
    status = drive.brief_status(op)             # a body is made: the real lines are kept from here
    assert status["present"] and status["changed_since_first"] == []
    assert "0.1 MPa" in status["prediction"] and (tmp_path / "brief_snapshots" / "brief.first.md").exists()
    # Edits before the first solve are the brief being written, not changes.
    (op / "brief.md").write_text(brief.replace("two supports", "two flat supports").replace("4 um", "5 um"))
    assert drive.brief_status(op)["changed_since_first"] == []
    (op / "brief.md").write_text(brief + "\n## Prediction (generated)\nmembers 60 mm thick: 0.07 MPa, 3 um")
    drive.snapshot_brief_at_solve(op)
    status = drive.brief_status(op)
    assert status["changed_since_first"] == [] and "3 um" in status["prediction_generated"]
    assert status["prediction_generated_changed"] is False
    (op / "brief.md").write_text(brief + "\n## Prediction (generated)\nmembers 60 mm thick: 0.07 MPa, 30 um")
    assert drive.brief_status(op)["prediction_generated_changed"] is True
    (op / "brief.md").write_text(brief.replace("4 um", "40 um"))
    assert drive.brief_status(op)["changed_since_first"] == ["Prediction"]
    # The real lines are read as they stood before the body; an edit is reported, never used.
    (op / "brief.md").write_text(brief.replace("Real mass: 30 kg", "Real mass: 90 kg"))
    assert drive.brief_status(op)["real_lines_edited"] == ["Real mass"] and drive.real_mass(op) == 30.0
    assert drive.brief_status(tmp_path / "none") == {"present": False}


def test_named_values_that_match_no_region_are_refused_and_supports_are_printed():
    card = {"dirichlet": [{"name": "support"}], "neumann": [{"name": "top"}]}
    plan = {**PLAN, "values": {"load": {"force": [0, -20, 0]},
                               "dirichlet_values": {"other_support": {"components": [1], "value": 0.0}}}}
    with pytest.raises(drive.PlanError, match="other_support"):
        drive.plan_answer("bc_values", card, plan)
    plan = {**PLAN, "values": {"load": {"force": [0, -20, 0]}, "neumann_values": {"region_b": [0, -5, 0]}}}
    with pytest.raises(drive.PlanError, match="region_b"):
        drive.plan_answer("bc_values", card, plan)
    regions = [{"name": "support", "category": "dirichlet"}, {"name": "top", "category": "neumann"}]
    plan = {**PLAN, "values": {"load": {"force": [0, -20, 0]},
                               "dirichlet_values": {"support": {"components": [1], "value": 0.0}, "edge": {"components": [0]}}}}
    out = drive.planned_supports(plan, regions)
    assert out[0] == {"name": "support", "components": [1], "from": "plan.values.dirichlet_values"}
    assert out[1]["name"] == "edge" and "refuse" in out[1]["error"]
    assert drive.planned_supports(PLAN, regions)[0]["from"].startswith("default")


def test_brief_run_keeps_step_warnings():
    value = {"ok": True, "stop": "regions", "steps": [
        {"kind": "bc_input", "action": "preview_regions", "ok": True, "elapsed_s": 1.0,
         "notes": [], "warnings": ["the preview kept the previous regions: api_error"]}]}
    out = drive.brief_run(value)
    assert out["steps"][0]["warnings"] == ["the preview kept the previous regions: api_error"]
    assert out["step_warnings"] == ["the preview kept the previous regions: api_error"]


def test_judgement_needs_a_verdict_and_a_deviations_heading():
    ok = "The brief's case was run.\n## Deviations\nnone\n## Case\n..."
    assert drive.judgement_problem(ok, {"present": True}) is None
    assert "verdict" in drive.judgement_problem("## Case\nfine", {"present": True})
    assert "Deviations" in drive.judgement_problem("The brief's case was run.\n## Case\nfine", {"present": True})
    # a deviation written down is reported as found, never a wording trap on the verdict
    listed = "The brief's case was run.\n## Deviations\n- the load landed on a neighbouring face\n"
    assert drive.judgement_problem(listed, {"present": True}) is None
    honest = "The brief's case could not be run.\n## Deviations\n- the load landed on a neighbouring face\n"
    assert drive.judgement_problem(honest, {"present": True}) is None
    assert "Brief changes" in drive.judgement_problem(ok, {"present": True, "changes": "share moved"})
    # The brief's edits are the runner's to state in the report: no phrase is asked for.
    assert drive.judgement_problem(ok, {"present": True, "changed_since_first": ["Load"]}) is None


def test_blank_bodies_are_watertight_and_sized():
    box = drive.build_blank("box", size=[0.018, 0.024, 0.26])
    assert box.is_watertight and [round(e, 3) for e in box.extents] == [0.018, 0.024, 0.26]
    cyl = drive.build_blank("cylinder", radius=0.01, length=0.3, axis="y")
    assert cyl.is_watertight and round(cyl.extents[1], 3) == 0.3
    ell = drive.build_blank("extrude", profile="0,0 0.02,0 0.02,0.005 0.005,0.005 0.005,0.02 0,0.02",
                            length=0.3, axis="z")
    assert ell.is_watertight and round(ell.extents[2], 3) == 0.3
    assert abs(ell.volume - 0.3 * (0.02 * 0.005 + 0.015 * 0.005)) < 1e-9
    with pytest.raises(drive.PlanError, match="three points"):
        drive.build_blank("extrude", profile="0,0 1,0", length=0.1)
    with pytest.raises(drive.PlanError, match="box needs"):
        drive.build_blank("box")


def test_member_thickness_reads_axis_parallel_members_within_a_voxel():
    import trimesh
    for extents, true in (([0.26, 0.018, 0.05], 0.018), ([0.26, 0.018, 0.024], 0.018),
                          ([0.25, 0.007, 0.18], 0.007), ([0.25, 0.011, 0.18], 0.011),
                          ([1.7, 0.13, 1.1], 0.13)):
        read = drive.member_thickness(trimesh.creation.box(extents=extents))
        assert abs(read["p10"] - true) <= read["voxel"], (extents, read)
        assert read["under_two_voxels"] is False
    rod = drive.member_thickness(trimesh.creation.cylinder(radius=0.025, height=1.0, sections=64))
    assert abs(rod["p10"] - 0.05) <= rod["voxel"]
    # Under two voxels the reading is within a voxel, never several thick.
    thin = drive.member_thickness(trimesh.creation.box(extents=[0.26, 0.003, 0.05]))
    assert thin["p10"] <= 0.003 + thin["voxel"]
    # Inclined faces read thick, within the bias the printed note states (1.8x).
    tilted = trimesh.creation.box(extents=[0.25, 0.011, 0.18])
    tilted.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1]))
    assert 0.011 <= drive.member_thickness(tilted)["p10"] <= 1.8 * 0.011


def test_an_extruded_profile_takes_through_holes():
    import math
    rect = "0,0 0.3,0 0.3,0.12 0,0.12"
    one = drive.build_blank("extrude", profile=rect, length=0.008, axis="y", circles=["0.07,0.04,0.012"])
    assert one.is_watertight and abs(one.volume - 0.008 * (0.036 - math.pi * 1.44e-4)) < 2e-8
    many = drive.build_blank("extrude", profile=rect, length=0.008, axis="z",
                             circles=["0.05,0.06,0.01", "0.25,0.03,0.01"],
                             holes=["0.12,0.02 0.17,0.02 0.17,0.09 0.12,0.09"])
    assert many.is_watertight and abs(many.volume - 0.008 * (0.036 - 2 * math.pi * 1e-4 - 0.0035)) < 3e-8
    # The profile's coordinates land on the other two axes in x, y, z order, unmirrored:
    # a hole near the profile's origin stays near the origin of those axes.
    for axis, (a, b) in (("x", (1, 2)), ("y", (0, 2)), ("z", (0, 1))):
        body = drive.build_blank("extrude", profile=rect, length=0.05, axis=axis,
                                 circles=["0.05,0.03,0.01"])
        assert body.is_watertight and body.volume > 0
        assert abs(body.extents["xyz".index(axis)] - 0.05) < 1e-6
        assert abs(body.extents[a] - 0.3) < 1e-6 and abs(body.extents[b] - 0.12) < 1e-6
        bore = body.vertices[np.abs(body.vertex_normals[:, a]) + np.abs(body.vertex_normals[:, b]) > 0.9]
        near = bore[(np.abs(bore[:, a] - 0.05) < 0.012) & (np.abs(bore[:, b] - 0.03) < 0.012)]
        assert len(near) > 0, axis
    with pytest.raises(drive.PlanError, match="outside"):
        drive.build_blank("extrude", profile=rect, length=0.008, circles=["0.25,0.06,0.07"])
    with pytest.raises(drive.PlanError, match="cross"):
        drive.build_blank("extrude", profile=rect, length=0.01,
                          circles=["0.05,0.06,0.03", "0.09,0.06,0.03"])
    with pytest.raises(drive.PlanError, match="extruded profile"):
        drive.build_blank("box", size=[0.1, 0.1, 0.1], circles=["0.03,0.04,0.01"])


def test_verdicts_variant_and_secondary():
    base = "\n## Deviations\nnone\n"
    assert drive.judgement_problem("A variant of the brief's case was run. Brief changes: the other part carries it." + base,
                                   {"present": True, "changes": "part"}) is None
    assert "Brief changes" in drive.judgement_problem("A variant of the brief's case was run." + base, {"present": True})
    assert "variant" in drive.judgement_problem("The brief's case was run." + base, {"present": True, "changes": "part"})
    assert drive.judgement_problem("The governing load cannot be applied; a secondary case was run: the second load." + base,
                                   {"present": True}) is None


def _min_angles(mesh):
    tri = mesh.triangles
    a = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    b = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)

    def angle(x, y, z):
        return np.degrees(np.arccos(np.clip((y * y + z * z - x * x) / (2 * y * z), -1, 1)))
    return np.minimum(np.minimum(angle(a, b, c), angle(b, a, c)), angle(c, a, b))


# The real MeshLab remesh (about 10 s a body): this test is about what it makes.
@pytest.mark.slow
@pytest.mark.real_remesh
def test_a_built_body_has_well_shaped_faces_sized_from_its_own_dimensions():
    for kwargs in (dict(shape="box", size=[1.6, 0.13, 0.035]),
                   dict(shape="extrude", profile="0,0 0.3,0 0.3,0.12 0,0.12", length=0.008,
                        axis="y", circles=["0.07,0.04,0.012"]),
                   dict(shape="cylinder", radius=0.02, length=0.45, axis="x")):
        body = drive.build_blank(**kwargs)
        edge = body.metadata["edge"]
        extents = sorted(body.extents)
        assert body.is_watertight
        # The target edge: half a depth-7 leaf at most, and a sixth of the thinnest side,
        # unless that would pass about 80k faces.
        assert (edge <= max(extents) / 150 * (1 + 1e-6) and edge <= extents[0] / 6 * (1 + 1e-6)) \
            or len(body.faces) > 60_000
        angles = _min_angles(body)
        assert np.percentile(angles, 1) > 20 and angles.min() > 10, kwargs
        lengths = body.edges_unique_length
        assert 0.8 < np.median(lengths) / edge < 1.3 and lengths.max() < 2.0 * edge, kwargs
    # A contact two leaves wide on the long face catches dozens of faces with no gap.
    box = drive.build_blank("box", size=[1.6, 0.13, 0.035])
    leaf = 1.92 * 1.6 / 128
    centres = box.triangles_center
    patch = (centres[:, 0] < 2 * leaf) & (box.face_normals[:, 1] < -0.99)
    assert patch.sum() >= 40
    assert abs(box.area_faces[patch].sum() - 2 * leaf * 0.035) < 0.15 * 2 * leaf * 0.035


def test_cross_sections_give_exact_areas_and_second_moments():
    import math
    import trimesh
    bar = trimesh.creation.box(extents=[0.9, 0.03, 0.05])
    out = drive.cross_sections(bar)
    assert out["axis"] == "x" and out["across"] == ["y", "z"]
    piece = out["stations"][2]["pieces"][0]
    assert piece["area"] == pytest.approx(0.0015, rel=1e-6)
    # I[0] about the first across axis (y): ∫(z-z̄)² = 0.03·0.05³/12; I[1] = 0.05·0.03³/12
    assert piece["I"] == pytest.approx([0.03 * 0.05 ** 3 / 12, 0.05 * 0.03 ** 3 / 12], rel=1e-6)
    rod = drive.cross_sections(trimesh.creation.cylinder(radius=0.02, height=0.8, sections=256))
    piece = rod["stations"][2]["pieces"][0]
    assert piece["area"] == pytest.approx(math.pi * 0.02 ** 2, rel=1e-3)
    assert piece["I"][0] == pytest.approx(math.pi * 0.04 ** 4 / 64, rel=1e-3)
    hollow = drive.build_blank("extrude", profile="0,0 0.06,0 0.06,0.04 0,0.04", length=0.7, axis="z",
                             holes=["0.01,0.01 0.05,0.01 0.05,0.03 0.01,0.03"])
    piece = drive.cross_sections(hollow)["stations"][2]["pieces"][0]
    assert piece["area"] == pytest.approx(0.06 * 0.04 - 0.04 * 0.02, rel=1e-3)
    assert piece["I"] == pytest.approx([(0.06 * 0.04 ** 3 - 0.04 * 0.02 ** 3) / 12,
                                        (0.04 * 0.06 ** 3 - 0.02 * 0.04 ** 3) / 12], rel=1e-3)
    two = trimesh.util.concatenate([bar, bar.copy().apply_translation([0.0, 0.0, 0.2])])
    pieces = drive.cross_sections(two)["stations"][1]["pieces"]
    assert len(pieces) == 2 and abs(abs(pieces[0]["centroid"][1] - pieces[1]["centroid"][1]) - 0.2) < 1e-9
    scaled = drive.planned_sections(bar, 2.0)
    row = scaled["stations"][2]["pieces"][0]
    assert scaled["length_m"] == pytest.approx(1.8) and row["area_m2"] == pytest.approx(0.006, rel=1e-3)
    assert row["I_m4"]["about_y"] == pytest.approx(16 * 0.03 * 0.05 ** 3 / 12, rel=1e-3)
    assert row["length_over_size"] == [30.0, 18.0]


def test_the_blank_scale_entry_is_the_longest_extent(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest "
        "member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall\n## Prediction\nmembers in compression, 0.1 MPa, microns"))
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    drive.cmd_blank(argparse.Namespace(
        output_root=str(tmp_path), session_id="abc", shape="extrude", size=None, radius=None,
        length=0.009, axis="z", profile="0,0 0.25,0 0.25,0.11 0,0.11", hole=None, circle=None))
    scale = printed[-1]["plan"]["scale"]
    assert scale["up_axis"] == "longest" and abs(scale["physical_extent"] - 0.25) < 1e-6
    # the axis convention is printed with the body, as build_blank applied it
    axis = printed[-1]["axis"]
    assert "the length runs along z" in axis and "lie on x and y" in axis and "default" not in axis
    assert printed[-1]["extents_m"][2] == pytest.approx(0.009, abs=1e-6)


def test_the_stress_exclusion_is_a_physical_distance(tmp_path, monkeypatch):
    import trimesh
    (tmp_path / "geometry").mkdir()
    trimesh.creation.box(extents=[0.6, 0.04, 0.1]).export(tmp_path / "geometry" / "physical.glb")
    reach = inspect_session.exclusion_distance(tmp_path, 0.001)
    assert abs(reach - 0.04) <= 0.6 / 160          # the member thickness, not 3 leaves
    assert inspect_session.exclusion_distance(tmp_path, 0.001) == reach   # cached
    assert inspect_session.exclusion_distance(tmp_path, 0.02) == pytest.approx(0.06)  # 3 leaves floor


def test_two_depths_of_one_case_give_the_first_order_limit(tmp_path, monkeypatch):
    solve = tmp_path / "solve"
    config = ("Workflow effective config: use_mixed_precision=True, relative_tol=1.000e-02, "
              "remesh_depth={d}, remesh_init_depth=4, run_mode=forward\n")
    runs = {}
    for name, depth, extra in (("run-a", 7, ""), ("run-b", 8, ""), ("run-c", 8, "load_steps=3, "),
                               ("run-d", 7, "")):
        run = solve / name
        run.mkdir(parents=True)
        (run / "log.txt").write_text(config.format(d=depth).replace("run_mode", extra + "run_mode"))
        runs[name] = run
    peaks = {"run-a": 3.0e-5, "run-b": 2.0e-5, "run-c": 5.0e-5, "run-d": 9.0e-5}
    monkeypatch.setattr(inspect_session, "run_succeeded", lambda run: True)
    monkeypatch.setattr(inspect_session, "run_peak_displacement", lambda run: peaks[run.name])
    monkeypatch.setattr(inspect_session, "surface_stress_summary",
                        lambda run, *a: {"surface_von_mises": {"max_away_from_regions": 1.0}})
    order = [runs["run-a"], runs["run-d"], runs["run-c"], runs["run-b"]]
    monkeypatch.setattr(lib, "solve_runs", lambda session: order)
    monkeypatch.setattr(lib, "newest_run", lambda session: order[-1])
    # run-d solved another case (other regions, values or material); run-a the same one.
    lib.write_json(tmp_path / "operator" / "run_cases.json",
                   {"run-a": "case1", "run-c": "case1", "run-d": "case2"})
    (tmp_path / "case").mkdir()
    monkeypatch.setattr(inspect_session, "case_digest", lambda session: "case1")
    out = inspect_session.depth_comparison(tmp_path, runs["run-b"], {}, 0.001)
    assert lib.read_json(tmp_path / "operator" / "run_cases.json")["run-b"] == "case1"
    # run-c has other settings; run-d another case; run-a is the same case one depth coarser.
    assert out["depths"] == [7, 8] and out["runs"] == ["run-a", "run-b"]
    assert out["displacement_first_order_limit"] == pytest.approx(2 * 2.0e-5 - 3.0e-5)
    # A run whose case was never recorded is not paired.
    lib.write_json(tmp_path / "operator" / "run_cases.json", {"run-b": "case1"})
    assert inspect_session.depth_comparison(tmp_path, runs["run-b"], {}, 0.001) is None


def test_an_exclusion_that_swallows_the_body_falls_back_and_says_so(tmp_path, monkeypatch):
    points = np.column_stack([np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)])
    stress = np.zeros((40, 6))
    stress[:, 0] = np.linspace(1.0, 2.0, 40)
    run = tmp_path / "run"
    (run / "vtu_surface").mkdir(parents=True)
    (run / "vtu_surface" / "surface_t0000.vtu").write_text("")
    monkeypatch.setattr(lib, "read_vtu", lambda path, names=None: {"stress": stress, "Points": points})
    monkeypatch.setattr(inspect_session, "exclusion_distance", lambda session, h: 0.5)
    monkeypatch.setattr(inspect_session, "_away_from_regions",
                        lambda pts, session, resolved, reach: pts[:, 0] > (0.9 if reach > 0.1 else 0.2) + 1)
    out = inspect_session.surface_stress_summary(run, tmp_path, {"regions": []}, 0.01)
    assert "no stress away from the regions" in out["surface_von_mises"]["exclusion"]
    monkeypatch.setattr(inspect_session, "_away_from_regions",
                        lambda pts, session, resolved, reach: pts[:, 0] > (2.0 if reach > 0.1 else 0.5))
    out = inspect_session.surface_stress_summary(run, tmp_path, {"regions": []}, 0.01)
    assert "3 leaves" in out["surface_von_mises"]["exclusion"]
    assert out["surface_von_mises"]["max_away_from_regions"] == pytest.approx(2.0)


def test_a_changed_scale_stops_at_the_geometry_again(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    plan = {**PLAN, "scale": {**PLAN["scale"], "physical_extent": 0.2}}
    lib.write_json(session / "operator" / "run_sent.json",
                   {"geometry": drive._normalized(plan["geometry"]),
                    "scale": drive._normalized(PLAN["scale"])})
    state = {"card": {"kind": "bc_input", "description": ""}}
    printed = []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": state["card"], "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "_newest_card", lambda g, sid, kind: "checkpoint")

    def fork(graph, sid, checkpoint):
        state["card"] = {"kind": "geometry_physics"}
    monkeypatch.setattr(drive, "fork", fork)
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "resolved_plan", lambda session, plan: plan)

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        state["card"] = {"kind": "bc_input", "description": ""}
        return {"ok": True, "card": state["card"]}
    monkeypatch.setattr(drive, "_run_step", step)
    path = session / "operator" / "plan.json"
    path.write_text(json.dumps(plan))
    drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(path), full=True))
    assert printed[-1]["stop"] == "geometry"




def test_a_drawing_already_generated_from_comes_back_from_the_cache_without_a_drawing_stop(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    lib.write_json(session / "operator" / "run_counts.json", {"generation": 1})     # the one text attempt made
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    state = {"card": {"kind": "geometry_input", "description": ""}, "drawn": None}
    printed, sent = [], []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": state["card"], "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "_newest_card", lambda g, sid, kind: "checkpoint")

    def drawn_card():
        words, seed = state["drawn"]
        return {"kind": "geometry_input", "description": "", "drawn_image": str(tmp_path / "d.png"),
                "drawn_image_prompt": words, "image_seed": seed}

    def fork(graph, sid, checkpoint):
        state["card"] = drawn_card()
    monkeypatch.setattr(drive, "fork", fork)
    monkeypatch.setattr(drive, "join_paint_ahead", lambda session: None)
    monkeypatch.setattr(drive, "start_paint_ahead", lambda *a, **k: None)
    monkeypatch.setattr(drive, "preload", lambda *a, **k: None)
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "resolved_plan", lambda session, plan: plan)

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        sent.append(prepared.answer)
        if prepared.answer.get("action") == "draw_image":
            state["drawn"] = (prepared.answer["description"], prepared.answer["image_seed"])
            state["card"] = drawn_card()
        else:
            state["card"] = {"kind": "bc_input", "description": ""}
        return {"ok": True, "card": state["card"]}
    monkeypatch.setattr(drive, "_run_step", step)
    path = session / "operator" / "plan.json"

    def run(words, **crop):
        path.write_text(json.dumps({**PLAN, "geometry_routes": ["drawing"],
                                    "geometry": {"description": words, "draw": True, "image_seed": 0, **crop}}))
        drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(path), full=True))
        return printed[-1]
    first = run("a solid part")
    assert first["stop"] == "drawing", first.get("error")
    assert run("a solid part")["stop"] == "geometry"               # generated from it: accepted
    assert run("a tall solid part")["stop"] == "drawing"          # new words: a new drawing is looked at
    out = run("a solid part")                                     # back to the drawing generated from before
    assert out["stop"] == "geometry", out
    assert [a.get("action") or "generate" for a in sent[-2:]] == ["draw_image", "generate"]
    assert any("no drawing stop" in n for s in out["steps"] for n in s["notes"])
    # the same words with another drawing_crop: looked at again
    assert run("a tall solid part")["stop"] == "drawing"
    assert run("a solid part", drawing_crop="largest")["stop"] == "drawing"


def _held_session(tmp_path, regions, components, contacts=None, body=None):
    """A session folder with a built bar (or ``body``) and resolved held regions, for the restraint checks."""
    import trimesh
    session = tmp_path / "abc"
    (session / "geometry").mkdir(parents=True)
    (session / "boundary_conditions").mkdir()
    (session / "operator").mkdir()
    (session / "checkpoints.sqlite").write_text("")
    (body or trimesh.creation.box(extents=[0.96, 0.09, 0.08])).export(session / "geometry" / "physical.glb")
    lib.write_json(session / "boundary_conditions" / "resolved.json",
                   {"regions": [{"category": "dirichlet", **r} for r in regions]})
    lib.write_json(session / "boundary_conditions" / "proposal.json",
                   {"dirichlet": [{"name": n, "components": c} for n, c in components.items()]})
    lib.write_json(session / "operator" / "declared.json", {"contacts": contacts or {}})
    lib.write_json(session / "operator" / "prediction_history.json", [{"run": "r", "solved": {}}])   # a solve ran
    return session


LEAF7 = 1.92 * 0.96 / 128          # the test bar's depth-7 leaf


def _side(name, x0, leaves):
    return {"name": name, "mean_normal": [0.0, -0.98, 0.0],
            "bounds": [[x0, 0.0, 0.0], [x0 + leaves * LEAF7, 0.0, 0.08]]}


def test_judge_accepts_a_simple_support_held_on_its_normal_component(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    ran = "The brief's case was run.\n## Deviations\nnone\n"

    def judge(session, text):
        (session / "operator" / "report.md").write_text("## Engineer's judgement\n\n_(fill in)_\n")
        (session / "operator" / "j.md").write_text(text)
        drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                           file=str(session / "operator" / "j.md"), text=None))
        return printed[-1]

    session = _held_session(tmp_path, [_side("left", 0.0, 4), _side("right", 0.6, 4)],
                            {"left": [1], "right": [1]},
                            {"ends": {"transmits": "force only", "regions": ["left", "right"]}})
    lib.write_json(session / "operator" / "prediction_check.json", {"displacement_ratio": 0.9})
    # held on its normal component only: the real support, accepted as it is
    assert judge(session, ran)["ok"]
    assert judge(session, "The brief's case could not be run.\n## Deviations\n- the holds clamp the ends\n")["ok"]


def test_the_verdict_does_not_hang_on_the_prediction_and_no_prose_moves_the_agreement(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = _held_session(tmp_path, [], {})
    (session / "operator" / "report.md").write_text("## Engineer's judgement\n\n_(fill in)_\n")
    entry = {"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "{}", "load_N": 10.0,
             "prediction": {"max_displacement": 1e-4, "max_stress": 2e6},
             "solved": {"max_displacement": 2.6e-4, "max_stress": 2.2e6}}
    lib.write_json(session / "operator" / "prediction_history.json", [entry])

    def judge(text):
        (session / "operator" / "j.md").write_text(text)
        drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                           file=str(session / "operator" / "j.md"), text=None))
        return printed[-1]
    ran = "The brief's case was run.\n## Deviations\nnone\n"
    out = judge(ran)
    assert out["ok"] and out["agreement"]["max_displacement"]["statement"] == "disagrees"
    assert out["agreement"]["max_stress"]["statement"] == "agrees"
    # A recomputation written after the result changes nothing the runner states.
    out = judge(ran + "Recomputed max_displacement: 2.0e-4 from `I_m4.about_y`\n")
    assert out["agreement"]["max_displacement"]["ratio"] == 2.6
    # A prediction declared after the first solve of the same body and ends is not the reference;
    # a real change of load scales it.
    later = {**entry, "run": "r2", "load_N": 20.0, "prediction": {"max_displacement": 5e-4, "max_stress": 4e6},
             "solved": {"max_displacement": 5.2e-4, "max_stress": 4.4e6}}
    agreed = drive.agreement([entry, later])
    assert agreed["reference_run"] == "r1" and agreed["max_displacement"]["ratio"] == 2.6
    assert drive.agreement([entry, {**later, "body": {"sha": "c", "size": 1.0}}])["reference_run"] == "r2"


def test_the_plan_declares_each_support_rotation_and_its_prediction(tmp_path):
    card = {"dirichlet": [{"name": "support"}, {"name": "other"}], "neumann": [{"name": "top"}]}
    with pytest.raises(drive.PlanError, match="support regions \\['other'\\] are in no contact"):
        drive.plan_answer("bc_values", card, PLAN)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**PLAN, "contacts": {"base": {"transmits": "pinned"}}}))
    with pytest.raises(drive.PlanError, match="keyed by the brief's"):
        drive.load_plan(path)
    path.write_text(json.dumps({**PLAN, "prediction": {"max_displacement": 1e-4, "max_stress": 2e6}}))
    assert drive.load_plan(path)["prediction"]["max_stress"] == 2e6
    solve = {"checks": {"max_displacement": 2e-4, "surface_von_mises": {"max_away_from_regions": 1e6},
                        "depths": {"displacement_first_order_limit": 3e-4}}}
    (tmp_path / "operator").mkdir(exist_ok=True)
    out = drive.prediction_check(tmp_path, solve, {"max_displacement": 1e-4, "max_stress": 2e6})
    # Always the solve at the plan's depth: checking convergence never moves the ratio.
    assert out["displacement"] == 2e-4 and out["displacement_ratio"] == 2.0
    assert out["stress_ratio"] == 0.5 and "plan's depth" in out["compared"]
    # No prediction is demanded: the runner's own stands where the plan declares none
    assert drive.prediction_required({**PLAN, "prediction": None}) is None


def test_a_body_built_at_the_real_section_is_predicted_once(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[0.7, 1.0, 0.4]))
    trimesh.creation.box(extents=[0.7, 1.0, 0.4]).export(session / "geometry" / "original.glb")
    # the body is built with `blank` (its staged name), so its route is derived "real" or "section"
    lib.write_json(session / "session.json", {"geometry_mode": "surface",
                                              "input_asset": {"source_name": "blank_extrude_0a1b2c3d.glb"}})
    (session / "operator" / "brief.md").write_text(
        "## Size and material\nReal mass: 470 kg\nReal thinnest member: 0.2 m\n"
        "Real section: 0.4 m deep, I 2.1e-3 m4, A 0.16 m2\nE 3e10 Pa, nu 0.2, rho 2400 kg/m3\n")
    mass = {"planned_mass": 470.4}
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan, *a: dict(mass))
    built = {**PLAN, "scale": {**PLAN["scale"], "physical_extent": 1.0},
             "material": {"youngs_modulus": 3e10, "poisson_ratio": 0.2, "density": 2400},
             "body_simplification": {"kept": "the load path as one profile", "left_out": "an inner part"}}
    # built at the brief's real section: the prediction is the brief's case, no brief_case demanded
    assert drive.body_route(session, built)["route"] == "real"
    assert drive.built_at_real(session, built)
    assert drive.prediction_required(built, session) is None
    mass["planned_mass"] = 300.0
    assert not drive.built_at_real(session, built)                  # a third of the mass off: not the real one
    mass["planned_mass"] = 470.4
    # a density the brief never wrote, tuned to land the mass in the window, does not count
    tuned = {**built, "material": {**built["material"], "density": 1530}}
    assert not drive.built_at_real(session, tuned)
    # no density line in the brief: the mass cannot be trusted, both are declared
    text = (session / "operator" / "brief.md").read_text()
    (session / "operator" / "brief.md").write_text(text.replace("rho 2400 kg/m3", "concrete"))
    assert not drive.built_at_real(session, built)
    (session / "operator" / "brief.md").write_text(text)
    # a real member under 2 leaves makes no stand-in: the body built at it is still the real one
    thin = {**built, "scale": {**built["scale"], "physical_extent": 100.0}}
    assert drive.member_unresolvable(session, thin) and drive.built_at_real(session, thin)
    assert drive.body_route(session, thin)["route"] == "real"
    # a body built elsewhere (an upload) always declares both
    lib.write_json(session / "session.json", {"geometry_mode": "surface", "input_asset": {"source_name": "part.stl"}})
    assert drive.body_route(session, built)["route"] == "upload"
    assert drive.prediction_required(built, session) is None        # no prediction is demanded


def test_the_brief_contact_names_are_said_before_the_contacts_are_written(tmp_path):
    op = tmp_path / "operator"
    op.mkdir()
    (op / "brief.md").write_text('## Contacts\nThe body is held at one end.\n'
                                 '{"hold at the end": "force and moment"}\n')
    note = drive.contacts_keys_note(tmp_path, {**PLAN, "contacts": {}})
    assert '"hold at the end" (force and moment)' in note and "not names" not in note
    wrong = drive.contacts_keys_note(tmp_path, {**PLAN, "contacts": {"hold": {"transmits": "force and moment"}}})
    assert "not names of the brief: ['hold']" in wrong


def test_the_plan_rotation_is_the_one_the_brief_declared_first(tmp_path):
    op = tmp_path / "operator"
    op.mkdir()
    brief = ("## Use\nA load rests on its top. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\nleft end on a ledge, right end on a ledge\n"
             '{"left": "force only", "right": "force only"}\n## Load\n785 N\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\nx\n'
             "## Prediction\nbending")
    (op / "brief.md").write_text(brief)
    drive.brief_status(op)                                     # the first run's snapshot
    declared = drive.brief_contacts(op)
    assert declared == {"left": "force only", "right": "force only"}
    # Keyed by the brief's contacts, so it can be written before any region exists.
    ok = {"left": {"transmits": "force only"}, "right": {"transmits": "force only"}}
    assert drive.contacts_problem(ok, declared) is None
    flipped = {**ok, "right": {"transmits": "force and moment"}}
    assert "the brief declares 'force only'" in drive.contacts_problem(flipped, declared)
    assert "not a contact of the brief" in drive.contacts_problem({**ok, "middle": {"transmits": "force only"}}, declared)
    assert "missing from plan.contacts" in drive.contacts_problem({"left": {"transmits": "force only"}}, declared)
    # Once regions exist, every support region is bound to one contact.
    bound = {"left": {"transmits": "force only", "regions": ["a"]}, "right": {"transmits": "force only", "regions": ["b"]}}
    assert drive.contacts_problem(bound, declared, ["a", "b"]) is None
    assert "support region c is in no contact" in drive.contacts_problem(bound, declared, ["a", "b", "c"])
    # An edit after the first run does not change the declaration the plan is held to.
    (op / "brief.md").write_text(brief.replace('"right": "force only"', '"right": "force and moment"'))
    assert drive.brief_contacts(op) == declared
    assert "declares no contact" in drive.contacts_problem(ok, drive.brief_contacts(tmp_path / "other" / "operator"))


def test_two_depths_from_a_finished_solve_solves_instead_of_finishing(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    plan = {**PLAN, "settings": {"remesh_octree_depth": 8}}
    from src.agents_wrapper.fem_workflow.models import ShapeOptSettings
    solved = {**ShapeOptSettings().model_dump(mode="json"), "use_mixed_precision": True,
              "remesh_octree_depth": 8}
    state = {"card": settings_card(solve_complete=True, defaults=dict(solved))}
    sent, printed = [], []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": state["card"], "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "resolved_plan", lambda session, plan: plan)

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        sent.append(prepared.answer.get("remesh_octree_depth", prepared.answer.get("action")))
        state["card"] = settings_card(solve_complete=True, defaults=dict(prepared.answer))
        return {"ok": True, "card": state["card"]}
    monkeypatch.setattr(drive, "_run_step", step)
    path = session / "operator" / "plan.json"
    path.write_text(json.dumps(plan))

    def run(two):
        drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(path),
                                         full=True, two_depths=two))
        return printed[-1]
    # One solve at the plan's depth is the result: two depths only after a failed solve.
    refused = run(True)
    assert refused["stop"] == "error" and "only after a solve" in refused["error"] and sent == []
    lib.write_json(session / "operator" / "solve_failed.json", {"time": 0})
    assert run(True)["stop"] == "solve" and sent == [7, 8]
    # Solved at both depths: the next run finishes, with or without the flag.
    assert run(True)["stop"] == "finished" and sent[-1] == "finish"


def test_two_depths_solves_one_depth_coarser_first(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    state = {"card": settings_card()}
    sent, printed = [], []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": state["card"], "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "resolved_plan", lambda session, plan: plan)

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        sent.append(prepared.answer["remesh_octree_depth"])
        state["card"] = settings_card(solve_complete=True, defaults=dict(prepared.answer))
        return {"ok": True, "card": state["card"]}
    monkeypatch.setattr(drive, "_run_step", step)
    path = session / "operator" / "plan.json"
    path.write_text(json.dumps({**PLAN, "settings": {"remesh_octree_depth": 8}}))
    lib.write_json(session / "operator" / "solve_failed.json", {"time": 0})    # two depths follow a failed solve
    drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(path),
                                     full=True, two_depths=True))
    # The coarser solve checks convergence: the pair costs one solve attempt.
    assert sent == [7, 8] and printed[-1]["stop"] == "solve" and printed[-1]["attempts"] == {"solve": 1}


SYNTHESIZED = {
    "dirichlet": [{"name": "left", "selector": {"type": "lowest", "params": {"axis": "x", "fraction": 0.05}},
                   "value": 0.0, "components": None}],
    "neumann": [{"name": "top", "selector": {"type": "highest", "params": {"axis": "y", "fraction": 0.02}},
                 "traction": [0.0, -1.0, 0.0]}],
}
WIDER = {"type": "lowest", "params": {"axis": "x", "fraction": 0.08}}


def test_a_revision_swaps_selectors_into_the_synthesized_regions_only(tmp_path):
    (tmp_path / "boundary_conditions").mkdir()
    with pytest.raises(drive.PlanError, match="preview the regions first"):
        drive.revision_answer(tmp_path, {"left": WIDER})
    lib.write_json(tmp_path / "boundary_conditions" / "synthesized.json", SYNTHESIZED)
    answer = drive.revision_answer(tmp_path, {"left": WIDER})
    assert answer["action"] == "revise_regions"
    assert answer["proposal"]["dirichlet"][0]["selector"] == WIDER
    assert answer["proposal"]["neumann"] == SYNTHESIZED["neumann"]       # untouched
    with pytest.raises(drive.PlanError, match="the synthesized regions are"):
        drive.revision_answer(tmp_path, {"right": WIDER})


def test_a_revision_of_a_region_a_reword_dropped_is_left_out_with_how_to_clear_it(tmp_path):
    (tmp_path / "boundary_conditions").mkdir()
    old = {"dirichlet": [{**SYNTHESIZED["dirichlet"][0], "name": "old_hold"}], "neumann": SYNTHESIZED["neumann"]}
    lib.write_json(tmp_path / "boundary_conditions" / "synthesized.json", old)
    assert drive.live_revisions(tmp_path, {"old_hold": WIDER}) == ({"old_hold": WIDER}, None)
    # reworded: a new synthesis without that region
    lib.write_json(tmp_path / "boundary_conditions" / "synthesized.json", SYNTHESIZED)
    live, said = drive.live_revisions(tmp_path, {"old_hold": WIDER, "left": WIDER})
    assert live == {"left": WIDER}
    assert "['old_hold']" in said and '{"region_revisions": {"old_hold": null}}' in said
    assert drive.revision_answer(tmp_path, live)["proposal"]["dirichlet"][0]["selector"] == WIDER
    # a name no synthesis made is a misspelt region: refused, with how to remove it
    assert drive.live_revisions(tmp_path, {"lfet": WIDER}) == ({"lfet": WIDER}, None)
    with pytest.raises(drive.PlanError, match='"lfet": null'):
        drive.revision_answer(tmp_path, {"lfet": WIDER})
    # the null patch clears it; an empty object does not (a merge patch merges)
    plan = {"region_revisions": {"old_hold": WIDER}}
    assert drive.merge_patch(plan, {"region_revisions": {}}) == plan
    assert drive.merge_patch(plan, json.loads('{"region_revisions": {"old_hold": null}}')) == {"region_revisions": {}}


def test_the_plan_validates_revisions_and_simplifications(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**PLAN, "region_revisions": {"left": {"type": "no_such_selector"}}}))
    with pytest.raises(drive.PlanError, match="not a selector"):
        drive.load_plan(path)
    free = {"contact": "base", "transmits": "force only", "simplified_to": "pinned"}
    path.write_text(json.dumps({**PLAN, "contacts": {"support": free}}))
    with pytest.raises(drive.PlanError, match="simplified_to is one of"):
        drive.load_plan(path)
    restrained = {"contact": "base", "transmits": "force and moment", "simplified_to": "clamped"}
    path.write_text(json.dumps({**PLAN, "contacts": {"support": restrained}}))
    with pytest.raises(drive.PlanError, match="idealises a contact that transmits 'force only'"):
        drive.load_plan(path)
    # Statics without contact: no yielding support, no bed.
    path.write_text(json.dumps({**PLAN, "contacts": {"support": {"transmits": "yielding", "regions": []}}}))
    with pytest.raises(drive.PlanError, match="keyed by the brief's"):
        drive.load_plan(path)
    # the route is the runner's (derived from where the body came from): a declared one is refused
    path.write_text(json.dumps({**PLAN, "body_simplification": {"route": "a generator, tried", "kept": "k",
                                                                "left_out": "l"}}))
    with pytest.raises(drive.PlanError, match="route is derived by the runner"):
        drive.load_plan(path)
    # every Load path member not absent is kept: "kept" and "left_out" are optional words
    path.write_text(json.dumps({**PLAN, "body_simplification": {"kept": "k"}}))
    assert drive.load_plan(path)["body_simplification"] == {"kept": "k"}
    path.write_text(json.dumps({**PLAN, "body_simplification": {"absent": "base"}}))
    with pytest.raises(drive.PlanError, match='"absent"'):
        drive.load_plan(path)
    path.write_text(json.dumps({**PLAN, "region_revisions": {"left": WIDER},
                                "contacts": {"support": {**free, "simplified_to": "clamped"}}}))
    assert drive.load_plan(path)["region_revisions"]["left"] == WIDER


def test_run_applies_the_plans_revision_after_the_synthesis(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "boundary_conditions").mkdir()
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    lib.write_json(session / "boundary_conditions" / "synthesized.json", SYNTHESIZED)
    card = {"kind": "bc_input", "description": PLAN["regions"], "has_segmentation": True,
            "has_regions": True, "can_revise": True, "segmentation": {}}
    state = {"card": card}
    sent, printed = [], []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": state["card"], "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "resolved_plan", lambda session, plan: plan)
    monkeypatch.setattr(drive, "_newest_card", lambda g, sid, kind: "checkpoint")

    def fork(graph, sid, checkpoint):
        state["card"] = card
    monkeypatch.setattr(drive, "fork", fork)

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        sent.append(prepared.answer)
        state["card"] = card if prepared.answer.get("action") == "revise_regions" else {
            "kind": "bc_values", "dirichlet": [], "neumann": []}
        return {"ok": True, "card": state["card"]}
    monkeypatch.setattr(drive, "_run_step", step)
    lib.write_json(session / "operator" / "run_sent.json",
                   {"regions": drive._normalized(PLAN["regions"]), "segmentation": drive._normalized(None),
                    "region_revisions": drive._normalized({})})
    path = session / "operator" / "plan.json"

    def run(plan):
        path.write_text(json.dumps(plan))
        drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(path),
                                         full=True, two_depths=False))
        return printed[-1]
    revised = {**PLAN, "region_revisions": {"left": WIDER}}
    assert run(revised)["stop"] == "regions"
    assert sent[-1]["action"] == "revise_regions" and sent[-1]["proposal"]["dirichlet"][0]["selector"] == WIDER
    assert printed[-1]["attempts"] == {"revisions": 1}                    # its own limit, no synthesis
    # Unchanged, the next run approves what the stop showed.
    run(revised)
    assert sent[-1] == {"action": "approve_regions"}
    # Dropping the revision after the approval goes back and restores the synthesized selectors.
    out = run(PLAN)
    assert sent[-1]["action"] == "revise_regions" and out["stop"] == "regions"
    assert sent[-1]["proposal"]["dirichlet"][0]["selector"] == SYNTHESIZED["dirichlet"][0]["selector"]
    # A revision of a region an earlier synthesis made is left out, said, never refused.
    lib.write_json(session / "operator" / "synthesized_names.json", ["gone", "left", "top"])
    out = run({**PLAN, "region_revisions": {"gone": WIDER}})
    assert "region_revisions" not in str(out.get("error"))      # the next card's own check may stop it
    assert sent[-1] == {"action": "approve_regions"}
    assert any('"gone": null' in w for w in out["warnings"])


def test_a_simplified_case_needs_a_stated_limitation_and_a_clean_idealisation(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    # Two supports away from the member's ends, where the free row cannot be held.
    session = _held_session(tmp_path, [_side("left", -0.2, 4), _side("right", 0.1, 4)],
                            {"left": [0, 1, 2], "right": [1]})
    simplified = "A simplified case was run: the free supports held clamped.\n## Deviations\nnone\n"

    def judge(contacts, prediction, text=simplified, body=None):
        # a simplified body's route is where the plan's geometry takes it (CAD-Coder here), never a field
        lib.write_json(session / "operator" / "declared.json",
                       {"contacts": contacts, "prediction": prediction, "body_simplification": body,
                        "geometry": {"generator": "cadcoder", "description": "two members"} if body else None})
        (session / "operator" / "report.md").write_text("## Engineer's judgement\n\n_(fill in)_\n")
        (session / "operator" / "j.md").write_text(text)
        drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                           file=str(session / "operator" / "j.md"), text=None))
        return printed[-1]
    lib.write_json(session / "operator" / "prediction_check.json", {"displacement": 1e-4, "displacement_ratio": 1.1})
    simple = {"max_displacement": 1e-4, "max_stress": 1e6,
              "brief_case": {"max_displacement": 4.5e-4, "max_stress": 2e6}}
    restrained = {"a": {"transmits": "force and moment", "regions": ["left"]},
                  "b": {"transmits": "force and moment", "regions": ["right"]}}
    # Nothing the workflow cannot do: no simplification to claim.
    # an honest simplified verdict the runner cannot see is taken as written, never made "was run"
    assert judge(restrained, simple)["ok"]
    clamped = {"a": {"transmits": "force only", "regions": ["left"], "simplified_to": "clamped"},
               "b": {"transmits": "force only", "regions": ["right"], "simplified_to": "clamped"}}
    lib.write_json(session / "boundary_conditions" / "proposal.json",
                   {"dirichlet": [{"name": "left", "components": [0, 1, 2]},
                                  {"name": "right", "components": [0, 1, 2]}]})
    assert judge(clamped, {"max_displacement": 1e-4, "max_stress": 1e6})["ok"]     # no brief_case demanded
    out = judge(clamped, simple)
    assert out["ok"] and out["limitation"]["supports"]
    # Claiming the brief's case for a simplified run is refused.
    ran = "The brief's case was run.\n## Deviations\nnone\n"
    assert judge(clamped, simple, ran)["ok"] and "A simplified case was run: " in (session / "operator" / "report.md").read_text()
    # A simplified body needs the routes the brief listed tried, and the brief's own prediction.
    body = {"kept": "member A", "left_out": "member B"}
    assert "declare plan.geometry_routes" in judge(restrained, simple, body=body)["error"]
    lib.write_json(session / "operator" / "geometry_routes.json", ["cadcoder"])
    assert "['cadcoder'] not yet" in judge(restrained, simple, body=body)["error"]
    lib.write_json(session / "operator" / "run_counts.json", {"cad": 1})
    # the body is measured against the brief's load path: every member kept or declared absent
    assert "Load path: {" in judge(restrained, simple, body=body)["error"]
    (session / "operator" / "brief.md").write_text('## Size and material\nLoad path: {"member A": "series"}\n')
    # no brief_case is demanded: the runner's own stands for the brief's case
    assert judge(restrained, {"max_displacement": 1e-4, "max_stress": 1e6}, body=body)["ok"]
    (session / "operator" / "driver_log.jsonl").write_text("\n".join(json.dumps(
        {"kind": "geometry_input", "ok": ok, "answer": {"description": f"words {i}", "generator": "cadcoder"}})
        for i, ok in ((1, True), (2, False), (3, True))) + "\n")
    out = judge(restrained, simple, body=body)
    assert out["ok"] and out["limitation"]["body"]["kept"] == "member A"
    report = (session / "operator" / "report.md").read_text()
    assert "Attempts at the body" in report and "'words 2'" in report and "failed" in report


def test_end_conditions_are_what_contacts_transmit_not_their_words():
    ends = drive.end_conditions({"contacts": {"c": {"transmits": "force only", "regions": ["x"]}}})
    assert drive.end_conditions({"contacts": {"c": {"transmits": "force only", "regions": ["y"]}}}) == ends
    assert drive.end_conditions({"contacts": {"c": {"transmits": "force only", "regions": ["x"],
                                                    "simplified_to": "clamped"}}}) != ends


def test_the_report_carries_the_runners_statement(tmp_path):
    (tmp_path / "operator").mkdir()
    report = tmp_path / "operator" / "report.md"
    report.write_text("# Session\n\n## Engineer's judgement\n\n_(fill in)_\n")
    entry = {"run": "r1", "body": {"sha": "b1", "size": 0.96}, "ends": "{}", "load_N": 10.0,
             "prediction": {"max_displacement": 1e-4, "max_stress": 2e6},
             "solved": {"max_displacement": 3e-4, "max_stress": None}}
    rows = ["| ends | force only | left: end face, held [1, 2], 0.0 leaves, 0.72 |"]
    drive.add_history_to_report(report, [entry], rows, {"changed_since_first": ["Load"]})
    drive.add_history_to_report(report, [entry, {**entry, "run": "r2"}], rows, {})
    text = report.read_text()
    assert text.count(drive.RUNNER_SECTION) == 1 and "| r1 |" in text and "| r2 |" in text
    assert "the result **disagrees** with the prediction" in text and "| ends | force only |" in text
    assert text.index(drive.RUNNER_SECTION) < text.index("## Engineer's judgement")


def test_held_pieces_the_contacts_do_not_account_for_stop_the_run(tmp_path):
    row = {"name": "hold#1", "contact": "base", "end": "low x", "face": "side face", "member_axis": "x",
           "length_m": 0.05, "load_beside": 0.1, "load_reach_m": 0.02}
    contacts = {"base": {"transmits": "force and moment", "regions": ["hold"]}}
    assert drive.pieces_problem([row], contacts) is None
    two = [row, {**row, "name": "hold#2", "end": "high x", "load_beside": 0.9}]
    out = drive.pieces_problem(two, contacts)
    # one message: the count the runner sees and what each answer means
    assert "the runner sees 2 held pieces" in out and "where it expects 1" in out and "revise the region" in out
    # a load beside a hold is said at the regions stop, never a declaration to clear
    assert "that load goes straight into the support" in drive.held_pieces_warning(two, contacts)
    declared = {"base": {**contacts["base"], "pieces": 2}}
    # a declared count needs the brief's reason
    assert "without \"pieces_why\"" in drive.pieces_problem(two, declared)
    # a reason that only cites the generated surface is none
    for why in ("as the generated body shows", "the mesh splits it in two"):
        declared["base"]["pieces_why"] = why
        assert "citing the generated surface" in drive.pieces_problem(two, declared), why
    # physical reasons are taken, whatever words they use
    for why in ("two feet in plan view", "each region of the base is a separate foot", "a segment of the rim on "
                "each support", "the base resolves into two pads", "the body has four feet"):
        declared["base"]["pieces_why"] = why
        assert drive.pieces_problem(two, declared) is None, why
    # refused once, the phrase quoted; a rewritten reason is taken the next time (never a loop)
    (tmp_path / "operator").mkdir(exist_ok=True)
    declared["base"]["pieces_why"] = "as the generated body shows"
    assert "'generated body shows'" in drive.pieces_problem(two, declared, tmp_path)
    assert drive.pieces_problem(two, declared, tmp_path)                       # unchanged: still refused
    declared["base"]["pieces_why"] = "two pads, as the generated surface has them"
    assert drive.pieces_problem(two, declared, tmp_path) is None
    declared["base"]["pieces_why"] = "the base stands on two feet"
    assert drive.pieces_problem(two, declared) is None
    # refused once for its count: another count is never taken on the agent's word; the count seen,
    # declared after the refusal, is taken (and is no simplification)
    session = tmp_path / "pieces"
    (session / "operator").mkdir(parents=True)
    assert "the runner sees 2 held pieces" in drive.pieces_problem(two, contacts, session)
    four = {"base": {**contacts["base"], "pieces": 4, "pieces_why": "four pads"}}
    assert "another count is never taken" in drive.pieces_problem(two, four, session)
    pair = {"base": {**contacts["base"], "pieces": 2, "pieces_why": "two pads"}}
    assert drive.pieces_problem(two, pair, session) is None
    assert not any("pieces where the runner sees" in s for s in drive.runner_simplifications(session, {}, []))


def test_held_regions_measure_cover_and_distance_to_the_load_from_the_masks(tmp_path):
    import numpy as np
    import trimesh
    bar = trimesh.creation.box(extents=[0.96, 0.09, 0.08])
    centres, normals = bar.triangles_center, bar.face_normals
    bottom = normals[:, 1] < -0.9
    top = normals[:, 1] > 0.9
    session = _held_session(tmp_path, [{"name": "bed", "key": "bed", "mean_normal": [0, -1, 0],
                                        "bounds": [[-0.48, -0.045, -0.04], [0.48, -0.045, 0.04]]}],
                            {"bed": [1]}, body=bar)
    resolved = lib.read_json(session / "boundary_conditions" / "resolved.json")
    resolved["regions"].append({"category": "neumann", "name": "top", "key": "top"})
    lib.write_json(session / "boundary_conditions" / "resolved.json", resolved)
    np.savez(session / "boundary_conditions" / "resolved.npz", bed=bottom, top=top)
    rows = drive.held_regions(session, {"bed": {"components": [1]}},
                              {"ground": {"transmits": "force and moment", "regions": ["bed"]}})
    # The whole top load lies within 1.5 member thicknesses of the bed: a short path, declared.
    assert rows[0]["load_beside"] == 1.0
    assert rows[0]["contact"] == "ground" and rows[0]["transmits"] == "force and moment"


def test_a_stand_in_keeps_the_real_mass_within_two(tmp_path, monkeypatch):
    op = tmp_path / "operator"
    op.mkdir()
    (op / "brief.md").write_text("## Size and material\nReal mass: 10 kg\n")
    plan = {**PLAN, "body_simplification": {"route": "triposg", "kept": "k", "left_out": "l"}}
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 53.0})
    assert "5.30 x the real object's 10 kg" in drive.stand_in_problem(tmp_path, plan)
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 12.0})
    assert drive.stand_in_problem(tmp_path, plan) is None
    assert drive.real_mass_in("Real mass: 2.5e3 kg, decided") == 2500.0


def test_hunyuan_is_one_ladder_step_from_a_drawing_with_its_own_limit():
    card = {"drawn_image": "/x.png", "drawn_image_prompt": "a thing", "image_seed": 1}
    plan = {**PLAN, "geometry": {"description": "a thing", "generator": "hunyuan3d", "draw": True, "image_seed": 1}}
    answer, stage, stop = drive.plan_answer("geometry_input", card, plan)
    assert answer["use_drawn_image"] and stage == "hunyuan" and stop == "geometry"
    assert drive.attempt_stage("geometry_input", answer) == "hunyuan" and drive.RUN_LIMITS["hunyuan"] == 2
    # A drawing made for it is one of its own attempts, not a TripoSG drawing (that limit may be
    # used up by then): counted when drawn, rejected or not (hunyuan_charge).
    draw, stage, stop = drive.plan_answer("geometry_input", {}, plan)
    assert draw["action"] == "draw_image" and stage == "hunyuan" and stop == "drawing"
    with pytest.raises(drive.PlanError, match="Hunyuan3D takes an image"):
        drive.plan_answer("geometry_input", {}, {**PLAN, "geometry": {"description": "a thing",
                                                                       "generator": "hunyuan3d"}})
    # The user's image, as well as a drawing: the same step and limit.
    user = {**PLAN, "geometry": {"description": "a thing", "generator": "hunyuan3d",
                                 "asset_path": "/x.png", "asset_kind": "image"}}
    assert drive.plan_answer("geometry_input", {}, user)[1] == "hunyuan"


def test_a_resegmentation_with_the_same_words_has_its_own_limit(tmp_path):
    card = {"has_segmentation": True, "has_regions": True, "description": PLAN["regions"],
            "segmentation": {"method": "partfield", "num_clusters": 5}}
    plan = {**PLAN, "segmentation": {"method": "partfield", "num_clusters": 8}}
    # The new segments are made and named first, then previewed.
    assert drive.plan_answer("bc_input", card, plan)[0]["action"] == "preview_segmentation"
    (tmp_path / "operator").mkdir()
    lib.write_json(tmp_path / "operator" / "last_preview.json",
                   {"words": PLAN["regions"].strip(),
                    "segmentation": drive._normalized({"method": "partfield", "num_clusters": 5})})
    assert drive.preview_stage(tmp_path, plan) == "segmentation"
    assert drive.preview_stage(tmp_path, {**plan, "regions": "other words"}) == "regions"


def test_the_body_is_the_surface_the_geometry_card_now_uses(tmp_path):
    (tmp_path / "geometry").mkdir()
    for name in ("generated.glb", "original.glb"):
        (tmp_path / "geometry" / name).write_text(name)
    lib.write_json(tmp_path / "session.json", {"geometry_mode": "surface"})
    assert drive.body_source(tmp_path).name == "original.glb"
    lib.write_json(tmp_path / "session.json", {"geometry_mode": "text"})
    assert drive.body_source(tmp_path).name == "generated.glb"


def test_the_regions_stop_shows_where_each_load_acts(tmp_path):
    import trimesh
    session = _held_session(tmp_path, [], {})
    bar = trimesh.creation.box(extents=[0.96, 0.09, 0.08]).subdivide_to_size(0.02)
    bar.apply_translation(-bar.bounds[0])
    bar.export(session / "geometry" / "physical.glb")
    bar = trimesh.load(session / "geometry" / "physical.glb", force="mesh", process=False)
    centres = bar.triangles_center
    end = bar.face_normals[:, 0] < -0.99
    # A load on part of one face only: its point lies off the section centroid in y.
    tip = (bar.face_normals[:, 0] > 0.99) & (centres[:, 1] > 0.045)
    np.savez(session / "boundary_conditions" / "resolved.npz", hold=end, pull=tip)
    lib.write_json(session / "boundary_conditions" / "resolved.json",
                   {"regions": [{"category": "dirichlet", "name": "hold", "key": "hold"},
                                {"category": "neumann", "name": "pull", "key": "pull"}]})
    rows = drive.load_resultants(session, [{"region": "pull", "force_N": [-1000.0, 0.0, 0.0]}])
    row = rows[0]
    assert row["point_m"][0] == pytest.approx(0.96, abs=1e-3)
    assert row["offset_from_section_centroid_m"]["y"] == pytest.approx(0.0225, abs=2e-3)
    # A force along x with a y offset of 22.5 mm gives 22.5 N·m about z.
    assert row["moment_about_section_centroid_Nm"][2] == pytest.approx(22.5, abs=2.0)


def test_cad_parts_are_read_from_the_revision_and_pieces_counted():
    text = ("For the first part, create a new coordinate system with Euler angles set to zero and a "
            "translation vector of (0.0, 0.0, 0.0). Draw a two-dimensional sketch on the first face: a "
            "rectangle of four lines from (0.0, 0.0) to (0.3, 0.0), to (0.3, 0.2), to (0.0, 0.2) and back "
            "to (0.0, 0.0). Extrude the sketch 0.1 units along the normal direction to create a new solid "
            "body. For the second part, create a new coordinate system with Euler angles set to zero and a "
            "translation vector of (0.05, 0.05, 0.1). Draw a two-dimensional sketch on the first face: a "
            "circle centered at (0.05, 0.05) with a radius of 0.02. Extrude the sketch 0.15 units in the "
            "opposite direction of the normal to remove material from the existing body.")
    parts = drive.cad_parts(text)
    assert parts[0]["height"] == [0.0, 0.1] and parts[0]["operation"] == "new"
    assert parts[0]["sketch_x"] == [0.0, 0.3] and parts[0]["sketch_y"] == [0.0, 0.2]
    assert parts[1]["operation"] == "cut" and parts[1]["direction"] == "opposite"
    assert parts[1]["height"] == [-0.05, 0.1] and parts[1]["sketch_x"] == [0.08, 0.12]
    import trimesh
    two = trimesh.util.concatenate([trimesh.creation.box(extents=[1, 1, 1]),
                                    trimesh.creation.box(extents=[1, 1, 1]).apply_translation([3, 0, 0])])
    assert drive.connected_solids(two) == 2




def test_motion_of_an_older_solve_is_not_judged_against_newer_regions(tmp_path, monkeypatch):
    import os
    import time
    session = tmp_path / "abc"
    run = session / "solve" / "api_fem_forward-1"
    run.mkdir(parents=True)
    (run / "log.txt").write_text(FORWARD_LOG)
    (session / "boundary_conditions").mkdir()
    lib.write_json(session / "boundary_conditions" / "proposal.json",
                   {"dirichlet": [{"name": "held", "components": [0, 1, 2]}]})
    (session / "boundary_conditions" / "resolved.json").write_text("{}")
    motion = [{"name": "held", "category": "dirichlet", "mean": 0.99, "max": 0.99,
               "held_mean": 0.99, "held_max": 0.99}]
    monkeypatch.setattr(inspect_session, "region_motion", lambda *a, **k: motion)
    monkeypatch.setattr(inspect_session, "surface_stress_summary", lambda *a, **k: {})
    old = time.time() - 100
    os.utime(run / "log.txt", (old, old))           # the regions were previewed after the solve
    warnings: list[str] = []
    solve = inspect_session.solve_summary(session, run, [1, 1, 1], 0.0, "uniform", {}, 0.01, None, warnings)
    assert "region_displacement" not in solve and "changed since" in solve["region_displacement_note"]
    assert not any("still moves" in w for w in warnings)


def _documented_plans():
    """Every plan example in the reading path (SKILL.md and references): whole plans, and plan
    sections shown on their own (``"<plan key>": …``), placeholders filled with a number."""
    import re
    skill = SCRIPTS.parent
    found = []
    for path in [skill / "SKILL.md", *sorted((skill / "references").rglob("*.md"))]:
        for block in re.findall(r"```json\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.S):
            text = re.sub(r"\s*//[^\n]*", "", block)
            text = re.sub(r'(?<!["\w])<[^<>"\n]*>(?!")', "1", text).strip()
            key = re.match(r'^"(\w+)"\s*:', text)
            if key and key.group(1) in drive.PLAN_KEYS:
                found.append((path.relative_to(skill).as_posix(), {**PLAN, **json.loads("{" + text + "}")}))
            elif text.startswith("{") and '"regions"' in text and '"geometry"' in text:
                found.append((path.relative_to(skill).as_posix(), json.loads(text)))
    return found


def test_every_documented_plan_passes_the_runners_own_validation(tmp_path):
    plans = _documented_plans()
    # plan.md keeps the field rules; the one skeleton is SKILL.md's (read first)
    assert {name for name, _ in plans} >= {"SKILL.md", "references/materials.md"}
    for k, (name, plan) in enumerate(plans):
        path = tmp_path / f"plan{k}.json"
        path.write_text(json.dumps(plan))
        drive.load_plan(path)        # a documented format the runner refuses fails here


def _bar_regions(length=1.0, s=0.05):
    """A bar along x from 0 to length, 0.06 deep: two bearing patches underneath and a top load."""
    return [{"name": "top", "category": "neumann", "centroid": [length / 2, 0.06, 0.03], "mean_normal": [0, 1, 0]},
            {"name": "lb", "category": "neumann", "centroid": [s / 2, 0.0, 0.03], "mean_normal": [0, -1, 0]},
            {"name": "rb", "category": "neumann", "centroid": [length - s / 2, 0.0, 0.03], "mean_normal": [0, -1, 0]},
            {"name": "end", "category": "dirichlet", "centroid": [length, 0.03, 0.03], "mean_normal": [1, 0, 0]}]


def test_a_pushing_support_is_a_load_statics_fixes_or_a_hold_on_its_components(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    regions = _bar_regions()
    contacts = {"left": {"transmits": "force only", "regions": ["lb"]},
                "right": {"transmits": "force only", "regions": ["rb"]}}
    plan = {**PLAN, "contacts": contacts, "values": {"acceleration": [0, 0, 0]},
            "reference_hold": {"regions": ["end"]}}

    def planned(left, right):
        return [{"region": "top", "force_N": [0, -1000, 0]}, {"region": "lb", "force_N": [0, left, 0]},
                {"region": "rb", "force_N": [0, right, 0]}]
    result = drive.statics(session, plan, regions, planned(500, 500))
    assert result["determinate"] and [r["force_N"][1] for r in result["reactions"]] == [500.0, 500.0]
    assert drive.statics_problem(result) is None and drive.statics_note(result) is None
    # the plan's own values do not matter: the runner applies statics' reactions
    assert drive.statics_problem(drive.statics(session, plan, regions, planned(700, 300))) is None
    # A third pushing contact along the same line: statics does not fix them alone; reported, not refused
    three = _bar_regions() + [{"name": "mb", "category": "neumann", "centroid": [0.5, 0.0, 0.03],
                               "mean_normal": [0, -1, 0]}]
    plan3 = {**plan, "contacts": {**contacts, "mid": {"transmits": "force only", "regions": ["mb"]}}}
    result3 = drive.statics(session, plan3, three, planned(500, 500) + [{"region": "mb", "force_N": [0, 0, 0]}])
    assert drive.statics_problem(result3) is None
    assert "3 simple support(s) as loads" in drive.statics_note(result3) and "least-squares" in drive.statics_note(result3)
    # A pushing contact held on some components is the real support; on every one, clamped and marked
    row = {"name": "lb", "region": "lb", "transmits": "force only", "held": [1], "simplified_to": None}
    assert drive.support_problem([row]) is None
    assert drive.support_notes([row]) == ["lb (lb): a simple support held on components [1] only"]
    assert "clamped (its rotation held), marked by the runner" in drive.support_notes([{**row, "held": [0, 1, 2]}])[0]
    # a force-only contact held on its region is a hold, not a reaction
    held = [{**r, "category": "dirichlet"} if r["name"] == "lb" else r for r in regions]
    out = drive.statics(session, plan, held, planned(500, 500))
    assert out["held_as"] == ["lb"] and [r["region"] for r in out["reactions"]] == ["rb"]


def test_reactions_as_loads_need_some_hold_anywhere(tmp_path):
    import trimesh
    body = trimesh.creation.box(extents=[1.0, 0.06, 0.06])
    body.apply_translation([0.5, 0.03, 0.03])                 # x from 0 to 1, as the regions
    session = _held_session(tmp_path, [], {}, body=body)
    regions = _bar_regions(1.0)
    result = {"reactions": [{"region": "lb"}, {"region": "rb"}]}
    row = {"name": "end", "region": "end", "member_axis": "x", "held": [0, 1, 2], "span_m": [1.0, 1.0],
           "face": "end face"}
    plan = {**PLAN, "reference_hold": {"regions": ["end"]}}
    assert drive.reference_hold_problem(session, plan, [row], result, regions) is None
    # where the hold sits is the agent's: a hold anywhere, or another hold, is accepted
    middle = {**row, "span_m": [0.48, 0.52], "face": "all round"}
    assert drive.reference_hold_problem(session, plan, [middle], result, regions) is None
    # a body held nowhere has no solution: refused
    loose = [r for r in regions if r.get("category") != "dirichlet"]
    assert "nothing holds the body" in drive.reference_hold_problem(session, plan, [], result, loose)


def test_a_contact_is_bound_only_by_the_regions_the_plan_lists_and_the_refusal_says_what_to_write():
    plan = {**PLAN, "contacts": {"support A": {"transmits": "force only"},
                                 "support C": {"transmits": "force and moment"}},
            "reference_hold": {}}
    rows = [{"name": "support_a", "category": "neumann"}, {"name": "base_clamp", "category": "dirichlet"},
            {"name": "hold", "category": "dirichlet"}, {"name": "top_load", "category": "neumann"}]
    bound = drive.bind_by_name(plan, rows)
    assert "regions" not in bound["contacts"]["support A"]            # no name is matched
    out = drive.binding_problem(bound, rows)
    assert 'plan.contacts["support A"].regions = [<names>]' in out and 'plan.contacts["support C"]' in out
    assert "held regions: ['base_clamp', 'hold']" in out and "Nothing was spent" in out
    # Listed by the plan, they bind; the hold keeps the held region no contact claims.
    plan["contacts"]["support A"]["regions"] = ["support_a"]
    plan["contacts"]["support C"]["regions"] = ["base_clamp"]
    bound = drive.bind_by_name(plan, rows)
    assert drive.binding_problem(bound, rows) is None and bound["reference_hold"]["regions"] == ["hold"]
    # A stray held region is not absorbed into the hold.
    stray = rows + [{"name": "stray_patch", "category": "dirichlet"}]
    assert "the reference hold took ['hold', 'stray_patch']" in drive.binding_problem(drive.bind_by_name(plan, stray), stray)


def test_a_region_revision_keeps_part_of_the_synthesized_selector(tmp_path):
    import trimesh
    (tmp_path / "boundary_conditions").mkdir()
    (tmp_path / "operator").mkdir()
    (tmp_path / "geometry").mkdir()
    trimesh.creation.box(extents=[1.0, 0.1, 0.1]).export(tmp_path / "geometry" / "physical.glb")
    lib.write_json(tmp_path / "boundary_conditions" / "synthesized.json", SYNTHESIZED)
    assert drive.revision_answer(tmp_path, {"left": WIDER})["action"] == "revise_regions"
    assert lib.read_json(tmp_path / "operator" / "revision_kept.json") == {"left": ["lowest"]}
    # a structural revision (another selector for the same region) is sent too: the preview judges it
    authored = {"type": "aligned_normal", "params": {"target_vector": [1, 0, 0], "max_angle_deg": 30}}
    answer = drive.revision_answer(tmp_path, {"left": authored})
    assert any(item["selector"] == authored for item in answer["proposal"]["dirichlet"] + answer["proposal"]["neumann"])


def test_the_report_head_carries_the_verdict_beside_the_agreement(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = _held_session(tmp_path, [], {})
    (session / "operator" / "report.md").write_text("# Session\n\n## Engineer's judgement\n\n_(fill in)_\n")
    lib.write_json(session / "operator" / "prediction_history.json", [
        {"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "{}", "load_N": 10.0,
         "prediction": {"max_displacement": 1e-4, "max_stress": 2e6},
         "solved": {"max_displacement": 3e-5, "max_stress": 2e6}}])
    (session / "operator" / "j.md").write_text("The brief's case was run.\n## Deviations\nnone\n")
    drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                       file=str(session / "operator" / "j.md"), text=None))
    head = (session / "operator" / "report.md").read_text().splitlines()[2]
    assert head.startswith("**Verdict:** The brief's case was run.") and "max_displacement disagrees (0.3)" in head
    assert lib.read_json(session / "operator" / "verdict.json")["verdict"] == "The brief's case was run."


def test_the_support_motion_is_extended_past_the_outermost_reaction():
    # A member on two patches set in from its ends, rotated and lifted as a rigid body, plus a
    # known bending between the patches and on the overhangs.
    x = np.linspace(0.0, 1.2, 241)
    points = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    bend = -1e-4 * np.sin(np.pi * (x - 0.2) / 0.8) * ((x > 0.2) & (x < 1.0))
    rigid = 2e-3 + 5e-3 * (x - 1.2)
    u = np.column_stack([np.zeros_like(x), rigid + bend, np.zeros_like(x)])
    anchors = np.array([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.array([[0.0, 2e-3 + 5e-3 * (0.2 - 1.2), 0.0], [0.0, 2e-3 + 5e-3 * (1.0 - 1.2), 0.0]])
    relative = np.linalg.norm(u - drive.support_motion(points, anchors, values), axis=1)
    assert np.allclose(relative, np.abs(bend), atol=1e-12)          # the overhangs read zero
    # Three patches not on a line: the plane through them.
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(50, 3))
    plane = lambda q: np.column_stack([1e-3 + q @ [2e-3, 0, -1e-3], q @ [3e-3, 0, 1e-3], np.zeros(len(q))])
    anchors3 = np.array([[0.1, 0.0, 0.1], [0.9, 0.0, 0.1], [0.5, 0.0, 0.9]])
    assert np.allclose(drive.support_motion(pts, anchors3, plane(anchors3)), plane(pts), atol=1e-12)


def test_the_reference_hold_takes_force_across_the_reactions_but_no_moment(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    regions = _bar_regions()
    contacts = {"left": {"transmits": "force only", "regions": ["lb"]},
                "right": {"transmits": "force only", "regions": ["rb"]}}
    plan = {**PLAN, "contacts": contacts, "reference_hold": {"regions": ["end"]},
            "values": {"acceleration": [0, 0, 0]}}
    # 1000 N down and 300 N along the bar on the top face (0.06 m above the patches).
    planned = [{"region": "top", "force_N": [300, -1000, 0]}]
    result = drive.statics(session, plan, regions, planned)
    assert result["determinate"] and result["unbalanced_share"] < 1e-6
    assert result["hold_force_N"][0] == pytest.approx(-300.0)
    left, right = (r["force_N"][1] for r in result["reactions"])
    # The top force acts 0.03 above the hold (the end face's centre): that couple shifts load to the right patch.
    assert left + right == pytest.approx(1000.0) and left - right == pytest.approx(-2 * 300 * 0.03 / 0.95, rel=1e-3)
    # A sideways load: its moment in the plane of the bar (about y) needs a lateral support the pushing contacts are not.
    off = [{"region": "top", "force_N": [0, 0, 500]}]
    result = drive.statics(session, plan, regions, off)
    assert drive.statics_problem(result) is None
    assert "moment about y" in drive.statics_note(result) and "the holds take the rest" in drive.statics_note(result)


def test_a_pushing_patch_along_the_whole_body_is_reported(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    regions = [{"name": "top", "category": "neumann", "centroid": [0.0, 0.03, 0.0], "mean_normal": [0, 1, 0]},
               {"name": "patch", "category": "neumann", "centroid": [0.0, -0.03, 0.0], "mean_normal": [0, -1, 0],
                "bounds": [[-0.5, -0.03, -0.03], [0.5, -0.03, 0.03]]}]
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": regions})
    plan = {**PLAN, "contacts": {"patch": {"transmits": "force only", "regions": ["patch"]}},
            "values": {"acceleration": [0, 0, 0]}}
    result = drive.statics(session, plan, regions, [{"region": "top", "force_N": [0, -100, 0]},
                                                     {"region": "patch", "force_N": [0, 100, 0]}])
    assert drive.statics_problem(result) is None and "the body's whole length" in drive.statics_note(result)


def test_statics_reactions_take_the_drag_lever_whatever_the_plan_typed(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    regions = _bar_regions()
    plan = {**PLAN, "contacts": {"left": {"transmits": "force only", "regions": ["lb"]},
                                 "right": {"transmits": "force only", "regions": ["rb"]}},
            "reference_hold": {"regions": ["end"]}, "values": {"acceleration": [0, 0, 0]}}
    # Statics: 4905.3 and 5094.7 N (the drag's lever over the hold shifts 94.7 N).
    exact = drive.statics(session, plan, regions, [{"region": "top", "force_N": [3000, -10000, 0]},
                                                   {"region": "lb", "force_N": [5000, 0, 0]},
                                                   {"region": "rb", "force_N": [0, 0, 0]}])
    left, right = (r["force_N"][1] for r in exact["reactions"])
    assert left == pytest.approx(4905.263, abs=0.01) and right == pytest.approx(5094.737, abs=0.01)
    assert drive.statics_problem(exact) is None


def test_a_contact_whose_region_resolves_into_several_pieces_is_named_not_crashed(tmp_path):
    import trimesh
    bar = trimesh.creation.box(extents=[0.96, 0.09, 0.08]).subdivide().subdivide().subdivide().subdivide()
    centres, normals = bar.triangles_center, bar.face_normals
    # One held region of two separate pieces: the underside near each end.
    two = (normals[:, 1] < -0.9) & (np.abs(centres[:, 0]) > 0.3)
    session = _held_session(tmp_path, [{"name": "feet", "key": "feet", "mean_normal": [0, -1, 0],
                                        "bounds": [[-0.48, -0.045, -0.04], [0.48, -0.045, 0.04]]}],
                            {"feet": [0, 1, 2]}, body=bar)
    np.savez(session / "boundary_conditions" / "resolved.npz", feet=two)
    # a contact named in the singular (one named in the plural is several parts by its name)
    contacts = {"support": {"transmits": "force and moment", "regions": ["feet"]}}
    rows = drive.held_regions(session, {"feet": {"components": [0, 1, 2]}}, contacts)
    out = drive.pieces_problem(rows, contacts)
    assert len(rows) == 2 and "the runner sees 2 held pieces" in out
    assert drive.pieces_problem(rows, {"support": {**contacts["support"], "pieces": 2,
                                                   "pieces_why": "two feet"}}) is None
    plural = {"supports": contacts["support"]}
    assert drive.pieces_problem(drive.held_regions(session, {"feet": {"components": [0, 1, 2]}}, plural), plural) is None
    assert not any(r.get("surface_split") for r in rows)


def test_a_held_face_the_surface_splits_at_one_level_is_one_contact(tmp_path):
    """Pieces of one held face at one height, a sliver gap apart, are the generated surface
    splitting one contact: one row, said as a note, no pieces to declare."""
    import trimesh
    bar = trimesh.creation.box(extents=[0.96, 0.09, 0.08]).subdivide_to_size(0.004)
    bar = trimesh.Trimesh(bar.vertices, bar.faces, process=False)
    centres, normals = bar.triangles_center, bar.face_normals
    # The underside, broken by a sliver strip under a leaf wide (the octree cannot resolve it).
    split = (normals[:, 1] < -0.9) & (np.abs(centres[:, 0]) > 0.003)
    session = _held_session(tmp_path, [{"name": "base", "key": "base", "mean_normal": [0, -1, 0],
                                        "bounds": [[-0.48, -0.045, -0.04], [0.48, -0.045, 0.04]]}],
                            {"base": [0, 1, 2]}, body=bar)
    np.savez(session / "boundary_conditions" / "resolved.npz", base=split)
    contacts = {"floor": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}
    assert len(drive._region_pieces(bar, split)) == 2
    rows = drive.held_regions(session, {"base": {"components": [0, 1, 2]}}, contacts)
    assert len(rows) == 1 and "split by the generated surface" in rows[0]["surface_split"]
    assert drive.pieces_problem(rows, contacts) is None
    assert drive.held_pieces_warning(rows, contacts) is None
    # a declared count left over from before: the runner says the count it sees
    assert "the runner sees 1 held pieces" in drive.pieces_problem(
        rows, {"floor": {**contacts["floor"], "pieces": 2, "pieces_why": "two slabs of the base"}})


def test_g4_reads_the_load_path_wall_against_the_real_member():
    import trimesh
    cat, leaf = trimesh.util.concatenate, 1.92 / 128
    # an exact 2 mm plate on a 1 m body, and a 7 mm plate with its real 2 mm tab (0.2 % of the surface): pass
    assert drive.g4_member_ratio(trimesh.creation.box(extents=[1.0, 0.002, 0.5]), leaf, 0.002) == (1.0, "wall")
    plate = trimesh.creation.box(extents=[1.0, 0.007, 0.5])
    tab = trimesh.creation.box(extents=[0.03, 0.002, 0.03])
    tab.apply_translation([0.515, 0, 0])
    assert drive.g4_member_ratio(cat([plate, tab]), leaf, 0.002)[0] <= 1.5
    assert drive.g4_member_ratio(plate, leaf, 0.002) == (3.5, "wall")       # the plate alone: over 3 x
    # both readings keep the same 0.1 % share: an exact member on 0.13 % of the surface is read, one
    # on 0.04 % is not the body's member (the denser second look no longer takes its 4th-smallest gap)
    small = trimesh.creation.box(extents=[0.024, 0.002, 0.024])
    small.apply_translation([0.512, 0, 0])
    assert drive.g4_member_ratio(cat([plate, small]), leaf, 0.002, 9.0) == (1.0, "wall")
    tiny = trimesh.creation.box(extents=[0.012, 0.002, 0.012])
    tiny.apply_translation([0.506, 0, 0])
    assert drive.g4_member_ratio(cat([plate, tiny]), leaf, 0.002, 9.0) == (3.5, "wall")
    # a 20 mm bar on a bulky block whose real member is 20 mm: its own size
    block = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
    bar = trimesh.creation.box(extents=[0.6, 0.02, 0.02])
    bar.apply_translation([0.45, 0, 0])
    assert drive.g4_member_ratio(cat([block, bar]), 1.92 * 0.9 / 128, 0.02)[0] <= 1.5
    # a 50 mm bar or a 40 mm plate where the real member is 2 mm: no wall within reach, the voxel ratio stands
    thick = trimesh.creation.box(extents=[0.6, 0.05, 0.05])
    thick.apply_translation([0.45, 0, 0])
    assert drive.g4_member_ratio(cat([block, thick]), 1.92 * 0.9 / 128, 0.002, 25.0) == (25.0, "voxel")
    assert drive.g4_member_ratio(trimesh.creation.box(extents=[1.0, 0.04, 0.5]), leaf, 0.002, 20.0) == (20.0, "voxel")
    assert drive.range_findings({"member_over_real": 25.0}) and drive.range_findings({"member_over_real": 1.0}) is None


def test_the_two_faces_of_one_thin_wall_held_are_one_piece():
    import trimesh
    leaf = 1.92 / 128
    wall = trimesh.creation.box(extents=[1.0, 0.004, 0.5])
    faces = drive._region_pieces(wall, np.abs(wall.face_normals[:, 1]) > 0.9)
    assert len(faces) == 2 and len(drive.merge_wall_faces(wall, faces, leaf)) == 1
    # the two sides of a narrow slot face each other across empty space: two pieces
    left = trimesh.creation.box(extents=[0.2, 0.2, 0.5])
    left.apply_translation([-0.102, 0.0, 0.0])
    right = trimesh.creation.box(extents=[0.2, 0.2, 0.5])
    right.apply_translation([0.102, 0.0, 0.0])
    slot = trimesh.util.concatenate([left, right])
    sides = np.abs(slot.triangles_center[:, 0]) < 0.01                  # the faces at x = -0.002 and +0.002
    faces = drive._region_pieces(slot, sides & (np.abs(slot.face_normals[:, 0]) > 0.9))
    assert len(faces) == 2 and len(drive.merge_wall_faces(slot, faces, leaf)) == 2
    block = trimesh.creation.box(extents=[1.0, 0.2, 0.5])                 # a thick part's two faces stay two
    assert len(drive.merge_wall_faces(block, drive._region_pieces(block, np.abs(block.face_normals[:, 1]) > 0.9),
                                      leaf)) == 2


def test_held_slivers_under_a_leaf_face_are_noted_not_counted(tmp_path):
    import trimesh
    bar = trimesh.creation.box(extents=[0.96, 0.09, 0.08]).subdivide_to_size(0.004)
    bar = trimesh.Trimesh(bar.vertices, bar.faces, process=False)
    centres, normals = bar.triangles_center, bar.face_normals
    leaf = 1.92 * 0.96 / 2 ** 7                     # 14 mm: a leaf face is 2.1e-4 m2
    underside = normals[:, 1] < -0.9
    # one triangle of the top face (8e-6 m2), away from the underside: a sliver of the region
    top = np.flatnonzero((normals[:, 1] > 0.9) & (np.abs(centres[:, 0]) < 0.003))[:1]
    held = underside.copy()
    held[top] = True
    session = _held_session(tmp_path, [{"name": "base", "key": "base", "mean_normal": [0, -1, 0],
                                        "bounds": [[-0.48, -0.045, -0.04], [0.48, 0.045, 0.04]]}],
                            {"base": [0, 1, 2]}, body=bar)
    np.savez(session / "boundary_conditions" / "resolved.npz", base=held)
    contacts = {"floor": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}
    assert len(drive.merge_surface_split(bar, drive._region_pieces(bar, held), leaf)) == 2
    rows = drive.held_regions(session, {"base": {"components": [0, 1, 2]}}, contacts)
    # the sliver keeps its row, marked and named apart; it is not counted, listed or named by a check
    assert len(rows) == 2 and [bool(r.get("sliver")) for r in rows].count(True) == 1
    counted = next(r for r in rows if not r.get("sliver"))
    sliver = next(r for r in rows if r.get("sliver"))
    assert counted["name"] == "base" and sliver["name"] == "base#sliver1"
    assert counted["pieces_of_region"] == 1 and counted["raw_pieces_of_region"] == 2
    assert "1 more connected piece(s)" in counted["slivers"] and "not counted" in counted["slivers"]
    assert drive.pieces_problem(rows, contacts) is None
    assert len(drive.collapse_held(rows)) == 1
    # a sliver beside a load is named by no check; the report's contacts table marks it
    beside = [{**r, "load_beside": 0.9 if r.get("sliver") else 0.0} for r in rows]
    assert drive.pieces_problem(beside, contacts) is None
    assert drive.held_pieces_warning([{**r, "contact": None} for r in beside],
                                      {"held": {"transmits": "force and moment"}}) is None
    unclamped = drive.support_notes([{**r, "simplified_to": None, "transmits": "force only"} for r in rows])
    assert len(unclamped) == 1 and unclamped[0].startswith("floor (base): a simple support") and "sliver" not in unclamped[0]
    table = drive.contacts_table(session, rows, contacts)
    assert sum("sliver (held, not counted" in line for line in table) == 1
    # a "pieces" that counts the sliver too is accepted
    raw = {"floor": {**contacts["floor"], "pieces": 2, "pieces_why": "the underside and a facet of the top"}}
    assert drive.pieces_problem(rows, raw) is None
    # a patch of several leaf faces is a held piece, counted
    patch = (normals[:, 1] > 0.9) & (np.abs(centres[:, 0]) < 0.03)
    np.savez(session / "boundary_conditions" / "resolved.npz", base=underside | patch)
    rows = drive.held_regions(session, {"base": {"components": [0, 1, 2]}}, contacts)
    assert len(rows) == 2 and not any(r.get("slivers") or r.get("sliver") for r in rows)
    # small feet of one size are each a contact, even under a leaf face; a tiny facet beside a big
    # face is a sliver
    feet = [{"area": 1.0e-4}, {"area": 1.1e-4}, {"area": 0.9e-4}, {"area": 1.0e-4}]
    assert drive.drop_slivers(feet, leaf) == (feet, [])
    # two half-leaf feet beside a 30-leaf-face base are each other's twin: both counted; a lone
    # small piece of another size beside them is still a sliver
    face = leaf ** 2
    base, two_feet = {"area": 30 * face}, [{"area": 0.5 * face}, {"area": 0.45 * face}]
    assert drive.drop_slivers([base, *two_feet], leaf) == ([base, *two_feet], [])
    odd = {"area": 0.05 * face}
    assert drive.drop_slivers([base, *two_feet, odd], leaf)[1] == [odd]
    assert drive.drop_slivers([{"area": 436 * face}, {"area": 17.9 * face}, odd], leaf)[1] == [odd]
    kept, slivers = drive.drop_slivers([{"area": 0.07}, {"area": 8e-6}], leaf)
    assert kept == [{"area": 0.07}] and slivers == [{"area": 8e-6}]
    # measured against the leaf at the plan's depth: a piece of 1.0e-4 m2 is over a leaf face at
    # depth 8 (5.2e-5 m2), so no sliver there, but under one at depth 7 (2.1e-4 m2)
    big = {"area": 0.07}
    assert drive.drop_slivers([big, {"area": 1.0e-4}], leaf)[1] == [{"area": 1.0e-4}]
    assert drive.drop_slivers([big, {"area": 1.0e-4}], leaf / 2)[1] == []


def _bottom_pieces(mesh):
    """The connected pieces of a mesh's downward faces at its lowest level."""
    bottom = (mesh.face_normals[:, 1] < -0.9) & (mesh.triangles_center[:, 1] < mesh.bounds[0][1] + 1e-6)
    return bottom, drive._region_pieces(mesh, bottom)


def test_close_but_distinct_patches_are_not_read_as_a_surface_split():
    """Only what the octree cannot tell apart is joined: posts a few leaves apart, a pad inside a
    ring, and the opposite faces of a plate thinner than two leaves stay separate pieces."""
    import trimesh
    leaf = 1.92 * 1.8 / 2 ** 7                      # a 1.8 m body at depth 7: 27 mm
    # eight posts, 50 mm apart, standing on one floor
    posts = trimesh.util.concatenate([
        trimesh.creation.box(extents=[0.05, 0.4, 0.05]).apply_translation([0.1 * k, 0.2, 0.0])
        for k in range(8)])
    _, pieces = _bottom_pieces(posts)
    assert len(pieces) == 8 and len(drive.merge_surface_split(posts, pieces, leaf)) == 8
    # a pad 50 mm inside a ring: their boxes overlap, their faces do not come near
    ring = trimesh.creation.annulus(r_min=0.15, r_max=0.25, height=0.1)
    pad = trimesh.creation.cylinder(radius=0.1, height=0.1)
    both = trimesh.util.concatenate([ring, pad])
    both.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    _, pieces = _bottom_pieces(both)
    assert len(pieces) == 2 and len(drive.merge_surface_split(both, pieces, leaf)) == 2
    # a plate 10 mm thick: its underside and part of its top are not one contact
    plate = trimesh.creation.box(extents=[0.5, 0.01, 0.5]).subdivide_to_size(0.05)
    plate = trimesh.Trimesh(plate.vertices, plate.faces, process=False)
    centres, normals = plate.triangles_center, plate.face_normals
    held = (normals[:, 1] < -0.9) | ((normals[:, 1] > 0.9) & (centres[:, 0] > 0.15))
    pieces = drive._region_pieces(plate, held)
    assert len(pieces) == 2 and len(drive.merge_surface_split(plate, pieces, leaf)) == 2


def test_the_runner_moves_a_synthesized_bound_off_the_bodys_extreme(tmp_path):
    import trimesh
    (tmp_path / "boundary_conditions").mkdir()
    (tmp_path / "operator").mkdir()
    (tmp_path / "geometry").mkdir()
    body = trimesh.creation.box(extents=[1.0, 0.1, 0.1])
    body.apply_translation([0.5, 0.05, 0.05])
    body.export(tmp_path / "geometry" / "physical.glb")
    at_end = {"type": "and", "params": {}, "children": [
        {"type": "aligned_normal", "params": {"target_vector": [1, 0, 0], "max_angle_deg": 45, "oriented": True},
         "children": []},
        {"type": "slab", "params": {"axis": "x", "min_val": 0.95, "max_val": 1.0}, "children": []}]}
    inside = {"type": "slab", "params": {"axis": "x", "min_val": 0.4, "max_val": 0.6}, "children": []}
    lib.write_json(tmp_path / "boundary_conditions" / "synthesized.json",
                   {"dirichlet": [{"name": "end", "selector": at_end}],
                    "neumann": [{"name": "mid", "selector": inside}]})
    fixes = drive.extreme_bound_fixes(tmp_path)
    leaf = 1.92 * 1.0 / 128
    assert list(fixes) == ["end"]
    assert fixes["end"]["children"][1]["params"]["max_val"] == pytest.approx(1.0 + leaf, abs=1e-6)
    answer = drive.revision_answer(tmp_path, {})
    assert answer["proposal"]["dirichlet"][0]["selector"] == fixes["end"]
    assert "the runner moved a bound" in lib.read_json(tmp_path / "operator" / "revision_kept.json")["end"]


def test_a_per_part_run_is_not_judged_on_the_number():
    entry = {"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "{}", "load_N": 10.0,
             "prediction": {"max_displacement": 1e-4, "max_stress": 2e6}, "material_mode": "parts",
             "solved": {"max_displacement": 3e-5, "max_stress": 2e6}}
    agreed = drive.agreement([entry])
    assert agreed["max_displacement"]["statement"].startswith("not judged")
    assert drive.agreement([{**entry, "material_mode": "uniform"}])["max_displacement"]["statement"] == "disagrees"


def test_the_geometry_stop_finds_surfaces_facing_across_less_than_a_leaf():
    import trimesh
    leaf = 0.01
    one = trimesh.creation.box(extents=[0.5, 0.05, 0.2])
    assert drive.thin_gaps(one, leaf)["share_of_surface"] == 0.0
    upper = trimesh.creation.box(extents=[0.5, 0.05, 0.2])
    upper.apply_translation([0, 0.054, 0])          # 4 mm of air under it
    gaps = drive.thin_gaps(trimesh.util.concatenate([one, upper]), leaf)
    assert gaps["share_of_surface"] > 0.1 and gaps["width_m"][1] == pytest.approx(0.004, abs=0.001)


def test_a_body_outside_the_range_cannot_carry_was_run(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = _held_session(tmp_path, [], {})
    (session / "operator" / "report.md").write_text("## Engineer's judgement\n\n_(fill in)_\n")
    lib.write_json(session / "operator" / "stop_geometry.json",
                   {"planned_resolution": {"member_over_real": 7.0, "mass_over_real": 2.6}})
    (session / "operator" / "j.md").write_text("The brief's case was run.\n## Deviations\nnone\n")
    drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                       file=str(session / "operator" / "j.md"), text=None))
    assert printed[-1]["ok"] and "could not be run: the body is outside the workflow's range (thinnest member 7.0 x" \
        in (session / "operator" / "report.md").read_text()
    assert drive.real_member(session / "operator") is None


def test_a_body_is_built_only_after_the_brief(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    drive.cmd_blank(argparse.Namespace(output_root=str(tmp_path), session_id="abc", shape="box",
                                       size=[1.0, 0.1, 0.1], radius=None, length=None, axis="y",
                                       profile=None, hole=None, circle=None))
    assert not printed[-1]["ok"] and "before any body exists" in printed[-1]["error"]



def test_every_documented_command_runs_from_the_repository_root():
    import re
    skill = SCRIPTS.parent
    repo = skill.parents[3]
    prefix = "python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/"
    commands = {sub.dest: sub for sub in drive.parser()._subparsers._group_actions}
    choices = set(next(iter(commands.values())).choices)
    for path in [skill / "SKILL.md", *sorted((skill / "references").rglob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"`([^`]*?(drive|inspect_session)\.py[^`]*)`", text):
            command = match.group(1)
            if not command.startswith(prefix):
                continue                                 # a mention of the file, not a command
            script, rest = command[len(prefix):].split(" ", 1)
            assert (repo / "src/agents_wrapper/fem_workflow/operator_skill/scripts" / script).exists()
            tokens, word = rest.split(), None
            valued = {"--output-root", "--gpu", "--gpu-wait", "--max-foreign-mib", "--max-util", "--prefetch"}
            while tokens:
                token = tokens.pop(0)
                if token in valued:
                    tokens.pop(0)
                elif not token.startswith("-"):
                    word = token
                    break
            if script == "drive.py":
                assert word in choices, (path.name, command[:120])
            plain = re.sub(r"<[^<>]*>", "X", rest.replace("\\|", ""))
            assert not re.search(r"\$[A-Z(]|&&|;| \| |>", plain), (path.name, command[:120])


def test_no_file_of_the_skill_names_an_absolute_path():
    import re
    skill = SCRIPTS.parent
    pattern = re.compile(r"(?:^|[\s`'\"(=\[])(?:/(?:home|usr|opt|tmp|root|mnt|data|Users|var|etc|srv)/|~/|[A-Za-z]:\\\\)")
    found = []
    for path in sorted(skill.rglob("*")):
        if path.is_file() and path.suffix in (".md", ".py", ".json", ".txt", ".sh") and "__pycache__" not in path.parts:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line) and "pattern = re.compile" not in line:
                    found.append(f"{path.relative_to(skill)}:{number}")
    assert not found, found


def test_a_backend_result_no_second_backend_confirmed_says_so(tmp_path, monkeypatch):
    import inspect_session
    (tmp_path / "operator").mkdir()
    monkeypatch.setattr(inspect_session, "case_digest", lambda session: "c1")
    lib.write_json(tmp_path / "operator" / "converged.json",
                   {"backend": "pyamgx", "run": "r1", "case": "c1", "final_rel": 0.009})
    note = drive.solver_note(tmp_path)
    assert "unconfirmed by a second backend" in note and "0.009" in note
    lib.write_json(tmp_path / "operator" / "converged.json",
                   {"backend": "pyamgx", "run": "r1", "case": "c1", "confirmed_by": "native"})
    assert drive.solver_note(tmp_path) is None


def test_the_reference_hold_takes_the_held_regions_no_contact_claims():
    plan = {**PLAN, "contacts": {"support A": {"transmits": "force only", "regions": ["support_A"]},
                                 "support C": {"transmits": "force and moment", "regions": ["support_C"]}},
            "reference_hold": {}}
    rows = [{"name": "support_A", "category": "neumann"}, {"name": "support_C", "category": "dirichlet"},
            {"name": "hold", "category": "dirichlet"}, {"name": "top_load", "category": "neumann"}]
    assert drive.bind_by_name(plan, rows)["reference_hold"]["regions"] == ["hold"]


def test_statics_own_leftover_is_not_counted_against_the_plans_reactions(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    regions = _bar_regions()
    regions[0] = {**regions[0], "centroid": [0.5, 0.06, 0.033]}     # the load 3 mm off the supports' line
    plan = {**PLAN, "contacts": {"left": {"transmits": "force only", "regions": ["lb"]},
                                 "right": {"transmits": "force only", "regions": ["rb"]}},
            "reference_hold": {"regions": ["end"]}, "values": {"acceleration": [0, 0, 0]}}
    planned = [{"region": "top", "force_N": [0, -1000, 0]}, {"region": "lb", "force_N": [0, 0, 0]},
               {"region": "rb", "force_N": [0, 0, 0]}]
    exact = [r["force_N"][1] for r in drive.statics(session, plan, regions, planned)["reactions"]]
    planned[1]["force_N"], planned[2]["force_N"] = [0, exact[0], 0], [0, exact[1], 0]
    assert drive.statics_problem(drive.statics(session, plan, regions, planned)) is None


def test_a_per_part_body_is_held_to_the_real_mass_from_the_solve(tmp_path):
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text("## Size and material\nReal mass: 1 kg\n")
    lib.write_json(tmp_path / "operator" / "stop_geometry.json", {"planned_resolution": {"member_over_real": 1.0}})
    lib.write_json(tmp_path / "operator" / "stop_solve.json", {"body": {"mass": 6.2}})
    assert "mass 6.2 x the real one" in drive.outside_range(tmp_path)


def test_another_held_support_beside_the_two_simple_supports_is_reported_not_refused(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    regions = _bar_regions() + [{"name": "stop", "category": "dirichlet", "centroid": [0.0, 0.03, 0.03],
                                 "mean_normal": [-1, 0, 0]}]
    plan = {**PLAN, "contacts": {"support A": {"transmits": "force only", "regions": ["lb"]},
                                 "support B": {"transmits": "force only", "regions": ["rb"]},
                                 "support C": {"transmits": "force and moment", "regions": ["stop"]}},
            "reference_hold": {"regions": ["end"]}, "values": {"acceleration": [0, 0, 0]}}
    planned = [{"region": "top", "force_N": [0, -1000, 0]}, {"region": "lb", "force_N": [0, 500, 0]},
               {"region": "rb", "force_N": [0, 500, 0]}]
    result = drive.statics(session, plan, regions, planned)
    assert drive.statics_problem(result) is None
    assert "['stop'] held besides the reference hold" in drive.statics_note(result)
    # Only held supports, no simple support: the reaction scheme is not involved.
    held_only = {**plan, "contacts": {"support C": {"transmits": "force and moment", "regions": ["stop", "end"]}}}
    assert drive.statics(session, held_only, regions, planned) is None


def _built_session(tmp_path, body, real_member, real_section):
    """A session whose body is built with ``blank`` (its staged name) and whose brief, written
    before the body, declares the real member and section."""
    session = _held_session(tmp_path, [], {}, body=body)
    body.export(session / "geometry" / "original.glb")
    lib.write_json(session / "session.json", {"geometry_mode": "surface",
                                              "input_asset": {"source_name": "blank_box_0a1b2c3d.glb"}})
    (session / "operator" / "brief.md").write_text(
        "## Size and material\nReal mass: 2 kg\n"
        f"Real thinnest member: {real_member} m\nReal section: {real_section}\n"
        'Load path: {"outline": "series"}\n')
    drive.brief_status(session / "operator")          # the snapshot a built body takes
    return session


def test_a_member_no_depth_resolves_runs_as_a_section_stand_in_with_its_ratios(tmp_path, gaps_not_read):
    import trimesh
    bar = trimesh.creation.box(extents=[1.0, 0.04, 0.04])     # a solid stand-in, 40 mm square
    real_i, real_a = (0.04 ** 4 - 0.036 ** 4) / 12, 0.04 ** 2 - 0.036 ** 2     # a 2 mm wall of that outline
    session = _built_session(tmp_path, bar, 0.002, f"0.04 m deep, I {real_i:.6g} m4, A {real_a:.6g} m2")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850},
            "settings": {"remesh_octree_depth": 8},
            "body_simplification": {"kept": "the outline", "left_out": "the hollow"}}
    resolution = drive.planned_resolution(session, plan)
    assert drive.body_route(session, plan, resolution)["route"] == "section"     # derived, never declared
    limit = drive.thin_member_limit(session, plan, resolution)
    assert limit["unresolvable"] and limit["real_member_leaves_at_depth_8"] < 1
    eq = drive.section_equivalence(session, plan, resolution)
    assert abs(eq["stiffness_ratio"] - (0.04 ** 4 / 12) / real_i) < 0.05 * eq["stiffness_ratio"]
    assert abs(eq["area_ratio"] - 0.04 ** 2 / real_a) < 0.05 * eq["area_ratio"]
    assert "not the real stress" in eq["stress"] and "x " in eq["deflection"]
    # the stand-in is a body the workflow resolves: no thin-member warning
    assert drive.resolution_warning(session, plan, resolution) is None


def test_a_member_under_two_leaves_is_warned_and_carried_as_a_resolution_caveat(tmp_path, gaps_not_read):
    import trimesh
    thin = trimesh.creation.box(extents=[1.0, 0.012, 0.012])
    session = _built_session(tmp_path, thin, 0.002, "0.012 m deep, I 1.5e-9 m4, A 8.0e-5 m2")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850}, "settings": {}}
    said = drive.resolution_warning(session, plan)
    assert said.startswith("thin member: the body's thinnest member is") and "under 2" in said
    assert "not a refusal" in said and "resolution caveat" in said
    deep = {**plan, "settings": {"remesh_octree_depth": 8}}
    assert "at depth 8" in drive.resolution_warning(session, deep)
    caveat = drive.resolution_caveat(session, deep)
    assert "leaves at depth 8" in caveat and "lower bound" in caveat
    # the report carries the caveat
    assert "resolution_caveat" in drive.thin_declaration(session, deep, summary={})
    # a body of 2 leaves or more: nothing said
    real = _built_session(tmp_path / "real", trimesh.creation.box(extents=[1.0, 0.04, 0.04]), 0.04,
                          "0.04 m deep, I 2.1e-7 m4, A 1.6e-3 m2")
    assert drive.resolution_warning(real, deep) is None and drive.resolution_caveat(real, deep) is None


def test_the_real_section_line_is_required_and_held_to_what_any_section_obeys(tmp_path):
    head = ("## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground\n"
            "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\n"
            "Real thinnest member: 0.002 m\n")
    tail = "0.9 m tall\n## Prediction\nmembers in compression, 0.1 MPa, microns"
    path = tmp_path / "brief.md"
    path.write_text(tabled(head + tail))
    assert "Real section" in drive.brief_missing(path)
    path.write_text(tabled(head + "Real section: 0.04 m deep, I 9e-7 m4, A 3e-4 m2\n" + tail))
    assert "which no section reaches" in drive.brief_missing(path)
    path.write_text(tabled(head + "Real section: 0.04 m deep, I 1e-9 m4, A 1e-5 m2\n" + tail))
    assert "less than a wall" in drive.brief_missing(path)
    path.write_text(tabled(head + "Real section: 0.04 m deep, I 7.3e-8 m4, A 3.04e-4 m2\n" + tail))
    assert drive.brief_missing(path) is None
    # The refusal names the conflicting numbers and what would agree with them.
    (tmp_path / "section_refused.json").unlink()
    path.write_text(tabled(head + "Real section: 0.118 m deep, I 5.8e-5 m4, A 0.015 m2\n" + tail))
    out = drive.brief_missing(path)
    assert "depth 0.118 m" in out and "2 x sqrt(I/A) = 0.1244 m" in out and "I is at most 5.221e-05" in out
    # I lowered under the bound, depth and A unchanged: the computation is shown, not a bare value
    path.write_text(tabled(head + "Real section: 0.118 m deep, I 4.5e-5 m4, A 0.015 m2\n"
                    "  (I estimated at 4.5e-5 m4, below the bound)\n" + tail))
    assert "show I's computation" in drive.brief_missing(path)
    # a formula that does not give the stated I, or prose with a number in it, does not pass
    path.write_text(tabled(head + "Real section: 0.118 m deep, I 4.5e-5 m4, A 0.015 m2\n"
                    "  (a hollow box: I = (0.162 x 0.118^3 - 0.11 x 0.066^3)/12)\n" + tail))
    assert "show I's computation" in drive.brief_missing(path)
    path.write_text(tabled(head + "Real section: 0.118 m deep, I 2e-5 m4, A 0.015 m2\n"
                    "  I = 2e-5 (about 1/3 of the solid block's)\n" + tail))
    assert "show I's computation" in drive.brief_missing(path)
    # a formula that evaluates to the stated I passes: on an indented line or bullet below, or inline
    for below in ("  (a hollow box: I = (0.162 x 0.118^3 - 0.11 x 0.066^3)/12)\n",
                  "  - hollow box, I = (0.162 × 0.118³ − 0.11 × 0.066³)/12\n",
                  "  (0.162 m x (0.118 m)^3 - 0.11 m x (0.066 m)^3) / 12\n"):
        path.write_text(tabled(head + "Real section: 0.118 m deep, I 1.955e-5 m4, A 0.015 m2\n" + below + tail))
        assert drive.brief_missing(path) is None, below
    assert drive.section_computed("Real section: 0.118 m deep, I 1.955e-5 m4, A 0.015 m2 "
                                  "(I = (0.162*0.118**3 - 0.11*0.066**3)/12)")
    assert drive.section_computed("Real section: 0.3 m deep, I 4.5e-3 m4, A 0.6 m2 (2.0 × 0.3³/12)")
    assert not drive.section_computed("Real section: 0.3 m deep, I 4.5e-3 m4, A 0.6 m2 (I estimated 4.5e-3)")
    # a change in the ninth digit of A is the same A
    path.write_text(tabled(head + "Real section: 0.118 m deep, I 4.5e-5 m4, A 0.015000001 m2\n" + tail))
    assert "show I's computation" in drive.brief_missing(path)
    path.write_text(tabled(head + "Real section: 0.178 m deep, I 4.5e-5 m4, A 0.015 m2\n" + tail))
    assert drive.brief_missing(path) is None                    # the depth corrected instead


def test_a_gap_under_a_leaf_stops_before_the_regions_until_the_plan_answers_it(tmp_path):
    import trimesh
    a = trimesh.creation.box(extents=[1.0, 0.04, 0.04])
    b = a.copy()
    b.apply_translation([0.0, 0.043, 0.0])                    # 3 mm of air between two bars
    session = _built_session(tmp_path, trimesh.util.concatenate([a, b]), 0.04,
                             "0.04 m deep, I 2.1e-7 m4, A 1.6e-3 m2")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "settings": {"remesh_octree_depth": 8}}
    out = drive.thin_gaps_problem(session, plan)
    assert "bonds the two sides" in out and "plan.bonded_gaps" in out
    assert drive.thin_gaps_problem(session, {**plan, "bonded_gaps": "the two bars act as one there"}) is None
    record = lib.read_json(session / "operator" / "thin_gaps.json")
    assert record["bonded_gaps"] and record["share_of_surface"] > 0.1
    line = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None,
                                 {"thin_gaps": record})
    assert "bonded in the solve: the two bars act as one there" in line
    with pytest.raises(drive.PlanError, match="bonded_gaps"):
        _load(tmp_path, dict(plan, bonded_gaps=" "))


def _load(tmp_path, plan):
    path = tmp_path / "plan_check.json"
    path.write_text(json.dumps(plan))
    return drive.load_plan(path)


def test_a_bending_moment_left_at_the_hold_is_reported_and_a_twist_is_printed(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    plan = {**PLAN, "contacts": {"left": {"transmits": "force only", "regions": ["lb"]},
                                 "right": {"transmits": "force only", "regions": ["rb"]}},
            "reference_hold": {"regions": ["end"]}, "values": {"acceleration": [0, 0, 0]}}

    def run(side):
        regions = _bar_regions() + [
            {"name": "s1", "category": "neumann", "centroid": [0.25, 0.03, 0.06], "mean_normal": [0, 0, 1]},
            {"name": "s2", "category": "neumann", "centroid": [0.75, 0.03, 0.0], "mean_normal": [0, 0, -1]}]
        planned = [{"region": "top", "force_N": [0, -1000, 0]}, {"region": "s1", "force_N": [0, 0, -side]},
                   {"region": "s2", "force_N": [0, 0, side]}, {"region": "lb", "force_N": [0, 500, 0]},
                   {"region": "rb", "force_N": [0, 500, 0]}]
        return drive.statics(session, plan, regions, planned)
    # A sideways couple along the member: two simple supports cannot resist it, the hold takes none.
    out = drive.statics_note(run(15.0))
    assert "left to the hold" in out and "about y" in out and drive.statics_problem(run(15.0)) is None
    assert drive.statics_note(run(0.5)) is None
    # A load off the supports' line twists the member: the seats' share across the width, printed.
    regions = _bar_regions()
    regions[0] = {**regions[0], "centroid": [0.5, 0.06, 0.04]}
    planned = [{"region": "top", "force_N": [0, -1000, 0]}, {"region": "lb", "force_N": [0, 500, 0]},
               {"region": "rb", "force_N": [0, 500, 0]}]
    result = drive.statics(session, plan, regions, planned)
    assert drive.statics_problem(result) is None
    assert abs(result["left_to_hold_by_statics"]["twist_about_member_axis_Nm"]) > 5


def test_reactions_act_in_the_plane_of_the_loads_whatever_a_patch_leans_sideways(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    regions = _bar_regions()
    regions[1] = {**regions[1], "mean_normal": [0.02, -0.98, 0.0]}      # selections a little lopsided
    regions[2] = {**regions[2], "mean_normal": [0.0, -0.98, 0.03]}
    plan = {**PLAN, "contacts": {"left": {"transmits": "force only", "regions": ["lb"]},
                                 "right": {"transmits": "force only", "regions": ["rb"]}},
            "reference_hold": {"regions": ["end"]}, "values": {"acceleration": [0, 0, 0]}}
    planned = [{"region": "top", "force_N": [0, -1000, 0]}, {"region": "lb", "force_N": [0, 500, 0]},
               {"region": "rb", "force_N": [0, 500, 0]}]
    result = drive.statics(session, plan, regions, planned)
    assert all(abs(r["force_N"][2]) < 1e-6 for r in result["reactions"])
    assert drive.statics_problem(result) is None


def test_the_simplified_verdict_for_a_section_stand_in_needs_it_declared_before_the_solve(tmp_path, gaps_not_read):
    import trimesh
    bar = trimesh.creation.box(extents=[1.0, 0.04, 0.04])
    real_i, real_a = (0.04 ** 4 - 0.036 ** 4) / 12, 0.04 ** 2 - 0.036 ** 2
    session = _built_session(tmp_path, bar, 0.002, f"0.04 m deep, I {real_i:.6g} m4, A {real_a:.6g} m2")
    body = {"kept": "the outline", "left_out": "the hollow"}
    declared = {"scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
                "settings": {"remesh_octree_depth": 8},
                "material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850}}
    prediction = {"max_displacement": 1.5e-3, "max_stress": 4e7,
                  "brief_case": {"max_displacement": 4.5e-3, "max_stress": 6e7}}
    history = [{"run": "r1", "body": {"sha": "a", "size": 1.0}, "ends": drive.end_conditions({"body_route": "section"}),
                "load_N": 200.0, "prediction": prediction, "solved": {"max_displacement": 1.6e-3, "max_stress": 3.9e7}}]
    lib.write_json(session / "operator" / "prediction_history.json", history)
    # the stand-in is the runner's finding (the real member resolves at no depth): nothing to declare first
    problem, info = drive.simplification_problem(session, [], body, prediction, declared)
    assert problem is None and info["body"]["route"] == "section"
    ratio = info["section"]["stiffness_ratio"]
    assert abs(info["section"]["real_deflection_m"] - 1.6e-3 * ratio) < 1e-3 * 1.6e-3 * ratio
    # without a body_simplification it is a simplified case all the same
    problem, info = drive.simplification_problem(session, [], None, prediction, declared)
    assert problem is None and "section" in info
    # No route attempts are needed for it: the limitation is the member, not the generator.
    assert drive.body_simplification_problem(session, {**declared, "body_simplification": body}) is None


ABSOLUTE = r"(?:^|[\s`'\"(=\[])(?:/(?:home|usr|opt|tmp|root|mnt|data|Users|var|etc|srv)/|~/|[A-Za-z]:\\\\)"


def test_what_the_runner_prints_for_the_agent_carries_no_absolute_path(tmp_path, monkeypatch, capsys):
    """The rule covers runner output the agent reads or copies: a built body's plan entry, every
    printed value, the saved last stop, and the report; a printed path resolves from the root."""
    import re
    monkeypatch.setattr(lib, "REPO_ROOT", tmp_path)          # the session sits under the root
    session = tmp_path / "output" / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest "
        "member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall\n## Prediction\nmembers in compression, 0.1 MPa, microns"))
    drive.cmd_blank(argparse.Namespace(
        output_root=str(tmp_path / "output"), session_id="abc", shape="box", size=[1.0, 0.1, 0.1], radius=None,
        length=None, axis="y", profile=None, hole=None, circle=None))
    printed = capsys.readouterr().out
    assert not re.search(ABSOLUTE, printed), printed
    path = json.loads(printed)["plan"]["geometry"]["asset_path"]
    assert path.startswith("output/abc/operator/blank_box_")
    monkeypatch.chdir(tmp_path / "output")                   # resolved from the root, not the shell
    errors = []
    assert drive._existing_path(path, "asset_path", errors) and not errors
    # Nested values, the saved stop and the report go through the same conversion.
    shown = lib.relative_paths({"look_at": [session / "operator" / "regions.png"],
                                "error": f"see {session / 'tool_receipts' / 'r.json'}"})
    assert not re.search(ABSOLUTE, json.dumps(shown)) and shown["look_at"] == ["output/abc/operator/regions.png"]
    report = session / "operator" / "report.md"
    report.write_text(f"# Session\n- {session / 'operator' / 'regions.png'}\n## Engineer's judgement\n")
    drive.add_history_to_report(report, [{"run": "r", "body": {}, "ends": "", "load_N": 1.0}])
    assert not re.search(ABSOLUTE, report.read_text())


def test_the_section_route_states_its_reach_by_the_declared_load_path(tmp_path, gaps_not_read, monkeypatch):
    import trimesh
    bar = trimesh.creation.box(extents=[1.0, 0.04, 0.04])
    real_i, real_a = (0.04 ** 4 - 0.036 ** 4) / 12, 0.04 ** 2 - 0.036 ** 2
    session = _built_session(tmp_path, bar, 0.002, f"0.04 m deep, I {real_i:.6g} m4, A {real_a:.6g} m2")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850},
            "settings": {"remesh_octree_depth": 8}, "body_simplification": {"kept": "k", "left_out": "l"}}
    resolution = drive.planned_resolution(session, plan)
    # one Load path member, the case run's effect there in N m (the default without a Use table): bending
    bent = drive.section_equivalence(session, plan, resolution)
    assert bent["deflection_factor"] == bent["stiffness_ratio"] and "whole load path is this one real" in bent["deflection"]
    # a Load path of several members: the ratios are estimates
    brief = session / "operator" / "brief.md"
    brief.write_text(brief.read_text().replace('{"outline": "series"}', '{"outline": "series", "foot": "series"}'))
    other = drive.section_equivalence(session, plan, resolution)
    assert other["deflection_factor"] is None and "estimates only" in other["deflection"]
    # The brief's real mass against its real section and the plan's density: the length it implies.
    assert abs(bent["real_section_length_implied_m"] - 2.0 / (7850 * real_a)) < 1e-3
    with pytest.raises(drive.PlanError, match="load_path is the runner's"):
        _load(tmp_path, {**plan, "body_simplification": {"load_path": "other", "kept": "k", "left_out": "l"}})


def test_a_thin_part_declared_off_the_load_path_is_reported_with_the_caveat(tmp_path, gaps_not_read):
    import trimesh
    thin = trimesh.creation.box(extents=[1.0, 0.012, 0.012])
    session = _built_session(tmp_path, thin, 0.04, "0.04 m deep, I 2.1e-7 m4, A 1.6e-3 m2")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850},
            "settings": {"remesh_octree_depth": 8}}
    assert drive.resolution_warning(session, plan)
    declared = {**plan, "thin_off_load_path": "a thin lip along one edge; the load enters and leaves elsewhere"}
    # a declaration is said beside the warning, never a refusal or a clearance
    said = drive.resolution_warning(session, declared)
    assert "declared off the load path: a thin lip" in said and "not a refusal" in said
    line = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None,
                                 drive.thin_declaration(session, declared))
    assert "- resolution caveat: the body's thinnest member is" in line and "declared off the load path: a thin lip" in line
    with pytest.raises(drive.PlanError, match="thin_off_load_path"):
        _load(tmp_path, {**plan, "thin_off_load_path": ""})


def test_a_twist_over_a_tenth_of_the_span_moment_is_reported(tmp_path, monkeypatch):
    import trimesh
    session = _held_session(tmp_path, [], {}, body=trimesh.creation.box(extents=[1.0, 0.06, 0.06]))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {})
    plan = {**PLAN, "contacts": {"left": {"transmits": "force only", "regions": ["lb"]},
                                 "right": {"transmits": "force only", "regions": ["rb"]}},
            "reference_hold": {"regions": ["end"]}, "values": {"acceleration": [0, 0, 0]}}
    planned = [{"region": "top", "force_N": [0, -1000, 0]}, {"region": "lb", "force_N": [0, 500, 0]},
               {"region": "rb", "force_N": [0, 500, 0]}]

    def at(z):
        regions = _bar_regions()
        regions[0] = {**regions[0], "centroid": [0.5, 0.06, z]}
        return drive.statics(session, plan, regions, planned)
    near_edge = at(0.045)                       # 15 mm off the centre line: 15 N m, 13 % of 118.75
    assert "of twist left to the hold" in drive.statics_note(near_edge) and drive.statics_problem(near_edge) is None
    assert drive.statics_note(at(0.033)) is None          # 3 mm: printed, not reported


def _lipped_bar():
    """A 40 mm square bar 1 m long with a 4 mm lip 60 mm wide along it at mid-height: the lip is
    under the voxel thickness's floor (two voxels, 12.5 mm here)."""
    import trimesh
    profile = [(0, 0), (0.018, 0), (0.018, -0.06), (0.022, -0.06), (0.022, 0), (0.04, 0), (0.04, 0.04), (0, 0.04)]
    return drive.build_blank("extrude", length=1.0, axis="x",
                             profile=" ".join(f"{a},{b}" for a, b in profile))


def test_a_part_under_the_voxel_floor_is_measured_by_its_faces_and_has_the_declared_exit(tmp_path, gaps_not_read):
    lipped = _lipped_bar()
    leaf = 1.92 * 1.0 / 2 ** 8
    walls = drive.thin_walls(lipped, leaf)
    assert walls["share_of_surface"] > 0.2 and abs(walls["width_m"][0] - 0.004) < 0.001
    import trimesh
    assert drive.thin_walls(trimesh.creation.box(extents=[1.0, 0.04, 0.04]), leaf)["share_of_surface"] < 0.01
    real_i, real_a = (0.04 ** 4 - 0.036 ** 4) / 12, 0.04 ** 2 - 0.036 ** 2 + 0.004 * 0.06
    session = _built_session(tmp_path, lipped, 0.002, f"0.04 m deep, I {real_i:.6g} m4, A {real_a:.6g} m2")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850},
            "settings": {"remesh_octree_depth": 8},
            "body_simplification": {"route": "section", "about": "z", "load_path": "one section in bending",
                                    "kept": "k", "left_out": "l"}}
    resolution = drive.planned_resolution(session, plan)
    assert drive.thinnest_leaves(resolution) < 1
    # the thin part is warned with its numbers, declared or not; nothing is refused
    assert "thin member" in drive.resolution_warning(session, plan, resolution)
    declared = {**plan, "thin_off_load_path": "the lip lies on the neutral axis; nothing loads it"}
    assert "declared off the load path" in drive.resolution_warning(session, declared, resolution)


# 12000 is the runner's default count and takes 9 s (6000: 3 s): the same readings at half the
# samples run by default, the default count with -m slow.
@pytest.mark.parametrize("samples", [6000, pytest.param(12000, marks=pytest.mark.slow)])
def test_a_wall_is_measured_straight_through_on_round_square_and_hollow_sections(samples):
    """At an 8.25 mm leaf (2 leaves = 16.5 mm): round and square bars just over 2 leaves pass, under
    them read at their own thickness, a thin hollow-section wall and a 1.5-leaf plate as what they are."""
    import math
    import trimesh
    leaf, length = 1.92 * 1.1 / 2 ** 8, 1.1

    def reading(mesh):
        drive._THIN_WALLS.clear()
        return drive.thinnest_leaves({"thin_walls": drive.thin_walls(mesh, leaf, samples=samples)})
    round_bar = lambda d: drive.build_blank("cylinder", radius=d / 2, length=length, axis="x")
    assert reading(round_bar(0.0175)) is None                          # 2.1 leaves
    assert abs(reading(round_bar(0.0124)) - 1.5) <= 0.15
    assert reading(trimesh.creation.box(extents=[length, 0.0175, 0.0175])) is None
    assert abs(reading(trimesh.creation.box(extents=[length, 0.0124, 0.0124])) - 1.5) <= 0.15
    ring = " ".join(f"{0.033 * math.cos(k * math.pi / 24):.6f},{0.033 * math.sin(k * math.pi / 24):.6f}"
                    for k in range(48))
    hollow_round = drive.build_blank("extrude", length=length, axis="x", profile=ring, circles=["0,0,0.0289"])
    assert abs(reading(hollow_round) - 0.5) <= 0.1                             # a 4.1 mm wall
    assert abs(reading(trimesh.creation.box(extents=[length, 0.0124, 0.3])) - 1.5) <= 0.1
    assert reading(trimesh.creation.box(extents=[length, 0.0175, 0.3])) is None


def test_the_saved_stop_files_carry_no_absolute_path(tmp_path, monkeypatch, capsys):
    import re
    monkeypatch.setattr(lib, "REPO_ROOT", tmp_path)
    session = tmp_path / "output" / "abc"
    (session / "operator").mkdir(parents=True)
    where = session / "geometry" / "views.png"
    payload = {"ok": True, "stop": "geometry", "look_at": [str(where)],
               "card": {"kind": "geometry_physics", "session_dir": str(session),
                        "solve_log_path": str(session / "solve" / "run" / "log.txt")},
               "error": f"see {session / 'tool_receipts' / 'r.json'}"}
    drive.emit(payload, session)                               # last_event.json, as a card stop saves it
    for name in ("last_run.json", "stop_geometry.json", "stop_solve.json"):
        drive.save_stop(session, name, payload)
    capsys.readouterr()
    for name in ("last_event.json", "last_run.json", "stop_geometry.json", "stop_solve.json"):
        text = (session / "operator" / name).read_text()
        assert not re.search(ABSOLUTE, text), (name, text)
        assert "output/abc/geometry/views.png" in text or name == "last_event.json"


def test_the_route_to_a_body_never_reruns_an_earlier_attempt_unchanged(tmp_path, monkeypatch):
    (tmp_path / "operator").mkdir()
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    text = {"description": "a solid block", "generator": "triposg"}
    assert drive.route_order_problem(tmp_path, "geometry_input", text) is None
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"generation": 1})
    drive.record_route(tmp_path, "geometry_input", text)                 # recorded once it ran
    assert drive.route_order_problem(tmp_path, "geometry_input", text) is None      # a cached resend, still
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"generation": 1, "drawing": 1})
    assert "not re-run unchanged" in drive.route_order_problem(tmp_path, "geometry_input", text)
    # TripoSG text is tried once: new words after it are refused (the drawing is next)
    assert "tried once" in drive.route_order_problem(tmp_path, "geometry_input", {**text, "description": "a block, tall"})
    drawn = {"description": "a solid block", "use_drawn_image": True}
    plan = {"geometry": {"draw": True, "description": "a solid block", "image_seed": 0}}
    assert drive.route_order_problem(tmp_path, "geometry_input", drawn, plan=plan) is None
    drive.record_route(tmp_path, "geometry_input", drawn, plan=plan)
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"generation": 1, "drawing": 3, "hunyuan": 1})
    assert "moved on to hunyuan" in drive.route_order_problem(tmp_path, "geometry_input", drawn, plan=plan)
    # two generated bodies are the budget: a third is refused, and a built body needs no argument
    plan = {"geometry": {"draw": True, "description": "a solid block", "image_seed": 1}}
    assert "the budget: build the body" in drive.route_order_problem(tmp_path, "geometry_input", drawn, plan=plan)
    draw = {"action": "draw_image", "description": "a solid block", "image_seed": 2}
    assert drive.route_order_problem(tmp_path, "geometry_input", draw) is None      # a drawing is no body
    built = {"geometry": {"asset_path": "o/blank_box_0a1b2c3d.glb", "asset_kind": "surface"}}
    assert drive.built_body_problem(tmp_path, built) is None
    # Undeclared: the free-form steps in order, then the simplified body.
    assert "Hunyuan3D from the drawing" in drive.limit_message("drawing", {"drawing": 4})
    assert "simplified body" in drive.limit_message("hunyuan", {"hunyuan": 3})


def test_a_poor_body_from_a_passed_image_is_refined_on_the_same_image_first(tmp_path, monkeypatch):
    (tmp_path / "operator").mkdir()
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    plan = {"geometry": {"draw": True, "description": "a block", "image_seed": 0}}
    drawn = {"description": "a block", "use_drawn_image": True}
    text = {"description": "a block"}
    for answer in (text, drawn):                                         # two bodies: the budget
        assert drive.route_order_problem(tmp_path, "geometry_input", answer, plan=plan) is None
        drive.record_route(tmp_path, "geometry_input", answer, plan=plan)
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"generation": 1, "drawing": 1})
    assert drive.generated_bodies(tmp_path) == 2
    hint = drive.refine_hint(tmp_path, plan)
    assert "init_octree_depth 6, octree_depth 9, then 5/8, then Hunyuan3D" in hint
    fine = {**drawn, "generation_settings": {"init_octree_depth": 6, "octree_depth": 9}}
    assert drive.route_identity("geometry_input", fine, plan=plan, session=tmp_path).startswith("refine|")
    assert drive.route_order_problem(tmp_path, "geometry_input", fine, plan=plan) is None   # past the budget
    drive.record_route(tmp_path, "geometry_input", fine, plan=plan)
    assert drive.generated_bodies(tmp_path) == 2                         # the same attempt refined
    finer = {**drawn, "generation_settings": {"init_octree_depth": 5, "octree_depth": 8}}
    refined = {**plan, "geometry": {**plan["geometry"], "generation_settings": fine["generation_settings"]}}
    assert "5/8, then" not in drive.refine_hint(tmp_path, refined)
    assert "init_octree_depth 5, octree_depth 8, then Hunyuan3D" in drive.refine_hint(tmp_path, refined)
    # no watertight surface at that depth: one new seed, which is no refinement
    drive.note_no_surface(tmp_path, "geometry_input", finer, "ValueError: no usable watertight surface")
    assert drive.route_order_problem(tmp_path, "geometry_input", finer, plan=plan) is None
    seeded = {**drawn, "generation_settings": {**finer["generation_settings"], "seed": 7}}
    assert drive.route_order_problem(tmp_path, "geometry_input", seeded, plan=plan) is None
    lib.write_json(tmp_path / "operator" / drive.NO_SURFACE, {})
    other = {**drawn, "generation_settings": {**fine["generation_settings"], "seed": 3}}
    assert "extracted finer 2 times" in drive.route_order_problem(tmp_path, "geometry_input", other, plan=plan)
    # a new image past the budget is a new body: refused
    new = {"geometry": {**plan["geometry"], "image_seed": 1}}
    assert "the budget" in drive.route_order_problem(tmp_path, "geometry_input", drawn, plan=new)
    assert drive.refine_hint(tmp_path, {"geometry": {"description": "a block"}}) is None   # text: no ladder
    # rung (c): Hunyuan3D from the same drawing past the budget, once, no new body
    hunyuan = {**drawn, "generator": "hunyuan3d"}
    from_drawing = {"geometry": {**plan["geometry"], "generator": "hunyuan3d"}}
    assert drive.route_identity("geometry_input", hunyuan, "hunyuan3d", from_drawing, tmp_path).startswith("rung|")
    assert drive.route_order_problem(tmp_path, "geometry_input", hunyuan, "hunyuan3d", from_drawing) is None
    drive.record_route(tmp_path, "geometry_input", hunyuan, "hunyuan3d", from_drawing)
    assert drive.generated_bodies(tmp_path) == 2
    assert drive.route_order_problem(tmp_path, "geometry_input", hunyuan, "hunyuan3d", from_drawing) is None  # resend
    again = {**hunyuan, "generation_settings": {"seed": 4}}
    assert "taken once" in drive.route_order_problem(tmp_path, "geometry_input", again, "hunyuan3d", from_drawing)
    # Hunyuan3D from a drawing no body was made from is a new body: the budget
    assert "the budget" in drive.route_order_problem(
        tmp_path, "geometry_input", hunyuan, "hunyuan3d", {"geometry": {**new["geometry"], "generator": "hunyuan3d"}})
    # a web photograph alike: once TripoSG made a body from it, Hunyuan3D from it once
    photo = {"asset_kind": "image", "asset_path": str(tmp_path / "inputs" / "web" / "q" / "a" / "cut.png")}
    drive.record_route(tmp_path, "geometry_input", photo, None, {"geometry": photo})
    on_photo = {**photo, "generator": "hunyuan3d"}
    assert drive.route_order_problem(tmp_path, "geometry_input", on_photo, "hunyuan3d", {"geometry": on_photo}) is None
    drive.record_route(tmp_path, "geometry_input", on_photo, "hunyuan3d", {"geometry": on_photo})
    assert "taken once" in drive.route_order_problem(
        tmp_path, "geometry_input", {**on_photo, "generation_settings": {"seed": 2}}, "hunyuan3d", {"geometry": on_photo})


def test_the_brief_narrows_the_routes_and_the_runner_orders_them(tmp_path, monkeypatch):
    (tmp_path / "operator").mkdir()
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    assert drive.routes_problem(["sketch"]) and drive.routes_problem(["drawing", "triposg_text"]) is None
    # any order is taken and sorted: generated first, built last, or first when the brief argues it
    assert drive.canonical_routes(["built", "hunyuan3d", "drawing", "drawing"], tmp_path) == [
        "triposg_text", "drawing", "hunyuan3d", "web_image", "built"]     # text first; a photograph before built
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", ["triposg_text", "drawing", "built"])
    built = {**PLAN, "geometry": {"asset_path": "o/blank_bars_0a1b2c3d.glb", "asset_kind": "surface"}}
    said = drive.built_body_problem(tmp_path, built)
    assert "a built body comes after 2 generated bodies" in said and "'Why not generated:'" in said
    # the line is checked against the brief's own Real lines: a few members, or a wall under 1/50 (1.5 % here)
    brief = ("## Size and material\nWhy not generated: the walls are too thin for a generated surface\n"
             "Real thinnest member: 0.05 m\nReal extents: 1 x 0.5 x 0.5 m\n"
             'Load path: {"a": "series", "b": "series", "c": "series", "d": "series", "e": "series"}\n')
    (tmp_path / "operator" / "brief.md").write_text(brief)
    assert drive.built_body_problem(tmp_path, built)                     # 5 members, a 5 % member
    (tmp_path / "operator" / "brief.md").write_text(brief.replace("0.05 m", "0.015 m"))
    assert drive.built_body_problem(tmp_path, built) is None
    assert drive.canonical_routes(["built", "drawing"], tmp_path) == ["web_image", "built", "triposg_text", "drawing"]
    few = brief.replace(', "b": "series", "c": "series", "d": "series", "e": "series"', "")
    (tmp_path / "operator" / "brief.md").write_text(few.replace(', "b": "series"', ""))
    assert drive.built_body_problem(tmp_path, built)                     # one member, no section shape: no
    (tmp_path / "operator" / "brief.md").write_text(few + "Real section (a): rectangle 0.1 x 0.05 m\n")
    assert drive.built_body_problem(tmp_path, built) is None             # one prismatic member
    (tmp_path / "operator" / "brief.md").write_text("")
    # a list without a ready generated route is no way round it
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", ["built"])
    assert "no ready generated route listed" in drive.built_body_problem(tmp_path, built)
    # nor is the photograph, once tried: it is no generated route (["built"] is declared as below)
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", drive.canonical_routes(["built"], tmp_path))
    done = drive.web_image_done
    monkeypatch.setattr(drive, "web_image_done", lambda session: True)
    assert "no ready generated route listed" in drive.built_body_problem(tmp_path, built)
    monkeypatch.setattr(drive, "web_image_done", done)
    for routes in (["built"], ["web_image", "built"], ["web_image"]):
        assert "lists no generated route" in drive.routes_problem(routes, tmp_path)
    assert drive.routes_problem(["cadcoder", "built"], tmp_path) is None
    assert drive.routes_problem(["built"], tmp_path, generated=False) is None      # the user's own file
    (tmp_path / "operator" / "brief.md").write_text(few + "Real section (a): rectangle 0.1 x 0.05 m\n")
    assert drive.routes_problem(["built"], tmp_path) is None                       # built first, argued
    (tmp_path / "operator" / "brief.md").write_text("")
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", ["triposg_text", "drawing", "built"])
    # a CAD body counts in the budget of two
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"cad": 2})
    cad = {"description": "a solid block", "generator": "cadcoder"}
    assert "the budget: build the body" in drive.route_order_problem(tmp_path, "geometry_input", cad)
    assert "the budget" in drive.route_order_problem(tmp_path, "geometry_input", {**cad, "revise_prompt": True})
    assert drive.built_body_problem(tmp_path, built) is None
    lib.write_json(tmp_path / "operator" / "run_counts.json", {})
    (tmp_path / "operator" / "brief.md").write_text("")
    text = {"description": "a solid block", "generator": "triposg"}
    drawn = {"description": "a solid block", "use_drawn_image": True}
    hunyuan = {"description": "a solid block", "generator": "hunyuan3d", "use_drawn_image": True}
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", ["triposg_text", "drawing"])
    # A listed route waits for the listed ones before it; one the brief left out is refused.
    assert "untried" in drive.route_order_problem(tmp_path, "geometry_input", drawn)
    assert drive.route_order_problem(tmp_path, "geometry_input", text) is None
    assert "not among plan.geometry_routes" in drive.route_order_problem(tmp_path, "geometry_input", hunyuan)
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"generation": 1})
    assert drive.route_order_problem(tmp_path, "geometry_input", drawn) is None
    assert "the next route plan.geometry_routes lists, an SDXL drawing" in drive.limit_message(
        "generation", {"generation": 4}, ["triposg_text", "drawing"])
    assert "every route plan.geometry_routes lists has been tried" in drive.limit_message(
        "drawing", {"drawing": 4}, ["triposg_text", "drawing"])
    # The simplified body once every listed route had an attempt.
    simplified = {"kept": "k", "left_out": "l"}
    body = {**PLAN, "geometry": {"description": "a solid block", "generator": "triposg"},
            "body_simplification": simplified}
    assert "['drawing'] not yet" in drive.body_simplification_problem(tmp_path, body)
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"generation": 1, "drawing": 1})
    assert drive.body_simplification_problem(tmp_path, body) is None
    built = {**built, "body_simplification": simplified}
    assert drive.body_simplification_problem(tmp_path, built) is None     # every listed route tried
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", ["built"])
    assert "lists none" in drive.body_simplification_problem(tmp_path, body)
    # The plan declares the routes unless the user gave the file.
    plan = {k: v for k, v in PLAN.items() if k != "geometry_routes"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(drive.PlanError, match="needs 'geometry_routes'"):
        drive.load_plan(path)
    path.write_text(json.dumps({**plan, "geometry": {"asset_path": "data/part.stl", "asset_kind": "surface"}}))
    drive.load_plan(path)


def test_one_thin_member_measure_reads_the_faces_not_the_voxels(tmp_path):
    import trimesh
    leaf = 1.92 * 1.1 / 2 ** 8
    bar = trimesh.creation.box(extents=[1.1, 2.2 * leaf, 2.2 * leaf])      # just over two leaves, square
    resolution = {"thin_walls": drive.thin_walls(bar, leaf), "leaf_size_h": leaf, "depth": 8,
                  "member_thickness": {"thinnest_leaves": 1.9}}           # the voxel measure, a voxel short
    assert drive.thinnest_leaves(resolution) is None
    thin = trimesh.creation.box(extents=[1.1, 1.5 * leaf, 1.5 * leaf])
    resolution["thin_walls"] = drive.thin_walls(thin, leaf)
    assert abs(drive.thinnest_leaves(resolution) - 1.5) <= 0.15


def test_the_load_path_is_printed_in_numbers_beside_the_object():
    # A profile with two uprights joined across the top: two members cross a horizontal cut, one a vertical cut.
    profile = "0,0 0.1,0 0.1,0.8 0.9,0.8 0.9,0 1.0,0 1.0,0.9 0,0.9"
    joined = drive.build_blank("extrude", length=0.1, axis="z", profile=profile)
    numbers = drive.load_path_numbers(joined, 1.0)
    assert numbers["members_crossing_cuts"]["y"] == [2, 2, 2] and numbers["members_crossing_cuts"]["x"][1] == 1
    assert max(numbers["reach_beyond_base_m"].values()) < 0.01
    # An inverted-L profile reaching out beyond its foot.
    inverted_l = drive.build_blank("extrude", length=0.1, axis="z", profile="0,0 0.1,0 0.1,0.8 0.6,0.8 0.6,0.9 0,0.9")
    assert drive.load_path_numbers(inverted_l, 1.0)["reach_beyond_base_m"]["+x"] > 0.45


def test_the_governing_load_follows_from_the_runners_effects(tmp_path):
    # the Use table's states, added member by member on the runner's effects: the governing state at
    # the Real section's member is the runner's sum, never a name the agent picks
    from test_use_table import TABLE, session_with, table
    session = session_with(tmp_path, TABLE)
    coverage = drive.use_coverage(session / "operator")
    assert coverage["cases"]["working"] == {"top plate": 75.0, "side grip": 75.0, "bottom plate": 75.0}
    assert coverage["governing"] == "working" and coverage["run"] == "working" and coverage["unit"] == "N m"
    # another state larger at the Real section's member governs
    session = session_with(tmp_path, table(rows={"A3": {"enters": "top plate", "size": 400,
                                                        "source": "40 kg x g x 1"}}))
    coverage = drive.use_coverage(session / "operator")
    assert coverage["governing"] == "handling" and coverage["governing_effect"] == pytest.approx(120.0)
    # a run that is no state is refused
    said = drive.use_gate(session_with(tmp_path, table(run="lift")) / "operator")
    assert any('"run" is "lift": the state the solve applies' in x for x in said)


def test_the_report_prints_the_briefs_comparison_beside_the_loads_applied(tmp_path, monkeypatch):
    import inspect_session
    from test_use_table import TABLE, session_with
    session = session_with(tmp_path, TABLE)
    monkeypatch.setattr(inspect_session, "summarize", lambda session, **kw: {"regions": [
        {"name": "top", "category": "neumann", "applied_force": [0, -500.0, 0]},
        {"name": "base", "category": "dirichlet"}]})
    plan = {**PLAN, "values": {"acceleration": [0, -9.8, 0]},
            "loads": {"A1": {"regions": ["top"]}, "A2": {"regions": ["top"]}}}
    extra = drive.thin_declaration(session, plan)
    assert extra["coverage"]["run"] == "working" and extra["applied_loads"] == ["top [0.0, -500.0, 0.0] N", "self-weight"]
    assert extra["unbound_loads"] == []
    line = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, extra)
    assert "at top plate (the Real section's member): the run state 'working' 75 N m" in line
    assert "loads applied: top [0.0, -500.0, 0.0] N; self-weight" in line
    assert "- Use table A1 (working): push 200 N (H5 top), entering top plate" in line


def test_a_per_part_stand_in_is_held_to_the_valued_mass_not_a_plan_density(tmp_path):
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text("## Size and material\nReal mass: 10 kg\n")
    plan = {**PLAN, "material": {"mode": "parts", "description": "two"},
            "body_simplification": {"route": "cadcoder", "kept": "k", "left_out": "l"}}
    assert drive.stand_in_problem(tmp_path, plan) is None              # not yet valued: checked at judge
    lib.write_json(tmp_path / "operator" / "stop_solve.json", {"body": {"mass": 31.0}})
    assert "3.10 x the real" in drive.stand_in_problem(tmp_path, plan)


def test_a_plan_with_an_absolute_asset_path_is_refused(tmp_path):
    plan = {**PLAN, "geometry": {"asset_path": str(tmp_path / "b.glb"), "asset_kind": "surface"}}
    with pytest.raises(drive.PlanError, match="relative to the repository root"):
        _load(tmp_path, plan)


def test_the_runner_takes_the_fallback_a_solver_message_names_and_cross_checks_only_an_absolute_only_newton(tmp_path, monkeypatch, capsys):
    assert drive.backend_fallback("Linear solver failed ... it was still converging and ran out of iterations") == ("jaxamg", 4000)
    assert drive.backend_fallback("Linear solver failed ... stalled ... ill-conditioned") == ("pyamgx", None)
    assert drive.backend_fallback("Newton solver did not converge in 20 iterations") is None
    calls, converged, above = [], {"value": True}, {"value": False}

    def fake(args):
        # a finished run has a converged Newton (the solver raises otherwise): ok, or an error
        calls.append((args.backend, args.max_iters))
        print(json.dumps({"ok": converged["value"] or args.backend == "native", "backend": args.backend,
                          "newton": {"final_rel": 1.4 if above["value"] and args.backend != "native" else 4e-3,
                                     "rel_above_one": above["value"] and args.backend != "native"},
                          **({} if converged["value"] else {"error": "Newton solver did not converge"}),
                          "look_at": [f"x_{args.backend}.png"], "report": "r.md"}))
    monkeypatch.setattr(drive, "cmd_solve_backend", fake)
    args = argparse.Namespace(output_root=str(tmp_path), session_id="abc", gpu="auto", allow_busy_gpu=False)
    value = drive.solve_fallback(args, tmp_path, "Linear solver failed: ran out of iterations")
    # the converged fallback is the result: no native cross-check after it (43 s, unconverged, at depth 8)
    assert calls == [("jaxamg", 4000)] and "cross_check" not in value
    assert value["ok"] and value["fallback"]["backend"] == "jaxamg" and "its own check" in value["next"]
    # an ordinary solve stop: the fallback's own pictures and report at the top level
    assert value["look_at"] == ["x_jaxamg.png"] and value["report"] == "r.md" and value["why"]
    # a fallback that failed too: still no cross-check, the next change named
    converged["value"] = False
    calls.clear()
    value = drive.solve_fallback(args, tmp_path, "Linear solver failed ... stalled ... ill-conditioned")
    assert calls == [("pyamgx", None)] and not value["ok"] and "did not converge either" in value["next"]
    # a fallback "converged" on the absolute tolerance alone, above its starting residual: its
    # Newton checked nothing, so the native cross-check runs, with its own pictures
    converged["value"], above["value"] = True, True
    calls.clear()
    value = drive.solve_fallback(args, tmp_path, "Linear solver failed ... stalled ... ill-conditioned")
    assert calls == [("pyamgx", None), ("native", None)]
    assert value["cross_check"]["ok"] and value["cross_check"]["look_at"] == ["x_native.png"]
    assert "absolute tolerance" in value["next"] and value["look_at"] == ["x_pyamgx.png"]


def test_blank_says_a_thin_real_member_is_a_caveat(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest "
        "member: 0.002 m\nReal section: 0.05 m deep, I 2.0e-7 m4, A 3.8e-4 m2\n0.9 m tall\n## Prediction\nmembers in compression, 0.1 MPa, microns"))
    drive.cmd_blank(argparse.Namespace(
        output_root=str(tmp_path), session_id="abc", shape="box", size=[1.0, 0.05, 0.05], radius=None,
        length=None, axis="y", profile=None, hole=None, circle=None))
    note = printed[-1]["real_member"]["note"]
    assert "a resolution caveat the report carries" in note and "not a refusal" in note


def test_the_drawing_hunyuan3d_uses_is_its_own_step_and_each_drawing_is_its_own_attempt(tmp_path):
    (tmp_path / "operator").mkdir()
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"generation": 3, "drawing": 3, "hunyuan": 1})
    draw = {"action": "draw_image", "description": "a solid block", "image_seed": 1}
    assert drive.route_order_problem(tmp_path, "geometry_input", draw, "hunyuan3d") is None
    drive.record_route(tmp_path, "geometry_input", draw, "triposg")               # recorded once it ran
    assert "moved on to hunyuan" in drive.route_order_problem(tmp_path, "geometry_input", draw, "triposg")
    answer = {"use_drawn_image": True, "generator": "hunyuan3d", "description": "a solid block"}
    plan = lambda seed: {"geometry": {"draw": True, "description": "a solid block", "image_seed": seed}}
    first = drive.charge_attempt(tmp_path, "hunyuan", drive.attempt_identity("hunyuan", answer, plan(0)))
    second = drive.charge_attempt(tmp_path, "hunyuan", drive.attempt_identity("hunyuan", answer, plan(1)))
    again = drive.charge_attempt(tmp_path, "hunyuan", drive.attempt_identity("hunyuan", answer, plan(1)))
    assert first["counted"] and second["counted"] and not again["counted"]


def test_the_thin_wall_reading_does_not_change_with_which_walls_the_depth_counts():
    """A thick plate and a small thin fin on one body: at the coarser depth the plate is also under
    4 leaves, and a percentile over all thin walls would read the plate; the reading is the
    thinnest wall over a hundredth of the surface, the fin, at both depths."""
    import trimesh
    plate = trimesh.creation.box(extents=[1.0, 0.05, 0.6])
    fin = trimesh.creation.box(extents=[0.3, 0.06, 0.012])
    fin.apply_translation([0.0, 0.055, 0.0])
    body = trimesh.util.concatenate([plate, fin])
    readings = []
    for depth in (7, 8):
        leaf = 1.92 * 1.0 / 2 ** depth
        walls = drive.thin_walls(body, leaf)
        readings.append(drive.thinnest_leaves({"thin_walls": walls}) * leaf)
    assert all(abs(r - 0.012) < 0.002 for r in readings), readings


def test_a_step_the_runner_takes_inside_its_own_run_does_not_take_the_lock_again(tmp_path):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    with drive.session_lock(session):
        with drive.session_lock(session):          # the solver fallback inside a run
            pass
        assert str(session.resolve()) in drive._HELD_LOCKS
    assert str(session.resolve()) not in drive._HELD_LOCKS


def _cut_out(tmp_path, name, mask):
    """A drawing's cut-out: an RGBA image whose alpha is the given foreground mask."""
    from PIL import Image
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[..., 3] = np.where(mask, 255, 0)
    path = tmp_path / f"{name}.png"
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def test_a_drawing_is_counted_by_the_separate_objects_in_its_cut_out(tmp_path):
    size = 512
    one = np.zeros((size, size), dtype=bool)
    one[60:460, 180:330] = True                                   # one upright block
    assert drive.drawing_objects(_cut_out(tmp_path, "one", one))["objects"] == 1
    two = np.zeros((size, size), dtype=bool)
    two[60:460, 40:200] = True
    two[70:470, 300:460] = True                                   # two apart
    assert drive.drawing_objects(_cut_out(tmp_path, "two", two))["objects"] == 2
    specks = one.copy()
    specks[10:14, 10:14] = True
    specks[480:485, 480:486] = True                               # specks of the cut-out
    counted = drive.drawing_objects(_cut_out(tmp_path, "specks", specks))
    assert counted["objects"] == 1 and counted["specks"] == 2
    # Two upright figures touching low down make one alpha piece; the column valley and the dip of
    # the top edge between them still read two.
    touching = two.copy()
    touching[400:430, 200:300] = True
    assert drive.drawing_objects(_cut_out(tmp_path, "touching", touching))["side_by_side"]
    # One object with a top across two supports is one: its top edge has no dip.
    spanning = two.copy()
    spanning[60:110, 40:460] = True
    assert drive.drawing_objects(_cut_out(tmp_path, "spanning", spanning))["objects"] == 1
    assert drive.drawing_objects(tmp_path / "missing.png") is None


def test_a_drawing_is_shown_at_its_own_stop_before_anything_is_generated(tmp_path):
    plan = {**PLAN, "geometry": {"description": "a block", "generator": "triposg", "draw": True, "image_seed": 0}}
    answer, stage, stop = drive.plan_answer("geometry_input", {}, plan)
    assert answer["action"] == "draw_image" and stage == "drawing" and stop == "drawing"
    card = {"drawn_image": "d.png", "drawn_image_prompt": "a block", "image_seed": 0}
    answer, stage, stop = drive.plan_answer("geometry_input", card, plan)       # unchanged: generate
    assert answer.get("use_drawn_image") and stop == "geometry" and drive.drawn_from_plan(card, plan)
    for changed in ({"description": "a solid block"}, {"image_seed": 3}):
        moved = {**plan, "geometry": {**plan["geometry"], **changed}}
        assert not drive.drawn_from_plan(card, moved)
        assert drive.plan_answer("geometry_input", card, moved)[2] == "drawing"   # drawn again, shown again


def test_a_drawing_the_count_reads_as_several_is_never_generated_from_whole(tmp_path):
    # A silhouette that splits (a U with its prongs up) reads 2, as several objects do.
    u = np.zeros((256, 256), dtype=bool)
    u[200:240, 30:226] = True
    u[30:240, 30:70] = True
    u[30:240, 186:226] = True
    drawing = _cut_out(tmp_path, "u", u)
    assert drive.drawing_objects(drawing)["objects"] == 2
    card = {"drawn_image": str(drawing), "drawn_image_prompt": "a block", "image_seed": 0}
    payload = drive._stop_payload("drawing", tmp_path, tmp_path, card, PLAN)
    assert "never generated from whole" in payload["drawing"]["note"]
    plan = {**PLAN, "geometry": {"description": "a block", "generator": "triposg", "draw": True, "image_seed": 0}}
    # refused every time, run on unchanged or not, and for Hunyuan3D as for TripoSG
    for generator in ("triposg", "triposg", "hunyuan3d"):
        moved = {**plan, "geometry": {**plan["geometry"], "generator": generator}}
        with pytest.raises(drive.PlanError, match="reads as 2 objects"):
            drive.plan_answer("geometry_input", card, moved)
    # its largest object alone is taken
    cut = drive.cut_largest_object(drawing, drive.drawing_objects(drawing))
    assert cut is not None and cut.exists()
    answer, stage, stop = drive.plan_answer(
        "geometry_input", card, {**plan, "geometry": {**plan["geometry"], "drawing_crop": "largest"}})
    assert answer["asset_path"] == str(cut) and stop == "geometry"
    # The runner's forced redraw is gone: nothing in the run swaps the seed or refuses the drawing.
    import inspect
    assert not hasattr(drive, "redraw_next_seed") and not hasattr(drive, "drawing_seed_plan")
    assert "drawing_objects" not in inspect.getsource(drive.cmd_run)


def test_the_drawing_stop_warns_of_words_sdxl_never_read(tmp_path):
    block = np.zeros((64, 64), dtype=bool)
    block[16:48, 16:48] = True
    drawing = str(_cut_out(tmp_path, "block", block))
    card = {"drawn_image": drawing, "drawn_image_prompt": "a long object", "image_seed": 0,
            "drawn_image_dropped": "plain white background, product photo", "drawn_image_description_cut": False}
    payload = drive._stop_payload("drawing", tmp_path, tmp_path, card, PLAN)
    assert len(payload["warnings"]) == 1 and "plain white background" in payload["warnings"][0]
    assert "single-object style" in payload["warnings"][0] and "about 20 words" in payload["warnings"][0]
    card["drawn_image_description_cut"] = True
    assert "words of your description" in drive._stop_payload("drawing", tmp_path, tmp_path, card, PLAN)["warnings"][0]
    assert "warnings" not in drive._stop_payload("drawing", tmp_path, tmp_path,
                                                 {"drawn_image": drawing, "drawn_image_prompt": "a block",
                                                  "image_seed": 0}, PLAN)


def test_the_geometry_stop_warns_of_a_thin_member_and_refuses_nothing(tmp_path, monkeypatch):
    calls = []

    def warning(session, plan, resolution):
        depth = (plan.get("settings") or {}).get("remesh_octree_depth", 7)
        calls.append(depth)
        return f"thin member: the body's thinnest member is {1.4 if depth == 7 else 1.9} leaves at depth {depth}"
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: {"depth": (p.get("settings") or {}).get("remesh_octree_depth", 7)})
    monkeypatch.setattr(drive, "resolution_warning", warning)
    monkeypatch.setattr(drive, "thin_member_limit", lambda *a: None)
    monkeypatch.setattr(drive, "_generated_mesh", lambda s: None)
    monkeypatch.setattr(drive, "_inspect", lambda *a, **k: {})
    monkeypatch.setattr(drive.inspect_session if hasattr(drive, "inspect_session") else inspect_session,
                        "summarize", lambda *a, **k: {"warnings": ["another"]})
    (tmp_path / "geometry").mkdir(exist_ok=True)
    payload = drive._stop_payload("geometry", tmp_path, tmp_path, {"kind": "geometry_texture"}, PLAN)
    assert payload["warnings"][0].startswith("thin member: the body's thinnest member is 1.4 leaves at depth 7")
    assert payload["warnings"][1] == "another" and payload.get("ok") is not False
    assert payload["planned_resolution"]["resolution_check"].endswith("at depth 8: under it too")


def test_the_drawing_and_texture_stops_show_the_current_pictures(tmp_path):
    two = np.zeros((256, 256), dtype=bool)
    two[30:220, 20:100] = True
    two[30:220, 150:230] = True
    (tmp_path / "inputs" / "sdxl").mkdir(parents=True)                # where drawings live
    drawing = _cut_out(tmp_path / "inputs" / "sdxl", "drawing", two)
    card = {"drawn_image": str(drawing), "drawn_image_prompt": "a block", "image_seed": 4}
    payload = drive._stop_payload("drawing", tmp_path, tmp_path, card, PLAN)
    cut = drive.largest_object_path(drawing)
    assert payload["look_at"] == [str(drawing), str(cut)] and payload["drawing"]["objects"] == 2
    assert "new words or a new image_seed draw again" in payload["drawing"]["note"]
    # the largest object alone, on a transparent square: one piece, offered as drawing_crop
    from PIL import Image
    alpha = np.asarray(Image.open(cut).getchannel("A")) > 127
    assert drive.mask_pieces(alpha)["objects"] == 1 and payload["drawing"]["largest_object"] == str(cut)
    plan = {**PLAN, "geometry": {"description": "a block", "draw": True, "image_seed": 4,
                                 "drawing_crop": "largest"}}
    answer, stage, stop = drive.plan_answer("geometry_input", card, plan)
    assert answer["asset_path"] == str(cut) and answer["asset_kind"] == "image" and stage is None
    assert "drawing_crop" not in answer and drive.route_step("geometry_input", answer) == "drawing"
    assert drive.attempt_stage("geometry_input", answer) is None       # the drawing's own attempt
    hun = {**plan, "geometry": {**plan["geometry"], "generator": "hunyuan3d"}}
    answer, stage, _ = drive.plan_answer("geometry_input", card, hun)
    assert stage == "hunyuan" and drive.route_step("geometry_input", answer) == "hunyuan"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**PLAN, "geometry": {"description": "a block", "drawing_crop": "largest"}}))
    with pytest.raises(drive.PlanError, match="drawing_crop"):
        drive.load_plan(path)                                          # only beside draw true
    painted, views = tmp_path / "painted.glb", tmp_path / "views.png"
    painted.write_text("x")
    views.write_text("x")
    card = {"textured_surface": str(painted), "texture_views": str(views), "texture_prompt": "grey stone",
            "texture_seed": 0}
    payload = drive._stop_payload("texture", tmp_path, tmp_path, card, PLAN)
    assert payload["look_at"] == [str(views)] and payload["texture"]["texture_prompt"] == "grey stone"
    brief = drive.brief_run({"stop": "texture", **payload})
    assert brief["texture"] and brief["look_at"]


def test_each_drawing_hunyuan_is_given_is_one_attempt(tmp_path):
    (tmp_path / "operator").mkdir()

    def charged(plan, card):
        answer, stage, _ = drive.plan_answer("geometry_input", card, plan)
        return answer, drive.hunyuan_charge(tmp_path, plan, answer, drive.drawing_charge(tmp_path, plan, answer, stage))

    hun = {**PLAN, "geometry": {"description": "a block", "generator": "hunyuan3d", "draw": True, "image_seed": 2}}
    # Drawn for Hunyuan3D: one attempt, counted now, whether or not the drawing is taken.
    draw, stage = charged(hun, {})
    assert draw["action"] == "draw_image" and stage == "hunyuan"
    # The body generated from that drawing is the same attempt; a second body from it is another.
    card = {"drawn_image": "d.png", "drawn_image_prompt": "a block", "image_seed": 2}
    use, stage = charged(hun, card)
    assert use["use_drawn_image"] and stage is None
    assert charged(hun, card)[1] == "hunyuan"
    # A good drawing taken over from the drawing route (the words and seed kept) is one attempt,
    # counted when Hunyuan3D generates from it; it is not drawn again.
    (tmp_path / "operator" / "hunyuan_drawings.json").unlink(missing_ok=True)
    taken = {**PLAN, "geometry": {"description": "a lamp", "generator": "hunyuan3d", "draw": True, "image_seed": 0}}
    use, stage = charged(taken, {"drawn_image": "d.png", "drawn_image_prompt": "a lamp", "image_seed": 0})
    assert use["use_drawn_image"] and stage == "hunyuan"


def test_check_one_judges_a_declared_simplification_by_its_declaration():
    text = (SCRIPTS.parent / "references" / "visual_checks.md").read_text(encoding="utf-8")
    assert "thickened into\n   another object" in text or "thickened into another object" in text
    assert "judged by its\n   own declaration" in text or "judged by its own declaration" in text
    assert "a thickened body fails" not in text
    geometry = (SCRIPTS.parent / "references" / "geometry.md").read_text(encoding="utf-8")
    routes = (SCRIPTS.parent / "references" / "objectives" / "routes.md").read_text(encoding="utf-8")
    assert "never a URL" in routes and "web photograph" in geometry


def _web_session(tmp_path, bodies=2):
    """A session whose listed generated routes made ``bodies`` bodies (the budget spent at 2)."""
    (tmp_path / "operator").mkdir(parents=True)
    lib.write_json(tmp_path / "operator" / "geometry_routes.json",
                   ["triposg_text", "drawing", "hunyuan3d", "web_image", "built"])
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"generation": 1, "drawing": 1})
    lib.write_json(tmp_path / "operator" / drive.ROUTE_SENT, [f"generation:{k}" for k in range(bodies)])


def _found(searched, candidates=1, found=None):
    """A stand-in for find_web_images: ``candidates`` survivors of the pre-screen."""
    def find(args, session, output_root, compute_env, query):
        searched.append(query)
        folder = drive.web_folder(session, query)
        shown = []
        for k in range(candidates):
            (folder / f"commons-{k}").mkdir(parents=True)
            (folder / f"commons-{k}" / "cut.png").write_bytes(b"png")
            lib.write_json(folder / f"commons-{k}" / "attribution.json",
                           {"credit": '"Item" by A, CC0 1.0', "id": f"commons-{k}", "retrieved": "2026-10-11"})
            shown.append({"id": f"commons-{k}", "image": str(folder / f"commons-{k}" / "cut.png"),
                          "thumb": str(folder / f"commons-{k}" / "cut.png"), "licence": "CC0 1.0"})
        out = {"query": query, "found": candidates if found is None else found, "sources": [],
               "candidates": shown, **({"montage": shown[0]["thumb"]} if shown else {})}
        folder.mkdir(parents=True, exist_ok=True)
        lib.write_json(folder / "candidates.json", out)
        return out
    return find


def test_a_web_photograph_is_searched_once_shown_then_picked_with_its_own_body(tmp_path, monkeypatch):
    _web_session(tmp_path)
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    assert drive.RUN_LIMITS["web_image"] == 2 and drive.ROUTE_ORDER[-1] == "web_image"
    assert "web photograph" in drive.limit_message("hunyuan", {"hunyuan": 3})
    searched = []
    monkeypatch.setattr(drive, "find_web_images", _found(searched))
    plan = {**PLAN, "geometry": {"description": "an item", "web_image": {"query": "item"}}}
    # past the generated budget the photograph needs no untried generated route first
    out = drive.web_image_route(None, tmp_path, tmp_path, "env", plan)
    assert out["stop"] == "web_image" and out["extra"]["look_at"] and searched == ["item"]
    assert lib.read_json(tmp_path / "operator" / "run_counts.json")["web_image"] == 1
    assert drive.web_image_route(None, tmp_path, tmp_path, "env", plan)["stop"] == "web_image"
    assert searched == ["item"]                                          # shown again, not searched again
    other = {**plan, "geometry": {**plan["geometry"], "web_image": {"query": "item, whole"}}}
    assert "one web search a session" in drive.web_image_route(None, tmp_path, tmp_path, "env", other)["extra"]["error"]
    picked = {**plan, "geometry": {**plan["geometry"], "web_image": {"query": "item", "pick": "commons-0"}}}
    geometry = drive.web_image_route(None, tmp_path, tmp_path, "env", picked)["plan"]["geometry"]
    assert geometry["asset_kind"] == "image" and drive.WEB_DIR in geometry["asset_path"] and "web_image" not in geometry
    answer, stage, stop = drive.plan_answer("geometry_input", {}, {**PLAN, "geometry": geometry})
    assert stage is None and stop == "geometry"                          # counted at the search
    assert drive.route_step("geometry_input", answer) == "web_image"
    assert drive.attempt_stage("geometry_input", answer) is None
    assert drive.route_order_problem(tmp_path, "geometry_input", answer, plan={**PLAN, "geometry": geometry}) is None
    drive.record_route(tmp_path, "geometry_input", answer, plan={**PLAN, "geometry": geometry})
    assert drive.generated_bodies(tmp_path) == 2 and drive.web_body_made(tmp_path)   # outside the budget
    assert drive.built_body_problem(tmp_path, {"geometry": {"asset_path": "o/blank_box_0a1b2c3d.glb",
                                                            "asset_kind": "surface"}}) is None
    # its one body made: a finer extraction of the same photograph is a refinement, another search is not
    fine = {**answer, "generation_settings": {"init_octree_depth": 6, "octree_depth": 9}}
    assert drive.route_order_problem(tmp_path, "geometry_input", fine, plan={**PLAN, "geometry": geometry}) is None
    assert "its budget" in drive.route_order_problem(tmp_path, "geometry_input", {"asset_path": drive.WEB_DIR + "x",
                                                                                  "asset_kind": "image"})
    bad = {**plan, "geometry": {**plan["geometry"], "web_image": {"query": "item", "pick": "met-9"}}}
    assert "none of the candidates" in drive.web_image_route(None, tmp_path, tmp_path, "env", bad)["extra"]["error"]
    # the credit reaches the stop and the report through the generating answer in the driver log
    lib.append_jsonl(tmp_path / "operator" / "driver_log.jsonl", {"kind": "geometry_input", "ok": True, "answer": answer})
    assert drive.web_image_credit(tmp_path)["credit"].startswith('"Item"')
    assert "generated from a photograph" in drive.history_section(
        [{"run": "r", "body": {"sha": "s"}, "ends": "e", "load_N": 1.0}], None,
        {"web_image": drive.web_image_credit(tmp_path)})


@pytest.mark.web_rule
def test_a_built_body_waits_for_the_web_photograph_unless_none_was_usable(tmp_path, monkeypatch):
    _web_session(tmp_path)
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    built = {"geometry": {"asset_path": "o/blank_box_0a1b2c3d.glb", "asset_kind": "surface"}}
    said = drive.built_body_problem(tmp_path, built)                     # past the generated budget, still
    assert "after a web photograph was tried" in said and "plan.geometry.web_image" in said
    assert "a web photograph" in drive.budget_spent(tmp_path)
    # with the brief's argument for building first, the photograph still comes first
    (tmp_path / "operator" / "brief.md").write_text(
        "## Size and material\nWhy not generated: the walls are too thin for a generated surface\n"
        "Real thinnest member: 0.005 m\nReal extents: 1 x 0.5 x 0.5 m\n")
    lib.write_json(tmp_path / "operator" / drive.ROUTE_SENT, [])
    assert drive.built_first_ok(tmp_path) and "after a web photograph" in drive.built_body_problem(tmp_path, built)
    # nothing found: no usable photograph is recorded, built follows at once, one more search is offered
    searched = []
    monkeypatch.setattr(drive, "find_web_images", _found(searched, candidates=0))
    plan = {**PLAN, "geometry": {"description": "an item", "web_image": {"query": "item"}}}
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", ["web_image", "built", "drawing"])
    out = drive.web_image_route(None, tmp_path, tmp_path, "env", plan)
    assert out["stop"] == "web_image" and out["extra"]["web_image"]["no_usable_photo"]
    assert "build the body now" in out["extra"]["web_image"]["next"]
    assert "other words" in out["extra"]["web_image"]["next"]
    assert drive.web_image_done(tmp_path) and drive.built_body_problem(tmp_path, built) is None
    second = {**plan, "geometry": {**plan["geometry"], "web_image": {"query": "item, whole"}}}
    assert drive.web_image_route(None, tmp_path, tmp_path, "env", second)["stop"] == "web_image"
    assert searched == ["item", "item, whole"]
    # candidates that all failed the pre-screen: no usable photograph, and no second search
    other = tmp_path / "other"
    _web_session(other)
    monkeypatch.setattr(drive, "find_web_images", _found(searched, candidates=0, found=4))
    out = drive.web_image_route(None, other, other, "env", plan)
    assert "other words" not in out["extra"]["web_image"]["next"] and drive.web_image_done(other)
    # turned down: done too
    third = tmp_path / "third"
    _web_session(third)
    monkeypatch.setattr(drive, "find_web_images", _found(searched))
    drive.web_image_route(None, third, third, "env", plan)
    assert drive.built_body_problem(third, built)
    none = {**plan, "geometry": {**plan["geometry"], "web_image": {"query": "item", "pick": "none"}}}
    # "none" names each shown photograph and the check it failed, or is refused unspent
    for why in (None, "all bad", "commons-0 looks odd"):
        said = drive.web_image_route(None, third, third, "env", {**none, "geometry": {
            **none["geometry"], "web_image": {**none["geometry"]["web_image"], **({"why": why} if why else {})}}})
        assert said["stop"] == "error" and "the check the photographs failed" in said["extra"]["error"]
        assert "['commons-0']" in said["extra"]["error"] and drive.built_body_problem(third, built)
    none["geometry"]["web_image"]["why"] = "commons-0: a print, not the object itself"
    out = drive.web_image_route(None, third, third, "env", none)["extra"]["web_image"]
    assert "build the body" in out["next"] and out["turned_down"].startswith("commons-0")
    assert drive.built_body_problem(third, built) is None
    assert drive.web_search_line(third) == "'item': turned down: commons-0: a print, not the object itself"
    assert drive.turned_down_problem(["a-1", "b-2"], "a-1: a person in front; b-2: the cut-out is incomplete") is None
    assert "['b-2']" in drive.turned_down_problem(["a-1", "b-2"], "a-1: not one whole object; b-2: odd")
    # one reason shared by every photograph shown is taken
    assert drive.turned_down_problem(["a-1", "b-2"], "none shows the whole object alone") is None
    # the query names the object: a word of the description or the instruction, not filler
    fourth = tmp_path / "fourth"
    _web_session(fourth)
    for query in ("photo of a real object", "unrelated words"):
        off = {**plan, "geometry": {**plan["geometry"], "web_image": {"query": query}}}
        said = drive.web_image_route(None, fourth, fourth, "env", off)
        assert said["stop"] == "error" and "names no word of plan.geometry.description" in said["extra"]["error"]
    assert not (lib.read_json(fourth / "operator" / "run_counts.json") or {}).get("web_image")   # unspent
    drive.store_instruction(fourth, "Simulate a heavy block resting on the floor")
    blocks = {**plan, "geometry": {**plan["geometry"], "web_image": {"query": "steel blocks"}}}
    out = drive.web_image_route(None, fourth, fourth, "env", blocks)
    assert out["stop"] == "web_image", out
    assert drive.web_search_line(fourth) == "'steel blocks': 1 shown"
    # no idle GPU: the search is refunded, run again; a failure that recurs (no CUDA, no interpreter,
    # RMBG-1.4 missing offline) is no usable photograph, so the body may be built
    for name, error in (("busy", drive.NoIdleGPU("no idle GPU after waiting", [])),
                        ("broken", drive.WebProcessingFailed("no CUDA device"))):
        folder = tmp_path / name
        _web_session(folder)

        def fail(*a, error=error):
            raise error
        monkeypatch.setattr(drive, "find_web_images", fail)
        out = drive.web_image_route(None, folder, folder, "env", plan)
        counted = (lib.read_json(folder / "operator" / "run_counts.json") or {}).get("web_image", 0)
        if name == "busy":
            assert out["stop"] == "error" and "run again" in out["extra"]["error"] and counted == 0
            assert not drive.web_image_done(folder)
        else:
            assert out["stop"] == "web_image" and counted == 1 and drive.web_image_done(folder)
            assert out["extra"]["web_image"]["no_usable_photo"] == "processing failed: no CUDA device"
            assert drive.built_body_problem(folder, built) is None


def test_the_plan_s_web_image_is_words_and_a_pick_only(tmp_path):
    for web, match in (({"query": ""}, "web_image is"), ({"query": "x", "url": "https://x"}, "web_image is"),
                       ({"query": "https://x/y.jpg"}, "never a URL"), ({"query": "x", "pick": "a"}, None)):
        path = tmp_path / "plan.json"
        path.write_text(json.dumps({**PLAN, "geometry": {"description": "x", "web_image": web}}))
        if match:
            with pytest.raises(drive.PlanError, match=match):
                drive.load_plan(path)
        else:
            assert drive.load_plan(path)["geometry"]["web_image"]["pick"] == "a"
    path.write_text(json.dumps({**PLAN, "geometry": {"description": "x", "draw": True, "web_image": {"query": "x"}}}))
    with pytest.raises(drive.PlanError, match="leave draw and asset_path out"):
        drive.load_plan(path)
    for written in ("output/fem_workflow_operator/x/inputs/web/abc/commons-1/cut.png",
                    "output\\fem_workflow_operator\\x\\inputs\\web\\abc\\cut.png"):
        path.write_text(json.dumps({**PLAN, "geometry": {"description": "x", "asset_path": written,
                                                         "asset_kind": "image"}}))
        with pytest.raises(drive.PlanError, match="cannot name a web photograph"):
            drive.load_plan(path)
    # the search is the runner's alone, through web_images.py: drive.py reaches no URL of its own
    source = (SCRIPTS / "drive.py").read_text()
    assert "urllib" not in source and "import requests" not in source


def test_the_photographs_are_cropped_cut_out_and_screened_before_three_are_shown(tmp_path, monkeypatch):
    import subprocess
    import web_images
    from PIL import Image
    (tmp_path / "operator").mkdir()
    shown = [web_images._candidate("commons", k, f"Item {k}", "A", "p", "CC0 1.0", web_images.CC0, "u", "item")
             for k in range(5)]
    monkeypatch.setattr(web_images, "search", lambda query, ledger: {
        "candidates": shown, "sources": [{"source": "commons", "passed": 5}, {"source": "met", "error": "x"}]})

    def download(candidate, folder, ledger):
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red").save(folder / "original.png")
        return {**candidate, "original": folder / "original.png"}
    monkeypatch.setattr(web_images, "download", download)
    monkeypatch.setattr(drive, "select_gpu", lambda *a, **k: {"index": 0})
    monkeypatch.setattr(inspect_session, "compute_interpreter", lambda env: "python")
    ran = []

    def run(command, **kw):
        ran.append(command)
        assert command[2:5] == ["process", "--phrase", "item"] and kw["env"]["HF_HUB_OFFLINE"] == "1"
        for k, folder in enumerate(command[6::2]):
            Image.new("RGB", (40, 30), "red").save(Path(folder) / "thumb.png")
            lib.write_json(Path(folder) / "screen.json",
                           {"reasons": ["2 large pieces: several objects, or one cut apart"] if k == 1 else []})
        # a library's own line before the JSON: the last JSON line is read
        return types.SimpleNamespace(returncode=0, stdout='{loading} weights\n{"processed": [], "detector": null, '
                                                          '"note": "no crop"}\n', stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    found = drive.find_web_images(None, tmp_path, tmp_path, "env", "item")
    assert len(ran) == 1 and [c["id"] for c in found["candidates"]] == ["commons-0", "commons-2", "commons-3"]
    assert found["dropped"] == [{"id": "commons-1", "why": "2 large pieces: several objects, or one cut apart"}]
    assert found["note"] == "no crop" and "unreachable" not in found
    montage = drive.lib.REPO_ROOT / found["montage"]
    assert montage.exists() and Image.open(montage).size == (3 * 40 + 2 * 8, 30 + 22)
    # a run again (after no idle GPU) reuses the downloads: no second search or download
    monkeypatch.setattr(web_images, "search", lambda query, ledger: pytest.fail("searched again"))
    monkeypatch.setattr(web_images, "download", lambda *a: pytest.fail("downloaded again"))
    assert len(drive.find_web_images(None, tmp_path, tmp_path, "env", "item")["candidates"]) == 3
    # a failure that recurs every run is WebProcessingFailed (the route records it), never "run again"
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: types.SimpleNamespace(
        returncode=1, stdout="", stderr="Traceback ...\nRuntimeError: no CUDA device: RMBG-1.4 cuts out on a GPU only\n"))
    with pytest.raises(drive.WebProcessingFailed, match="no CUDA device"):
        drive.find_web_images(None, tmp_path, tmp_path, "env", "item")
    monkeypatch.setattr(inspect_session, "compute_interpreter", lambda env: None)
    with pytest.raises(drive.WebProcessingFailed, match="no interpreter"):
        drive.find_web_images(None, tmp_path, tmp_path, "env", "item")


def test_a_planned_painting_a_coloured_body_passed_by_is_said_at_the_geometry_stop(tmp_path):
    import trimesh
    (tmp_path / "geometry").mkdir()
    (tmp_path / "operator").mkdir()
    body = trimesh.creation.box()
    body.visual.vertex_colors = np.tile([[180, 120, 60, 255], [90, 60, 30, 255]], (4, 1))
    body.export(tmp_path / "geometry" / "generated.glb")
    plan = {**PLAN, "texture": {"texture_prompt": "light maple"}}
    note = drive.texture_not_applied(tmp_path, {"kind": "geometry_physics"}, plan)
    assert note["applied"] is False and "paints only a colourless surface" in note["note"]
    assert "views.png" not in note["note"]                     # no colours judged at the geometry stop
    assert drive.texture_not_applied(tmp_path, {"kind": "geometry_physics"}, PLAN) is None
    assert drive.texture_not_applied(tmp_path, {"kind": "geometry_texture"}, plan) is None
    lib.write_json(tmp_path / "operator" / "run_sent.json", {"texture": "{}"})    # the card was answered
    assert drive.texture_not_applied(tmp_path, {"kind": "geometry_physics"}, plan) is None
    (tmp_path / "operator" / "run_sent.json").unlink()
    trimesh.creation.box().export(tmp_path / "geometry" / "generated.glb")         # colourless: no painter here
    assert "painter is not available" in drive.texture_not_applied(tmp_path, {"kind": "geometry_physics"}, plan)["note"]


def test_the_texture_words_are_checked_with_the_plan_before_any_paint():
    """One colour and a few words, per part in parts mode; the refusal carries the rule."""
    ok = lambda texture, material=None: drive.check_plan(  # noqa: E731
        {**PLAN, "texture": texture, **({"material": material} if material else {})})
    ok({"texture_prompt": "light grey stone"})
    ok({"texture_prompt": "medium reddish-brown glazed porcelain", "texture_seed": 3})
    ok({"texture_prompt": "red painted metal, red all over"})                   # one colour twice
    with pytest.raises(drive.PlanError, match=r"11 words; .*<colour> <material>"):
        ok({"texture_prompt": "a rough weathered cast concrete surface with fine pores and marks"})
    with pytest.raises(drive.PlanError, match=r"colours \['black', 'green'\].*texture_seed"):
        ok({"texture_prompt": "dark green painted shaft, black grip"})
    with pytest.raises(drive.PlanError, match="whole number"):
        ok({"texture_prompt": "light grey stone", "texture_seed": "1"})
    with pytest.raises(drive.PlanError, match="texture_prompt"):
        ok({"texture_seed": 1})
    parts = {"mode": "parts", "description": "steel part one, wood part two"}
    ok({"texture_prompt": "mid grey steel frame, light yellow pine seat"}, parts)
    with pytest.raises(drive.PlanError, match="in one part"):
        ok({"texture_prompt": "mid grey and blue steel frame, light yellow pine seat"}, parts)


def test_one_colour_of_two_colour_words_passes_and_two_joined_colours_do_not(tmp_path):
    ok = lambda texture, material=None: drive.check_plan(  # noqa: E731
        {**PLAN, "texture": texture, **({"material": material} if material else {})})
    # adjacent or hyphenated colour words are one colour
    for words in ("olive green painted steel", "blue grey steel", "silver grey aluminium", "charcoal grey steel",
                  "golden brown oak", "creamy white ceramic", "rust red steel", "reddish-brown clay",
                  "blue-grey steel", "grey green stone", "blue green glass", "light blue steel",
                  "pale grey concrete"):
        ok({"texture_prompt": words})
    # two basic hues side by side, or apart by a comma or slash, are two
    for words in ("red white steel", "red, white steel", "black/white steel", "red-white steel",
                  "black grey steel", "red green steel"):
        with pytest.raises(drive.PlanError, match="colours"):
            ok({"texture_prompt": words})
    # colours joined by and / with / & / -and-, or apart with a word between, are two
    for words in ("red-and-white painted steel", "red and white painted steel", "red & white steel",
                  "grey steel with brown wood", "grey steel plate, brown wood rail"):
        with pytest.raises(drive.PlanError, match="colours"):
            ok({"texture_prompt": words})
    ok({"texture_prompt": "blue grey steel", "texture_seed": None})            # no seed: the default
    parts = {"mode": "parts", "description": "steel plate, wood rail"}
    ok({"texture_prompt": "grey steel plate and brown wood rail"}, parts)    # parts joined by "and"
    ok({"texture_prompt": "olive green steel frame with golden brown oak seat"}, parts)
    # a side with one word besides its colour is still a part
    ok({"texture_prompt": "grey steel plate and brown rail"}, parts)
    ok({"texture_prompt": "silver aluminium and black plastic"}, parts)
    assert drive.texture_parts("grey steel plate and brown rail") == ["grey steel plate", "brown rail"]
    for words in ("black and grey steel frame", "grey steel frame with white", "mid grey and blue steel frame"):
        with pytest.raises(drive.PlanError, match="in one part"):
            ok({"texture_prompt": words}, parts)
    with pytest.raises(drive.PlanError, match="in one part"):
        ok({"texture_prompt": "red and white steel plate, brown wood rail"}, parts)


def test_a_texture_already_sent_or_painted_is_not_checked_again(tmp_path):
    (tmp_path / "operator").mkdir()
    old = {"texture_prompt": "a weathered dark green painted shaft with a black rubber grip"}
    plan = {**PLAN, "texture": old}
    with pytest.raises(drive.PlanError, match="colours"):
        drive.check_plan(plan, tmp_path)
    lib.write_json(tmp_path / "operator" / "run_sent.json", {"texture": drive._normalized(old)})
    assert drive.check_plan(plan, tmp_path)["texture"] == old                  # a resumed session runs on
    (tmp_path / "operator" / "run_sent.json").unlink()
    lib.write_json(tmp_path / "operator" / drive.PAINT_AHEAD, {"answer": {"action": "texture_surface", **old}})
    assert drive.check_plan(plan, tmp_path)["texture"] == old
    with pytest.raises(drive.PlanError, match="colours"):                      # a changed one is checked
        drive.check_plan({**PLAN, "texture": {**old, "texture_seed": 2}}, tmp_path)


def test_a_painting_waits_for_the_final_geometry_is_keyed_to_its_words_and_shown_first():
    same = {**PLAN, "texture": {"texture_prompt": "grey stone"}}
    # The geometry's texture card is skipped: no paint is spent on a body regenerated later.
    assert drive.plan_answer("geometry_texture", {"surface_has_colour": False}, same)[0] == {"action": "skip_texture"}
    # At the material card (the regions approved), a bare body is painted and shown first.
    bare = {"surface_has_colour": False}
    answer, stage, stop = drive.plan_answer("material_input", bare, same)
    assert answer == {"action": "texture_surface", "texture_prompt": "grey stone"} and stage is None and stop == "texture"
    painted = {**bare, "textured_surface": "p.glb", "texture_prompt": "grey stone", "texture_seed": 0}
    assert drive.plan_answer("material_input", painted, same)[0]["material_mode"] == "uniform"
    other = {**PLAN, "texture": {"texture_prompt": "red brick"}}           # new words: painted again
    answer, stage, stop = drive.plan_answer("material_input", painted, other)
    assert answer["action"] == "texture_surface" and stop == "texture"
    # A body that arrived coloured (Hunyuan3D) is not repainted; no plan.texture, no paint.
    assert drive.plan_answer("material_input", {"surface_has_colour": True}, same)[0]["material_mode"] == "uniform"
    assert drive.plan_answer("material_input", bare, PLAN)[0]["material_mode"] == "uniform"
    # Editing plan.texture after the material card rewinds to that card, not to the geometry.
    assert dict(drive.SECTION_CARDS)["texture"] == "material_input"


def test_an_inferred_uniform_material_is_shown_before_the_solve_and_keyed_to_its_words(tmp_path):
    plan = {**PLAN, "material": {"action": "infer_material", "material_description": "grey cast iron"}}
    answer, stage, stop = drive.plan_answer("material_input", {"youngs_modulus": None}, plan)
    assert answer["action"] == "infer_material" and stop == "materials"
    inferred = {"youngs_modulus": 1.1e11, "poisson_ratio": 0.26, "density": 7200,
                "material_description": "grey cast iron"}
    assert drive.plan_answer("material_input", inferred, plan)[0]["material_mode"] == "uniform"
    reworded = {**plan, "material": {**plan["material"], "material_description": "mild steel"}}
    assert drive.plan_answer("material_input", inferred, reworded)[2] == "materials"
    with pytest.raises(drive.PlanError, match="VoMP"):
        drive.plan_answer("material_input", {"youngs_modulus": None, "material_mode": "vomp"}, plan)
    payload = drive._stop_payload("materials", tmp_path, tmp_path, inferred, plan)
    assert payload["material"]["youngs_modulus"] == 1.1e11
    assert drive.brief_run({"stop": "solve", "material": {"youngs_modulus": 1}})["material"]


def test_per_part_values_made_from_other_words_are_valued_again():
    card = {"material_parts": [{"id": 0, "youngs_modulus": 2e11}], "material_description": "steel head"}
    plan = {**PLAN, "material": {"mode": "parts", "description": "steel head, wooden handle"}}
    answer, stage, stop = drive.plan_answer("material_input", card, plan)
    assert answer == {"action": "infer_parts", "material_description": "steel head, wooden handle"}
    assert stage == "materials" and stop == "materials"
    same = {**plan, "material": {**plan["material"], "description": "steel head"}}
    assert drive.plan_answer("material_input", card, same)[0] == {"material_mode": "parts"}


def test_a_stop_lists_only_the_pictures_its_render_wrote():
    assert drive.fresh_images({"ok": True, "images": ["a.png"]}) == (["a.png"], None)
    images, failed = drive.fresh_images({"ok": False, "error": "no display"})
    assert images == [] and "no display" in failed



def test_the_nearest_distance_is_the_brute_force_one():
    rng = np.random.default_rng(1)
    points, target = rng.normal(size=(5000, 3)), rng.normal(size=(700, 3))
    brute = np.sqrt(((points[:, None, :] - target[None, :, :]) ** 2).sum(-1)).min(1)
    assert np.allclose(inspect_session._min_distance(points, target, chunk=999), brute, atol=1e-9)


def test_the_gpu_choice_keeps_a_warm_worker_where_it_is():
    def verdict(index, free, idle=True):
        return lib.GPUVerdict(index=index, name="g", total_mib=48000, used_mib=48000 - free,
                              ours_mib=0, foreign_mib=0, util=0, idle=idle)
    verdicts = [verdict(0, 20000), verdict(1, 40000)]
    assert lib.choose_gpu(verdicts).index == 1                    # the most free memory
    assert lib.choose_gpu(verdicts, {0}).index == 0               # the GPU holding the worker
    assert lib.choose_gpu([verdict(0, 20000, idle=False), verdict(1, 40000)], {0}).index == 1


def test_a_batch_slot_has_its_own_root_and_gpus_and_prints_the_skill_s_paths(monkeypatch):
    monkeypatch.delenv(lib.OUTPUT_ROOT_ENV, raising=False)
    monkeypatch.delenv(lib.GPUS_ENV, raising=False)
    assert lib.output_root_default() == lib.DEFAULT_OUTPUT_ROOT and lib.allowed_gpus() is None
    monkeypatch.setenv(lib.OUTPUT_ROOT_ENV, "output/fem_workflow_operator/_slots/b-1")
    monkeypatch.setenv(lib.GPUS_ENV, "1")
    slot = lib.output_root_default()
    assert slot == (lib.REPO_ROOT / "output/fem_workflow_operator/_slots/b-1").resolve() and lib.allowed_gpus() == {1}
    # The session lives in the slot; the agent reads and writes it at the linked path.
    assert lib.relative_paths(str(slot / "abc" / "operator" / "plan.json")) == "output/fem_workflow_operator/abc/operator/plan.json"


def test_our_own_load_does_not_make_a_gpu_busy_but_another_users_does():
    ours = [{"pid": 20, "uuid": "GPU-0", "used_mib": 9000}]
    loading = [[gpu(0, 9100, 0)], [gpu(0, 9100, 90)], [gpu(0, 9100, 40)]]
    assert lib.classify(loading, ours, owners={20: "hunyuan3d"})[0].idle         # our worker loading
    foreign = ours + [{"pid": 30, "uuid": "GPU-0", "used_mib": 300}]
    busy = [[gpu(0, 9400, 0)], [gpu(0, 9400, 90)]]
    verdict = lib.classify(busy, foreign, owners={20: "hunyuan3d", 30: None})[0]
    assert not verdict.idle and "another user" in verdict.reason                # someone else is on it
    nobody = [[gpu(0, 100, 95)]]
    assert not lib.classify(nobody, [], owners={})[0].idle                      # no process of ours there
    # a worker that exits between samples is not counted as foreign (memory per sample)
    samples = [[gpu(0, 9000, 0)], [gpu(0, 200, 0)]]
    assert lib.classify(samples, [], owners={20: "tools"}, sample_apps=[ours, []])[0].idle


def test_a_fresh_idle_verdict_lets_the_next_gpu_step_sample_once(tmp_path):
    assert not lib.recent_idle(tmp_path)
    idle = lib.GPUVerdict(0, "g", 48000, 100, 0, 0, 0, idle=True)
    lib.note_verdicts(tmp_path, [idle])
    assert lib.recent_idle(tmp_path) and lib.recent_idle(tmp_path, 0) and not lib.recent_idle(tmp_path, 1)
    lib.note_verdicts(tmp_path, [lib.GPUVerdict(0, "g", 48000, 100, 0, 0, 0, foreign_pids=[7], idle=True)])
    assert not lib.recent_idle(tmp_path)                                        # someone else was there
    lib.note_verdicts(tmp_path, [idle])
    record = lib.read_json(tmp_path / ".runtime_cache" / "gpu_verdict.json")
    lib.write_json(tmp_path / ".runtime_cache" / "gpu_verdict.json",
                   {**record, "time": record["time"] - lib.VERDICT_FRESH_S - 1})
    assert not lib.recent_idle(tmp_path)                                        # stale


def test_planned_resolution_is_computed_once_per_body_and_plan(tmp_path, monkeypatch, gaps_not_read, walls_not_read):
    import time as _time
    import trimesh
    (tmp_path / "geometry").mkdir()
    (tmp_path / "operator").mkdir()
    trimesh.creation.box(extents=[1.0, 2.0, 0.5]).export(tmp_path / "geometry" / "generated.glb")
    calls = []
    real = drive._planned_resolution
    monkeypatch.setattr(drive, "_planned_resolution", lambda s, p: calls.append(1) or real(s, p))
    first = drive.planned_resolution(tmp_path, PLAN)
    again = drive.planned_resolution(tmp_path, PLAN)
    assert first == again and len(calls) == 1
    again["planned_mass"] = -1                                                   # callers get a copy
    assert drive.planned_resolution(tmp_path, PLAN) == first
    drive.planned_resolution(tmp_path, {**PLAN, "settings": {"remesh_octree_depth": 8}})
    assert len(calls) == 2                                                       # another depth
    _time.sleep(0.01)
    trimesh.creation.box(extents=[1.0, 2.0, 0.6]).export(tmp_path / "geometry" / "generated.glb")
    drive.planned_resolution(tmp_path, PLAN)
    assert len(calls) == 3                                                       # another body


def test_the_paint_worker_is_preloaded_only_where_the_runner_paints(tmp_path):
    path = tmp_path / "plan.json"
    args = argparse.Namespace(plan=str(path))
    path.write_text(json.dumps(PLAN))
    assert drive.card_will_paint({"kind": "geometry_texture", "surface_has_colour": False}, args) is False
    assert drive.card_will_paint({"kind": "material_input", "surface_has_colour": False}, args) is False
    path.write_text(json.dumps({**PLAN, "texture": {"texture_prompt": "oak"}}))
    assert drive.card_will_paint({"kind": "material_input", "surface_has_colour": False}, args) is True
    assert drive.card_will_paint({"kind": "material_input", "surface_has_colour": True}, args) is False
    assert drive.card_will_paint({"kind": "bc_input"}, args) is None


def test_a_starting_process_s_unlisted_memory_does_not_make_the_gpu_busy():
    # The spare's CUDA context shows in the total a moment before nvidia-smi lists it, while the
    # last step's tail still runs: one sample of unlisted memory beside utilisation is not
    # someone else's job ...
    ours = [{"pid": 20, "uuid": "GPU-0", "used_mib": 300}]
    listed = ours + [{"pid": 21, "uuid": "GPU-0", "used_mib": 340}]
    samples = [[gpu(0, 700, 60)], [gpu(0, 650, 0)], [gpu(0, 650, 0)]]
    owners = {20: "tools", 21: "solver"}
    assert lib.classify(samples, listed, owners, sample_apps=[ours, listed, listed])[0].idle
    # ... but unlisted memory in every sample is, and so is one sample alone
    assert not lib.classify(samples, ours, owners, sample_apps=[ours, ours, ours])[0].idle
    assert not lib.classify(samples[:1], ours, owners, sample_apps=[ours])[0].idle
    # A stopped worker's memory stays in the total a moment after it leaves the list: the paint
    # worker's 17 GB in one sample of three is not another user's memory; in every sample it is.
    tool = [{"pid": 20, "uuid": "GPU-0", "used_mib": 426}]
    exiting = [[gpu(0, 17363, 0)], [gpu(0, 449, 0)], [gpu(0, 449, 0)]]
    assert lib.classify(exiting, tool, owners, sample_apps=[tool, tool, tool])[0].idle
    stays = [[gpu(0, 17363, 0)]] * 3
    verdict = lib.classify(stays, tool, owners, sample_apps=[tool, tool, tool])[0]
    assert not verdict.idle and "foreign memory" in verdict.reason


def test_a_worker_on_its_way_out_is_ours_while_it_lasts(monkeypatch, tmp_path):
    import subprocess
    import time as _time
    # A stopped worker is a zombie a moment, its command line empty, its pid file gone.
    child = subprocess.Popen(["true"])
    deadline = _time.time() + 5
    while Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
        assert _time.time() < deadline
        _time.sleep(0.01)
    try:
        assert lib.workflow_process(child.pid) == "exiting"
    finally:
        child.wait()
    # Owners are taken sample by sample: the exiting worker, labelled in the first sample, is not
    # forgotten when it has gone by the last.
    present = [{"pid": 77, "uuid": "GPU-0", "used_mib": 16900}]
    queries = iter([present, [], []])
    monkeypatch.setattr(lib, "query_gpus", lambda: [gpu(0, 17000)])
    monkeypatch.setattr(lib, "query_apps", lambda: next(queries))
    monkeypatch.setattr(lib, "allowed_gpus", lambda: None)
    monkeypatch.setattr(lib, "app_owners", lambda apps, root: {a["pid"]: "exiting" for a in apps})
    verdict = lib.gpu_table(tmp_path, count=3, interval=0.0)[0]
    assert "exiting(pid 77" in verdict.our_workers[0]


def test_a_one_sample_busy_check_is_repeated_at_once_with_three(monkeypatch, tmp_path):
    counts = []

    def table(root, count=3, **k):
        counts.append(count)
        return _verdicts(False, count == 3)
    monkeypatch.setattr(lib, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(lib, "recent_idle", lambda *a, **k: True)
    monkeypatch.setattr(lib, "gpu_table", table)
    monkeypatch.setattr(lib, "worker_files", lambda root: [])
    # no real GPU state: nvidia-smi's apps and GPUs, the warm workers and the verdict record are mocked
    monkeypatch.setattr(lib, "query_apps", lambda: [])
    monkeypatch.setattr(lib, "query_gpus", lambda: [])
    monkeypatch.setattr(lib, "note_verdicts", lambda *a, **k: None)
    monkeypatch.setattr(drive, "warm_gpus", lambda *a, **k: set())
    monkeypatch.setattr(drive.time, "sleep", lambda s: pytest.fail("waited for a GPU it had not re-checked"))
    chosen = drive.select_gpu(_guard_args(gpu_wait=60.0), tmp_path, True, ["solver"], "env")
    assert chosen["index"] == 1 and counts == [1, 3]


def test_every_root_shares_the_user_s_warp_kernel_cache(monkeypatch, tmp_path):
    from src.agents_wrapper.fem_workflow import workers
    monkeypatch.delenv("WARP_CACHE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert workers.kernel_cache_env() == {"WARP_CACHE_PATH": str(tmp_path / "xdg" / "warp")}
    monkeypatch.setenv("WARP_CACHE_PATH", "/elsewhere")
    assert workers.kernel_cache_env() == {"WARP_CACHE_PATH": "/elsewhere"}
    # a worker gets it beside its own root's XDG cache
    seen = {}

    class Popen:
        pid = 4242

        def __init__(self, command, env, **kw):
            seen.update(env)
    monkeypatch.setattr(workers.subprocess, "Popen", Popen)
    assert workers._launch(tmp_path, "tools", "env", ["true"], cwd=tmp_path, cache=tmp_path / "cache")
    assert seen["WARP_CACHE_PATH"] == "/elsewhere" and seen["XDG_CACHE_HOME"] == str(tmp_path / "cache")


def test_the_slow_loads_start_one_stop_early(tmp_path, monkeypatch):
    import trimesh
    from src.agents_wrapper.fem_workflow import run_workflow, workers
    session = tmp_path / "s"
    (session / "geometry").mkdir(parents=True)
    bare = session / "geometry" / "physical.glb"
    trimesh.creation.box().export(bare)
    lib.write_json(session / "session.json", {"surface_path": str(bare)})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({**PLAN, "texture": {"texture_prompt": "oak"}}))
    args = argparse.Namespace(plan=str(plan_path), no_preload=False, max_foreign_mib=1024.0, max_util=10.0)
    assert drive.paint_ahead(session, {"kind": "bc_input"}, args)
    assert drive.paint_ahead(session, {"kind": "bc_values"}, args)
    assert not drive.paint_ahead(session, {"kind": "geometry_physics"}, args)
    calls = []
    monkeypatch.setattr(lib, "recent_idle", lambda *a, **k: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(run_workflow, "preload_for",
                        lambda *a, **k: calls.append(("preload_for", a[3]["kind"], k.get("keep_paint"))))
    monkeypatch.setattr(workers, "ensure_solver_worker", lambda *a, **k: calls.append(("spare",)))
    monkeypatch.setattr(workers, "ensure_tool_worker", lambda *a, **k: None)
    monkeypatch.setattr(workers, "ensure_hunyuan3d_worker", lambda *a, **k: calls.append(
        ("paint", k.get("paint"), k.get("keep_triposg"), k.get("keep_partfield"))))
    monkeypatch.setattr(workers, "ensure_partfield_worker", lambda *a, **k: calls.append(("partfield", k.get("keep"))))
    monkeypatch.setattr(workers, "stop_hunyuan3d_worker", lambda *a, **k: calls.append(("stop hunyuan3d",)))
    monkeypatch.setattr(drive, "_partfield_wanted", lambda *a: True)
    monkeypatch.setattr(drive, "_gpu_free_mib", lambda: 40_000)
    worker = [None]
    monkeypatch.setattr(drive, "_hunyuan3d_worker", lambda root: worker[0])
    # The paint worker and PartField side by side, no preload_for (it would stop one for the other)
    # and no spare beside them.
    drive.preload(tmp_path, session, "env", {"kind": "bc_input"}, args)
    assert calls == [("paint", True, False, True), ("partfield", ("hunyuan3d",))]
    calls.clear()
    # From the geometry stop on, TripoSG kept for a regeneration
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == [("paint", True, True, True), ("partfield", ("hunyuan3d", "triposg"))]
    calls.clear()
    # A Hunyuan3D body's shape worker stays at its stop, and gives way once the body is accepted
    lib.write_json(session / "session.json", {"surface_path": str(bare), "generation": {"generator": "hunyuan3d"}})
    worker[0] = "shape"
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == [("partfield", ("hunyuan3d", "triposg"))]
    calls.clear()
    drive.preload(tmp_path, session, "env", {"kind": "geometry_physics"}, args)
    assert calls == [("stop hunyuan3d",), ("paint", True, False, True), ("partfield", ("hunyuan3d",))]
    calls.clear()
    # A CAD body's worker stays at its stop
    lib.write_json(session / "session.json", {"surface_path": str(bare), "generation": {"generator": "cadcoder"}})
    worker[0] = None
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == []
    lib.write_json(session / "session.json", {"surface_path": str(bare)})
    # No room: the paint loads when it runs; a worker already up is kept as it is
    monkeypatch.setattr(drive, "_gpu_free_mib", lambda: 15_000)
    drive.preload(tmp_path, session, "env", {"kind": "bc_values"}, args)
    assert calls == []
    worker[0] = "paint"
    drive.preload(tmp_path, session, "env", {"kind": "bc_input"}, args)
    assert calls == [("partfield", ("hunyuan3d",))]
    calls.clear()
    plan_path.write_text(json.dumps(PLAN))                        # no paint: the spare instead
    drive.preload(tmp_path, session, "env", {"kind": "bc_input"}, args)
    assert calls == [("preload_for", "bc_input", False), ("spare",)]
    calls.clear()
    drive.preload(tmp_path, session, "env", {"kind": "material_input", "solve_complete": True}, args)
    assert calls == [("preload_for", "material_input", False)]   # nothing follows a finished solve


def test_the_hunyuan3d_worker_is_warmed_for_the_plan_and_kept_for_a_regeneration(tmp_path, monkeypatch):
    from src.agents_wrapper.fem_workflow import run_workflow
    session = tmp_path / "s"
    session.mkdir()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({**PLAN, "geometry": {"description": "an object", "draw": True,
                                                         "generator": "hunyuan3d"}}))
    args = argparse.Namespace(plan=str(plan_path), no_preload=False, max_foreign_mib=1024.0, max_util=10.0)
    calls = []
    monkeypatch.setattr(lib, "recent_idle", lambda *a, **k: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(run_workflow, "preload_for",
                        lambda *a, **k: calls.append((a[3].get("generator"), k.get("paint"))))
    # after a drawing the card still names TripoSG: the plan's generator is warmed
    drive.preload(tmp_path, session, "env", {"kind": "geometry_input", "generator": "triposg"}, args)
    # a Hunyuan3D body at the geometry stop keeps its worker (paint None, not False)
    lib.write_json(session / "session.json", {"generation": {"generator": "hunyuan3d"}})
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    lib.write_json(session / "session.json", {"generation": {"generator": "triposg"}})
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == [("hunyuan3d", None), (None, None), (None, False)]


def test_hunyuan3d_generates_the_shape_alone_unless_the_plan_asks_for_colour():
    answer = {"description": "an object", "generator": "hunyuan3d", "use_drawn_image": True}
    shaped = drive.hunyuan_shape_only(answer, {"geometry": {"generator": "hunyuan3d"}})
    assert shaped["generation_settings"] == {"texture": False}
    seeded = drive.hunyuan_shape_only({**answer, "generation_settings": {"seed": 3}}, {"geometry": {}})
    assert seeded["generation_settings"] == {"seed": 3, "texture": False}
    asked = {"geometry": {"generation_settings": {"texture": True}}}
    assert drive.hunyuan_shape_only(answer, asked) is answer                # the plan's own choice
    drawing = {"action": "draw_image", "description": "an object"}
    assert drive.hunyuan_shape_only(drawing, {"geometry": {"generator": "hunyuan3d"}}) is drawing
    assert drive.hunyuan_shape_only({"description": "an object"}, {"geometry": {}}) == {"description": "an object"}


def test_hunyuan3d_is_warmed_beside_triposg_when_the_routes_go_there_next(tmp_path, monkeypatch):
    from src.agents_wrapper.fem_workflow import run_workflow, workers
    session = tmp_path / "s"
    (session / "operator").mkdir(parents=True)
    lib.write_json(session / "session.json", {"generation": {"generator": "triposg"}})
    plan_path = tmp_path / "plan.json"
    plan = {**PLAN, "geometry": {"description": "an object", "draw": True},
            "geometry_routes": ["drawing", "hunyuan3d"]}
    plan_path.write_text(json.dumps(plan))
    args = argparse.Namespace(plan=str(plan_path), no_preload=False, max_foreign_mib=1024.0, max_util=10.0)
    calls = []
    monkeypatch.setattr(lib, "recent_idle", lambda *a, **k: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    monkeypatch.setattr(drive, "_gpu_free_mib", lambda: 40_000)
    monkeypatch.setattr(run_workflow, "preload_for", lambda *a, **k: calls.append(("preload_for", a[3]["kind"])))
    monkeypatch.setattr(workers, "ensure_tool_worker", lambda *a, **k: calls.append(("tools", a[2])))
    monkeypatch.setattr(workers, "ensure_hunyuan3d_worker", lambda *a, **k: calls.append(("hunyuan3d", k)))
    note = drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == [("tools", "cuda"), ("hunyuan3d", {"shape_only": True, "keep_triposg": True})]
    assert "Hunyuan3D" in note
    calls.clear()
    monkeypatch.setattr(drive, "_gpu_free_mib", lambda: 9_000)              # no room beside TripoSG
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == [("tools", "cuda")]
    calls.clear()
    plan_path.write_text(json.dumps({**plan, "geometry_routes": ["drawing"]}))   # not a next route
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == [("preload_for", "geometry_texture")]
    calls.clear()
    plan_path.write_text(json.dumps(plan))
    lib.write_json(session / "session.json", {"generation": {"generator": "hunyuan3d"}})   # already it
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == [("preload_for", "geometry_texture")]
    calls.clear()
    # past the stop (the TripoSG body accepted) the cards preload as before, PartField included
    lib.write_json(session / "session.json", {"generation": {"generator": "triposg"}})
    drive.preload(tmp_path, session, "env", {"kind": "geometry_physics"}, args)
    assert calls == [("preload_for", "geometry_physics")]
    calls.clear()
    # a plan that generates coloured bodies is not warmed shape-only
    coloured = {**plan, "geometry": {**plan["geometry"], "generation_settings": {"texture": True}}}
    plan_path.write_text(json.dumps(coloured))
    drive.preload(tmp_path, session, "env", {"kind": "geometry_texture"}, args)
    assert calls == [("preload_for", "geometry_texture")]
    assert not drive.plan_shape_only(coloured) and drive.plan_shape_only(plan)


def test_a_drawing_card_keeps_the_hunyuan3d_worker_warmed_for_the_next_route(tmp_path, monkeypatch):
    # A trial run (2026-10-09): a redrawn drawing's card stopped the worker warmed at the geometry
    # stop, and the Hunyuan3D generation that followed loaded it cold.
    from src.agents_wrapper.fem_workflow import run_workflow, workers
    session = tmp_path / "s"
    (session / "operator").mkdir(parents=True)
    lib.write_json(session / "session.json", {"generation": {"generator": "triposg"}})
    plan_path = tmp_path / "plan.json"
    plan = {**PLAN, "geometry": {"description": "a stand", "draw": True},
            "geometry_routes": ["drawing", "hunyuan3d"]}
    plan_path.write_text(json.dumps(plan))
    args = argparse.Namespace(plan=str(plan_path), no_preload=False, max_foreign_mib=1024.0, max_util=10.0)
    calls = []
    monkeypatch.setattr(lib, "recent_idle", lambda *a, **k: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(drive, "_gpu_free_mib", lambda: 30_000)
    worker = ["shape"]
    monkeypatch.setattr(drive, "_hunyuan3d_worker", lambda root: worker[0])
    monkeypatch.setattr(run_workflow, "preload_for", lambda *a, **k: calls.append(("preload_for", a[3]["kind"])))
    monkeypatch.setattr(workers, "ensure_tool_worker", lambda *a, **k: calls.append(("tools", a[2])))
    monkeypatch.setattr(workers, "ensure_geometry_worker", lambda *a, **k: calls.append(("triposg", k)))
    card = {"kind": "geometry_input", "generator": "triposg"}
    note = drive.preload(tmp_path, session, "env", card, args)
    assert calls == [("triposg", {"keep_hunyuan3d": True}), ("tools", "light")]
    assert "kept" in note
    for change in ({"_gpu_free_mib": lambda: 9_000},              # no room for a drawing beside it
                   {"_hunyuan3d_worker": lambda root: None}):     # none was warmed
        calls.clear()
        with monkeypatch.context() as m:
            for name, value in change.items():
                m.setattr(drive, name, value)
            drive.preload(tmp_path, session, "env", card, args)
        assert calls == [("preload_for", "geometry_input")]
    calls.clear()
    plan_path.write_text(json.dumps({**plan, "geometry_routes": ["drawing", "generation"]}))
    drive.preload(tmp_path, session, "env", card, args)
    assert calls == [("preload_for", "geometry_input")]


def test_the_monitor_keeps_a_warm_worker_beside_a_run_that_fits(tmp_path, monkeypatch):
    from src.agents_wrapper.fem_workflow import monitor
    stopped = []
    monkeypatch.setattr(monitor.ToolMonitor, "_shutdown_hunyuan3d_worker", lambda self: stopped.append("hunyuan3d"))
    monkeypatch.setattr(monitor.ToolMonitor, "_shutdown_partfield_worker", lambda self: stopped.append("partfield"))
    for name in ("_shutdown_geometry_worker", "_shutdown_vomp_worker"):
        monkeypatch.setattr(monitor.ToolMonitor, name, lambda self: None)
    monkeypatch.setattr(monitor.ToolMonitor, "_shutdown_cad_workers", lambda self, keep=None: None)
    monkeypatch.setattr(monitor.ToolMonitor, "_hunyuan3d_paint_loaded", lambda self: True)
    monkeypatch.setattr(monitor.ToolMonitor, "_worker_up", lambda self, kind, environment: True)

    class Ran(Exception):
        pass

    def run_worker(self, *a, **k):
        raise Ran()
    monkeypatch.setattr(monitor.ToolMonitor, "_run_worker", run_worker)
    monkeypatch.setattr(monitor, "partfield_status", lambda: types.SimpleNamespace(
        ready=True, interpreter="python", environment="pf", summary=""))
    session = tmp_path / "s"
    session.mkdir()
    surface = session / "a.glb"
    surface.write_bytes(b"x")
    args = {"method": "partfield", "surface": str(surface), "output_npz": str(session / "o.npz"),
            "output_json": str(session / "o.json"), "output_glb": str(session / "o.glb"), "clusters": 5,
            "feature_cache": str(session / "features")}
    for free, expected in ((40_000, []), (4_000, ["hunyuan3d"]), (None, ["hunyuan3d"])):
        stopped.clear()
        monkeypatch.setattr(monitor, "gpu_free_mib", lambda free=free: free)
        with pytest.raises(Ran):
            monitor.ToolMonitor(session, "env").execute("segment_surface", args)
        assert stopped == expected, free


def test_the_material_card_s_paint_runs_ahead_and_every_conflicting_step_waits(tmp_path, monkeypatch):
    import subprocess
    import trimesh
    from src.agents_wrapper.fem_workflow import monitor
    session = tmp_path / "s"
    (session / "geometry").mkdir(parents=True)
    (session / "operator").mkdir()
    bare = session / "geometry" / "physical.glb"
    trimesh.creation.box().export(bare)
    lib.write_json(session / "session.json", {"surface_path": str(bare), "session_dir": str(session),
                                              "segmentation_method": "partfield"})
    plan = {**PLAN, "texture": {"texture_prompt": "oak"}}
    assert drive.paint_ahead_answer(session, plan) == {"action": "texture_surface", "texture_prompt": "oak"}
    assert drive.paint_ahead_answer(session, PLAN) is None                  # a uniform plan paints nothing
    parts = {**PLAN, "material": {"mode": "parts", "description": "oak and steel"}}
    assert drive.paint_ahead_answer(session, parts) is None                 # coloured flat by part
    parts["material"]["texture_prompt"] = "brown oak, grey steel"
    assert drive.paint_ahead_answer(session, parts)["action"] == "texture_surface"
    launched = []

    class Popen:
        pid = 4242

        def __init__(self, command, **kw):
            launched.append(command)
    monkeypatch.setattr(subprocess, "Popen", Popen)
    monkeypatch.setattr(monitor, "gpu_free_mib", lambda: 40_000)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    # not before the geometry is segmented
    assert drive.start_paint_ahead(session, tmp_path, "env", {"kind": "bc_input"}, plan) is None
    card = {"kind": "bc_input", "has_segmentation": True}
    assert "background" in drive.start_paint_ahead(session, tmp_path, "env", card, plan)
    record = lib.read_json(session / "operator" / drive.PAINT_AHEAD)
    assert record["state"] == "running" and record["pid"] == 4242 and record["gpu_index"] == 1
    assert launched[0][2].endswith("paint_ahead.py") and json.loads(launched[0][5]) == record["answer"]
    monkeypatch.setattr(lib, "_alive", lambda pid: True)
    assert drive.paint_ahead_running(session)
    assert drive.start_paint_ahead(session, tmp_path, "env", card, plan) is None     # once
    assert len(launched) == 1
    # what may run beside it, and what waits
    assert not drive.paint_ahead_conflicts(session, "bc_input", {"action": "preview_regions"})
    assert not drive.paint_ahead_conflicts(session, "bc_input", {"action": "preview_regions",
                                                                 "segmentation": {"method": "partfield"}})
    assert drive.paint_ahead_conflicts(session, "bc_input", {"action": "preview_regions",
                                                             "segmentation": {"num_clusters": 9}})
    assert not drive.paint_ahead_conflicts(session, "bc_values", {"action": "apply_values"})
    for kind, answer in (("bc_input", {"action": "preview_segmentation"}), ("material_input", {}),
                         ("geometry_input", {"description": "x"}), ("shape_opt_settings", {})):
        assert drive.paint_ahead_conflicts(session, kind, answer)
    # still running at the timeout: stopped (its group and the worker) and recorded failed, said once
    from src.agents_wrapper.fem_workflow import workers
    stopped = []
    monkeypatch.setattr(drive.os, "killpg", lambda pid, sig: stopped.append(("group", pid)))
    monkeypatch.setattr(workers, "stop_hunyuan3d_worker", lambda root: stopped.append(("worker", root)))
    drive.join_paint_ahead(session, timeout=0.0)
    assert stopped == [("group", 4242), ("worker", tmp_path)]
    assert lib.read_json(session / "operator" / drive.PAINT_AHEAD)["state"] == "failed"
    assert "background paint failed" in drive.paint_ahead_warning(session)
    assert drive.paint_ahead_warning(session) is None
    assert drive.start_paint_ahead(session, tmp_path, "env", card, plan) is None     # not again
    # a child gone without recording its end is failed, not waited for or started again
    lib.write_json(session / "operator" / drive.PAINT_AHEAD, {**record, "state": "running"})
    monkeypatch.setattr(lib, "_alive", lambda pid: False)
    assert not drive.paint_ahead_running(session)
    assert lib.read_json(session / "operator" / drive.PAINT_AHEAD)["state"] == "failed"
    assert drive.start_paint_ahead(session, tmp_path, "env", card, plan) is None and len(launched) == 1
    # the join returns once the helper has recorded its end
    lib.write_json(session / "operator" / drive.PAINT_AHEAD, {**record, "state": "done"})
    assert drive.join_paint_ahead(session, timeout=5.0) == 0.0 and not drive.paint_ahead_running(session)
    # too little free memory: the card paints for itself
    (session / "operator" / drive.PAINT_AHEAD).unlink()
    monkeypatch.setattr(monitor, "gpu_free_mib", lambda: 8_000)
    assert "not started ahead" in drive.start_paint_ahead(session, tmp_path, "env", card, plan)
    assert len(launched) == 1


def test_a_backend_solve_joins_the_paint_and_frees_the_models_first(tmp_path, monkeypatch, capsys):
    """solve-backend runs outside the graph's card, so it does what the monitor does before the
    card's solve: once a GPU is chosen, the background paint ends, then the model workers are
    stopped and waited out, before the solve runs.  With no GPU free nothing is stopped."""
    import subprocess
    from src.agents_wrapper.fem_workflow.monitor import ToolMonitor
    session = tmp_path / "s"
    (session / "case").mkdir(parents=True)
    (session / "operator").mkdir()
    lib.write_json(session / "case" / "case.json", {"output": str(session / "solve")})
    calls = []
    monkeypatch.setattr(drive, "_load", lambda args: (tmp_path, session, "env"))
    monkeypatch.setattr(drive, "charge_attempt", lambda *a: {"counted": True, "over": False, "counts": {}})
    monkeypatch.setattr(inspect_session, "compute_interpreter", lambda env: "python")
    monkeypatch.setattr(drive, "join_paint_ahead", lambda s: calls.append("join") or 0.0)
    monkeypatch.setattr(ToolMonitor, "release_models_for_solve", lambda self: calls.append("release"))

    def no_gpu(*a, **k):
        calls.append("select")
        raise drive.NoIdleGPU("busy", [])

    monkeypatch.setattr(drive, "select_gpu", no_gpu)
    args = argparse.Namespace(session_id="s", backend="native", max_iters=None, two_depths=False)
    drive.cmd_solve_backend(args)
    assert calls == ["select"]
    calls.clear()
    monkeypatch.setattr(drive, "select_gpu", lambda *a, **k: calls.append("select") or {"index": None})

    def solve(*a, **k):
        calls.append("solve")
        raise RuntimeError("stop here")

    monkeypatch.setattr(subprocess, "run", solve)
    with pytest.raises(RuntimeError, match="stop here"):
        drive.cmd_solve_backend(args)
    assert calls == ["select", "join", "release", "solve"]
    capsys.readouterr()


def test_commands_hand_themselves_to_the_environment_without_conda(monkeypatch, tmp_path):
    rf = lib.runtime_files
    # Found beside the running interpreter even when the shell's conda variables are gone.
    for name in ("CONDA_EXE", "CONDA_ROOT", "CONDA_PREFIX_1"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "envs" / "mechfoundry-x" / "bin"
    env.mkdir(parents=True)
    (env / "python").write_text("")
    monkeypatch.setattr(rf.sys, "executable", str(tmp_path / "envs" / "other" / "bin" / "python"))
    assert rf.direct_interpreter("mechfoundry-x") == env / "python"
    assert rf.direct_interpreter("../mechfoundry-x") is None and rf.direct_interpreter("nope") is None
    # Already in it: nothing happens; elsewhere: this process becomes the env's interpreter.
    calls = []
    monkeypatch.setattr(rf.os, "execvp", lambda path, argv: calls.append(argv))
    monkeypatch.setattr(rf.sys, "prefix", str(tmp_path / "envs" / "mechfoundry-x"))
    rf.run_in_environment("mechfoundry-x", "drive.py", ["start"])
    assert calls == []
    monkeypatch.setattr(rf.sys, "prefix", "/usr")
    monkeypatch.delenv(rf.HANDED_OVER, raising=False)
    rf.run_in_environment("mechfoundry-x", "drive.py", ["start"])
    assert calls == [[str(env / "python"), "-u", "drive.py", "start"]]
    rf.run_in_environment("mechfoundry-x", "drive.py", ["start"])          # handed over once only
    assert len(calls) == 1


def test_region_pieces_and_solids_by_graph_components():
    import numpy as np
    import trimesh
    two = trimesh.util.concatenate([trimesh.creation.box(), trimesh.creation.box().apply_translation([3, 0, 0])])
    assert drive.connected_solids(two) == 2 and drive.connected_solids(trimesh.creation.box()) == 1
    mask = np.zeros(len(two.faces), dtype=bool)
    mask[:] = True
    pieces = drive._region_pieces(two, mask)
    assert [p["faces"] for p in pieces] == [12, 12]
    assert pieces[0]["bounds"][0][0] < pieces[1]["bounds"][0][0]
    assert pieces[0]["members"] == sorted(pieces[0]["members"])
    assert drive._region_pieces(two, np.zeros(len(two.faces), dtype=bool)) == []


def test_render_takes_several_pictures_in_one_process():
    assert inspect_session._what("regions,segments") == "regions,segments"
    with pytest.raises(Exception):
        inspect_session._what("regions,photos")


# ------------------------------------------------- the compact view, read, judge --finish --

def _regions_stop(pieces: int = 33) -> dict:
    """A regions stop as brief_run hands it to the printer: one support region in many pieces
    (two under a leaf, one beside a load), the brief's prose, every attempt counter."""
    held = [{"name": f"base#{k + 1}", "region": "base", "contact": "feet", "transmits": "force and moment",
             "pieces_of_region": pieces, "load_beside": 0.8 if k == 5 else 0.0, "load_reach_m": 0.04,
             "member_axis": "x", "face": "side face", "held": [0, 1, 2],
             "span_m": [0.01 * k, 0.01 * k + 0.0081234567], "length_m": 0.0081234567,
             "length_leaves7": 0.4 if k in (3, 7) else 2.3456789, "simplified_to": None}
            for k in range(pieces)]
    return {"ok": True, "session_id": "abc", "stop": "regions", "kind": "bc_input", "elapsed_s": 3.21,
            "attempts_left": {k: v for k, v in drive.RUN_LIMITS.items()},
            "why": drive.STOPS["regions"], "look_at": ["output/fem_workflow_operator/abc/operator/regions.png"],
            "next": ("run again to continue (that accepts what this stop shows), or edit the plan first and "
                     "run again: the runner rewinds to the card of the earliest changed section"),
            "brief": {"present": True, "prediction": "- Mode: bending of the top. " * 60,
                      "prediction_generated": "- Mode: bending of the generated top. " * 60, "changes": None,
                      "changed_since_first": [], "prediction_generated_changed": False,
                      "real_lines_edited": [], "digest": "0123456789ab"},
            "held_regions": held, "warnings": [],
            "synthesized_selectors": {"base": {"type": "and", "params": {}, "children": [
                {"type": "lowest", "params": {"axis": "y", "fraction": 0.0512345678}, "children": []}]}},
            "steps": [{"kind": "bc_input", "action": "preview_regions", "ok": True, "elapsed_s": 3.0,
                       "notes": [], "warnings": []}]}


def _printed(capsys, value, session):
    drive.emit(drive.compact_stop(value, session), session)
    return capsys.readouterr().out


def test_the_compact_view_is_one_json_object_a_line_per_key_and_small(tmp_path, capsys):
    session = tmp_path / "abc"
    value = _regions_stop()
    full = len(json.dumps(value, indent=1))
    out = _printed(capsys, value, session)
    shown = json.loads(out)                                   # still one JSON object
    assert len(out.splitlines()) == len(shown) + 2            # one line per key
    assert len(out) < 3000 and len(out) < full / 5, (len(out), full)
    # The brief as its digest and flags, never the prose the agent wrote.
    assert shown["brief"] == {"digest": "0123456789ab"}
    # Only the stages this stop's next run can spend.
    assert set(shown["attempts_left"]) == {"regions", "revisions", "segmentation", "materials", "solve"}
    # Numbers to four figures, a selector exactly and without its empty parts.
    assert shown["elapsed_s"] == 3.21
    assert shown["synthesized_selectors"]["base"] == {"type": "and", "children": [
        {"type": "lowest", "params": {"axis": "y", "fraction": 0.0512345678}}]}
    assert shown["steps"] == [{"kind": "bc_input", "action": "preview_regions", "ok": True, "elapsed_s": 3.0}]
    assert shown["warnings"] == [] and "card" not in shown
    # The saved stop keeps every number.
    assert value["brief"]["prediction"].startswith("- Mode")


def test_held_pieces_collapse_to_a_row_per_region_keeping_short_and_loaded_ones():
    rows = drive.collapse_held(_regions_stop(33)["held_regions"])
    assert len(rows) == 1
    row = rows[0]
    assert row["pieces"] == 33 and row["faces"] == {"side face": 33} and row["held"] == [0, 1, 2]
    assert row["contact"] == "feet" and row["transmits"] == "force and moment"
    assert row["length_leaves7_min_max"] == [0.4, 2.3456789] and row["load_beside_max"] == 0.8
    assert [p["name"] for p in row["under_a_leaf"]] == ["base#4", "base#8"]
    assert [p["name"] for p in row["beside_a_load"]] == ["base#6"]
    assert row["span_m"][0] == 0.0 and abs(row["span_m"][1] - (0.32 + 0.0081234567)) < 1e-9
    # A region of few pieces stays piece by piece, without its empty fields.
    few = drive.collapse_held(_regions_stop(2)["held_regions"])
    assert [r["name"] for r in few] == ["base#1", "base#2"] and "simplified_to" not in few[0]
    # Two regions are collapsed apart; a different hold within one region is listed.
    mixed = _regions_stop(5)["held_regions"]
    mixed[0]["held"] = [1]
    other = [{**r, "region": "top", "name": "top"} for r in _regions_stop(1)["held_regions"]]
    rows = drive.collapse_held(mixed + other)
    assert [r.get("region") for r in rows] == ["base", "top"] and rows[0]["held"] == [[0, 1, 2], [1]]


def test_boilerplate_and_read_sections_are_printed_once_per_session(tmp_path, capsys):
    session = tmp_path / "abc"
    first = _regions_stop()
    first["read"] = drive.read_for("regions", [])
    one = json.loads(_printed(capsys, first, session))
    assert one["next"].startswith("run again") and one["why"] == drive.STOPS["regions"]
    # each entry one string, "file:a-b why"
    assert {e.split(" ")[0].split(":")[0] for e in one["read"]} >= {"references/visual_checks.md",
                                                                   "references/objectives/judging.md"}
    entry = drive.reference_section("visual_checks.md", "## 2. ", "why")
    assert f"references/visual_checks.md:{entry['lines'][0]}-{entry['lines'][1]} the region checks and the decision" \
        in one["read"]
    record = lib.read_json(session / "operator" / "shown.json")
    assert ["references/visual_checks.md", "2. `operator/segments.png` and `operator/regions.png` (after the preview)"] \
        in record["read"]
    second = _regions_stop()
    second["read"] = drive.read_for("regions", []) + drive.read_for("solve", [])
    two = json.loads(_printed(capsys, second, session))
    assert two["next"] == "as before" and two["why"] == "as before"
    assert two["read"] and not set(two["read"]) & set(one["read"])
    assert any(e.startswith("references/visual_checks.md:") and "field checks" in e for e in two["read"])
    third = _regions_stop()
    third["read"] = drive.read_for("solve", [])
    # nothing new to read: at most the regions stop's own sections, named again in one line
    assert all(e.startswith(drive.READ_AGAIN) for e in json.loads(_printed(capsys, third, session)).get("read", []))
    # Another session starts afresh.
    assert json.loads(_printed(capsys, _regions_stop(), tmp_path / "other"))["next"].startswith("run again")


def test_a_stops_own_sections_are_named_again_until_the_session_runs_on_from_it(tmp_path, capsys):
    session = tmp_path / "abc"

    def stop(kind):
        value = {**_regions_stop(), "stop": kind}
        value["read"] = drive.read_for(kind, [])
        return json.loads(_printed(capsys, value, session))
    first = stop("geometry")
    regions_md = next(e for e in first["read"] if e.startswith("references/objectives/regions.md"))
    assert not any(e.startswith(drive.READ_AGAIN) for e in first["read"])
    # a regeneration: the next geometry stop names them again, in one line, never "as before"
    for kind in ("geometry", "drawing", "geometry"):
        again = stop(kind)
        if kind == "geometry":
            line = [e for e in again["read"] if e.startswith(drive.READ_AGAIN)]
            assert len(line) == 1 and regions_md.split(" ")[0] in line[0], again["read"]
            assert again["next"] == "as before" and "as before" not in again["read"]
            assert " the " not in line[0].split(": ", 1)[1]           # file:a-b only, no reasons
    # once per run of stops of a kind: a third geometry stop in a row adds nothing
    assert "read" not in stop("geometry")
    # the regions stop: the geometry was run on from, so its sections are not named again
    regions = stop("regions")
    assert not any("objectives/regions.md" in e for e in regions["read"])
    assert "read" not in stop("geometry")
    record = lib.read_json(session / "operator" / "shown.json")
    assert "geometry" in record["passed"] and "regions" not in record["passed"]
    # a section first named at start that serves a stop's decision (the drawing's words) is
    # listed again at that stop
    other = tmp_path / "other"
    reads = drive.read_for("start", [])
    lib.write_json(other / "operator" / "shown.json",
                   {"notes": [], "read": [[e["file"], e.get("section")] for e in reads]})
    value = {**_regions_stop(), "stop": "drawing", "read": drive.read_for("drawing", [])}
    drawing = json.loads(_printed(capsys, value, other))
    sdxl = drive.reference_section("geometry.md", "## SDXL drawing", "")
    assert drawing["read"] == [drive.read_line(drive.read_for("drawing", [])[1]),
                               drive.READ_AGAIN + f"{sdxl['file']}:{sdxl['lines'][0]}-{sdxl['lines'][1]}"]


def test_the_drawing_stop_names_the_drawing_checks_not_the_whole_visual_checks():
    entries = [e for e in drive.read_for("drawing", []) if e["file"] == "references/visual_checks.md"]
    assert len(entries) == 1 and "lines" in entries[0]
    lines = (SCRIPTS.parent / "references" / "visual_checks.md").read_text(encoding="utf-8").splitlines()
    a, b = entries[0]["lines"]
    text = "\n".join(lines[a - 1:b])
    assert lines[a - 1].startswith("## 1.") and "**Is it the object?**" in text and "**Not fused.**" in text
    assert "6. **" not in text and "## 2." not in text and b - a < len(lines) // 3
    # the range ends at check 6's line: renumbered or removed, the range runs on and this fails
    assert lines[b].startswith("6. **The right size"), lines[b]
    assert sum(line.startswith("6. **") for line in lines[a - 1:]) >= 1
    # kept apart from the whole §1, which the geometry stop names for its decision
    whole = drive.reference_section("visual_checks.md", "## 1.", "")
    assert entries[0]["section"] != whole["section"] and whole["lines"][1] > b


def test_every_read_entry_names_an_existing_section_and_every_reference_is_named():
    skill = SCRIPTS.parent
    named = set()
    for stop, entries in [*drive.READ.items(), *(("error", e) for _, e in drive.READ_ON_ERROR)]:
        for file, heading, why in entries:
            entry = drive.reference_section(file, heading, why)
            assert entry and (skill / entry["file"]).is_file(), (stop, file)
            if heading is not None:
                assert "lines" in entry, (stop, file, heading)        # the heading still exists
                lines = (skill / entry["file"]).read_text(encoding="utf-8").splitlines()
                a, b = entry["lines"]
                assert lines[a - 1].startswith("#") and a <= b <= len(lines)
            named.add(entry["file"])
    skill_text = (skill / "SKILL.md").read_text(encoding="utf-8")
    index = {"references/objectives.md", "references/cli_protocol.md"}
    for path in (skill / "references").rglob("*.md"):
        rel = path.relative_to(skill).as_posix()
        assert rel in index or rel in named or rel in skill_text or path.name in skill_text, rel


def test_read_names_an_errors_sections_and_what_a_message_cites():
    out = drive.read_for("error", ["the solve failed: Linear solver failed at Newton iteration 0"])
    assert ("references/troubleshooting.md", "Solve") in {(e["file"], e.get("section")) for e in out}
    # A solve stop whose graph solve failed names the solve table; one the fallback rescued does not.
    failed = {"ok": True, "stop": "solve", "solve_error": "Newton solver did not converge in 20 iterations"}
    assert "references/troubleshooting.md" in {e["file"] for e in drive.read_for("solve", drive._texts(failed), failed)}
    rescued = {**failed, "fallback": {"backend": "pyamgx", "error": None}}
    assert "references/troubleshooting.md" not in {e["file"] for e in drive.read_for("solve", drive._texts(rescued), rescued)}
    # A flat reference cited without its section is not named whole; an optimization's solve adds its own.
    assert drive.read_for("error", ["see geometry.md"]) == []
    optimized = {"solve": {"optimization": {"iterations": 3}}}
    assert ("references/judging_results.md", "7. Shape-optimization results") in {
        (e["file"], e.get("section")) for e in drive.read_for("solve", [], optimized)}
    cited = drive.read_for("error", ["refused: build the simplified body (objectives/routes.md §3.4); see "
                                     "visual_checks.md §1"])
    files = {(e["file"], e.get("section")) for e in cited}
    assert ("references/objectives/routes.md", None) in files
    assert ("references/visual_checks.md", "1. `geometry/views.png` (after generation)") in files
    geometry = drive.read_for("geometry", [])
    assert ("references/objectives/regions.md", None) in {(e["file"], e.get("section")) for e in geometry}
    start = {e["file"] for e in drive.read_for("start", [])}
    assert {"references/objectives/brief.md", "references/objectives/routes.md",
            "references/cli_protocol/plan.md"} <= start


def test_a_section_file_of_the_split_references_keeps_its_rules():
    ref = SCRIPTS.parent / "references"
    assert "## 2. The case brief" in (ref / "objectives" / "brief.md").read_text(encoding="utf-8")
    assert "### 3.4 Getting the body" in (ref / "objectives" / "routes.md").read_text(encoding="utf-8")
    assert "## 6. The plan file" in (ref / "cli_protocol" / "plan.md").read_text(encoding="utf-8")
    skill = " ".join((SCRIPTS.parent / "SKILL.md").read_text(encoding="utf-8").split())
    assert "The plan skeleton (objectives §4)" in skill and "--finish" in skill
    assert "conda --version" not in skill and "All are part of the skill" not in skill
    assert "nothing already in the conversation" in skill and "Never cat, wc, head or tail" in skill


def _judged_session(tmp_path):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "report.md").write_text(
        "# Session abc\n\n## Warnings\n- none\n\n## Engineer's judgement\n\n_(fill in)_\n")
    lib.write_json(session / "operator" / "prediction_history.json", [{"run": "r", "solved": {}}])   # a solve ran
    return session


def test_judge_finish_puts_the_judgement_in_and_ends_the_session(tmp_path, monkeypatch):
    session = _judged_session(tmp_path)
    judgement = tmp_path / "j.md"
    judgement.write_text("The brief's case was run.\n## Deviations\nnone\nThe body deflects 13 um.\n")
    printed, finished = [], []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "finish_session", lambda args, show=True: finished.append(show) or {
        "ok": True, "stage": "complete", "complete": True, "card": {"kind": "big"}})
    drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc", file=str(judgement),
                                       text=None, finish=True))
    assert finished == [False] and len(printed) == 1                 # one command, one result
    assert printed[-1]["ok"] and printed[-1]["finished"] == {"ok": True, "stage": "complete", "complete": True}
    assert "The body deflects 13 um." in (session / "operator" / "report.md").read_text()
    log = [json.loads(line) for line in (session / "operator" / "driver_log.jsonl").read_text().splitlines()]
    assert log[-1]["command"] == "judge --finish" and log[-1]["stop"] == "finished"
    # A finish the session refuses (another card waiting) is the command's result.
    monkeypatch.setattr(drive, "finish_session", lambda args, show=True: {"ok": False, "error": "the bc_input card"})
    drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc", file=str(judgement),
                                       text=None, finish=True))
    assert not printed[-1]["ok"] and printed[-1]["finished"]["error"] == "the bc_input card"


def test_judge_finish_finishes_nothing_when_the_judgement_is_refused(tmp_path, monkeypatch):
    _judged_session(tmp_path)
    printed, finished = [], []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "finish_session", lambda args, show=True: finished.append(1) or {"ok": True})
    drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc", file=None,
                                       text="## Case\nfine", finish=True))
    assert not printed[-1]["ok"] and "verdict" in printed[-1]["error"] and finished == []
    assert drive.parser().parse_args(["judge", "abc", "--file", "j.md", "--finish"]).finish


def test_start_names_the_drawing_words_before_the_first_drawing():
    # the ~20-word limit decides the first drawing's words, so it is read before that spend
    named = {(e["file"], e.get("section")) for e in drive.read_for("start", [])}
    assert any(f == "references/geometry.md" and (s or "").startswith("SDXL drawing") for f, s in named)
    assert "about 20 words" in (SCRIPTS.parent / "SKILL.md").read_text(encoding="utf-8")


def test_the_compact_view_keeps_every_warning_error_and_solve_status(tmp_path, capsys):
    value = {**_regions_stop(), "ok": False, "stop": "error",
             "warnings": ["thin walls: 2 members under 4 leaves"],
             "step_warnings": ["SDXL never read: plain background, seen from"],
             "error": "the solve failed", "errors": ["plan.scale: missing"],
             "solve_error": "Newton solver did not converge in 20 iterations",
             "solve": {"succeeded": False, "newton": {"converged": False, "iterations": 20},
                       "checks": {"agreement": 0.42}}}
    shown = json.loads(_printed(capsys, value, tmp_path / "abc"))
    for key in ("warnings", "step_warnings", "error", "errors", "solve_error"):
        assert shown[key] == value[key], key
    assert shown["solve"]["newton"] == {"converged": False, "iterations": 20}
    assert shown["solve"]["checks"]["agreement"] == 0.42 and shown["ok"] is False


def test_a_stop_prints_its_compact_view_unless_full_and_records_what_it_named(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    value = {**_regions_stop(), "attempts": {"regions": 1}, "read": drive.read_for("regions", [])}
    drive.show_stop(value, session, full=False)
    assert printed[-1]["_compact"] and "attempts" not in printed[-1] and printed[-1]["read"]
    drive.show_stop(value, session, full=True)
    assert printed[-1] is value                                   # every number, as saved
    assert lib.read_json(session / "operator" / "shown.json")["read"]


# ------------------------------------------- one round of token, time and retry fixes (2026-10-09) --

def test_merge_patch_follows_rfc_7396():
    target = {"a": {"b": 1, "c": [1, 2]}, "d": "x"}
    assert drive.merge_patch(target, {"a": {"b": None, "c": [3]}, "e": {"f": 1}}) == \
        {"a": {"c": [3]}, "d": "x", "e": {"f": 1}}
    assert target == {"a": {"b": 1, "c": [1, 2]}, "d": "x"}           # the target is not changed
    assert drive.merge_patch(target, ["whole"]) == ["whole"]


def test_run_patch_writes_the_plan_back_and_refuses_a_bad_patch_leaving_it_unchanged(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    state = {"card": {"kind": "geometry_input", "description": ""}}
    sent, printed = [], []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": state["card"], "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        sent.append(prepared.answer)
        state["card"] = {"kind": "geometry_input", "description": "revised " + prepared.answer["description"]}
        return {"ok": True, "card": state["card"]}
    monkeypatch.setattr(drive, "_run_step", step)
    plan_path = session / "operator" / "plan.json"
    plan_path.write_text(json.dumps({**PLAN, "geometry": {"description": "first words", "generator": "cadcoder",
                                                          "revise_prompt": True}}))

    def run(patch):
        drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(plan_path),
                                         full=True, patch=patch))
        return printed[-1]

    first = run('{"geometry": {"description": "second words"}}')
    assert first["stop"] == "revision"
    # The session's first patch says the earlier Read is stale: keep to --patch, or Read first.
    assert first["warnings"][0] == drive.PATCH_REWROTE
    assert "keep changing the plan with --patch, or Read plan.json before a Write or Edit" in drive.PATCH_REWROTE
    assert sent[-1]["description"] == "second words"
    on_disk = json.loads(plan_path.read_text())
    assert on_disk["geometry"] == {"description": "second words", "generator": "cadcoder", "revise_prompt": True}
    before = plan_path.read_text()
    for bad in ('{"geometry": ', '["not an object"]', '{"regions": ""}', '{"material": null}'):
        out = run(bad)
        assert out["stop"] == "error" and not out["ok"] and "unchanged" in out["error"], (bad, out)
        assert plan_path.read_text() == before
    assert len(sent) == 1                                          # nothing sent, nothing spent
    # The patch keeps the file's own layout, so an Edit of a line as written still finds it.
    assert plan_path.read_text() == json.dumps(json.loads(plan_path.read_text()))
    for indent in (2, "\t"):
        plan_path.write_text(json.dumps(json.loads(before), indent=indent) + "\n")
        out = run('{"geometry": {"description": "third words"}}')
        assert drive.PATCH_REWROTE not in (out.get("warnings") or [])      # said once, layout kept
        text = plan_path.read_text()
        assert text == json.dumps(json.loads(text), indent=indent) + "\n" and "third words" in text
    # A file laid out by hand is written in one layout, and the stop says to re-read it.
    hand = json.dumps(json.loads(before), indent=1).replace('"cadcoder",\n  "revise', '"cadcoder", "revise')
    assert drive.json_layout(hand) is None
    plan_path.write_text(hand)
    out = run('{"geometry": {"description": "fourth words"}}')
    assert out["warnings"][0] == drive.PATCH_REWROTE and "fourth words" in plan_path.read_text()


def test_the_drawing_stop_shows_the_drawing_and_its_cut_out_as_one_picture(tmp_path):
    from PIL import Image
    block = np.zeros((512, 512), dtype=bool)
    block[100:400, 150:350] = True
    cut = _cut_out(tmp_path, "d_s0", block)
    raw = tmp_path / "d_s0_drawn.png"
    Image.new("RGB", (512, 512), (240, 240, 240)).save(raw)
    card = {"drawn_image": str(cut), "drawn_image_raw": str(raw), "drawn_image_prompt": "a block", "image_seed": 0}
    payload = drive._stop_payload("drawing", tmp_path, tmp_path, card, PLAN)
    assert len(payload["look_at"]) == 1
    pair = Image.open(payload["look_at"][0])
    width, height = pair.size
    assert width == 2 * (height - max(14, (width // 2) // 24))         # two square panels and a label band
    assert width * height <= 1.1 * 512 * 512                            # about one drawing's pixels


def test_field_and_texture_pictures_are_shrunk_to_about_1000_by_700(tmp_path):
    from PIL import Image
    big = tmp_path / "field.png"
    Image.new("RGB", (1400, 980), (1, 2, 3)).save(big)
    inspect_session.shrink(big, inspect_session.FIELD_PICTURE)
    assert Image.open(big).size == (1000, 700)
    views = tmp_path / "views.png"
    Image.new("RGB", (1400, 980)).save(views)
    small = drive.downscaled(views, tmp_path / "operator" / "texture_views.png")
    assert Image.open(small).size == (1000, 700) and Image.open(views).size == (1400, 980)


def _geometry_stop() -> dict:
    stations = [{"at_m": 0.03 + 0.06 * k, "pieces": [
        {"area_m2": 0.005 - 0.001 * k, "centroid_m": [0.05, 0.04], "I_m4": {"about_y": 2e-6, "about_z": 3e-6},
         "size_m_along_y_z": [0.1, 0.08], "length_over_size": [2.9, 3.6]},
        {"area_m2": 1e-9, "centroid_m": [0.0, 0.0], "I_m4": {"about_y": 1e-15, "about_z": 1e-15},
         "size_m_along_y_z": [0.001, 0.001], "length_over_size": [300, 300]}]} for k in range(5)]
    return {"ok": True, "session_id": "abc", "stop": "geometry", "kind": "geometry_physics",
            "warnings": ["thin walls: the body's thinnest member is 0.3 leaves at depth 8, under 4"],
            "planned_resolution": {
                "physical_extents": [0.29, 0.1131, 0.1552], "depth": 7, "leaf_size_h": 0.00435,
                "planned_mass": 0.6197, "mass_over_real": 1.24, "member_over_real": 2.38,
                "thin_gaps": {"share_of_surface": 0.0157, "leaf_m": 0.00435, "width_m": [0.00051, 0.00123],
                              "where": [[-0.14, -0.05, -0.005], [0.13, 0.04, 0.14]]},
                "thin_walls": {"share_of_surface": 0.2189, "leaf_m": 0.00435, "width_m": [0.00084, 0.00932],
                               "where": [[-0.14, -0.06, -0.005], [0.14, 0.04, 0.14]]},
                "thinnest_member_leaves": 0.1, "thinnest_member_leaves_at_depth_8": 0.3,
                "sections": {"along": "x", "across": ["y", "z"], "length_m": 0.29, "stations": stations},
                "load_path": {"members_crossing_cuts": {"x": [2, 5, 1], "y": [2, 8, 26], "z": [4, 3, 1]},
                              "reach_beyond_base_m": {"+x": 0.0387, "-x": 0.0182},
                              "base_footprint_m": [[-0.12, 0.0], [0.1, 0.14]], "note": "how to read it"},
                "real_member": {"real_member_m": 0.008, "unresolvable": True},
                "resolution_check": "the body's thinnest member is 0.1 leaves at depth 7, under 4"},
            "read": [drive.reference_section("visual_checks.md", "## 1.", "the geometry checks and the decision"),
                     drive.reference_section("objectives/size.md", None, "the generated prediction")]}


def test_the_geometry_stop_prints_sections_load_path_and_boxes_as_one_line_each(tmp_path, capsys):
    value = _geometry_stop()
    shown = json.loads(_printed(capsys, value, tmp_path / "abc"))
    res = shown["planned_resolution"]
    # the stations decide nothing at a body that is no stand-in: in the saved stop only
    assert "sections" not in res
    assert res["load_path"]["summary"].startswith("members crossing cuts at 1/4, 1/2, 3/4: x [2, 5, 1]")
    assert "+x 0.0387" in res["load_path"]["summary"] and "base_footprint" not in json.dumps(res["load_path"])
    assert res["load_path"]["note"] == "how to read it"
    assert "where" not in res["thin_gaps"] and res["thin_gaps"]["width_m"] == [0.00051, 0.00123]
    assert "where" not in res["thin_walls"]
    for key in ("thinnest_member_leaves", "thinnest_member_leaves_at_depth_8", "resolution_check",
                "mass_over_real", "member_over_real", "planned_mass", "real_member"):
        assert res[key] == value["planned_resolution"][key], key
    assert shown["warnings"] == value["warnings"]
    lines = drive.reference_section("visual_checks.md", "## 1.", "x")["lines"]
    assert shown["read"] == [f"references/visual_checks.md:{lines[0]}-{lines[1]} the geometry checks and the decision",
                             "references/objectives/size.md the generated prediction"]
    # The saved stop keeps every station; a section stand-in's are not printed either (the runner
    # picks its cut), and its ratios are printed without their prose
    assert isinstance(value["planned_resolution"]["sections"]["stations"], list)
    stand_in = _geometry_stop()
    stand_in["planned_resolution"]["section_stand_in"] = {"station_m": 0.15, "stiffness_ratio": 2.0,
                                                          "stress": "what the ratio means, at length"}
    res = json.loads(_printed(capsys, stand_in, tmp_path / "other"))["planned_resolution"]
    assert "sections" not in res and res["section_stand_in"] == {"station_m": 0.15, "stiffness_ratio": 2.0}
    # a repeat geometry stop prints what changed, the rest as before
    again = _geometry_stop()
    again["planned_resolution"]["mass_over_real"] = 9.9
    res = json.loads(_printed(capsys, again, tmp_path / "abc"))["planned_resolution"]
    assert res["mass_over_real"] == 9.9 and res["rest"].startswith("as before") and "thinnest_member_leaves" not in res
    # a changed measure the decisions do not read is left to operator/stop_geometry.json
    again["planned_resolution"]["fill_fraction"] = 0.5
    res = json.loads(_printed(capsys, again, tmp_path / "abc"))["planned_resolution"]
    assert "fill_fraction" not in res


def test_start_prints_what_is_not_ready_and_no_paths_or_field_lists():
    started = {"ok": True, "session_id": "abc", "checkpoint_id": "c", "stage": "new", "kind": "geometry_input",
               "card": {"kind": "geometry_input", "fields": ["description", "asset_path"],
                        "cadcoder": {"ready": True, "summary": "CAD-Coder ready"},
                        "hunyuan3d": {"ready": False, "summary": "Hunyuan3D weights missing"},
                        "sdxl": {"ready": True, "root": "/x/y", "interpreter": "/x/bin/python", "summary": "SDXL ready"}},
               "notes": ["workers warming"], "model": "m", "vision_model": "m",
               "read": [drive.reference_section("objectives/brief.md", None, "the case brief")]}
    out = drive.compact_start(started)
    text = json.dumps(out)
    assert "interpreter" not in text and "/x/" not in text and "fields" not in text
    assert out["ready"] == ["cadcoder", "sdxl"] and out["not_ready"] == {"hunyuan3d": "Hunyuan3D weights missing"}
    assert out["read"] == ["references/objectives/brief.md the case brief"] and out["session_id"] == "abc"
    assert "vision_model" not in out


@pytest.mark.parametrize("member, section, warned", [
    (0.008, "0.02 m deep, I 6.0e-8 m4, A 1.8e-3 m2", True),        # a solid sole, a side wall named
    (0.008, "0.08 m deep, I 2.01e-6 m4, A 5.03e-3 m2", True),      # a solid shank, a shed rim named
    (0.02, "0.02 m deep, I 6.0e-8 m4, A 1.8e-3 m2", False),        # the sole itself
    (0.019, "0.25 m deep, I 2.08e-4 m4, A 2.14e-2 m2", False),     # a hollow box's wall
    (0.3, "0.3 m deep, I 4.5e-3 m4, A 0.6 m2", False),             # a slab
    (0.012, "0.118 m deep, I 3.0e-5 m4, A 0.010 m2", False),       # a hollow casting's wall
])
def test_a_real_member_far_thinner_than_a_solid_real_section_is_warned(tmp_path, member, section, warned):
    (tmp_path / "brief.md").write_text(f"## Size and material\nReal thinnest member: {member} m\nReal section: {section}\n")
    out = drive.brief_member_warning(tmp_path)
    assert bool(out) == warned
    if warned:
        assert "the thinnest member the load passes through" in out and "thin_off_load_path" in out


def _pre_body(tmp_path, first, then, warned):
    """A brief written at the first run, maybe warned at a drawing stop, edited, then a body."""
    operator = tmp_path / "operator"
    operator.mkdir(parents=True)
    (operator / "brief.md").write_text(first)
    assert drive.brief_status(operator, snapshot=False)["present"]          # a drawing stop
    assert not (tmp_path / "brief_snapshots" / "brief.first.md").exists()
    assert (tmp_path / "brief_snapshots" / "brief_initial.md").exists()     # the first run's brief
    if warned:
        assert drive.brief_member_warning(operator)
        drive.record_member_warning(operator)
    (operator / "brief.md").write_text(then)
    return operator, drive.brief_status(operator)                            # the body's snapshot


SOLE = "Real section: 0.02 m deep, I 6.0e-8 m4, A 1.8e-3 m2\n"


def test_a_real_member_moved_after_the_member_warning_is_allowed_and_stated(tmp_path):
    operator, status = _pre_body(tmp_path, "## Size and material\nReal thinnest member: 0.008 m\n" + SOLE,
                                 "## Size and material\nReal thinnest member: 0.02 m\n"
                                 "Thinnest member why: the loaded web is 20 mm\n" + SOLE, warned=True)
    # taken with its reason (without one it is not, test_loophole_rules)
    assert drive.real_member(operator) == 0.02 and status["real_lines_edited"] == []
    [edit] = status["real_lines_edited_before_body"]
    assert "Real thinnest member: 0.008 -> 0.02" in edit and "allowed, stated" in edit
    line = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, status)
    assert "Real thinnest member: 0.008 -> 0.02" in line


def test_a_real_line_edited_before_any_body_is_the_brief_being_written(tmp_path):
    # the Real lines freeze at the first body, not at the first run (round 10 audit): any edit before
    # the body is taken, and stated
    hollow = "Real section: 0.25 m deep, I 2.08e-4 m4, A 2.14e-2 m2\n"
    operator, status = _pre_body(tmp_path, "## Size and material\nReal thinnest member: 0.019 m\n" + hollow,
                                 "## Size and material\nReal thinnest member: 0.03 m\n" + hollow, warned=False)
    assert drive.real_member(operator) == 0.03
    [edit] = status["real_lines_edited_before_body"]
    assert "0.019 -> 0.03" in edit and "allowed, stated" in edit
    other = tmp_path / "other"
    operator, status = _pre_body(other, "## Size and material\nReal thinnest member: 0.008 m\n" + SOLE,
                                 "## Size and material\nReal thinnest member: 0.008 m\n"
                                 "Real section: 0.04 m deep, I 4.8e-7 m4, A 3.6e-3 m2\n", warned=True)
    assert drive.real_section(operator)["depth_m"] == 0.04
    shown = drive.compact_stop({"stop": "geometry", "brief": status}, None)
    assert shown["brief"]["real_lines_edited_before_body"]
    report = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, status)
    assert "Real section" in report and "allowed, stated" in report


def test_no_generated_prediction_is_asked_for(tmp_path):
    operator = tmp_path / "abc" / "operator"
    operator.mkdir(parents=True)
    (operator / "brief.md").write_text("## Use\nsits\n## Prediction\n- Mode: bending.\n")
    shown = drive.compact_stop({"stop": "geometry", "brief": drive.brief_status(operator)}, None)["brief"]
    assert "add" not in shown and "why_not" not in shown and "prediction_generated" not in shown


def test_the_first_drawing_stop_says_the_real_member_in_leaves(tmp_path):
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text("## Size and material\nReal thinnest member: 0.008 m\n")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 0.3}}
    payload = drive._stop_payload("drawing", tmp_path, tmp_path, {"drawn_image_prompt": "an object"}, plan)
    member = payload["real_member"]
    leaf = 1.92 * 0.3 / 256
    assert member["real_member_leaves_at_depth_8"] == round(0.008 / leaf, 2)
    assert abs(member["members_needed_m"] - 2 * leaf) < 1e-4
    assert "never refused" in member["note"] and "are not changed to meet it" in member["note"]
    assert "real_member" in drive.brief_run({**payload, "stop": "drawing"})


def test_the_regions_stop_warns_of_held_pieces_the_approval_will_refuse():
    rows = [{"name": "ground#1", "region": "ground", "face": "side face", "member_axis": "x", "span_m": [0, 0.1]},
            {"name": "ground#2", "region": "ground", "face": "side face", "member_axis": "x", "span_m": [0.2, 0.3]}]
    # reference_hold {} takes a region no contact lists, and the approval does not count it
    assert drive.held_pieces_warning(rows, {}) is None
    waiting = {"floor": {"transmits": "force and moment", "regions": []}}
    out = drive.held_pieces_warning(rows, waiting)
    assert "ground splits into 2 held pieces" in out and "\"pieces\": 2" not in out and "pieces_why" in out
    assert drive.held_pieces_warning(rows, {"floor": {**waiting["floor"], "pieces": 2, "pieces_why": "two feet"}}) is None
    declared = {"floor": {"transmits": "force and moment", "regions": ["ground"], "pieces": 2, "pieces_why": "two feet"}}
    assert drive.held_pieces_warning([{**r, "contact": "floor"} for r in rows], declared) is None
    bound = {"floor": {"transmits": "force and moment", "regions": ["ground"]}}
    assert "the runner sees 2 held pieces" in drive.held_pieces_warning([{**r, "contact": "floor"} for r in rows], bound)
    assert drive.held_pieces_warning(rows[:1], waiting) is None


def test_held_pieces_that_jump_after_a_reword_are_flagged_and_the_report_gives_the_reason(tmp_path):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    rows = lambda n: [{"name": f"base#{k}", "region": "base"} for k in range(n)]
    assert drive.held_pieces_jump(session, "Fix the base.", rows(12)) is None
    assert drive.held_pieces_jump(session, "Fix the base.", rows(12)) is None      # same words: no change
    out = drive.held_pieces_jump(session, "Fix the flat base.", rows(20))
    assert "went 12 -> 20 after the reword" in out and "say so in pieces_why" in out
    assert "rather than" not in out
    assert drive.held_pieces_jump(session, "Fix the flat base.", rows(20)) == out  # the same stop reads the same
    assert drive.held_pieces_jump(session, "Fix the base again.", rows(22)) is None  # a small change
    # the same words on a new segmentation: said as that, not as a reword
    seg = drive.held_pieces_jump(session, "Fix the base again.", rows(40), {"num_clusters": 8})
    assert "after the segmentation changed (the words did not)" in seg and "reword" not in seg
    # the declared reason is printed beside the contact in the report
    (session / "operator" / "brief.md").write_text("## Contacts\nfeet on the floor\n")
    contacts = {"feet": {"transmits": "force and moment", "regions": ["base"], "pieces": 2,
                         "pieces_why": "the stand has two feet"}}
    row = {"name": "base#1", "contact": "feet", "face": "end face", "held": [0, 1, 2], "length_leaves7": 3}
    lines = drive.contacts_table(session, [row], contacts)
    assert "declared 2 pieces (the stand has two feet)" in lines[0]


def test_depth_8_is_offered_for_thin_gaps_only_where_its_leaf_opens_them(tmp_path):
    import trimesh
    a = trimesh.creation.box(extents=[1.0, 0.04, 0.04])
    b = a.copy()
    b.apply_translation([0.0, 0.05, 0.0])                    # 10 mm of air: under a depth-7 leaf only
    session = _built_session(tmp_path, trimesh.util.concatenate([a, b]), 0.04,
                             "0.04 m deep, I 2.1e-7 m4, A 1.6e-3 m2")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "settings": {}}
    out = drive.thin_gaps_problem(session, plan)
    assert "bonds the two sides" in out and "take depth 8, whose leaf" in out
    c = a.copy()
    c.apply_translation([0.0, 0.043, 0.0])                   # 3 mm: under both leaves
    close = _built_session(tmp_path / "close", trimesh.util.concatenate([a, c]), 0.04,
                           "0.04 m deep, I 2.1e-7 m4, A 1.6e-3 m2")
    out = drive.thin_gaps_problem(close, plan)
    assert "bonds the two sides" in out and "take depth 8" not in out


def test_blank_names_every_missing_item_in_one_error(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    args = argparse.Namespace(output_root=str(tmp_path), session_id="abc", shape="extrude", size=None, radius=None,
                              length=None, axis="z", profile="0,0 1,0 1,1 0,1", hole=None, circle=None)
    drive.cmd_blank(args)
    error = printed[-1]["error"]
    assert "operator/brief.md" in error and "--length L for --shape extrude" in error
    (session / "operator" / "brief.md").write_text("## Size and material\nReal mass: 3 kg\n")
    drive.cmd_blank(args)
    error = printed[-1]["error"]
    assert "'Real thinnest member: <number> m'" in error and "'Real section (<Load path member>):" in error and "--length L" in error
    assert "'Real mass" not in error
    # a missing heading and a missing Real line are named together, not one run each
    assert "without a 'Use' heading" in error and "without a 'Contacts' heading" in error
    assert "'Prediction'" not in error                      # the runner's prediction stands
    cylinder = argparse.Namespace(**{**vars(args), "shape": "cylinder", "profile": None})
    drive.cmd_blank(cylinder)
    assert "--radius R for --shape cylinder" in printed[-1]["error"] and "--length L" in printed[-1]["error"]
    help_text = drive.parser()._subparsers._group_actions[0].choices["blank"].format_help()
    assert "cylinder and extrude: required" in help_text and "cylinder: required" in help_text
    # the usage error and the help say the axis convention
    assert "--axis z: the length runs along z" in printed[-1]["error"]
    assert "required, the axis the length runs along" in " ".join(help_text.split())
    assert "--axis does not apply" in drive.blank_axis("box", "y")
    assert "lie on y and z" in drive.blank_axis("extrude", "x")


def test_the_geometry_stop_keeps_a_stand_in_under_two_leaves_with_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: {
        "depth": (p.get("settings") or {}).get("remesh_octree_depth", 7), "physical_extents": [1.0, 0.5, 0.4]})
    monkeypatch.setattr(drive, "resolution_warning", lambda s, p, r: "thin member: 1.8 leaves at depth 8")
    monkeypatch.setattr(drive, "thin_member_limit", lambda *a: {"real_member_m": 0.019, "unresolvable": True})
    monkeypatch.setattr(drive, "_generated_mesh", lambda s: None)
    monkeypatch.setattr(inspect_session, "summarize", lambda *a, **k: {"warnings": []})
    monkeypatch.setattr(drive, "section_equivalence", lambda *a: {})
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["x"]})
    monkeypatch.setattr(drive, "body_gate", lambda *a, **k: None)
    (tmp_path / "geometry").mkdir(exist_ok=True)
    payload = drive._stop_payload("geometry", tmp_path, tmp_path, {"kind": "geometry_physics"}, PLAN)
    assert payload.get("ok") is not False and payload["warnings"][0] == "thin member: 1.8 leaves at depth 8"
    note = payload["planned_resolution"]["real_member"]["note"]
    assert "resolution caveat" in note and "not a refusal" in note and "`blank`" not in note


THIN_BRIEF = ("## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", "
              "\"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
              "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\n"
              "Real thinnest member: MEMBER m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n"
              "0.9 m tall, wood 1.2e10\nLoad path: {\"outline\": \"series\"}\n"
              "## Prediction\nmembers in compression, 0.1 MPa, microns")
THIN_BRIEF = tabled(THIN_BRIEF)


def test_the_plan_check_warns_of_a_real_member_under_two_leaves_with_its_numbers(tmp_path):
    (tmp_path / "operator").mkdir()
    brief = tmp_path / "operator" / "brief.md"
    # extent 0.1 m: leaf 1.5 mm at depth 7, 0.75 mm at depth 8
    brief.write_text(THIN_BRIEF.replace("MEMBER", "0.004"))
    assert drive.member_plan_warning(tmp_path, PLAN) is None
    brief.write_text(THIN_BRIEF.replace("MEMBER", "0.002"))          # 1.3 leaves at 7, 2.7 at 8
    said = drive.member_plan_warning(tmp_path, PLAN)
    assert said.startswith("thin member (plan check): the brief's Real thinnest member 0.002 m is 1.33 leaves at depth 7")
    assert "2.67 at depth 8" in said and "not a refusal" in said
    deep = {**PLAN, "settings": {"remesh_octree_depth": 8}}
    assert drive.member_plan_warning(tmp_path, deep) is None
    brief.write_text(THIN_BRIEF.replace("MEMBER", "0.001"))          # 1.3 leaves even at depth 8
    said = drive.member_plan_warning(tmp_path, deep)
    assert "1.33 leaves at depth 8" in said and "depth 8" in said and "resolution caveat" in said
    assert drive.member_unresolvable(tmp_path, PLAN)


def test_one_threshold_of_two_leaves_for_measured_and_nominal_sizes(tmp_path):
    (tmp_path / "operator").mkdir()
    brief = tmp_path / "operator" / "brief.md"
    assert drive.RESOLVED_WITH_ALLOWANCE == drive.RESOLVED_LEAVES == 2
    assert drive.stand_in_minimum({"physical_extents": [0.1]}) == float(f"{2 * 0.00075:.3g}")
    brief.write_text(THIN_BRIEF.replace("MEMBER", "0.0014"))         # 1.9 leaves at depth 8
    assert drive.member_unresolvable(tmp_path, PLAN)
    brief.write_text(THIN_BRIEF.replace("MEMBER", "0.0016"))         # 2.1 leaves at depth 8
    assert not drive.member_unresolvable(tmp_path, PLAN)
    leaf = 0.00075
    measured = {"depth": 8, "leaf_size_h": leaf,
                "thin_walls": {"share_of_surface": 0.05, "width_m": [1.9 * leaf], "leaf_m": leaf}}
    assert drive.thinnest_leaves(measured) == 1.9
    assert "under 2" in drive.resolution_warning(tmp_path, PLAN, measured)


def _thin_run(tmp_path, monkeypatch, member, plan, card_kind="geometry_input", body=None):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True, exist_ok=True)
    (session / "checkpoints.sqlite").write_text("")
    if not (session / "operator" / "brief.md").exists():
        (session / "operator" / "brief.md").write_text(THIN_BRIEF.replace("MEMBER", member))
    sent, printed = [], []
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": {"kind": card_kind}, "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "body_source", lambda s: body)

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        sent.append(prepared.answer)
        return {"ok": True, "card": {"kind": "geometry_texture"}}
    monkeypatch.setattr(drive, "_run_step", step)
    plan_path = session / "operator" / "plan.json"
    plan_path.write_text(json.dumps(plan))
    drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(plan_path), full=True))
    return printed[-1], sent


def test_the_runner_leaves_the_brief_as_the_agent_wrote_it(tmp_path, monkeypatch):
    out, _ = _thin_run(tmp_path / "run", monkeypatch, "0.05", PLAN)
    brief = (tmp_path / "run" / "abc" / "operator" / "brief.md").read_text()
    assert "Prediction (generated)" not in brief and "Prediction (generated)" not in str(out.get("warnings"))


def test_run_warns_of_the_plan_check_without_a_stop_and_refuses_no_step(tmp_path, monkeypatch):
    plan = {**PLAN, "settings": {"remesh_octree_depth": 8}}
    out, sent = _thin_run(tmp_path, monkeypatch, "0.001", plan)
    # the free-form step is sent: the warning rides on the run's stop, said once
    assert out["stop"] != "plan" and sent
    assert any(w.startswith("thin member (plan check)") for w in out.get("warnings") or [])
    out, sent = _thin_run(tmp_path, monkeypatch, "0.001", plan)
    assert sent and not any("plan check" in w for w in out.get("warnings") or [])
    # a member that resolves at the plan's depth: nothing said
    out, sent = _thin_run(tmp_path / "other", monkeypatch, "0.05", plan)
    assert sent and not any("plan check" in w for w in out.get("warnings") or [])


def test_a_patch_is_written_only_once_the_session_lock_is_held_and_apostrophes_are_named(tmp_path, monkeypatch):
    import contextlib
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    plan_path = session / "operator" / "plan.json"
    plan_path.write_text(json.dumps(PLAN))
    before = plan_path.read_text()

    @contextlib.contextmanager
    def busy(session):
        raise SystemExit('{"ok": false, "error": "another driver is working on this session"}')
        yield
    monkeypatch.setattr(drive, "session_lock", busy)
    with pytest.raises(SystemExit):
        drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(plan_path),
                                         full=True, patch='{"regions": "Fix the base. Apply a traction on the top."}'))
    assert plan_path.read_text() == before
    with pytest.raises(drive.PlanError, match="u0027"):
        drive.patched_plan(plan_path, "{\"regions\": \"the segment")


def test_the_drawing_composite_names_the_largest_object_when_drawing_crop_takes_it(tmp_path, monkeypatch):
    from PIL import Image
    labels = []
    monkeypatch.setattr(drive, "drawing_pair", lambda raw, cut, out, label="": labels.append(label) or None)
    block = np.zeros((64, 64), dtype=bool)
    block[16:48, 16:48] = True
    cut = _cut_out(tmp_path, "d", block)
    raw = tmp_path / "d_drawn.png"
    Image.new("RGB", (64, 64)).save(raw)
    card = {"drawn_image": str(cut), "drawn_image_raw": str(raw), "drawn_image_prompt": "a block", "image_seed": 0}
    crop = {**PLAN, "geometry": {"description": "a block", "draw": True, "drawing_crop": "largest"}}
    drive._stop_payload("drawing", tmp_path, tmp_path, card, crop)
    drive._stop_payload("drawing", tmp_path, tmp_path, card, PLAN)
    assert "largest object alone" in labels[0] and labels[1] == "cut-out (what the body is generated from)"


# ------------------------------------------------------------- the description --

def test_the_free_form_description_is_short_with_no_mesh_or_negated_words():
    routes = " ".join((drive.SKILL_DIR / "references" / "objectives" / "routes.md").read_text().split())
    geometry = (drive.SKILL_DIR / "references" / "geometry.md").read_text()
    assert '"<material> <object>, <2-3 defining shape features>"' in routes and "solid, watertight" not in routes
    writing = " ".join(geometry.split("## Writing the description")[1].split("\n## ")[0].split())
    assert "at most about 20 words" in writing and "No negation" in writing and "no mesh words" in writing


# --------------------------------------------------------------- the drawing limit --

def test_the_last_drawing_says_a_failed_one_is_not_generated_from(tmp_path):
    (tmp_path / "operator").mkdir()
    card = {"drawn_image_prompt": "an object", "image_seed": 2}
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"drawing": 2})
    assert "last_drawing" not in drive._stop_payload("drawing", tmp_path, tmp_path, card, PLAN)["drawing"]
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"drawing": 3})
    last = drive._stop_payload("drawing", tmp_path, tmp_path, card, PLAN)["drawing"]["last_drawing"]
    assert "3 of 3" in last and "generated from by no route" in last
    assert "fresh drawing made for Hunyuan3D" in last and "Hunyuan3D's own attempt" in last
    # a drawing made for Hunyuan3D is its own attempt, under its own limit
    hunyuan = {**PLAN, "geometry": {**PLAN["geometry"], "generator": "hunyuan3d", "draw": True}}
    assert "last_drawing" not in drive._stop_payload("drawing", tmp_path, tmp_path, card, hunyuan)["drawing"]
    assert "fresh drawing made for Hunyuan3D" in drive.limit_message("drawing", {"drawing": 4})
    geometry = (drive.SKILL_DIR / "references" / "geometry.md").read_text()
    drawing = geometry.split("## SDXL drawing")[1].split("\n## ")[0]
    assert "is never generated from" in drawing and "Hunyuan3D's own attempt" in drawing


# ------------------------------------------------------------------ blank's usage --

def test_blank_needs_the_axis_and_says_what_the_brief_is(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    args = argparse.Namespace(output_root=str(tmp_path), session_id="abc", shape="extrude", size=None, radius=None,
                              length=0.05, axis=None, profile="0,0 1,0 1,1 0,1", hole=None, circle=None)
    drive.cmd_blank(args)
    error = printed[-1]["error"]
    # the brief, whole: its five headings and its Real lines, then blank again
    for words in ("'## Use'", "'## Contacts'", "'## Load'", "'## Size and material'", "'## Prediction'",
                  "'Real mass: <n> kg'", "'Real thinnest member: <n> m'", "'Real section:", "then run blank again"):
        assert words in error, words
    # --axis has no default, and the convention comes with the refusal
    assert "--axis x|y|z for --shape extrude, which has no default" in error
    assert "--axis z: profile in x,y; --axis y: in x,z; --axis x: in y,z" in error
    assert "the axis is the profile's depth" in error
    box = argparse.Namespace(**{**vars(args), "shape": "box", "size": [0.1, 0.2, 0.3], "profile": None,
                                "length": None})
    drive.cmd_blank(box)
    assert "--axis" not in printed[-1]["error"]
    assert drive.parser().parse_args(["blank", "abc", "--shape", "extrude"]).axis is None
    skill = " ".join((drive.SKILL_DIR / "SKILL.md").read_text().split())
    assert "`blank` comes after the brief" in skill and "`--axis` is required" in skill


def _generated_body(session, content=b"body", generator="triposg", mode="image"):
    (session / "geometry").mkdir(parents=True, exist_ok=True)
    (session / "geometry" / "generated.glb").write_bytes(content)
    lib.write_json(session / "session.json", {"geometry_mode": mode, "generation": {"generator": generator}})


THIN_RESOLUTION = {"depth": 8, "leaf_size_h": 0.0075, "physical_extents": [1.0, 0.5, 0.4],
                   "thin_walls": {"share_of_surface": 0.5, "leaf_m": 0.0075, "width_m": [0.003, 0.004],
                                  "thinnest_over_1pct_m": 0.003, "volume_share": 0.45}}


def test_cad_prompt_revisions_have_a_budget_before_each_generation(tmp_path):
    (tmp_path / "operator").mkdir()
    revise = {"description": "a frame of three bars", "generator": "cadcoder", "revise_prompt": True}
    assert drive.cad_revision_stage(revise) == drive.CAD_REVISIONS
    assert drive.cad_revision_stage({**revise, "generator": "triposg"}) is None
    assert drive.cad_revision_stage({**revise, "revise_prompt": False}) is None
    for n in range(drive.RUN_LIMITS[drive.CAD_REVISIONS]):
        charged = drive.charge_attempt(tmp_path, drive.CAD_REVISIONS, {**revise, "description": f"words {n}"})
        assert charged["counted"] and not charged["over"]
    # the same words again are still a revision call: counted
    assert drive.charge_attempt(tmp_path, drive.CAD_REVISIONS, revise)["over"]
    out = drive.limit_message(drive.CAD_REVISIONS, {}, ["cadcoder", "drawing"])
    assert "generate from the revision the card shows" in out and "SDXL drawing" in out
    # a CAD generation restores the budget, which every stop before a CAD step shows
    assert drive.CAD_REVISIONS not in drive.restart_cad_revisions(tmp_path)
    assert drive.CAD_REVISIONS in drive.STOP_STAGES["revision"] and drive.CAD_REVISIONS in drive.STOP_STAGES["plan"]


def test_the_cad_revision_budget_is_shown_only_on_a_cad_route(tmp_path):
    (tmp_path / "operator").mkdir()
    counts = {"drawing": 1}
    drawn = {**PLAN, "geometry": {"description": "a frame of three bars", "draw": True}}
    cad = {**PLAN, "geometry": {"description": "a frame of three bars", "generator": "cadcoder", "revise_prompt": True}}
    # a drawn or generated body off the CAD route: no CAD budget among the stages left
    left = drive.attempts_left(tmp_path, counts, drawn)
    assert drive.CAD_REVISIONS not in left and left["drawing"] == drive.RUN_LIMITS["drawing"] - 1
    # the plan names a CAD generator: shown
    assert drive.attempts_left(tmp_path, {}, cad)[drive.CAD_REVISIONS] == drive.RUN_LIMITS[drive.CAD_REVISIONS]
    # the body is a CAD generator's now (the session's generation settings), whatever the plan says now
    state = {"geometry_mode": "text", "generation": {"generator": "cadcoder"}}
    (tmp_path / "session.json").write_text(json.dumps(state))
    assert drive.CAD_REVISIONS in drive.attempts_left(tmp_path, {}, drawn)
    # an upload replaced it: off the route again
    (tmp_path / "session.json").write_text(json.dumps({**state, "geometry_mode": "surface"}))
    assert drive.CAD_REVISIONS not in drive.attempts_left(tmp_path, {}, drawn)
    # no plan given: the plan as the last run declared it
    (tmp_path / "operator" / "declared.json").write_text(json.dumps(cad))
    assert drive.CAD_REVISIONS in drive.attempts_left(tmp_path, {})
    # and the compact stop keeps it off where it is absent (the 0-or-1 rule adds nothing back)
    shown = drive.compact_stop({"stop": "plan", "attempts_left": drive.attempts_left(tmp_path, {}, drawn)}, None)
    assert drive.CAD_REVISIONS not in shown["attempts_left"]


def _painted(tmp_path, name, colour_of):
    """A painted sphere: each vertex coloured by ``colour_of(vertex)`` (sRGB 0..255), with a little noise."""
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=4)
    rng = np.random.default_rng(0)
    rgb = np.array([colour_of(v) for v in sphere.vertices], dtype=float) + rng.normal(0, 3, (len(sphere.vertices), 3))
    sphere.visual = trimesh.visual.ColorVisuals(sphere, vertex_colors=np.clip(rgb, 0, 255).astype(np.uint8))
    path = tmp_path / f"{name}.glb"
    sphere.export(path)
    return path


def _held_cap(session, axis, sign):
    """A session whose physical body is the unit sphere, held on the cap facing ``sign`` along ``axis``."""
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=4)
    (session / "geometry").mkdir(parents=True, exist_ok=True)
    (session / "boundary_conditions").mkdir(parents=True, exist_ok=True)
    sphere.export(session / "geometry" / "physical.glb")
    held = sign * sphere.triangles_center[:, axis] > 0.85
    np.savez(session / "boundary_conditions" / "resolved.npz", dirichlet_0=held)
    (session / "boundary_conditions" / "resolved.json").write_text(json.dumps(
        {"regions": [{"key": "dirichlet_0", "category": "dirichlet", "name": "held"}]}))


def _counted(said):
    """The counted facings' list of a near-black warning."""
    return said.split("counted, by facing:")[1].split(";")[0]


def test_a_partial_paint_is_warned_never_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "mark_paint_views", lambda stats, out, compute_env=None: None)
    wood = (170, 110, 60)
    good = _painted(tmp_path, "good", lambda v: wood)
    assert drive.paint_defect_warning(good, "warm brown wood") is None
    # a black patch on a side, on a colour that is not dark: the share, by facing
    patch = _painted(tmp_path, "patch", lambda v: (5, 5, 5) if v[0] < -0.85 else wood)
    said = drive.paint_defect_warning(patch, "warm brown wood")
    assert "near-black" in said and "-x " in _counted(said) and "a warning only" in said
    # no answer to paste: the views first, a repaint only for a defect they show
    assert '"texture_seed": ' not in said and "Look at the painted views first" in said
    assert "only if the views show the defect" in said
    assert drive.paint_stats(patch)["dark_share"] > drive.PAINT_DARK_SHARE
    # the same black asked for is no defect
    assert drive.paint_defect_warning(patch, "black painted steel") is None
    # two tones between facings: one side much darker than the other
    two = _painted(tmp_path, "two", lambda v: (90, 90, 90) if v[0] > 0 else (215, 215, 215))
    said = drive.paint_defect_warning(two, "light grey stone")
    assert "facings differ by" in said and '"texture_seed": ' not in said and "near-black" not in said
    assert drive.paint_defect_warning(tmp_path / "missing.glb", "x") is None
    # the texture stop carries it as a warning beside the views
    (tmp_path / "operator").mkdir()
    payload = drive._stop_payload("texture", tmp_path, tmp_path, {
        "textured_surface": str(patch), "texture_prompt": "warm brown wood", "texture_seed": 0}, PLAN)
    assert "near-black" in payload["warnings"][0] and payload["texture"]["textured_surface"] == str(patch)


def test_near_black_is_dark_grey_and_a_resting_facing_reads_as_shading(tmp_path):
    wood = (170, 110, 60)
    # a deep saturated colour is dark but not near-black: no black patch
    deep = _painted(tmp_path, "deep", lambda v: (60, 0, 8) if v[0] < -0.85 else wood)
    assert drive.paint_stats(deep)["dark_share"] == 0.0
    assert drive.paint_defect_warning(deep, "warm brown wood") is None
    # dark grey facing straight down (the views' -y) is listed apart and does not warn on its own
    under = _painted(tmp_path, "under", lambda v: (5, 5, 5) if v[1] < -0.85 else wood)
    stats = drive.paint_stats(under)
    assert stats["dark_share"] > drive.PAINT_DARK_SHARE and stats["dark_share_away_from_resting"] == 0.0
    assert stats["dark_share_by_facing"]["-y"] > 0 and "-y" in stats["resting_facings"]
    assert drive.paint_defect_warning(under, "warm brown wood") is None
    # a facing that rests on a held region likewise; the same patch with nothing held there warns
    side = _painted(tmp_path, "side", lambda v: (5, 5, 5) if v[0] > 0.85 else wood)
    assert "near-black" in drive.paint_defect_warning(side, "warm brown wood")
    session = tmp_path / "session"
    _held_cap(session, 0, +1)
    assert drive.held_facings(session) == [1]
    assert drive.paint_defect_warning(side, "warm brown wood", session=session) is None
    # black off the resting facings still warns, the resting share named apart
    both = _painted(tmp_path, "both", lambda v: (5, 5, 5) if v[0] > 0.85 or v[2] < -0.85 else wood)
    said = drive.paint_defect_warning(both, "warm brown wood", session=session)
    assert "near-black" in said and "-z " in _counted(said) and "+x " not in _counted(said)
    assert "+x " in said.split("not counted):")[1]


def test_a_darker_underside_is_shading_and_a_black_base_is_not(tmp_path):
    wood = (170, 110, 60)
    # the downward facing a dark brown, not near-black: no two tones (it is not counted)
    under = _painted(tmp_path, "under", lambda v: (60, 40, 20) if v[1] < -0.5 else wood)
    assert drive.paint_defect_warning(under, "warm brown oak wood") is None
    # the same split on a side facing is two tones
    side = _painted(tmp_path, "side", lambda v: (60, 40, 20) if v[0] < -0.5 else wood)
    assert "facings differ" in drive.paint_defect_warning(side, "warm brown oak wood")
    # a downward facing near-black all over is a black base: counted, and warned
    base = _painted(tmp_path, "base", lambda v: (5, 5, 5) if v[1] < -0.5 else wood)
    stats = drive.paint_stats(base)
    assert "-y" not in stats["resting_facings"] and stats["dark_share_away_from_resting"] > drive.PAINT_DARK_SHARE
    assert "near-black" in drive.paint_defect_warning(base, "warm brown oak wood")


def test_a_paint_mostly_off_the_one_hue_asked_for_is_warned(tmp_path):
    grey, red = (150, 150, 150), (200, 30, 30)
    # a grey body with the hue only on small parts (here the caps)
    spotted = _painted(tmp_path, "spotted", lambda v: red if abs(v[1]) > 0.9 else grey)
    said = drive.paint_defect_warning(spotted, "bright red painted cast iron")
    assert "of the surface is red" in said and "Look at the painted views first" in said
    # the hue asked for over the body, a deep shade of it underneath included: no warning
    assert drive.paint_defect_warning(_painted(tmp_path, "red", lambda v: red), "red painted cast iron") is None
    shaded = _painted(tmp_path, "shaded", lambda v: (90, 10, 15) if v[1] < 0 else red)
    assert drive.paint_defect_warning(shaded, "red painted cast iron") is None
    for name, colour, words in (("brown", (170, 110, 60), "warm brown oak wood"),
                                ("tan", (205, 170, 120), "light tan maple wood"),
                                ("oak", (215, 180, 130), "light brown oak wood")):
        assert drive.paint_defect_warning(_painted(tmp_path, name, lambda v: colour), words) is None, words
    # no single hue named, or a paint of several colours or parts: hue is not judged
    for words in ("light grey concrete", "grey body, red trim", "white glazed ceramic"):
        assert drive.paint_defect_warning(spotted, words) is None, words
    assert drive.paint_defect_warning(spotted, "bright red painted cast iron", mode="parts") is None
    assert drive.paint_colours("bright red painted cast iron")["hue"]["word"] == "red"
    assert drive.paint_colours("light grey steel")["hue"] is None
    assert drive.paint_colours("blue grey steel")["hue"] is None


def test_the_hue_bands_read_shades_and_keep_neighbouring_hues_apart(tmp_path):
    # a pale or dark shade word reads the hue from a lower chroma: these shades come out correct
    for name, colour, words in (("pale", (245, 215, 222), "pale pink ceramic"),
                                ("light", (190, 210, 225), "light blue painted steel"),
                                ("dark", (80, 58, 45), "dark brown oak wood")):
        painted = _painted(tmp_path, name, lambda v: colour)
        assert drive.paint_defect_warning(painted, words) is None, words
    # without the shade word the same pale colour is under the hue's chroma
    pale = _painted(tmp_path, "pale2", lambda v: (245, 215, 222))
    assert "of the surface is pink" in drive.paint_defect_warning(pale, "pink ceramic")
    # red is held off wood brown; pure blue sits well inside blue's band
    brown = _painted(tmp_path, "brownish", lambda v: (170, 110, 60))
    assert "of the surface is red" in drive.paint_defect_warning(brown, "red painted steel")
    assert drive.paint_defect_warning(_painted(tmp_path, "blue", lambda v: (0, 0, 255)), "blue painted steel") is None
    centre, half = drive.HUE_ANGLES["blue"]
    assert half - abs(306 - centre) >= 15
    assert abs(drive.HUE_ANGLES["cyan"] - 197) <= 2
    # a dark colour word right before a hue is one colour, a shade of it
    asked = drive.paint_colours("navy blue painted steel")
    assert asked["several"] is False and asked["hue"]["word"] == "navy blue"
    navy = _painted(tmp_path, "navy", lambda v: (30, 40, 120))
    assert drive.paint_defect_warning(navy, "navy blue painted steel") is None


def test_the_paint_check_reads_crimson_as_red_pale_teal_as_light_blue_and_dark_materials_as_dark(tmp_path):
    """A held-out run (2026-10-10): a crimson red read as "9 % red", a pale teal glass asked light
    blue as "1 % blue", and a carbon fibre weave as a near-black patch, which pushed a repaint."""
    for name, colour in (("crimson", (220, 20, 60)), ("magenta_red", (200, 0, 80)), ("pink_red", (230, 40, 90)),
                         ("pure", (255, 0, 0))):
        painted = _painted(tmp_path, name, lambda v: colour)
        assert drive.paint_defect_warning(painted, "red painted steel") is None, name
    # still held off wood brown
    brown = _painted(tmp_path, "brown_again", lambda v: (150, 90, 50))
    assert "of the surface is red" in drive.paint_defect_warning(brown, "red painted steel")
    # a light blue reaches through cyan to teal; a plain blue does not
    teal = _painted(tmp_path, "teal_glass", lambda v: (160, 210, 200))
    assert drive.paint_defect_warning(teal, "light blue glass") is None
    assert "of the surface is blue" in drive.paint_defect_warning(
        _painted(tmp_path, "teal_paint", lambda v: (0, 128, 128)), "blue painted steel")
    # a dark material asks for dark paint, whatever colour word stands beside it
    weave = _painted(tmp_path, "weave", lambda v: (20, 20, 22) if int(5 * (v[0] + v[1] + v[2] + 3)) % 2
                     else (120, 120, 120))       # a weave on every facing, not one dark side
    for words in ("mid grey woven carbon fibre", "black rubber", "light grey graphite composite"):
        assert drive.paint_colours(words)["dark"], words
        assert drive.paint_defect_warning(weave, words) is None, words
    assert "near-black" in drive.paint_defect_warning(weave, "mid grey painted steel")


def test_a_failing_hatch_keeps_the_plain_views(tmp_path, monkeypatch):
    from PIL import Image
    patch = _painted(tmp_path, "patch", lambda v: (5, 5, 5) if v[0] < -0.85 else (170, 110, 60))
    views = tmp_path / "views.png"
    Image.new("RGB", (60, 40), "white").save(views)
    (tmp_path / "operator").mkdir()

    def broken(stats, out, compute_env=None):
        raise RuntimeError("renderer failed")
    monkeypatch.setattr(drive, "mark_paint_views", broken)
    payload = drive._stop_payload("texture", tmp_path, tmp_path, {
        "textured_surface": str(patch), "texture_views": str(views), "texture_prompt": "warm brown wood",
        "texture_seed": 0}, PLAN)
    assert payload["look_at"] == [str(views)] and "near-black" in payload["warnings"][0]
    assert "hatched" not in payload["warnings"][0] and "look at the faces named" in payload["warnings"][0]


def test_the_texture_stop_shows_the_measured_vertices_hatched(tmp_path, monkeypatch):
    from PIL import Image
    wood = (170, 110, 60)
    patch = _painted(tmp_path, "patch", lambda v: (5, 5, 5) if v[0] < -0.85 else wood)
    views = tmp_path / "views.png"
    Image.new("RGB", (60, 40), "white").save(views)
    card = {"textured_surface": str(patch), "texture_views": str(views), "texture_prompt": "warm brown wood",
            "texture_seed": 0}
    (tmp_path / "operator").mkdir()
    drawn = []

    def mark(stats, out, compute_env=None):
        drawn.append(int(stats["flagged"].sum()))
        Image.new("RGB", (60, 40), "magenta").save(out)
        return "magenta"
    monkeypatch.setattr(drive, "mark_paint_views", mark)
    payload = drive._stop_payload("texture", tmp_path, tmp_path, card, PLAN)
    # the marked views replace the plain ones, and the warning says what the hatching is
    assert payload["look_at"][0].endswith("texture_views_marked.png") and str(views) not in payload["look_at"]
    assert drawn and drawn[0] > 0 and "hatched in magenta" in payload["warnings"][0]
    # a paint that passes draws nothing and keeps the plain views
    drawn.clear()
    good = _painted(tmp_path, "good", lambda v: wood)
    payload = drive._stop_payload("texture", tmp_path, tmp_path, {**card, "textured_surface": str(good)}, PLAN)
    assert not drawn and payload["look_at"] == [str(views)] and "warnings" not in payload


def test_a_plain_colour_fallback_is_a_marked_texture_stop_and_paint_errors_read_paint(tmp_path):
    """A held-out run (2026-10-10): the painter crashed four times and the error named the
    geometry section.  A crash now ends in a plain colour the stop marks, with nothing to repaint;
    a paint message names the Paint section, not Geometry."""
    (tmp_path / "operator").mkdir()
    flat = _painted(tmp_path, "flat", lambda v: (243, 218, 121))
    fallback = {"kind": "plain_colour", "crashed_seeds": [0, 1], "colour": [243, 218, 121],
                "colour_word": "light yellow", "note": "NOT PAINTED: the painter crashed on this surface"}
    payload = drive._stop_payload("texture", tmp_path, tmp_path, {
        "textured_surface": str(flat), "texture_prompt": "light yellow metal", "texture_seed": 0,
        "texture_fallback": fallback}, PLAN)
    assert payload["texture"]["texture_fallback"] == fallback
    assert payload["warnings"][0].startswith("NOT PAINTED") and "never repaint" in payload["warnings"][0]
    crash = ("RuntimeError: paint_hunyuan3d_surface failed: crashed (SIGSEGV, return code -11) without "
             "a Python error; the same inputs crash again; see x.stderr.log")
    sections = {entry.get("section") for entry in drive.read_for("error", [crash])}
    assert "Paint" in sections and "Geometry" not in sections
    generator = "RuntimeError: generate_hunyuan3d_image_surface failed: Hunyuan3D-2.1 produced an empty surface"
    assert "Geometry" in {entry.get("section") for entry in drive.read_for("error", [generator])}


def test_the_marked_views_are_drawn_by_the_workflow_renderer(tmp_path):
    import inspect_session
    if inspect_session.compute_interpreter(lib.DEFAULT_COMPUTE_ENV) is None:
        pytest.skip("no compute environment to render in")
    from PIL import Image
    patch = _painted(tmp_path, "patch", lambda v: (5, 5, 5) if v[0] < -0.5 else (170, 110, 60))
    found, stats = drive.paint_findings(patch, "warm brown wood")
    assert found and stats["flagged"].any()
    out = tmp_path / "marked.png"
    assert drive.mark_paint_views(stats, out) == "magenta"
    pixels = np.asarray(Image.open(out).convert("RGB")).reshape(-1, 3).astype(int)
    hatched = (pixels[:, 0] > 150) & (pixels[:, 1] < 60) & (pixels[:, 2] > 150)
    assert hatched.sum() > 100


def test_a_paint_of_several_colours_or_a_dark_finish_is_read_as_asked(tmp_path):
    wood = (170, 110, 60)
    patch = _painted(tmp_path, "patch", lambda v: (5, 5, 5) if v[0] < -0.85 else wood)
    two = _painted(tmp_path, "two", lambda v: (90, 90, 90) if v[0] > 0 else (215, 215, 215))
    # a dark colour or a dark finish alone asks for dark paint
    for prompt in ("navy painted steel", "oiled walnut", "slate", "cast iron"):
        assert "near-black" not in (drive.paint_defect_warning(patch, prompt) or ""), prompt
    # a dark material painted a lighter colour still asks for the lighter colour
    assert "near-black" in drive.paint_defect_warning(patch, "blue painted cast iron")
    # a dark colour among several: the black is not judged
    assert drive.paint_defect_warning(patch, "black steel frame, beige fabric seat") is None
    # several colours, or per-part materials: two tones may be meant
    assert drive.paint_defect_warning(two, "white frame, grey panel") is None
    assert drive.paint_defect_warning(two, "light grey stone", mode="parts") is None
    assert "facings differ" in drive.paint_defect_warning(two, "light grey stone", mode="uniform")
    # a shade word is no second colour
    asked = drive.paint_colours("dark brown oak")
    assert asked["dark"] is True and asked["several"] is False and asked["hue"]["word"] == "brown"


def test_a_run_refused_for_the_lock_writes_no_once_records(tmp_path, monkeypatch):
    import contextlib
    import trimesh
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "geometry").mkdir()
    (session / "checkpoints.sqlite").write_text("")
    trimesh.creation.box(extents=[0.7, 1.0, 0.4]).export(session / "geometry" / "original.glb")
    lib.write_json(session / "session.json", {"geometry_mode": "surface"})
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n"
        "## Contacts\nheld at the base, fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\n"
        "Real section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    plan_path = session / "operator" / "plan.json"
    plan_path.write_text(json.dumps({**PLAN, "contacts": {}}))
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))

    @contextlib.contextmanager
    def busy(session):
        raise SystemExit("another driver is working on this session")
        yield
    monkeypatch.setattr(drive, "session_lock", busy)
    with pytest.raises(SystemExit):
        drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(plan_path), full=True))
    assert not printed            # nothing said: the lock refused the run before any record
    assert not (session / "operator" / drive.CONTACTS_NAMED).exists()
    assert not (session / "operator" / drive.DEPTH_SET).exists()


def test_the_contact_names_are_printed_once_at_a_stop():
    note = drive.CONTACTS_KEYS_HEAD + '"hold at the base" (force and moment)'
    assert drive.stop_run_notes({"contacts_keys": note}, [note, "another note"]) == ["another note"]
    assert drive.stop_run_notes({}, [note]) == [note]


def test_a_declared_thin_part_over_an_appendage_is_warned_with_its_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: json.loads(json.dumps(THIN_RESOLUTION)))
    monkeypatch.setattr(drive, "thin_member_limit", lambda *a: {"real_member_m": 0.002, "unresolvable": True,
                                                                "real_member_leaves_at_depth_8": 0.3})
    monkeypatch.setattr(drive, "section_equivalence", lambda *a: {})
    monkeypatch.setattr(drive, "_generated_mesh", lambda s: None)
    monkeypatch.setattr(inspect_session, "summarize", lambda *a, **k: {"warnings": []})
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["x"]})
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text('## Size and material\nLoad path: {"k": "series"}\n')
    _generated_body(tmp_path)
    plan = {**PLAN, "settings": {"remesh_octree_depth": 8}, "thin_off_load_path": "the braces; they carry nothing",
            "body_simplification": {"kept": "k", "left_out": "l"}}
    payload = drive._stop_payload("geometry", tmp_path, tmp_path, {"kind": "geometry_physics"}, plan)
    assert payload.get("ok") is not False and "error" not in payload
    warning = next(w for w in payload["warnings"] if w.startswith("thin part declared"))
    assert "50 % of the surface and 45 % of the volume" in warning and "resolution caveat" in warning
    # no volume share (a body that is not watertight): the surface share stands in, said so
    walls = {**THIN_RESOLUTION, "thin_walls": {**THIN_RESOLUTION["thin_walls"], "volume_share": None}}
    said = drive.thin_cover_warning(tmp_path, plan, walls)
    assert "not measured (the body is not watertight)" in said and "50 % of the surface and 50 % of the volume" in said


# ------------------------------------------------------- thin_off_load_path's reach --

def _thin_on_base(tmp_path, held_faces):
    """A thick base (8 leaves) beside a thin plate (1.6 leaves at depth 8, half the volume), with
    supports on the faces given: a list of face masks, one Dirichlet region each."""
    import trimesh
    base = trimesh.creation.box(extents=[1.0, 0.06, 0.06])
    plate = trimesh.creation.box(extents=[1.0, 0.3, 0.012])
    plate.apply_translation([0.0, 0.3, 0.0])
    body = trimesh.util.concatenate([base, plate])
    session = _built_session(tmp_path, body, 0.04, "0.04 m deep, I 2.1e-7 m4, A 1.6e-3 m2")
    regions, masks = [], {}
    for k, faces in enumerate(held_faces):
        mask = np.zeros(len(body.faces), dtype=bool)
        mask[faces] = True
        masks[f"dirichlet_{k}"] = mask
        regions.append({"category": "dirichlet", "key": f"dirichlet_{k}", "name": f"held{k}"})
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": regions})
    np.savez(session / "boundary_conditions" / "resolved.npz", **masks)
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850},
            "settings": {"remesh_octree_depth": 8}, "thin_off_load_path": "the plate; it carries nothing"}
    return session, plan


def test_only_thin_walls_inside_the_supports_faces_are_taken_off(tmp_path, gaps_not_read):
    import os
    base, plate = list(range(12)), list(range(12, 24))
    # supports on the thick base (two overlapping regions, counted once) do not clear the thin plate
    session, plan = _thin_on_base(tmp_path / "base", [base, base[:6]])
    resolution = drive.planned_resolution(session, plan)
    measured = drive.thin_volume_outside_holds(session, resolution["leaf_size_h"])
    assert measured["volume_share"] > drive.THIN_OFF_PATH_SHARE
    assert measured["outside_share"] == pytest.approx(measured["volume_share"])
    said = drive.thin_cover_warning(session, plan, resolution)
    assert "outside the supports' faces" in said and "main members" in said
    # a support holding the thin plate whole: nothing thin is left outside it
    session, plan = _thin_on_base(tmp_path / "plate", [plate])
    resolution = drive.planned_resolution(session, plan)
    assert drive.thin_volume_outside_holds(session, resolution["leaf_size_h"])["outside_share"] < 0.01
    assert drive.thin_cover_warning(session, plan, resolution) is None
    # regions resolved on an earlier body do not count
    for name in ("resolved.json", "resolved.npz"):
        os.utime(session / "boundary_conditions" / name, ns=(1, 1))
    assert drive.thin_volume_outside_holds(session, resolution["leaf_size_h"]) is None
    assert "main members" in drive.thin_cover_warning(session, plan, resolution)


def test_a_thin_part_declaration_covers_an_appendage_not_the_main_members(tmp_path, gaps_not_read):
    import trimesh
    leaf = 1.92 / 256
    # a fin reads far more of the surface than of the volume: the volume share is the measure
    lipped = drive.thin_walls(_lipped_bar(), leaf)
    assert lipped["share_of_surface"] > 0.4 and lipped["volume_share"] < drive.THIN_OFF_PATH_SHARE
    plate = drive.thin_walls(trimesh.creation.box(extents=[1.0, 0.012, 0.3]), leaf)
    assert plate["volume_share"] > 0.9
    # a body that is all thin wall: the declaration does not clear it, and the numbers are named
    session = _built_session(tmp_path, trimesh.creation.box(extents=[1.0, 0.012, 0.012]), 0.04,
                             "0.04 m deep, I 2.1e-7 m4, A 1.6e-3 m2")
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.0},
            "material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850},
            "settings": {"remesh_octree_depth": 8}, "thin_off_load_path": "the bracing bars; they carry nothing"}
    resolution = drive.planned_resolution(session, plan)
    said = drive.thin_cover_warning(session, plan, resolution)
    assert "99 % of the surface and 100 % of the volume" in said and "1.6-1.6 leaves thick" in said
    assert "over the 20 %" in said and "depth 8" in said
    # no declaration: the callers' own message, not this one
    assert drive.thin_cover_warning(session, {**plan, "thin_off_load_path": None}, resolution) is None
    # a cached measure from before the volume share existed is measured again
    path = session / "operator" / "planned_resolution.json"
    cache = lib.read_json(path, {})
    for entry in cache.values():
        entry["thin_walls"].pop("volume_share", None)
    lib.write_json(path, cache)
    assert drive.planned_resolution(session, plan)["thin_walls"]["volume_share"] == pytest.approx(1.0)
    # a body that is not watertight: the volume share is said missing, not left out
    open_box = trimesh.creation.box(extents=[1.0, 0.012, 0.3])
    open_box.update_faces(np.arange(len(open_box.faces)) != 0)
    assert drive.thin_walls(open_box, leaf)["volume_share"] is None


def test_blank_s_usage_line_shows_the_axis_as_required():
    commands = (drive.SKILL_DIR / "references" / "cli_protocol" / "commands.md").read_text()
    usage = next(line for line in commands.splitlines() if "drive.py blank SID" in line)
    assert "[--axis" not in usage and "--length L --axis x\\|y\\|z" in usage


# ------------------------------------------------------- frames of bars (`blank --bars`) --

def _apex_frame(section=0.045, members=4):
    """``members`` bars radiating from one joint down to points on a circle, a block over the joint."""
    import math
    bars = [{"from": [round(0.6 * math.cos(a), 6), 0.0, round(0.6 * math.sin(a), 6)], "to": [0.0, 1.2, 0.0],
             "section": [section, section]} for a in (2 * math.pi * k / members for k in range(members))]
    return {"bars": bars, "boxes": [{"min": [-0.07, 1.2, -0.07], "max": [0.07, 1.26, 0.07]}]}


# Two posts and a beam in one plane, a strut from the beam's middle to the ground out of that
# plane (a T joint), and a cross member whose ends sit a little off the posts' axes (as points
# written by hand do).
TWO_PLANE_FRAME = [{"from": [0.0, 0.0, 0.0], "to": [0.0, 1.2, 0.0], "section": [0.05, 0.05]},
                   {"from": [0.8, 0.0, 0.0], "to": [0.8, 1.2, 0.0], "section": [0.05, 0.05]},
                   {"from": [0.0, 1.2, 0.0], "to": [0.8, 1.2, 0.0], "section": [0.05, 0.05]},
                   {"from": [0.4, 0.0, 0.7], "to": [0.4, 1.2, 0.0], "section": [0.05, 0.05]},
                   {"from": [0.02, 0.5, 0.0], "to": [0.78, 0.5, 0.01], "section": [0.055, 0.055]}]


def _bars_of(spec):
    return spec["bars"] if isinstance(spec, dict) else spec


def test_a_frame_of_bars_in_several_planes_is_one_watertight_body_that_resolves(monkeypatch):
    # the remesh (MeshLab) only reshapes the faces; the union is what is under test here
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    for spec, members in ((_apex_frame(), (4, 1)), (_apex_frame(0.05, 3), (3, 1)), (TWO_PLANE_FRAME, (5, 0))):
        mesh, built = drive.build_frame(spec)
        assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1
        assert (built["bars"], built["boxes"]) == members and built["joints"] >= 1
        points = np.array([p for b in _bars_of(spec) for p in (b["from"], b["to"])])
        span = points.max(axis=0) - points.min(axis=0)
        assert (mesh.extents >= span - 1e-6).all() and (mesh.extents <= span + 0.25).all()
        assert np.allclose(mesh.bounds[0], 0.0)                  # the lowest corner at the origin
        # the geometry stop's measures at depth 8: no wall under 4 leaves (the joints leave no lip)
        # and no gap under a leaf (the hubs fill the notches between arms)
        leaf = 1.92 * max(mesh.extents) / 2 ** drive.DEEPEST_DEPTH
        walls = drive.thin_walls(mesh, leaf)
        assert drive.thinnest_leaves({"thin_walls": {**walls, "leaf_m": leaf}}) is None
        assert drive.thin_gaps(mesh, leaf)["share_of_surface"] < drive.THIN_GAP_SHARE
        assert built["smallest_section_m"] == min(min(b["section"]) if isinstance(b["section"], list)
                                                  else b["section"]["d"] for b in _bars_of(spec))


def test_a_frame_names_its_faults_before_building(monkeypatch):
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    apart = [{"from": [0, 0, 0], "to": [0, 1, 0], "section": [0.05, 0.05]},
             {"from": [1, 0, 0], "to": [1, 1, 0], "section": [0.05, 0.05]}]
    with pytest.raises(drive.PlanError, match=r"\[bar 0\]; \[bar 1\] are apart"):
        drive.build_frame(apart)
    with pytest.raises(drive.PlanError, match="section"):
        drive.build_frame([{"from": [0, 0, 0], "to": [0, 1, 0], "section": 0.05}])
    with pytest.raises(drive.PlanError, match="two different points"):
        drive.build_frame([{"from": [0, 0, 0], "to": [0, 0, 0], "section": [0.05, 0.05]}])
    with pytest.raises(drive.PlanError, match="above"):
        drive.build_frame({"bars": [], "boxes": [{"min": [0, 0, 0], "max": [1, 0, 1]}]})
    # a rectangle's w side follows w_along across the bar
    u, v, t = drive._bar_frame([0, 1, 0], [0, 0, 1])
    assert np.allclose(u, [0, 0, 1]) and np.allclose(t, [0, 1, 0]) and np.isclose(np.dot(np.cross(u, v), t), 1)
    # a stub shorter than the half sections, both ends on one joint, is kept as given, not refused
    stub = TWO_PLANE_FRAME + [{"from": [0.0, 1.2, 0.0], "to": [0.0, 1.21, 0.0], "section": [0.06, 0.06]}]
    mesh, built = drive.build_frame(stub)
    assert mesh.is_watertight and built["stubs_kept_as_given"] == ["bar 5"]


def test_blank_bars_prints_what_it_built_and_a_section_that_does_not_resolve(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(THIN_BRIEF.replace("MEMBER", "0.002"))
    args = dict(output_root=str(tmp_path), session_id="abc", shape=None, size=None, radius=None, length=None,
                axis=None, profile=None, hole=None, circle=None)
    bars = tmp_path / "frame.json"
    bars.write_text(json.dumps(_apex_frame(0.012)))
    drive.cmd_blank(argparse.Namespace(**args, bars=str(bars)))
    # under 2 leaves: built, with the thin bars named in the warning
    out = printed[-1]
    assert out["ok"] and out["path"].endswith(".glb") and "blank_bars_" in out["path"]
    assert "thin member: bar 0, bar 1, bar 2, bar 3 (0.012 m thinnest)" in out["warning"]
    # the bars are kept beside the body, for a kept bar's check
    assert lib.read_json(Path(out["path"]).with_suffix(".bars.json"), None) == _apex_frame(0.012)
    built = out["built"]
    assert (built["bars"], built["boxes"], built["smallest_section_m"]) == (4, 1, 0.012)
    assert built["smallest_section_leaves_at_depth_8"] < drive.RESOLVED_WITH_ALLOWANCE
    assert out["plan"]["scale"]["physical_extent"] == max(out["extents_m"]) and "axis" not in out
    bars.write_text(json.dumps(_apex_frame(0.09)))
    drive.cmd_blank(argparse.Namespace(**args, bars=str(bars)))
    assert "warning" not in printed[-1]["built"] and "thin member" not in str(printed[-1].get("warning"))
    # one of --shape or --bars, each named
    drive.cmd_blank(argparse.Namespace(**args, bars=None))
    assert "--shape box|cylinder|extrude for one section, or --bars FILE.json" in printed[-1]["error"]
    drive.cmd_blank(argparse.Namespace(**{**args, "shape": "box"}, bars=str(bars)))
    assert "--shape or --bars, not both" in printed[-1]["error"]
    # the frame route is in the skill text read before the plan, and its command in commands.md
    routes = (drive.SKILL_DIR / "references" / "objectives" / "routes.md").read_text()
    assert "**Members in several planes**" in routes and "`blank SID --bars FILE.json`" in routes
    commands = (drive.SKILL_DIR / "references" / "cli_protocol" / "commands.md").read_text()
    assert "drive.py blank SID --bars FILE.json" in commands
    assert drive.parser().parse_args(["blank", "abc", "--bars", "f.json"]).bars == "f.json"


# --------------------------------------------- a contact whose member the stand-in leaves out --

ABSENT_ROWS = [{"name": "base", "category": "dirichlet"}, {"name": "joint", "category": "dirichlet"},
               {"name": "joint_load", "category": "neumann"}]


def test_a_contact_absent_from_the_stand_in_keeps_a_support_of_its_own(tmp_path):
    declared = {"support A": "force only", "support B": "force only"}
    kept = {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}
    moved = {"transmits": "force only", "regions": ["joint_load"],
             "absent_in_stand_in": {"carried_by": "load_at_joint"}}
    contacts = {"support A": kept, "support B": moved}
    # the missing contact names the declaration on a stand-in
    said = drive.contacts_problem({"support A": kept}, declared, stand_in=True)
    assert "missing from plan.contacts" in said and "absent_in_stand_in" in said
    # structured, with a region of its own: the contacts pass, and the binding reads it as any contact
    assert drive.contacts_problem(contacts, declared, ["base"], stand_in=True) is None
    assert drive.binding_problem({"contacts": contacts}, ABSENT_ROWS) is None
    # free words (the mapping written in prose), no region, another contact's region: refused; no word count
    prose = {**moved, "absent_in_stand_in": "carried by the other supports of the body"}
    assert "carried_by is one of" in drive.contacts_problem({**contacts, "support B": prose}, declared, stand_in=True)
    bare = {**moved, "regions": []}
    assert "names its region on the stand-in" in drive.contacts_problem({**contacts, "support B": bare}, declared,
                                                                         stand_in=True)
    mapped = {**moved, "regions": ["base"]}
    said = drive.contacts_problem({**contacts, "support B": mapped}, declared, stand_in=True)
    assert "mapped onto another contact's region ['base (support A)']" in said
    assert "is for a stand-in" in drive.contacts_problem(contacts, declared)
    # the joint's regions may sit inside the declaration: they become the contact's own
    nested = {"contacts": {"support B": {"transmits": "force only", "absent_in_stand_in": {
        "carried_by": "held_at_joint", "regions": ["joint"]}}}}
    assert drive.normalize_absent(nested)["contacts"]["support B"]["regions"] == ["joint"]
    # on the preview: load_at_joint needs this contact's statics reaction on a load region of its own
    plan = {"contacts": contacts, "body_simplification": {"kept": "k", "left_out": "l"}}
    reacted = {"reactions": [{"region": "joint_load", "force_N": [0, 10, 0]}]}
    assert drive.absent_support_problem(plan, ABSENT_ROWS, reacted) is None
    assert "do not carry this contact's reaction" in drive.absent_support_problem(plan, ABSENT_ROWS, {"reactions": []})
    clamped = {**contacts, "support B": {**moved, "transmits": "force and moment"}}
    assert "transmits \"force only\"" in drive.absent_support_problem({**plan, "contacts": clamped}, ABSENT_ROWS, reacted)
    # held_at_joint needs a held region; a region not on the preview is named
    held = {**moved, "regions": ["joint"], "absent_in_stand_in": {**moved["absent_in_stand_in"], "carried_by": "held_at_joint"}}
    assert drive.absent_support_problem({**plan, "contacts": {**contacts, "support B": held}}, ABSENT_ROWS, None) is None
    loose = {**held, "regions": ["joint_load"]}
    assert "are not held regions" in drive.absent_support_problem({**plan, "contacts": {**contacts, "support B": loose}},
                                                                  ABSENT_ROWS, None)
    gone = {**held, "regions": ["elsewhere"]}
    assert "not regions of this preview" in drive.absent_support_problem(
        {**plan, "contacts": {**contacts, "support B": gone}}, ABSENT_ROWS, None)
    # bar_kept: a bar of the built frame runs along the declared direction
    body = tmp_path / "blank_bars_x.glb"
    lib.write_json(body.with_suffix(".bars.json"), TWO_PLANE_FRAME)
    bar = {**held, "absent_in_stand_in": {"carried_by": "bar_kept", "direction": [0, 1, 0]}}
    framed = {**plan, "geometry": {"asset_path": str(body)}, "contacts": {**contacts, "support B": bar}}
    assert drive.contacts_problem(framed["contacts"], declared, stand_in=True) is None
    assert drive.absent_support_problem(framed, ABSENT_ROWS, None) is None
    askew = {**bar, "absent_in_stand_in": {**bar["absent_in_stand_in"], "direction": [1, 0, 1]}}
    assert "no bar of the stand-in runs within 15 degrees" in drive.absent_support_problem(
        {**framed, "contacts": {**contacts, "support B": askew}}, ABSENT_ROWS, None)
    assert "no bars record" in drive.absent_support_problem({**framed, "geometry": {"asset_path": "/x/b.glb"}},
                                                            ABSENT_ROWS, None)
    # the report carries it
    lines = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, {"absent_in_stand_in": {
        n: drive.absent_said(d) for n, d in drive.absent_contacts(contacts).items()}})
    assert "contact support B has no part on the stand-in: load_at_joint" in lines
    # the rule is read before the stand-in is built (routes.md, read at the plan stop)
    routes = " ".join((drive.SKILL_DIR / "references" / "objectives" / "routes.md").read_text().split())
    assert "A member left out of the stand-in that carries load to a support" in routes
    assert '"carried_by": "held_at_joint" | "load_at_joint" | "bar_kept"' in routes


def test_a_region_two_contacts_list_is_refused_on_a_new_plan_and_warned_on_a_solved_one():
    declared = {"support A": "force and moment", "support B": "force and moment"}
    shared = {"support A": {"transmits": "force and moment", "regions": ["base"]},
              "support B": {"transmits": "force and moment", "regions": ["base"]}}
    said = drive.contacts_problem(shared, declared, ["base"])
    assert "region base is listed by contacts ['support A', 'support B']" in said
    # kept from a plan the session already solved with: a warning, not a refusal (off a stand-in)
    assert drive.contacts_problem(shared, declared, ["base"], shared_before={"base"}) is None
    assert "kept from this session's earlier plans" in drive.shared_regions_warning(shared, {"base"})
    assert drive.shared_regions_warning(shared, set()) is None
    assert "is listed by contacts" in drive.contacts_problem(shared, declared, ["base"], stand_in=True,
                                                             shared_before={"base"})


# ---------------------------------------------------- a stand-in's self-weight is the real one --

def test_a_heavy_section_stand_in_carries_the_real_self_weight(tmp_path, monkeypatch):
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text(THIN_BRIEF.replace("MEMBER", "0.002"))   # Real mass 30 kg
    planned = {"planned_mass": 600.0}
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: dict(planned))
    # a stand-in built thicker than the real member: the section stand-in, derived from the body
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["built thicker"]})
    plan = {**PLAN, "body_simplification": {"kept": "the outline", "left_out": "the hollow"}}
    said = drive.self_weight_problem(tmp_path, plan)
    want = round(9.80665 * 30 / 600, 5)
    assert "20 x the brief's Real mass 30 kg" in said and f"[0.0, -{want}, 0.0]" in said
    assert "E and the density stay the real material's" in said and "acceleration_use" in said
    scaled = {**plan, "values": {**PLAN["values"], "acceleration": [0, -want, 0]}}
    assert drive.self_weight_problem(tmp_path, scaled) is None
    assert drive.self_weight_scaling(tmp_path, scaled)["scaled"]
    assert not drive.self_weight_scaling(tmp_path, plan)["scaled"]
    # no self-weight, within 2 x the Real mass, or not a section stand-in: nothing asked
    assert drive.self_weight_problem(tmp_path, {**plan, "values": {"load": {"force": [0, -20, 0]}}}) is None
    planned["planned_mass"] = 55.0
    assert drive.self_weight_problem(tmp_path, plan) is None
    planned["planned_mass"] = 600.0
    # any route over 2 x the Real mass carries the real self-weight, a generated body too
    for route in ("generated", "real", "cad"):
        monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": route, "reasons": []})
        assert "the body weighs 600 kg" in drive.self_weight_problem(tmp_path, plan)
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": None, "reasons": []})
    assert drive.self_weight_problem(tmp_path, plan) is None
    # the report says it, and the rule is in routes.md
    lines = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, {"self_weight_scaled": "acceleration 0.49 m/s2"})
    assert "self-weight scaled to the Real mass: acceleration 0.49 m/s2" in lines
    routes = " ".join((drive.SKILL_DIR / "references" / "objectives" / "routes.md").read_text().split())
    assert "takes its self-weight scaled to the Real mass" in routes and "values.acceleration_use" in routes


def test_a_stand_in_s_self_weight_scales_the_use_s_own_acceleration(tmp_path, monkeypatch):
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text(THIN_BRIEF.replace("MEMBER", "0.002"))   # Real mass 30 kg
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: {"planned_mass": 600.0})
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["built thicker"]})
    section = {"route": "section", "about": "x", "load_path": "one section in bending"}
    # a swing of 3 g sideways and down: the scaled acceleration is that x 30 / 600, not g x 30 / 600
    use = [0.0, -3 * 9.80665, 3 * 9.80665]
    want = [round(c * 30 / 600, 5) for c in use]
    plan = {**PLAN, "body_simplification": section, "values": {**PLAN["values"], "acceleration": want,
                                                               "acceleration_use": use}}
    assert drive.self_weight_problem(tmp_path, plan) is None
    scaling = drive.self_weight_scaling(tmp_path, plan)
    assert scaling["scaled"] and abs(scaling["use_m_s2"] - 3 * 9.80665 * 2 ** 0.5) < 1e-6
    # scaled from g instead: refused, with the vector from the use's acceleration
    from_g = {**plan, "values": {**plan["values"], "acceleration": [0, -9.80665 * 30 / 600, 0]}}
    said = drive.self_weight_problem(tmp_path, from_g)
    assert json.dumps(want) in said
    # braking along x: the same rule along the use's direction
    braking = [-6.0, -9.80665, 0.0]
    plan_b = {**plan, "values": {**plan["values"], "acceleration_use": braking,
                                 "acceleration": [round(c * 30 / 600, 5) for c in braking]}}
    assert drive.self_weight_problem(tmp_path, plan_b) is None


# ------------------------------------------------------- one hue, a side painted another colour --

def test_one_hue_missing_from_a_whole_facing_is_warned(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "mark_paint_views", lambda stats, out, compute_env=None: None)
    yellow, grey = (230, 200, 30), (195, 195, 195)
    # the hue on the front and the sides, the back grey: over half the surface is the hue, and the
    # two colours are close in lightness, so neither the overall share nor the split finds it
    back = _painted(tmp_path, "back", lambda v: grey if v[2] < -0.3 else yellow)
    found, stats = drive.paint_findings(back, "yellow painted steel")
    assert stats["hue_share"] > drive.PAINT_HUE_SHARE and stats["facing_split"] < drive.PAINT_FACING_SPLIT
    assert len(found) == 1 and "on its -z facing 0 %" in found[0] and "painted another colour" in found[0]
    assert stats["flagged"].any()
    # the hue all round is no finding; nor a paint asked in two colours
    assert drive.paint_findings(_painted(tmp_path, "all", lambda v: yellow), "yellow painted steel")[0] == []
    assert drive.paint_findings(back, "yellow and grey painted steel")[0] == []
