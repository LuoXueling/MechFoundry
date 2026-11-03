#!/usr/bin/env python
"""Collect, check and draw the evidence a workflow session leaves behind.

An agent cannot open a GLB, and the numbers that decide whether a run is
credible are spread over a dozen files.  This gathers them into one report and
flags what an engineer would question: a support that holds nothing, a load
spread over surfaces facing every way, a strain too small for float32, a solve
that finished on a residual that was growing, a displacement that is not small.

    inspect_session.py summary SID [--yield-strength S]
    inspect_session.py render SID [--what regions|segments|geometry|field|design|all]
                                  [--field stress --component von_mises] [--deform auto|0|<scale>]
    inspect_session.py report SID [--yield-strength S]

``summary`` and ``report`` need only numpy and trimesh (either conda env).
``render`` draws with the workflow's own rasterizer and re-runs itself under
the compute environment's interpreter when matplotlib is missing here.  Every
file it writes goes to ``<session>/operator/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))
import sessionlib as lib  # noqa: E402

# Thresholds an engineer would question.  Each is a prompt to look, not a verdict.
MIN_REGION_FACES = 50
MIN_REGION_WIDTH_H = 3.0
MIN_LOAD_NORMAL_COHERENCE = 0.7
MAX_REGION_AREA_FRACTION = 0.4
SMALL_STRAIN = 1e-2
SMALL_DISPLACEMENT = 0.01            # of the largest extent
SUPPORT_MOTION = 0.05               # of the peak displacement, a held region's mean
SUPPORT_PEAK_MOTION = 0.25          # of the peak displacement, anywhere in a held region
UNIT_LABELS = {
    "SI (m-kg-s)": {"length": "m", "stress": "Pa", "force": "N", "density": "kg/m^3",
                    "accel": "m/s^2", "g": 9.80665},
    "SI-mm (mm-tonne-s)": {"length": "mm", "stress": "MPa", "force": "N",
                           "density": "tonne/mm^3", "accel": "mm/s^2", "g": 9806.65},
    "US-inch (in-lbf-s)": {"length": "in", "stress": "psi", "force": "lbf",
                           "density": "lbf s^2/in^4", "accel": "in/s^2", "g": 386.089},
}


def _norm(vector) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in vector)) if vector else 0.0


def _generate_receipt(session: Path) -> dict[str, Any] | None:
    """The generator's own report (faces kept, components discarded)."""
    newest = None
    for path in (session / "tool_receipts").glob("*.json"):
        receipt = lib.read_json(path, {}) or {}
        if str(receipt.get("tool", "")).startswith("generate_"):
            if newest is None or path.stat().st_mtime > newest[0]:
                newest = (path.stat().st_mtime, receipt)
    if newest is None:
        return None
    receipt = newest[1]
    stdout = [line for line in lib.tail(Path(receipt.get("stdout_path", "")), 50)
              if line.strip().startswith("{")]
    report = {"tool": receipt.get("tool"), "returncode": receipt.get("returncode")}
    for line in reversed(stdout):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        report.update({k: value[k] for k in ("vertices", "faces", "discarded_components")
                       if k in value})
        break
    return report


def segment_names(description: str) -> dict[int, str]:
    """``"0: part a; 1: part b"`` or ``"0: part a. 1: part b."`` -> ``{0: "part a", 1: "part b"}``.

    The vision model separates entries with semicolons or full stops, so each
    name runs from its ``id:`` to the next one.
    """
    text = description or ""
    marks = list(re.finditer(r"(?:^|(?<=[\s;.,]))(\d+)\s*:\s*", text))
    names = {}
    for mark, following in zip(marks, marks[1:] + [None]):
        end = following.start() if following else len(text)
        names[int(mark[1])] = text[mark.end():end].strip(" ;.,\n")
    return names


def _load_mesh(path: Path):
    import trimesh
    return trimesh.load(path, force="mesh", process=False)


def leaf_size(session: Path, workflow: dict[str, Any],
              remesh_depth: int | None = None) -> tuple[float | None, int | None]:
    """The analysis leaf size h, from the scaling transform and the solver depth.

    ``remesh_depth`` is the depth a plan will solve at before its settings card
    is answered (the runner passes it), so a warning about thin members at the
    regions stop is about the depth that will be used.
    """
    transform = lib.read_json(session / "geometry" / "physical.glb.transform.json", {}) or {}
    h_target = transform.get("h_target")
    depth = transform.get("octree_depth") or workflow.get("octree_depth")
    remesh = remesh_depth or (workflow.get("shape_opt") or {}).get("remesh_octree_depth", depth)
    if h_target is None or depth is None:
        return None, remesh
    return float(h_target) * 2 ** (int(depth) - int(remesh)), remesh


def region_rows(resolved: dict[str, Any], proposal: dict[str, Any], h: float | None,
                warnings: list[str]) -> tuple[list[dict[str, Any]], list[float], float]:
    """One row per resolved region, with its value, total force and red flags.

    A load region carries one normal, the average over its faces, and the
    solver loads each point in proportion to max(0, n . n_avg), scaling the
    region's traction so the total is exactly traction x ``loaded_area`` (the
    cosine-weighted area resolve-bcs records), not traction x area.
    """
    values = {item["name"]: item for item in
              (proposal.get("dirichlet") or []) + (proposal.get("neumann") or [])}
    rows = []
    total_force = [0.0, 0.0, 0.0]
    max_traction = 0.0
    for region in resolved.get("regions", []):
        bounds = region.get("bounds") or [[0, 0, 0], [0, 0, 0]]
        size = sorted(float(b - a) for a, b in zip(*bounds))
        coherence = _norm(region.get("mean_normal") or [])
        row = {
            "name": region.get("name"), "category": region.get("category"),
            "faces": region.get("faces"), "area": region.get("area"),
            "area_fraction": round(float(region.get("area_fraction") or 0), 4),
            "centroid": [round(c, 4) for c in region.get("centroid") or []],
            "bbox_size": [round(s, 4) for s in size],
            "normal_coherence": round(coherence, 3),
            "mean_normal": [round(c, 3) for c in region.get("mean_normal") or []],
            "selector": (region.get("selector") or {}).get("type"),
        }
        value = values.get(region.get("name")) or {}
        if region.get("category") == "dirichlet":
            row["components"] = value.get("components")
            row["value"] = value.get("value")
        else:
            traction = value.get("traction")
            row["traction"] = traction
            loaded = float(region["loaded_area"])
            row["loaded_area"] = round(loaded, 6)
            row["loaded_fraction"] = round(loaded / max(float(region["area"]), 1e-30), 3)
            if traction:
                nominal = [float(t) * float(region["area"]) for t in traction]
                applied = [float(t) * loaded for t in traction]
                row["nominal_force"] = [round(f, 6) for f in nominal]
                row["applied_force"] = [round(f, 6) for f in applied]
                total_force = [a + b for a, b in zip(total_force, applied)]
                max_traction = max(max_traction, _norm(traction))
            fraction = row["loaded_fraction"]
            if coherence < MIN_LOAD_NORMAL_COHERENCE or fraction < 0.8:
                warnings.append(
                    f"load {row['name']!r} covers surfaces facing many ways (normal coherence "
                    f"{coherence:.2f}, {fraction:.0%} of its area loaded): the solver loads "
                    "each point by the cosine to the region's average "
                    "normal, so the applied force is smaller than traction x area. Narrow the "
                    "region to one facing (e.g. and + aligned_normal) or split it")
        if (region.get("faces") or 0) < MIN_REGION_FACES:
            warnings.append(f"region {row['name']!r} has only {region.get('faces')} faces")
        if h and size[1] < MIN_REGION_WIDTH_H * h:
            warnings.append(f"region {row['name']!r} is only {size[1] / h:.1f} leaves wide; "
                            "the diffused boundary spreads it about three quarters of a leaf "
                            "beyond its faces, a large share of so small a region")
        if row["area_fraction"] > MAX_REGION_AREA_FRACTION:
            warnings.append(f"region {row['name']!r} covers {row['area_fraction']:.0%} of the "
                            "surface; is that what was meant?")
        rows.append(row)
    if resolved:
        if not any(r["category"] == "dirichlet" for r in rows):
            warnings.append("no support region: the stiffness is singular")
        if any("Overlapping" in (item.get("label") or "") for item in resolved.get("legend", [])):
            warnings.append("support and load regions overlap (purple in the preview)")
    return rows, total_force, max_traction


def summarize(session: Path, yield_strength: float | None = None,
              remesh_depth: int | None = None, solve: bool = True) -> dict[str, Any]:
    workflow = lib.read_json(session / "session.json", {}) or {}
    warnings: list[str] = []
    report: dict[str, Any] = {"session": {
        "id": session.name, "stage": workflow.get("stage"),
        "geometry_mode": workflow.get("geometry_mode"),
        "generator": ("upload" if workflow.get("geometry_mode") == "surface"
                      else (workflow.get("generation") or {}).get("generator")),
        "description": workflow.get("generation_prompt") or workflow.get("geometry_description"),
        "unit_system": workflow.get("unit_system"),
        "api_calls": sum(not r.get("cached") for r in workflow.get("api_receipts") or []),
        "errors": (workflow.get("errors") or [])[-5:],
    }}
    units = UNIT_LABELS.get(workflow.get("unit_system") or "", UNIT_LABELS["SI (m-kg-s)"])
    h, remesh_depth = leaf_size(session, workflow, remesh_depth)

    # ---- geometry
    inspection = lib.read_json(session / "geometry" / "inspection.json", {}) or {}
    transform = lib.read_json(session / "geometry" / "physical.glb.transform.json", {}) or {}
    geometry: dict[str, Any] = {
        "watertight": inspection.get("watertight"), "components": inspection.get("components"),
        "faces": inspection.get("faces"), "generator_report": _generate_receipt(session),
    }
    physical = session / "geometry" / "physical.glb"
    extents = None
    volume = None
    mesh = None
    if physical.exists():
        mesh = _load_mesh(physical)
        extents = [float(v) for v in mesh.extents]
        volume = float(abs(mesh.volume)) if mesh.is_watertight else None
        geometry.update({"physical_extents": [round(e, 6) for e in extents],
                         "physical_volume": volume,
                         "reference_axis": transform.get("axis"),
                         "physical_extent_requested": transform.get("physical_extent")})
    if h:
        geometry["leaf_size_h"] = h
        geometry["analysis_depth"] = remesh_depth
        if extents:
            geometry["leaves_across_smallest_extent"] = round(min(extents) / h, 1)
            geometry["leaves_across_largest_extent"] = round(max(extents) / h, 1)
            if min(extents) / h < 6:
                warnings.append(f"the smallest overall dimension spans only {min(extents) / h:.1f} "
                                "leaves: thin members bend too stiffly or vanish; raise the depth "
                                "or reconsider the part")
    if inspection and not inspection.get("watertight", True):
        warnings.append("surface is not watertight")
    if (inspection.get("components") or 1) > 1:
        # Counted by solid, not by surface: a closed hollow body's void has an inner skin of its
        # own, which one run was warned of at three stops as a loose piece.
        solids = None
        if mesh is not None:
            try:
                solids = lib.solid_pieces(mesh)[1]
            except Exception:       # a count that fails falls back to the surface's
                solids = None
        if solids is not None and solids < inspection["components"]:
            geometry["solids"] = solids
        if (solids or inspection["components"]) > 1:
            warnings.append(f"surface has {solids or inspection['components']} separate solid pieces; loose "
                            "pieces carry no load and a floating piece makes the stiffness singular "
                            "(separate pieces each held by a support of their own solve)")
    discarded = (geometry["generator_report"] or {}).get("discarded_components")
    if discarded:
        geometry["note"] = (f"the generator discarded {discarded} small components; check "
                            "views.png for a missing functional part")
    report["geometry"] = geometry

    # ---- segmentation
    segments_json = lib.read_json(session / "boundary_conditions" / "segments.json", {}) or {}
    if segments_json:
        names = segment_names(workflow.get("segment_description") or "")
        rows = []
        for segment in segments_json.get("segments", []):
            facing = segment.get("facing_fraction_45deg") or {}
            dominant = max(facing, key=facing.get) if facing else None
            rows.append({
                "id": segment["id"], "name": names.get(int(segment["id"])),
                "area_fraction": round(float(segment.get("area_fraction", 0)), 4),
                "faces": segment.get("faces"),
                "centroid": [round(c, 4) for c in segment.get("centroid", [])],
                "dominant_facing": dominant, "cad_surface": segment.get("cad_surface"),
            })
            # A shape-diameter segmentation reports each segment's thickness as
            # a length (shape_diameter) beside the normalized figures
            # (shape_diameter_relative); without a known range only the latter.
            if "shape_diameter_relative" in segment and segment.get("shape_diameter"):
                thickness = float(segment["shape_diameter"]["mean"])
                rows[-1]["thickness"] = round(thickness, 6)
                if h:
                    rows[-1]["thickness_leaves"] = round(thickness / h, 1)
        report["segments"] = {"method": segments_json.get("method"),
                              "num_clusters": segments_json.get("num_clusters"),
                              "count": len(rows), "segments": rows}
        if len(rows) > 20:
            warnings.append(f"{len(rows)} segments: montage colours repeat above 20; identify "
                            "segments by id and bounds, not colour")
        thin = [r for r in rows if r.get("thickness_leaves") is not None
                and r["thickness_leaves"] < 3 and r["area_fraction"] >= 0.01]
        if thin:
            warnings.append(
                "thin members (shape-diameter thickness under 3 leaves): "
                + ", ".join(f"segment {r['id']}{' (' + r['name'] + ')' if r['name'] else ''} "
                            f"{r['thickness']:.3g} = {r['thickness_leaves']} leaves" for r in thin[:5])
                + "; they come out too stiff in bending or vanish -- raise the depth if they carry load")
        tiny = [r["id"] for r in rows if r["area_fraction"] < 0.005]
        if tiny:
            report["segments"]["tiny_segments"] = tiny

    # ---- regions and loads
    resolved = lib.read_json(session / "boundary_conditions" / "resolved.json", {}) or {}
    proposal = workflow.get("bc_proposal") or lib.read_json(
        session / "boundary_conditions" / "proposal.json", {}) or {}
    regions, total_force, max_traction = region_rows(resolved, proposal, h, warnings)
    loads: dict[str, Any] = {"applied_surface_force": [round(f, 6) for f in total_force],
                             "applied_surface_force_magnitude": round(_norm(total_force), 6),
                             "force_unit": units["force"]}
    body = [float(v) for v in workflow.get("body_force") or [0, 0, 0]]
    acceleration = [float(v) for v in workflow.get("acceleration") or [0, 0, 0]]
    material = workflow.get("material") or {}
    density = material.get("density")
    if material.get("mode") in ("parts", "vomp"):
        loads["self_weight"] = ("per-point density in the material field; not totalled here"
                                if any(acceleration) else "no gravity")
    elif volume is not None and (any(body) or any(acceleration)):
        body_total = [b + (density or 0.0) * a for b, a in zip(body, acceleration)]
        loads["body_load_total"] = [round(volume * v, 6) for v in body_total]
    if volume is not None and density and material.get("mode") not in ("parts", "vomp"):
        key = "self_weight" if any(acceleration) else "weight_if_gravity_were_applied"
        loads[key] = round(volume * density * units["g"], 6)
    report["regions"] = regions
    report["loads"] = loads

    # ---- material
    mode = material.get("mode")
    report["material"] = {
        "mode": mode, "youngs_modulus": material.get("youngs_modulus"),
        "poisson_ratio": material.get("poisson_ratio"), "density": density,
        "parts": workflow.get("material_parts") if mode == "parts" else None,
        "stress_unit": units["stress"],
    }
    settings = workflow.get("shape_opt") or {}
    if mode == "uniform" and material.get("youngs_modulus") and max_traction:
        estimate = max_traction / float(material["youngs_modulus"])
        floor = 10 * 1.1920929e-7 / settings.get("relative_tol", 1e-2)
        report["material"]["strain_estimate"] = estimate
        report["material"]["float32_floor"] = floor
        if estimate < floor and not settings.get("use_mixed_precision"):
            warnings.append(f"estimated strain {estimate:.2e} is below the float32 floor "
                            f"{floor:.2e}: use_mixed_precision true")

    # ---- solve (``solve=False``: a caller that reads only the geometry, regions or material, so
    # a regions or geometry stop after a first solve does not pay the 2-7 s solve summary)
    run = lib.newest_run(session) if solve else None
    if run is not None:
        report["solve"] = solve_summary(session, run, extents, max_traction, mode, resolved,
                                        h, yield_strength, warnings)
        case = (lib.read_json(session / "case" / "case.json", {}) or {}).get("shape_opt") or {}
        if report["solve"].get("succeeded") and (settings.get("scale_system_to_unit_diagonal")
                                                 or case.get("scale_system_to_unit_diagonal")):
            # the ladder's last rung (solver_settings.md §2): the report says so, whatever the verdict
            warnings.append("the solve used unit-diagonal scaling (scale_system_to_unit_diagonal, the "
                            "solver ladder's last resort): its ||b - Kx||/||b|| is the scaled system's; "
                            "the result stands on Newton's converged unscaled residual")
    report["warnings"] = warnings
    return report


def run_succeeded(run: Path) -> bool:
    """A run that finished: the finalized marker, its PVD collections, a surface."""
    info = lib.parse_solve_log(run / "log.txt")
    return (bool(info.get("finalized")) and any(run.glob("*.pvd"))
            and any(run.glob("vtu_surface/surface_t*.vtu")))


def run_leaf_size(session: Path, info: dict[str, Any]) -> float | None:
    """The leaf size this run actually used, from its own logged depth."""
    transform = lib.read_json(session / "geometry" / "physical.glb.transform.json", {}) or {}
    depth = (info.get("effective_config") or {}).get("remesh_depth")
    if not transform.get("h_target") or not depth:
        return None
    return float(transform["h_target"]) * 2 ** (int(transform["octree_depth"]) - int(depth))


def solve_summary(session: Path, run: Path, extents, max_traction: float, mode,
                  resolved: dict[str, Any], h, yield_strength, warnings: list[str]) -> dict[str, Any]:
    info = lib.parse_solve_log(run / "log.txt")
    pvd = sorted(p.name for p in run.glob("*.pvd"))
    surfaces = sorted(run.glob("vtu_surface/surface_t*.vtu"))
    succeeded = bool(info.get("finalized")) and bool(pvd) and bool(surfaces)
    h = run_leaf_size(session, info) or h
    solve: dict[str, Any] = {
        "run": run.name, "succeeded": succeeded, "duration_s": info.get("duration_s"),
        "depth": (info.get("effective_config") or {}).get("remesh_depth"), "leaf_size_h": h,
        "newton": lib.newton_summary(info.get("newton") or []),
        "nodes": (info.get("nodes") or [None])[-1], "beta_range": info.get("beta_range"),
        "effective_config": info.get("effective_config"),
        "failures": info.get("failures") if not succeeded else [],
        "saved_surfaces": len(surfaces),
    }
    # The workflow marks a failed solve in solve/failed.json (it may have failed
    # before writing a run folder, leaving an older run the newest one here).
    failure = lib.runtime_files.solve_failure(session)
    if failure:
        solve["latest_attempt_failed"] = {k: failure.get(k) for k in ("error", "log")}
        if (run / "log.txt").stat().st_mtime < float(failure.get("started", 0)) - 1.0:
            succeeded = False
            solve["succeeded"] = False
            warnings.append("the latest solve failed before writing a run folder: "
                            + str(failure.get("error", ""))[:300]
                            + f"; {run.name} is an older run")
    if not succeeded and run_succeeded(run):
        pass
    elif not succeeded:
        warnings.append(f"newest run {run.name} did not finish: "
                        + str((info.get("failures") or [info.get("last_line")])[-1]))
        earlier = [r for r in lib.solve_runs(session)[:-1] if run_succeeded(r)]
        solve["last_successful_run"] = earlier[-1].name if earlier else None
        if earlier and any((session / "solve").glob("*.glb.json")):
            warnings.append(f"the field values below come from {earlier[-1].name}, the last run "
                            "that finished, not from the newest settings")
    fields = {}
    for field in ("displacement_magnitude", "stress_von_mises", "strain_von_mises"):
        candidates = [p for p in (session / "solve").glob(f"{field}*.glb.json")
                      if "_d" not in p.name[len(field):]]
        candidates.sort(key=lambda p: p.stat().st_mtime)
        if not candidates:
            continue
        sidecar = lib.read_json(candidates[-1], {}) or {}
        stale = candidates[-1].stat().st_mtime < (run / "log.txt").stat().st_mtime - 5
        fields[field] = {
            "file": candidates[-1].name,
            "max": sidecar.get("observed_maximum", sidecar.get("maximum")),
            "min": sidecar.get("observed_minimum", sidecar.get("minimum")),
            "stale": stale,
        }
        if stale:
            warnings.append(f"{candidates[-1].name} predates the newest run; it belongs to an "
                            "earlier solve")
    solve["fields"] = fields
    u_max = (fields.get("displacement_magnitude") or {}).get("max")
    s_max = (fields.get("stress_von_mises") or {}).get("max")
    e_max = (fields.get("strain_von_mises") or {}).get("max")
    checks: dict[str, Any] = {}
    if u_max is not None and extents:
        checks["max_displacement"] = u_max
        checks["max_displacement_over_extent"] = u_max / max(extents)
        if u_max / max(extents) > SMALL_DISPLACEMENT:
            warnings.append(f"peak displacement is {u_max / max(extents):.1%} of the part size: "
                            "compare it with the brief's prediction; right for a flexible "
                            "object that predicted it, a units or load slip otherwise")
    if e_max is not None:
        checks["max_von_mises_strain"] = e_max
        if e_max > SMALL_STRAIN:
            warnings.append(f"peak strain {e_max:.2e} exceeds 1e-2: outside linear-elastic "
                            "behaviour for most engineering materials")
    if s_max is not None:
        checks["max_von_mises_stress"] = s_max
        if max_traction:
            checks["stress_concentration_vs_traction"] = s_max / max_traction
        if yield_strength and s_max:
            checks["safety_factor_vs_yield"] = yield_strength / s_max
            if yield_strength / s_max < 1.0:
                warnings.append(f"peak von Mises {s_max:.3g} exceeds the yield strength "
                                f"{yield_strength:.3g}: the part would yield (plasticity is "
                                "not modelled)")
    if mode in ("parts", "vomp"):
        checks["note"] = ("spatial material: surface stress takes each node's own material; "
                          "judge each material by stress_by_material against its own strength")
        try:
            checks["stress_by_material"] = stress_by_material(run)
        except Exception as exc:  # a diagnostic, never a blocker
            checks["stress_by_material_error"] = str(exc)[:200]
    try:
        checks.update(surface_stress_summary(run, session, resolved, h))
    except Exception as exc:  # a diagnostic, never a blocker
        checks["surface_stress_error"] = str(exc)[:200]
    if succeeded:
        try:
            compared = depth_comparison(session, run, resolved, h)
            if compared:
                checks["depths"] = compared
        except Exception as exc:  # a diagnostic, never a blocker
            checks["depths_error"] = str(exc)[:200]
    solve["checks"] = checks
    if info.get("trajectory"):
        solve["optimization"] = optimization_summary(info, run, h)
        warnings.extend(solve["optimization"].pop("warnings", []))
    proposal = lib.read_json(session / "boundary_conditions" / "proposal.json", {}) or {}
    held_components = {item.get("name"): item.get("components")
                       for item in proposal.get("dirichlet") or []}
    try:
        motion = region_motion(session, resolved, h, held_components)
    except Exception as exc:  # a diagnostic, never a blocker
        motion, solve["region_displacement_error"] = None, str(exc)
    resolved_path = session / "boundary_conditions" / "resolved.json"
    if motion and resolved_path.exists() and \
            resolved_path.stat().st_mtime > (run / "log.txt").stat().st_mtime:
        # The regions were previewed again after this solve: its motion belongs to the old
        # regions, and the new regions' held components would mislabel it.
        solve["region_displacement_note"] = "the regions changed since this solve: not compared"
    elif motion:
        solve["region_displacement"] = motion
        warnings.extend(support_motion_warnings(motion, held_components, u_max))

    return solve


def support_motion_warnings(motion: list[dict[str, Any]], held_components: dict[str, Any],
                            u_max: float | None) -> list[str]:
    """A warning per support that moves more than SUPPORT_MOTION of the peak.

    A support held in all three components cannot move by a rigid motion, so
    its motion means the region did not hold in the solve; that is checked in
    the region, not the solver settings.  A partly held one may be moving by
    the rigid motion its released components leave free.
    """
    out = []
    for row in motion:
        if row["category"] != "dirichlet" or not u_max:
            continue
        mean, peak = row.get("held_mean", row["mean"]), row.get("held_max", row["max"])
        if mean <= SUPPORT_MOTION * u_max and peak <= SUPPORT_PEAK_MOTION * u_max:
            continue
        held = held_components.get(row["name"])
        share = (f"{mean / u_max:.0%}" if mean > SUPPORT_MOTION * u_max
                 else f"up to {peak / u_max:.0%} (part of the region)")
        if "held_mean" in row and held is not None and sorted(held) != [0, 1, 2]:
            out.append(
                f"support {row['name']!r} moves {share} of the peak displacement in the "
                f"components it holds {sorted(held)}: the region did not hold in the solve. "
                "Check the region before any solver setting (boundary_conditions.md §2)")
        elif held is not None and sorted(held) == [0, 1, 2]:
            out.append(
                f"support {row['name']!r} is held in all components and still moves {share} of "
                "the peak displacement: the region did not hold in the solve. Check the region "
                "before any solver setting: the solve re-selects a positional region on its own "
                "marching-cubes surface, which can lie a fraction of a leaf outside the previewed "
                "one, so a bound placed exactly at the body's extreme can cut the face off "
                "(boundary_conditions.md §2)")
        else:
            out.append(
                f"support {row['name']!r} moves {share} of the peak displacement on average: a "
                "rigid motion its released components leave free, or a region that did not hold "
                "(objectives/judging.md §5, the solve)")
    return out


def surface_stress_summary(run: Path, session: Path | None = None,
                           resolved: dict[str, Any] | None = None, h: float | None = None) -> dict[str, Any]:
    """Percentiles of the surface von Mises, and the maximum principal stress.

    A percentile describes how much of the surface is lightly stressed, not a
    section stress, so the number to compare with a hand prediction is the
    maximum away from the held and loaded regions (``max_away_from_regions``),
    where their singular peaks are not.  Brittle materials are judged on
    tension, so the largest principal stress (the Voigt tensor's largest
    eigenvalue) is given the same way.
    """
    import numpy as np

    surfaces = sorted(run.glob("vtu_surface/*.vtu"))
    if not surfaces:
        return {}
    data = lib.read_vtu(surfaces[-1], ["stress"])
    voigt = np.asarray(data.get("stress"), dtype=np.float64)
    if voigt.ndim != 2 or voigt.shape[1] != 6 or len(voigt) < 10:
        return {}
    vm = lib.von_mises(voigt)
    reach = exclusion_distance(session, h)
    points_all = np.asarray(data["Points"], dtype=np.float64).reshape(-1, 3)
    away = _away_from_regions(points_all, session, resolved, reach)
    exclusion_note = None
    if h and reach and reach > 3.0 * h and (away is None or away.sum() < 10):
        exclusion_note = (f"no surface lies farther than the member thickness ({reach:.3g}) from "
                          f"the regions; 3 leaves ({3.0 * h:.3g}) used instead")
        reach = 3.0 * h
        away = _away_from_regions(points_all, session, resolved, reach)
    if away is not None and away.sum() < 10:
        exclusion_note = (f"every surface point lies within {reach:.3g} of a held or loaded "
                          "region: no stress away from the regions is measured")
    xx, yy, zz, xy, yz, xz = (voigt[:, i] for i in range(6))
    tensors = np.empty((len(voigt), 3, 3))
    tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 2, 2] = xx, yy, zz
    tensors[:, 0, 1] = tensors[:, 1, 0] = xy
    tensors[:, 1, 2] = tensors[:, 2, 1] = yz
    tensors[:, 0, 2] = tensors[:, 2, 0] = xz
    principal = np.linalg.eigvalsh(tensors)[:, -1]
    pct = lambda a, q: float(np.percentile(a, q))
    out = {"surface_von_mises": {"p50": pct(vm, 50), "p95": pct(vm, 95), "p99": pct(vm, 99),
                                 "max": float(vm.max())},
           "surface_max_principal": {"p95": pct(principal, 95), "p99": pct(principal, 99),
                                     "max": float(principal.max()),
                                     "note": "the largest principal (tensile) stress, for brittle materials"}}
    if exclusion_note:
        out["surface_von_mises"]["exclusion"] = exclusion_note
    if away is not None and away.sum() >= 10:
        # A hand prediction is a section stress at one place; the comparable
        # number is the maximum away from the held and loaded patches, where
        # the clamp's and the load edge's singular peaks are not.
        points = np.asarray(data["Points"], dtype=np.float64).reshape(-1, 3)
        where = int(np.flatnonzero(away)[np.argmax(vm[away])])
        out["surface_von_mises"]["max_away_from_regions"] = float(vm[away].max())
        out["surface_von_mises"]["max_away_at"] = [round(float(c), 5) for c in points[where]]
        out["surface_max_principal"]["max_away_from_regions"] = float(principal[away].max())
        out["surface_von_mises"]["away_distance"] = reach
        out["surface_von_mises"]["note"] = (
            "max_away_from_regions is the largest stress farther than away_distance (the "
            "body's median member thickness, or 3 leaves if more) from any held or loaded "
            "region; compare the brief's prediction with it, at its place (max_away_at). A "
            "value that rises with depth at a corner is a mesh-dependent local peak "
            "(checks.depths)")
    return out


def exclusion_distance(session: Path | None, h: float | None) -> float | None:
    """How far from a held or loaded region its local disturbance reaches.

    By Saint-Venant the stress near a patch settles to the section stress
    within about the member's thickness, a physical distance that does not
    shrink when the mesh is refined; 3 leaves is the floor, the width over
    which a region is blurred.  The median member thickness of the scaled body
    is cached in ``operator/body_thickness.json`` against the surface's mtime.
    """
    if not h:
        return None
    thickness = None
    physical = session / "geometry" / "physical.glb" if session else None
    if physical is not None and physical.exists():
        cache = lib.operator_dir(session) / "body_thickness.json"
        cached = lib.read_json(cache, {}) or {}
        if cached.get("mtime") == physical.stat().st_mtime:
            thickness = cached.get("median")
        else:
            try:
                from drive import member_thickness
                measured = member_thickness(_load_mesh(physical))
                thickness = measured["p50"] if measured else None
            except Exception:
                thickness = None
            lib.write_json(cache, {"mtime": physical.stat().st_mtime, "median": thickness})
    return max(3.0 * float(h), float(thickness or 0.0))


def run_peak_displacement(run: Path) -> float | None:
    """The largest displacement magnitude on a run's solved surface."""
    import numpy as np

    surfaces = sorted(run.glob("vtu_surface/*.vtu"))
    if not surfaces:
        return None
    u = np.asarray(lib.read_vtu(surfaces[-1], ["displacement"]).get("displacement"), dtype=float)
    return float(np.linalg.norm(u, axis=1).max()) if u.ndim == 2 and len(u) else None


def case_digest(session: Path) -> str | None:
    """What makes two solves the same case apart from the mesh: the resolved regions,
    the values (loads and holds) and the material."""
    import hashlib

    parts = [session / "boundary_conditions" / "resolved.npz",
             session / "boundary_conditions" / "proposal.json",
             session / "case" / "loads.json", session / "case" / "material.json"]
    digest = hashlib.sha256()
    found = False
    for path in parts:
        if path.exists():
            digest.update(path.name.encode() + path.read_bytes())
            found = True
    return digest.hexdigest()[:16] if found else None


def record_run_case(session: Path, run: Path) -> dict[str, str]:
    """The case each solve ran, recorded when the newest solve is first summarised.

    The run folder does not keep the case; the session's case files are the
    newest solve's, so the digest is taken for the newest run only, once.
    """
    path = lib.operator_dir(session) / "run_cases.json"
    cases = lib.read_json(path, {}) or {}
    newest = lib.newest_run(session)
    if newest is not None and newest.name == run.name and run.name not in cases:
        digest = case_digest(session)
        if digest:
            cases[run.name] = digest
            lib.write_json(path, cases)
    return cases


def depth_comparison(session: Path, run: Path, resolved: dict[str, Any],
                     h: float | None) -> dict[str, Any] | None:
    """The same case solved at two depths: the change, and its first-order limit.

    The displacement error of the diffused boundary is first order in the leaf
    size (concepts.md, Resolution), so two depths give the limit by Richardson
    extrapolation: u_inf = u_f + (u_f - u_c) / (r - 1), with r the ratio of the
    leaf sizes (2 per depth step).  The stress away from the regions is listed
    per depth: one that settles is the section stress, one that keeps rising is
    a local peak.  Runs are paired only when the same case was solved
    (``record_run_case``: the regions, values and material) and their logged
    settings, all but the depth, match.
    """
    import numpy as np

    def signature(path: Path):
        info = lib.parse_solve_log(path / "log.txt")
        config = dict(info.get("effective_config") or {})
        depth = config.pop("remesh_depth", None)
        text = (path / "log.txt").read_text(encoding="utf-8", errors="replace")
        areas = sorted(line.split("loaded area")[1].split(",")[0].strip()
                       for line in text.splitlines() if "loaded area" in line)
        return depth, (json.dumps(config, sort_keys=True, default=str), tuple(areas))

    depth, key = signature(run)
    if depth is None:
        return None
    cases = record_run_case(session, run)
    other = None
    for earlier in reversed(lib.solve_runs(session)):
        if earlier == run or not run_succeeded(earlier):
            continue
        if cases.get(earlier.name) != cases.get(run.name):
            continue                   # another case (regions, values or material), or unknown
        d, k = signature(earlier)
        if (k[0] == key[0] and (not k[1] or not key[1] or k[1] == key[1])
                and d is not None and int(d) != int(depth)):
            other = (earlier, int(d))
            break
    if other is None:
        return None
    pairs = sorted([(int(depth), run), other[::-1]], key=lambda item: item[0])
    (coarse_depth, coarse), (fine_depth, fine) = pairs
    u_coarse, u_fine = run_peak_displacement(coarse), run_peak_displacement(fine)
    out: dict[str, Any] = {"depths": [coarse_depth, fine_depth], "runs": [coarse.name, fine.name]}
    if u_coarse and u_fine:
        r = 2.0 ** (fine_depth - coarse_depth)
        out["max_displacement"] = [u_coarse, u_fine]
        out["displacement_first_order_limit"] = u_fine + (u_fine - u_coarse) / (r - 1.0)
    stresses = []
    for depth_value, path in ((coarse_depth, coarse), (fine_depth, fine)):
        info = lib.parse_solve_log(path / "log.txt")
        summary = surface_stress_summary(path, session, resolved, run_leaf_size(session, info) or h)
        stresses.append((summary.get("surface_von_mises") or {}).get("max_away_from_regions"))
    out["max_away_from_regions"] = stresses
    out["note"] = ("the same case at two depths: the displacement's first-order limit is a "
                   "measured quantity; a stress away from the regions that keeps rising with "
                   "depth is a local peak, reported as mesh-dependent")
    return out


def _min_distance(points, target, chunk: int = 8192):
    """Each point's distance to the nearest target point, exactly.

    Expanded as |a|^2 + |b|^2 - 2 a.b in float64, so each chunk is one matrix
    product instead of a (chunk, targets, 3) difference array: on a depth-8 solved
    surface (about 100k vertices against 6000 region points) that was 13.8 of the
    16.4 s a solve stop's summary took (one run), and is now well
    under a second.  The clamp at zero absorbs the expansion's rounding.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_sq = (target ** 2).sum(1)
    out = np.empty(len(points))
    for start in range(0, len(points), chunk):
        a = points[start:start + chunk]
        squared = (a ** 2).sum(1)[:, None] + target_sq[None, :] - 2.0 * (a @ target.T)
        out[start:start + chunk] = np.sqrt(np.maximum(squared.min(1), 0.0))
    return out


def _away_from_regions(vertices, session: Path | None, resolved: dict[str, Any] | None,
                       reach: float | None):
    """A mask of the solved surface's vertices farther than ``reach`` from every region."""
    import numpy as np

    if session is None or not resolved or not reach:
        return None
    masks_path = session / "boundary_conditions" / "resolved.npz"
    physical = session / "geometry" / "physical.glb"
    if not masks_path.exists() or not physical.exists():
        return None
    centroids = np.asarray(_load_mesh(physical).triangles_center)
    masks = np.load(masks_path)
    rng = np.random.default_rng(0)
    targets = []
    for region in resolved.get("regions", []):
        key = region.get("key")
        if key in masks.files and len(masks[key]) == len(centroids):
            points = centroids[masks[key]]
            if len(points) > 3000:
                points = points[rng.choice(len(points), 3000, replace=False)]
            targets.append(points)
    if not targets:
        return None
    return _min_distance(vertices, np.concatenate(targets)) > reach


def stress_by_material(run: Path) -> list[dict[str, Any]]:
    """Von Mises in the volume, grouped by Young's modulus (one row per material).

    Each material must be judged against its own strength -- two materials can
    fail at very different stresses -- so the peak over the whole part says
    little.  Read from the volume VTU, which carries each node's E.
    The 99th percentile stands beside the maximum because the maximum sits at a
    singular point (a clamped edge, a re-entrant corner) and grows with the mesh.
    """
    import numpy as np

    volumes = sorted(run.glob("vtu/solution_t*.vtu"))
    if not volumes:
        return []
    data = lib.read_vtu(volumes[-1], ["stress", "youngs_modulus"])
    if "youngs_modulus" not in data or "stress" not in data:
        return []
    stress = lib.von_mises(data["stress"])
    modulus = np.asarray(data["youngs_modulus"], dtype=np.float64)
    # Materials are distinct values in a per-part field; bin a continuous one
    # (VoMP) by decade-thirds so the table stays short.
    keys = np.round(np.log10(np.maximum(modulus, 1e-30)) * 3) / 3
    rows = []
    for key in np.unique(keys):
        mask = keys == key
        if mask.sum() < 20:
            continue
        rows.append({"youngs_modulus": float(np.median(modulus[mask])), "nodes": int(mask.sum()),
                     "von_mises_p99": float(np.percentile(stress[mask], 99)),
                     "von_mises_max": float(stress[mask].max())})
    return rows


def optimization_summary(info: dict[str, Any], run: Path, h: float | None = None) -> dict[str, Any]:
    trajectory = info["trajectory"]
    first, last = trajectory[0], trajectory[-1]
    objective = info.get("objective") or [row["total"] for row in trajectory]
    increases = sum(1 for a, b in zip(objective, objective[1:]) if b > a * (1 + 1e-6))
    summary: dict[str, Any] = {
        "iterations": len(trajectory),
        "se_first": first["se"], "se_last": last["se"],
        "se_raw_first": first["se_raw"], "se_raw_last": last["se_raw"],
        "volume_first": first["vol"], "volume_last": last["vol"], "volume_target": last["target"],
        "volume_last_over_first": last["vol_raw"] / first["vol_raw"] if first["vol_raw"] else None,
        "objective_first": objective[0] if objective else None,
        "objective_last": objective[-1] if objective else None,
        "objective_increases": increases,
        "warnings": [],
    }
    if len(objective) > 1 and objective[-1] >= objective[0]:
        summary["warnings"].append("the objective did not decrease over the optimization")
    if last["target"] and last["vol"] > last["target"] * 1.05 and last["penalty"] > 0:
        summary["warnings"].append(
            f"volume {last['vol']:.3f} is still above the target {last['target']:.3f}: more "
            "iterations or a larger gd_lr")
    if len(trajectory) >= 10 and abs(last["vol"] - first["vol"]) < 0.01 * first["vol"] \
            and abs(last["se"] - first["se"]) < 0.01:
        summary["warnings"].append("the design barely changed: gd_lr is too small for this part size")
    surfaces = sorted((run / "glb").glob("surface_t*.glb"))
    if surfaces:
        try:
            pieces = design_pieces(_load_mesh(surfaces[-1]), h)
            summary["final_design_file"] = str(surfaces[-1])
            summary["final_design_pieces"] = pieces
            detached = [p for p in pieces[1:] if p["kind"] == "detached solid"]
            necks = [p for p in pieces[1:] if p["kind"] == NECK]
            seams = [p for p in pieces[1:] if p["kind"] == SEAM]
            if detached:
                summary["warnings"].append(
                    f"final design has {len(detached)} detached solid piece(s) (volumes "
                    f"{[p['volume'] for p in detached][:4]}): material cut off from the load path")
            if necks:
                whole = latest_whole_design(surfaces[:-1], h)
                if whole:
                    summary["latest_whole_design"] = whole
                summary["warnings"].append(
                    f"final design has {len(necks)} solid piece(s) within one leaf of the main "
                    f"body (gaps {[p['gap_to_main'] for p in necks][:4]}): a neck thinner than "
                    "the resolution, which the diffuse solve still loads but the surface drops. "
                    + (f"{Path(whole).name} is the latest design in one body; " if whole else "")
                    + "quote a design in one body, or lower gd_lr")
            if seams:
                summary["note"] = (
                    f"{len(seams)} open surface fragment(s) near the main body: BC-zip seams or "
                    "marching-cubes slivers in the output surface, not detached material "
                    "(check 'BC zip: ... non-manifold edges' in the log)")
        except Exception as exc:
            summary["final_design_error"] = str(exc)
    return summary


NECK = "solid behind a sub-leaf neck"
SEAM = "open fragment (seam or sliver)"


def latest_whole_design(surfaces: list[Path], h: float | None, look_back: int = 6) -> str | None:
    """The newest of the last few saved surfaces that is one solid body, if any."""
    for surface in reversed(surfaces[-look_back:]):
        pieces = design_pieces(_load_mesh(surface), h)
        if not any(p["kind"] in ("detached solid", NECK) for p in pieces[1:]):
            return str(surface)
    return None


def design_pieces(mesh, h: float | None = None) -> list[dict[str, Any]]:
    """The connected pieces of an optimized surface, each classified.

    A closed piece with volume farther than one leaf ``h`` from the main body
    is material the optimizer cut loose.  One closer than that usually hangs on
    a neck thinner than the resolution: the SDF drops below zero across it, so
    marching cubes splits the surface there, while the diffuse phase field
    (phi = 1/2 at d = 0, not 0) can still carry load through it, and such a
    piece can flicker between split and joined with the volume penalty's on/off
    cycle while the compliance stays on its trend.  An open fragment lying against the main body is
    a stitching seam of the BC zip (or a marching-cubes sliver): the solver's
    design is the volumetric SDF, which this surface only approximates.
    """
    import numpy as np

    mesh.merge_vertices()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices)
    labels = _face_components(faces)
    pieces = []
    for label in np.unique(labels):
        sub = faces[labels == label]
        edges = np.sort(np.concatenate([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        closed = bool((counts == 2).all())
        corners = vertices[sub]
        volume = float(abs(np.einsum("ij,ij->i", corners[:, 0],
                                     np.cross(corners[:, 1], corners[:, 2])).sum()) / 6.0)
        pieces.append({"faces": int(len(sub)), "closed": closed,
                       "volume": round(volume, 9) if closed else None,
                       "bounds": np.round(corners.reshape(-1, 3).min(0), 4).tolist()
                       + np.round(corners.reshape(-1, 3).max(0), 4).tolist(),
                       "_vertices": np.unique(sub)})
    pieces.sort(key=lambda p: -p["faces"])
    main = vertices[pieces[0]["_vertices"]]
    scale = float(np.ptp(main, axis=0).max()) or 1.0
    for piece in pieces:
        own = vertices[piece.pop("_vertices")]
        if piece is pieces[0]:
            piece["kind"] = "main body"
            continue
        sample = own[:: max(1, len(own) // 500)]
        target = main[:: max(1, len(main) // 4000)]
        gap = float(np.sqrt(((sample[:, None] - target[None]) ** 2).sum(-1)).min())
        piece["gap_to_main"] = round(gap, 6)
        if not (piece["closed"] and piece["volume"] and piece["volume"] > 1e-6 * scale ** 3):
            piece["kind"] = SEAM
        else:
            piece["kind"] = NECK if h and gap < h else "detached solid"
    return pieces


def _face_components(faces):
    """A component label per face, by union-find over shared vertices."""
    import numpy as np

    parent = np.arange(int(faces.max()) + 1 if len(faces) else 0)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for a, b, c in faces:
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    return np.fromiter((find(a) for a in faces[:, 0]), dtype=np.int64, count=len(faces))


def region_motion(session: Path, resolved: dict[str, Any], h: float | None,
                  held: dict[str, Any] | None = None):
    """Mean and peak displacement on each BC region, read off the solved surface.

    For a support whose held components are known, ``held_mean`` and
    ``held_max`` are the motion in those components only: a support may move
    freely in the components it releases.

    Supports should barely move and loads usually move most.  The solved
    surface is the solver's own marching-cubes mesh, so each of its vertices is
    matched to the nearest face of each region on the reviewed surface.
    """
    import numpy as np

    field = [p for p in (session / "solve").glob("displacement_magnitude*.glb")
             if "_d" not in p.name[len("displacement_magnitude"):]]
    field.sort(key=lambda p: p.stat().st_mtime)
    masks_path = session / "boundary_conditions" / "resolved.npz"
    physical = session / "geometry" / "physical.glb"
    if not field or not masks_path.exists() or not physical.exists() or not h:
        return None
    disp_path = Path(str(field[-1]) + ".disp.bin")
    if not disp_path.exists():
        return None
    surface = _load_mesh(field[-1])
    displacement = np.frombuffer(disp_path.read_bytes(), dtype="<f4").reshape(-1, 3)
    if len(displacement) != len(surface.vertices):
        return None
    magnitude = np.linalg.norm(displacement, axis=1)
    reviewed = _load_mesh(physical)
    centroids = np.asarray(reviewed.triangles_center)
    masks = np.load(masks_path)
    vertices = np.asarray(surface.vertices)
    rows = []
    rng = np.random.default_rng(0)
    for region in resolved.get("regions", []):
        key = region.get("key")
        if key not in masks.files or len(masks[key]) != len(centroids):
            continue
        points = centroids[masks[key]]
        if len(points) > 3000:
            points = points[rng.choice(len(points), 3000, replace=False)]
        near = _min_distance(vertices, points) < 0.75 * h
        if not near.any():
            continue
        row = {"name": region.get("name"), "category": region.get("category"),
               "vertices": int(near.sum()), "mean": float(magnitude[near].mean()),
               "max": float(magnitude[near].max())}
        comps = (held or {}).get(region.get("name"))
        if region.get("category") == "dirichlet" and comps:
            part = np.linalg.norm(displacement[near][:, sorted(int(c) for c in comps)], axis=1)
            row["held_mean"], row["held_max"] = float(part.mean()), float(part.max())
        rows.append(row)
    return rows


# --------------------------------------------------------------- rendering --

WHAT = ("regions", "segments", "geometry", "field", "material", "design", "all")


def _what(value: str) -> str:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    bad = [p for p in parts if p not in WHAT]
    if not parts or bad:
        raise argparse.ArgumentTypeError(f"--what takes one or more of {', '.join(WHAT)}")
    return ",".join(parts)


def compute_interpreter(compute_env: str) -> str | None:
    """The compute env's interpreter, by the workflow's own lookup (runtime_files, read without
    importing the package, which cost every render subprocess the whole package)."""
    found = lib.runtime_files.direct_interpreter(compute_env)
    return str(found) if found is not None else None


def ramp_legend(minimum: float, maximum: float, unit: str) -> list[tuple[str, tuple]]:
    """Five labelled swatches of the exporter's blue-to-red ramp."""
    colour_stops = [(35, 65, 180), (30, 175, 210), (55, 190, 95), (245, 205, 55), (205, 35, 45)]
    labels = []
    for i, rgb in enumerate(colour_stops):
        value = minimum + (maximum - minimum) * i / (len(colour_stops) - 1)
        labels.append((f"{value:.3g} {unit}".strip(), tuple(c / 255 for c in rgb)))
    return labels


FIELD_PICTURE = (1000, 700)   # a field's six views: the colours and the peak's place read at this size


def shrink(path: Path, size: tuple[int, int]) -> None:
    """Shrink a rendered picture in place to fit ``size`` (aspect kept).  The renderer draws
    1400 x 980; the solve stop's two field pictures need no more than about half those pixels
    (a picture's tokens go as its pixels).  The regions, segments and geometry views keep theirs:
    their legends and small patches are read there."""
    from PIL import Image
    with Image.open(path) as image:
        if image.size[0] <= size[0] and image.size[1] <= size[1]:
            return
        image.load()
        small = image.copy()
    small.thumbnail(size, Image.LANCZOS)
    small.save(path)


def render(session: Path, what: str, field: str, component: str, deform: str) -> list[str]:
    sys.path.insert(0, os.fspath(lib.PACKAGE_DIR / "scripts"))
    import workflow_tool as wt  # the workflow's own renderer and exporter

    wt._prepare_rendering()
    import numpy as np

    out = lib.operator_dir(session)
    workflow = lib.read_json(session / "session.json", {}) or {}
    units = UNIT_LABELS.get(workflow.get("unit_system") or "", UNIT_LABELS["SI (m-kg-s)"])
    written: list[str] = []
    todo = []
    for one in what.split(","):              # several pictures from one process: "regions,segments"
        if one == "all":
            todo += ["regions:if-stale", "field:displacement:magnitude", "field:stress:von_mises", "design"]
        else:
            todo.append(f"field:{field}:{component}" if one == "field" else one)
    for item in todo:
        if item.startswith("regions"):
            glb = session / "boundary_conditions" / "resolved.glb"
            if not glb.exists():
                continue
            if (item == "regions:if-stale" and (out / "regions.png").exists()
                    and (out / "regions.png").stat().st_mtime >= glb.stat().st_mtime):
                written.append(str(out / "regions.png"))   # the regions stop drew these regions
                continue
            resolved = lib.read_json(session / "boundary_conditions" / "resolved.json", {}) or {}
            legend = [(entry["label"], entry["color"]) for entry in resolved.get("legend", [])]
            mesh = wt._mesh(glb)
            target = out / "regions.png"
            wt._render(mesh, target, colors=wt._vertex_face_colors(mesh),
                       title="BC regions (cool = held, warm = loaded)", legend=legend)
            written.append(str(target))
        elif item == "segments":
            glb = session / "boundary_conditions" / "segments.glb"
            if not glb.exists():
                continue
            names = segment_names(workflow.get("segment_description") or "")
            meta = lib.read_json(session / "boundary_conditions" / "segments.json", {}) or {}
            legend = [(f"{s['id']}: {names.get(int(s['id']), '')}".rstrip(": "), s["color"])
                      for s in meta.get("segments", [])]
            mesh = wt._mesh(glb)
            target = out / "segments.png"
            wt._render(mesh, target, colors=wt._vertex_face_colors(mesh),
                       title=f"Segments ({meta.get('method')})", legend=legend)
            written.append(str(target))
        elif item == "geometry":
            surface = Path(workflow.get("surface_path") or session / "geometry" / "physical.glb")
            mesh = wt._mesh(surface)
            target = out / "geometry.png"
            wt._render(mesh, target, colors=wt._vertex_face_colors(mesh), title="Analysis surface")
            written.append(str(target))
        elif item.startswith("field:"):
            _, name, comp = item.split(":")
            from src.agents_wrapper.fem_workflow.assets import newest_surface_vtu
            vtu = newest_surface_vtu(session / "solve")
            if vtu is None:
                continue
            run_name = Path(vtu).parent.parent.name
            asset = out / "fields" / f"{name}_{comp}.glb"
            # The solve step's prefetch exports this same field (newest surface, no deformation,
            # its own range: assets.field_asset_request) into solve/: use it when it is that
            # surface's, instead of exporting it a second time.
            prefetched = session / "solve" / f"{name}_{comp}.glb"
            if (prefetched.exists() and Path(str(prefetched) + ".json").exists()
                    and prefetched.stat().st_mtime >= Path(vtu).stat().st_mtime):
                asset = prefetched
            else:
                wt.export_surface_field(argparse.Namespace(
                    vtu=Path(vtu), output=asset, field=name,
                    component="" if comp == "magnitude" else comp,
                    deform_scale=0.0, value_range_from=""))
            sidecar = lib.read_json(Path(str(asset) + ".json"), {}) or {}
            mesh = wt._mesh(asset)
            colors = wt._vertex_face_colors(mesh)
            scale = 0.0
            disp_path = Path(str(asset) + ".disp.bin")
            if deform != "0" and disp_path.exists():
                displacement = np.frombuffer(disp_path.read_bytes(), dtype="<f4").reshape(-1, 3)
                peak = float(np.linalg.norm(displacement, axis=1).max())
                extent = float(np.ptp(np.asarray(mesh.vertices), axis=0).max())
                scale = ((0.05 * extent / peak if peak > 0 else 0.0) if deform == "auto"
                         else float(deform))
                mesh.vertices = np.asarray(mesh.vertices) + scale * displacement
            unit = {"displacement": units["length"], "stress": units["stress"], "strain": ""}[name]
            legend = ramp_legend(float(sidecar.get("minimum", 0.0)),
                                 float(sidecar.get("maximum", 0.0)), unit)
            title = (f"{name} {comp}" + (f"   (deformation x{scale:.3g})" if scale else "")
                     + f"\n{run_name}")
            target = out / f"{name}_{comp}.png"
            wt._render(mesh, target, colors=colors, title=title, legend=legend)
            shrink(target, FIELD_PICTURE)
            written.append(str(target))
        elif item == "material":
            # Young's modulus of every material field the session holds, as built
            # (log-scaled colours; the sidecar gives the range).
            for label, glb in (("parts", session / "materials" / "parts" / "field" / "youngs_modulus.glb"),
                               ("vomp", session / "materials" / "vomp" / "youngs_modulus.glb")):
                if not glb.exists():
                    continue
                sidecar = lib.read_json(Path(str(glb) + ".json"), {}) or {}
                mesh = wt._mesh(glb)
                low, high = (sidecar.get("minimum"), sidecar.get("maximum"))
                title = f"Young's modulus, {label} field ({units['stress']}, log colour scale)"
                legend = ([(f"2nd pct {low:.3g}", (35 / 255, 65 / 255, 180 / 255)),
                           (f"98th pct {high:.3g}", (205 / 255, 35 / 255, 45 / 255)),
                           (f"data {sidecar.get('data_minimum', 0):.3g}..{sidecar.get('data_maximum', 0):.3g}",
                            (0.8, 0.8, 0.8))]
                          if low is not None and high is not None else None)
                target = out / f"material_{label}.png"
                wt._render(mesh, target, colors=wt._vertex_face_colors(mesh), title=title,
                           legend=legend)
                written.append(str(target))
        elif item == "design":
            run = lib.newest_run(session)
            surfaces = sorted((run / "glb").glob("surface_t*.glb")) if run else []
            if len(surfaces) < 2:
                continue
            for label, surface in (("initial", surfaces[0]), ("final", surfaces[-1])):
                mesh = wt._mesh(surface)
                target = out / f"design_{label}.png"
                wt._render(mesh, target, title=f"Design, {label} ({surface.stem})")
                written.append(str(target))
    return written


def write_report(session: Path, summary: dict[str, Any]) -> Path:
    images = sorted(str(p) for p in (session / "operator").glob("*.png"))
    s = summary["session"]
    g = summary.get("geometry", {})
    lines = [f"# Session {session.name}", "",
             f"- Object: {s.get('description')}",
             f"- Generator: {s.get('generator')} ({s.get('geometry_mode')})",
             f"- Units: {s.get('unit_system')}",
             f"- Stage: {s.get('stage')}; API calls: {s.get('api_calls')}", "",
             "## Geometry",
             f"- Extents: {g.get('physical_extents')}; volume {g.get('physical_volume')}",
             f"- Leaf size h = {g.get('leaf_size_h')} (depth {g.get('analysis_depth')}); "
             f"{g.get('leaves_across_smallest_extent')} leaves across the smallest extent", "",
             "## Boundary conditions",
             "| region | kind | area | faces | normal coherence | value | applied force |",
             "|---|---|---|---|---|---|---|"]
    for r in summary.get("regions", []):
        value = (r.get("traction") if r["category"] == "neumann"
                 else f"components {r.get('components')} = {r.get('value')}")
        lines.append(f"| {r['name']} | {r['category']} | {r.get('area')} | {r.get('faces')} | "
                     f"{r.get('normal_coherence')} | {value} | {r.get('applied_force', '')} |")
    revised = (lib.read_json(session / "boundary_conditions" / "revision.json", {}) or {}).get("regions")
    if revised:
        # Regions whose selectors are not the synthesis's own, for a reader of the result.
        lines += ["", "- Revised after the synthesis (boundary_conditions/revision.json): "
                  + ", ".join(r["name"] for r in revised)]
    lines += ["", f"- Loads: {summary.get('loads')}", "",
              "## Material", f"- {summary.get('material')}", ""]
    if "solve" in summary:
        lines += ["## Solve", "```json", json.dumps(summary["solve"], indent=1, default=str),
                  "```", ""]
    lines += ["## Warnings"] + [f"- {w}" for w in summary.get("warnings", [])] + [""]
    lines += ["## Images"] + [f"- {p}" for p in images] + [""]
    lines += ["## Engineer's judgement", "",
              "_(fill in: is the result credible, what it means, limitations)_", ""]
    target = lib.operator_dir(session) / "report.md"
    target.write_text(lib.relative_paths("\n".join(lines)), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> None:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root.add_argument("--output-root", default=str(lib.output_root_default()))
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("summary", "report"):
        p = sub.add_parser(name)
        p.add_argument("session_id")
        p.add_argument("--yield-strength", type=float,
                       help="in the session's stress unit, for a safety factor")
    p = sub.add_parser("render")
    p.add_argument("session_id")
    p.add_argument("--what", default="all", type=_what,
                   help="one or several, comma-separated, of " + ", ".join(WHAT))
    p.add_argument("--field", default="displacement", choices=("displacement", "stress", "strain"))
    p.add_argument("--component", default=None,
                   help="magnitude for displacement, von_mises for stress and strain unless given")
    p.add_argument("--deform", default="auto", help="auto (5%% of the size), 0, or a factor")
    p.add_argument("--compute-env", default=lib.DEFAULT_COMPUTE_ENV)
    args = root.parse_args(argv)
    session = lib.session_dir(args.session_id, Path(args.output_root))
    if not session.is_dir():
        raise SystemExit(f"no such session: {session}")

    if args.command == "render":
        if args.component is None:
            args.component = "magnitude" if args.field == "displacement" else "von_mises"
        if importlib.util.find_spec("matplotlib") is None:
            interpreter = compute_interpreter(args.compute_env)
            if interpreter is None:
                raise SystemExit(f"matplotlib is missing and no {args.compute_env} "
                                 "interpreter was found")
            os.environ["MPLBACKEND"] = "Agg"
            os.execv(interpreter, [interpreter, os.fspath(Path(__file__).resolve()),
                                   *(argv if argv is not None else sys.argv[1:])])
        sys.path.insert(0, os.fspath(lib.REPO_ROOT))
        written = render(session, args.what, args.field, args.component, args.deform)
        print(json.dumps(lib.relative_paths({"ok": True, "images": written}), indent=1))
        return
    summary = summarize(session, args.yield_strength)
    if args.command == "report":
        path = write_report(session, summary)
        print(json.dumps(lib.relative_paths({"ok": True, "report": str(path), "warnings": summary["warnings"]}),
                         indent=1))
        return
    print(json.dumps(lib.relative_paths(summary), indent=1, default=str))


if __name__ == "__main__":
    # Typed as `python3 -u .../inspect_session.py ...`: into mechfoundry-agents' own interpreter (a
    # render then goes on into the compute environment's).
    lib.runtime_files.run_in_environment("mechfoundry-agents", os.fspath(Path(__file__).resolve()), sys.argv[1:])
    main()
