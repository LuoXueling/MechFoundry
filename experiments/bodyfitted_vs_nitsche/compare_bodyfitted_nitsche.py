#!/usr/bin/env python
"""Diffused Nitsche against body-fitted P1 tetrahedra, per workflow session.

For one finished API-workflow session this solves the same case two ways and
measures one against the other, directly (there is no third, finer reference):

* **Nitsche:** the workflow's own ``workflow_tool.run_forward`` on a copy of the
  session's ``case.json``, float64 (the sessions' precision), diffuse volume,
  at ``remesh_octree_depth`` 7 and 8 (``--depths``).  The code is the current
  dev: saved-segment and solver-resolved BC selectors alike use
  ``core.nitsche_field.bc_match_tolerance`` = max(1.5 eps, 0.75 h_min).
* **Body-fitted P1:** TetGen tetrahedra of the session's surface, solved by the
  production ``core.neoHookean.NeoHookean`` Newton solver on a Warp ``Tetmesh``
  (``TetNeoHookean``), float64, posed as the Nitsche solve poses it:
  - **clamp band:** every node within ``bc_match_tolerance(eps, h_min)`` of a
    Dirichlet region's faces is held, in that region's components (eps and h_min
    of the depth being compared);
  - **Nitsche load definition:** each Neumann region loads t * max(0, n . n_r)
    per unit area with n_r the region's area-weighted mean outward normal, so
    its total force is t times its ``loaded_area``, the area the Nitsche solve
    rescales its band weight to (``SurfaceSelector.loaded_area``);
  for three meshes re-matched at each depth: ``dofs`` (vertex count ~ the
  Nitsche node count), ``surface`` (surface triangles ~ h_min) and ``graded``
  (~h_min at the surface, coarser inside like the octree; TetGen ``-m``).

Metrics per (depth, mode), on the P1 mesh's own quadrature (the P1 field is
exact there; the Nitsche field is the trilinear Q1 interpolant of its octree
cell, so neither is resampled): e = u_nitsche - u_P1,

* headline: rel L2 of e with its best-fit infinitesimal rigid motion removed,
  over ||u_P1||;
* the raw rel L2 and the compliance ratio l(u_N) / l(u_P1), with l the P1 load
  functional (body force and traction at the P1 nodes).

Timing: wall time per phase, GPU-synchronized at every boundary, after an
untimed warm-up of each solver kind in the same process (JIT excluded).  Every
solve runs to a tight common tolerance (``--tight-rtol``); the time to the
session's own tolerance (rel 1e-2) is read from the same solve's Newton log.
Every TetGen attempt, failed ones included, is logged.

Two phases: ``--phase solve`` runs and saves every solve (Nitsche VTUs, P1
solution npz, meshes, TetGen log, timings; no metrics; resumable) and writes
``solve_done``; ``--phase evaluate`` computes every metric from the stored files
only and writes ``comparison.json``.  Usage (env mechfoundry-warp117, idle GPU):

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      python experiments/bodyfitted_vs_nitsche/compare_bodyfitted_nitsche.py \\
      --session <session folder> --output output/bodyfitted_vs_nitsche/<object> --phase solve
"""

import argparse
import base64
import copy
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _main_checkout(root: Path) -> Path:
    """The main checkout: a git worktree has no output/ (it is git-ignored), so
    outputs and the batch inputs live beside the shared .git directory."""
    import subprocess
    try:
        common = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
        if common:
            main = (root / common).resolve().parent
            if (main / "output").is_dir():
                return main
    except Exception:  # noqa: BLE001
        pass
    return root


DATA_ROOT = _main_checkout(REPO_ROOT)
for _path in (REPO_ROOT, REPO_ROOT / "src" / "warp_wrapper"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402
import trimesh  # noqa: E402
import warp as wp  # noqa: E402
import warp.fem as fem  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

# ``core`` must be the same module object ``workflow_tool`` imports, so it is
# imported by that name, not as ``src.warp_wrapper.core``.
from core.neoHookean import NeoHookean  # noqa: E402
from core.nitsche_field import bc_match_tolerance  # noqa: E402
from src.agents_wrapper.fem_workflow.models import (  # noqa: E402
    BCProposal, MaterialParameters, ShapeOptSettings,
)
from src.agents_wrapper.fem_workflow.scripts import workflow_tool  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tetgen_child
import cpu_p1_worker  # noqa: E402  (spawned TetGen child: numpy + tetgen only)

MATCH_TOL = 1e-6
TETGEN_TIMEOUT = 1800.0
DIRECT_TIMEOUT = 1800.0
CPU_PY = "/home/xl2254/implicit_optimize/output/bodyfitted_vs_nitsche/_cpu_env/bin/python"
S_TMP = Path(tempfile.gettempdir())
OOM_MARKERS = ("out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "cudaErrorMemoryAllocation",
               "RESOURCE_EXHAUSTED", "OutOfMemoryError", "AMGX_RC_NO_MEMORY",
               "Failed to allocate")
# A CUDA illegal address is not a memory error here: large meshes go to the CPU
# by node count (CPU_P1_MIN_NODES), and in the ladders an illegal address moves
# on to the next rung, in a fresh process like every rung.
MODES = ("dofs", "surface", "graded")
# Per-part stand-in lattice (copied from experiments/geometry_cleanup/setup.py).
PART_GRID = 160

# Nitsche phases (``nitsche_phase_patches``).  ``run_forward``'s own remainder
# is ``case_parse``; ``output`` (VTU/GLB writes) belongs to neither side.
NITSCHE_PREP_PHASES = ("case_parse", "surface_load", "bc_resolution", "sdf_octree",
                       "outer_setup", "step_overhead", "trim_inner_build", "bc_freeze",
                       "build_case")
NITSCHE_SOLVE_PHASES = ("newton", "inner_rebuild")

TETGEN_LOG = []
_MESH_TAG = ["(untagged)"]


@wp.kernel
def _ray_kernel(mesh: wp.uint64, o: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.vec3),
                out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, o[i], d[i], 1.0e6)
    if q.result:
        out[i] = q.t
    else:
        out[i] = -1.0


@fem.integrand
def _sample_position(s: fem.Sample, domain: fem.Domain):
    return domain(s)


@fem.integrand
def _sample_measure(s: fem.Sample, domain: fem.Domain):
    return fem.measure(domain, s) * s.qp_weight


@fem.integrand
def _sample_normal(s: fem.Sample, domain: fem.Domain):
    return fem.normal(domain, s)


@fem.integrand
def _sample_element(s: fem.Sample, domain: fem.Domain):
    return s.element_index


@fem.integrand
def _sample_value(s: fem.Sample, f: fem.Field):
    return f(s)


@fem.integrand
def _sample_grad(s: fem.Sample, f: fem.Field):
    return fem.grad(f, s)


def _make_side_traction_integrand(vec_type):
    """t_side[side] . v, with the traction looked up per boundary side.

    ``s.element_index`` on a ``BoundarySides`` domain is the geometry's side
    index, the same index ``warp_boundary_sides`` records, so the table built
    there is read back here without any reordering.  The script checks this by
    comparing the assembled total force against the sum of t * area.
    """

    @fem.integrand
    def side_traction(s: fem.Sample, v: fem.Field, t_side: wp.array(dtype=vec_type)):
        return wp.dot(t_side[s.element_index], v(s))

    return side_traction


@wp.kernel
def _closest_face_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    out: wp.array(dtype=int),
):
    tid = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, points[tid], max_dist)
    if q.result:
        out[tid] = q.face
    else:
        out[tid] = -1


@wp.kernel
def _surface_distance_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    p = points[tid]
    q = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    if q.result:
        out[tid] = wp.length(wp.mesh_eval_position(mesh, q.face, q.u, q.v) - p)
    else:
        out[tid] = max_dist


def _sync():
    """Wait for all queued GPU work, so a clock read measures finished work."""
    wp.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PhaseClock:
    """Exclusive wall time per named phase, GPU-synchronized at every boundary.

    Phases nest.  A phase's time excludes its children's, so the totals add up
    to the outermost phase's wall time and nothing is counted twice.  Each
    boundary calls ``wp.synchronize()`` and ``torch.cuda.synchronize()``
    before reading ``perf_counter``; that removes the async overlap of
    consecutive phases, which is what a per-phase split has to give up.
    """

    def __init__(self):
        self.totals = {}
        self.counts = {}
        self._stack = []

    @contextmanager
    def phase(self, name):
        _sync()
        start = time.perf_counter()
        self._stack.append([name, 0.0])
        try:
            yield
        finally:
            _sync()
            duration = time.perf_counter() - start
            _, child = self._stack.pop()
            self.totals[name] = self.totals.get(name, 0.0) + duration - child
            self.counts[name] = self.counts.get(name, 0) + 1
            if self._stack:
                self._stack[-1][1] += duration

    def total(self, names):
        return float(sum(self.totals.get(name, 0.0) for name in names))


@contextmanager
def patched_phases(clock, patches):
    """Time methods/functions of production objects without editing them.

    ``patches`` is ``[(owner, attribute, name_or_callable)]``.  A callable gets
    ``self`` (or the first argument) and returns the phase name, or None to
    leave that call untimed.  Every patch is undone on exit, including those on
    a class that only inherited the attribute (they are deleted again).
    """
    undo = []
    try:
        for owner, attr, namer in patches:
            own = attr in vars(owner)
            original = vars(owner)[attr] if own else getattr(owner, attr)

            def wrapper(*a, __original=original, __namer=namer, **k):
                name = __namer(a[0] if a else None) if callable(__namer) else __namer
                if name is None:
                    return __original(*a, **k)
                with clock.phase(name):
                    return __original(*a, **k)

            setattr(owner, attr, wrapper)
            undo.append((owner, attr, own, original))
        yield
    finally:
        for owner, attr, own, original in reversed(undo):
            if own:
                setattr(owner, attr, original)
            else:
                delattr(owner, attr)


def nitsche_phase_patches():
    """The production calls ``run_forward`` makes, mapped to phase names.

    * ``surface_load``: ``workflow_tool._mesh`` (its in-process cache is cleared
      before the timed run, so the file is parsed).
    * ``bc_resolution``: every ``SurfaceSelector.resolve`` -- the saved-segment
      selectors ``run_forward`` resolves on the source surface and the ones the
      solver resolves on its own surface.
    * ``sdf_octree``: ``ShapeOptTrimmed._init_remeshing_state`` -- surface to
      SDF (BVH queries) and the adaptive octree extraction.
    * ``outer_setup``: the rest of ``ShapeOptTrimmed.__init__`` (the outer,
      untrimmed field and its Nitsche state).
    * ``trim_inner_build``: the inner ``ShapeOpt`` built on the trimmed mesh
      in the first ``step()`` -- exterior trimming, beta, the correction
      domain, the Neumann rescale and the solver cache.
    * ``bc_freeze``: the BC-preservation masks/patches (``_init_bg_frozen_mask``,
      ``_init_bc_graft``), if a forward run builds them.
    * ``newton``: the inner ``step()`` (assembly + linear solves + line
      search), i.e. the Newton solve itself; ``inner_rebuild`` is the inner
      re-initialization a multi-step load ramp does between steps.
    * ``step_overhead``: the remainder of the outer ``step()``.
    * ``output``: ``save``/``finalize``/``_export_displacement_contour``.
    """
    from core.bc_surface_selectors import SurfaceSelector
    from core.shape_opt import ShapeOpt, ShapeOptTrimmed

    def inner_only(name):
        return lambda obj: name if type(obj) is ShapeOpt else None

    return [
        (workflow_tool, "_mesh", "surface_load"),
        (workflow_tool, "_export_displacement_contour", "output"),
        (SurfaceSelector, "resolve", "bc_resolution"),
        (ShapeOptTrimmed, "__init__", "outer_setup"),
        (ShapeOptTrimmed, "_init_remeshing_state", "sdf_octree"),
        (ShapeOptTrimmed, "step", "step_overhead"),
        (ShapeOptTrimmed, "_init_bg_frozen_mask", "bc_freeze"),
        (ShapeOptTrimmed, "_init_bc_graft", "bc_freeze"),
        (ShapeOptTrimmed, "save", "output"),
        (ShapeOptTrimmed, "finalize", "output"),
        (ShapeOpt, "__init__", inner_only("trim_inner_build")),
        (ShapeOpt, "reinit_with_sdf", inner_only("inner_rebuild")),
        (ShapeOpt, "step", inner_only("newton")),
    ]






def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(error) -> bool:
    text = f"{type(error).__name__}: {error}"
    return isinstance(error, torch.cuda.OutOfMemoryError) or any(m in text for m in OOM_MARKERS)


class TetNeoHookean(NeoHookean):
    """The production neo-Hookean solver on a body-fitted Warp ``Tetmesh``.

    Newton, the tangent, the linear solvers and the load ramp are inherited
    unchanged.  Only three things differ from the structured-grid parent:

    * the mesh state is a ``Tetmesh`` (P1 or P2) built from ``params.tet_mesh``;
    * node positions come from the function space, because for P2 the
      geometry's vertex positions are only the first ``V`` of the nodes;
    * the traction is integrated over the boundary sides from a per-side
      table (``params.tet_side_traction``).  The parent's
      PCA + Delaunay nodal lumping assumes a planar region, and a workflow
      region is generally curved.

    The solver writes no VTU of its own.  The comparison script writes one
    VTU holding every field.
    """

    def _build_mesh_state(self):
        mesh = self.params["tet_mesh"]
        mixed = self.use_mixed_precision
        self._tet_nodes_t = torch.as_tensor(
            mesh["nodes"], dtype=torch.float64, device=self.device
        ).contiguous()
        self._tet_elements_t = torch.as_tensor(
            mesh["elements"], dtype=torch.int32, device=self.device
        ).contiguous()
        self._tet_positions_t = (
            self._tet_nodes_t if mixed else self._tet_nodes_t.float().contiguous()
        )
        self.geo = fem.Tetmesh(
            tet_vertex_indices=wp.from_torch(self._tet_elements_t, dtype=wp.int32),
            positions=wp.from_torch(
                self._tet_positions_t, dtype=wp.vec3d if mixed else wp.vec3
            ),
        )
        self._node_positions_cache = None
        self.space = fem.make_polynomial_space(
            geo=self.geo, degree=self.params.elem_order, dtype=self.dtype
        )
        self.u = self.space.make_field()
        self.e_space = fem.make_polynomial_space(
            geo=self.geo, degree=1, dtype=self.design_dtype
        )
        self.e = self.e_space.make_field()
        self._init_e_field()
        self._build_e_fixed_mask()
        if self._has_spatial_material:
            self._init_material_field()

        self.dirichlet_bcs = []
        self.neumann_bcs = []
        self.affine_constraints = []
        self.load_boundary_conditions()

        self.vtu_writer = None
        self._mesh_cache = {
            "points": self._tet_nodes_t,
            "cells": self._tet_elements_t.long(),
        }
        self.residual_history = []
        self.relative_residual_history = []
        self.newton_iterations = 0
        self.initial_residual = None
        self.final_residual = None
        self.final_relative_residual = None
        self.line_search_alphas = []
        self.current_load_scale = 1.0
        self._init_solver_cache()

    def _default_quadrature(self):
        """Order 2 for R and K alike, as the parent pins in float64.

        Order 2 integrates the P2 stiffness grad(v).grad(w) exactly on affine
        tets.  Left to dispatch, a float32 P2 run asks Warp for an order-7 tet
        rule, which Warp does not implement.  Order 4 compiled, but it broke
        both P2 cases: the table's tangent assembly ended in a CUDA
        illegal-address error, and the bar's first Newton step made no progress
        in 25 min.  Both run with order 2.
        """
        return fem.RegularQuadrature(self._mesh_cache["domain"], order=2)

    def _node_positions_torch(self):
        if self._node_positions_cache is None:
            self._node_positions_cache = wp.to_torch(
                self.space.node_positions()
            ).to(dtype=torch.float64)
        return self._node_positions_cache

    def _init_material_field(self):
        """The parent's VoMP/part sampler, fed vertex (not P2 node) positions.

        The material space is degree 1, so it has one value per vertex.  The
        parent samples at ``_node_positions_torch()``, which for P2 is the longer
        node list.
        """
        from core.material_field import sample_material_artifact

        sampled = sample_material_artifact(
            self.params.material_artifact, self._tet_nodes_t
        )
        torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        self.material_space = fem.make_polynomial_space(
            geo=self.geo, degree=1, dtype=self.dtype
        )
        self.material_field = self.material_space.make_field()
        self.material_field.dof_values = wp.from_torch(
            sampled.values.to(dtype=torch_dtype).contiguous(), dtype=self.dtype
        )
        self._material_values_torch = sampled.values
        self._material_manifest = sampled.manifest

    def load_boundary_conditions(self):
        super().load_boundary_conditions()
        if self.params.get("tet_side_traction", None) is not None:
            # A non-empty list is what makes _init_solver_cache assemble the
            # load; the entry itself is never read.
            self.neumann_bcs = [("tet_side_traction", None)]

    def _assemble_neumann_bcs(self):
        scalar = wp.float64 if self.use_mixed_precision else wp.float32
        vec_type = wp.types.vector(length=3, dtype=scalar)
        table = np.asarray(self.params["tet_side_traction"])
        table = table.astype(np.float64 if scalar is wp.float64 else np.float32)
        t_side = wp.array(table, dtype=vec_type, device=self.device)
        mc = self._mesh_cache
        boundary = mc.get("boundary") or fem.BoundarySides(self.geo)
        test = mc.get("bd_test") or fem.make_test(space=self.space, domain=boundary)
        return fem.integrate(
            _make_side_traction_integrand(vec_type),
            fields={"v": test},
            values={"t_side": t_side},
            quadrature=fem.RegularQuadrature(
                boundary, order=2 * int(self.params.elem_order)
            ),
            output_dtype=self.dtype,
        )


def rebase_path(value, session: Path):
    """A path recorded in a session, moved to where the session now lives.

    Batch sessions were run in a slot folder and copied afterwards, so
    ``case.json`` names files under ``.../_slots/<slot>/<session_id>/`` that no
    longer exist.  The part after the session id is re-rooted on the session
    folder given on the command line.
    """
    if not value:
        return value
    path = Path(value)
    if path.exists():
        return str(path)
    parts = path.parts
    if session.name in parts:
        index = len(parts) - 1 - parts[::-1].index(session.name)
        candidate = session.joinpath(*parts[index + 1:])
        if candidate.exists():
            return str(candidate)
    return str(path)


def load_inputs(args):
    """The case JSON and the approved region masks, checked against each other.

    The masks in ``resolved.npz`` are keyed ``<category>_<n>`` in the order
    ``resolve-bcs`` visited the proposal (every Dirichlet entry, then every
    Neumann entry).  Values are chosen later, but the regions are not.  So the
    names and categories in ``resolved.json`` must match ``case.json`` entry for
    entry.  Otherwise the masks are from a different proposal and would load
    the wrong faces.
    """
    if args.case:
        case_path = Path(args.case).resolve()
        session = Path(args.session).resolve() if args.session else case_path.parents[1]
    elif args.session:
        session = Path(args.session).resolve()
        case_path = session / "case" / "case.json"
    else:
        raise SystemExit("give --session DIR or --case case.json")
    regions_npz = (
        Path(args.regions).resolve() if args.regions
        else session / "boundary_conditions" / "resolved.npz"
    )
    regions_json = regions_npz.with_suffix(".json")
    for path in (case_path, regions_npz, regions_json):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    case = json.loads(case_path.read_text(encoding="utf-8"))
    for key in ("surface", "segments_npz"):
        if case.get(key):
            case[key] = rebase_path(case[key], session)
    if case.get("material", {}).get("artifact"):
        case["material"]["artifact"] = rebase_path(case["material"]["artifact"], session)
    if not Path(case["surface"]).is_file():
        raise SystemExit(f"case surface not found: {case['surface']}")
    bcs = BCProposal.model_validate(case["boundary_conditions"])
    if any(item.traction is None for item in bcs.neumann):
        raise SystemExit("every traction must be set in case.json before a solve")
    records = json.loads(regions_json.read_text(encoding="utf-8"))["regions"]
    expected = [("dirichlet", d.name) for d in bcs.dirichlet] + [
        ("neumann", n.name) for n in bcs.neumann
    ]
    found = [(r["category"], r["name"]) for r in records]
    if found != expected:
        raise SystemExit(
            "resolved.json does not describe the case's BCs "
            f"(case: {expected}, resolved: {found}); re-resolve the regions first"
        )
    standin = None
    if case["material"]["mode"] == "parts" and not args.no_standin:
        # Both solves get the same uniform stand-in, written into the in-memory
        # copy of the case only; the session folder is never written.
        mean, parts = part_standin(session, case)
        material = dict(case["material"])
        material.update(mode="uniform", artifact=None, **mean)
        case["material"] = MaterialParameters.model_validate(material).model_dump(mode="json")
        standin = {"material": mean, "parts": parts,
                   "cross_check": standin_cross_check(session, mean)}
    masks = np.load(regions_npz)
    regions = [
        SimpleNamespace(
            key=record["key"], category=record["category"], name=record["name"],
            mask=np.asarray(masks[record["key"]], dtype=bool), bc=item,
            color=record.get("color"),
        )
        for record, item in zip(records, [*bcs.dirichlet, *bcs.neumann])
    ]
    return SimpleNamespace(
        session=session, case_path=case_path, case=case, regions=regions,
        regions_npz=regions_npz,
        material=MaterialParameters.model_validate(case["material"]),
        settings=ShapeOptSettings.model_validate(case["shape_opt"]),
        standin=standin,
    )


def part_standin(session, case):
    """Part-volume-weighted mean (E, nu, rho) of a per-part session's parts.

    Part volumes follow the rule the part field itself is built by
    (``material_parts.field.part_grid``: every lattice cell the surface crosses
    or encloses takes the part of the nearest surface point), on a 160^3 lattice
    instead of the field's 64^3 so thin parts are not over-weighted by the
    surface shell.
    """
    from src.material_parts.field import part_grid

    parts = json.loads((Path(session) / "materials" / "parts" / "materials.json")
                       .read_text(encoding="utf-8"))["parts"]
    table = {int(p["id"]): p for p in parts}
    mesh = trimesh.load(case["surface"], force="mesh", process=False)
    labels = np.load(case["segments_npz"])["segment_ids"]
    coords, cell_labels, _, scale = part_grid(mesh, labels, grid_size=PART_GRID)
    cell = scale / PART_GRID
    ids, counts = np.unique(cell_labels, return_counts=True)
    vol = {int(i): float(n) * cell ** 3 for i, n in zip(ids, counts)}
    missing = sorted(set(vol) - set(table))
    if missing:
        raise ValueError(f"no material for parts {missing}")
    w = np.array([vol[i] for i in vol]) / sum(vol.values())
    mean = {k: float(sum(wi * table[i][k] for wi, i in zip(w, vol)))
            for k in ("youngs_modulus", "poisson_ratio", "density")}
    return mean, [{"id": i, "volume_fraction": vol[i] / sum(vol.values()),
                   **{k: table[i][k] for k in ("youngs_modulus", "poisson_ratio", "density")}}
                  for i in sorted(vol)]


def object_name(session: Path):
    """.../bar-stool-20261013-213800/workflow/<id> -> bar-stool (else the session id)."""
    if session.parent.name == "workflow":
        return re.sub(r"-\d{8}-\d{6}$", "", session.parent.parent.name)
    return session.name


def standin_cross_check(session, mean):
    path = DATA_ROOT / "output" / "geometry_cleanup" / object_name(Path(session)) / "setup.json"
    if not path.is_file():
        return {"setup_json": None}
    other = json.loads(path.read_text(encoding="utf-8")).get("standin_material") or {}
    diff = {k: abs(mean[k] - other[k]) / max(abs(other[k]), 1e-300) for k in mean if k in other}
    return {"setup_json": str(path), "max_rel_diff": max(diff.values()) if diff else None}


def _edge_lengths(vertices, faces):
    return np.linalg.norm(vertices[faces[:, [1, 2, 0]]] - vertices[faces], axis=2)


def clean_surface(path: Path, n_mask_faces: int):
    """The source surface as TetGen needs it, keeping the map back to its faces.

    The surface is loaded the way ``workflow_tool._mesh`` loads it
    (``process=False``), because the region masks index THAT face order.  A GLB
    duplicates vertices along UV/normal seams, so coincident vertices are
    merged by position, and faces that collapse or repeat are dropped.
    ``source_face`` maps every kept face back to the index the masks use.
    """
    raw = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(raw.vertices, dtype=np.float64)
    faces = np.asarray(raw.faces, dtype=np.int64)
    if len(faces) != n_mask_faces:
        raise SystemExit(
            f"{path} has {len(faces)} faces but the region masks have "
            f"{n_mask_faces}; they were resolved on a different surface"
        )
    scale = float(np.ptp(vertices, axis=0).max())
    key = np.round(vertices / (1e-9 * scale)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    vertices, faces = vertices[first], inverse.reshape(-1)[faces]

    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2]) & (area > 1e-14 * scale**2)
    )
    _, unique_rows = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    first_copy = np.zeros(len(faces), dtype=bool)
    first_copy[unique_rows] = True
    source_face = np.flatnonzero(keep & first_copy)
    faces = faces[source_face]
    used, faces = np.unique(faces, return_inverse=True)
    vertices, faces = vertices[used], faces.reshape(-1, 3)

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not (mesh.is_watertight and mesh.is_winding_consistent):
        raise SystemExit(
            f"{path} is not a closed, consistently oriented surface after "
            "merging seam vertices; TetGen cannot mesh its interior"
        )
    tri = vertices[faces]
    face_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return SimpleNamespace(
        path=path, vertices=vertices, faces=faces, source_face=source_face,
        n_source_faces=n_mask_faces, dropped_faces=n_mask_faces - len(faces),
        scale=scale, volume=float(abs(mesh.volume)), mesh=mesh, face_area=face_area,
        median_edge=float(np.median(_edge_lengths(vertices, faces))),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def preserve_surface(surface, h_tet: float, mode: str) -> bool:
    """Whether TetGen may keep every surface triangle as it is (-Y).

    -Y makes every boundary side a meshing-surface face.  But TetGen then
    cannot split a surface triangle much larger than ``h_tet``, so the mesh
    keeps that triangle's coarse resolution along the whole boundary and fills
    in with tets fanned from its vertices.  A bar whose 768 faces were 5 cm
    across (h = 6.6 mm) came out at quality 0.0045 that way, against 0.0135
    without -Y.  ``auto`` keeps -Y only when no surface edge is longer than
    twice ``h_tet``.  Without it, every side still lies inside one source face
    and is mapped back by closest point, so the regions carry over just as
    exactly.
    """
    if mode != "auto":
        return mode == "on"
    return float(_edge_lengths(surface.vertices, surface.faces).max()) <= 2.0 * h_tet


def isotropic_remesh(source, s, iterations, max_surf_dist):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(source.vertices, source.faces))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations, targetlen=pymeshlab.PureValue(s), featuredeg=30.0, checksurfdist=True,
        maxsurfdist=pymeshlab.PureValue(max_surf_dist * s))
    current = ms.current_mesh()
    return current.vertex_matrix(), current.face_matrix()


def surface_at_size(source, s: float, args):
    """The source surface brought to edge length ~s, and how it was done.

    * median edge already in [0.75 s, 1.33 s]: the source itself, untouched;
    * coarser: ``trimesh.remesh.subdivide_to_size`` to edges <= 1.4 s, which
      splits triangles in their own plane, so the geometry is exact;
    * finer: pymeshlab isotropic explicit remeshing to target length s, with
      feature edges (> 30 deg) kept and the vertices reprojected onto the source.
      Coarsening necessarily moves the surface by O(s^2 curvature); the
      distance of the new vertices to the source is reported.
    """
    lo, hi = args.surface_band
    median = source.median_edge
    if lo * s <= median <= hi * s:
        return source, {"method": "source", "median_edge_over_s": median / s}
    if median > hi * s:
        vertices, faces = trimesh.remesh.subdivide_to_size(
            source.vertices, source.faces, max_edge=1.4 * s, max_iter=30
        )
        method = "subdivide"
    else:
        # The remesh rule: our own isotropic remesh can make the surface
        # self-intersect (checked, and confirmed by TetGen -d).  Then it is redone
        # with a tighter surface distance (0.05 s); if it still intersects, the
        # cleaned source surface is meshed as it is, with -Y ("meshed at source
        # resolution").  A source that already intersects is never repaired.
        method = "isotropic_remesh"
        for attempt, max_dist in enumerate((args.remesh_max_surf_dist, 0.05)):
            vertices, faces = isotropic_remesh(source, s, args.remesh_iterations, max_dist)
            if not self_intersects(vertices, faces):
                break
            log_attempt(None, False, 0.0, f"remesh at maxsurfdist {max_dist} s self-intersects",
                        stage="remesh_retry" if attempt == 0 else "remesh_fallback_to_source")
        else:
            return source, {"method": "source", "surface_source": "original", "force_preserve": True,
                            "median_edge_over_s": median / s,
                            "note": "meshed at source resolution: the remeshed surface self-intersects"}
        if attempt:
            method = "isotropic_remesh_maxsurfdist_0.05"
    mesh = trimesh.Trimesh(np.asarray(vertices, np.float64), np.asarray(faces, np.int64),
                           process=True)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        raise RuntimeError(f"{method} at s = {s:.4g} gave an open surface")
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    if mesh.volume < 0:
        mesh.invert()
    out = SimpleNamespace(
        vertices=np.asarray(mesh.vertices, np.float64),
        faces=np.asarray(mesh.faces, np.int64), sha256=source.sha256,
    )
    edges = _edge_lengths(out.vertices, out.faces)
    deviation = surface_distance(source, torch.as_tensor(out.vertices, device="cuda")).cpu().numpy()
    return out, {
        "method": method, "faces": int(len(out.faces)), "vertices": int(len(out.vertices)),
        "median_edge_over_s": float(np.median(edges) / s), "max_edge_over_s": float(edges.max() / s),
        "volume_over_source": float(abs(mesh.volume) / source.volume),
        "vertex_distance_to_source_max_over_s": float(deviation.max() / s),
        "vertex_distance_to_source_mean_over_s": float(deviation.mean() / s),
    }


@contextmanager
def mesh_tag(tag):
    previous = _MESH_TAG[0]
    _MESH_TAG[0] = tag
    try:
        yield
    finally:
        _MESH_TAG[0] = previous


def failure_type(message: str) -> str:
    m = message or ""
    if "crashed (exit -11)" in m:
        return "segfault in the isolated TetGen child"
    if "crashed" in m:
        return "TetGen child crashed"
    if "did not finish" in m:
        return "TetGen timeout"
    if "repair_tet" in m:
        return "TetGen repair_tet error"
    if "self-intersect" in m.lower() or "intersect" in m.lower():
        return "TetGen self-intersection"
    if "locate" in m.lower():
        return "TetGen point location failure"
    if "degenerate" in m:
        return "degenerate tetrahedron"
    if "open surface" in m:
        return "surface remesh gave an open surface"
    if "self-intersects" in m:
        return "surface self-intersects"
    return "other"


def log_attempt(switches, ok, seconds, message=None, stage="tetgen", **extra):
    TETGEN_LOG.append({"mesh": _MESH_TAG[0], "stage": stage, "switches": switches, "ok": bool(ok),
                       "seconds": seconds, "failure_type": None if ok else failure_type(message),
                       "message": None if ok else (message or "")[:400], **extra})




def tetgen_detects_intersections(vertices, faces):
    """TetGen -d (exact predicates) on a surface, in a spawned child: True when it
    reports self-intersections, False when it says the surface is correct, None
    when it says neither."""
    import multiprocessing as mp
    import tempfile

    tmp = tempfile.mkdtemp(prefix="tetgen_d_")
    proc = mp.get_context("spawn").Process(target=tetgen_child.detect_intersections, args=(vertices, faces, tmp))
    proc.start()
    proc.join(TETGEN_TIMEOUT)
    if proc.is_alive():
        proc.kill()
        proc.join()
    try:
        text = Path(tmp, "tetgen_d.txt").read_text(errors="replace")
    except OSError:
        text = ""
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    if "self-intersection" in text or "intersect" in text.lower() and "is correct" not in text:
        return True
    if "is correct" in text:
        return False
    return None


def self_intersects(vertices, faces):
    """pymeshlab per-face self-intersection selection, confirmed by TetGen -d.

    pymeshlab gives floating-point false positives on near-degenerate
    marching-cubes triangles; TetGen -d uses exact predicates and decides.
    Logged as a ``selfint_check`` attempt.
    """
    import pymeshlab

    t0 = time.perf_counter()
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(vertices, np.float64), np.asarray(faces, np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    n_pml = int(ms.current_mesh().selected_face_number())
    verdict = tetgen_detects_intersections(vertices, faces) if n_pml > 0 else False
    hit = bool(verdict)
    log_attempt(None, not hit, time.perf_counter() - t0,
                None if not hit else "surface self-intersects (TetGen -d)", stage="selfint_check",
                pymeshlab_faces=n_pml, tetgen_d=verdict)
    return hit






def _tetgen_isolated(vertices, faces, switches, background):
    """TetGen in a spawned child (``tetgen_child``), so a crash in it is an
    exception, not the end of the job, and no fork of a multithreaded
    (JAX/Warp/Torch) process can deadlock.  A child that does not answer in
    TETGEN_TIMEOUT is killed and recorded as a timeout."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=tetgen_child.tetrahedralize, args=(child, vertices, faces, switches, background))
    proc.start()
    child.close()
    result = None
    try:
        if parent.poll(TETGEN_TIMEOUT):
            result = parent.recv()
    except EOFError:
        result = None
    if result is None and proc.is_alive():
        proc.kill()
        proc.join()
        raise RuntimeError(f"TetGen did not finish in {TETGEN_TIMEOUT} s with switches '{switches}'")
    proc.join()
    if result is None:
        raise RuntimeError(f"TetGen crashed (exit {proc.exitcode}) with switches '{switches}'")
    if result[0] != "ok":
        raise RuntimeError(f"TetGen failed with switches '{switches}': {result[1]}")
    return result[1], result[2], result[3]


def tetrahedralize(surface, h_tet: float, minratio: float, mindihedral: float,
                   preserve: bool, background=None):
    """TetGen with interior tets bounded by h_tet, optionally preserving the surface.

    ``a`` bounds the volume by that of a regular tetrahedron of edge
    ``h_tet``.  With ``-Y`` the boundary size is set by the surface triangles,
    which are kept as they are.

    ``background = (nodes, tets, target_size)`` replaces the uniform bound by a
    sizing field (TetGen ``-m``): the wrapper's ``load_bgmesh_from_arrays``, the
    call ``TetGen._process_bgmesh`` makes from a pyvista grid, fed the arrays
    directly (pyvista is not installed in mechfoundry-warp117).
    """
    try:
        import tetgen  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "TetGen is required: python -m pip install tetgen==0.8.4 "
            "(in mechfoundry-warp117; see README.md)"
        ) from error
    max_volume = h_tet**3 / (6.0 * np.sqrt(2.0))
    bound = "" if background is not None else f"a{max_volume:.6g}"
    switches = (f"pq{minratio:g}/{mindihedral:g}{bound}{'Y' if preserve else ''}"
                f"{'m' if background is not None else ''}Q")
    t0 = time.perf_counter()
    k = tetgen_child.background_scale(background[0]) if background is not None else 1.0
    try:
        nodes, elements, k = _tetgen_isolated(surface.vertices, surface.faces, switches, background)
    except Exception as exc:
        log_attempt(switches, False, time.perf_counter() - t0, str(exc), bg_scale=k)
        raise
    seconds = time.perf_counter() - t0
    log_attempt(switches, True, seconds, bg_scale=k)
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    used, elements = np.unique(elements, return_inverse=True)
    nodes, elements = nodes[used], elements.reshape(-1, 4)
    # Warp integrates with |det J|, but a consistent positive orientation keeps
    # the side normals and the 6V volume checks below meaningful.
    p = nodes[elements]
    six_v = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    flip = six_v < 0
    elements[flip, 1], elements[flip, 2] = elements[flip, 2], elements[flip, 1].copy()
    six_v = np.abs(six_v)
    if six_v.min() <= 0:
        raise RuntimeError("TetGen produced a degenerate tetrahedron")
    return {
        "nodes": nodes, "elements": elements.astype(np.int32),
        "switches": switches, "seconds": seconds, "h_tet": h_tet,
        "surface_sha256": surface.sha256,
    }


def mesh_quality(mesh):
    p = mesh["nodes"][mesh["elements"]]
    edges = np.stack(
        [p[:, a] - p[:, b] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))],
        axis=1,
    )
    length = np.linalg.norm(edges, axis=2)
    volume = np.abs(np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )) / 6.0
    # Normalised volume: 1 for a regular tet, -> 0 for a sliver.
    quality = 6.0 * np.sqrt(2.0) * volume / np.maximum(length.max(axis=1), 1e-300) ** 3
    return {
        "nodes": int(len(mesh["nodes"])), "tets": int(len(mesh["elements"])),
        "edge_min": float(length.min()), "edge_median": float(np.median(length)),
        "edge_max": float(length.max()), "volume": float(volume.sum()),
        "quality_min": float(quality.min()),
        "quality_p01": float(np.quantile(quality, 0.01)),
    }


def boundary_edge_median(mesh):
    """Median edge of the boundary triangles of a tet mesh (its surface size)."""
    faces = np.sort(mesh["elements"][:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3), axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    return float(np.median(_edge_lengths(mesh["nodes"], boundary))), int(len(boundary))


def build_tet_mesh(source, s, args, clock, family: bool):
    """One tet mesh: the reference kind (source surface) or a family member M(s)."""
    if family:
        with clock.phase("surface_remesh"):
            t0 = time.perf_counter()
            try:
                meshing_surface, surface_info = surface_at_size(source, s, args)
            except Exception as exc:
                log_attempt(None, False, time.perf_counter() - t0, str(exc), stage="surface_remesh")
                raise
    else:
        meshing_surface, surface_info = source, {"method": "source"}
    preserve = (True if surface_info.get("force_preserve")
                else preserve_surface(meshing_surface, s, args.preserve_surface))
    with clock.phase("tetgen"):
        mesh = tetrahedralize(meshing_surface, s, args.tet_minratio, args.tet_mindihedral, preserve)
    mesh["surface_info"] = surface_info
    mesh["family"] = bool(family)
    mesh["phase_seconds"] = {k: clock.totals.get(k, 0.0) for k in ("surface_remesh", "tetgen")}
    return mesh


def save_mesh(path: Path, mesh, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, nodes=mesh["nodes"], elements=mesh["elements"])
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_mesh(path: Path, surface_sha256: str):
    meta_path = path.with_suffix(".json")
    if not (path.is_file() and meta_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("surface_sha256") != surface_sha256:
        return None
    data = np.load(path)
    return {"nodes": data["nodes"], "elements": data["elements"], **meta, "reused": str(path)}


def mesh_meta(mesh, h):
    quality = mesh_quality(mesh)
    bmed, bfaces = boundary_edge_median(mesh)
    return {
        "switches": mesh["switches"], "h_tet": mesh["h_tet"], "s_over_h": mesh["h_tet"] / h,
        "surface_sha256": mesh["surface_sha256"], "tetgen_seconds": mesh["seconds"],
        "surface_info": mesh.get("surface_info"), "family": mesh.get("family"),
        "phase_seconds": mesh.get("phase_seconds", {}),
        "boundary_edge_median_over_h": bmed / h, "boundary_faces": bfaces,
        # "nodes"/"tets" would shadow the arrays when meta and mesh are merged
        **{("n_" + k if k in ("nodes", "tets") else k): v for k, v in quality.items()},
    }


def _scalar_types(mixed: bool):
    scalar = wp.float64 if mixed else wp.float32
    return (
        scalar,
        wp.types.vector(length=3, dtype=scalar),
        wp.types.matrix(shape=(3, 3), dtype=scalar),
    )


def warp_boundary_sides(geo, mixed: bool):
    """Index, centroid, outward normal, area and vertices of every boundary side.

    An order-1 rule on a triangle is its centroid, so one sample per side.
    """
    scalar, vec, _ = _scalar_types(mixed)
    boundary = fem.BoundarySides(geo)
    quadrature = fem.RegularQuadrature(boundary, order=1)
    n = quadrature.total_point_count()
    if n != boundary.element_count():
        raise RuntimeError("expected one order-1 sample per boundary side")
    index = wp.zeros(n, dtype=int)
    centroid = wp.zeros(n, dtype=vec)
    normal = wp.zeros(n, dtype=vec)
    area = wp.zeros(n, dtype=scalar)
    fem.interpolate(_sample_element, dest=index, at=quadrature)
    fem.interpolate(_sample_position, dest=centroid, at=quadrature)
    fem.interpolate(_sample_normal, dest=normal, at=quadrature)
    fem.interpolate(_sample_measure, dest=area, at=quadrature)
    index = index.numpy().astype(np.int64)
    return SimpleNamespace(
        index=index,
        centroid=centroid.numpy().astype(np.float64),
        normal=normal.numpy().astype(np.float64),
        area=area.numpy().astype(np.float64),
        vertices=geo.face_vertex_indices.numpy()[index].astype(np.int64),
        table_size=int(geo.side_count()),
    )


def match_sides_to_faces(sides, surface):
    """Source-face index of every boundary side, and how many were not exact.

    Under -Y on the source every boundary side IS a source triangle, so its
    centroid is that triangle's centroid to rounding.  A side TetGen split (no
    -Y) lies inside one source triangle, and its centroid is strictly inside
    it, so the closest face on the surface BVH is that triangle, with no tie.
    A side of a remeshed surface lies within O(s^2 curvature) of the source,
    and the closest source face is where its region membership is read.
    """
    centroids = surface.vertices[surface.faces].mean(axis=1)
    distance, face = cKDTree(centroids).query(sides.centroid)
    # 1e-6 of the part size: far below any triangle, far above float32 rounding.
    exact = distance <= MATCH_TOL * surface.scale
    if not exact.all():
        face[~exact] = closest_faces(surface, sides.centroid[~exact])
    return surface.source_face[face], int((~exact).sum())


def closest_faces(surface, points):
    """Index of the surface face closest to each point (GPU BVH query)."""
    mesh = wp.Mesh(
        points=wp.array(surface.vertices.astype(np.float32), dtype=wp.vec3),
        indices=wp.array(surface.faces.astype(np.int32).ravel(), dtype=wp.int32),
    )
    out = wp.zeros(len(points), dtype=int)
    wp.launch(
        _closest_face_kernel, dim=len(points),
        inputs=[mesh.id, wp.array(points.astype(np.float32), dtype=wp.vec3),
                float(2.0 * surface.scale), out],
    )
    face = out.numpy()
    if (face < 0).any():
        raise RuntimeError("a boundary side found no surface face")
    return face


def side_traction_table(inputs, sides, side_source_face, weighting):
    """Per-side traction, indexed like ``s.element_index``, and per-region totals.

    Where two Neumann regions overlap, their tractions are averaged, as
    ``nitsche_field`` averages ``t_bar`` over overlapping predicates.
    """
    total = np.zeros((sides.table_size, 3))
    count = np.zeros(sides.table_size)
    report = []
    for region in inputs.regions:
        if region.category != "neumann":
            continue
        on = region.mask[side_source_face]
        if not on.any():
            raise RuntimeError(f"Neumann region {region.name!r} covers no boundary side")
        n_avg = (sides.normal[on] * sides.area[on, None]).sum(axis=0)
        n_avg /= max(float(np.linalg.norm(n_avg)), 1e-300)
        weight = (
            np.clip(sides.normal[on] @ n_avg, 0.0, None)
            if weighting == "nitsche" else np.ones(int(on.sum()))
        )
        traction = np.asarray(region.bc.traction, dtype=np.float64)
        np.add.at(total, sides.index[on], weight[:, None] * traction)
        np.add.at(count, sides.index[on], 1.0)
        loaded_area = float((sides.area[on] * weight).sum())
        report.append({
            "name": region.name, "sides": int(on.sum()),
            "area": float(sides.area[on].sum()), "loaded_area": loaded_area,
            "mean_normal": n_avg.tolist(), "traction": traction.tolist(),
            "force": (traction * loaded_area).tolist(),
        })
    table = total / np.maximum(count, 1.0)[:, None]
    return table, report


def dirichlet_node_masks(inputs, sides, side_source_face, node_positions, degree, scale):
    """Space nodes on each Dirichlet region's boundary sides.

    Vertex nodes come straight from the sides' vertex indices.  For P2 the
    midside nodes are found by position, which does not depend on Warp's
    internal edge numbering.
    """
    n_nodes = len(node_positions)
    tree = cKDTree(node_positions) if degree > 1 else None
    masks = []
    for region in inputs.regions:
        if region.category != "dirichlet":
            continue
        on = region.mask[side_source_face]
        if not on.any():
            raise RuntimeError(f"Dirichlet region {region.name!r} covers no boundary side")
        mask = np.zeros(n_nodes, dtype=bool)
        corner = sides.vertices[on]
        mask[np.unique(corner)] = True  # P1/P2 vertex nodes are numbered as vertices
        if tree is not None:
            p = node_positions[corner]
            mid = np.concatenate(
                [0.5 * (p[:, 0] + p[:, 1]), 0.5 * (p[:, 1] + p[:, 2]), 0.5 * (p[:, 0] + p[:, 2])]
            )
            distance, node = tree.query(mid, distance_upper_bound=MATCH_TOL * scale)
            if not np.isfinite(distance).all():
                raise RuntimeError("a P2 midside node was not found by position")
            mask[node] = True
        masks.append((region, mask))
    return masks


def encode_dirichlet(item):
    """The Dirichlet value spec exactly as ``workflow_tool.run_forward`` builds it."""
    value = item.value
    if item.components is None:
        return value
    if isinstance(value, tuple):
        if sorted(item.components) == [0, 1, 2]:
            return value
        selected = [value[index] for index in item.components]
        if len(set(selected)) != 1:
            raise SystemExit("partial Dirichlet components require one shared scalar value")
        return (item.components, selected[0])
    return (item.components, value)


def bf_boundary_data(inputs, source, mesh, order, weighting, clock, grow=0.0):
    """Boundary-side -> source-face map and the BCs carried onto one tet mesh.

    ``grow > 0`` (a diagnostic, ``--bf-dirichlet-grow``) also clamps every node
    within that distance of a Dirichlet patch, inside the body: the clamp a
    diffused Nitsche band applies (``NitscheNeoHookean._bc_match_tolerance``,
    max(1.5 eps, 0.75 h)).
    """
    device = "cuda"
    with clock.phase("boundary_mapping"):
        geo = fem.Tetmesh(
            tet_vertex_indices=wp.array(mesh["elements"], dtype=wp.int32, device=device),
            positions=wp.array(mesh["nodes"], dtype=wp.vec3d, device=device),
        )
        sides = warp_boundary_sides(geo, True)
        side_source_face, inexact = match_sides_to_faces(sides, source)
    with clock.phase("bc_transfer"):
        space = fem.make_polynomial_space(geo=geo, degree=order, dtype=wp.vec3d)
        node_positions = space.node_positions().numpy().astype(np.float64)
        if not np.allclose(node_positions[: len(mesh["nodes"])], mesh["nodes"],
                           rtol=0.0, atol=MATCH_TOL * source.scale):
            raise RuntimeError("the first space nodes are not the mesh vertices")
        side_table, neumann_report = side_traction_table(inputs, sides, side_source_face, weighting)
        dirichlet_masks = dirichlet_node_masks(
            inputs, sides, side_source_face, node_positions, order, source.scale
        )
        if grow:
            # grow: one radius for every region, or {region name: radius}
            grown = []
            x = torch.as_tensor(node_positions, device="cuda")
            for region, mask in dirichlet_masks:
                radius = grow.get(region.name, 0.0) if isinstance(grow, dict) else grow
                if radius <= 0:
                    grown.append((region, mask))
                    continue
                faces = source.faces[region.mask[source.source_face]]
                patch = SimpleNamespace(vertices=source.vertices, faces=faces, scale=source.scale)
                near = (surface_distance(patch, x) <= radius).cpu().numpy()
                grown.append((region, mask | near))
            dirichlet_masks = grown
    # How well each region carried over: its area on this mesh against the
    # source faces' area (1 exactly for -Y on the source surface).
    region_area = {}
    for region in inputs.regions:
        on = region.mask[side_source_face]
        src = float(source.face_area[region.mask[source.source_face]].sum())
        region_area[f"{region.category}:{region.name}"] = (
            float(sides.area[on].sum()) / src if src > 0 else None
        )
    del geo, space
    return SimpleNamespace(
        sides=sides, side_source_face=side_source_face, inexact=inexact,
        side_table=side_table, neumann_report=neumann_report,
        dirichlet_masks=dirichlet_masks, node_positions=node_positions,
        region_area_ratio=region_area,
    )


def material_params(inputs):
    """E, nu, body force and material artifact, as ``run_forward`` sets them."""
    material, case = inputs.material, inputs.case
    loads = case.get("loads") or {
        "body_force": material.body_force, "acceleration": (0.0, 0.0, 0.0)
    }
    body = tuple(float(v) for v in loads.get("body_force", (0.0, 0.0, 0.0)))
    acceleration = tuple(float(v) for v in loads.get("acceleration", (0.0, 0.0, 0.0)))
    if material.mode == "uniform":
        density = float(material.density or 0.0)
        return {
            "E": material.youngs_modulus, "nu": material.poisson_ratio,
            "B": tuple(body[i] + density * acceleration[i] for i in range(3)),
        }
    grid = np.load(Path(material.artifact) / "material_grid.npy", mmap_mode="r")
    return {
        "E": float(np.median(grid[..., 0])), "nu": float(np.median(grid[..., 1])),
        "B": body, "material_artifact": material.artifact,
        "material_acceleration": acceleration,
    }


def build_case_replay(inputs, out_dir: Path):
    """Time the workflow's ``build-case`` on this session's own inputs.

    It is the last preparation step before ``run-forward`` (it writes
    ``case.json``).  The replay writes into the output folder, never the session.
    """
    session = inputs.session
    try:
        transform = Path(rebase_path(str(session / "geometry" / "physical.glb.transform.json"), session))
        args = SimpleNamespace(
            surface=Path(inputs.case["surface"]),
            bc_json=session / "boundary_conditions" / "proposal.json",
            segments_npz=Path(inputs.case["segments_npz"]) if inputs.case.get("segments_npz") else None,
            material_json=session / "case" / "material.json",
            loads_json=session / "case" / "loads.json",
            settings_json=session / "case" / "shape_opt.json",
            transform_json=transform,
            output=out_dir / "replay" / "case.json",
            solve_output=out_dir / "replay" / "solve",
            octree_depth=int(inputs.case["octree_depth"]),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        workflow_tool.build_case(args)
        return {"seconds": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001 - timing only
        return {"seconds": 0.0, "error": f"{type(exc).__name__}: {exc}"}


def newton_trace(log_path, solve_seconds, tol):
    """Per-iteration (seconds since solve start, rel) from a solver log, and the
    time the solve took to first reach ``rel <= tol``.

    The log stamps each Newton row when its residual is known.  The solve's own
    wall time (from the phase clock) anchors the last row, so the time to the
    tolerance is ``solve_seconds - (t_last - t_k)``.  Used to compare solve times
    at one tolerance: the body-fitted solves run to rel 1e-8, the float32
    Nitsche solve stops at the case's rel 1e-2.
    """
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rows = re.findall(r"^\[(\d+)-(\d+)-(\d+),(\d+):(\d+):([\d.]+)\].*Iter (\d+): \|\|R\|\| = "
                      r"([0-9.eE+-]+), rel = ([0-9.eE+-]+)", text, re.M)
    if not rows or solve_seconds is None:
        return None
    stamp = [int(r[3]) * 3600 + int(r[4]) * 60 + float(r[5]) for r in rows]
    stamp = [s + 86400 if s < stamp[0] else s for s in stamp]
    rel = [float(r[8]) for r in rows]
    hit = next((k for k, r in enumerate(rel) if r <= tol and int(rows[k][6]) > 0), None)
    to_tol = None if hit is None else max(0.0, solve_seconds - (stamp[-1] - stamp[hit]))
    return {"tol": tol, "seconds_to_tol": to_tol,
            "iterations_to_tol": None if hit is None else int(rows[hit][6]),
            "rel": rel, "seconds": [s - stamp[0] for s in stamp]}


def parse_nitsche_log(run_dir: Path):
    text = (run_dir / "log.txt").read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"Iter (\d+): \|\|R\|\| = ([0-9.eE+-]+), rel = ([0-9.eE+-]+)", text)
    config = re.search(r"Workflow effective config: (.*)", text)
    inner = re.findall(r"inner (\d+) nodes", text)
    result = {"log": str(run_dir / "log.txt"), "newton_rows": len(rows)}
    if inner:
        result["inner_nodes"] = int(inner[-1])
        result["dofs"] = 3 * int(inner[-1])
    if rows:
        result.update(
            newton_iterations=sum(1 for it, _, _ in rows if int(it) > 0),
            final_residual=float(rows[-1][1]), final_relative_residual=float(rows[-1][2]),
        )
    if config:
        result["effective_config"] = config.group(1)
        mode = re.search(r"run_mode=(\w+)", config.group(1))
        result["run_mode"] = mode.group(1) if mode else None
    return result


def check_load(solver, bc):
    """The assembled traction against sum(t * area): a side-indexing check."""
    load = (wp.to_torch(solver._mesh_cache["neumann"]).to(torch.float64)
            if solver._mesh_cache.get("neumann") is not None else None)
    expected = (bc.side_table[bc.sides.index] * bc.sides.area[:, None]).sum(axis=0)
    assembled = load.sum(dim=0).cpu().numpy() if load is not None else np.zeros(3)
    error = float(np.linalg.norm(assembled - expected) / max(np.linalg.norm(expected), 1e-300))
    # A side-indexing error shows up as an O(1) mismatch.  Rounding over tens of
    # thousands of split sides reached 1e-8 on the bar check (7.6e-14 on the table).
    if np.linalg.norm(expected) > 0 and error > 1e-6:
        raise RuntimeError(f"assembled traction force is off by {error:.2e}")
    return load, assembled, error


def external_load_vector(solver, load):
    body = solver._mesh_cache.get("body")
    f_ext = torch.zeros(solver.space.node_count(), 3, dtype=torch.float64, device="cuda")
    if load is not None:
        f_ext += load
    if body is not None:
        f_ext += wp.to_torch(body).to(torch.float64)
    return f_ext


def read_vtu(path: Path):
    """Arrays of a solver VTU: inline base64 of a UInt64 byte count + raw values."""
    dtypes = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8",
              "UInt8": "u1", "UInt32": "<u4", "UInt64": "<u8"}
    text = path.read_text(encoding="latin-1")
    arrays = {}
    for match in re.finditer(r"<DataArray([^>]*)>\s*([^<]*)</DataArray>", text):
        attrs, payload = match.group(1), match.group(2).strip()
        name = re.search(r'Name="([^"]+)"', attrs).group(1)
        kind = re.search(r'type="([^"]+)"', attrs).group(1)
        comps = re.search(r'NumberOfComponents="(\d+)"', attrs)
        raw = base64.b64decode(payload)
        size = int(np.frombuffer(raw[:8], dtype="<u8")[0])
        values = np.frombuffer(raw[8:8 + size], dtype=dtypes[kind])
        if comps and int(comps.group(1)) > 1:
            values = values.reshape(-1, int(comps.group(1)))
        arrays.setdefault(name, values)  # the writer can repeat a name; keep the first
    return arrays


class OctreeQ1Field:
    """The Nitsche solution as the solver holds it: Q1 on axis-aligned octree boxes.

    Every retained hex is a box on the octree lattice, so a dense voxel grid at
    the finest cell size maps each point to its cell with one lookup.  Inside
    the cell, a corner's trilinear weight is read from the SIGN of its offset
    about the cell centre, not from an assumed corner order, as in
    ``illustrations/nitscheconv/nc_prep.hex_eval``.  Hanging nodes change the
    corner values, not the field inside a cell.

    A point that falls in no retained cell (a feature thinner than the trimmed
    octree resolves) is evaluated in its nearest cell with clamped local
    coordinates.  Those points are counted and reported as ``uncovered``.
    """

    def __init__(self, vtu: dict, device):
        self.device = device
        points = np.asarray(vtu["Points"], dtype=np.float64)
        cells = np.asarray(vtu["connectivity"], dtype=np.int64).reshape(-1, 8)
        if not np.all(np.asarray(vtu["types"]) == 12):
            raise RuntimeError("Nitsche VTU is not all hexahedra")
        corner = points[cells]
        lo, hi = corner.min(axis=1), corner.max(axis=1)
        h = float((hi - lo).min())
        origin = points.min(axis=0)
        start = np.round((lo - origin) / h).astype(np.int64)
        size = np.round((hi - lo) / h).astype(np.int64)
        if np.abs((lo - origin) / h - start).max() > 1e-2:
            raise RuntimeError("Nitsche hexes are not on one lattice")
        dims = (start + size).max(axis=0)
        grid = np.full(tuple(dims), -1, dtype=np.int64)
        for shape in np.unique(size, axis=0):
            members = np.flatnonzero((size == shape).all(axis=1))
            offsets = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
            for off in offsets:
                ijk = start[members] + off
                grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = members
        centre = 0.5 * (lo + hi)
        t = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        self.h, self.origin_np = h, origin
        self.origin, self.dims = t(origin), t(dims, torch.int64)
        self.grid = t(grid, torch.int64)
        self.lo, self.span = t(lo), t(hi - lo)
        self.cells = t(cells, torch.int64)
        self.sign = t(corner > centre[:, None, :], torch.bool)  # (M, 8, 3)
        self.u = t(np.asarray(vtu["displacement"], dtype=np.float64))
        self.phi = t(np.asarray(vtu["phi"], dtype=np.float64))
        self._centres = centre
        self._tree = None
        self.n_cells = len(cells)
        self.n_points = len(points)
        # Volume of the Nitsche body: the integral of the trilinear phi, which
        # is each cell's volume times the mean of its corner values.
        phi_np = np.asarray(vtu["phi"], dtype=np.float64)
        self.phi_volume = float((np.prod(hi - lo, axis=1) * phi_np[cells].mean(axis=1)).sum())
        u_np = np.asarray(vtu["displacement"], dtype=np.float64)
        self.max_u_body = float(np.linalg.norm(u_np[phi_np >= 0.5], axis=1).max())

    def _locate(self, x):
        ijk = torch.floor((x - self.origin) / self.h).long()
        inside = ((ijk >= 0) & (ijk < self.dims)).all(dim=1)
        ijk = torch.minimum(torch.clamp(ijk, min=0), self.dims - 1)
        cell = self.grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        cell = torch.where(inside, cell, torch.full_like(cell, -1))
        miss = cell < 0
        if miss.any():
            if self._tree is None:
                self._tree = cKDTree(self._centres)
            _, nearest = self._tree.query(x[miss].cpu().numpy())
            cell[miss] = torch.as_tensor(nearest, dtype=torch.int64, device=self.device)
        return cell, miss

    def evaluate(self, x, chunk=1 << 21):
        """u, grad u (component, direction), and the uncovered flag at x (n, 3)."""
        out_u, out_g, out_miss = [], [], []
        for begin in range(0, len(x), chunk):
            xc = x[begin:begin + chunk]
            cell, miss = self._locate(xc)
            lo, span = self.lo[cell], self.span[cell]
            xi = ((xc - lo) / span).clamp(0.0, 1.0)  # (n, 3)
            sign = self.sign[cell]  # (n, 8, 3)
            factor = torch.where(sign, xi[:, None, :], 1.0 - xi[:, None, :])  # (n, 8, 3)
            dfactor = (2.0 * sign.to(torch.float64) - 1.0) / span[:, None, :]  # (n, 8, 3)
            weight = factor.prod(dim=2)  # (n, 8)
            dweight = torch.stack(
                [
                    dfactor[..., 0] * factor[..., 1] * factor[..., 2],
                    factor[..., 0] * dfactor[..., 1] * factor[..., 2],
                    factor[..., 0] * factor[..., 1] * dfactor[..., 2],
                ],
                dim=2,
            )  # (n, 8, 3)
            corner = self.cells[cell]  # (n, 8)
            uc = self.u[corner]  # (n, 8, 3)
            out_u.append(torch.einsum("nk,nkc->nc", weight, uc))
            out_g.append(torch.einsum("nkd,nkc->ncd", dweight, uc))
            out_miss.append(miss)
        return torch.cat(out_u), torch.cat(out_g), torch.cat(out_miss)

    def evaluate_phi(self, x, chunk=1 << 21):
        out = []
        for begin in range(0, len(x), chunk):
            xc = x[begin:begin + chunk]
            cell, _ = self._locate(xc)
            xi = ((xc - self.lo[cell]) / self.span[cell]).clamp(0.0, 1.0)
            factor = torch.where(self.sign[cell], xi[:, None, :], 1.0 - xi[:, None, :])
            out.append(torch.einsum("nk,nk->n", factor.prod(dim=2), self.phi[self.cells[cell]]))
        return torch.cat(out)


def sample_cells(solver, order: int):
    """Positions, weights, element, u and grad u at the tet quadrature points."""
    mixed = solver.use_mixed_precision
    scalar, vec, mat = _scalar_types(mixed)
    quadrature = fem.RegularQuadrature(fem.Cells(solver.geo), order=order)
    n = quadrature.total_point_count()
    arrays = {
        "x": (wp.zeros(n, dtype=vec), _sample_position, {}),
        "w": (wp.zeros(n, dtype=scalar), _sample_measure, {}),
        "cell": (wp.zeros(n, dtype=int), _sample_element, {}),
        "u": (wp.zeros(n, dtype=vec), _sample_value, {"f": solver.u}),
        "grad": (wp.zeros(n, dtype=mat), _sample_grad, {"f": solver.u}),
    }
    if solver._has_spatial_material:
        arrays["material"] = (wp.zeros(n, dtype=vec), _sample_value, {"f": solver.material_field})
    result = {}
    for key, (dest, integrand, fields) in arrays.items():
        fem.interpolate(integrand, dest=dest, at=quadrature, fields=fields)
        tensor = wp.to_torch(dest)
        result[key] = tensor.long() if key == "cell" else tensor.to(torch.float64)
    return result


def surface_distance(surface, x):
    """Unsigned distance from each point to the true (source) surface, on the GPU."""
    mesh = wp.Mesh(
        points=wp.array(surface.vertices.astype(np.float32), dtype=wp.vec3),
        indices=wp.array(surface.faces.astype(np.int32).ravel(), dtype=wp.int32),
    )
    points = wp.from_torch(x.float().contiguous(), dtype=wp.vec3)
    out = wp.zeros(len(x), dtype=float)
    wp.launch(
        _surface_distance_kernel, dim=len(x),
        inputs=[mesh.id, points, float(2.0 * surface.scale), out],
    )
    return wp.to_torch(out).to(torch.float64)


def secant_size_search(build, x0, target, tol, max_iter, lo, hi):
    """Find x whose mesh has ~target vertices; vertices fall as x grows.

    ``build(x)`` returns (vertices, payload) or raises.  Secant on log(vertices)
    vs log(x) through the last two good trials (slope -2.5 until there are
    two, or when the pair is degenerate), clipped to [lo, hi].  Returns the
    best (x, vertices, payload) and every trial.
    """
    trials, best, x = [], None, x0
    for _ in range(max_iter):
        x = float(np.clip(x, lo, hi))
        try:
            n, payload = build(x)
        except Exception as exc:  # noqa: BLE001 - try a neighbouring size
            trials.append({"x": x, "vertices": None, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            x *= 1.15
            continue
        trials.append({"x": x, "vertices": int(n)})
        miss = abs(np.log(n / target))
        if best is None or miss < best[0]:
            best = (miss, x, n, payload)
        if abs(n / target - 1.0) <= tol:
            break
        good = [t for t in trials if t["vertices"]]
        slope = -2.5
        if len(good) >= 2:
            (x1, n1), (x2, n2) = [(t["x"], t["vertices"]) for t in good[-2:]]
            if abs(np.log(x2 / x1)) > 1e-6 and n1 != n2:
                slope = (np.log(n2) - np.log(n1)) / (np.log(x2) - np.log(x1))
            if not np.isfinite(slope) or slope > -0.5:
                slope = -2.5
        x = x * (target / n) ** (1.0 / slope)
    if best is None:
        raise RuntimeError(f"no mesh could be built: {trials}")
    return best[1], best[2], best[3], trials


def octree_leaf_sizes(field, x):
    """Edge of the Nitsche octree cell containing each point (nearest cell outside)."""
    cell, _ = field._locate(torch.as_tensor(x, dtype=torch.float64, device=field.device))
    return field.span[cell].min(dim=1).values.cpu().numpy()


def distance_leaf_sizes(source, x, h, band):
    """The octree's 2:1 levels from the distance to the surface:
    s = h * 2**floor(max(0, log2(d / band)))."""
    d = surface_distance(source, torch.as_tensor(x, device="cuda")).cpu().numpy()
    level = np.floor(np.maximum(0.0, np.log2(np.maximum(d, 1e-300) / band)))
    return h * 2.0 ** level


def box_background(lo, hi, step):
    """A Kuhn (6 tets per cube) tet grid of the box [lo, hi], positively oriented.

    TetGen's ``-m`` interpolates the sizing on this background mesh.  A
    background built by TetGen from the body's own surface failed: TetGen could
    not locate the PLC's surface vertices in it ("Failed to locate point ... in
    source mesh", hundreds on the wrench) and the wrapper then segfaulted on
    return.  A box strictly containing the body has every point inside it.
    """
    n = np.maximum(1, np.ceil((hi - lo) / step).astype(int))
    axes = [np.linspace(lo[i], hi[i], n[i] + 1) for i in range(3)]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    index = np.arange(len(nodes)).reshape(n[0] + 1, n[1] + 1, n[2] + 1)
    corners = np.stack([
        index[i:n[0] + i, j:n[1] + j, k:n[2] + k].ravel()
        for i in (0, 1) for j in (0, 1) for k in (0, 1)
    ], axis=1)  # corner bits (i j k): 000 001 010 011 100 101 110 111
    paths = [(0, 4, 6, 7), (0, 4, 5, 7), (0, 1, 5, 7), (0, 1, 3, 7), (0, 2, 3, 7), (0, 2, 6, 7)]
    tets = np.concatenate([corners[:, list(p)] for p in paths])
    p = nodes[tets]
    neg = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) < 0
    tets[neg, 1], tets[neg, 2] = tets[neg, 2], tets[neg, 1].copy()
    return nodes, tets.astype(np.int32)


# ---------------------------------------------------------------------------
# Session geometry flags and rigid fit
# ---------------------------------------------------------------------------


def _face_normals_areas(vertices, faces):
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2 * a, 1e-300)[:, None], a


def ray_lengths(surface, faces_idx, inward: bool, h):
    """Length of the ray from each face centroid along -n (wall) or +n (crevice)."""
    v, f = surface.vertices, surface.faces
    n, _ = _face_normals_areas(v, f)
    c = v[f[faces_idx]].mean(axis=1)
    d = -n[faces_idx] if inward else n[faces_idx]
    mesh = wp.Mesh(points=wp.array(v.astype(np.float32), dtype=wp.vec3),
                   indices=wp.array(f.astype(np.int32).ravel(), dtype=wp.int32))
    out = wp.zeros(len(c), dtype=float)
    wp.launch(_ray_kernel, dim=len(c), inputs=[
        mesh.id, wp.array((c + 1e-4 * h * d).astype(np.float32), dtype=wp.vec3),
        wp.array(d.astype(np.float32), dtype=wp.vec3), out])
    t = out.numpy().astype(np.float64)
    return np.where(t > 0, t, np.inf)


def _area_quantile(values, area, q):
    order = np.argsort(values)
    cum = np.cumsum(area[order]) / max(area.sum(), 1e-300)
    return float(values[order][min(np.searchsorted(cum, q), len(values) - 1)])


def face_components(faces):
    """Connected components of a face set through shared edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    m = len(faces)
    if m == 0:
        return 0, np.zeros(0, int)
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1)
    owner = np.tile(np.arange(m), 3)
    _, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv)
    inv_s, own_s = inv[order], owner[order]
    same = inv_s[1:] == inv_s[:-1]
    g = coo_matrix((np.ones(int(same.sum())), (own_s[:-1][same], own_s[1:][same])), shape=(m, m))
    return connected_components(g, directed=False)


def support_rigid_mode_ratio(source, inputs, area):
    """How well the supports hold the six rigid modes: lambda_min / lambda_max of
    the area-weighted Gram matrix of the rigid modes (translations, and rotations
    about the support centroid in units of the part size) over the clamped
    components of every Dirichlet face.  Near 0: a support set that leaves a
    near rigid-body mode (a near-singular system the solvers cannot fix)."""
    rows, wts = [], []
    for region in inputs.regions:
        if region.category != "dirichlet":
            continue
        idx = np.flatnonzero(region.mask[source.source_face])
        if not len(idx):
            continue
        comps = list(range(3)) if region.bc.components is None else list(region.bc.components)
        c = source.vertices[source.faces[idx]].mean(axis=1)
        rows.append((c, comps))
        wts.append(area[idx])
    if not rows:
        return None
    allc = np.concatenate([c for c, _ in rows])
    allw = np.concatenate(wts)
    xc = (allw[:, None] * allc).sum(0) / allw.sum()
    M = np.zeros((6, 6))
    for (c, comps), w in zip(rows, wts):
        r = (c - xc) / max(float(source.scale), 1e-300)
        A = np.zeros((len(c), 3, 6))
        A[:, 0, 0] = A[:, 1, 1] = A[:, 2, 2] = 1.0
        A[:, 0, 4], A[:, 0, 5] = r[:, 2], -r[:, 1]
        A[:, 1, 3], A[:, 1, 5] = -r[:, 2], r[:, 0]
        A[:, 2, 3], A[:, 2, 4] = r[:, 1], -r[:, 0]
        A = A[:, comps, :]
        M += np.einsum("n,nij,nik->jk", w / allw.sum(), A, A)
    ev = np.linalg.eigvalsh(M)
    return float(ev[0] / ev[-1]) if ev[-1] > 0 else None


def session_flags(source, inputs, h):
    """Thin parts, sub-h crevices, a load next to a support, small or partially
    clamped supports, at the case's analysis h (the session's own depth)."""
    allf = np.arange(len(source.faces))
    _, area = _face_normals_areas(source.vertices, source.faces)
    wall = ray_lengths(source, allf, True, h) / h
    crev = ray_lengths(source, allf, False, h) / h
    flags = {
        "h": h,
        "wall_thickness_over_h_p05": _area_quantile(wall, area, 0.05),
        "wall_thickness_over_h_p50": _area_quantile(wall, area, 0.5),
        "crevice_area_share_below_h": float(area[crev < 1.0].sum() / area.sum()),
    }
    flags["thin"] = flags["wall_thickness_over_h_p50"] < 4.0
    flags["crevices"] = flags["crevice_area_share_below_h"] > 0.01
    supports = []
    for region in inputs.regions:
        if region.category != "dirichlet":
            continue
        idx = np.flatnonzero(region.mask[source.source_face])
        faces = source.faces[idx]
        ncomp, lab = face_components(faces)
        comps = list(range(3)) if region.bc.components is None else list(region.bc.components)
        for k in range(ncomp):
            on = lab == k
            a = float(area[idx[on]].sum())
            if a < 0.01 * float(area[idx].sum()):
                continue
            supports.append(SimpleNamespace(name=f"{region.name}#{k}" if ncomp > 1 else region.name,
                                            faces=faces[on], idx=idx[on], area=a, comps=comps))
    gap = np.inf
    for nreg in [r for r in inputs.regions if r.category == "neumann"]:
        pts = source.vertices[np.unique(source.faces[nreg.mask[source.source_face]])]
        for s in supports:
            if len(pts):
                patch = SimpleNamespace(vertices=source.vertices, faces=s.faces, scale=source.scale)
                gap = min(gap, surface_distance(patch, torch.as_tensor(pts, device="cuda")).min().item() / h)
    flags["load_to_support_gap_over_h"] = None if not np.isfinite(gap) else float(gap)
    flags["load_near_support"] = bool(np.isfinite(gap) and gap < 3.0)
    flags["small_supports"] = [s.name for s in supports if s.area < 50 * h * h]
    flags["partial_clamps_on_thin_sections"] = [
        s.name for s in supports
        if len(s.comps) < 3 and np.median(ray_lengths(source, s.idx, True, h) / h) < 4.0]
    flags["support_issue"] = bool(flags["small_supports"] or flags["partial_clamps_on_thin_sections"])
    flags["support_rigid_mode_ratio"] = support_rigid_mode_ratio(source, inputs, area)
    flags["any"] = bool(flags["thin"] or flags["crevices"] or flags["load_near_support"]
                        or flags["support_issue"])
    return flags


def neo_hookean_stress(grad, mu, kappa, chunk=1 << 21):
    """Cauchy stress and von Mises of the solver's neo-Hookean model
    (PK1 as in ``NeoHookean``'s stress integrand, sigma = P F^T / J)."""
    sigmas, vms = [], []
    for s in range(0, len(grad), chunk):
        g = grad[s:s + chunk]
        eye = torch.eye(3, dtype=g.dtype, device=g.device).expand_as(g)
        F = eye + g
        J = torch.linalg.det(F)
        F_inv_T = torch.linalg.inv(F).mT
        I1 = (F * F).sum(dim=(1, 2))
        iso = mu * J.pow(-2.0 / 3.0)
        P = iso[:, None, None] * F + (kappa * torch.log(J) - iso * I1 / 3.0)[:, None, None] * F_inv_T
        sigma = P @ F.mT / J[:, None, None]
        dev = sigma - sigma.diagonal(dim1=1, dim2=2).mean(dim=1)[:, None, None] * eye
        sigmas.append(sigma)
        vms.append(torch.sqrt(1.5 * (dev * dev).sum(dim=(1, 2))))
    return torch.cat(sigmas), torch.cat(vms)


def weighted_quantile(values, w, q):
    order = torch.argsort(values)
    cum = torch.cumsum(w[order], 0) / w.sum()
    i = int(torch.searchsorted(cum, torch.tensor(q, dtype=cum.dtype, device=cum.device)).clamp(max=len(order) - 1))
    return float(values[order][i])


def rigid_fit(x, d, w, chunk=1 << 22):
    """Weighted least-squares infinitesimal rigid motion t + theta x (x - x_c) of d.

    Built in chunks: the dental crown d8 surface mesh has 73 M quadrature
    points, and the whole (n, 3, 6) design matrix alone is 9.8 GiB."""
    xc = (w[:, None] * x).sum(0) / w.sum()

    def design(s):
        r = x[s:s + chunk] - xc
        A = torch.zeros((len(r), 3, 6), dtype=x.dtype, device=x.device)
        A[:, 0, 0] = A[:, 1, 1] = A[:, 2, 2] = 1.0
        A[:, 0, 4], A[:, 0, 5] = r[:, 2], -r[:, 1]
        A[:, 1, 3], A[:, 1, 5] = -r[:, 2], r[:, 0]
        A[:, 2, 3], A[:, 2, 4] = r[:, 1], -r[:, 0]
        return A

    M = torch.zeros((6, 6), dtype=x.dtype, device=x.device)
    rhs = torch.zeros(6, dtype=x.dtype, device=x.device)
    for s in range(0, len(x), chunk):
        A, ws = design(s), w[s:s + chunk]
        M += torch.einsum("n,nij,nik->jk", ws, A, A)
        rhs += torch.einsum("n,nij,ni->j", ws, A, d[s:s + chunk])
    p = torch.linalg.solve(M, rhs)
    return p, torch.cat([torch.einsum("nij,j->ni", design(s), p) for s in range(0, len(x), chunk)])


def rigid_gradient(p):
    """The gradient (component, direction) of ``rigid_fit``'s motion t + theta x (x - x_c).

    It is constant: the skew matrix of theta = p[3:6], in the column order of
    ``rigid_fit``'s design matrix.  An infinitesimal rotation has a non-zero
    (skew) gradient, so removing the rigid motion changes the H1 seminorm of
    the error too, not only its L2 part; it carries no strain (sym = 0)."""
    W = torch.zeros((3, 3), dtype=p.dtype, device=p.device)
    W[0, 1], W[0, 2] = -p[5], p[4]
    W[1, 0], W[1, 2] = p[5], -p[3]
    W[2, 0], W[2, 1] = -p[4], p[3]
    return W


def h1_errors(w, u_p, g_p, e, g_n, p_rigid, rigid, chunk=1 << 22):
    """Relative H1 errors of e = u_N - u_P1 at the quadrature points (x, w).

    With r* = ``rigid_fit``'s best-fit rigid motion of e (the one the
    ``rel_L2_rigid_removed`` metric removes; ``rigid`` is r* at the points,
    ``p_rigid`` its parameters) and grad r* = ``rigid_gradient(p_rigid)``:

    * ``rel_H1_rigid_removed`` = ||e - r*||_H1 / ||u_P1||_H1, with
      ||v||_H1^2 = int |v|^2 + int |grad v|_F^2 over the body;
    * ``rel_H1_seminorm_rigid_removed`` = |e - r*|_H1 / |u_P1|_H1 (gradients only);
    * ``rel_H1`` = ||e||_H1 / ||u_P1||_H1 (nothing removed).

    The gradients are each method's own: g_p the P1 gradient, g_n the Nitsche
    Q1 cell gradient, the same arrays the stress metrics use.  r* is fitted in
    L2 (not re-fitted in H1).  The norm is not scale invariant: the gradient
    part scales as 1/length^2 against the L2 part, so with lengths in m the
    seminorm dominates the full norm on parts much smaller than 1 m
    (``h1_seminorm_share_of_ref2`` is that share of ||u_P1||_H1^2)."""
    W = rigid_gradient(p_rigid)
    acc = {k: 0.0 for k in ("l2_rem", "semi_rem", "l2_err", "semi_err", "l2_ref", "semi_ref")}
    for s in range(0, len(w), chunk):
        ww = w[s:s + chunk]
        ge = g_n[s:s + chunk] - g_p[s:s + chunk]
        ec = e[s:s + chunk]
        acc["l2_err"] += float((ww * (ec**2).sum(1)).sum())
        acc["semi_err"] += float((ww * (ge**2).sum((1, 2))).sum())
        acc["l2_rem"] += float((ww * ((ec - rigid[s:s + chunk])**2).sum(1)).sum())
        acc["semi_rem"] += float((ww * ((ge - W)**2).sum((1, 2))).sum())
        acc["l2_ref"] += float((ww * (u_p[s:s + chunk]**2).sum(1)).sum())
        acc["semi_ref"] += float((ww * (g_p[s:s + chunk]**2).sum((1, 2))).sum())
    ref = acc["l2_ref"] + acc["semi_ref"]
    return {"rel_H1_rigid_removed": float(np.sqrt((acc["l2_rem"] + acc["semi_rem"]) / ref)),
            "rel_H1_seminorm_rigid_removed": float(np.sqrt(acc["semi_rem"] / acc["semi_ref"])),
            "rel_H1": float(np.sqrt((acc["l2_err"] + acc["semi_err"]) / ref)),
            "h1_seminorm_share_of_ref2": acc["semi_ref"] / ref}


# ---------------------------------------------------------------------------
# Nitsche
# ---------------------------------------------------------------------------


def nitsche_case(inputs, out_root: Path, depth: int, rtol: float, max_iter: int, unit_diagonal: bool,
                 load_steps=None, line_search=None, shape_opt_extra=None):
    case = copy.deepcopy(inputs.case)
    case["output"] = str(out_root)
    so = case["shape_opt"]
    so.update(run_mode="forward", use_mixed_precision=True, relative_tol=rtol, absolute_tol=1e-30,
              max_newton_iters=max_iter, remesh_octree_depth=int(depth))
    if int(so["remesh_init_octree_depth"]) >= depth:
        so["remesh_init_octree_depth"] = depth - 1
    if unit_diagonal:
        so["scale_system_to_unit_diagonal"] = True
    if load_steps:
        so["load_steps"] = int(load_steps)
    if line_search:
        so["line_search_enabled"] = True
    so.update(shape_opt_extra or {})
    return case


# ---------------------------------------------------------------------------
# Solver ladders.  Each rung is one attempt in its own child process
# (``--phase attempt``); the session's orchestrator (``--phase solve``) never
# initializes AMGX itself: pyamgx cannot initialize AMGX after JAX-AMG in one
# process, and a crash or a NaN-poisoned CUDA context then costs one rung only.
# ---------------------------------------------------------------------------

# Nitsche: the operator skill's order for a stalled JAX-AMG solve
# (operator_skill troubleshooting.md, drive.py backend_fallback), unit-diagonal
# scaling as the last GPU resort (a user exception; JAX-AMG only -- pyamgx
# ignores the flag), then the PARDISO hybrid, then continuation, and last the
# session's own tolerance (a flagged looser-tolerance entry).
NITSCHE_LADDER = (
    {"rung": "jaxamg"},
    {"rung": "jaxamg_4000", "overrides": {"jaxamg_max_iters": 4000}, "when": "ran out of iterations"},
    {"rung": "pyamgx", "overrides": {"cg_backend_assembled": "pyamgx"}},
    {"rung": "pyamgx_beta500", "overrides": {"cg_backend_assembled": "pyamgx"},
     "shape_opt": {"beta_max_over_e": 500.0}, "uniform_only": True},
    {"rung": "jaxamg_unit_diagonal", "unit_diagonal": True},
    {"rung": "direct", "direct_solver": True},
    {"rung": "direct_continuation", "direct_solver": True, "load_steps": 10, "line_search": True},
    {"rung": "direct_continuation_session_tol", "direct_solver": True, "load_steps": 10, "line_search": True,
     "rtol": "session", "looser": True},
)
NITSCHE_RUNG = {r["rung"]: r for r in NITSCHE_LADDER}
TRANSIENT = ("WarpCodegen", "Error while parsing function")


@contextmanager
def solver_overrides(overrides: dict):
    """Merge solver parameters into the ShapeOptTrimmed that run_forward builds
    (a retry setting only; run_forward fixes the backend to "amgx")."""
    if not overrides:
        yield
        return
    from core.shape_opt import ShapeOptTrimmed

    original = ShapeOptTrimmed.__init__

    def patched(self, config_module, override_params=None):
        merged = dict(override_params or {})
        merged.update(overrides)
        original(self, config_module, override_params=merged)

    ShapeOptTrimmed.__init__ = patched
    try:
        yield
    finally:
        ShapeOptTrimmed.__init__ = original


def rung_backend(spec):
    if spec.get("direct_solver"):
        return "pardiso-direct"
    return (spec.get("overrides") or spec.get("extra") or {}).get("cg_backend_assembled") or "jaxamg"


def rung_budget(spec, first):
    return ATTEMPT_BUDGET_DIRECT if spec.get("direct_solver") else (
        ATTEMPT_BUDGET_FIRST if first else ATTEMPT_BUDGET_RETRY)


def nitsche_attempt(S, depth, rung):
    """One Nitsche rung (child process): warm-up on the rung's backend, then the
    workflow's run_forward at ``depth`` in float64, timed per phase."""
    args, inputs = S.args, S.inputs
    spec = NITSCHE_RUNG[rung]
    root = S.out_dir / f"nitsche_d{depth}"
    overrides = spec.get("overrides") or {}
    warm = None
    if args.warmup:
        case = nitsche_case(inputs, S.out_dir / "warmup" / "nitsche", depth, args.tight_rtol, 2, False)
        path = S.out_dir / "warmup" / f"nitsche_case_{rung}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(case, indent=2), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            with solver_overrides(overrides):
                workflow_tool.run_forward(SimpleNamespace(case=path))
        except Exception:  # noqa: BLE001 - a warm-up only has to compile
            pass
        warm = time.perf_counter() - t0
        free_gpu()
    rtol = args.session_rtol if spec.get("rtol") == "session" else spec.get("rtol", args.tight_rtol)
    unit_diagonal = bool(spec.get("unit_diagonal"))
    case = nitsche_case(inputs, root, depth, rtol, spec.get("max_iter", args.newton_max_iter), unit_diagonal,
                        spec.get("load_steps"), spec.get("line_search"), spec.get("shape_opt"))
    path = S.out_dir / f"nitsche_d{depth}_case.json"
    path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    workflow_tool._mesh_cache.clear()
    clock = PhaseClock()
    error = None
    started = time.time()
    try:
        with patched_phases(clock, nitsche_phase_patches()), solver_overrides(overrides), \
                direct_solver(spec.get("direct_solver")), attempt_deadline(rung_budget(spec, rung == "jaxamg")):
            with clock.phase("case_parse"):
                workflow_tool.run_forward(SimpleNamespace(case=path))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:600]}"
    runs = [p for p in sorted(root.glob("api_fem_forward-*"), key=lambda p: p.stat().st_mtime)
            if p.stat().st_mtime >= started - 2]
    phases = dict(clock.totals)
    phases["build_case"] = S.build_case_seconds
    rec = {"rung": rung, "backend": rung_backend(spec), "unit_diagonal": unit_diagonal,
           "shape_opt_extra": spec.get("shape_opt"), "rtol": rtol, "overrides": overrides,
           "direct_solver": bool(spec.get("direct_solver")), "looser": bool(spec.get("looser")),
           "load_steps": case["shape_opt"].get("load_steps"), "line_search": case["shape_opt"].get("line_search_enabled"),
           "error": error, "phases": phases, "run_dir": str(runs[-1]) if runs else None,
           "remesh_init_octree_depth": case["shape_opt"]["remesh_init_octree_depth"], "warmup_seconds": warm,
           "direct_rel_residual_max": max(DIRECT_RESIDUALS) if DIRECT_RESIDUALS else None,
           "prep_seconds": sum(phases.get(k, 0.0) for k in NITSCHE_PREP_PHASES),
           "solve_seconds": sum(phases.get(k, 0.0) for k in NITSCHE_SOLVE_PHASES)}
    if rec["run_dir"]:
        log = parse_nitsche_log(Path(rec["run_dir"]))
        rec.update(log=log, dofs=log.get("dofs"),
                   has_vtu=bool(list((Path(rec["run_dir"]) / "vtu").glob("solution_t*.vtu"))),
                   # unit-diagonal scales the linear solve only: Newton's rel is the unscaled force residual
                   unscaled_final_rel=log.get("final_relative_residual"))
        trace = newton_trace(log["log"], rec["solve_seconds"], args.session_rtol)
        rec["solve_seconds_session_tol"] = (trace or {}).get("seconds_to_tol")
        rec["newton_iterations_session_tol"] = (trace or {}).get("iterations_to_tol")
    return rec


# ---------------------------------------------------------------------------
# Body-fitted P1
# ---------------------------------------------------------------------------


# P1 on the standard GPU path (the session's own load steps): the same backend
# order, unit-diagonal last (accepted only if the unscaled residual, recomputed
# from the solution, meets the tolerance); then the CPU queue.
P1_LADDER = (
    {"rung": "jaxamg"},
    {"rung": "jaxamg_4000", "extra": {"jaxamg_max_iters": 4000}, "when": "ran out of iterations"},
    {"rung": "pyamgx", "extra": {"cg_backend_assembled": "pyamgx"}},
    {"rung": "jaxamg_unit_diagonal", "extra": {"jaxamg_max_iters": 4000}, "unit_diagonal": True},
)
P1_RUNG = {r["rung"]: r for r in P1_LADDER}


@contextmanager
def attempt_deadline(seconds):
    """Wall-time cap of one solver attempt, checked before every linear solve
    of its Newton loop (jackhammer d7 graded and school-chair d8 Nitsche
    crawled in their direct-solve attempts at ~6.5 min per iteration without
    converging).  An attempt over its budget fails like any other and the
    chain moves on to the next fallback."""
    import core.abstract_field as af

    original = af.solve_linear_system
    deadline = time.perf_counter() + seconds

    def capped(*a, **k):
        if time.perf_counter() > deadline:
            raise TimeoutError(f"solver attempt exceeded its {seconds:.0f} s budget")
        return original(*a, **k)

    af.solve_linear_system = capped
    try:
        yield
    finally:
        af.solve_linear_system = original


DIRECT_RESIDUALS = []  # the PARDISO linear residual of every direct solve of this process
ATTEMPT_BUDGET_FIRST, ATTEMPT_BUDGET_RETRY, ATTEMPT_BUDGET_DIRECT = 1800.0, 900.0, 5400.0


@contextmanager
def direct_solver(enabled):
    """Replace the Krylov/AMG linear solve by a sparse direct solve (MKL PARDISO
    on the CPU, ``cpu_p1_worker.py --direct``) of the same assembled system: the fallback for systems whose
    conditioning stalls every AMG setup (sliver tets of a -Y source surface)."""
    if not enabled:
        yield
        return
    import core.abstract_field as af
    import scipy.sparse as sp
    original = af.solve_linear_system

    def direct(K, b, *a, **k):
        if K is None or k.get("mv_routine") is not None:
            return original(K, b, *a, **k)
        offsets = K.offsets.numpy()[:K.nrow + 1].astype(np.int64)
        nnz = int(offsets[-1])
        values = K.values.numpy()[:nnz].astype(np.float64)
        bs = values.shape[1]
        A = sp.bsr_matrix((values, K.columns.numpy()[:nnz], offsets), shape=(K.nrow * bs, K.ncol * bs)).tocsc()
        rhs = b.numpy().astype(np.float64).reshape(-1)
        # MKL PARDISO in a child of the CPU venv (multi-threaded, nested
        # dissection; symmetric-indefinite or general factorization): SciPy
        # SuperLU took ~6.5 min per Newton iteration on the jackhammer d8 Nitsche
        # system.  A crash or timeout fails this attempt only.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory(prefix="p1direct_", dir=str(S_TMP)) as tmp:
            src, dst = Path(tmp) / "A.npz", Path(tmp) / "x.npy"
            A = A.tocsr()
            np.savez(src, data=A.data, indices=A.indices, indptr=A.indptr, shape=np.asarray(A.shape), rhs=rhs)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MKL_NUM_THREADS="24", OMP_NUM_THREADS="24")
            try:
                proc = subprocess.run([CPU_PY, str(Path(__file__).resolve().parent / "cpu_p1_worker.py"),
                                       "--direct", str(src), str(dst)], env=env, capture_output=True, text=True,
                                      timeout=DIRECT_TIMEOUT)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"direct solve timed out after {DIRECT_TIMEOUT:.0f} s")
            if proc.returncode != 0 or not dst.is_file():
                found = re.findall(r"rel residual ([0-9.eE+-]+)", proc.stdout)
                if found:
                    DIRECT_RESIDUALS.append(float(found[-1]))
                raise RuntimeError(f"direct solve failed (exit {proc.returncode}): {proc.stdout[-300:]} "
                                   f"{proc.stderr[-300:]}")
            print(f"[direct] {proc.stdout.strip()[-200:]}", flush=True)
            found = re.findall(r"rel residual ([0-9.eE+-]+)", proc.stdout)
            if found:
                DIRECT_RESIDUALS.append(float(found[-1]))
            result = ("ok", np.load(dst))
        return wp.array(np.asarray(result[1]).reshape(rhs.size // bs, bs), dtype=b.dtype, device=b.device)

    af.solve_linear_system = direct
    try:
        yield
    finally:
        af.solve_linear_system = original


@contextmanager
def amg_selector(selector):
    """Swap the JAX-AMG aggregation for one solve (``core.jax_amg_solver.jaxamg_config``)."""
    if not selector:
        yield
        return
    import core.jax_amg_solver as jax_amg

    base = jax_amg.jaxamg_config

    def tuned(block_dim, **kwargs):
        config = base(block_dim, **kwargs)
        config["solver"]["preconditioner"].update({"selector": selector, "presweeps": 2, "postsweeps": 2})
        return config

    jax_amg.jaxamg_config = tuned
    try:
        yield
    finally:
        jax_amg.jaxamg_config = base


def run_bodyfitted(S, tag, mesh, bc, newton, clock, unit_diagonal, extra=None, budget=None):
    inputs = S.inputs
    settings = inputs.settings
    dirichlet = {}
    for region, mask in bc.dirichlet_masks:
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device="cuda")
        # identify_boundary_nodes calls the predicate on the full node list, in
        # node order, which is what the mask indexes.
        dirichlet[lambda x, y, z, m=mask_t: m] = encode_dirichlet(region.bc)
    params = {
        "project_name": f"bodyfitted_{tag}", "output_root": str(S.out_dir / "bodyfitted"),
        "dim": 3, "elem_order": 1, "device": "cuda",
        "tet_mesh": {"nodes": mesh["nodes"], "elements": mesh["elements"]},
        "tet_side_traction": bc.side_table,
        "dirichlet_bcs_dict": dirichlet, "neumann_bcs_dict": {},
        "load_steps": settings.load_steps, "line_search_enabled": settings.line_search_enabled,
        "update_damping_limit": 0.0, "linear_solver": "cg", "ew_enabled": False,
        "cg_backend_assembled": S.args.cg_backend, "scale_system_to_unit_diagonal": unit_diagonal,
        "use_mixed_precision": True, **newton, **material_params(inputs),
        **{k: v for k, v in (extra or {}).items() if k not in ("amg_selector", "direct_solver")},
    }
    with clock.phase("solver_setup"):
        solver = TetNeoHookean("src.agents_wrapper.fem_workflow.config", override_params=params)
    with clock.phase("solve"), amg_selector((extra or {}).get("amg_selector")), \
            direct_solver((extra or {}).get("direct_solver")), \
            attempt_deadline(budget or ATTEMPT_BUDGET_FIRST):
        solver.forward()
    rel = solver.relative_residual_history[-1] if solver.relative_residual_history else None
    return solver, {"log": str(Path(solver.path) / "log.txt"),
                    "newton_iterations": int(solver.newton_iterations),
                    "final_relative_residual": None if rel is None else float(rel),
                    "dofs": int(3 * solver.space.node_count())}


def warmup_bodyfitted(S, newton):
    """Compile/load the P1 kernels once, untimed: a small clamped and loaded
    icosphere through the same TetNeoHookean path, plus the lookup kernels."""
    t0 = time.perf_counter()
    try:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5 * S.source.scale)
        surface = SimpleNamespace(vertices=np.asarray(sphere.vertices, np.float64),
                                  faces=np.asarray(sphere.faces, np.int64), sha256="warmup")
        with mesh_tag("warmup"):
            mesh = tetrahedralize(surface, 0.25 * S.source.scale, 1.5, 10.0, False)
        geo = fem.Tetmesh(tet_vertex_indices=wp.array(mesh["elements"], dtype=wp.int32),
                          positions=wp.array(mesh["nodes"], dtype=wp.vec3d))
        sides = warp_boundary_sides(geo, True)
        z = sides.centroid[:, 2]
        low, high = z < np.quantile(z, 0.15), z > np.quantile(z, 0.85)
        dirichlet = [r for r in S.inputs.regions if r.category == "dirichlet"]
        neumann = [r for r in S.inputs.regions if r.category == "neumann"]
        table = np.zeros((sides.table_size, 3))
        if neumann:
            table[sides.index[high]] = np.asarray(neumann[0].bc.traction, np.float64)
        masks = []
        if dirichlet:
            mask = np.zeros(len(mesh["nodes"]), dtype=bool)
            mask[np.unique(sides.vertices[low])] = True
            masks.append((dirichlet[0], mask))
        bc = SimpleNamespace(side_table=table, dirichlet_masks=masks)
        solver, _ = run_bodyfitted(S, "warmup", mesh, bc, newton, PhaseClock(), S.args.unit_diagonal)
        sample_cells(solver, S.args.eval_order)
        del solver
    except Exception as exc:  # noqa: BLE001 - a warm-up only has to compile
        print(f"[warmup] body-fitted: {exc}")
    S.warmups["bodyfitted"] = time.perf_counter() - t0
    S.warm["bodyfitted"] = True
    free_gpu()


def build_mesh(S, tag, s, family, args_override=None):
    """One tet mesh with TetGen switch fallbacks (all attempts logged)."""
    args = S.args
    path = S.out_dir / "meshes" / f"{tag}.npz"
    cached = None if args.recompute else load_mesh(path, S.source.sha256)
    if cached is not None and abs(cached["h_tet"] - s) <= 1e-9 * s:
        TETGEN_LOG.extend({**a, "cached": True} for a in cached.get("tetgen_attempts", []))
        return cached
    start = len(TETGEN_LOG)
    variants = [{}, {"preserve_surface": "off" if args.preserve_surface != "off" else "on"},
                {"tet_minratio": 2.0, "tet_mindihedral": 0.0}]
    with mesh_tag(tag):
        for variant in variants:
            clock = PhaseClock()
            try:
                mesh = build_tet_mesh(S.source, s, SimpleNamespace(**{**vars(args), **variant}), clock, family)
                break
            except RuntimeError as exc:
                print(f"[mesh {tag}] {variant or 'default'} failed: {str(exc)[:200]}")
        else:
            raise RuntimeError(f"no TetGen variant meshed {tag}")
    meta = mesh_meta(mesh, S.h_by_depth[S.depth])
    meta.update(tetgen_attempts=TETGEN_LOG[start:], phase_seconds=dict(clock.totals))
    save_mesh(path, mesh, meta)
    return {"nodes": mesh["nodes"], "elements": mesh["elements"], **meta}


def dof_mesh(S, tag, target_vertices, known):
    """M(s*) with ~target vertices: secant on log(vertices) vs log(s)."""
    args, h = S.args, S.h_by_depth[S.depth]
    path = S.out_dir / "meshes" / f"{tag}.npz"
    cached = None if args.recompute else load_mesh(path, S.source.sha256)
    if cached is not None and cached.get("dof_target_vertices") == int(target_vertices):
        TETGEN_LOG.extend({**a, "cached": True} for a in cached.get("tetgen_attempts", []))
        return cached
    start = len(TETGEN_LOG)
    t0 = time.perf_counter()
    x0 = h * (known / target_vertices) ** (1.0 / 3.0) if known else h

    def build(s):
        clock = PhaseClock()
        mesh = build_tet_mesh(S.source, s, args, clock, family=True)
        meta = mesh_meta(mesh, h)
        meta["phase_seconds"] = dict(clock.totals)
        return len(mesh["nodes"]), (mesh, meta)

    with mesh_tag(tag):
        s, n, (mesh, meta), trials = secant_size_search(
            build, x0, target_vertices, args.dof_match_tol, args.dof_match_max_iter, h / 3.0, 12.0 * h)
    meta.update(dof_target_vertices=int(target_vertices), dof_match_ratio=n / target_vertices,
                search_trials=trials, search_seconds=time.perf_counter() - t0,
                tetgen_attempts=TETGEN_LOG[start:])
    save_mesh(path, mesh, meta)
    return {"nodes": mesh["nodes"], "elements": mesh["elements"], **meta}


def graded_mesh(S, tag, target_vertices, octree_field):
    """~h_min at the surface (M(h_min)'s surface), interior sized by the
    Nitsche octree's leaf sizes on a box background (TetGen -m), one factor c
    found by the same secant search on the vertex count."""
    args, h = S.args, S.h_by_depth[S.depth]
    path = S.out_dir / "meshes" / f"{tag}.npz"
    cached = None if args.recompute else load_mesh(path, S.source.sha256)
    if cached is not None and cached.get("dof_target_vertices") == int(target_vertices):
        TETGEN_LOG.extend({**a, "cached": True} for a in cached.get("tetgen_attempts", []))
        return cached
    start = len(TETGEN_LOG)
    t0 = time.perf_counter()
    clock = PhaseClock()
    with mesh_tag(tag):
        with clock.phase("surface_remesh"):
            surface, surface_info = surface_at_size(S.source, h, args)
        preserve = [True if surface_info.get("force_preserve")
                    else preserve_surface(surface, h, args.preserve_surface)]
        with clock.phase("sizing_field"):
            lo = surface.vertices.min(axis=0) - 2.0 * h
            hi = surface.vertices.max(axis=0) + 2.0 * h
            step = max(h, (float(np.prod(hi - lo)) / args.graded_bg_max_nodes) ** (1.0 / 3.0))
            bg_nodes, bg_tets = box_background(lo, hi, step)
            if octree_field is not None:
                size, size_source = octree_leaf_sizes(octree_field, bg_nodes), "nitsche_octree"
            else:
                size = distance_leaf_sizes(S.source, bg_nodes, h, args.graded_band * h)
                size_source = f"distance rule, band {args.graded_band} h"
        fixed = dict(clock.totals)

        def build(c):
            local = PhaseClock()
            with local.phase("tetgen"):
                mesh = tetrahedralize(surface, h, args.tet_minratio, args.tet_mindihedral, preserve[0],
                                      background=(bg_nodes, bg_tets, c * size))
            mesh["surface_info"] = surface_info
            meta = mesh_meta(mesh, h)
            meta["phase_seconds"] = {**fixed, **dict(local.totals)}
            return len(mesh["nodes"]), (mesh, meta)

        try:
            c, n, (mesh, meta), trials = secant_size_search(
                build, args.graded_scale0, target_vertices, args.dof_match_tol, args.dof_match_max_iter, 0.2, 8.0)
        except RuntimeError:
            if not preserve[0]:
                raise
            preserve[0] = False  # TetGen -m crashed at every scale: once more without -Y
            c, n, (mesh, meta), trials = secant_size_search(
                build, args.graded_scale0, target_vertices, args.dof_match_tol, args.dof_match_max_iter, 0.2, 8.0)
    meta.update(graded_scale=c, graded_size_source=size_source, dof_target_vertices=int(target_vertices),
                dof_match_ratio=n / target_vertices, search_trials=trials,
                search_seconds=time.perf_counter() - t0, tetgen_attempts=TETGEN_LOG[start:])
    save_mesh(path, mesh, meta)
    return {"nodes": mesh["nodes"], "elements": mesh["elements"], **meta}


# P1 meshes this large skip the GPU and go to the CPU queue.  The d8 surface
# meshes of the hamburger, arch, Pantheon, sofa, lighthouse and mouse (0.49 to
# 2.2 M nodes) ended in GPU out-of-memory or CUDA illegal-address errors; the
# largest GPU-solved meshes so far are ~0.4 M nodes.
CPU_P1_MIN_NODES = 450_000


def _torch_tangent(F, mu, kappa):
    """``cpu_p1_worker.material_tangent`` in torch (dP/dF, (m, 3, 3, 3, 3))."""
    J = torch.linalg.det(F)
    a = torch.linalg.inv(F).mT
    I1 = (F * F).sum(dim=(1, 2))
    iso = mu * J.pow(-2.0 / 3.0)
    eye = torch.eye(3, dtype=F.dtype, device=F.device)
    dev = F - (I1 / 3.0)[:, None, None] * a
    a_il_a_kj = a[:, :, None, None, :] * a.mT[:, None, :, :, None]
    C = (eye[:, None, :, None] * eye[None, :, None, :])[None] \
        - (2.0 / 3.0) * a[:, None, None, :, :] * dev[:, :, :, None, None] \
        - (2.0 / 3.0) * F[:, None, None, :, :] * a[:, :, :, None, None] \
        + (I1 / 3.0)[:, None, None, None, None] * a_il_a_kj
    C = iso[:, None, None, None, None] * C
    C += kappa * a[:, :, :, None, None] * a[:, None, None, :, :] \
        - (kappa * torch.log(J))[:, None, None, None, None] * a_il_a_kj
    return C


def _torch_pk1(F, mu, kappa):
    J = torch.linalg.det(F)
    a = torch.linalg.inv(F).mT
    I1 = (F * F).sum(dim=(1, 2))
    iso = mu * J.pow(-2.0 / 3.0)
    return iso[:, None, None] * F + (kappa * torch.log(J) - iso * I1 / 3.0)[:, None, None] * a


def torch_newton(X, T, f, fixed, u0, mu, kappa, rtol, backend="jaxamg", load_steps=5, max_iter=30,
                 device="cuda", chunk=1_000_000):
    """Full Newton (load steps, backtracking only from non-finite or 1e4x residuals) for one P1
    clamp-band problem, everything on ``device`` in torch float64.

    The 3x3-block sparsity pattern is built once (each tet's 16 vertex pairs ->
    unique blocks, sorted by row then column, as a Warp BSR orders them), and
    every iteration scatter-adds the element tangents into a persistent block
    value array; the clamped components are eliminated in place (identity on
    the diagonal).  ``backend="jaxamg"`` solves through the repo's JAX-AMG on a
    persistent Warp BSR (float64, 5000 Krylov iterations, hierarchy reused across
    iterations); ``"scipy"`` is the CPU test path (sparse LU)."""
    t0 = time.perf_counter()
    dt = torch.float64
    Xt = torch.as_tensor(np.asarray(X, np.float64), dtype=dt, device=device)
    Tt = torch.as_tensor(np.asarray(T, np.int64), device=device)
    n, m = len(Xt), len(Tt)
    vol = torch.empty(m, dtype=dt, device=device)
    G = torch.empty((m, 4, 3), dtype=dt, device=device)
    for s in range(0, m, chunk):
        p = Xt[Tt[s:s + chunk]]
        D = torch.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], dim=2)
        vol[s:s + chunk] = torch.linalg.det(D).abs() / 6.0
        Dinv = torch.linalg.inv(D)
        G[s:s + chunk] = torch.cat([-Dinv.sum(dim=1, keepdim=True), Dinv], dim=1)
    # Block pattern: key = row * n + col for the 16 (a, b) vertex pairs of every tet.
    inv = torch.empty(m * 16, dtype=torch.int32, device=device)
    keys = (Tt[:, :, None] * n + Tt[:, None, :]).reshape(-1)
    ukeys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    inv.copy_(inverse.to(torch.int32))
    del keys, inverse
    brow, bcol = ukeys // n, ukeys % n
    del ukeys
    nnzb = len(brow)
    freeN = torch.as_tensor(~np.asarray(fixed, bool), dtype=dt, device=device)
    elim = freeN[brow][:, :, None] * freeN[bcol][:, None, :]
    diag = torch.nonzero(brow == bcol).squeeze(1)
    diag_add = torch.diag_embed(1.0 - freeN[brow[diag]])
    if backend == "jaxamg":
        from core.jax_amg_solver import solve_jaxamg

        K = wp.sparse.bsr_from_triplets(
            n, n, wp.from_torch(brow.to(torch.int32)), wp.from_torch(bcol.to(torch.int32)),
            wp.zeros(nnzb, dtype=wp.mat33d, device=device), prune_numerical_zeros=False)
        K.nnz_sync()
        if K.nnz != nnzb or not torch.equal(wp.to_torch(K.columns)[:nnzb].long(), bcol):
            raise RuntimeError("Warp BSR block order differs from the pattern")
        vals = wp.to_torch(K.values).view(-1, 3, 3)[:nnzb]
        cache = {"jaxamg_force_rebuild": True}
    else:
        vals = torch.zeros((nnzb, 3, 3), dtype=dt, device=device)
    iters = []

    def assemble(u):
        vals.zero_()
        for s in range(0, m, chunk):
            g, Tc = G[s:s + chunk], Tt[s:s + chunk]
            F = torch.eye(3, dtype=dt, device=device) + torch.einsum("mac,mad->mcd", u[Tc], g)
            C = _torch_tangent(F, mu, kappa)
            Ke = torch.einsum("maikl,mbl->mabik", torch.einsum("maj,mijkl->maikl", g, C), g) \
                * vol[s:s + chunk, None, None, None, None]
            vals.index_add_(0, inv[16 * s:16 * (s + len(g))].long(), Ke.reshape(-1, 3, 3))
            del F, C, Ke
        vals.mul_(elim)
        vals[diag] += diag_add

    def linsolve(b):
        if backend == "jaxamg":
            rhs = wp.from_torch(b.contiguous(), dtype=wp.vec3d)
            x, niter, _ = solve_jaxamg(K, rhs, use_mixed_precision=True, cache=cache, device=device,
                                       rebuild_every_iter=False, scale_to_unit_diagonal=False, max_iters=5000)
            iters.append(int(niter))
            return wp.to_torch(x).clone().view(n, 3)
        import scipy.sparse as sps
        import scipy.sparse.linalg as spla
        A = sps.bsr_matrix((vals.cpu().numpy(), bcol.cpu().numpy(),
                            np.concatenate([[0], np.cumsum(np.bincount(brow.cpu().numpy(), minlength=n))])),
                           shape=(3 * n, 3 * n)).tocsc()
        return torch.as_tensor(spla.spsolve(A, b.cpu().numpy().reshape(-1)), dtype=dt, device=device).view(n, 3)

    def residual(u, scale):
        out = torch.zeros((n, 3), dtype=dt, device=device)
        for s in range(0, m, chunk):
            g, Tc = G[s:s + chunk], Tt[s:s + chunk]
            F = torch.eye(3, dtype=dt, device=device) + torch.einsum("mac,mad->mcd", u[Tc], g)
            fe = torch.einsum("mcd,mad->mac", _torch_pk1(F, mu, kappa), g) * vol[s:s + chunk, None, None]
            out.index_add_(0, Tc.reshape(-1), fe.reshape(-1, 3))
        return (out - scale * ft) * freeN

    ft = torch.as_tensor(np.asarray(f, np.float64), dtype=dt, device=device)
    u = torch.as_tensor(np.asarray(u0, np.float64), dtype=dt, device=device)
    t_setup = time.perf_counter() - t0
    steps, stamps, alphas, t_asm, t_lin = [], [], [], 0.0, 0.0
    for step in range(1, load_steps + 1):
        scale = step / load_steps
        hist, R = [], residual(u, scale)
        for it in range(max_iter + 1):
            norm = float(torch.linalg.norm(R))
            if not np.isfinite(norm):
                raise RuntimeError(f"Newton diverged (non-finite residual) in load step {step}")
            hist.append(norm)
            stamps.append(time.perf_counter() - t0)
            print(f"[lean-gpu] scale={scale:.3f} Iter {it}: ||R|| = {norm:.6e}, rel = {norm / hist[0]:.6e}",
                  flush=True)
            if (it > 0 and norm / hist[0] <= rtol) or it == max_iter:
                break
            ta = time.perf_counter()
            assemble(u)
            tb = time.perf_counter()
            du = linsolve(-R) * freeN
            t_asm, t_lin = t_asm + tb - ta, t_lin + time.perf_counter() - tb
            alpha = 1.0
            for _ in range(7):
                trial = residual(u + alpha * du, scale)
                tn = float(torch.linalg.norm(trial))
                # Plain Newton, as the GPU path runs it (teddy bear d8 dofs: rel 287
                # after the first iteration, then quadratic); back off only from
                # a non-finite residual (J <= 0) or a 1e4x growth.
                if np.isfinite(tn) and tn < 1e4 * norm:
                    break
                alpha *= 0.5
            if alpha < 1.0:
                alphas.append((step, it, alpha))
            u = u + alpha * du
            R = trial if np.isfinite(tn) and alpha >= 1.0 / 64 else residual(u, scale)
        steps.append([h / hist[0] for h in hist])
        if hist[-1] / hist[0] > rtol:
            raise RuntimeError(f"Newton did not reach rel {rtol:g} in load step {step}: {steps[-1]}")
    stats = {"solver": "lean-gpu" if backend == "jaxamg" else f"torch-{backend}",
             "method": f"full Newton, {load_steps} load steps, backtracking only from non-finite or 1e4x residuals; "
                       "torch tangent into a fixed block pattern, " + ("JAX-AMG" if backend == "jaxamg" else backend),
             "load_step_histories": steps, "relative_residuals": steps[-1], "line_search_steps": alphas,
             "newton_iterations": sum(len(s) - 1 for s in steps), "krylov_iterations": iters,
             "final_relative_residual": steps[-1][-1], "dofs": 3 * n, "blocks": nnzb, "iteration_seconds": stamps,
             "seconds": {"setup": t_setup, "tangent_assembly": t_asm, "linear_solves": t_lin,
                         "total": time.perf_counter() - t0}}
    return u.cpu().numpy(), stats


LEAN_GPU_VALIDATION = ("teddy-bear/p1_solution_d7_dofs.npz", "elephant/p1_solution_d8_surface.npz")
_lean_validated = {}


def validate_lean_gpu(S, rtol):
    """End-to-end check of ``torch_newton`` on this GPU against two stored
    standard-path (TetNeoHookean) solutions: the teddy bear d7 dofs
    (nonlinear) and the elephant d8 surface (near-linear).  Each is re-solved
    from its stored mesh, clamp and load; rel L2 and the compliance f.u must
    match to 1e-6, else the lean path is not used.  Once per process; written to
    ``<output root>/_batch/lean_gpu_validation.json``."""
    if "result" in _lean_validated:
        return _lean_validated["result"]
    root = Path(S.args.validation_root)
    cases = []
    for rel in LEAN_GPU_VALIDATION:
        z = np.load(root / rel)
        t0 = time.perf_counter()
        u, st = torch_newton(z["nodes"], z["elements"], z["f_ext"], z["clamped"], np.zeros_like(z["u"]),
                             float(z["mu"]), float(z["kappa"]), rtol, backend="jaxamg", load_steps=5)
        ref = z["u"]
        c_ref = float((z["f_ext"] * ref).sum())
        cases.append({"case": rel, "dofs": st["dofs"], "seconds": time.perf_counter() - t0,
                      "rel_L2_nodal": float(np.linalg.norm(u - ref) / np.linalg.norm(ref)),
                      "compliance_rel_diff": float(abs((z["f_ext"] * u).sum() - c_ref) / abs(c_ref)),
                      "newton_iterations": st["newton_iterations"],
                      "final_relative_residual": st["final_relative_residual"]})
        free_gpu()
    ok = all(c["rel_L2_nodal"] <= 1e-6 and c["compliance_rel_diff"] <= 1e-6 for c in cases)
    result = {"ok": ok, "threshold": 1e-6, "rtol": rtol, "cases": cases, "time": datetime.now().isoformat()}
    path = S.out_dir.parent / "_batch" / "lean_gpu_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"[lean-gpu validation] {json.dumps(result)}", flush=True)
    _lean_validated["result"] = result
    return result


def cpu_load_and_clamp(S, mesh, bc):
    """The GPU path's load vector and clamp, built on the CPU from the same BC data.

    Traction: the per-side table is constant per side and P1 test functions
    integrate to area/3 at each side vertex (what the order-2 side rule gives).
    Body force: vol/4 * B per tet vertex.  Clamp: every component listed by a
    Dirichlet region, with its value (``encode_dirichlet``'s reading)."""
    X, T = np.asarray(mesh["nodes"], np.float64), np.asarray(mesh["elements"], np.int64)
    n = len(X)
    mat = material_params(S.inputs)
    vol, _ = cpu_p1_worker.tet_gradients(X, T)
    body = np.asarray(mat["B"], np.float64)
    f = cpu_p1_worker.nodal_sum(n, T, np.broadcast_to((vol / 4.0)[:, None, None] * body, (len(T), 4, 3)))
    side_force = bc.side_table[bc.sides.index] * (bc.sides.area / 3.0)[:, None]
    for k in range(3):
        idx = bc.sides.vertices[:, k]
        f += np.stack([np.bincount(idx, weights=side_force[:, c], minlength=n) for c in range(3)], axis=1)
    fixed, u0 = np.zeros((n, 3), bool), np.zeros((n, 3))
    for region, mask in bc.dirichlet_masks:
        item = region.bc
        comps = [0, 1, 2] if item.components is None else list(item.components)
        value = item.value
        vals = np.asarray(value if isinstance(value, (tuple, list)) else (value,) * 3, np.float64)
        for c in comps:
            fixed[mask, c] = True
            u0[mask, c] = vals[c]
    E, nu = float(mat["E"]), float(mat["nu"])
    return f, fixed, u0, E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))


def p1_bc(S, mesh):
    """The clamp-band boundary data of one P1 mesh (BC tolerance of the depth)."""
    inputs, h = S.inputs, S.h_by_depth[S.depth]
    eps = float(inputs.case["shape_opt"].get("phase_field_epsilon_factor", 0.5)) * h
    radius = bc_match_tolerance(eps, h)
    grow = {r.name: radius for r in inputs.regions if r.category == "dirichlet"}
    clock = PhaseClock()
    bc = bf_boundary_data(inputs, S.source, mesh, 1, S.args.traction_weighting, clock, grow=grow)
    return bc, radius, clock


def load_force_error(S, mesh, bc, f):
    """The traction part of an assembled load against sum(t * area)."""
    expected = (bc.side_table[bc.sides.index] * bc.sides.area[:, None]).sum(axis=0)
    vol, _ = cpu_p1_worker.tet_gradients(np.asarray(mesh["nodes"], np.float64), np.asarray(mesh["elements"], np.int64))
    body = vol.sum() * np.asarray(material_params(S.inputs)["B"], np.float64)
    return float(np.linalg.norm(f.sum(axis=0) - body - expected) / max(np.linalg.norm(expected), 1e-300))


def p1_info(S, tag, mesh, bc, radius, clock, stats, force_error, trace):
    """The entry's record (everything but the metrics)."""
    h = S.h_by_depth[S.depth]
    phases = {**mesh.get("phase_seconds", {}), **dict(clock.totals)}
    prep_success = sum(phases.get(k, 0.0) for k in ("surface_remesh", "sizing_field", "tetgen",
                                                     "boundary_mapping", "bc_transfer", "solver_setup"))
    failed_tetgen = sum(a.get("seconds") or 0.0 for a in mesh.get("tetgen_attempts") or [] if not a.get("ok"))
    return {
        "p1_dofs": stats.get("dofs", 3 * len(mesh["nodes"])), "p1_vertices": len(mesh["nodes"]),
        "band_radius_over_h": radius / h, "assembled_force_rel_error": force_error,
        "neumann": bc.neumann_report, "region_area_ratio": bc.region_area_ratio,
        "solve": stats,
        "timing": {
            "phases": phases, "prep_seconds": prep_success,
            "prep_seconds_incl_failed_tetgen": prep_success + failed_tetgen,
            "tetgen_seconds_success": phases.get("tetgen"),
            "tetgen_seconds_incl_failed": (phases.get("tetgen") or 0.0) + failed_tetgen,
            "search_seconds": mesh.get("search_seconds"),
            "solve_device": stats.get("solver"),
            "solve_seconds": clock.totals.get("solve"),
            "solve_seconds_session_tol": (trace or {}).get("seconds_to_tol"),
            "newton_iterations_session_tol": (trace or {}).get("iterations_to_tol"),
        },
        "mesh": {k: mesh.get(k) for k in ("switches", "s_over_h", "boundary_edge_median_over_h", "n_tets",
                                          "quality_min", "dof_match_ratio", "graded_scale", "graded_size_source",
                                          "surface_info")},
        "surface_source": (mesh.get("surface_info") or {}).get("surface_source", "remeshed/source"),
    }


def unscaled_p1_rel(npz):
    """||f_int(u) - f||/||f|| over the free components, recomputed from a stored
    solution (the check for a unit-diagonal solve, whose Krylov tolerance is the
    scaled system's)."""
    z = np.load(npz)
    X, T = z["nodes"].astype(np.float64), z["elements"].astype(np.int64)
    vol, G = cpu_p1_worker.tet_gradients(X, T)
    free = ~np.asarray(z["clamped"], bool)
    r = (cpu_p1_worker.internal_force(X, T, vol, G, z["u"].astype(np.float64), float(z["mu"]), float(z["kappa"]))
         - z["f_ext"])[free]
    return float(np.linalg.norm(r) / max(np.linalg.norm(z["f_ext"][free]), 1e-300))


def stored_mesh(S, tag):
    mesh = load_mesh(S.out_dir / "meshes" / f"{tag}.npz", S.source.sha256)
    if mesh is None:
        raise RuntimeError(f"no cached mesh for {tag}")
    return mesh


def p1_attempt(S, tag, rung):
    """One P1 rung (child process) on the standard GPU path; the solution is kept
    only if the rung converged (and, for unit-diagonal, the unscaled residual
    meets the tolerance)."""
    args = S.args
    spec = P1_RUNG[rung]
    mesh = stored_mesh(S, tag)
    newton = {"relative_tol": args.tight_rtol, "absolute_tol": 1e-30, "max_iter": args.newton_max_iter}
    extra = dict(spec.get("extra") or {})
    if extra.get("cg_backend_assembled") == "pyamgx":
        args.cg_backend = "pyamgx"  # the warm-up too: AMGX first initialized by pyamgx
    warm = None
    if args.warmup:
        warmup_bodyfitted(S, newton)
        warm = S.warmups.get("bodyfitted")
    bc, radius, clock = p1_bc(S, mesh)
    unit_diagonal = bool(spec.get("unit_diagonal"))
    rec = {"rung": rung, "backend": rung_backend(spec) if extra.get("cg_backend_assembled") else "jaxamg",
           "unit_diagonal": unit_diagonal, "extra": extra, "warmup_seconds": warm}
    local = PhaseClock()
    try:
        solver, stats = run_bodyfitted(S, tag, mesh, bc, newton, local, unit_diagonal, extra,
                                       rung_budget(spec, rung == "jaxamg"))
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:500]}", oom=is_oom(exc),
                   seconds=sum(local.totals.values()))
        return rec
    for k, v in local.totals.items():
        clock.totals[k] = clock.totals.get(k, 0.0) + v
    stats.update(solver="gpu", backend=rec["backend"], unit_diagonal=unit_diagonal, rung=rung, solver_extra=extra)
    load, _, force_error = check_load(solver, bc)
    f_ext = external_load_vector(solver, load)
    tmp = S.out_dir / f"p1_solution_{tag}.{rung}.npz"
    np.savez_compressed(
        tmp, nodes=mesh["nodes"], elements=mesh["elements"],
        u=wp.to_torch(solver.u.dof_values).to(torch.float64).cpu().numpy(),
        clamped=solver.bc_mask.cpu().numpy() if torch.is_tensor(solver.bc_mask) else np.asarray(solver.bc_mask),
        side_traction=bc.side_table, f_ext=f_ext.cpu().numpy(), mu=float(solver.mu), kappa=float(solver.kappa),
        band_radius=radius)
    if unit_diagonal:
        stats["unscaled_rel"] = unscaled_p1_rel(tmp)
        if stats["unscaled_rel"] > args.tight_rtol:
            tmp.unlink()
            rec.update(ok=False, error=f"unit-diagonal solve not accepted: unscaled rel {stats['unscaled_rel']:.2e} "
                                       f"> {args.tight_rtol:g}", unscaled_rel=stats["unscaled_rel"])
            return rec
    tmp.replace(S.out_dir / f"p1_solution_{tag}.npz")
    trace = newton_trace(stats["log"], clock.totals.get("solve"), args.session_rtol)
    rec.update(ok=True, info=p1_info(S, tag, mesh, bc, radius, clock, stats, force_error, trace))
    return rec


def defer_cpu(S, tag, mesh, reason, attempts):
    """Write the entry's problem for the driver's CPU queue (``cpu_p1_worker.py``)
    with the GPU path's discretization, and return the deferred record."""
    args = S.args
    bc, radius, clock = p1_bc(S, mesh)
    f_np, fixed, u0, mu, kappa = cpu_load_and_clamp(S, mesh, bc)
    jobs = S.out_dir / "cpu_jobs"
    jobs.mkdir(exist_ok=True)
    for suffix in (".done", ".failed", ".running"):
        (jobs / f"{tag}{suffix}").unlink(missing_ok=True)
    np.savez(jobs / f"{tag}.npz", nodes=mesh["nodes"], elements=mesh["elements"], f=f_np, fixed=fixed,
             u0=u0, mu=mu, kappa=kappa, side_traction=bc.side_table, band_radius=radius)
    stats = {"solver": "cpu", "dofs": 3 * len(mesh["nodes"]), "reason": reason, "gpu_attempts": attempts}
    info = p1_info(S, tag, mesh, bc, radius, clock, stats, load_force_error(S, mesh, bc, f_np), None)
    (jobs / f"{tag}.json").write_text(json.dumps(
        {"info": info, "reason": reason, "rtol": args.tight_rtol, "session_rtol": args.session_rtol,
         "linear": "pardiso", "load_steps": int(S.inputs.settings.load_steps or 5),
         "written": datetime.now().isoformat()}, indent=2, default=str))
    return {"ok": False, "deferred_cpu": True, "cpu_reason": reason,
            "error": f"deferred to the CPU queue: {reason}", "p1_vertices": len(mesh["nodes"]),
            "gpu_attempts": attempts}


def lean_attempt(S, tag):
    """The lean GPU full Newton (child process) for a CPU job the CPU worker
    found too nonlinear for its fixed tangent and too large for PARDISO; the
    path validates itself on this GPU first (``validate_lean_gpu``)."""
    args = S.args
    jobs = S.out_dir / "cpu_jobs"
    z = np.load(jobs / f"{tag}.npz")
    job = json.loads((jobs / f"{tag}.json").read_text())
    validation = validate_lean_gpu(S, args.tight_rtol)
    if not validation["ok"]:
        return {"ok": False, "error": f"lean GPU path not used: its validation failed {validation['cases']}"}
    t0 = time.perf_counter()
    u, st = torch_newton(z["nodes"], z["elements"], z["f"], z["fixed"], z["u0"], float(z["mu"]), float(z["kappa"]),
                         float(job["rtol"]), backend="jaxamg", load_steps=int(job.get("load_steps") or 5))
    np.savez_compressed(S.out_dir / f"p1_solution_{tag}.npz", nodes=z["nodes"], elements=z["elements"], u=u,
                        clamped=z["fixed"], side_traction=z["side_traction"], f_ext=z["f"], mu=z["mu"],
                        kappa=z["kappa"], band_radius=z["band_radius"])
    info = job["info"]
    rows = [(k, r) for hist in st["load_step_histories"] for k, r in enumerate(hist)]
    hit = next((i for i, (k, r) in enumerate(rows) if k > 0 and r <= job["session_rtol"]), None)
    cpu_failure = (jobs / f"{tag}.failed").read_text()[-400:] if (jobs / f"{tag}.failed").is_file() else None
    info["solve"] = {**st, "validation": validation, "unit_diagonal": False, "solver_extra": {"lean_gpu": True},
                     "reason": f"{job['reason']}; the CPU's fixed-tangent iteration did not contract: lean GPU "
                               f"full Newton", "cpu_failure": cpu_failure}
    info["timing"]["phases"]["solve"] = time.perf_counter() - t0
    info["timing"].update(solve_device="lean-gpu", solve_seconds=time.perf_counter() - t0,
                          solve_seconds_session_tol=None if hit is None else st["iteration_seconds"][hit],
                          newton_iterations_session_tol=hit)
    info.update(p1_dofs=st["dofs"], ok=True)
    (S.out_dir / f"result_{tag}.json").write_text(json.dumps(info, indent=2, default=str))
    (jobs / f"{tag}.done").write_text(datetime.now().isoformat())
    (jobs / f"{tag}.failed").unlink(missing_ok=True)
    return {"ok": True}


def stored_p1(path: Path):
    """A stored P1 solution as the fields the evaluation needs (no solve)."""
    z = np.load(path)
    geo = fem.Tetmesh(tet_vertex_indices=wp.array(z["elements"].astype(np.int32), dtype=wp.int32),
                      positions=wp.array(z["nodes"].astype(np.float64), dtype=wp.vec3d))
    space = fem.make_polynomial_space(geo=geo, degree=1, dtype=wp.vec3d)
    u = space.make_field()
    u.dof_values = wp.array(z["u"].astype(np.float64), dtype=wp.vec3d)
    dev = "cuda"
    return SimpleNamespace(
        geo=geo, u=u, use_mixed_precision=True, _has_spatial_material=False,
        nodes=torch.as_tensor(z["nodes"], dtype=torch.float64, device=dev),
        u_nodes=torch.as_tensor(z["u"], dtype=torch.float64, device=dev),
        f_ext=torch.as_tensor(z["f_ext"], dtype=torch.float64, device=dev),
        mu=float(z["mu"]), kappa=float(z["kappa"]), source_volume=None)


def evaluate_p1(S, tag, sol, field):
    """Every metric of one (depth, mode) from the stored P1 solution and the
    stored Nitsche field, on the P1 mesh's own quadrature."""
    args, inputs, h = S.args, S.inputs, S.h_by_depth[S.depth]
    qp = sample_cells(sol, args.eval_order)
    x, w, u_p = qp["x"], qp["w"], qp["u"]
    u_n, g_n, outside = field.evaluate(x)
    e = u_n - u_p
    ref2 = (w * (u_p**2).sum(1)).sum()
    p_rig, rig = rigid_fit(x, e, w)
    rem = e - rig
    # H1 (an addition; the metrics below are unchanged): the same r*, points
    # and per-method gradients as the rigid-removed L2 and the stress metrics.
    h1 = h1_errors(w, u_p, qp["grad"], e, g_n, p_rig, rig)
    u_n_nodes, _, _ = field.evaluate(sol.nodes)
    if S.depth_stats is None:
        dist = surface_distance(S.source, x)
        o = torch.argsort(dist)
        cum = torch.cumsum(w[o], 0) / w.sum()
        S.depth_stats = {"max_depth_over_h": float(dist.max() / S.h_case),
                         "median_depth_over_h": float(dist[o][int(torch.searchsorted(
                             cum, torch.tensor(0.5, dtype=cum.dtype, device=cum.device)).clamp(max=len(o) - 1))]
                             / S.h_case)}
    # Stress: each method's Cauchy stress from its own gradient, same material.
    # An infinitesimal rigid motion carries no strain, so no rigid removal here.
    sig_p, vm_p = neo_hookean_stress(qp["grad"], sol.mu, sol.kappa)
    sig_n, vm_n = neo_hookean_stress(g_n, sol.mu, sol.kappa)
    del g_n
    qp.pop("grad", None)
    torch.cuda.empty_cache()

    def rel(a, b, m=None, chunk=1 << 22):
        # Accumulated in chunks: the dental crown d8 surface mesh has 73 M
        # quadrature points, and whole-array temporaries ran out of GPU memory.
        num = den = 0.0
        for s in range(0, len(w), chunk):
            ww, aa, bb = w[s:s + chunk], a[s:s + chunk], b[s:s + chunk]
            if m is not None:
                mm = m[s:s + chunk]
                ww, aa, bb = ww[mm], aa[mm], bb[mm]
            num += float((ww * ((aa - bb) ** 2).reshape(len(ww), -1).sum(1)).sum())
            den += float((ww * (bb ** 2).reshape(len(ww), -1).sum(1)).sum())
        return float(np.sqrt(num / den)) if den > 0 else None
    away = torch.full((len(x),), float("inf"), dtype=torch.float64, device=x.device)
    for region in inputs.regions:
        faces = S.source.faces[region.mask[S.source.source_face]]
        if len(faces):
            patch = SimpleNamespace(vertices=S.source.vertices, faces=faces, scale=S.source.scale)
            away = torch.minimum(away, surface_distance(patch, x))
    away = away >= 2.0 * h
    return {
        "rel_L2_rigid_removed": float(torch.sqrt((w * (rem**2).sum(1)).sum() / ref2)),
        "rel_L2": float(torch.sqrt((w * (e**2).sum(1)).sum() / ref2)),
        "rigid_share_of_err2": float(1.0 - (w * (rem**2).sum(1)).sum() / (w * (e**2).sum(1)).sum()),
        "compliance_ratio": float((sol.f_ext * u_n_nodes).sum() / (sol.f_ext * sol.u_nodes).sum()),
        "outside_octree_volume_fraction": float(w[outside].sum() / w.sum()),
        "mesh_volume_over_source": float(w.sum()) / S.source.volume,
        "h1": h1,
        "stress": {
            "rel_L2_cauchy": rel(sig_n, sig_p), "rel_L2_von_mises": rel(vm_n, vm_p),
            "von_mises_p95_ratio": weighted_quantile(vm_n, w, 0.95) / max(weighted_quantile(vm_p, w, 0.95), 1e-300),
            "von_mises_p99_ratio": weighted_quantile(vm_n, w, 0.99) / max(weighted_quantile(vm_p, w, 0.99), 1e-300),
            "away_from_patches_volume_fraction": float(w[away].sum() / w.sum()),
            "rel_L2_cauchy_away_from_patches": rel(sig_n, sig_p, away) if bool(away.any()) else None,
            "rel_L2_von_mises_away_from_patches": rel(vm_n, vm_p, away) if bool(away.any()) else None,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", help="workflow session folder")
    ap.add_argument("--case", help="case.json (default: <session>/case/case.json)")
    ap.add_argument("--regions", help="resolved.npz (default: <session>/boundary_conditions/resolved.npz)")
    ap.add_argument("--output", required=True, help="output folder of this session")
    ap.add_argument("--depths", type=int, nargs="+", default=[7, 8], help="Nitsche remesh_octree_depth values")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="body-fitted P1 meshes")
    ap.add_argument("--tight-rtol", type=float, default=1e-6,
                    help="common Newton tolerance both methods are solved to (relative)")
    ap.add_argument("--session-rtol", type=float, default=None,
                    help="the session's own tolerance, timed from the same solve (default: case.json's)")
    ap.add_argument("--newton-max-iter", type=int, default=30)
    ap.add_argument("--unit-diagonal", action="store_true",
                    help="solve both methods' systems with scale_system_to_unit_diagonal from the start "
                         "(default: only as the retry after a stall)")
    ap.add_argument("--cg-backend", default="amgx")
    ap.add_argument("--traction-weighting", choices=("nitsche", "full"), default="nitsche",
                    help="body-fitted load: the Nitsche definition t*max(0, n.n_r) (default) or plain t")
    ap.add_argument("--eval-order", type=int, default=2, help="P1 quadrature order of the error integrals")
    ap.add_argument("--no-standin", action="store_true",
                    help="keep a per-part material (by default both methods use the volume-weighted stand-in)")
    ap.add_argument("--tet-minratio", type=float, default=1.5)
    ap.add_argument("--tet-mindihedral", type=float, default=10.0)
    ap.add_argument("--preserve-surface", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--surface-band", type=float, nargs=2, default=(0.75, 1.33))
    ap.add_argument("--remesh-iterations", type=int, default=10)
    ap.add_argument("--remesh-max-surf-dist", type=float, default=0.2)
    ap.add_argument("--dof-match-tol", type=float, default=0.05)
    ap.add_argument("--dof-match-max-iter", type=int, default=6)
    ap.add_argument("--graded-scale0", type=float, default=1.15)
    ap.add_argument("--graded-band", type=float, default=2.0)
    ap.add_argument("--graded-bg-max-nodes", type=float, default=2.0e6)
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--phase", choices=("solve", "evaluate", "attempt"), default="solve",
                    help="solve: the session's orchestrator (meshes here, every solver rung in its own process; "
                         "resumable); evaluate: every metric from the stored outputs only; attempt: one rung "
                         "(launched by solve)")
    ap.add_argument("--recompute", action="store_true", help="ignore cached meshes / redo the evaluation")
    ap.add_argument("--retry-failed", action="store_true", help="kept for the driver; the ladders resume anyway")
    ap.add_argument("--kind", choices=("nitsche", "p1", "lean"), help="attempt: which rung")
    ap.add_argument("--depth", type=int, help="attempt (nitsche): the depth")
    ap.add_argument("--tag", help="attempt (p1, lean): the entry, e.g. d8_dofs")
    ap.add_argument("--rung", help="attempt: the rung name (NITSCHE_LADDER / P1_LADDER)")
    ap.add_argument("--record", help="attempt: where the rung's record is written")
    ap.add_argument("--validation-root", default=str(DATA_ROOT / "output" / "bodyfitted_vs_nitsche"),
                    help="where the lean GPU path finds its two stored standard-path validation cases")
    return ap.parse_args(argv)


CUDA_FATAL = ("an illegal memory access", "CUDA error", "cudaErrorIllegalAddress", "unspecified launch failure")


def environment_info(args):
    """Code version, library versions, GPU and options, for reproducibility."""
    import importlib.metadata as md
    import subprocess

    def version(name):
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001
            return None
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=30).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "experiments", "src"],
                                    capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:  # noqa: BLE001
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "python": sys.version.split()[0],
            "versions": {n: version(n) for n in ("tetgen", "warp-lang", "torch", "pymeshlab", "numpy", "trimesh", "scipy")},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
            "time": datetime.now().isoformat()}


def setup_session(args):
    np.random.seed(0)
    torch.manual_seed(0)
    wp.init()
    inputs = load_inputs(args)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.session_rtol is None:
        args.session_rtol = float(inputs.settings.relative_tol)
    h_case = float(inputs.case["h_target"]) * 2.0 ** (int(inputs.case["octree_depth"])
                                                      - inputs.settings.remesh_octree_depth)
    source = clean_surface(Path(inputs.case["surface"]), len(inputs.regions[0].mask))
    S = SimpleNamespace(args=args, inputs=inputs, out_dir=out_dir, source=source, warm={}, warmups={},
                        failures=[], depth=None, depth_stats=None, h_case=h_case,
                        h_by_depth={d: float(inputs.case["h_target"]) * 2.0 ** (int(inputs.case["octree_depth"]) - d)
                                    for d in args.depths})
    S.env = environment_info(args)
    print(f"[env] {json.dumps({k: v for k, v in S.env.items() if k != 'options'})}")
    return S


def mark(S, stage):
    (S.out_dir / "status.json").write_text(
        json.dumps({"stage": stage, "time": datetime.now().isoformat()}), encoding="utf-8")


def record_failure(S, stage, exc):
    text = f"{type(exc).__name__}: {exc}"
    S.failures.append({"stage": stage, "reason": text[:1500], "oom": is_oom(exc),
                       "traceback": traceback.format_exc(limit=6)})
    print(f"[{stage}] FAILED: {text[:300]}")


def true_tetgen_failure(error):
    """A failure the meshing itself cannot get past: a generated surface that
    self-intersects, or TetGen failing for every variant / every size."""
    e = error or ""
    return ("self-intersects" in e or "no TetGen variant meshed" in e or "no mesh could be built" in e)


def tetgen_failure_evidence(S):
    """For a true TetGen failure: TetGen -d (exact predicates) on the cleaned
    source with its intersection counts, the pymeshlab per-face count, and
    TetGen with every variant of the chain on the source as-is (incl. -Y);
    written to ``tetgen_evidence.json`` (the summary prints it)."""
    import multiprocessing as mp
    import pymeshlab
    import tempfile as tf

    path = S.out_dir / "tetgen_evidence.json"
    if path.is_file():
        return json.loads(path.read_text())
    v, f = S.source.vertices, S.source.faces
    ev = {"object": S.out_dir.name, "surface": str(S.source.path), "cleaned_vertices": len(v), "cleaned_faces": len(f)}
    try:
        ev["generator"] = json.loads((Path(S.inputs.session) / "session.json").read_text()).get(
            "generation", {}).get("generator")
    except Exception:  # noqa: BLE001
        ev["generator"] = None
    tmp = tf.mkdtemp(prefix="tg_d_")
    p = mp.get_context("spawn").Process(target=tetgen_child.detect_intersections, args=(v, f, tmp))
    p.start()
    p.join(TETGEN_TIMEOUT)
    if p.is_alive():
        p.kill()
        p.join()
    text = Path(tmp, "tetgen_d.txt").read_text(errors="replace") if Path(tmp, "tetgen_d.txt").is_file() else ""
    ev["tetgen_d_counts"] = {
        "facet_facet_warnings": text.count("Two facets exactly intersect") + text.count("facets intersect"),
        "segment_facet_warnings": text.count("A segment and a facet intersect"),
        "skipped_triangles": re.findall(r"(\d+) input triangles are skipped", text)}
    ev["tetgen_d_output"] = "\n".join(l for l in text.splitlines() if l.strip())[-1500:]
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(v, np.float64), np.asarray(f, np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    ev["pymeshlab_self_intersecting_faces"] = int(ms.current_mesh().selected_face_number())
    ev["tetgen_variants_on_source"] = []
    for sw in ("pq1.5/10YQ", "pq1.5/10Q", "pq2/0YQ", "pq2/0Q", "pYQ", "pQ"):
        t0 = time.perf_counter()
        try:
            _, elems, _ = _tetgen_isolated(v, f, sw, None)
            res = {"switches": sw, "ok": True, "tets": int(len(elems))}
        except Exception as exc:  # noqa: BLE001
            res = {"switches": sw, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        res["seconds"] = round(time.perf_counter() - t0, 1)
        ev["tetgen_variants_on_source"].append(res)
    path.write_text(json.dumps(ev, indent=2))
    return ev


def run_child(S, extra, timeout, record_path):
    """One ladder rung in its own process; its record, or a crash record.  A
    record written before a non-zero exit (an AMGX destructor abort at exit)
    stands."""
    import subprocess

    record_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["timeout", "--kill-after=60", str(int(timeout)), sys.executable, str(Path(__file__).resolve()),
           *S.child_argv, "--phase", "attempt", "--record", str(record_path), *extra]
    t0 = time.time()
    sys.stdout.flush()
    with open(S.out_dir / "run.log", "a") as fh:
        fh.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} attempt: {' '.join(extra)}\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if record_path.is_file():
        rec = json.loads(record_path.read_text())
    else:
        rec = {"ok": False, "error": f"the attempt process ended with exit code {rc} and no record "
                                     f"({'timeout' if rc in (124, 137) else 'crash'})"}
    rec.update(process_returncode=rc, process_seconds=time.time() - t0)
    record_path.write_text(json.dumps(rec, indent=2, default=str))
    return rec


def _skip(spec, recs, S):
    """None if the rung runs, else why it is skipped."""
    if spec.get("when"):
        last = next((r for r in reversed(recs) if not r.get("skipped")), None)
        if last is None or spec["when"] not in str(last.get("error")):
            return f"only after '{spec['when']}'"
    if spec.get("uniform_only") and (S.inputs.standin or S.inputs.case["material"]["mode"] != "uniform"):
        return "beta_max_over_e applies to a uniform material only (this session is a stand-in or not uniform)"
    return None


def nitsche_ladder(S, depth, upto=None):
    """The Nitsche rungs (resumable from their records) until one converges;
    ``upto`` stops after that many run rungs."""
    recs = []
    for i, spec in enumerate(NITSCHE_LADDER):
        why = _skip(spec, recs, S)
        if why:
            if spec.get("uniform_only"):
                recs.append({"rung": spec["rung"], "backend": rung_backend(spec), "skipped": True,
                             "error": f"skipped: {why}"})
            continue
        rp = S.out_dir / "attempts" / f"nitsche_d{depth}_{spec['rung']}.json"
        if rp.is_file():
            rec = json.loads(rp.read_text())
        else:
            mark(S, f"nitsche:d{depth}:{spec['rung']}")
            rec = run_child(S, ["--kind", "nitsche", "--depth", str(depth), "--rung", spec["rung"]],
                            rung_budget(spec, i == 0) + 900, rp)
        recs.append(rec)
        print(f"[nitsche d{depth} {spec['rung']}] {'ok' if rec.get('error') is None else rec.get('error')[:160]}",
              flush=True)
        if rec.get("error") is None and rec.get("run_dir"):
            return rec, recs
        if upto is not None and sum(1 for r in recs if not r.get("skipped")) >= upto:
            return None, recs
    return None, recs


def nitsche_record(S, depth, final, recs):
    run = [r for r in recs if not r.get("skipped")]
    base = final if final is not None else (run[-1] if run else {})
    res = {k: base.get(k) for k in ("run_dir", "unit_diagonal", "rtol", "overrides", "direct_solver", "backend",
                                    "shape_opt_extra", "looser", "remesh_init_octree_depth", "prep_seconds",
                                    "solve_seconds", "phases", "log", "solve_seconds_session_tol",
                                    "newton_iterations_session_tol", "unscaled_final_rel")}
    res.update(depth=depth, attempts=recs, ok=final is not None,
               error=None if final is not None else (base.get("error") or "no Nitsche rung ran"),
               fallback=max(len(run) - 1, 0), rung=base.get("rung"),
               dofs=next((r.get("dofs") for r in [base, *recs] if r.get("dofs")), None),
               leaf_h=float(S.inputs.case["h_target"]) * 2.0 ** (int(S.inputs.case["octree_depth"]) - depth))
    return res


def p1_ladder(S, tag, mesh):
    """The P1 rungs (resumable) on the standard GPU path; then the CPU queue."""
    if len(mesh["nodes"]) >= CPU_P1_MIN_NODES:
        return defer_cpu(S, tag, mesh, f"{len(mesh['nodes'])} nodes >= {CPU_P1_MIN_NODES} (GPU memory)", [])
    recs = []
    for i, spec in enumerate(P1_LADDER):
        if _skip(spec, recs, S):
            continue
        rp = S.out_dir / "attempts" / f"p1_{tag}_{spec['rung']}.json"
        if rp.is_file():
            rec = json.loads(rp.read_text())
        else:
            mark(S, f"p1:{tag}:{spec['rung']}")
            rec = run_child(S, ["--kind", "p1", "--tag", tag, "--rung", spec["rung"]],
                            rung_budget(spec, i == 0) + 900, rp)
        recs.append({k: v for k, v in rec.items() if k != "info"})
        print(f"[{tag} {spec['rung']}] {'ok' if rec.get('ok') else str(rec.get('error'))[:160]}", flush=True)
        if rec.get("ok"):
            info = rec["info"]
            info["solve"]["gpu_attempts"] = recs
            info["ok"] = True
            return info
        if rec.get("oom"):
            return defer_cpu(S, tag, mesh, f"GPU out of memory: {str(rec.get('error'))[:200]}", recs)
    return defer_cpu(S, tag, mesh, "every GPU rung failed: " + "; ".join(
        f"{r.get('rung')}: {str(r.get('error'))[:100]}" for r in recs), recs)


def p1_mode(S, depth, mode, nit_dofs, field, source_bad, known):
    """One P1 entry: cached record, or mesh (TetGen chain) and its ladder."""
    tag = f"d{depth}_{mode}"
    out_dir, h = S.out_dir, S.h_by_depth[depth]
    rpath, npz = out_dir / f"result_{tag}.json", out_dir / f"p1_solution_{tag}.npz"
    if rpath.is_file():
        res = json.loads(rpath.read_text(encoding="utf-8"))
        deferred = res.get("deferred_cpu") and (out_dir / "cpu_jobs" / f"{tag}.json").is_file()
        lean_pending = "needs full Newton" in str(res.get("error"))  # the driver's lean GPU phase has it
        if (res.get("ok") and npz.is_file()) or deferred or lean_pending or true_tetgen_failure(res.get("error")):
            mp_ = out_dir / "meshes" / f"{tag}.json"
            if mp_.is_file():
                TETGEN_LOG.extend({**a, "cached": True} for a in json.loads(mp_.read_text()).get("tetgen_attempts", []))
            return res
    if source_bad:
        res = {"ok": False, "error": "generated surface self-intersects (TetGen -d); not repaired"}
    elif mode != "surface" and not nit_dofs:
        res = {"ok": False, "error": "no Nitsche DOF count at this depth (the octree build failed)"}
    elif mode == "graded" and field is None:
        res = {"ok": False, "error": "no Nitsche octree at this depth (no Nitsche run wrote a field)"}
    else:
        mark(S, f"p1:{tag}")
        try:
            if mode == "surface":
                mesh = build_mesh(S, tag, h, family=True)
            elif mode == "dofs":
                # The secant search starts from the surface mesh's vertex count:
                # read it from the cached surface mesh, so the start (and so the
                # mesh) does not depend on whether the surface entry's record
                # exists yet (a crash or transient error there changed it).
                sm = out_dir / "meshes" / f"d{depth}_surface.json"
                if sm.is_file():
                    known = json.loads(sm.read_text()).get("n_nodes") or known
                mesh = dof_mesh(S, tag, nit_dofs // 3, known)
            else:
                mesh = graded_mesh(S, tag, nit_dofs // 3, field)
            try:
                res = p1_ladder(S, tag, mesh)
            except Exception as exc:  # noqa: BLE001 - a transient Warp code-generation error: once more
                if not any(m in str(exc) for m in TRANSIENT):
                    raise
                print(f"[{tag}] transient ({str(exc)[:120]}); once more", flush=True)
                free_gpu()
                res = p1_ladder(S, tag, mesh)
            res.setdefault("surface_source", (mesh.get("surface_info") or {}).get("surface_source", "remeshed/source"))
        except Exception as exc:  # noqa: BLE001
            record_failure(S, f"p1:{tag}", exc)
            res = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"}
            if true_tetgen_failure(res["error"]):
                tetgen_failure_evidence(S)
    rpath.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"[{tag}] ok={res.get('ok')} {res.get('error') or ''}", flush=True)
    free_gpu()
    return res


def octree_field(nit, recs):
    """The Nitsche octree (Q1 field on its leaves) from any rung that wrote a VTU:
    the octree does not depend on the load or the solver."""
    for r in [nit, *reversed(recs)]:
        if r and r.get("run_dir"):
            vtus = sorted((Path(r["run_dir"]) / "vtu").glob("solution_t*.vtu"))
            if vtus:
                return OctreeQ1Field(read_vtu(vtus[-1]), "cuda")
    return None


def solve_phase(S):
    """Phase 1, the session's orchestrator: TetGen meshes here (spawned children),
    every solver rung in its own process, every output and record saved; no
    metrics.  Resumable from the records.  P1 does not wait on Nitsche: the
    matched surface needs h only, the matched DOFs the octree's DOF count (known
    after the first Nitsche rung), the graded mesh any Nitsche VTU (the octree)."""
    args, inputs, out_dir = S.args, S.inputs, S.out_dir
    replay = build_case_replay(inputs, out_dir)
    S.build_case_seconds = replay.get("seconds", 0.0)
    (out_dir / "replay").mkdir(exist_ok=True)
    (out_dir / "replay" / "timing.json").write_text(json.dumps(replay))
    if (out_dir / "solve_done").is_file():
        (out_dir / "solve_done").unlink()
    record_path = out_dir / "solve.json"
    prev = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    if "source_self_intersects" in prev:
        source_bad = prev["source_self_intersects"]
        TETGEN_LOG.extend(a for a in prev.get("tetgen_log", []) if a.get("mesh") == "source")
    else:
        with mesh_tag("source"):
            source_bad = self_intersects(S.source.vertices, S.source.faces)
    if source_bad:
        tetgen_failure_evidence(S)
    depths, warm = {}, {}
    for depth in args.depths:
        S.depth = depth
        D = depths[str(depth)] = {"h": S.h_by_depth[depth], "modes": {}}
        cache = out_dir / f"nitsche_d{depth}.json"
        nit = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else None
        if nit is not None and nit.get("ok"):
            final, recs = nit, nit.get("attempts") or []
        else:
            final, recs = nitsche_ladder(S, depth, upto=1)
        dofs = next((r.get("dofs") for r in [final or {}, *recs] if r and r.get("dofs")), None)
        known = None
        for mode in ("surface", "dofs"):
            if mode in args.modes:
                res = p1_mode(S, depth, mode, dofs, None, source_bad, known)
                D["modes"][mode] = str(out_dir / f"result_d{depth}_{mode}.json")
                if mode == "surface":
                    known = res.get("p1_vertices")
        if final is None:
            final, recs = nitsche_ladder(S, depth)
        if not (nit is not None and nit.get("ok")):
            nit = nitsche_record(S, depth, final, recs)
            cache.write_text(json.dumps(nit, indent=2, default=str), encoding="utf-8")
        D["nitsche"] = str(cache)
        warm.setdefault("nitsche", next((r.get("warmup_seconds") for r in recs if r.get("warmup_seconds")), None))
        print(f"[nitsche d{depth}] ok={nit.get('ok')} rung {nit.get('rung')} dofs {nit.get('dofs')} "
              f"{nit.get('error') or ''}", flush=True)
        if "graded" in args.modes:
            field = octree_field(nit, recs)
            p1_mode(S, depth, "graded", nit.get("dofs") or dofs, field, source_bad, known)
            D["modes"]["graded"] = str(out_dir / f"result_d{depth}_graded.json")
            field = None
        free_gpu()
    record = {
        "inputs": {"session": str(inputs.session), "case": str(inputs.case_path), "surface": str(S.source.path),
                   "material": inputs.case["material"], "standin": inputs.standin,
                   "case_remesh_octree_depth": inputs.settings.remesh_octree_depth, "h_case": S.h_case},
        "environment": S.env, "source_self_intersects": source_bad,
        "depths": depths, "warmup_seconds": warm, "build_case_replay": replay,
        "tetgen_log": [a for a in TETGEN_LOG if a["mesh"] != "warmup"],
        "failures": S.failures,
    }
    with open(out_dir / "failures_history.jsonl", "a") as fh:
        for f in S.failures:
            fh.write(json.dumps({**f, "time": datetime.now().isoformat()}) + "\n")
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    (out_dir / "solve_done").write_text(datetime.now().isoformat(), encoding="utf-8")
    mark(S, "solve_done")
    print(f"[solve done] {record_path}", flush=True)


def attempt_phase(S):
    """One rung in this process (``--kind nitsche|p1|lean``); its record to ``--record``."""
    a = S.args
    timing = S.out_dir / "replay" / "timing.json"
    S.build_case_seconds = json.loads(timing.read_text()).get("seconds", 0.0) if timing.is_file() else 0.0
    try:
        if a.kind == "nitsche":
            S.depth = a.depth
            rec = nitsche_attempt(S, a.depth, a.rung)
        else:
            S.depth = int(a.tag[1:].split("_")[0])
            rec = p1_attempt(S, a.tag, a.rung) if a.kind == "p1" else lean_attempt(S, a.tag)
    except Exception as exc:  # noqa: BLE001
        rec = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}", "oom": is_oom(exc),
               "traceback": traceback.format_exc(limit=8)}
    Path(a.record).write_text(json.dumps(rec, indent=2, default=str))


def _child_argv(argv):
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x in ("--phase", "--record", "--kind", "--depth", "--tag", "--rung"):
            skip = True
            continue
        out.append(x)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    S = setup_session(args)
    S.child_argv = _child_argv(argv)
    if args.phase == "solve":
        solve_phase(S)
    elif args.phase == "attempt":
        attempt_phase(S)
    else:
        evaluate_phase(S)


def evaluate_phase(S):
    """Phase 2: every metric from the stored files only (the Nitsche VTUs and
    the P1 solution npz), then ``comparison.json`` with the summary row."""
    out_dir = S.out_dir
    cmp_path = out_dir / "comparison.json"
    inputs_ = [out_dir / "solve.json", *out_dir.glob("result_d*.json"), *out_dir.glob("nitsche_d?.json"),
               *out_dir.glob("p1_solution_*.npz")]
    if (cmp_path.is_file() and not S.args.recompute
            and cmp_path.stat().st_mtime > max(p.stat().st_mtime for p in inputs_ if p.is_file())):
        print(f"[evaluate] {cmp_path} is newer than every input; skipped (--recompute to redo)")
        return
    record = json.loads((out_dir / "solve.json").read_text(encoding="utf-8"))
    TETGEN_LOG.extend(record.get("tetgen_log") or [])
    try:
        S.flags = session_flags(S.source, S.inputs, S.h_case)
    except Exception as exc:  # noqa: BLE001
        S.failures.append({"stage": "flags", "reason": f"{type(exc).__name__}: {exc}"})
        S.flags = None
    results = {}
    for d, D in record["depths"].items():
        S.depth = int(d)
        nit = json.loads(Path(D["nitsche"]).read_text(encoding="utf-8"))
        R = results[d] = {"h": D["h"], "nitsche": nit, "modes": {}}
        field = None
        for mode, rpath in D["modes"].items():
            res = json.loads(Path(rpath).read_text(encoding="utf-8"))
            npz = out_dir / f"p1_solution_d{d}_{mode}.npz"
            if res.get("ok") and nit.get("ok"):
                if field is None:
                    vtus = sorted((Path(nit["run_dir"]) / "vtu").glob("solution_t*.vtu"))
                    field = OctreeQ1Field(read_vtu(vtus[-1]), "cuda")
                mark(S, f"evaluate:d{d}_{mode}")
                try:
                    res.update(evaluate_p1(S, f"d{d}_{mode}", stored_p1(npz), field))
                    res["evaluated_from_stored"] = [str(npz), nit["run_dir"]]
                except Exception as exc:  # noqa: BLE001
                    res = {**res, "ok": False, "error": f"evaluation: {type(exc).__name__}: {str(exc)[:400]}"}
                    S.failures.append({"stage": f"evaluate:d{d}_{mode}", "reason": res["error"]})
                free_gpu()
            R["modes"][mode] = res
        field = None
        free_gpu()
    report = {
        "inputs": record["inputs"], "environment_solve": record.get("environment"),
        "environment_evaluate": S.env, "flags": {**(S.flags or {}), **(S.depth_stats or {})},
        "source_self_intersects": record.get("source_self_intersects"),
        "depths": results, "warmup_seconds": record.get("warmup_seconds"),
        "tetgen_log": record.get("tetgen_log") or [],
        "failures": (record.get("failures") or []) + S.failures,
    }
    report["tetgen_summary"] = tetgen_summary(report["tetgen_log"])
    report["row"] = summary_row(report)
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    mark(S, "evaluated")
    print(f"[evaluate done] {out_dir / 'comparison.json'}")


def tetgen_summary(log):
    out = {}
    for e in log:
        m = out.setdefault(e["mesh"], {"attempts": 0, "failures": [], "succeeded": False,
                                       "success_switches": None})
        m["attempts"] += 1
        if e["ok"]:
            m["succeeded"], m["success_switches"] = True, e["switches"]
        else:
            m["failures"].append({"stage": e.get("stage"), "switches": e["switches"],
                                  "type": e["failure_type"], "message": e["message"]})
    return out


def summary_row(report):
    row = {"session": Path(report["inputs"]["session"]).name,
           "standin": "yes" if report["inputs"].get("standin") else "no",
           "E": report["inputs"]["material"].get("youngs_modulus"),
           "case_depth": report["inputs"]["case_remesh_octree_depth"]}
    fl = report.get("flags") or {}
    for k in ("any", "thin", "crevices", "load_near_support", "support_issue", "wall_thickness_over_h_p50",
              "crevice_area_share_below_h", "load_to_support_gap_over_h", "max_depth_over_h", "median_depth_over_h",
              "support_rigid_mode_ratio"):
        row[f"flag_{k}"] = fl.get(k)
    row["flag_small_supports"] = ",".join(fl.get("small_supports") or [])
    row["flag_partial_thin"] = ",".join(fl.get("partial_clamps_on_thin_sections") or [])
    tg = report.get("tetgen_summary") or {}
    row["tetgen_failed_meshes"] = ",".join(sorted(k for k, v in tg.items() if v["failures"]))
    row["tetgen_unmeshed"] = ",".join(sorted(k for k, v in tg.items() if not v["succeeded"]))
    for d, D in report["depths"].items():
        nit = D.get("nitsche") or {}
        row[f"d{d}_nitsche_dofs"] = nit.get("dofs")
        row[f"d{d}_nitsche_prep_s"] = nit.get("prep_seconds")
        row[f"d{d}_nitsche_solve_s"] = nit.get("solve_seconds")
        row[f"d{d}_nitsche_solve_session_tol_s"] = nit.get("solve_seconds_session_tol")
        row[f"d{d}_nitsche_unit_diagonal"] = nit.get("unit_diagonal")
        row[f"d{d}_nitsche_load_steps"] = ((nit.get("attempts") or [{}])[-1] or {}).get("load_steps")
        for mode, r in (D.get("modes") or {}).items():
            p = f"d{d}_{mode}"
            if not r.get("ok"):
                row[f"{p}_error"] = (r.get("error") or "")[:200]
                continue
            if "rel_L2" not in r:  # solved, but its depth's Nitsche solve did not converge
                row[f"{p}_error"] = "solved; not compared (no converged Nitsche solve at this depth)"
                row[f"{p}_p1_dofs"] = r.get("p1_dofs")
                continue
            t = r["timing"]
            row.update({
                f"{p}_p1_dofs": r["p1_dofs"], f"{p}_relL2_rigid_removed": r["rel_L2_rigid_removed"],
                f"{p}_relL2": r["rel_L2"], f"{p}_compliance": r["compliance_ratio"],
                f"{p}_stress_relL2": r["stress"]["rel_L2_cauchy"],
                f"{p}_vm_relL2": r["stress"]["rel_L2_von_mises"],
                f"{p}_vm_p95": r["stress"]["von_mises_p95_ratio"],
                f"{p}_vm_p99": r["stress"]["von_mises_p99_ratio"],
                f"{p}_stress_relL2_away": r["stress"]["rel_L2_cauchy_away_from_patches"],
                f"{p}_away_fraction": r["stress"]["away_from_patches_volume_fraction"],
                f"{p}_p1_prep_s": t["prep_seconds"], f"{p}_p1_prep_incl_failed_s": t["prep_seconds_incl_failed_tetgen"],
                f"{p}_p1_solve_s": t["solve_seconds"], f"{p}_p1_solve_session_tol_s": t["solve_seconds_session_tol"],
                f"{p}_p1_unit_diagonal": r["solve"].get("unit_diagonal"),
                f"{p}_p1_solver_fallback": json.dumps(r["solve"].get("solver_extra") or {}),
                f"{p}_p1_solve_device": t.get("solve_device") or "gpu",
                f"{p}_p1_solve_method": r["solve"].get("method") or "GPU Newton (TetNeoHookean, AMG)",
                f"{p}_surface_source": r.get("surface_source"),
            })
    row["failures"] = "; ".join(f"{f['stage']}: {f['reason'][:160]}" for f in report["failures"])
    return row


if __name__ == "__main__":
    main()
