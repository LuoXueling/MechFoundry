"""Deterministic LangGraph coordinator for the economical FEM workflow."""

from __future__ import annotations

import contextlib
import hashlib
import json
import mimetypes
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, TypedDict

from pydantic import ValidationError

from src.agents_wrapper import AgentsWrapperError

from .models import (
    AssetRecord, BCProposal, BCValuesInput, FIELD_MODES, GeometryGenerationSettings,
    MaterialParameters, NeumannForce, OPTIMIZATION_FAILURE_ADVICE_MARKER, PartMaterials,
    SegmentationSettings, ShapeOptSettings, Stage, UNIT_SYSTEM_DETAILS, UNIT_SYSTEMS,
    WorkflowState, optimization_failure_advice, write_text_atomically,
)
from src.cad_wrapper.cad_common.mesh import face_labels_path
from src.cad_wrapper.cadcoder_wrapper.availability import cadcoder_status
from src.cad_wrapper.text2cad_wrapper.availability import text2cad_status
from src.hunyuan3d_wrapper.availability import hunyuan3d_status, sdxl_status, texture_status
from src.hunyuan3d_wrapper.surface import has_colour, plain_colour
from src.partfield_wrapper.availability import partfield_status
from .assets import SOLVE_FAILURE
from .runtime_files import OPTIMIZATION_STATUS
from .monitor import ToolFailed, ToolMonitor, _digest, stopped
from .specialists import SpecialistCalls


class GraphState(TypedDict, total=False):
    workflow: dict[str, Any]
    pending: dict[str, Any]


def _session_folder() -> str | None:
    """The folder this run's checkpoints live in, when the caller names it."""
    try:
        from langgraph.config import get_config

        return get_config().get("configurable", {}).get("session_dir")
    except (ImportError, RuntimeError):
        return None


def _rebased(value: Any, old: str, new: str) -> Any:
    """``value`` with every path under ``old`` moved to the same place under ``new``."""
    if isinstance(value, str):
        if value == old or value.startswith(old + os.sep):
            return new + value[len(old):]
        return value
    if isinstance(value, dict):
        return {key: _rebased(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebased(item, old, new) for item in value]
    return value


def _state(raw: GraphState) -> WorkflowState:
    """The workflow state, with its paths in the folder the session is run from.

    Every path the state holds is absolute and was recorded where the session
    was created, so a session folder that had been copied or moved kept
    reading and writing the original: every node writes under
    ``state.session_dir``.  The front ends name the folder they opened in the
    run's config, and a state recorded elsewhere is moved there on the way in,
    whichever checkpoint it comes from.
    """
    workflow = raw["workflow"]
    folder, recorded = _session_folder(), workflow.get("session_dir")
    if folder and recorded and Path(recorded) != Path(folder):
        workflow = _rebased(workflow, str(Path(recorded)), str(Path(folder)))
    state = WorkflowState.model_validate(workflow)
    _restore_branch_files(state)
    return state


def _dump(state: WorkflowState) -> GraphState:
    _record_branch_files(state)
    snapshot = state.root / "session.json"
    write_text_atomically(snapshot, state.model_dump_json(indent=2))
    return {"workflow": state.model_dump(mode="json")}


# The folders whose files later steps read back by path.  The paths are fixed
# -- the scaled surface is always geometry/physical.glb, the labels always
# boundary_conditions/segments.npz, a part field always materials/parts/field
# -- so every branch of a rewound session wrote over the others', and a rewind
# found the newest branch's: an earlier branch's settings card built its case
# on the newer segments, and would have on a newer surface, scale or field.
BRANCH_FOLDERS = ("inputs", "geometry", "boundary_conditions", "materials", "case")
# Inside them but no one branch's: PartField's features, keyed by the surface's
# own digest, so a rewind finds the right ones or makes them again.
SHARED_FOLDERS = ("boundary_conditions/partfield_features",)
# Every version any branch left, by content, for a rewind to put back.
FILE_VERSIONS = "versions"


def _branch_files(root: Path) -> list[str]:
    """The session's branch files, relative to it, leaving out partial writes."""
    found = []
    for folder in BRANCH_FOLDERS:
        for directory, subfolders, files in os.walk(root / folder):
            here = Path(directory)
            subfolders[:] = [name for name in subfolders if not name.startswith(".")
                             and (here / name).relative_to(root).as_posix() not in SHARED_FOLDERS]
            found.extend((here / name).relative_to(root).as_posix()
                         for name in files if not name.startswith("."))
    return found


def _file_version(root: Path, digest: str, relative: str) -> Path:
    return root / FILE_VERSIONS / f"{digest}{Path(relative).suffix}"


def _copy_atomically(source: Path, target: Path) -> None:
    """Replace ``target`` with a copy of ``source`` in one step (see write_text_atomically).

    The copy keeps the source's modification time.  The viewer opens on the
    newest file and picks the newer material field by it, so a file put back
    has to be as new as when its branch wrote it, not as new as the rewind.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(handle)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _keep_version(root: Path, relative: str, digest: str) -> None:
    if not _file_version(root, digest, relative).is_file():
        _copy_atomically(root / relative, _file_version(root, digest, relative))


def _record_branch_files(state: WorkflowState) -> None:
    """Note the branch files as this step leaves them, and keep a copy of each."""
    recorded = {}
    for relative in _branch_files(state.root):
        digest = _digest(state.root / relative)
        if digest:
            _keep_version(state.root, relative, digest)
            recorded[relative] = digest
    state.session_files = recorded


def _restore_branch_files(state: WorkflowState) -> None:
    """Put back the files this checkpoint left, whichever branch wrote them since.

    Only what the checkpoint recorded: a file it did not have is left where it
    is.  No step takes one for its own unless the state names it or its stamp
    names this surface, and the viewer shows the last surface and field made
    while an earlier card is decided again.  What is replaced is kept among the
    versions first.
    """
    if state.session_files is None:
        return
    root = state.root.resolve()
    for relative, wanted in state.session_files.items():
        path = root / relative
        if root not in path.resolve().parents:
            continue
        found = _digest(path)
        if found == wanted or not _file_version(root, wanted, relative).is_file():
            continue
        if found:
            _keep_version(root, relative, found)
        _copy_atomically(_file_version(root, wanted, relative), path)


def _api_error_detail(exc: Exception) -> str:
    """Summarize a failure in the terms needed to fix it.

    A pydantic error's first line only says how many problems there were; the
    field and the reason live on the lines after it, so reporting line one alone
    leaves "1 validation error for BCProposal" and nothing to act on.
    """
    if isinstance(exc, ValidationError):
        problems = [
            f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
            for item in exc.errors()[:3]
        ]
        remaining = exc.error_count() - len(problems)
        summary = "; ".join(problems)
        if remaining > 0:
            summary += f" (+{remaining} more)"
        return summary[:400]
    return str(exc).splitlines()[0][:240]


class _APIFailed(Exception):
    """A specialist call failed; the card that triggered it reports it inline."""


def _clear_api_error(state: WorkflowState) -> None:
    """Drop a reported failure once the user acts on the card again."""
    state.api_error = None
    state.api_error_source = None


def _material_artifact_summary(path: str | None) -> dict[str, Any] | None:
    """Read only the tiny review manifest; never load the 64^3 array in the UI."""
    if not path:
        return None
    try:
        manifest = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return {
        key: manifest[key]
        for key in ("ranges", "occupied_voxels", "source_voxels", "unit_system")
        if key in manifest
    }


def _surface_has_colour(path: str | None) -> bool | None:
    """Whether the analysis surface carries colour VoMP can see; None if it cannot be read."""
    if not path:
        return None
    try:
        return has_colour(path)
    except Exception:  # an unreadable file is reported elsewhere; here it is just unknown
        return None


def _surface_digest(path: str | None) -> str:
    """The contents of a surface file, "" when there is none (cached by file identity)."""
    return _digest(Path(path)) if path else ""


def _current_texture(state: WorkflowState) -> str | None:
    """The painted copy of the analysis surface, if it was painted from the current one."""
    if (state.textured_surface and state.textured_from == state.surface_path
            and Path(state.textured_surface).is_file()
            and state.textured_from_digest == _surface_digest(state.surface_path)):
        return state.textured_surface
    return None


# What any solid a case is about falls within, in SI.  E runs from the softest
# foams to diamond and density from aerogel-light foam to osmium; E/density
# (the specific modulus, m²/s²) runs from soft silicone (~90) to diamond
# (~3e8), which is what catches a slip that the absolute spans let through: a
# steel part's 2e11 Pa given as 2e5 is a plausible foam's E, but not at 7850
# kg/m³.  A part outside any of them is taken for a unit slip, not a material.
PLAUSIBLE_MODULUS_PA = (1.0e3, 1.2e12)
PLAUSIBLE_DENSITY_KG_M3 = (1.0, 2.5e4)
PLAUSIBLE_SPECIFIC_MODULUS = (50.0, 4.0e8)


def _implausible_part(parts: list[dict[str, Any]], unit_system: str) -> str | None:
    """Why the first implausible part's values look like a unit slip, or None."""
    from src.vomp_wrapper.artifact import unit_scales

    e_scale, rho_scale = unit_scales(unit_system)
    for part in parts:
        modulus = float(part["youngs_modulus"]) / e_scale
        density = float(part["density"]) / rho_scale
        low, high = PLAUSIBLE_SPECIFIC_MODULUS
        if not (PLAUSIBLE_MODULUS_PA[0] <= modulus <= PLAUSIBLE_MODULUS_PA[1]
                and PLAUSIBLE_DENSITY_KG_M3[0] <= density <= PLAUSIBLE_DENSITY_KG_M3[1]
                and low <= modulus / density <= high):
            return (
                f"part {part['id']}: E {modulus:.3g} Pa with density {density:.3g} kg/m³ "
                f"is no solid's (E {PLAUSIBLE_MODULUS_PA[0]:.0e}–{PLAUSIBLE_MODULUS_PA[1]:.1e} Pa, "
                f"density {PLAUSIBLE_DENSITY_KG_M3[0]:.0f}–{PLAUSIBLE_DENSITY_KG_M3[1]:.1e} kg/m³, "
                f"E/density {low:.0f}–{high:.0e} m²/s²); check the units -- a factor "
                "of 10⁶ is the usual slip"
            )
    return None


def _current_parts(state: WorkflowState) -> bool:
    """Whether the per-part field was valued on the surface and parts as they are now.

    The surface alone does not decide it: the same body re-segmented -- PartField
    swapped for its CAD faces, say -- keeps its digest while its parts change, and
    a table judged by the surface kept showing, and approving, values for parts
    the labels no longer have.  The labels the field was built from must also be
    the ones the material step reads now (``_material_segments_npz``).
    """
    if not (state.material_parts and state.parts_artifact
            and (Path(state.parts_artifact) / "manifest.json").is_file()
            and state.parts_surface_digest == _surface_digest(state.surface_path)):
        return False
    npz = _material_segments_npz(state)
    if npz is None:
        return False
    if state.parts_segments_digest is None:
        # Valued before the labels were recorded (2026-10-20).  Those sessions
        # read the boundary stage's labels for PartField only, so the table is
        # still theirs exactly when that rule picks the same file today.
        return _bc_segmentation_method(state) != "cad_faces"
    return state.parts_segments_digest == _surface_digest(str(npz))


# The boundary-condition segmentations whose segments the material step takes as
# its parts: PartField's parts, and a CAD part's own B-rep faces.  Both are what
# the user saw, named and picked the supports by, and a per-part material that
# ignored them looked as if it had not noticed the segmentation change: switched
# to CAD faces, the material step went on reading its own PartField parts, and
# with them the same montage and the same cached answer (525c794ba2ae,
# 2026-10-20).  Shape-diameter segments are thickness patches, not parts, so with
# them the material step still segments the body with PartField itself.
BC_PART_SEGMENTATIONS = ("partfield", "cad_faces")


def _bc_segmentation_method(state: WorkflowState) -> str | None:
    """The method the boundary stage's saved labels were made with, if they are current.

    Read from the labels' own metadata, which is what the npz beside it holds;
    the state's setting is only the fallback, since a card can change it without
    segmenting again.
    """
    bc = state.root / "boundary_conditions"
    if not (state.segmentation_metadata_path and _segments_current(state, bc / "segments.npz")):
        return None
    try:
        method = json.loads(Path(state.segmentation_metadata_path).read_text(
            encoding="utf-8")).get("method")
    except (OSError, ValueError, AttributeError):
        method = None
    return method or state.segmentation_method


def _material_segments_npz(state: WorkflowState) -> Path | None:
    """The part labels the material step reads now, or None when it would segment first."""
    if _bc_segmentation_method(state) in BC_PART_SEGMENTATIONS:
        return state.root / "boundary_conditions" / "segments.npz"
    own = state.root / "materials" / "parts" / "segments.npz"
    return own if _segments_current(state, own) else None


def _segments_source(state: WorkflowState) -> dict[str, str]:
    """What a segmentation stamp records: the surface it was computed on, and its contents."""
    return {"surface": str(state.surface_path), "sha256": _surface_digest(state.surface_path)}


def _segments_current(state: WorkflowState, npz: Path) -> bool:
    """Whether saved part labels were computed on the surface as it is now.

    Judged by the surface's contents alone.  The stamp's path is where the
    surface was, and a session opened from another folder read as a new
    surface: the material step segmented again, and PartField does not return
    the same parts twice, so the per-part values already assigned no longer
    fitted the labels ("no material for parts [5]").
    """
    stamp = npz.with_name(npz.stem + ".source.json")
    try:
        recorded = json.loads(stamp.read_text(encoding="utf-8")).get("sha256")
        return npz.is_file() and bool(recorded) and recorded == _segments_source(state)["sha256"]
    except (OSError, ValueError, AttributeError):
        return False


def _texture_fields(state: WorkflowState) -> dict[str, Any]:
    """What a card offering to paint the surface shows: its colour, the painting, the painter."""
    textured = _current_texture(state)
    return {
        "surface_has_colour": _surface_has_colour(state.surface_path),
        "textured_surface": textured,
        "texture_reference": state.texture_reference if textured else None,
        "texture_views": str(Path(textured).with_name("views.png")) if textured else None,
        "texture_prompt": state.texture_prompt or state.geometry_description or "",
        "texture_seed": state.texture_seed,
        # Set when the painter crashed on this surface: the seed the painting was made with
        # instead, or the plain colour it was given (_paint_surface).
        "texture_fallback": state.texture_fallback if textured else None,
        # The image a surface was generated from can guide its painting.
        "texture_input_image": (
            state.input_asset.path
            if state.input_asset and state.geometry_mode == "image" else None
        ),
        "texture_status": texture_status().model_dump(),
    }


def _paint_surface(state: WorkflowState, answer: dict[str, Any], directory: Path) -> None:
    """Paint the current surface with Hunyuan3D-2.1 into ``directory``, as ``answer`` asks.

    The painted copy has the surface's vertices and faces, plus colour; it is
    recorded as ``textured_surface`` against the surface it was painted from.
    """
    state.texture_prompt = str(answer.get("texture_prompt", "")).strip()
    state.texture_seed = int(answer.get("texture_seed", state.texture_seed))
    image = answer.get("texture_image")
    args = {
        "surface": state.surface_path, "prompt": state.texture_prompt,
        "seed": state.texture_seed,
        "output": str(directory / "textured.glb"),
        "output_reference": str(directory / "reference.png"),
    }
    monitor = ToolMonitor(state.root, state.compute_env)
    if image:
        # An upload is staged into the session first, as geometry inputs are.
        source = Path(str(image)).expanduser().resolve()
        if state.root.resolve() not in source.parents:
            staged = directory / f"upload{source.suffix.lower() or '.png'}"
            state.tool_receipts.append(monitor.execute(
                "stage_asset", {"source": str(source), "output": str(staged)}))
            source = staged
        args["input"] = str(source)
    elif not state.texture_prompt:
        raise ValueError("describe the surface's look, or give a reference image, to paint it")
    fallback = _paint_crash_safe(state, monitor, args)
    if fallback is None or fallback["kind"] == "reseeded":
        # The painter refuses a painting that never reached the surface; this also catches one
        # served from a receipt made before it did (an acceptance run, 2026-10-10, recorded two
        # "painted" surfaces whose vertex colours were all black).
        if _surface_has_colour(args["output"]) is False:
            raise RuntimeError(
                "the painting failed: the painted surface carries no colour (every vertex the same "
                "plain value), so the views never reached it; paint again with another seed")
    # Six views of the painted surface, for the card to show beside its reference.
    state.tool_receipts.append(monitor.execute("render_surface", {
        "surface": args["output"], "output": str(directory / "views.png")}))
    state.textured_surface = args["output"]
    # A plain colour follows no reference (one drawn before a crash is not what the surface shows).
    state.texture_reference = (args["output_reference"]
                               if fallback is None or fallback["kind"] == "reseeded" else None)
    state.textured_from = state.surface_path
    state.textured_from_digest = _surface_digest(state.surface_path)
    state.texture_fallback = fallback


# Failures of the painter (a signal, or a Python exception), one JSON line each, kept with the receipts:
# the receipts and this file are not part of the checkpointed files, so a rewind or the background
# paint (whose state is thrown away) never forgets one.
PAINT_CRASHES = "paint_crashes.jsonl"
# Seeds tried on one surface (whatever the words) before it is given a plain colour: the seed
# asked for and one other.  A held-out run (2026-10-10) crashed the painter four times on one surface
# (twice at the same seed, then at another) and lost the run at that stop.
PAINT_CRASH_SEEDS = 2


def _paint_inputs(state: WorkflowState, args: dict[str, Any]) -> dict[str, Any]:
    """What a crash is counted against: the surface's contents and the reference image, if any.
    Not the words: the painter's crashes come from the mesh (paint_frame in the wrapper), so new
    words on a surface that crashed it would only crash it again."""
    return {"surface": _surface_digest(args["surface"]),
            "input": _digest(Path(args["input"])) if args.get("input") else None}


def _paint_crashes(state: WorkflowState, inputs: dict[str, Any]) -> dict[int, str]:
    """Seeds that crashed the painter on these inputs, and each crash's message."""
    crashes: dict[int, str] = {}
    try:
        lines = (state.root / "tool_receipts" / PAINT_CRASHES).read_text(encoding="utf-8").splitlines()
    except OSError:
        return crashes
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if {key: record.get(key) for key in inputs} == inputs:
            crashes[int(record["seed"])] = str(record.get("error") or "")
    return crashes


def _record_paint_crash(state: WorkflowState, inputs: dict[str, Any], seed: int, error: str) -> None:
    path = state.root / "tool_receipts" / PAINT_CRASHES
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**inputs, "seed": int(seed), "prompt": state.texture_prompt,
                                 "error": error[:600],
                                 "time": round(time.time(), 1)}) + "\n")


def _paint_crash_safe(state: WorkflowState, monitor: ToolMonitor, args: dict[str, Any]) -> dict[str, Any] | None:
    """Run the paint in ``args``; when the painter crashes, try one other seed, then a plain colour.

    A crash (killed by a signal, or "the surface cannot be painted") names nothing to change, and
    the same inputs crash again: they are never run twice (PAINT_CRASHES).  A Python exception
    from the painter is a paint failure too: a batch of 2026-10-12 met ``RuntimeError: max() ...
    numel() == 0`` in MeshRender.back_project on a long thin body, the same at every seed, and the
    run looped on it three times while it was re-raised.  After the asked seed and one more have
    failed on a surface (any words), the surface is given one flat colour read from the words
    (``plain_colour``) instead of an error, so the run goes on to the materials and the solve.
    Returns None for a normal painting, else what was done instead (the card's
    ``texture_fallback``).  Only a painter stopped from outside (SIGKILL: the out-of-memory killer
    or a user; SIGTERM, SIGINT) raises as before: that says nothing about the inputs.
    """
    inputs = _paint_inputs(state, args)
    asked = int(args["seed"])
    crashes = _paint_crashes(state, inputs)
    seeds = [asked + step for step in range(PAINT_CRASH_SEEDS * 4)]
    for seed in seeds:
        if len(crashes) >= PAINT_CRASH_SEEDS or any("cannot be painted" in error for error in crashes.values()):
            break
        if seed in crashes:
            continue
        try:
            state.tool_receipts.append(monitor.execute("paint_hunyuan3d_surface", {**args, "seed": seed}))
        except ToolFailed as failure:
            if stopped(failure.receipt) and "cannot be painted" not in str(failure):
                raise
            _record_paint_crash(state, inputs, seed, str(failure))
            crashes[seed] = str(failure)
            continue
        if seed == asked:
            return None
        return {"kind": "reseeded", "seed_asked": asked, "seed_used": seed,
                "crashed_seeds": sorted(crashes),
                "note": (f"the painter crashed at seed(s) {sorted(crashes)}; painted with seed {seed} "
                         "instead")}
    rgb, word = plain_colour(args.get("prompt") or "")
    receipt = monitor.execute("plain_colour_surface", {
        "surface": args["surface"], "rgb": ",".join(str(c) for c in rgb), "output": args["output"]})
    state.tool_receipts.append(receipt)
    last = crashes[max(crashes)] if crashes else ""
    return {"kind": "plain_colour", "seed_asked": asked, "crashed_seeds": sorted(crashes),
            "colour": list(rgb), "colour_word": word, "error": last[:600],
            "note": (f"NOT PAINTED: the painter crashed or failed on this surface (seeds {sorted(crashes)}); it was "
                     f"given one plain colour {tuple(rgb)} read from '{word}' in the texture words, so "
                     "the run can go on. Say so in the report: the body is shown in a flat colour, "
                     "and anything read from colour (per-part materials, VoMP) sees one colour")}


# Where the material card paints (its texture_surface answer).
MATERIAL_TEXTURE_DIR = Path("materials") / "texture"


def paint_material_texture_ahead(session_dir: Path, answer: dict[str, Any]) -> dict[str, Any]:
    """Paint the session's current surface now, as the material card's ``texture_surface`` would.

    For a front end that knows the material card will paint (the operator runner, from its plan)
    and has the GPU free before the card is reached: the card's own paint then finds the same
    tool calls in the receipts (same surface contents, words, seed and output paths) and is served
    from them at once.  Nothing here touches the graph or its checkpoints: the state is read from
    the session's snapshot and thrown away, so a surface changed in between, or other words, only
    means the card paints for itself, as it always did.
    """
    session_dir = Path(session_dir)
    workflow = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    recorded = workflow.get("session_dir")
    if recorded and Path(recorded) != session_dir:
        workflow = _rebased(workflow, str(Path(recorded)), str(session_dir))
    state = WorkflowState.model_validate(workflow)
    if _surface_has_colour(state.surface_path) is not False:
        return {"painted": False, "reason": "the surface already has colour"}
    _paint_surface(state, answer, state.root / MATERIAL_TEXTURE_DIR)
    return {"painted": True, "surface": state.surface_path, "textured_surface": state.textured_surface,
            "texture_fallback": state.texture_fallback}


def _colour_parts(state: WorkflowState, answer: dict[str, Any], directory: Path,
                  npz: Path, metadata_path: Path) -> None:
    """Each part of the per-part segmentation in one flat colour, in place of a painting.

    The painter draws one reference and spreads its colours over every face, so it places no
    colour by part: in a held-out run (2026-10-10) seven paints (about 170 s) never put the
    asked colours on the asked parts.  The parts are the ones the material is valued on
    (``_material_segments``, passed as ``npz`` and ``metadata_path``); ``per_part_colours`` is
    true (each part in its montage colour) or {"<part id>": "<colour words>"} (read as
    ``plain_colour`` reads a texture prompt), the rest in their montage colour.  Deterministic,
    no model and no seed.
    """
    wanted = answer.get("per_part_colours")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    present = sorted(int(segment["id"]) for segment in metadata["segments"])
    named = {int(key): str(value) for key, value in (wanted.items() if isinstance(wanted, dict) else [])}
    unknown = sorted(set(named) - set(present))
    if unknown:
        raise ValueError(f"per_part_colours names parts {unknown}; the parts are {present}")
    rgb = {key: list(plain_colour(words)[0]) for key, words in named.items()}
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "textured.glb"
    monitor = ToolMonitor(state.root, state.compute_env)
    state.tool_receipts.append(monitor.execute("plain_colour_surface", {
        "surface": state.surface_path, "rgb": "128,128,129", "output": str(output),
        "segments_npz": str(npz), "segments_json": str(metadata_path),
        "part_rgb": json.dumps({str(key): value for key, value in sorted(rgb.items())})}))
    state.tool_receipts.append(monitor.execute("render_surface", {
        "surface": str(output), "output": str(directory / "views.png")}))
    state.texture_prompt = "per-part colours"
    state.textured_surface = str(output)
    state.texture_reference = None
    state.textured_from = state.surface_path
    state.textured_from_digest = _surface_digest(state.surface_path)
    state.texture_fallback = {
        "kind": "part_colours", "asked": wanted, "parts": present,
        "colours": {str(key): value for key, value in sorted(rgb.items())},
        "note": ("each part of the per-part segmentation is one flat colour (no painting): the named "
                 "parts in their words' colour, the rest in the colour the part montage shows them in")}


def _draw_image(state: WorkflowState, answer: dict[str, Any], description: str) -> None:
    """Draw an image input with SDXL from the card's description, as ``draw_image`` asks.

    The description is the drawing's prompt -- the one text box says what the
    object is for every input -- and an answer that leaves it out draws what
    the card shows.  The card comes back with the drawing, cut out from its
    background, and as it stood -- the input kind, the generator and its
    settings -- so the drawing can be sent to any image generator, or drawn
    again.  Each drawing is its own file, named for what it was drawn from, so
    an earlier card still shows its own and an identical request is served from
    the cache.
    """
    status = sdxl_status()
    if not status.ready:
        raise ValueError(f"SDXL is not available: {status.summary}")
    prompt = (description or state.geometry_description or "").strip()
    if not prompt:
        raise ValueError("describe the geometry: SDXL draws the description")
    try:
        seed = int(answer.get("image_seed", state.drawn_image_seed))
    except (TypeError, ValueError) as exc:
        raise ValueError("image_seed must be a whole number") from exc
    if not 0 <= seed < 2 ** 32:
        raise ValueError("image_seed must be between 0 and 2**32 - 1")
    name = hashlib.sha256(prompt.encode()).hexdigest()[:10]
    directory = state.root / "inputs" / "sdxl"
    output = directory / f"{name}_s{seed}.png"
    raw = directory / f"{name}_s{seed}_drawn.png"
    receipt = ToolMonitor(state.root, state.compute_env).execute(
        "draw_sdxl_image",
        {"prompt": prompt, "seed": seed, "output": str(output), "output_raw": str(raw)},
    )
    state.tool_receipts.append(receipt)
    state.drawn_image, state.drawn_image_raw = str(output), str(raw)
    state.drawn_image_prompt, state.drawn_image_seed = prompt, seed
    # The words SDXL dropped, as draw_image.py measured them with the pipeline's tokenizer.
    report = _last_json_line(getattr(receipt, "stdout_path", None))
    state.drawn_image_dropped = str(report.get("dropped") or "")
    state.drawn_image_description_cut = bool(report.get("description_cut"))
    state.geometry_description = prompt
    state.geometry_draft = {**_card_draft(answer), "asset_kind": "image"}


def _last_json_line(path: str | None) -> dict[str, Any]:
    """The JSON object a tool printed last, {} when there is none."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines() if path else []
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _card_draft(answer: dict[str, Any]) -> dict[str, Any]:
    """What of a geometry answer the card shows again when it comes back to itself."""
    return {key: answer[key]
            for key in ("asset_kind", "generation_settings", "condition_on_text")
            if answer.get(key) is not None}


def _geometry_views(state: WorkflowState) -> Path | None:
    """Six orthographic views of the geometry itself, rendered on demand.

    Every stage downstream of generation reasons about the object that was
    actually built.  What the user supplied -- a photograph, a scribble, a line
    of text -- is evidence about the request, and a generator does not always
    build what was asked for, so the reference used to stand in for a mesh
    nobody had looked at.  Now the mesh is what gets looked at, whichever way it
    arrived, and the user's own words travel beside the reading rather than
    being replaced by it.

    Re-rendering is left to the tool monitor: its fingerprint covers the input
    mesh's bytes, so a regenerated or rescaled surface misses the cache and a
    view of a shape that no longer exists is never handed on.
    """
    if not state.surface_path or not Path(state.surface_path).exists():
        return None
    views = state.root / "geometry" / "views.png"
    state.tool_receipts.append(ToolMonitor(
        state.root, state.compute_env
    ).execute("render_surface", {
        "surface": state.surface_path, "output": str(views),
    }))
    return views


# Single precision carries ~1.19e-7 relative resolution, and the best relative
# residual an f32 solve reaches scales as ``C * eps / strain``.  Fitting C to three
# measured solves in this workflow gives 28.5 (strain 9.2e-8, stalled at 3.7e+01),
# 20.8 (strain 9.2e-5, stalled at 2.7e-02), and 8.6 (strain 2.6e-4, converged at
# 4.0e-03).  Ten separates the two that failed from the one that converged; it is
# an order-of-magnitude rule from three points, not a guarantee either way.
FLOAT32_EPSILON = 1.1920929e-7
FLOAT32_RESIDUAL_FACTOR = 10.0


def measured_extents(state: WorkflowState) -> dict[str, float]:
    """The uploaded mesh's own extents, in whatever units it was modelled in.

    Only meaningful for a surface someone supplied: a generated one lives in the
    generator's arbitrary [-1, 1] domain, where a measured extent says nothing
    about physical size.  ``inspect_surface`` has always recorded this and
    nothing read it back, so an uploaded mesh modelled at its true size was
    rescaled to a placeholder unless the user retyped the number.
    """
    if state.geometry_mode != "surface":
        return {}
    try:
        inspection = json.loads(
            (state.root / "geometry" / "inspection.json").read_text(encoding="utf-8")
        )
        extents = inspection["extents"]
    except (OSError, KeyError, TypeError, ValueError):
        return {}
    if not isinstance(extents, list) or len(extents) != 3:
        return {}
    measured = {}
    for axis, value in zip("xyz", extents):
        try:
            size = float(value)
        except (TypeError, ValueError):
            return {}
        if size > 0:
            measured[axis] = size
    return measured


def _strain_estimate(state: WorkflowState) -> float | None:
    """Rough peak strain, from the loads and stiffness already chosen.

    Strain is dimensionless, so it is exactly what decides whether single
    precision has any significant digits left for the displacement -- and it
    cannot be improved by a change of units or by scaling the system.
    """
    if state.material is None or not state.material.youngs_modulus:
        return None
    modulus = float(state.material.youngs_modulus)
    tractions = [
        max(abs(component) for component in item.traction)
        for item in (state.bc_proposal.neumann if state.bc_proposal else [])
        if item.traction
    ]
    peak = max(tractions, default=0.0) / modulus
    # A body force acts over the part, so its stress scale is force per volume
    # times a characteristic length.
    density = float(state.material.density or 0.0)
    effective_body = tuple(
        state.body_force[i] + density * state.acceleration[i] for i in range(3)
    )
    body = max(abs(component) for component in effective_body)
    if body and state.physical_extent:
        peak = max(peak, body * float(state.physical_extent) / modulus)
    return peak or None


def _matched_values(returned, regions, current, label: str) -> dict[str, Any]:
    """Put each inferred value on the region it was worked out for.

    Named values are matched by name and nothing else.  Falling back to position
    for a name that did not match is what made a mis-keyed answer indeterminate:
    the values were right and landed on the wrong regions, with nothing in the
    result to say so.  A region the answer leaves out keeps what it already has.

    A bare array is still accepted, in the order the regions were sent, because
    a provider that cannot emit an object has no other way to answer.
    """
    names = [item.name for item in regions]
    if isinstance(returned, dict):
        matched = {name: returned[name] for name in names if name in returned}
        # Nothing matched is evidence of a mis-keyed answer only when there was
        # something to match: a proposal with no loaded region has no names, and
        # the empty answer it gets back is the right one rather than a miss.
        if names and not matched:
            raise ValueError(
                f"{label} values name none of the approved regions "
                f"({', '.join(sorted(returned)) or 'nothing'}); expected "
                f"{', '.join(names)}"
            )
        return {name: matched.get(name, current[name]) for name in names}
    return {
        name: returned[index] if index < len(returned) else current[name]
        for index, name in enumerate(names)
    }


def _as_traction(value, area: float | None, name: str):
    """A traction vector, dividing a stated total force by the area it loads.

    ``area`` is the region's loaded (cosine-weighted) area where known: the
    solver loads a region along its one average normal, so a traction reaches
    the part as traction x loaded area, and dividing by the nominal area lost
    the force on every face tilted away from that normal.
    """
    force = getattr(value, "force", None)
    if force is None:
        return value
    if not area or area <= 0:
        raise ValueError(f"cannot spread a force over {name!r}: its area is unknown")
    return tuple(float(component) / area for component in force)


def _loaded_area(face_areas, face_normals) -> float:
    """How much of a region a traction actually loads.

    A Neumann region carries one normal, the average over its faces, and every
    point is loaded in proportion to max(0, n . n_avg) (nitsche_field.py:
    ``n_face`` and ``dphi_dn``).  A flat patch is loaded in full; a region
    wrapping round a curved or boxy part loses its tilted faces to the cosine
    and its back-facing ones entirely.  Matches resolve-bcs in workflow_tool.
    """
    import numpy as np

    mean = (face_normals * face_areas[:, None]).sum(0)
    mean = mean / max(float(np.linalg.norm(mean)), 1.0e-12)
    return float((face_areas * np.clip(face_normals @ mean, 0.0, None)).sum())


def _region_statistics(state: WorkflowState) -> list[dict[str, Any]]:
    """Compact physical-region statistics, as the value stage is shown them."""
    keys = ("name", "category", "faces", "area", "area_fraction", "bounds", "centroid",
            "mean_normal")
    return [{key: region[key] for key in keys if key in region}
            for region in _region_records(state)]


def _loaded_areas(state: WorkflowState) -> dict[str, float]:
    """Each region's loaded area, the one a total force is divided by."""
    return {region["name"]: region["loaded_area"] for region in _region_records(state)
            if region.get("loaded_area")}


def _region_records(state: WorkflowState) -> list[dict[str, Any]]:
    """The resolved regions' records, upgrading older previews in place."""
    root = state.root / "boundary_conditions"
    resolved_path = root / "resolved.json"
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    regions = payload.get("regions", [])
    if regions and any("loaded_area" not in region for region in regions):
        import numpy as np
        import trimesh

        mesh = trimesh.load(state.surface_path, force="mesh")
        masks = np.load(root / "resolved.npz")
        face_areas = np.asarray(mesh.area_faces)
        centers = np.asarray(mesh.triangles_center)
        normals = np.asarray(mesh.face_normals)
        triangles = np.asarray(mesh.triangles)
        total_area = float(face_areas.sum())
        for region in regions:
            mask = np.asarray(masks[region["key"]], dtype=bool)
            selected_areas = face_areas[mask]
            area = float(selected_areas.sum())
            points = triangles[mask].reshape(-1, 3)
            region.update({
                "area": area,
                "loaded_area": _loaded_area(selected_areas, normals[mask]),
                "area_fraction": area / max(total_area, 1.0e-12),
                "bounds": [points.min(0).tolist(), points.max(0).tolist()],
                "centroid": np.average(centers[mask], axis=0, weights=selected_areas).tolist(),
                "mean_normal": np.average(normals[mask], axis=0, weights=selected_areas).tolist(),
            })
        resolved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return regions


def _write_loads(state: WorkflowState) -> None:
    """The body force and acceleration ``build-case`` reads beside the material."""
    loads_json = state.root / "case" / "loads.json"
    loads_json.parent.mkdir(parents=True, exist_ok=True)
    loads_json.write_text(json.dumps({
        "body_force": list(state.body_force),
        "acceleration": list(state.acceleration),
    }, indent=2), encoding="utf-8")


def resume_command(graph, config, answer):
    """Resume the pending step with `answer`, addressed to that step by id.

    LangGraph reads a dict whose keys all look like interrupt ids as a map from
    id to answer -- and an empty dict passes that test vacuously, so `{}` ("keep
    the defaults") resumed nothing and the same card came back.  Naming the
    interrupt removes the ambiguity for every answer.
    """
    from langgraph.types import Command

    pending = graph.get_state(config).interrupts
    if len(pending) == 1:
        return Command(resume={pending[0].id: answer})
    return Command(resume=answer)


def failed_step(graph, config) -> bool:
    """Whether the thread stopped on a step that raised."""
    return any(task.error for task in graph.get_state(config).tasks)


def session_thread(session_id: str, session_dir: Path | str) -> dict[str, Any]:
    """The run config of a session's thread.  Naming the folder lets a copied or moved session
    work in its own folder rather than the one it was recorded in."""
    return {"configurable": {"thread_id": session_id, "session_dir": str(session_dir)}}


@contextlib.contextmanager
def open_session_graph(session_dir: Path, build=None):
    """The session's graph on its own checkpoints (``checkpoints.sqlite``), closed afterwards.

    How every front end opens a session -- the app's controller, the operator runner -- so they
    read and resume the same thread the same way.  ``build`` is the graph builder, the module's
    own by default (a caller that holds its own reference, as the controller does, passes it).
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(Path(session_dir) / "checkpoints.sqlite", check_same_thread=False)
    try:
        yield (build or build_graph)().compile(checkpointer=SqliteSaver(connection))
    finally:
        connection.close()


def reopen_card(graph, config, checkpoint_id: str | None = None):
    """Put a card back as it stood before it was answered; return the run that raises it.

    LangGraph keeps an answer beside the error it caused and replays that
    answer on every later attempt -- a fresh answer included -- so one rejected
    answer used to break its card for good.  Forking the card's checkpoint and
    running the fork raises the card again with nothing stored against it.
    Without ``checkpoint_id``, the newest checkpoint that waits on a card;
    None when there is none.
    """
    thread = config["configurable"]
    if checkpoint_id is None:
        checkpoint_id = next((
            snapshot.config["configurable"]["checkpoint_id"]
            for snapshot in graph.get_state_history(
                {"configurable": {"thread_id": thread["thread_id"]}})
            if any(task.interrupts for task in snapshot.tasks)
        ), None)
        if checkpoint_id is None:
            return None
    forked = graph.update_state({"configurable": {
        "thread_id": thread["thread_id"], "checkpoint_ns": "",
        "checkpoint_id": checkpoint_id,
    }}, None)
    # The caller's other keys -- the session folder -- ride along to the fork.
    return graph.invoke(None, config={"configurable": {**thread, **forked["configurable"]}})


# PartField parts for a material per part, when the boundary-condition stage
# did not already make some: the count the GVM benchmark was run with.
MATERIAL_PART_CLUSTERS = 6
# The text-to-CAD generators: each builds a CAD part from text alone, is
# checked for readiness the same way, and is segmented by its own B-rep faces.
CAD_GENERATORS = {"text2cad": "Text2CAD", "cadcoder": "CAD-Coder"}
# Neither knows object names -- "a hex nut" comes back a box -- so each needs its
# own revision of the description before it can build anything useful.
CAD_REVISION_NOTE = (
    "Text2CAD and CAD-Coder need a revised prompt: send the description with "
    '"revise_prompt": true first, review the rewrite, then send it to generate. '
    "They do not know object names, so an unrevised prompt usually comes back a box."
)


def needs_cad_revision(generator: str | None, description: str, cad_revision: dict | None) -> bool:
    """Whether text bound for a CAD generator has not been revised for it.

    True for a CAD generator unless the description is exactly what that
    generator's own revision last wrote.
    """
    if generator not in CAD_GENERATORS:
        return False
    cad_revision = cad_revision or {}
    return not (cad_revision.get("generator") == generator
                and (cad_revision.get("prompt") or "").strip() == description.strip())


def cad_generator_status(generator: str):
    return {"text2cad": text2cad_status, "cadcoder": cadcoder_status}[generator]()


def build_graph():
    """Build the graph lazily so importing tools does not require LangGraph."""

    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed; follow the API-backed agents installation "
            "steps in README.md"
        ) from exc

    def recover_api(state, call, source: str):
        """Record a failed specialist call for the owning card to show inline.

        Replacing the card with a recovery screen loses whatever the user had
        typed into it.  The card stays on screen instead, reports the failure
        under the button that triggered it, and stays editable so the next click
        retries with whatever the user changed.
        """
        specialist = SpecialistCalls(
            state.root, state.provider, state.model, state.vision_model,
            state.vision_provider, requested=state.generation_prompt,
            overview=state.geometry_overview)
        try:
            value, receipt = call(specialist)
            # An image-bearing stage bills two calls: the vision description and
            # the text answer.  Record both, not just the one returned.  A stage
            # that is only the vision call has already recorded itself and
            # returns no second receipt.
            state.api_receipts.extend(getattr(specialist, "receipts", ()))
            if receipt is not None:
                state.api_receipts.append(receipt)
            return value, receipt
        except (AgentsWrapperError, ValidationError) as exc:
            state.api_receipts.extend(getattr(specialist, "receipts", ()))
            message = f"{type(exc).__name__}: {_api_error_detail(exc)}"
            state.errors.append(message)
            state.api_error = message
            state.api_error_source = source
            raise _APIFailed(message) from exc

    def geometry_input(raw):
        state = _state(raw)
        answer = interrupt({
            "kind": "geometry_input",
            "message": (
                "Describe the analyzed geometry or upload a surface mesh, image, or scribble; "
                "an image can also be drawn from text with SDXL. "
                + CAD_REVISION_NOTE
            ),
            "description": state.geometry_description or "",
            # Which generator the text in the box was last revised for, if any.
            "cad_revision": {
                "generator": state.cad_revision_generator,
                "prompt": state.cad_revision_prompt,
            },
            "revise_required": sorted(CAD_GENERATORS),
            "fields": [
                "description", "asset_path", "asset_kind", "condition_on_text",
                "generator", "revise_prompt", "action", "image_seed",
                "use_drawn_image",
            ],
            # The optional text-to-CAD generators, offered only when ready.
            **{name: cad_generator_status(name).model_dump() for name in CAD_GENERATORS},
            # The optional image-to-3D generator with colour, likewise.
            "hunyuan3d": hunyuan3d_status().model_dump(),
            # SDXL, which draws an image input from text, likewise; and the
            # newest drawing, which any image generator can take.
            "sdxl": sdxl_status().model_dump(),
            "drawn_image": state.drawn_image,
            "drawn_image_raw": state.drawn_image_raw,
            # The description it was drawn from, which the box may since have left.
            "drawn_image_prompt": state.drawn_image_prompt,
            "image_seed": state.drawn_image_seed,
            # What of that prompt SDXL never read (77 tokens), so the card can say so.
            "drawn_image_dropped": state.drawn_image_dropped,
            "drawn_image_description_cut": state.drawn_image_description_cut,
            # The card as it stood when the image was drawn.
            **state.geometry_draft,
            "generator": state.generation.generator,
            "api_error": state.api_error,
            "api_error_source": state.api_error_source,
        })
        _clear_api_error(state)
        if not isinstance(answer, dict):
            raise ValueError("geometry input requires an object")
        description = str(answer.get("description", "")).strip()
        asset_path = answer.get("asset_path")
        asset_kind = answer.get("asset_kind")
        if answer.get("use_drawn_image"):
            # The newest drawing, as though it had been uploaded.
            if not state.drawn_image or not Path(state.drawn_image).is_file():
                raise ValueError("draw an image before using it")
            if asset_kind not in (None, "image") or asset_path not in (None, state.drawn_image):
                raise ValueError("use_drawn_image takes the drawing as the image input; "
                                 "leave asset_path and asset_kind out")
            asset_path, asset_kind = state.drawn_image, "image"
        generation_settings = dict(answer.get("generation_settings") or {})
        if "inference_steps" not in generation_settings:
            # The route's own default, whether or not other settings are given: a block holding
            # only a seed used to take the model's 16 for an image too, and 8 of 10 image
            # generations in the 2026-10-08 operator batches ran at 16 steps instead of 50.
            generator = answer.get("generator") or generation_settings.get("generator")
            generation_settings["inference_steps"] = (
                30 if generator == "hunyuan3d" else 50 if asset_kind == "image" else 16)
        if answer.get("generator") is not None:
            generation_settings = {**generation_settings, "generator": answer["generator"]}
        state.generation = GeometryGenerationSettings.model_validate(generation_settings)
        if str(answer.get("action", "")) == "draw_image":
            _draw_image(state, answer, description)
            return {**_dump(state), "pending": {"next": "geometry_input"}}
        if answer.get("action") not in (None, ""):
            raise ValueError("the geometry card's only action is draw_image")
        # A revision returns to this card too, and should find it as it was sent.
        state.geometry_draft = _card_draft(answer) if answer.get("revise_prompt") else {}
        label = CAD_GENERATORS.get(state.generation.generator)
        if label and asset_kind != "surface":
            if asset_path or asset_kind:
                raise ValueError(f"{label} generates from text only; clear the image or scribble")
            status = cad_generator_status(state.generation.generator)
            if not status.ready:
                raise ValueError(f"{label} is not ready: {status.summary}")
        if state.generation.generator == "hunyuan3d" and asset_kind != "surface":
            if asset_kind != "image" or not asset_path:
                raise ValueError("Hunyuan3D-2.1 generates from an image; upload one")
            status = hunyuan3d_status()
            if not status.ready:
                raise ValueError(f"Hunyuan3D-2.1 is not ready: {status.summary}")
        if not description and asset_kind != "surface":
            raise ValueError("geometry input requires a description unless a surface is uploaded")
        # Left empty when an upload carries no words: a stand-in phrase would be
        # prefilled back into the box on the next redraw, and would reach the
        # vision stage as if the user had claimed the mesh was one.
        state.geometry_description = description
        state.refined_prompt = state.geometry_description
        if asset_path:
            if asset_kind not in ("image", "scribble", "surface"):
                raise ValueError("asset_kind must be image, scribble, or surface")
            source = Path(asset_path).expanduser().resolve()
            suffix = source.suffix.lower() or ".png"
            if asset_kind == "surface" and suffix == ".stl":
                # STL has no shared vertices, so a closed part reads as open
                # triangles; staging welds it into a GLB, which keeps them and
                # which the viewer can draw.
                suffix = ".glb"
            target = state.root / ("geometry" if asset_kind == "surface" else "inputs") / f"original{suffix}"
            monitor = ToolMonitor(state.root, state.compute_env)
            receipt = monitor.execute("stage_asset", {"source": str(source), "output": str(target)})
            metadata = json.loads(target.with_suffix(target.suffix + ".json").read_text())
            state.input_asset = AssetRecord(
                source_name=metadata["source_name"], path=str(target), sha256=metadata["sha256"],
                mime_type=mimetypes.guess_type(target.name)[0],
            )
            state.tool_receipts.append(receipt)
            if asset_kind == "surface":
                inspection = state.root / "geometry" / "inspection.json"
                state.tool_receipts.append(monitor.execute(
                    "inspect_surface", {"surface": str(target), "output": str(inspection)}
                ))
                state.geometry_mode = "surface"
                state.surface_path = str(target)
                state.cad_faces_path = None
                # An upload was not generated from a prompt, so the only words
                # for it are the user's own -- and only if they wrote any.
                state.generation_prompt = description
                state.stage = Stage.GEOMETRY_PHYSICS
                return {**_dump(state), "pending": {"surface_ready": True}}
            state.geometry_mode = (
                "text_scribble"
                if asset_kind == "scribble" and answer.get("condition_on_text", True)
                else asset_kind
            )
        else:
            state.geometry_mode = "text"
        state.stage = Stage.GEOMETRY_INTAKE
        return {
            **_dump(state),
            "pending": {
                "surface_ready": False,
                "revise_prompt": bool(answer.get("revise_prompt", False)),
            },
        }

    def geometry_input_route(raw):
        pending = raw.get("pending", {})
        if pending.get("next") == "geometry_input":
            return "geometry_input"
        if pending.get("surface_ready"):
            return "texture_geometry"
        if not pending.get("revise_prompt"):
            return "generate_geometry"
        # Each generator reads a different kind of prompt, so each has its own revision.
        state = _state(raw)
        return {"text2cad": "refine_cad_geometry", "cadcoder": "refine_cadcoder_geometry"}.get(
            state.generation.generator, "refine_geometry")

    def texture_geometry(raw):
        """Optionally paint a surface that arrived without colour, before it is described.

        Painted here, the colour is in every later render -- the six views the
        geometry is described from, which scaling, boundary conditions and
        material all read -- and VoMP later needs no painting of its own.  The
        card appears only for a bare surface when Hunyuan3D-2.1 is installed;
        skipping it goes on exactly as before.
        """
        state = _state(raw)
        if (_surface_has_colour(state.surface_path) is not False
                or not hunyuan3d_status().ready):
            return {**_dump(state), "pending": {"next": "describe_geometry"}}
        views = _geometry_views(state)
        answer = interrupt({
            "kind": "geometry_texture",
            "message": "Optionally paint the surface before scaling, so later steps see its colours.",
            "surface_views": str(views) if views else None,
            **_texture_fields(state),
            "api_error": state.api_error,
            "api_error_source": state.api_error_source,
        })
        _clear_api_error(state)
        if not isinstance(answer, dict):
            raise ValueError("geometry texture requires an object")
        action = str(answer.get("action", "skip_texture"))
        if action == "texture_surface" and answer.get("per_part_colours"):
            raise ValueError("colours by part are given at the material card, once the parts are segmented")
        if action == "texture_surface":
            _paint_surface(state, answer, state.root / "geometry" / "texture")
            return {**_dump(state), "pending": {"next": "texture_geometry"}}
        if action == "use_texture":
            textured = _current_texture(state)
            if textured is None:
                raise ValueError("generate a texture before continuing with it")
            # Same vertices and faces, now coloured: the surface from here on.
            state.surface_path = textured
        elif action != "skip_texture":
            raise ValueError("action must be texture_surface, use_texture or skip_texture")
        return {**_dump(state), "pending": {"next": "describe_geometry"}}

    def texture_geometry_route(raw):
        return raw.get("pending", {}).get("next", "describe_geometry")

    def describe_geometry(raw):
        """Read the six-view render once, as soon as there is something to read.

        Scaling, material and a later revision all want to know what was built,
        and leaving each to ask for itself meant they shared one call only by
        producing byte-identical payloads -- and that nothing was read at all
        until somebody pressed Infer, so a freshly generated shape sat there
        undescribed.  One reading, taken here, kept on the state.

        It is advisory: every stage that uses it also works without it, so a
        failed call is dropped rather than stopping the workflow at a step with
        no card of its own to report on.
        """
        state = _state(raw)
        views = _geometry_views(state)
        if views is None or not views.exists():
            return _dump(state)
        try:
            described, _ = recover_api(
                state,
                lambda specialist: (
                    specialist.describe_image("geometry_overview", views), None
                ),
                "describe_geometry",
            )
        except (_APIFailed, OSError):
            # Advisory, so a failure here is dropped rather than stopping the
            # workflow at a step with no card of its own to report on.  OSError
            # covers a render that cannot be opened -- PIL raises a subclass of
            # it -- which is a reason to go without a description, not to die.
            _clear_api_error(state)
            return _dump(state)
        state.geometry_overview = described
        return _dump(state)

    def refine_geometry(raw):
        state = _state(raw)
        try:
            result, receipt = recover_api(
                state,
                lambda specialist: specialist.refine_geometry(
                    state.geometry_description or "", state.geometry_overview,
                ),
                "revise_geometry_prompt",
            )
        except _APIFailed:
            return _dump(state)
        state.refined_prompt = result.refined_prompt
        state.geometry_description = result.refined_prompt
        state.stage = Stage.GEOMETRY_INTAKE
        return _dump(state)

    def refine_cad_geometry(raw):
        state = _state(raw)
        try:
            result, receipt = recover_api(
                state,
                lambda specialist: specialist.refine_cad_geometry(state.geometry_description or ""),
                "revise_cad_prompt",
            )
        except _APIFailed:
            return _dump(state)
        state.refined_prompt = result.refined_prompt
        state.geometry_description = result.refined_prompt
        state.cad_revision_generator = "text2cad"
        state.cad_revision_prompt = result.refined_prompt
        state.stage = Stage.GEOMETRY_INTAKE
        return _dump(state)

    def refine_cadcoder_geometry(raw):
        state = _state(raw)
        try:
            result, receipt = recover_api(
                state,
                lambda specialist: specialist.refine_cadcoder_geometry(
                    state.geometry_description or ""),
                "revise_cadcoder_prompt",
            )
        except _APIFailed:
            return _dump(state)
        state.refined_prompt = result.refined_prompt
        state.geometry_description = result.refined_prompt
        state.cad_revision_generator = "cadcoder"
        state.cad_revision_prompt = result.refined_prompt
        state.stage = Stage.GEOMETRY_INTAKE
        return _dump(state)

    def generate_geometry(raw):
        state = _state(raw)
        state.stage = Stage.GEOMETRY_GENERATION
        mode = state.geometry_mode or "text"
        state.generation_prompt = state.refined_prompt or ""
        output = state.root / "geometry" / "generated.glb"
        args = {
            "output": str(output), "prompt": state.refined_prompt,
            "steps": state.generation.inference_steps,
            "seed": state.generation.seed,
            "faces": state.generation.faces,
            "octree_depth": state.generation.octree_depth,
            "init_octree_depth": state.generation.init_octree_depth,
            "prompt_confidence": state.generation.prompt_confidence,
            "scribble_confidence": state.generation.scribble_confidence,
            "erosion_radius": state.generation.erosion_radius,
        }
        if mode == "image":
            # The setting is carried for every mode, because the card keeps one
            # set of settings whatever is being generated -- but only the image
            # checkpoint does classifier-free guidance, and sending the number
            # to the others would have it recorded in a receipt that then reads
            # as though it had been applied.
            args["guidance_scale"] = state.generation.guidance_scale
        if state.input_asset:
            args["input"] = state.input_asset.path
        if state.generation.generator in CAD_GENERATORS:
            # Text only, and none of the diffusion or octree controls apply.
            tool = f"generate_{state.generation.generator}_text_surface"
            args = {key: args[key] for key in ("output", "prompt", "seed", "faces")}
        elif state.generation.generator == "hunyuan3d":
            # The image alone; the text is kept for the vision stage.
            tool = "generate_hunyuan3d_image_surface"
            args = {
                **{key: args[key] for key in ("output", "input", "seed", "faces", "steps")},
                "octree_resolution": state.generation.hunyuan3d_octree_resolution,
                "texture": state.generation.texture,
                "flashvdm": state.generation.hunyuan3d_flashvdm,
                "texture_size": state.generation.hunyuan3d_texture_size,
                "upscale": state.generation.hunyuan3d_upscale,
            }
        else:
            tool = f"generate_{mode}_surface"
        receipt = ToolMonitor(state.root, state.compute_env).execute(tool, args)
        state.tool_receipts.append(receipt)
        state.surface_path = str(output)
        labels = face_labels_path(output)
        state.cad_faces_path = (
            str(labels) if state.generation.generator in CAD_GENERATORS and labels.is_file() else None
        )
        metadata = state.root / "geometry" / "inspection.json"
        monitor = ToolMonitor(state.root, state.compute_env)
        state.tool_receipts.append(monitor.execute("inspect_surface", {"surface": str(output), "output": str(metadata)}))
        state.stage = Stage.GEOMETRY_PHYSICS
        return _dump(state)

    def geometry_physics(raw):
        state = _state(raw)
        measured = measured_extents(state)
        # A supplied mesh is usually modelled at its real size, so start from
        # that rather than from a placeholder that would silently resize it.
        # The axis is filled with the extent it was measured along: an extent
        # shown beside an empty axis read as settled, and whichever axis was
        # then picked was scaled to z's figure.
        if state.physical_extent is None and measured:
            axis = state.up_axis or ("z" if "z" in measured else max(measured, key=measured.get))
            state.up_axis, state.physical_extent = axis, measured.get(axis)
        answer = interrupt({
            "kind": "geometry_physics", "message": "Confirm physical scaling.",
            "fields": ["up_axis", "physical_extent", "unit_system"],
            # Segmentation will start from the CAD model's faces, not PartField.
            "cad_faces": bool(state.cad_faces_path),
            "unit_systems": list(UNIT_SYSTEMS),
            "unit_details": UNIT_SYSTEM_DETAILS,
            "up_axis": state.up_axis,
            "physical_extent": state.physical_extent,
            "unit_system": state.unit_system,
            "scale_description": state.scale_description,
            "measured_extents": measured,
            "api_error": state.api_error,
            "api_error_source": state.api_error_source,
        })
        _clear_api_error(state)
        if not isinstance(answer, dict):
            raise ValueError("physical scaling requires an object")
        # A field the answer leaves out keeps what the card showed.
        unit_system = str(answer.get("unit_system", state.unit_system) or "").strip()
        if unit_system not in UNIT_SYSTEMS:
            raise ValueError(f"unit_system must be one of: {', '.join(UNIT_SYSTEMS)}")
        state.unit_system = unit_system
        state.scale_description = str(
            answer.get("scale_description", state.scale_description) or "").strip()
        if str(answer.get("action", "apply_scaling")) == "infer_scale":
            try:
                inferred, receipt = recover_api(
                    state,
                    lambda specialist: specialist.infer_physical_scale(
                        state.geometry_description or "", state.scale_description,
                        UNIT_SYSTEM_DETAILS[unit_system]["length"],
                        state.geometry_overview,
                    ),
                    "infer_scale",
                )
            except _APIFailed:
                return {**_dump(state), "pending": {"next": "geometry_physics"}}
            state.up_axis = inferred.reference_axis
            state.physical_extent = inferred.physical_extent
            # Return to the same step so both values stay reviewable before scaling.
            return {**_dump(state), "pending": {"next": "geometry_physics"}}
        state.up_axis = answer.get("up_axis", state.up_axis)
        state.physical_extent = answer.get("physical_extent", state.physical_extent)
        if state.up_axis not in ("x", "y", "z") or not state.physical_extent:
            raise ValueError("up_axis, positive physical_extent, and unit_system are required")
        output = state.root / "geometry" / "physical.glb"
        receipt = ToolMonitor(state.root, state.compute_env).execute("scale_surface", {
            "surface": state.surface_path, "output": str(output), "axis": state.up_axis,
            "extent": state.physical_extent, "units": state.unit_system,
            "octree_depth": state.octree_depth,
        })
        state.tool_receipts.append(receipt)
        state.surface_path = str(output)
        state.stage = Stage.GEOMETRY_READY
        return {**_dump(state), "pending": {"next": "bc_input"}}

    def bc_input(raw):
        state = _state(raw)
        root = state.root / "boundary_conditions"
        partfield = partfield_status()
        # A CAD part comes already partitioned into its faces, which is the
        # segmentation to start from; otherwise prefer the learned backend.
        # Never change an existing/reviewed choice when the user returns here.
        cad_faces = bool(state.cad_faces_path) and Path(state.cad_faces_path).is_file()
        default_method = state.segmentation_method
        if default_method == "cad_faces" and not cad_faces:
            default_method = "shape_diameter"
        # This branch's segmentation, not any on disk: a rewind leaves a later
        # branch's labels in place for the viewer, and taking them for this one's
        # skipped segmenting on to a synthesis whose state named no labels.
        segmented = (state.segmentation_metadata_path is not None
                     and (root / "segments.json").is_file())
        if not segmented:
            if cad_faces:
                default_method = "cad_faces"
            elif partfield.ready:
                default_method = "partfield"
        current_settings = {
            "method": default_method,
            "num_clusters": state.segmentation_num_clusters,
            "num_rays": state.segmentation_num_rays,
            "smoothing": state.segmentation_smoothing,
            "cone_angle_deg": state.segmentation_cone_angle_deg,
        }
        answer = interrupt({
            "kind": "bc_input",
            "message": (
                "Describe supports and applied-load regions, preview them, then approve. "
                "Previewing BC regions automatically runs segmentation with the selected settings when needed."
            ),
            "description": state.bc_description or "",
            "segmentation": current_settings,
            "compute_env": state.compute_env,
            "partfield": partfield.model_dump(),
            "cad_faces": cad_faces,
            "has_segmentation": segmented and (root / "segments.glb").is_file(),
            "has_regions": state.bc_proposal is not None and (root / "resolved.glb").is_file(),
            # Regions synthesized from the description now in the box: the
            # proposal revise_regions may edit (synthesized.json holds it).
            "can_revise": (state.bc_synthesized is not None
                           and state.bc_synthesized_description == (state.bc_description or "")),
            "segment_description": state.segment_description,
            "preview_error": next(
                (item.removeprefix("BC preview: ") for item in reversed(state.errors)
                 if item.startswith("BC preview: ")),
                None,
            ),
            "api_error": state.api_error,
            "api_error_source": state.api_error_source,
        })
        _clear_api_error(state)
        # Keep the former CLI shorthand: a plain response means preview these regions.
        if isinstance(answer, str):
            answer = {"action": "preview_regions", "description": answer}
        if not isinstance(answer, dict):
            raise ValueError("boundary-condition input requires an action")
        action = str(answer.get("action", "")).strip()
        if action in ("preview_segmentation", "preview_regions"):
            state.errors = [item for item in state.errors if not item.startswith("BC preview: ")]
        if action == "approve_regions":
            if state.bc_proposal is None or not (root / "resolved.glb").is_file():
                raise ValueError("preview boundary-condition regions before approving them")
            return {**_dump(state), "pending": {"next": "bc_values"}}
        if action == "revise_regions":
            # A correction of what the synthesis made, never a substitute for it:
            # the synthesized proposal for the current wording must exist, and the
            # revision may change only the selectors of the regions it produced.
            synthesized = state.bc_synthesized
            if (synthesized is None
                    or state.bc_synthesized_description != (state.bc_description or "")):
                raise ValueError("revise_regions edits the regions synthesized from the current "
                                 "description: preview_regions first")
            revised = BCProposal.model_validate(answer.get("proposal") or {})
            for kind_name, made, sent in (("support", synthesized.dirichlet, revised.dirichlet),
                                          ("load", synthesized.neumann, revised.neumann)):
                if sorted(item.name for item in made) != sorted(item.name for item in sent):
                    raise ValueError(
                        f"revise_regions keeps the synthesized {kind_name} regions "
                        f"{sorted(item.name for item in made)}; it got "
                        f"{sorted(item.name for item in sent)}")
            # Everything but the selectors stays as synthesized.
            selectors = {item.name: item.selector
                         for item in [*revised.dirichlet, *revised.neumann]}
            proposal = BCProposal(
                dirichlet=[item.model_copy(update={"selector": selectors[item.name]})
                           for item in synthesized.dirichlet],
                neumann=[item.model_copy(update={"selector": selectors[item.name]})
                         for item in synthesized.neumann])
            state.errors = [item for item in state.errors if not item.startswith("BC preview: ")]
            return {**_dump(state), "pending": {
                "next": "revise_bcs", "proposal": proposal.model_dump(mode="json")}}
        if action not in ("preview_segmentation", "preview_regions", "revise_prompt"):
            raise ValueError(
                "action must be preview_segmentation, preview_regions, "
                "revise_prompt, revise_regions, or approve_regions"
            )

        # Settings the answer leaves out keep the card's, the method included:
        # a partial object used to drop back to the model's shape_diameter.
        settings = SegmentationSettings.model_validate(
            {**current_settings, **(answer.get("segmentation") or {})})
        if settings.method == "cad_faces" and not cad_faces:
            raise ValueError("CAD faces are only available for a surface generated by a CAD model")
        settings_changed = settings.model_dump() != current_settings
        state.segmentation_num_clusters = settings.num_clusters
        state.segmentation_method = settings.method
        state.segmentation_num_rays = settings.num_rays
        state.segmentation_smoothing = settings.smoothing
        state.segmentation_cone_angle_deg = settings.cone_angle_deg
        old_description = state.bc_description or ""
        description = str(answer.get("description", old_description)).strip()
        if description:
            state.bc_description = description
        if action == "revise_prompt":
            if not state.bc_description:
                raise ValueError("describe supports and loads before revising the prompt")
            return {**_dump(state), "pending": {"next": "refine_bcs"}}
        if action == "preview_segmentation":
            if settings_changed or description != old_description:
                state.bc_proposal = None
                state.bc_synthesized = None
            state.stage = Stage.BC_SEGMENTATION
            return {**_dump(state), "pending": {"next": "segment", "after_segment": "bc_input"}}
        if not state.bc_description:
            raise ValueError("describe the boundary-condition regions before previewing them")
        state.stage = Stage.BC_SEGMENTATION
        # Previewing regions is self-contained: create segmentation with the submitted
        # settings first unless an artifact for the unchanged settings already exists.
        if settings_changed or not segmented:
            return {**_dump(state), "pending": {"next": "segment", "after_segment": "synthesize_bcs"}}
        return {**_dump(state), "pending": {"next": "synthesize_bcs"}}

    def bc_input_route(raw):
        return raw.get("pending", {}).get("next", "bc_input")

    def refine_bcs(raw):
        state = _state(raw)
        previous = state.bc_description or ""
        try:
            result, receipt = recover_api(
                state,
                # What the object is: the vision stage's reading of the render.
                # The description box may hold a CAD generator's construction
                # text, which says how the part was built, not what it is.
                lambda specialist: specialist.refine_bc_prompt(
                    state.geometry_overview or state.geometry_description or "",
                    previous, state.segment_description,
                ),
                "revise_prompt",
            )
        except _APIFailed:
            return _dump(state)
        state.bc_description = result.refined_prompt
        if result.refined_prompt.strip() != previous.strip():
            # Any previewed regions were synthesized from the old wording.
            state.bc_proposal = None
        return _dump(state)

    def segment(raw):
        state = _state(raw)
        root = state.root / "boundary_conditions"
        monitor = ToolMonitor(state.root, state.compute_env)
        # Drop the montage before re-segmenting, not after.  It is derived from
        # the ids and colours this run is about to replace, and it doubles as
        # the signal that the slow local step is over and the vision call is
        # what is left: leaving the old one in place said "describing" while
        # segmentation was still running.
        (root / "segments.png").unlink(missing_ok=True)
        receipt = monitor.execute("segment_surface", {
            "method": state.segmentation_method,
            "surface": state.surface_path, "output_npz": str(root / "segments.npz"),
            "output_json": str(root / "segments.json"),
            "output_glb": str(root / "segments.glb"),
            "clusters": state.segmentation_num_clusters,
            "rays": state.segmentation_num_rays,
            "smoothing": state.segmentation_smoothing,
            "cone_angle_deg": state.segmentation_cone_angle_deg,
            "feature_cache": str(root / "partfield_features"),
            **({"cad_faces": state.cad_faces_path}
               if state.segmentation_method == "cad_faces" else {}),
        })
        state.tool_receipts.append(receipt)
        # Which surface these labels belong to, for the material step to reuse them.
        (root / "segments.source.json").write_text(
            json.dumps(_segments_source(state)), encoding="utf-8")
        state.segmentation_metadata_path = str(root / "segments.json")
        state.segmentation_montage_path = str(root / "segments.png")
        after_segment = raw.get("pending", {}).get("after_segment", "synthesize_bcs")
        state.segment_description = ""
        return {**_dump(state), "pending": {"after_segment": after_segment}}

    def segment_route(raw):
        after = raw.get("pending", {}).get("after_segment", "synthesize_bcs")
        # A preview goes on to read its own montage: the description is wanted
        # on the card, and having it already cached is what makes the later
        # region synthesis a single text call.
        return "describe_segments" if after == "bc_input" else after

    def describe_segments(raw):
        state = _state(raw)
        root = state.root / "boundary_conditions"
        montage = Path(state.segmentation_montage_path)
        if not montage.exists():
            state.tool_receipts.append(ToolMonitor(
                state.root, state.compute_env
            ).execute("render_segmentation", {
                "surface": state.surface_path,
                "segments_npz": str(root / "segments.npz"),
                "segments_json": str(root / "segments.json"),
                "output": str(montage),
            }))
        metadata = json.loads(
            Path(state.segmentation_metadata_path).read_text(encoding="utf-8")
        )
        try:
            described, _ = recover_api(
                state,
                lambda specialist: (
                    specialist.describe_image(
                        "bc_synthesis", montage, specialist._segment_legend(metadata)
                    ),
                    None,
                ),
                "describe_segments",
            )
        except _APIFailed:
            return {**_dump(state), "pending": {"next": "bc_input"}}
        state.segment_description = described
        return {**_dump(state), "pending": {"next": "bc_input"}}

    def synthesize_bcs(raw):
        state = _state(raw)
        metadata = json.loads(Path(state.segmentation_metadata_path).read_text(encoding="utf-8"))
        montage = Path(state.segmentation_montage_path)
        if not montage.exists():
            root = state.root / "boundary_conditions"
            state.tool_receipts.append(ToolMonitor(
                state.root, state.compute_env
            ).execute("render_segmentation", {
                "surface": state.surface_path,
                "segments_npz": str(root / "segments.npz"),
                "segments_json": str(root / "segments.json"),
                "output": str(montage),
            }))
        try:
            proposal, receipt = recover_api(
                state,
                lambda specialist: specialist.synthesize_bcs(
                    state.bc_description or "", metadata, montage
                ),
                "preview_regions",
            )
        except _APIFailed:
            return _dump(state)
        state.bc_proposal = proposal
        state.bc_synthesized = proposal
        state.bc_synthesized_description = state.bc_description or ""
        root = state.root / "boundary_conditions"
        (root / "synthesized.json").write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
        (root / "revision.json").unlink(missing_ok=True)
        bc_json = root / "proposal.json"
        bc_json.write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
        monitor = ToolMonitor(state.root, state.compute_env)
        try:
            state.tool_receipts.append(monitor.execute("resolve_bcs", {
                "surface": state.surface_path, "bc_json": str(bc_json),
                "segments_npz": str(root / "segments.npz"),
                "output_npz": str(root / "resolved.npz"), "output_json": str(root / "resolved.json"),
                "output_glb": str(root / "resolved.glb"),
            }))
        except RuntimeError as exc:
            state.bc_proposal = None
            state.errors.append(f"BC preview: {str(exc).split('; see ', 1)[0]}")
            state.stage = Stage.BC_SEGMENTATION
            return _dump(state)
        state.stage = Stage.BC_REVIEW
        return _dump(state)

    def revise_bcs(raw):
        """Resolve a revised proposal with the same tool a synthesized one goes through.

        The revision arrives validated by ``bc_input``; what changed is recorded
        beside the proposal (``revision.json``: per region, the synthesized and the
        revised selector), so a reader of the session sees which regions were not
        the synthesis's own.
        """
        state = _state(raw)
        proposal = BCProposal.model_validate(raw.get("pending", {}).get("proposal") or {})
        root = state.root / "boundary_conditions"
        made = {item.name: item for item in
                [*state.bc_synthesized.dirichlet, *state.bc_synthesized.neumann]}
        changes = [{"name": item.name,
                    "category": "dirichlet" if item in proposal.dirichlet else "neumann",
                    "synthesized": made[item.name].selector.model_dump(mode="json"),
                    "revised": item.selector.model_dump(mode="json")}
                   for item in [*proposal.dirichlet, *proposal.neumann]
                   if item.selector != made[item.name].selector]
        (root / "revision.json").write_text(json.dumps({
            "description": state.bc_synthesized_description, "regions": changes}, indent=2),
            encoding="utf-8")
        bc_json = root / "proposal.json"
        bc_json.write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
        state.bc_proposal = proposal
        monitor = ToolMonitor(state.root, state.compute_env)
        try:
            state.tool_receipts.append(monitor.execute("resolve_bcs", {
                "surface": state.surface_path, "bc_json": str(bc_json),
                "segments_npz": str(root / "segments.npz"),
                "output_npz": str(root / "resolved.npz"), "output_json": str(root / "resolved.json"),
                "output_glb": str(root / "resolved.glb"),
            }))
        except RuntimeError as exc:
            # The synthesized proposal stays, so the revision can be corrected.
            state.bc_proposal = None
            state.errors.append(f"BC preview: {str(exc).split('; see ', 1)[0]}")
            state.stage = Stage.BC_SEGMENTATION
            return _dump(state)
        state.stage = Stage.BC_REVIEW
        return _dump(state)

    def bc_values(raw):
        state = _state(raw)
        if state.bc_proposal is None or state.unit_system is None:
            raise ValueError("boundary-condition regions and a unit system are required")
        dirichlet_defaults: list[dict] = []
        for item in state.bc_proposal.dirichlet:
            components = item.components if item.components is not None else [0, 1, 2]
            if isinstance(item.value, tuple):
                selected = [item.value[index] for index in components]
                value = selected[0] if selected and len(set(selected)) == 1 else 0.0
            else:
                value = item.value
            dirichlet_defaults.append({
                "name": item.name, "components": components, "value": value,
            })
        current_values = {
            "dirichlet_values": {
                item["name"]: {"components": item["components"], "value": item["value"]}
                for item in dirichlet_defaults
            },
            "neumann_values": {
                item.name: list(item.traction or (0.0, 0.0, 0.0))
                for item in state.bc_proposal.neumann
            },
            "body_force": list(state.body_force),
            "acceleration": list(state.acceleration),
        }
        # Region areas turn a traction into the total force it applies, which is
        # the quantity a user actually reasons about when setting a load.  The
        # loaded area is the one that force is traction times.
        try:
            region_areas = {
                region["name"]: region["area"]
                for region in _region_statistics(state) if "area" in region
            }
            loaded_areas = _loaded_areas(state)
        except (OSError, KeyError, ValueError):
            region_areas, loaded_areas = {}, {}
        answer = interrupt({
            "kind": "bc_values",
            "message": "Set prescribed displacements, tractions, and body force.",
            "unit_system": state.unit_system,
            "units": UNIT_SYSTEM_DETAILS[state.unit_system],
            "dirichlet": dirichlet_defaults,
            "neumann": [
                {"name": item.name, "traction": list(item.traction or (0.0, 0.0, 0.0)),
                 "area": region_areas.get(item.name),
                 "loaded_area": loaded_areas.get(item.name)}
                for item in state.bc_proposal.neumann
            ],
            "body_force": list(state.body_force),
            "acceleration": list(state.acceleration),
            "value_description": state.bc_value_description,
            "api_error": state.api_error,
            "api_error_source": state.api_error_source,
        })
        _clear_api_error(state)
        if not isinstance(answer, dict):
            raise ValueError("boundary-condition values require an object")
        action = str(answer.get("action", "apply_values"))
        if action == "infer_values":
            description = str(answer.get("value_description", "")).strip()
            if not description:
                raise ValueError("describe the boundary-condition values to infer")
            state.bc_value_description = description
            statistics = _region_statistics(state)
            try:
                values, receipt = recover_api(
                    state,
                    lambda specialist: specialist.infer_bc_values(
                        state.bc_description or "", description, state.unit_system,
                        UNIT_SYSTEM_DETAILS[state.unit_system],
                        statistics,
                        bool(state.bc_proposal.neumann),
                    ),
                    "infer_values",
                )
            except _APIFailed:
                return {**_dump(state), "pending": {"next": "bc_values"}}
            areas = {region["name"]: region.get("area") for region in statistics}
            areas.update(loaded_areas)
            values = BCValuesInput.model_validate({
                "dirichlet_values": _matched_values(
                    values.dirichlet_values, state.bc_proposal.dirichlet,
                    current_values["dirichlet_values"], "Dirichlet",
                ),
                "neumann_values": {
                    name: _as_traction(value, areas.get(name), name)
                    for name, value in _matched_values(
                        values.neumann_values, state.bc_proposal.neumann,
                        current_values["neumann_values"], "Neumann",
                    ).items()
                },
                "body_force": values.body_force,
                "acceleration": values.acceleration,
            })
            next_node = "bc_values"
        elif action == "apply_values":
            supplied = {
                key: answer.get(key, current_values[key])
                for key in ("dirichlet_values", "neumann_values", "body_force", "acceleration")
            }
            # A region the answer leaves out keeps the value the card showed,
            # and a support given only some of its fields keeps the rest.
            for key in ("dirichlet_values", "neumann_values"):
                given = answer.get(key)
                if isinstance(given, dict):
                    merged = dict(current_values[key])
                    for name, value in given.items():
                        shown = merged.get(name)
                        merged[name] = ({**shown, **value}
                                        if isinstance(shown, dict) and isinstance(value, dict)
                                        else value)
                    supplied[key] = merged
            # A total force is spread over the area it loads, as an inferred one is.
            if isinstance(supplied["neumann_values"], dict):
                supplied["neumann_values"] = {
                    name: (_as_traction(NeumannForce.model_validate(value),
                                        loaded_areas.get(name) or region_areas.get(name), name)
                           if isinstance(value, dict) and "force" in value else value)
                    for name, value in supplied["neumann_values"].items()
                }
            values = BCValuesInput.model_validate(supplied)
            state.bc_value_description = str(
                answer.get("value_description", state.bc_value_description)
            ).strip()
            next_node = "material_input"
        else:
            raise ValueError("action must be infer_values or apply_values")
        for label, given, expected in (
                ("Dirichlet", values.dirichlet_values,
                 {item.name for item in state.bc_proposal.dirichlet}),
                ("Neumann", values.neumann_values,
                 {item.name for item in state.bc_proposal.neumann})):
            if set(given) != expected:
                unknown = ", ".join(sorted(set(given) - expected))
                raise ValueError(
                    f"{label} values must match all approved region names "
                    f"({', '.join(sorted(expected)) or 'none'})"
                    + (f"; not a region: {unknown}" if unknown else ""))
        for item in state.bc_proposal.dirichlet:
            supplied = values.dirichlet_values[item.name]
            item.components = supplied.components
            item.value = supplied.value
        for item in state.bc_proposal.neumann:
            item.traction = values.neumann_values[item.name]
        state.body_force = values.body_force
        state.acceleration = values.acceleration
        (state.root / "boundary_conditions" / "proposal.json").write_text(
            state.bc_proposal.model_dump_json(indent=2), encoding="utf-8"
        )
        state.stage = Stage.BC_VALUES
        return {**_dump(state), "pending": {"next": next_node}}

    def geometry_physics_route(raw):
        return raw.get("pending", {}).get("next", "bc_input")

    def material_input_route(raw):
        return raw.get("pending", {}).get("next", "shape_opt_input")

    def bc_values_route(raw):
        return raw.get("pending", {}).get("next", "material_input")

    def _material_segments(state, describe: bool = True):
        """The parts a material is assigned to, and a reading of what each is.

        The boundary-condition stage's parts when it made PartField parts or
        CAD faces (``BC_PART_SEGMENTATIONS``): the parts the supports and loads
        were named by, usually described already.  Otherwise the material step
        runs PartField itself.  The reading is the
        segment stage's own (``describe_image`` on the segment montage), so a
        repeat costs nothing; if it fails, parts are valued from the picture
        alone rather than not at all.
        """
        bc = state.root / "boundary_conditions"
        if _bc_segmentation_method(state) in BC_PART_SEGMENTATIONS:
            npz, metadata_path = bc / "segments.npz", Path(state.segmentation_metadata_path)
            montage = Path(state.segmentation_montage_path or bc / "segments.png")
            if state.segment_description or not describe:
                return npz, metadata_path, state.segment_description
        else:
            root = state.root / "materials" / "parts"
            npz, metadata_path, montage = (
                root / "segments.npz", root / "segments.json", root / "segments.png")
            if not _segments_current(state, npz):
                if not partfield_status().ready:
                    raise ValueError("assigning materials by part needs PartField, which is not ready")
                root.mkdir(parents=True, exist_ok=True)
                # Labels from another surface, and the picture of them, are dropped first.
                for stale in (npz, metadata_path, montage):
                    stale.unlink(missing_ok=True)
                state.tool_receipts.append(ToolMonitor(state.root, state.compute_env).execute(
                    "segment_surface", {
                        "method": "partfield", "surface": state.surface_path,
                        "output_npz": str(npz), "output_json": str(metadata_path),
                        "output_glb": str(root / "segments.glb"),
                        "clusters": MATERIAL_PART_CLUSTERS,
                        "feature_cache": str(bc / "partfield_features"),
                    }))
                npz.with_name(npz.stem + ".source.json").write_text(
                    json.dumps(_segments_source(state)), encoding="utf-8")
        if not describe:
            return npz, metadata_path, ""
        if not montage.exists():
            state.tool_receipts.append(ToolMonitor(state.root, state.compute_env).execute(
                "render_segmentation", {
                    "surface": state.surface_path, "segments_npz": str(npz),
                    "segments_json": str(metadata_path), "output": str(montage),
                }))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            described, _ = recover_api(
                state,
                lambda specialist: (specialist.describe_image(
                    "bc_synthesis", montage, specialist._segment_legend(metadata)), None),
                "infer_parts",
            )
        except _APIFailed:
            _clear_api_error(state)
            described = ""
        return npz, metadata_path, described

    def _build_part_field(state, parts, segments_npz):
        """Write the parts' values and the field made from them."""
        root = state.root / "materials" / "parts"
        root.mkdir(parents=True, exist_ok=True)
        materials = root / "materials.json"
        materials.write_text(json.dumps({"parts": parts}, indent=2), encoding="utf-8")
        field = root / "field"
        state.tool_receipts.append(ToolMonitor(state.root, state.compute_env).execute(
            "build_part_field", {
                "surface": state.surface_path, "segments_npz": str(segments_npz),
                "materials_json": str(materials), "unit_system": state.unit_system,
                "output": str(field),
            }))
        state.material_parts = parts
        state.parts_artifact = str(field)
        state.parts_surface_digest = _surface_digest(state.surface_path)
        # And the labels they were valued on, so a re-segmentation retires them.
        state.parts_segments_digest = _surface_digest(str(segments_npz))
        state.material = None
        state.material_inferred = True
        state.stage = Stage.MATERIAL_INFERENCE

    def material_input(raw):
        state = _state(raw)
        # What the card shows, which an answer that leaves a field out keeps.
        shown = {
            "material_mode": state.material.mode if state.material else (
                "vomp" if state.material_artifact or state.vomp_suggested else
                "parts" if _current_parts(state) else "uniform"
            ),
            "youngs_modulus": state.material.youngs_modulus if state.material else None,
            "poisson_ratio": state.material.poisson_ratio if state.material else None,
            "density": state.material.density if state.material else None,
            "yield_strength": state.material.yield_strength if state.material else None,
        }
        answer = interrupt({
            "kind": "material_input", "message": "Choose uniform parameters, a material per part, or a spatial VoMP material field.",
            "unit_system": state.unit_system,
            "units": UNIT_SYSTEM_DETAILS[state.unit_system],
            "fields": ["youngs_modulus", "poisson_ratio", "density", "yield_strength"],
            **shown,
            "material_artifact": state.material_artifact,
            "material_summary": _material_artifact_summary(state.material_artifact),
            # The per-part field: each part's values, the picture they were
            # read from, and the resulting field.
            **({
                "material_parts": state.material_parts,
                "parts_montage": state.parts_montage,
                "parts_artifact": state.parts_artifact,
                "parts_summary": _material_artifact_summary(state.parts_artifact),
            } if _current_parts(state) else {
                "material_parts": None, "parts_montage": None,
                "parts_artifact": None, "parts_summary": None,
            }),
            "material_description": state.material_description,
            "material_inferred": state.material_inferred,
            # Painting the surface before VoMP, which reads colour: whether it
            # has any, the painted copy and its reference image if made, and
            # whether the painter is ready (text needs the reference models).
            **_texture_fields(state),
            "api_error": state.api_error,
            "api_error_source": state.api_error_source,
        })
        _clear_api_error(state)
        if not isinstance(answer, dict):
            raise ValueError("material input requires an object")
        state.material_description = str(
            answer.get("material_description", state.material_description) or "").strip()
        action = str(answer.get("action", "apply_material"))
        if action == "texture_surface" and answer.get("per_part_colours"):
            npz, metadata_path, _ = _material_segments(state, describe=False)
            _colour_parts(state, answer, state.root / MATERIAL_TEXTURE_DIR, npz, metadata_path)
            return {**_dump(state), "pending": {"next": "material_input"}}
        if action == "texture_surface":
            _paint_surface(state, answer, state.root / MATERIAL_TEXTURE_DIR)
            return {**_dump(state), "pending": {"next": "material_input"}}
        if action == "infer_material":
            try:
                inferred, receipt = recover_api(
                    state,
                    lambda specialist: specialist.infer_material(
                        state.geometry_description or "", state.material_description,
                        UNIT_SYSTEM_DETAILS[state.unit_system]["stress_modulus_traction"],
                        state.geometry_overview,
                    ),
                    "infer_material",
                )
            except _APIFailed:
                return {**_dump(state), "pending": {"next": "material_input"}}
            if inferred.mode == "vomp":
                if _current_texture(state) is None and _surface_has_colour(state.surface_path) is False:
                    # A bare surface is painted first: open the VoMP card at
                    # its texture step rather than run VoMP on grey.
                    state.material, state.material_artifact = None, None
                    state.vomp_suggested = True
                    return {**_dump(state), "pending": {"next": "material_input"}}
                action = "infer_vomp"
            else:
                state.material = MaterialParameters(
                    unit_system=state.unit_system or "unspecified",
                    body_force=state.body_force,
                    youngs_modulus=inferred.youngs_modulus,
                    poisson_ratio=inferred.poisson_ratio,
                    density=inferred.density,
                    # Strength is supplied manually in the results viewer.
                    yield_strength=None,
                )
                # Inferred values can coincide exactly with the card's defaults,
                # so record that they came from the call rather than leaving the
                # user to guess whether it did anything.
                state.material_inferred = True
                return {**_dump(state), "pending": {"next": "material_input"}}
        if action == "infer_parts":
            # Read off colour, like VoMP: a bare surface is painted first.
            if _current_texture(state) is None and _surface_has_colour(state.surface_path) is False:
                raise ValueError("the surface has no colour; generate a texture before assigning materials by part")
            segments_npz, segments_json, description = _material_segments(state)
            root = state.root / "materials" / "parts"
            montage = root / "montage.png"
            state.tool_receipts.append(ToolMonitor(state.root, state.compute_env).execute(
                "render_part_montage", {
                    # The painted copy when there is one: same faces, with colour.
                    "surface": _current_texture(state) or state.surface_path,
                    "segments_npz": str(segments_npz), "segments_json": str(segments_json),
                    "output": str(montage),
                }))
            metadata = json.loads(segments_json.read_text(encoding="utf-8"))
            try:
                assigned, _ = recover_api(
                    state,
                    lambda specialist: specialist.assign_part_materials(
                        montage, metadata,
                        UNIT_SYSTEM_DETAILS[state.unit_system]["stress_modulus_traction"],
                        request=state.material_description, segments=description,
                    ),
                    "infer_parts",
                )
            except _APIFailed:
                return {**_dump(state), "pending": {"next": "material_input"}}
            present = {int(segment["id"]) for segment in metadata["segments"]}
            named = {part.id for part in assigned.parts}
            if named != present:
                raise ValueError(f"the part stage valued parts {sorted(named)} of {sorted(present)}")
            values = [part.model_dump() for part in assigned.parts]
            slip = _implausible_part(values, state.unit_system)
            if slip:
                # Shown under the button, like a failed call: nothing is built
                # from values off by orders of magnitude.
                state.errors.append(slip)
                state.api_error = f"The part stage's values look like a unit slip: {slip}"
                state.api_error_source = "infer_parts"
                return {**_dump(state), "pending": {"next": "material_input"}}
            state.parts_montage = str(montage)
            _build_part_field(state, values, segments_npz)
            return {**_dump(state), "pending": {"next": "material_input"}}
        if action == "apply_parts":
            # The user's edits to the per-part table: the same parts, new values.
            if not _current_parts(state):
                raise ValueError("assign materials by part on this surface before editing them")
            edited = PartMaterials.model_validate({"parts": answer.get("parts") or []})
            if {part.id for part in edited.parts} != {int(part["id"]) for part in state.material_parts}:
                raise ValueError("edited parts must match the assigned parts")
            slip = _implausible_part([part.model_dump() for part in edited.parts], state.unit_system)
            if slip:
                raise ValueError(slip)
            segments_npz, _, _ = _material_segments(state, describe=False)
            _build_part_field(state, [part.model_dump() for part in edited.parts], segments_npz)
            state.material_inferred = False
            return {**_dump(state), "pending": {"next": "material_input"}}
        if action == "infer_vomp":
            # VoMP reads colour: a bare surface is painted first (texture_surface).
            if _current_texture(state) is None and _surface_has_colour(state.surface_path) is False:
                raise ValueError("the surface has no colour; generate a texture before running VoMP")
            artifact = state.root / "materials" / "vomp"
            receipt = ToolMonitor(state.root, state.compute_env).execute(
                "infer_material_field", {
                    # The painted copy when there is one: same vertices, with colour.
                    "surface": _current_texture(state) or state.surface_path,
                    "artifact": str(artifact),
                    "cache": str(state.root.parent / ".runtime_cache" / "vomp"),
                    "unit_system": state.unit_system,
                },
            )
            state.tool_receipts.append(receipt)
            state.material_artifact = str(artifact)
            state.material = None
            state.material_inferred = True
            state.stage = Stage.MATERIAL_INFERENCE
            return {**_dump(state), "pending": {"next": "material_input"}}
        # An answer without a mode means the card's, unless it brings a uniform
        # material's own numbers; the fields it omits keep the card's values.
        if "material_mode" not in answer and {"youngs_modulus", "poisson_ratio"} & set(answer):
            answer = {**answer, "material_mode": "uniform"}
        answer = {**shown, **answer}
        mode = str(answer["material_mode"] or "uniform")
        if mode == "vomp":
            if not state.material_artifact:
                raise ValueError("run VoMP inference before approving adaptive material")
            state.material = MaterialParameters(
                unit_system=state.unit_system or "unspecified", mode="vomp",
                artifact=state.material_artifact, approved=True,
            )
        elif mode == "parts":
            if not _current_parts(state):
                raise ValueError("assign materials by part on this surface before approving them")
            state.material = MaterialParameters(
                unit_system=state.unit_system or "unspecified", mode="parts",
                artifact=state.parts_artifact, approved=True,
            )
        else:
            if answer["youngs_modulus"] is None or answer["poisson_ratio"] is None:
                raise ValueError("uniform material requires youngs_modulus and poisson_ratio")
            if any(state.acceleration) and answer.get("density") in (None, ""):
                raise ValueError("uniform material density is required when acceleration is nonzero")
            state.material = MaterialParameters(
                unit_system=state.unit_system or "unspecified", body_force=state.body_force,
                youngs_modulus=answer["youngs_modulus"], poisson_ratio=answer["poisson_ratio"],
                density=answer.get("density"),
                yield_strength=answer.get("yield_strength"),
            )
        state.material_inferred = mode == "vomp" or (mode == "parts" and state.material_inferred)
        material_json = state.root / "case" / "material.json"
        material_json.parent.mkdir(parents=True, exist_ok=True)
        material_json.write_text(state.material.model_dump_json(indent=2), encoding="utf-8")
        _write_loads(state)
        state.stage = Stage.MATERIAL_INTAKE
        return {**_dump(state), "pending": {"next": "shape_opt_input"}}

    def build_case(raw):
        state = _state(raw)
        case_root = state.root / "case"
        bc_json = state.root / "boundary_conditions" / "proposal.json"
        case = case_root / "case.json"
        if state.material is not None:
            # Written again from the state, which names the field artifact in
            # the folder the session runs from; the same bytes when it has not moved.
            (case_root / "material.json").write_text(
                state.material.model_dump_json(indent=2), encoding="utf-8")
        # So are the values and loads.  Only their own cards wrote these files,
        # and a rewind restores the state, not the files: a rewind to a settings
        # card of an earlier load case solved with the values applied last
        # (7cf4533ae018 solved 2353.6 N from a 784.5 N branch's card).
        if state.bc_proposal is not None:
            bc_json.write_text(state.bc_proposal.model_dump_json(indent=2), encoding="utf-8")
        _write_loads(state)
        receipt = ToolMonitor(state.root, state.compute_env).execute("build_case", {
            "surface": state.surface_path, "bc_json": str(bc_json),
            "segments_npz": str(state.root / "boundary_conditions" / "segments.npz"),
            "material_json": str(case_root / "material.json"),
            "loads_json": str(case_root / "loads.json"),
            "settings_json": str(case_root / "shape_opt.json"), "output": str(case),
            "transform_json": str(Path(state.surface_path).with_suffix(Path(state.surface_path).suffix + ".transform.json")),
            "solve_output": str(state.root / "solve"), "octree_depth": state.octree_depth,
        })
        state.tool_receipts.append(receipt)
        state.case_path = str(case)
        state.stage = Stage.FORWARD_PENDING
        return _dump(state)

    def shape_opt_input(raw):
        state = _state(raw)
        pending = raw.get("pending", {})
        strain = _strain_estimate(state)
        answer = interrupt({
            "kind": "shape_opt_settings",
            "message": (
                "The FEA solve failed. Review the log and edit settings before running again."
                if pending.get("solve_failed") else
                "Simulation complete. Review the results or edit settings and run again."
                if pending.get("solve_complete") else
                "Review ShapeOptTrimmed solver and optimization settings."
            ),
            "defaults": state.shape_opt.model_dump(mode="json"),
            "strain_estimate": strain,
            "strain_floor": (
                FLOAT32_RESIDUAL_FACTOR * FLOAT32_EPSILON / state.shape_opt.relative_tol
                if state.shape_opt.relative_tol else None
            ),
            "solve_error": pending.get("solve_error"),
            # A failed optimization: the iteration it failed at and the one the
            # viewer shows instead (run_forward's optimization_status.json).
            "optimization": pending.get("optimization"),
            "solve_log": pending.get("solve_log"),
            "solve_log_path": pending.get("solve_log_path"),
            "solve_complete": bool(pending.get("solve_complete")),
            # The card stays open after a solve so it can be run again; this is
            # the answer that ends the session instead.  Anything else runs.
            "finish_answer": {"action": "finish"},
        })
        if not isinstance(answer, dict):
            raise ValueError("ShapeOptTrimmed settings require an object")
        if answer.get("action") == "finish":
            state.stage = Stage.COMPLETE
            return {**_dump(state), "pending": {"next": "complete"}}
        # Settings the answer leaves out keep the card's, not the class defaults.
        state.shape_opt = ShapeOptSettings.model_validate(
            {**state.shape_opt.model_dump(mode="json"), **answer})
        if (state.material is not None and state.material.mode in FIELD_MODES
                and state.shape_opt.run_mode != "forward"):
            raise ValueError("spatial material fields currently support forward solves only")
        settings_json = state.root / "case" / "shape_opt.json"
        settings_json.parent.mkdir(parents=True, exist_ok=True)
        settings_json.write_text(state.shape_opt.model_dump_json(indent=2), encoding="utf-8")
        state.stage = Stage.SHAPE_OPT_SETTINGS
        return {**_dump(state), "pending": {"next": "build_case"}}

    def shape_opt_route(raw):
        return raw.get("pending", {}).get("next", "build_case")

    def solve(raw):
        state = _state(raw)
        solve_root = state.root / "solve"
        started = time.time()
        # The viewer assets already here, by modification time: what a failed
        # attempt did not write or rewrite is an earlier run's (file times trail
        # the clock by a tick, so they are not compared with ``started``).
        earlier_assets = {}
        for path in solve_root.glob("*.glb*"):
            try:
                earlier_assets[path] = path.stat().st_mtime_ns
            except OSError:
                pass
        try:
            receipt = ToolMonitor(state.root, state.compute_env).execute(
                "run_forward", {"case": state.case_path}
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:4000]
            # This run's log only: a solve that failed before writing one would
            # otherwise be reported with the last good run's log.
            logs = sorted(
                (path for path in solve_root.glob("*/log.txt")
                 if path.stat().st_mtime >= started - 1.0),
                key=lambda path: path.stat().st_mtime,
            )
            log_path = logs[-1] if logs else None
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path else ""
            if len(log_text) > 30000:
                log_text = "[Earlier log output omitted]\n" + log_text[-30000:]
            # What an earlier run left is not this case's result.  The coloured
            # surfaces at the top of solve/ from before this attempt are dropped
            # (they are built on demand), and the marker tells assets.py that
            # every surface from before this attempt started belongs to an
            # earlier run; whatever this one wrote before failing -- part of an
            # optimization -- stays, and so do the assets built from it: the
            # displacement asset run_forward rebuilt from the last iteration
            # that solved (scripts/workflow_tool.py, _OptimizationView).
            solve_root.mkdir(parents=True, exist_ok=True)
            for stale, written in earlier_assets.items():
                try:
                    if stale.is_file() and stale.stat().st_mtime_ns == written:
                        stale.unlink(missing_ok=True)
                except OSError:
                    pass
            # Which iteration a failed optimization shows, and which one failed.
            optimization = None
            if log_path is not None:
                try:
                    optimization = json.loads(
                        (log_path.parent / OPTIMIZATION_STATUS).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    optimization = None
            # A failed optimization says what to try next, wherever the error
            # is shown (the settings card, the failure record, the CLI).  The
            # monitor quotes one line of the tool's stderr, which may not be
            # the one run_forward appended its advice to, so it is added here
            # unless it came through -- sized from the step record run_forward
            # left in optimization_status.json when there is one.
            if (state.shape_opt.run_mode == "shape_optimization"
                    and OPTIMIZATION_FAILURE_ADVICE_MARKER not in message):
                step_size = (optimization or {}).get("step_size")
                rate = state.shape_opt.gd_lr
                if isinstance(step_size, dict) and step_size.get("gd_lr_auto"):
                    # The steps were taken at the rates gd_lr_auto chose; the last.
                    rate = step_size.get("gd_lr", rate)
                advice = optimization_failure_advice(rate, step_size)
                message = f"{message} {advice}"
            write_text_atomically(solve_root / SOLVE_FAILURE, json.dumps({
                "started": started, "error": message,
                "log": str(log_path) if log_path else None,
                **({"optimization": optimization} if optimization else {}),
            }, indent=2))
            state.errors.append(f"FEA solve: {message}")
            state.stage = Stage.FORWARD_PENDING
            return {**_dump(state), "pending": {
                "solve_failed": True, "solve_error": message,
                **({"optimization": optimization} if optimization else {}),
                "solve_log": log_text,
                "solve_log_path": str(log_path) if log_path else None,
            }}
        (solve_root / SOLVE_FAILURE).unlink(missing_ok=True)
        state.tool_receipts.append(receipt)
        state.stage = Stage.COMPLETE
        return {**_dump(state), "pending": {
            "solve_failed": False, "solve_complete": True,
        }}

    def solve_route(raw):
        return "shape_opt_input"

    graph = StateGraph(GraphState)
    graph.add_node("geometry_input", geometry_input)
    graph.add_node("refine_geometry", refine_geometry)
    graph.add_node("refine_cad_geometry", refine_cad_geometry)
    graph.add_node("refine_cadcoder_geometry", refine_cadcoder_geometry)
    graph.add_node("generate_geometry", generate_geometry)
    graph.add_node("texture_geometry", texture_geometry)
    graph.add_node("describe_geometry", describe_geometry)
    graph.add_node("geometry_physics", geometry_physics)
    graph.add_node("bc_input", bc_input)
    graph.add_node("refine_bcs", refine_bcs)
    graph.add_node("segment", segment)
    graph.add_node("describe_segments", describe_segments)
    graph.add_node("synthesize_bcs", synthesize_bcs)
    graph.add_node("revise_bcs", revise_bcs)
    graph.add_node("bc_values", bc_values)
    graph.add_node("material_input", material_input)
    graph.add_node("shape_opt_input", shape_opt_input)
    graph.add_node("build_case", build_case)
    graph.add_node("solve", solve)
    graph.add_edge(START, "geometry_input")
    graph.add_conditional_edges("geometry_input", geometry_input_route)
    graph.add_edge("refine_geometry", "geometry_input")
    graph.add_edge("refine_cad_geometry", "geometry_input")
    graph.add_edge("refine_cadcoder_geometry", "geometry_input")
    graph.add_edge("generate_geometry", "texture_geometry")
    graph.add_conditional_edges("texture_geometry", texture_geometry_route)
    graph.add_edge("describe_geometry", "geometry_physics")
    graph.add_conditional_edges("geometry_physics", geometry_physics_route)
    graph.add_conditional_edges("bc_input", bc_input_route)
    graph.add_edge("refine_bcs", "bc_input")
    graph.add_conditional_edges("segment", segment_route)
    graph.add_edge("describe_segments", "bc_input")
    graph.add_edge("synthesize_bcs", "bc_input")
    graph.add_edge("revise_bcs", "bc_input")
    graph.add_conditional_edges("bc_values", bc_values_route)
    graph.add_conditional_edges("material_input", material_input_route)
    graph.add_conditional_edges(
        "shape_opt_input", shape_opt_route,
        {"build_case": "build_case", "complete": END},
    )
    graph.add_edge("build_case", "solve")
    graph.add_conditional_edges(
        "solve", solve_route, {"shape_opt_input": "shape_opt_input"}
    )
    return graph
