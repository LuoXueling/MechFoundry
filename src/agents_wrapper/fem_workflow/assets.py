"""What a finished solve leaves behind, and the viewer assets built from it.

Everything here is shared by the two front ends.  The Streamlit app reads these
to draw the result viewer; the command line runs the same prefetch after a solve
so that opening that session in the app later finds its assets already built.
Neither imports the other, and nothing here imports Streamlit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .monitor import ToolMonitor

TENSOR_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")
# Left in solve/ by a solve that failed, holding when it started.  Every
# surface older than that was written by an earlier run, for a case that may
# since have changed, so it is not shown as this one's result.  A successful
# solve removes it.
from .runtime_files import SOLVE_FAILURE, solve_failure_in  # noqa: E402  (one definition, runtime_files)
# What each field offers in the viewport.  "magnitude" is the whole vector's
# length and is the one asset built eagerly; the rest are coloured on demand.
FIELD_COMPONENTS = {
    "displacement": ("magnitude", "x", "y", "z"),
    "strain": ("von_mises",) + TENSOR_COMPONENTS,
    "stress": ("von_mises",) + TENSOR_COMPONENTS,
}


def fully_written(path: Path) -> bool:
    """Whether this VTU has been closed off, rather than caught mid-write.

    The writer streams a surface into place rather than writing it aside and
    renaming, so a reader that arrives while a save is in progress finds a file
    with a header and no end.  That never mattered while these were only read
    after the run, but the viewer now follows an optimization as it writes, and
    a surface half on disk would reach it as an export failure.  The closing tag
    is what says the whole of it is there.
    """
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(size - 64, 0))
            return b"</VTKFile>" in handle.read()
    except OSError:
        return False


def _failed_since(solve_root: Path) -> float:
    """When the solve that last failed started, or 0 if the last one did not fail."""
    try:
        return float((solve_failure_in(solve_root) or {})["started"])
    except (ValueError, KeyError, TypeError):
        return 0.0


def _current(path: Path, since: float) -> bool:
    try:
        return path.is_file() and path.stat().st_mtime >= since and fully_written(path)
    except OSError:
        return False


def load_step_surfaces(solve_root: Path) -> list[Path]:
    """Every surface a solve wrote, in order, newest solve only.

    A run that ramps its load writes one of these per step, so the viewer can
    step through the deformation rather than only seeing where it ended up; a
    shape optimization writes one per saved iteration, and the same list is
    what its iteration slider walks.  None from before a failed solve.
    """
    runs = [d for d in solve_root.glob("*/vtu_surface") if d.is_dir()]
    if not runs:
        return []
    newest = max(runs, key=lambda d: d.stat().st_mtime)
    since = _failed_since(solve_root)
    return sorted(
        (p for p in newest.glob("surface_t*.vtu") if _current(p, since)),
        key=lambda p: p.name,
    )


def newest_surface_vtu(solve_root: Path) -> Path | None:
    steps = load_step_surfaces(solve_root)
    if steps:
        return steps[-1]
    since = _failed_since(solve_root)
    files = [p for p in solve_root.glob("*/vtu_surface/surface_t*.vtu") if _current(p, since)]
    return max(files, key=lambda p: p.stat().st_mtime, default=None)


def surface_point_fields(path: str) -> tuple[str, ...]:
    """The point fields a solve's surface carries, by name.

    Solves predating these fields have none of them, so the viewer offers only
    what a given run actually wrote.
    """
    try:
        text = Path(path).read_text(encoding="latin-1")
    except OSError:
        return ()
    block = re.search(r"<PointData.*?</PointData>", text, re.S)
    return tuple(re.findall(r'Name="([^"]+)"', block.group(0))) if block else ()


def field_asset_request(
    session_dir: str, vtu: str, field: str, component: str,
    deform_scale: float = 0.0, step: int = 0, total_steps: int = 1,
    reference_vtu: str = "",
) -> tuple[str, dict[str, Any]]:
    """Where one coloured asset lives, and the call that builds it.

    The monitor caches a tool on a fingerprint of its arguments, so building an
    asset before it is asked for only saves the click that asks if the two ask
    in exactly the same words.  Both routes come through here, so they cannot
    drift apart.
    """
    # Stamped whenever there is a ramp, step zero included: without that, the
    # first step of a ramp and a run that never ramped share a filename while
    # holding different surfaces.
    stamp = f"_s{step:02d}" if total_steps > 1 else ""
    scaled = f"_d{deform_scale:.6g}".replace(".", "p") if deform_scale else ""
    output = Path(session_dir) / "solve" / f"{field}_{component}{stamp}{scaled}.glb"
    arguments = {
        "vtu": vtu, "output": str(output), "field": field,
        # The exporter spells a vector's magnitude as "no component"; the
        # viewport spells it "magnitude" so it can sit in the picker beside the
        # named ones.  Translating here is what that costs.
        "component": "" if component == "magnitude" else component,
        "deform_scale": deform_scale,
        # Which surface's extremes the colours are stretched over.  Empty, or
        # this surface itself, means its own -- see the callers for why a ramp
        # names its last step instead.
        "value_range_from": reference_vtu,
        "hover_version": 1,
    }
    if component == "safety_factor":
        # Carry stress samples into the browser. Manual strength and the target
        # only change the presentation; neither requires another FEM solve.
        arguments["component"] = "von_mises"
    return str(output), arguments


def plan_solution_prefetch(
    session_dir: Path, steps: list[Path], carried: Iterable[str],
    field_components: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    """The field assets the viewer opens on, in the order it will want them.

    A solve leaves one surface per load step, each carrying up to eighteen
    field/component pairs; building every one would be a hundred exports nobody
    opens.  What is worth having ready is what a click actually lands on.  Every
    field opens on its first component, and a ramp draws that component for each
    of its steps at once -- which is the whole of the stall this removes.  The
    remaining displacement components follow, at the step the viewer starts on,
    because they are what people reach for next.
    """
    if not steps:
        return []
    carried = set(carried)
    last = len(steps) - 1
    wanted = [(field, components[0], index)
              for field, components in field_components.items() if field in carried
              for index in range(len(steps))]
    if "displacement" in carried:
        wanted += [("displacement", component, last)
                   for component in field_components["displacement"][1:]]
    # Named here exactly as the viewer names it, so what is built ahead is what
    # a click asks for: every step of a ramp is coloured against the last one.
    reference = str(steps[last]) if len(steps) > 1 else ""
    return [
        field_asset_request(str(session_dir), str(steps[index]), field, component,
                            0.0, index, len(steps), reference)[1]
        for field, component, index in wanted
    ]


def solution_prefetch_requests(session_dir: Path) -> list[dict[str, Any]]:
    """What the viewer will ask for, found from what the solve left on disk."""
    solve_root = session_dir / "solve"
    steps = load_step_surfaces(solve_root)
    if not steps:
        newest = newest_surface_vtu(solve_root)
        steps = [newest] if newest is not None else []
    if not steps:
        return []
    carried = surface_point_fields(str(steps[-1]))
    return plan_solution_prefetch(session_dir, steps, carried, FIELD_COMPONENTS)


def prefetch_solution_assets(session_dir: str, compute_env: str,
                             requests: list[dict[str, Any]]) -> None:
    """Build speculative assets one at a time, and never complain.

    Nobody asked for this work, so nothing it does may reach the caller.  An
    export that fails here fails the same way when it is clicked, and that is
    where it is worth reporting.
    """
    monitor = ToolMonitor(Path(session_dir), compute_env)
    for arguments in requests:
        try:
            monitor.execute("export_surface_field", arguments)
        except Exception:
            continue
