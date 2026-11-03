"""Browsing, previewing and packing the files a conversation leaves behind.

A session's folder is the whole record of a run -- the geometry it started
from, the previews it drew, the case it built, and a solve tree of logs,
surfaces and coloured assets.  None of it was reachable without leaving the app
for a shell.  Everything here is the part of reaching it that has a decision in
it, kept out of the page so it can be tested: the page cannot be imported
outside a Streamlit runtime, and nothing in this module imports Streamlit.
"""

from __future__ import annotations

import contextlib
import io
import mimetypes
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# What a download may weigh.  Streamlit keeps a finished download in the
# server's own memory until the browser has taken it, so the cap is not about
# patience -- a session here runs to several gigabytes, and zipping one whole
# would put all of it in the process that is also running the conversation.
ARCHIVE_LIMIT = 1 << 30
# How much of a text file the preview window shows.  Solve logs and surface XML
# run to megabytes, and a window is for looking at the start of something, not
# for reading all of it.
PREVIEW_TEXT_LIMIT = 256 << 10

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
TEXT_SUFFIXES = {
    ".txt", ".log", ".json", ".jsonl", ".md", ".py", ".yaml", ".yml", ".csv",
    ".cfg", ".ini", ".toml", ".xml", ".vtu", ".pvd", ".sh", ".obj",
}


@dataclass(frozen=True)
class Entry:
    """One row of a listing, as the table shows it."""

    name: str
    path: Path
    is_dir: bool
    # Folders carry no size: the only honest one is a recursive walk, and doing
    # that per row would cost the whole tree to draw a folder nobody opened.
    size: int | None
    mtime: float


def listing(directory: Path) -> list[Entry]:
    """What is in this folder, folders first and then files, by name.

    One ``scandir`` and one ``stat`` per row, no recursion: the solve folders
    here hold a thousand entries apiece, and the listing is redrawn whenever
    anything is clicked.
    """
    entries: list[Entry] = []
    try:
        found = list(os.scandir(directory))
    except OSError:
        return []
    for item in found:
        try:
            is_dir = item.is_dir()
            info = item.stat()
        except OSError:
            # Vanished between the listing and the stat -- a solve writing into
            # the folder being looked at is enough to do it.
            continue
        entries.append(
            Entry(item.name, Path(item.path), is_dir,
                  None if is_dir else info.st_size, info.st_mtime)
        )
    return sorted(entries, key=lambda entry: (not entry.is_dir, entry.name.lower()))


def resolve_within(root: Path, relative: str) -> Path | None:
    """This path under the session, or nothing if it is not under it.

    The browser's position is held as a path relative to the session folder and
    re-resolved on every draw, so this is the one place that decides what the
    browser may reach.  A ``..``, an absolute path and a symlink pointing out
    of the session all come back as nothing, and so does one that has been
    deleted underneath the browser -- the caller falls back to the root either
    way, which is the right answer to both.
    """
    base = root.resolve()
    try:
        candidate = (base / relative).resolve()
        if not candidate.is_relative_to(base) or not candidate.exists():
            return None
    except OSError:
        return None
    return candidate


def breadcrumb(root: Path, here: Path) -> str:
    """Where the browser is, written from the session down."""
    try:
        rest = here.relative_to(root).parts
    except ValueError:
        rest = ()
    return " › ".join([root.name, *rest])


def human_size(size: float) -> str:
    """A byte count the way a person reads it."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def selection_bytes(paths: Iterable[Path]) -> int:
    """What this selection weighs, folders walked.

    Uncached on purpose: this is the number the size cap is decided on, and a
    remembered one would be decided on a folder as it used to be.  Walking two
    thousand files costs milliseconds; being wrong about it costs the process.
    """
    total = 0
    for path in paths:
        if path.is_dir():
            for folder, _, names in os.walk(path):
                for name in names:
                    with contextlib.suppress(OSError):
                        total += (Path(folder) / name).stat().st_size
        else:
            with contextlib.suppress(OSError):
                total += path.stat().st_size
    return total


def needs_archive(paths: Sequence[Path]) -> bool:
    """Whether this selection has to be packed rather than sent as it is.

    One plain file is itself; anything else -- several things, or a folder --
    only travels as an archive.
    """
    return len(paths) > 1 or any(path.is_dir() for path in paths)


def archive_name(root: Path, here: Path, paths: Sequence[Path]) -> str:
    """What to call the zip.

    One thing is named for itself.  A handful is named for where it came from,
    session included, because a folder called ``solve`` says nothing once it is
    in a downloads folder beside four others.
    """
    if len(paths) == 1:
        return f"{paths[0].name}.zip"
    if here == root:
        return f"{root.name}.zip"
    return f"{root.name}-{here.name}.zip"


def _members(base: Path, path: Path) -> Iterator[tuple[Path | None, str]]:
    """Every file this selection contributes, named as the browser showed it.

    Members are named relative to the folder being browsed, so a zip taken in
    the session root holds ``solve/log.txt`` and one taken inside ``solve``
    holds ``log.txt``.  Anything resolving outside that folder is dropped: the
    listing will not navigate through a symlink out of the session, and the
    archive must not carry one out either.  A folder with nothing left in it
    still gets an entry of its own, so an empty one survives the round trip
    instead of quietly disappearing.
    """
    root = base.resolve()

    def inside(candidate: Path) -> bool:
        try:
            return candidate.resolve().is_relative_to(root)
        except OSError:
            return False

    if not inside(path):
        return
    if path.is_file():
        yield path, path.relative_to(base).as_posix()
        return
    if not path.is_dir():
        return
    for folder, folders, names in os.walk(path):
        here = Path(folder)
        folders[:] = [name for name in folders if not (here / name).is_symlink()]
        kept = [here / name for name in names
                if (here / name).is_file() and inside(here / name)]
        for member in kept:
            yield member, member.relative_to(base).as_posix()
        if not kept and not folders:
            yield None, here.relative_to(base).as_posix() + "/"


def build_archive(base: Path, paths: Sequence[Path]) -> bytes:
    """Pack this selection, called only once a download has been asked for.

    Deflate at level one: the surface XML compresses hard at any setting, the
    GLBs are packed already, and past level one the time stops buying anything.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            for member, arcname in _members(base, path):
                if member is None:
                    folder = zipfile.ZipInfo(arcname)
                    folder.external_attr = (0o755 << 16) | 0x10
                    archive.writestr(folder, b"")
                    continue
                with contextlib.suppress(OSError):
                    archive.write(member, arcname)
    return buffer.getvalue()


def preview_kind(path: Path) -> str | None:
    """What the preview window can make of this file, if anything."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES:
        return "text"
    return None


def preview_text(path: Path) -> tuple[str, int]:
    """The head of a text file, and how long the whole of it is.

    The head rather than the tail: a surface or a case is read from the top,
    and a solve log's tail is what the Backend section is already for.
    """
    try:
        size = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(PREVIEW_TEXT_LIMIT), size
    except OSError as exc:
        return f"Could not read this file: {exc}", 0


def download_mime(path: Path) -> str:
    """What to tell the browser one file is."""
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def viewer_surface(surface: Path, previews: Path) -> Path | None:
    """The surface as the 3-D viewer can load it: itself if a GLB, else a GLB copy.

    The viewer reads GLB alone, and an uploaded OBJ or PLY is the analysis
    surface as uploaded until scaling writes ``physical.glb`` -- so until then
    the geometry was handed to the image widget, which cannot draw a mesh.  The
    copy lives under ``previews/`` and carries the surface's modification
    time: it is rebuilt when the two differ, which a rewind that puts back an
    older upload does as much as a newer upload.  None if the file cannot be
    read as a mesh.
    """
    if surface.suffix.lower() == ".glb":
        return surface
    target = previews / f"{surface.stem}{surface.suffix.lower().replace('.', '_')}.glb"
    try:
        modified = surface.stat().st_mtime_ns
        if not target.is_file() or target.stat().st_mtime_ns != modified:
            import trimesh

            mesh = trimesh.load(surface, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                return None
            target.parent.mkdir(parents=True, exist_ok=True)
            # Written aside and renamed, so a page drawn meanwhile never loads
            # half a file.
            partial = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            partial.write_bytes(mesh.export(file_type="glb"))
            os.utime(partial, ns=(modified, modified))
            os.replace(partial, target)
    except Exception:  # an unreadable upload is reported by the geometry step itself
        return None
    return target
