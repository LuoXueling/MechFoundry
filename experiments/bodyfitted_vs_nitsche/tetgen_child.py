"""Child processes (spawned, not forked) for compare_bodyfitted_nitsche.py: TetGen, TetGen -d, SuperLU.

A TetGen crash or hang must not take the comparison with it, and a ``fork`` of
a process that already runs JAX/Warp/Torch threads can deadlock silently.  So
TetGen runs in a ``spawn``-ed child that imports only numpy and tetgen (this
module), under a wall-clock timeout.
"""

import os

import numpy as np


def background_scale(bg_nodes):
    """The power of two that brings the background box diagonal D to <= 0.5.

    TetGen 1.6.0's ``randomsample`` compares a squared distance with the
    unsquared bounding-box diagonal; with ``-m`` and a background mesh of
    D >~ 4.6 it rejects every sample, ``searchtet`` stays NULL and the run
    segfaults in ``interpolatemeshsize`` (pyvista/tetgen#65).  Scaling the input
    surface, the background nodes and the sizes by an exact power of two
    k = 2^-ceil(log2(D / 0.5)) avoids it and is exact in binary floating point,
    so where it already worked the mesh is bit-identical after dividing by k.
    """
    diag = float(np.linalg.norm(np.ptp(np.asarray(bg_nodes, np.float64), axis=0)))
    return float(2.0 ** -np.ceil(np.log2(diag / 0.5))) if diag > 0 else 1.0


def tetrahedralize(conn, vertices, faces, switches, background):
    import shutil
    import tempfile

    import tetgen

    # TetGen writes tetgen-tmpfile_skipped.{node,face} into the working
    # directory when it meets a self-intersection; the child inherits the repo
    # root as cwd, so the TetGen-failure evidence variants left those files
    # there.  A private temp cwd keeps them out of the source tree.
    tmp = tempfile.mkdtemp(prefix="tg_mesh_")
    os.chdir(tmp)
    try:
        _tetrahedralize(conn, tetgen, vertices, faces, switches, background)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _tetrahedralize(conn, tetgen, vertices, faces, switches, background):
    k = 1.0
    if background is not None:
        k = background_scale(background[0])
    generator = tetgen.TetGen(np.asarray(vertices, np.float64) * k, np.asarray(faces, np.int32))
    if background is not None:
        bg_nodes, bg_tets, bg_size = background
        generator._tetgen.load_bgmesh_from_arrays(
            np.ascontiguousarray(np.asarray(bg_nodes, np.float64) * k), np.ascontiguousarray(bg_tets, np.int32),
            np.ascontiguousarray(np.asarray(bg_size, np.float64) * k))
    try:
        nodes, elements = generator.tetrahedralize(switches=switches)[:2]
        conn.send(("ok", np.asarray(nodes, np.float64) / k, np.asarray(elements, np.int64), k))
    except Exception as exc:  # noqa: BLE001
        conn.send(("error", f"{type(exc).__name__}: {exc}", None, k))
    conn.close()


def detect_intersections(vertices, faces, out_dir):
    """TetGen -d (exact predicates); its report goes to ``out_dir/tetgen_d.txt``."""
    import tetgen

    os.chdir(out_dir)
    fd = os.open(os.path.join(out_dir, "tetgen_d.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    try:
        tetgen.TetGen(np.asarray(vertices, np.float64), np.asarray(faces, np.int32)).tetrahedralize(switches="d")
    except Exception as exc:  # noqa: BLE001 - -d always ends in an exception (no tets)
        print(f"EXC {exc}", flush=True)


def direct_solve(conn, data, indices, indptr, shape, rhs):
    """Sparse direct solve (SciPy SuperLU) in a child process: a crash in
    SuperLU (it segfaulted on a 1.1 M-DOF system) then fails one attempt only."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    try:
        A = sp.csc_matrix((data, indices, indptr), shape=shape)
        conn.send(("ok", spla.spsolve(A, rhs, permc_spec="MMD_AT_PLUS_A")))
    except Exception as exc:  # noqa: BLE001
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    conn.close()
