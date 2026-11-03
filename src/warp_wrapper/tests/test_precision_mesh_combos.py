"""Tests for dimension × mesh-type combinations in SINGLE precision (float32).

Covers:
  2D uniform, 3D uniform, 3D octree (no hanging), 3D octree (with hanging)

NOTE: Warp JIT-compiles different kernel variants for float32 vs float64
geometry.  Running both precisions in one process causes codegen conflicts.
Mixed precision tests live in test_precision_mesh_combos_mixed.py.

Run with:
    conda activate mechfoundry-warp117
    pytest src/warp_wrapper/tests/test_precision_mesh_combos.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest
import warp as wp

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.neoHookean import NeoHookean, OctreeNeoHookean
from tests.test_octree_mesh import make_octree_mesh_with_hanging, make_uniform_hex_mesh

wp.init()

_OVERRIDES = dict(
    use_mixed_precision=False,
    relative_tol=1e-5,
    absolute_tol=1e-6,
    max_iter=30,
    linear_solver="cg",
    cg_backend_assembled="amgx",
    ew_enabled=False,
    line_search_enabled=False,
    load_steps=1,
)

_RESIDUAL_TOL = 1e-4


class TestUniform2DSingle:
    CONFIG = "tests.examples.two_d_affine_shear.config"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_converges(self):
        s = NeoHookean(
            self.CONFIG,
            override_params={"output_root": self.tmp, "nx": 6, "ny": 4, **_OVERRIDES},
        )
        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"


class TestUniform3DSingle:
    CONFIG = "tests.examples.three_d_affine_shear.config"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_converges(self):
        s = NeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "nx": 3,
                "ny": 2,
                "nz": 2,
                **_OVERRIDES,
            },
        )
        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"


class TestOctreeUniform3DSingle:
    CONFIG = "tests.examples.three_d_affine_shear.config"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_converges(self):
        nodes, elements = make_uniform_hex_mesh(3, 3, 3, [0, 0, 0], [1.0, 0.6, 0.4])
        s = OctreeNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements},
                **_OVERRIDES,
            },
        )
        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"


class TestOctreeHanging3DSingle:
    CONFIG = "tests.examples.three_d_affine_shear.config"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_converges_and_constraints_hold(self):
        nodes, elements, edge_h, face_h = make_octree_mesh_with_hanging()
        mesh = {
            "nodes": nodes,
            "elements": elements,
            "edge_hanging": edge_h,
            "face_hanging": face_h,
        }
        s = OctreeNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "octree_mesh": mesh,
                **_OVERRIDES,
            },
        )
        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"

        disp = s.get_displacement()
        for row in edge_h:
            slave, m1, m2 = row
            err = np.linalg.norm(disp[slave] - 0.5 * (disp[m1] + disp[m2]))
            assert err < 1e-8, f"Edge hanging violated: slave={slave}, err={err:.3e}"
        for row in face_h:
            slave, m1, m2, m3, m4 = row
            err = np.linalg.norm(
                disp[slave] - 0.25 * (disp[m1] + disp[m2] + disp[m3] + disp[m4])
            )
            assert err < 1e-8, f"Face hanging violated: slave={slave}, err={err:.3e}"


class TestUniform3DNeumannSingle:
    """3D uniform mesh with Neumann BC to test lump_boundary_tractions_3d torch path."""

    CONFIG = "tests.examples.three_d_affine_shear.config"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_converges(self):
        s = NeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "nx": 3,
                "ny": 2,
                "nz": 2,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.zeros_like(x)): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, 1.0)): (
                        0.01,
                        0.0,
                        0.0,
                    ),
                },
                **_OVERRIDES,
            },
        )
        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"


class TestEFixedRegionSingle:
    """Verify _build_e_fixed_mask works with torch lambda."""

    CONFIG = "tests.examples.two_d_affine_shear.config"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_mask_built_correctly(self):
        s = NeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "nx": 4,
                "ny": 3,
                "e_fixed_region": lambda x, y: x < 0.25,
                **_OVERRIDES,
            },
        )
        assert s._e_free_mask is not None
        mask_np = s._e_free_mask.numpy()
        assert mask_np.shape[0] == s.e_space.node_count()
        assert mask_np.min() == 0.0  # some fixed
        assert mask_np.max() == 1.0  # some free
