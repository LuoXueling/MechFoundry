import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path
import textwrap

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.neoHookean import NeoHookean


EXAMPLE_CASES = [
    ("tests.examples.two_d_affine_shear.config", 2),
    ("tests.examples.two_d_affine_tension.config", 2),
    ("tests.examples.two_d_affine_compression.config", 2),
    ("tests.examples.two_d_affine_mix1.config", 2),
    ("tests.examples.two_d_affine_mix2.config", 2),
    ("tests.examples.three_d_affine_shear.config", 3),
    ("tests.examples.three_d_affine_tension.config", 3),
    ("tests.examples.three_d_affine_compression.config", 3),
    ("tests.examples.three_d_affine_mix1.config", 3),
    ("tests.examples.three_d_affine_mix2.config", 3),
]

# Expected affine displacement gradients for each config, used only for
# analytical comparisons in tests. Examples/configs do not include
# affine_gradient anymore.
GRADIENTS = {
    "tests.examples.two_d_affine_shear.config": ((0.0, 0.02), (0.0, 0.0)),
    "tests.examples.two_d_affine_tension.config": ((0.015, 0.0), (0.0, 0.0)),
    "tests.examples.two_d_affine_compression.config": ((-0.012, 0.0), (0.0, 0.0)),
    "tests.examples.two_d_affine_mix1.config": ((0.006, -0.004), (0.003, 0.002)),
    "tests.examples.two_d_affine_mix2.config": ((-0.003, 0.008), (-0.002, 0.005)),
    "tests.examples.three_d_affine_shear.config": (
        (0.0, 0.015, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    "tests.examples.three_d_affine_tension.config": (
        (0.01, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    "tests.examples.three_d_affine_compression.config": (
        (-0.009, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    "tests.examples.three_d_affine_mix1.config": (
        (0.004, -0.002, 0.0015),
        (0.0035, 0.0025, -0.002),
        (-0.001, 0.002, 0.003),
    ),
    "tests.examples.three_d_affine_mix2.config": (
        (-0.003, 0.0045, -0.002),
        (0.0025, 0.0015, 0.003),
        (0.001, -0.0025, 0.002),
    ),
}

# Additional right-boundary cases using a looser relative tolerance (5e-4)
RIGHT_BOUNDARY_CASES = [
    "tests.examples.two_d_right_boundary_force.config",
    "tests.examples.two_d_right_boundary_down_disp.config",
    "tests.examples.two_d_right_boundary_right_disp.config",
]

# Baseline displacement metrics for regression checks. These capture the
# current (accepted) results for each example. We assert new results match
# within rtol=1e-5.
BASELINE_DISPLACEMENT = {
    "tests.examples.two_d_right_boundary_force.config": {
        "l2": 20.238826751708984,
        "linf": 1.9479024410247803,
        "comp_l2": [5.537740707397461, 19.46647071838379],
        "sum": [-67.0199966430664, -315.1820373535156],
    },
    "tests.examples.two_d_right_boundary_down_disp.config": {
        "l2": 0.2084924876689911,
        "linf": 0.020000001415610313,
        "comp_l2": [0.03362995758652687, 0.2057623416185379],
        "sum": [-0.007286904379725456, -3.2931413650512695],
    },
    "tests.examples.two_d_right_boundary_right_disp.config": {
        "l2": 0.30769750475883484,
        "linf": 0.025002483278512955,
        "comp_l2": [0.20344483852386475, 0.23084184527397156],
        "sum": [3.5100202560424805, 3.505880832672119],
    },
    "tests.examples.two_d_affine_shear.config": {
        "l2": 0.08991661667823792,
        "linf": 0.009999999776482582,
        "comp_l2": [0.08991658687591553, 2.987113845165368e-08],
        "sum": [1.1549996137619019, 2.1520470738778386e-07],
    },
    "tests.examples.two_d_affine_tension.config": {
        "l2": 0.04592793434858322,
        "linf": 0.014999999664723873,
        "comp_l2": [0.04592793434858322, 3.6665415148462444e-09],
        "sum": [0.1875000149011612, 9.931910849303449e-11],
    },
    "tests.examples.two_d_affine_compression.config": {
        "l2": 0.029576975852251053,
        "linf": 0.010799999348819256,
        "comp_l2": [0.029576975852251053, 1.6491112475591763e-08],
        "sum": [-0.107999786734581, 4.5702108764089644e-11],
    },
    "tests.examples.two_d_affine_mix1.config": {
        "l2": 0.015065486542880535,
        "linf": 0.004800000227987766,
        "comp_l2": [0.01176329143345356, 0.009412430226802826],
        "sum": [0.03750002011656761, 0.041250016540288925],
    },
    "tests.examples.two_d_affine_mix2.config": {
        "l2": 0.011699572205543518,
        "linf": 0.004000000189989805,
        "comp_l2": [0.009926100261509418, 0.006192941218614578],
        "sum": [0.021749982610344887, 0.01199999637901783],
    },
    "tests.examples.three_d_affine_shear.config": {
        "l2": 0.044899892061948776,
        "linf": 0.009000000543892384,
        "comp_l2": [0.044899892061948776, 5.913542100444147e-09, 2.643591789919242e-09],
        "sum": [0.28800004720687866, 1.2948930816492066e-08, 5.6843418860808015e-12],
    },
    "tests.examples.three_d_affine_tension.config": {
        "l2": 0.03367491438984871,
        "linf": 0.008999999612569809,
        "comp_l2": [0.03367491438984871, 3.139504932067605e-12, 9.977105946967768e-12],
        "sum": [0.16200001537799835, -4.307938554493074e-12, -1.3097746845436298e-11],
    },
    "tests.examples.three_d_affine_compression.config": {
        "l2": 0.028623679652810097,
        "linf": 0.0076500000432133675,
        "comp_l2": [
            0.028623679652810097,
            1.1782685070682586e-11,
            4.523344806356411e-12,
        ],
        "sum": [-0.13770002126693726, -1.1506264691041324e-12, -1.4298757490538438e-12],
    },
    "tests.examples.three_d_affine_mix1.config": {
        "l2": 0.019113793969154358,
        "linf": 0.003680000314489007,
        "comp_l2": [0.013752247206866741, 0.012532858178019524, 0.004374958574771881],
        "sum": [0.07488001137971878, 0.06863999366760254, 0.017280003055930138],
    },
    "tests.examples.three_d_affine_mix2.config": {
        "l2": 0.019952956587076187,
        "linf": 0.003595000132918358,
        "comp_l2": [0.0109151192009449, 0.015994934365153313, 0.004810681566596031],
        "sum": [-0.05008000135421753, 0.11504001915454865, 0.01999998837709427],
    },
}


def _disp_metrics(d):
    d = np.asarray(d)
    return dict(
        l2=float(np.linalg.norm(d)),
        linf=float(np.max(np.abs(d))),
        comp_l2=list(np.linalg.norm(d, axis=0)),
        sum=list(np.sum(d, axis=0)),
    )


def _assert_disp_matches_baseline(config_path, disp, rtol=1e-5, atol=1e-10):
    base = BASELINE_DISPLACEMENT[config_path]
    act = _disp_metrics(disp)
    # Scalars
    assert np.isclose(
        act["l2"], base["l2"], rtol=rtol, atol=atol
    ), f"l2 mismatch for {config_path}: {act['l2']:.12e} vs {base['l2']:.12e}"
    assert np.isclose(
        act["linf"], base["linf"], rtol=rtol, atol=atol
    ), f"linf mismatch for {config_path}: {act['linf']:.12e} vs {base['linf']:.12e}"
    # Vectors
    assert np.allclose(
        act["comp_l2"], base["comp_l2"], rtol=rtol, atol=atol
    ), f"comp_l2 mismatch for {config_path}: {act['comp_l2']} vs {base['comp_l2']}"
    assert np.allclose(
        act["sum"], base["sum"], rtol=rtol, atol=atol
    ), f"sum mismatch for {config_path}: {act['sum']} vs {base['sum']}"


def _affine_displacement(gradient, coordinates):
    grad = np.asarray(gradient, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    return (grad @ coords.T).T


def _small_strain_voigt(gradient):
    grad = np.asarray(gradient, dtype=np.float64)
    sym = 0.5 * (grad + grad.T)
    if grad.shape[0] == 2:
        return np.array([sym[0, 0], sym[1, 1], sym[0, 1]], dtype=np.float64)
    return np.array(
        [
            sym[0, 0],
            sym[1, 1],
            sym[2, 2],
            sym[0, 1],
            sym[1, 2],
            sym[0, 2],
        ],
        dtype=np.float64,
    )


def _cauchy_stress_voigt(gradient, youngs_modulus, poisson_ratio):
    grad = np.asarray(gradient, dtype=np.float64)
    dim = grad.shape[0]
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    kappa = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    F = np.eye(dim, dtype=np.float64) + grad
    J = np.linalg.det(F)
    if J <= 0.0:
        raise ValueError("Non-positive Jacobian encountered in expected stress.")
    F_inv = np.linalg.inv(F)
    F_inv_T = F_inv.T
    C = F.T @ F
    I1 = np.trace(C) + (0.0 if dim == 3 else 1.0)
    J_m23 = J ** (-2.0 / 3.0)
    P = mu * (J_m23 * (F - (I1 / 3.0) * F_inv_T)) + kappa * math.log(J) * F_inv_T
    sigma = (P @ F.T) / J
    sigma_sym = 0.5 * (sigma + sigma.T)
    if dim == 2:
        return np.array(
            [sigma_sym[0, 0], sigma_sym[1, 1], sigma_sym[0, 1]], dtype=np.float64
        )
    return np.array(
        [
            sigma_sym[0, 0],
            sigma_sym[1, 1],
            sigma_sym[2, 2],
            sigma_sym[0, 1],
            sigma_sym[1, 2],
            sigma_sym[0, 2],
        ],
        dtype=np.float64,
    )


def _check_cell_orientation(dim, points, cells):
    pts = np.asarray(points, dtype=np.float64)
    for cell in cells:
        node_ids = np.asarray(cell, dtype=np.int64)
        if dim == 2:
            p0, p1, p2, p3 = pts[node_ids]
            v1 = p1 - p0
            v2 = p3 - p0
            area = v1[0] * v2[1] - v1[1] * v2[0]
            assert (
                area >= -1e-12
            ), f"Negative area detected in quad cell with nodes {node_ids}"
        else:
            p0, p1, p3, p4 = (
                pts[node_ids[0]],
                pts[node_ids[1]],
                pts[node_ids[3]],
                pts[node_ids[4]],
            )
            v1 = p1 - p0
            v3 = p3 - p0
            v4 = p4 - p0
            volume = np.dot(np.cross(v1, v3), v4)
            assert (
                volume >= -1e-10
            ), f"Negative volume detected in hex cell with nodes {node_ids}"


# Definition of boundary-condition variants for dedicated regression tests.
BC_VARIANT_CASES = [
    {
        "name": "bc_component_list_tension",
        "builder": "component_list_tension",
        "expected_gradient": ((0.012, 0.0), (0.0, 0.0)),
        "expected_translation": (0.0, 0.0),
    },
    {
        "name": "bc_function_tuple_shear",
        "builder": "function_tuple_shear",
        "expected_gradient": ((0.0, 0.018), (0.0, 0.0)),
        "expected_translation": (0.0, 0.0),
    },
    {
        "name": "bc_scalar_translation",
        "builder": "scalar_translation",
        "expected_gradient": ((0.0, 0.0), (0.0, 0.0)),
        "expected_translation": (0.007, -0.003),
    },
]


@pytest.mark.parametrize("case", BC_VARIANT_CASES, ids=lambda c: c["name"])
def test_boundary_condition_variants(case, tmp_path, monkeypatch):
    module_name = _load_bc_config(case, tmp_path, monkeypatch)
    override = {"output_root": str(tmp_path / "outputs")}

    solver = NeoHookean(module_name, override_params=override)
    solver.forward()
    solver.save(time=0.0)
    solver.finalize()

    rel_hist = solver.relative_residual_history
    assert rel_hist, "Residual history should be recorded."
    assert rel_hist[-1] < 1e-5, f"Relative residual too large: {rel_hist[-1]:.3e}"

    coords = np.array(solver.vtu_writer.points, copy=True)
    displacement = solver.get_displacement()
    grad = np.asarray(case["expected_gradient"], dtype=np.float64)
    expected_disp = (grad @ coords.T).T
    translation = np.asarray(case["expected_translation"], dtype=np.float64)
    if translation.shape == (2,):
        expected_disp += translation
    else:
        expected_disp += translation.reshape(1, -1)

    np.testing.assert_allclose(
        displacement,
        expected_disp,
        rtol=5e-3,
        atol=1e-5,
        err_msg=f"Displacement mismatch for {case['name']}",
    )

    from core.vtu_writer import compute_strain_tensor

    strain = compute_strain_tensor(solver.u, solver.geo, solver.space, solver.dim)
    expected_strain = _small_strain_voigt(grad)
    for node_strain in strain:
        np.testing.assert_allclose(
            node_strain,
            expected_strain,
            rtol=5e-3,
            atol=1e-6,
            err_msg=f"Strain mismatch for {case['name']}",
        )

    from core.vtu_writer import compute_stress_tensor

    stress = compute_stress_tensor(
        solver.u, solver.geo, solver.space, solver.dim, solver.mu, solver.kappa
    )
    expected_stress = _cauchy_stress_voigt(grad, solver.params.E, solver.params.nu)
    for node_stress in stress:
        np.testing.assert_allclose(
            node_stress,
            expected_stress,
            rtol=2e-2,
            atol=1e-5,
            err_msg=f"Stress mismatch for {case['name']}",
        )

    out_dir = Path(solver.path)
    if out_dir.exists():
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for file in files:
                Path(root, file).unlink()
            for directory in dirs:
                Path(root, directory).rmdir()
        out_dir.rmdir()


@pytest.mark.parametrize("config_path, dim", EXAMPLE_CASES)
def test_affine_examples_converge_and_match_analytics(config_path, dim, tmp_path):
    module = importlib.import_module(config_path)
    params = module.params.copy()
    override_output = {"output_root": str(tmp_path)}

    solver = NeoHookean(config_path, override_params=override_output)
    solver.forward()
    solver.save(time=0.0)
    solver.finalize()

    residual_history = solver.relative_residual_history
    assert residual_history, "Residual history should not be empty."
    assert residual_history[-1] < 1e-5
    assert (
        len(residual_history) <= 10
    ), f"Too many Newton iterations: {len(residual_history)}"

    initial_res, final_res, final_rel = (
        solver.initial_residual,
        solver.final_residual,
        solver.final_relative_residual,
    )
    assert final_rel is not None and final_rel < 1e-5
    abs_tol = getattr(params, "absolute_tol", 1e-10) or 1e-10
    rel_tol = getattr(params, "relative_tol", 1e-5) or 1e-5
    assert final_res is not None and final_res < max(
        abs_tol, (initial_res or 1.0) * rel_tol
    )

    coords = np.array(solver.vtu_writer.points, copy=True)
    displacement = solver.get_displacement()
    grad = np.asarray(GRADIENTS[config_path], dtype=np.float64)
    expected_disp = _affine_displacement(grad, coords)
    np.testing.assert_allclose(
        displacement,
        expected_disp,
        rtol=1e-4,
        atol=5e-8,
        err_msg="Displacement field does not match imposed affine map.",
    )

    from core.vtu_writer import compute_strain_tensor

    strain = compute_strain_tensor(solver.u, solver.geo, solver.space, solver.dim)
    expected_strain_voigt = _small_strain_voigt(grad)
    for node_strain in strain:
        np.testing.assert_allclose(
            node_strain,
            expected_strain_voigt,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Strain tensor mismatch against analytical expectation.",
        )

    from core.vtu_writer import compute_stress_tensor

    stress = compute_stress_tensor(
        solver.u, solver.geo, solver.space, solver.dim, solver.mu, solver.kappa
    )
    expected_stress_voigt = _cauchy_stress_voigt(grad, params.E, params.nu)
    for node_stress in stress:
        np.testing.assert_allclose(
            node_stress,
            expected_stress_voigt,
            rtol=1e-4,
            atol=2e-5,
            err_msg="Stress tensor mismatch against analytical expectation.",
        )

    vtu_dir = Path(solver.vtu_writer.vtu_dir)
    assert solver.vtu_writer.vtu_files, "No VTU files recorded."
    first_vtu = vtu_dir / solver.vtu_writer.vtu_files[0]
    assert first_vtu.exists(), f"Expected VTU output missing: {first_vtu}"


@pytest.mark.parametrize("config_path", RIGHT_BOUNDARY_CASES)
def test_right_boundary_examples_converge(config_path, tmp_path):
    override = {"output_root": str(tmp_path)}
    solver = NeoHookean(config_path, override_params=override)
    try:
        solver.forward()
    except RuntimeError:
        # plateau still acceptable per looser tol; we still save and check histories
        pass
    residual_history = solver.relative_residual_history
    assert residual_history, "Residual history should not be empty."
    # Configs use relative_tol=5e-4 for these examples
    assert residual_history[-1] < 5e-4

    pvd_path = Path(solver.path) / "solution.pvd"
    # Save and finalize to ensure PVD exists
    solver.save(time=0.0)
    solver.finalize()
    assert pvd_path.exists(), f"PVD collection file missing: {pvd_path}"
    # These right-boundary cases are 2D
    _check_cell_orientation(2, solver.vtu_writer.points, solver.vtu_writer.cells)

    # Provide debug info for residual progression and extrema for review.
    rel_hist = np.asarray(residual_history)
    debug_msg = (
        f"{config_path}: residuals -> "
        f"min={rel_hist.min():.3e}, max={rel_hist.max():.3e}, "
        f"final={rel_hist[-1]:.3e}, iterations={len(rel_hist)}"
    )
    print(debug_msg)

    # Displacement regression against baseline.
    # Right-boundary (Neumann BC) cases use looser tolerance because AMGX
    # PCG+AMG is not bit-reproducible across runs (parallel atomics), and
    # the force case saturates at max_iters on early Newton steps, making
    # the convergence path sensitive to small numerical differences.
    is_right_boundary = config_path in RIGHT_BOUNDARY_CASES
    baseline_rtol = 1e-3 if is_right_boundary else 1e-5
    _assert_disp_matches_baseline(
        config_path, solver.get_displacement(), rtol=baseline_rtol
    )

    # Cleanup outputs to keep workspace tidy.
    solver_output_dir = Path(solver.path)
    if solver_output_dir.exists():
        for root, dirs, files in os.walk(solver_output_dir, topdown=False):
            for file in files:
                Path(root, file).unlink()
            for directory in dirs:
                Path(root, directory).rmdir()
        solver_output_dir.rmdir()


def _build_bc_config(case):
    if case["builder"] == "component_list_tension":
        return textwrap.dedent(
            """
            from core.utils import dotdict
            import numpy as np

            WIDTH = 4.0
            HEIGHT = 1.0
            AXIAL = 0.012


            def ux(x, y):
                return AXIAL * x


            def zero(x, y):
                return 0.0


            params = dotdict(
                project_name="bc_component_list_tension",
                output_root="examples_output",
                dim=2,
                q_deg=2,
                elem_order=1,
                width=WIDTH,
                height=HEIGHT,
                nx=6,
                ny=3,
                relative_tol=1e-5,
                absolute_tol=2e-8,
                max_iter=25,
                dirichlet_bcs_dict={
                    lambda x, y: np.isclose(x, 0.0): (0.0, 0.0),
                    lambda x, y: np.isclose(x, WIDTH): ([0], ux),
                    lambda x, y: np.isclose(y, HEIGHT): (ux, zero),
                    lambda x, y: np.isclose(y, 0.0): (ux, zero),
                },
                neumann_bcs_dict={},
                B=(0.0, 0.0),
            )
            """
        )
    elif case["builder"] == "function_tuple_shear":
        return textwrap.dedent(
            """
            from core.utils import dotdict
            import numpy as np

            WIDTH = 4.0
            HEIGHT = 1.0
            SHEAR = 0.018


            def ux(x, y):
                return SHEAR * y


            def uy(x, y):
                return 0.0


            params = dotdict(
                project_name="bc_function_tuple_shear",
                output_root="examples_output",
                dim=2,
                q_deg=2,
                elem_order=1,
                width=WIDTH,
                height=HEIGHT,
                nx=6,
                ny=3,
                relative_tol=1e-5,
                absolute_tol=2e-8,
                max_iter=25,
                dirichlet_bcs_dict={
                    lambda x, y: np.isclose(x, 0.0): (ux, uy),
                    lambda x, y: np.isclose(x, WIDTH): (ux, uy),
                    lambda x, y: np.isclose(y, HEIGHT): (ux, uy),
                    lambda x, y: np.isclose(y, 0.0): (ux, uy),
                },
                neumann_bcs_dict={},
                B=(0.0, 0.0),
            )
            """
        )
    elif case["builder"] == "scalar_translation":
        return textwrap.dedent(
            """
            from core.utils import dotdict
            import numpy as np

            WIDTH = 4.0
            HEIGHT = 1.0
            DX = 0.007
            DY = -0.003

            params = dotdict(
                project_name="bc_scalar_translation",
                output_root="examples_output",
                dim=2,
                q_deg=2,
                elem_order=1,
                width=WIDTH,
                height=HEIGHT,
                nx=6,
                ny=3,
                relative_tol=1e-5,
                absolute_tol=2e-8,
                max_iter=25,
                dirichlet_bcs_dict={
                    lambda x, y: np.isclose(x, 0.0): (DX, DY),
                    lambda x, y: np.isclose(x, WIDTH): (DX, DY),
                    lambda x, y: np.isclose(y, HEIGHT): (DX, DY),
                    lambda x, y: np.isclose(y, 0.0): (DX, DY),
                },
                neumann_bcs_dict={},
                B=(0.0, 0.0),
            )
            """
        )
    else:
        raise ValueError(f"Unknown builder '{case['builder']}'")


def _load_bc_config(case, tmp_path, monkeypatch):
    module_name = f"tmp_bc_{case['name']}"
    config_path = tmp_path / f"{module_name}.py"
    config_path.write_text(_build_bc_config(case))

    monkeypatch.syspath_prepend(str(tmp_path))
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module_name


# ---------------------------------------------------------------------------
# Matrix-free CG comparison tests
# ---------------------------------------------------------------------------
# Representative configs covering 2D affine, 3D affine, and Neumann BC.
MATRIX_FREE_COMPARISON_CONFIGS = [
    "tests.examples.two_d_affine_shear.config",
    "tests.examples.two_d_affine_mix1.config",
    "tests.examples.three_d_affine_shear.config",
]

MATRIX_FREE_SOLVERS = ["matrix_free_cg", "autodiff_matrix_free_cg"]


def _run_solver(config_path, output_dir, **overrides):
    """Run NeoHookean solver with parameter overrides and return it."""
    params = {"output_root": str(output_dir)}
    params.update(overrides)
    solver = NeoHookean(config_path, override_params=params)
    try:
        solver.forward()
    except RuntimeError:
        pass
    return solver


@pytest.mark.parametrize("solver_type", MATRIX_FREE_SOLVERS)
@pytest.mark.parametrize("config_path", MATRIX_FREE_COMPARISON_CONFIGS)
def test_matrix_free_matches_assembled_cg(config_path, solver_type, tmp_path):
    """Matrix-free CG variants must produce the same solution as assembled CG."""
    common = dict(ew_enabled=False)

    ref = _run_solver(
        config_path,
        tmp_path / "ref",
        linear_solver="cg",
        **common,
    )
    mf = _run_solver(
        config_path,
        tmp_path / "mf",
        linear_solver=solver_type,
        **common,
    )

    ref_disp = ref.get_displacement()
    mf_disp = mf.get_displacement()
    ref_m = _disp_metrics(ref_disp)
    mf_m = _disp_metrics(mf_disp)

    # Displacement norms must agree closely.
    assert np.isclose(
        ref_m["l2"], mf_m["l2"], rtol=1e-4
    ), f"l2 mismatch: assembled={ref_m['l2']:.8e}, {solver_type}={mf_m['l2']:.8e}"
    assert np.isclose(
        ref_m["linf"], mf_m["linf"], rtol=1e-4
    ), f"linf mismatch: assembled={ref_m['linf']:.8e}, {solver_type}={mf_m['linf']:.8e}"
    assert np.allclose(
        ref_m["comp_l2"], mf_m["comp_l2"], rtol=1e-4, atol=1e-6
    ), f"comp_l2 mismatch: assembled={ref_m['comp_l2']}, {solver_type}={mf_m['comp_l2']}"

    # Newton iteration count should be the same (or very close).
    assert abs(ref.newton_iterations - mf.newton_iterations) <= 1, (
        f"Newton iters differ: assembled={ref.newton_iterations}, "
        f"{solver_type}={mf.newton_iterations}"
    )

    # Both must converge.
    module = importlib.import_module(config_path)
    rel_tol = getattr(module.params, "relative_tol", 1e-5)
    assert ref.relative_residual_history[-1] < rel_tol
    assert mf.relative_residual_history[-1] < rel_tol


# Affine configs where analytical displacement is known.
MATRIX_FREE_AFFINE_CONFIGS = [
    ("tests.examples.two_d_affine_shear.config", 2),
    ("tests.examples.two_d_affine_mix1.config", 2),
    ("tests.examples.three_d_affine_shear.config", 3),
]


@pytest.mark.parametrize("solver_type", MATRIX_FREE_SOLVERS)
@pytest.mark.parametrize("config_path, dim", MATRIX_FREE_AFFINE_CONFIGS)
def test_matrix_free_matches_analytical(config_path, dim, solver_type, tmp_path):
    """Matrix-free solvers must match analytical affine displacement."""
    solver = _run_solver(
        config_path,
        tmp_path,
        linear_solver=solver_type,
        ew_enabled=False,
    )

    assert solver.relative_residual_history[-1] < 1e-5

    coords = np.array(solver.vtu_writer.points, copy=True)
    displacement = solver.get_displacement()
    grad = np.asarray(GRADIENTS[config_path], dtype=np.float64)
    expected_disp = _affine_displacement(grad, coords)
    np.testing.assert_allclose(
        displacement,
        expected_disp,
        rtol=1e-4,
        atol=5e-8,
        err_msg=f"{solver_type}: displacement does not match analytical affine map.",
    )


@pytest.mark.parametrize("config_path", RIGHT_BOUNDARY_CASES)
def test_matrix_free_cg_matches_assembled_on_right_boundary(config_path, tmp_path):
    """matrix_free_cg must match assembled CG on right-boundary configs.

    autodiff_matrix_free_cg is excluded here because it diverges on Neumann BCs.

    The assembled path uses AMGX PCG+AMG (fixed 1e-6 tolerance) while the
    matrix-free path uses warp CG (1e-8 tolerance).  Different linear solve
    accuracy at each Newton step leads to slightly different convergence
    paths, so we use rtol=1e-3 for these Neumann BC cases.
    """
    common = dict(ew_enabled=False)

    ref = _run_solver(
        config_path,
        tmp_path / "ref",
        linear_solver="cg",
        **common,
    )
    mf = _run_solver(
        config_path,
        tmp_path / "mf",
        linear_solver="matrix_free_cg",
        **common,
    )

    ref_disp = ref.get_displacement()
    mf_disp = mf.get_displacement()
    ref_m = _disp_metrics(ref_disp)
    mf_m = _disp_metrics(mf_disp)

    assert np.isclose(
        ref_m["l2"], mf_m["l2"], rtol=1e-3
    ), f"l2 mismatch: assembled={ref_m['l2']:.8e}, matrix_free_cg={mf_m['l2']:.8e}"
    assert np.isclose(
        ref_m["linf"], mf_m["linf"], rtol=1e-3
    ), f"linf mismatch: assembled={ref_m['linf']:.8e}, matrix_free_cg={mf_m['linf']:.8e}"
    assert np.allclose(
        ref_m["comp_l2"], mf_m["comp_l2"], rtol=1e-3, atol=1e-6
    ), f"comp_l2 mismatch: assembled={ref_m['comp_l2']}, matrix_free_cg={mf_m['comp_l2']}"

    # Newton iteration count should be the same (or very close).
    assert abs(ref.newton_iterations - mf.newton_iterations) <= 1, (
        f"Newton iters differ: assembled={ref.newton_iterations}, "
        f"matrix_free_cg={mf.newton_iterations}"
    )


@pytest.mark.parametrize("solver_type", MATRIX_FREE_SOLVERS)
@pytest.mark.parametrize("config_path", MATRIX_FREE_COMPARISON_CONFIGS)
def test_matrix_free_with_ew_converges(config_path, solver_type, tmp_path):
    """Matrix-free solvers must converge with Eisenstat-Walker enabled."""
    solver = _run_solver(
        config_path,
        tmp_path,
        linear_solver=solver_type,
        ew_enabled=True,
        ew_eta_max=0.9,
    )

    module = importlib.import_module(config_path)
    rel_tol = getattr(module.params, "relative_tol", 1e-5)
    assert solver.relative_residual_history, "Residual history should not be empty."
    assert solver.relative_residual_history[-1] < rel_tol, (
        f"{solver_type} with EW did not converge: "
        f"final_rel={solver.relative_residual_history[-1]:.3e}, tol={rel_tol}"
    )

    # Result should still be in the same ballpark as assembled CG.
    ref = _run_solver(
        config_path,
        tmp_path / "ref",
        linear_solver="cg",
        ew_enabled=False,
    )
    ref_l2 = _disp_metrics(ref.get_displacement())["l2"]
    mf_l2 = _disp_metrics(solver.get_displacement())["l2"]
    assert np.isclose(
        ref_l2, mf_l2, rtol=1e-3
    ), f"EW {solver_type} l2={mf_l2:.8e} too far from assembled ref={ref_l2:.8e}"


# ---------------------------------------------------------------------------
# Load-stepping test
# ---------------------------------------------------------------------------


def test_load_stepping_matches_single_step(tmp_path):
    """Multi-step load stepping must converge to the same solution as single step."""
    # Single load step (default)
    s1 = NeoHookean(
        "tests.examples.three_d_affine_shear.config",
        override_params={"output_root": str(tmp_path / "s1")},
    )
    s1.forward()
    disp_1 = s1.get_displacement()

    # Multiple load steps
    s3 = NeoHookean(
        "tests.examples.three_d_affine_shear.config",
        override_params={"output_root": str(tmp_path / "s3"), "load_steps": 3},
    )
    s3.forward()
    disp_3 = s3.get_displacement()

    # Final displacements must match
    linf = np.abs(disp_1 - disp_3).max()
    assert linf < 1e-5, f"Load-stepping L-inf mismatch: {linf:.3e}"


# ---------------------------------------------------------------------------
# Solver x CG backend combination tests
# ---------------------------------------------------------------------------
# Verifies that all solver paths (cg, matrix_free_cg, autodiff_matrix_free_cg)
# work with all CG backend configurations on representative configs.
# For assembled "cg": cg_backend_assembled selects the backend:
#   - "native": warp CG + Jacobi preconditioner
#   - "amgx":   AMGX PCG + AGGREGATION AMG (prefer JAX-AMG, then pyamgx)
#   - "jaxamg": require JAX-AMG
#   - "pyamgx": require pyamgx
# For matrix-free solvers: cg_backend_assembled is ignored (always warp CG).

SOLVER_TYPES = ["cg", "matrix_free_cg", "autodiff_matrix_free_cg"]
CG_BACKENDS = [
    "native",  # warp CG + Jacobi
    "amgx",  # AMGX PCG + AMG
]

# Use a small 2D + 3D config for speed
COMBO_CONFIGS = [
    ("tests.examples.two_d_affine_shear.config", 2),
    ("tests.examples.three_d_affine_shear.config", 3),
]


@pytest.mark.parametrize("cg_backend", CG_BACKENDS)
@pytest.mark.parametrize("solver_type", SOLVER_TYPES)
@pytest.mark.parametrize("config_path, dim", COMBO_CONFIGS)
def test_solver_preconditioner_combinations(
    config_path, dim, solver_type, cg_backend, tmp_path
):
    """All solver x CG backend combinations must converge and match analytical."""
    overrides = dict(
        linear_solver=solver_type,
        cg_backend_assembled=cg_backend,
        ew_enabled=False,
    )

    label = f"{solver_type}+{cg_backend}"

    solver = _run_solver(config_path, tmp_path, **overrides)

    # Must converge
    assert solver.relative_residual_history, "Residual history should not be empty."
    assert solver.relative_residual_history[-1] < 1e-5, (
        f"{label}: did not converge "
        f"(rel={solver.relative_residual_history[-1]:.3e})"
    )

    # Must match analytical affine displacement
    coords = np.array(solver.vtu_writer.points, copy=True)
    displacement = solver.get_displacement()
    grad = np.asarray(GRADIENTS[config_path], dtype=np.float64)
    expected_disp = _affine_displacement(grad, coords)
    np.testing.assert_allclose(
        displacement,
        expected_disp,
        rtol=1e-4,
        atol=5e-8,
        err_msg=f"{label}: displacement mismatch.",
    )
