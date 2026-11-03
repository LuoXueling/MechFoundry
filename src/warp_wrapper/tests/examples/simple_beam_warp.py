"""
Simple beam bending example using Warp Neo-Hookean solver

This example demonstrates:
1. Creating a 2D rectangular beam mesh
2. Setting up Neo-Hookean material
3. Applying boundary conditions (fixed left, prescribed displacement right)
4. Solving the nonlinear problem
5. Extracting and displaying results

Run with: PYTHONPATH=. python examples/simple_beam_warp.py
"""

import sys

sys.path.insert(0, ".")

import warp as wp
import numpy as np
from core.neoHookean_warp import NeoHookeanWarp
from core.geometry_warp import MeshConfig
from core.material_warp import MaterialProperties
from core.solver_warp import SolverConfig


def main():
    """Run simple beam bending example"""

    print("=" * 60)
    print("Neo-Hookean Beam Bending Example (Warp)")
    print("=" * 60)

    # Initialize Warp
    wp.init()

    # ========== Problem Setup ==========
    print("\n1. Setting up problem...")

    # Mesh configuration: 4m x 1m beam, 20x5 elements
    mesh_config = MeshConfig(
        width=4.0, height=1.0, nx=20, ny=5, cell_type="quad", dim=2
    )

    # Material: Steel-like properties
    # E = 200 GPa (scaled to 200 for numerical stability)
    # nu = 0.3
    material = MaterialProperties(E=200.0, nu=0.3)

    print(
        f"   Mesh: {mesh_config.nx}x{mesh_config.ny} = {mesh_config.nx * mesh_config.ny} elements"
    )
    print(f"   Material: {material}")

    # Create solver
    solver = NeoHookeanWarp(
        mesh_config=mesh_config,
        material=material,
        degree=1,  # Linear elements
        device="cuda",  # Use GPU
    )

    print(f"   Nodes: {solver.space.node_count()}")
    print(f"   DOFs: {solver.space.node_count() * 2}")

    # ========== Boundary Conditions ==========
    print("\n2. Applying boundary conditions...")

    # Fixed left edge (all DOFs = 0)
    solver.add_dirichlet_bc("left", value=0.0, components=None)
    print("   ✓ Fixed left edge (clamped)")

    # Small prescribed displacement on right edge (x-direction only)
    # Using small displacement for linear-like behavior
    prescribed_disp = 0.05  # 5% strain
    solver.add_dirichlet_bc("right", value=prescribed_disp, components=[0])
    print(
        f"   ✓ Prescribed displacement: {prescribed_disp}m on right edge (x-direction)"
    )

    num_dirichlet = solver.bc_manager.num_dirichlet_bcs()
    print(f"   Total Dirichlet BCs: {num_dirichlet}")

    # ========== Solver Configuration ==========
    print("\n3. Configuring solver...")

    solver_config = SolverConfig(
        max_iter=20,
        rel_tol=1e-8,
        abs_tol=1e-10,
        linear_solver="direct",  # Use direct solver for small problem
        verbose=True,
    )

    print(f"   Max iterations: {solver_config.max_iter}")
    print(f"   Relative tolerance: {solver_config.rel_tol}")
    print(f"   Linear solver: {solver_config.linear_solver}")

    # ========== Solve ==========
    print("\n4. Solving nonlinear problem...")
    print("-" * 60)

    try:
        results = solver.solve(solver_config=solver_config)

        print("-" * 60)
        print("\n5. Results:")
        print("   " + "✓" * 30)

        if results["converged"]:
            print(f"   ✅ Solution converged in {results['num_iterations']} iterations")
        else:
            print(f"   ⚠️  Did not converge (max iterations reached)")

        print(f"   Residual norm: {results['residual_norm']:.3e}")
        print(f"   Solve time: {results['solve_time']:.3f} seconds")
        print(f"   Strain energy: {results['strain_energy']:.6e}")

        # Get displacement
        u = solver.get_displacement()

        # Compute norms
        u_L2 = solver.get_displacement_norm("L2")
        u_Linf = solver.get_displacement_norm("Linf")

        print(f"\n   Displacement norms:")
        print(f"   L2: {u_L2:.6e}")
        print(f"   L∞: {u_Linf:.6e}")

        # Statistics on displacement components
        u_x = u[:, 0]
        u_y = u[:, 1]

        print(f"\n   Displacement statistics:")
        print(f"   X-displacement: min={np.min(u_x):.6f}, max={np.max(u_x):.6f}")
        print(f"   Y-displacement: min={np.min(u_y):.6f}, max={np.max(u_y):.6f}")

        # ========== Save Results ==========
        print("\n6. Saving results...")

        output_dir = "output_warp"
        solver.save_numpy(output_dir)
        print(f"   ✓ Saved to {output_dir}/")

        # Print solver info
        print("\n7. Solver information:")
        info = solver.get_info()
        print(f"   Mesh: {info['mesh']}")
        print(f"   Material: {info['material']}")

        print("\n" + "=" * 60)
        print("Example completed successfully! 🎉")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during solve: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback

        traceback.print_exc()

        print("\n⚠️  Note: This example demonstrates the setup workflow.")
        print("   Full Newton solver with Warp FEM integration is complex")
        print("   and may require additional Warp-specific implementation.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
