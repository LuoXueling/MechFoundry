"""
Warp kernels and utility functions for the diffused Nitsche method.

Provides:
- Trilinear hex shape functions and gradients in reference [0,1]^3 space
- Pre-computation kernels for phi, grad_phi at quadrature points
- Beta stabilization matrix assembly kernel
"""

import warp as wp

# VTK hex ordering for [0,1]^3 reference space:
# Node 0: (0,0,0), Node 1: (1,0,0), Node 2: (1,1,0), Node 3: (0,1,0)
# Node 4: (0,0,1), Node 5: (1,0,1), Node 6: (1,1,1), Node 7: (0,1,1)

# Corner coordinates stored as constant arrays for VTK hex ordering
_HEX_CORNERS_X = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
_HEX_CORNERS_Y = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
_HEX_CORNERS_Z = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


@wp.func
def hex_shape_value(xi: wp.vec3, node: int) -> float:
    """Evaluate trilinear shape function N_node at reference point xi in [0,1]^3.

    VTK hex ordering:
      0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
      4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
    """
    x = xi[0]
    y = xi[1]
    z = xi[2]

    # Each shape function is a product of (1-coord) or coord for each axis
    # depending on whether the node's corner coordinate is 0 or 1
    nx = float(0.0)
    ny = float(0.0)
    nz = float(0.0)

    if node == 0:
        nx = 1.0 - x
        ny = 1.0 - y
        nz = 1.0 - z
    elif node == 1:
        nx = x
        ny = 1.0 - y
        nz = 1.0 - z
    elif node == 2:
        nx = x
        ny = y
        nz = 1.0 - z
    elif node == 3:
        nx = 1.0 - x
        ny = y
        nz = 1.0 - z
    elif node == 4:
        nx = 1.0 - x
        ny = 1.0 - y
        nz = z
    elif node == 5:
        nx = x
        ny = 1.0 - y
        nz = z
    elif node == 6:
        nx = x
        ny = y
        nz = z
    elif node == 7:
        nx = 1.0 - x
        ny = y
        nz = z

    return nx * ny * nz


@wp.func
def hex_shape_grad_ref(xi: wp.vec3, node: int) -> wp.vec3:
    """Gradient of shape function N_node w.r.t. reference coords (xi, eta, zeta).

    Returns dN/d(xi), dN/d(eta), dN/d(zeta) as a vec3.
    """
    x = xi[0]
    y = xi[1]
    z = xi[2]

    dx = float(0.0)
    dy = float(0.0)
    dz = float(0.0)

    if node == 0:
        dx = -(1.0 - y) * (1.0 - z)
        dy = -(1.0 - x) * (1.0 - z)
        dz = -(1.0 - x) * (1.0 - y)
    elif node == 1:
        dx = (1.0 - y) * (1.0 - z)
        dy = -x * (1.0 - z)
        dz = -x * (1.0 - y)
    elif node == 2:
        dx = y * (1.0 - z)
        dy = x * (1.0 - z)
        dz = -x * y
    elif node == 3:
        dx = -y * (1.0 - z)
        dy = (1.0 - x) * (1.0 - z)
        dz = -(1.0 - x) * y
    elif node == 4:
        dx = -(1.0 - y) * z
        dy = -(1.0 - x) * z
        dz = (1.0 - x) * (1.0 - y)
    elif node == 5:
        dx = (1.0 - y) * z
        dy = -x * z
        dz = x * (1.0 - y)
    elif node == 6:
        dx = y * z
        dy = x * z
        dz = x * y
    elif node == 7:
        dx = -y * z
        dy = (1.0 - x) * z
        dz = (1.0 - x) * y

    return wp.vec3(dx, dy, dz)


@wp.func
def hex_jacobian(
    element_nodes: wp.array2d(dtype=wp.int32),
    elem_idx: int,
    node_coords: wp.array(dtype=wp.vec3),
    xi: wp.vec3,
) -> wp.mat33:
    """Compute 3x3 Jacobian J_ij = dx_i/d(xi_j) for a hex element.

    J = sum_k x_k (outer) dN_k/d(xi)
    """
    J = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for k in range(8):
        nid = element_nodes[elem_idx, k]
        xk = node_coords[nid]
        dNk = hex_shape_grad_ref(xi, k)
        # J_ij += xk_i * dNk_j
        for i in range(3):
            for j in range(3):
                J[i, j] = J[i, j] + xk[i] * dNk[j]
    return J


@wp.func
def hex_physical_grad(
    J_inv_T: wp.mat33,
    xi: wp.vec3,
    node: int,
) -> wp.vec3:
    """Compute physical gradient dN_node/dx = J^{-T} * dN_node/d(xi)."""
    dN_ref = hex_shape_grad_ref(xi, node)
    return J_inv_T * dN_ref


# --------------------------------------------------------------------------
# Kernel: Pre-compute phi and grad_phi at all quadrature points
# --------------------------------------------------------------------------
@wp.kernel
def precompute_qp_phi_grad(
    # Per-QP mapping: which element in the source mesh and reference coords
    qp_source_elem: wp.array(dtype=wp.int32),  # source element index (child or parent)
    qp_ref_coords: wp.array(dtype=wp.vec3),  # reference coords in source element
    qp_active: wp.array(dtype=wp.int32),  # 1=active, 0=padding
    # Source mesh data
    source_elements: wp.array2d(dtype=wp.int32),  # (n_source_elems, 8) connectivity
    source_node_coords: wp.array(dtype=wp.vec3),  # node positions
    source_node_sdf: wp.array(dtype=wp.float32),  # nodal SDF values
    epsilon: float,  # phase-field half-width
    # Output (flattened: total_qps = n_coarse_elems * qp_per_cell)
    phi_out: wp.array(dtype=wp.float32),
    grad_phi_out: wp.array(dtype=wp.vec3),
    grad_phi_mag_out: wp.array(dtype=wp.float32),
):
    """Interpolate SDF (smooth) then compute phi, ∇phi analytically.

    SDF is nearly linear inside each element, so shape-function interpolation
    of SDF and its gradient is accurate.  The nonlinear tanh is applied
    analytically via the chain rule, with ∇sdf normalized to enforce the
    eikonal property |∇sdf| = 1 (trilinear interpolation underestimates
    the gradient at edges/corners where the SDF has a kink):

        φ        = 0.5 (1 − tanh(sdf/ε))
        n̂        = ∇sdf / |∇sdf|           (unit outward normal)
        ∇φ       = −2φ(1−φ)/ε  n̂
        |∇φ|     = 2φ(1−φ)/ε
    """
    tid = wp.tid()
    if qp_active[tid] == 0:
        phi_out[tid] = 0.0
        grad_phi_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        grad_phi_mag_out[tid] = 0.0
        return

    elem = qp_source_elem[tid]
    xi = qp_ref_coords[tid]

    # Interpolate SDF and its reference gradient from 8 nodes
    sdf_val = float(0.0)
    grad_sdf_ref = wp.vec3(0.0, 0.0, 0.0)
    for k in range(8):
        nid = source_elements[elem, k]
        Nk = hex_shape_value(xi, k)
        dNk = hex_shape_grad_ref(xi, k)
        sdf_k = source_node_sdf[nid]
        sdf_val = sdf_val + Nk * sdf_k
        grad_sdf_ref = grad_sdf_ref + dNk * sdf_k

    # Transform grad_sdf to physical space via Jacobian
    J = hex_jacobian(source_elements, elem, source_node_coords, xi)
    J_inv = wp.inverse(J)
    J_inv_T = wp.transpose(J_inv)
    grad_sdf_phys = J_inv_T * grad_sdf_ref

    # Normalize ∇sdf to unit length (eikonal: |∇sdf| = 1 near boundary).
    # Trilinear interpolation underestimates |∇sdf| at edges/corners where
    # the SDF gradient has a kink that linear shape functions cannot capture.
    grad_sdf_mag = wp.length(grad_sdf_phys)
    if grad_sdf_mag > 1.0e-12:
        n_hat = grad_sdf_phys / grad_sdf_mag
    else:
        n_hat = wp.vec3(0.0, 0.0, 0.0)

    # Analytical phi from SDF
    t = wp.tanh(sdf_val / epsilon)
    phi_val = 0.5 * (1.0 - t)
    phi_val = wp.clamp(phi_val, 0.0, 1.0)

    # Analytical grad_phi via chain rule with normalized direction:
    #   ∇φ = −2φ(1−φ)/ε * n̂,   |∇φ| = 2φ(1−φ)/ε
    scale = -2.0 * phi_val * (1.0 - phi_val) / epsilon
    grad_phi_phys = scale * n_hat

    phi_out[tid] = wp.float32(phi_val)
    grad_phi_out[tid] = grad_phi_phys
    grad_phi_mag_out[tid] = wp.abs(scale)


# --------------------------------------------------------------------------
# Kernel: Interpolate a vector field at QPs using parent element shape functions
# --------------------------------------------------------------------------
@wp.kernel
def interpolate_vector_field_at_qps(
    # Per-QP mapping to parent element
    qp_parent_elem: wp.array(dtype=wp.int32),  # parent element index
    qp_parent_ref: wp.array(dtype=wp.vec3),  # reference coords in parent
    qp_active: wp.array(dtype=wp.int32),
    # Parent mesh data
    parent_elements: wp.array2d(dtype=wp.int32),
    # Nodal field (n_nodes, 3) stored as flat array of vec3
    nodal_field: wp.array(dtype=wp.vec3),
    # Output
    field_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if qp_active[tid] == 0:
        field_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    elem = qp_parent_elem[tid]
    xi = qp_parent_ref[tid]

    val = wp.vec3(0.0, 0.0, 0.0)
    for k in range(8):
        nid = parent_elements[elem, k]
        Nk = hex_shape_value(xi, k)
        val = val + Nk * nodal_field[nid]

    field_out[tid] = val


# --------------------------------------------------------------------------
# Kernel: Interpolate a scalar field at QPs using parent element shape functions
# --------------------------------------------------------------------------
@wp.kernel
def interpolate_scalar_field_at_qps(
    qp_parent_elem: wp.array(dtype=wp.int32),
    qp_parent_ref: wp.array(dtype=wp.vec3),
    qp_active: wp.array(dtype=wp.int32),
    parent_elements: wp.array2d(dtype=wp.int32),
    nodal_field: wp.array(dtype=wp.float32),
    field_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if qp_active[tid] == 0:
        field_out[tid] = 0.0
        return

    elem = qp_parent_elem[tid]
    xi = qp_parent_ref[tid]

    val = float(0.0)
    for k in range(8):
        nid = parent_elements[elem, k]
        Nk = hex_shape_value(xi, k)
        val = val + Nk * nodal_field[nid]

    field_out[tid] = wp.float32(val)


# --------------------------------------------------------------------------
# Kernel: Cast float64 SDF to float32 (for tape-recorded SDF recomputation)
# --------------------------------------------------------------------------
@wp.kernel
def cast_sdf_f64_to_f32(
    sdf_f64: wp.array(dtype=wp.float64),
    sdf_f32: wp.array(dtype=wp.float32),
):
    """Element-wise cast of float64 SDF to float32.

    Used on tape so backward flows from float32 QP outputs to float64 design variables.
    """
    i = wp.tid()
    sdf_f32[i] = wp.float32(sdf_f64[i])


# --------------------------------------------------------------------------
# Kernel: SDF → φ (phase-field) via φ = ½(1 − tanh(d/ε))
# --------------------------------------------------------------------------
@wp.kernel
def sdf_to_phi_f64(
    sdf_f64: wp.array(dtype=wp.float64),
    epsilon: float,
    phi_f64_out: wp.array(dtype=wp.float64),
):
    """Elementwise φ = ½(1 − tanh(d/ε)), float64 in/out.

    Launched on tape each forward to populate ``e_phi.dof_values`` — the
    "real" integrand φ derived from SDF.  Tape thus records SDF →
    ``sdf_to_phi_f64`` → ``e_phi`` → integrand → J, so a single backward
    yields ∂J/∂e_phi at ``e_phi.dof_values.grad`` (the intermediate
    φ-gradient).  This intermediate is extracted off-tape and applied to a
    *separate* persistent ``_hidden_phi`` field that drives the post-opt
    bg-grid scatter for void formation.
    """
    i = wp.tid()
    eps64 = wp.float64(epsilon)
    half = wp.float64(0.5)
    one = wp.float64(1.0)
    phi_f64_out[i] = half * (one - wp.tanh(sdf_f64[i] / eps64))


@wp.kernel
def sdf_to_phi_f32(
    sdf_f32: wp.array(dtype=wp.float32),
    epsilon: float,
    phi_f32_out: wp.array(dtype=wp.float32),
):
    """Elementwise φ = ½(1 − tanh(d/ε)), float32 in/out.

    This is the native single-precision counterpart of ``sdf_to_phi_f64``.
    Keeping the complete SDF → phase-field tape path in one scalar type is
    required when the assembled JAX-AMG solve is configured for float32.
    """
    i = wp.tid()
    eps32 = wp.float32(epsilon)
    half = wp.float32(0.5)
    one = wp.float32(1.0)
    phi_f32_out[i] = half * (one - wp.tanh(sdf_f32[i] / eps32))


# --------------------------------------------------------------------------
# Kernel: Scatter child SDF gradients to parent nodes via interpolation weights
# --------------------------------------------------------------------------
@wp.kernel
def accumulate_child_grad_to_parent(
    child_grad: wp.array(dtype=wp.float32),
    child_parent_elem: wp.array(dtype=wp.int32),
    child_ref_coords: wp.array(dtype=wp.vec3),
    parent_elements: wp.array2d(dtype=wp.int32),
    parent_grad_f64: wp.array(dtype=wp.float64),
):
    """Accumulate child node SDF gradients to parent nodes.

    For each child node j with gradient g_j:
        parent_grad[k] += N_k(xi_j) * g_j
    where k runs over the 8 nodes of the parent element containing child j.
    """
    i = wp.tid()
    elem = child_parent_elem[i]
    xi = child_ref_coords[i]
    g = child_grad[i]
    for k in range(8):
        nid = parent_elements[elem, k]
        Nk = hex_shape_value(xi, k)
        wp.atomic_add(parent_grad_f64, nid, wp.float64(Nk * g))


# --------------------------------------------------------------------------
# Kernel: Concatenate parent + child SDF into combined array
# --------------------------------------------------------------------------
@wp.kernel
def concat_sdf_arrays(
    parent_sdf: wp.array(dtype=wp.float32),
    child_sdf: wp.array(dtype=wp.float32),
    n_parent: int,
    combined_sdf: wp.array(dtype=wp.float32),
):
    """combined_sdf = [parent_sdf; child_sdf]."""
    i = wp.tid()
    if i < n_parent:
        combined_sdf[i] = parent_sdf[i]
    else:
        combined_sdf[i] = child_sdf[i - n_parent]


# --------------------------------------------------------------------------
# Kernel: Expand per-element beta to per-QP beta (constant within element)
# --------------------------------------------------------------------------
@wp.kernel
def expand_element_beta_to_qp(
    beta_element: wp.array(dtype=wp.float32),  # (n_coarse_elems,)
    qp_per_cell: int,
    beta_qp: wp.array(dtype=wp.float32),  # (n_coarse_elems * qp_per_cell,)
):
    elem_idx = wp.tid()
    beta_val = beta_element[elem_idx]
    base = elem_idx * qp_per_cell
    for q in range(qp_per_cell):
        beta_qp[base + q] = beta_val
