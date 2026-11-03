"""
Utility functions for Neo-Hookean solver
"""

from datetime import datetime
import math
import warp.fem as fem
import warp as wp
import numpy as np
import torch


class dotdict(dict):
    """
    Dictionary subclass that allows attribute-style access.

    Example:
        d = dotdict({'a': 1, 'b': 2})
        print(d.a)  # Returns 1
        d.c = 3     # Sets d['c'] = 3
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        """Return a copy of the dotdict"""
        return dotdict(super().copy())


def log(*args, **kwargs):
    """
    Logging function with timestamp and optional file output.

    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments, including:
            - output_file: Optional file path to write log to
            - Other kwargs passed to print()

    Example:
        log("Starting simulation")
        log("Step 1 complete", output_file="log.txt")
    """
    # Format timestamp
    timestamp = f"[{datetime.now().strftime('%m-%d-%Y,%H:%M:%S.%f')[:-3]}]"

    # Extract output file if provided
    output_file = kwargs.get("output_file", None)
    _kwargs = kwargs.copy()
    _kwargs.update(dict(flush=True))

    if output_file is not None:
        del _kwargs["output_file"]

    # Print to console
    if len(args) == 0:
        print(*args, **_kwargs)
    else:
        print(timestamp, *args, **_kwargs)

    # Write to file if specified
    if output_file is not None:
        with open(output_file, "a") as file:
            _kwargs.update(dict(file=file))
            if len(args) == 0:
                print(*args, **_kwargs)
            else:
                print(timestamp, *args, **_kwargs)


"""
Boundary-condition lambda factories removed. Use inline lambdas in configs,
e.g., lambda x,y: x < 0.01 or lambda x,y,z: x > WIDTH-0.01
"""

from typing import Any, Union, Tuple, List, Callable, Optional


class FlexibleBC:
    """Container for flexible boundary condition specification"""

    def __init__(
        self,
        boundary_spec: Callable,
        value_spec: Union[float, tuple, Callable, List],
        components: Optional[List[int]] = None,
    ):
        """
        Initialize flexible BC

        Args:
            boundary_spec: Function predicate (lambda) defining the boundary
            value_spec: Value specification (see formats above)
            components: Optional list of components to constrain (e.g., [0] for x only)
        """
        if not callable(boundary_spec):
            raise TypeError(
                f"boundary_spec must be callable (lambda function), got {type(boundary_spec).__name__}"
            )
        self.boundary_spec = boundary_spec
        self.value_spec = value_spec
        self.components = components

    def __repr__(self):
        return f"FlexibleBC(boundary={self.boundary_spec}, value={self.value_spec}, components={self.components})"


def identify_boundary_nodes(
    boundary_spec: Callable,
    node_positions,
    dim: int,
):
    """
    Identify nodes on boundary using function predicate (2D or 3D).

    Accepts torch.Tensor (GPU) or np.ndarray positions.
    Returns torch.LongTensor on GPU (if torch input) or np.ndarray (if numpy).

    Tries vectorized evaluation first, falls back to per-node scalar
    for lambdas using Python ``and``/``or``/``if``.
    """
    if not callable(boundary_spec):
        raise TypeError(
            f"boundary_spec must be callable (lambda function), got {type(boundary_spec).__name__}"
        )

    # --- Torch fast path (GPU) ---
    if isinstance(node_positions, torch.Tensor):
        coords = node_positions
        n = coords.shape[0]
        try:
            if dim == 2:
                mask = boundary_spec(coords[:, 0], coords[:, 1])
            else:
                mask = boundary_spec(coords[:, 0], coords[:, 1], coords[:, 2])
            if isinstance(mask, torch.Tensor):
                mask = mask.bool().view(-1)
                if mask.shape[0] == n:
                    return torch.where(mask)[0]  # LongTensor on GPU
        except Exception:
            pass
        # Fallback: try vectorized numpy call before per-node scalar loop
        coords_np = coords.cpu().numpy()
        try:
            if dim == 2:
                mask_np = boundary_spec(coords_np[:, 0], coords_np[:, 1])
            else:
                mask_np = boundary_spec(
                    coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]
                )
            mask_np = np.asarray(mask_np, dtype=bool).ravel()
            if mask_np.shape == (n,):
                return torch.from_numpy(np.where(mask_np)[0]).long().to(coords.device)
        except Exception:
            pass
        # Last resort: per-node scalar (slow)
        if dim == 2:
            mask_np = np.array(
                [
                    boundary_spec(float(coords_np[i, 0]), float(coords_np[i, 1]))
                    for i in range(n)
                ],
                dtype=bool,
            )
        else:
            mask_np = np.array(
                [
                    boundary_spec(
                        float(coords_np[i, 0]),
                        float(coords_np[i, 1]),
                        float(coords_np[i, 2]),
                    )
                    for i in range(n)
                ],
                dtype=bool,
            )
        return torch.from_numpy(np.where(mask_np)[0]).long().to(coords.device)

    # --- Numpy path (backward compat) ---
    coords = np.asarray(node_positions, dtype=float)
    n = len(coords)

    try:
        if dim == 2:
            mask = boundary_spec(coords[:, 0], coords[:, 1])
        else:
            mask = boundary_spec(coords[:, 0], coords[:, 1], coords[:, 2])
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.shape == (n,):
            return np.where(mask)[0].astype(int)
    except Exception:
        pass

    if dim == 2:
        mask = np.array(
            [boundary_spec(float(coords[i, 0]), float(coords[i, 1])) for i in range(n)],
            dtype=bool,
        )
    else:
        mask = np.array(
            [
                boundary_spec(
                    float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
                )
                for i in range(n)
            ],
            dtype=bool,
        )
    return np.where(mask)[0].astype(int)


# Keep old names as thin wrappers for backward compatibility
def identify_boundary_nodes_2d(boundary_spec, node_positions):
    return identify_boundary_nodes(boundary_spec, node_positions, 2)


def identify_boundary_nodes_3d(boundary_spec, node_positions):
    return identify_boundary_nodes(boundary_spec, node_positions, 3)


def _call_coord_func(func, coord, dim):
    """Call a user-provided coordinate function with the right number of args."""
    if dim == 2:
        return func(float(coord[0]), float(coord[1]))
    return func(float(coord[0]), float(coord[1]), float(coord[2]))


def _call_coord_func_vectorized(func, coords, dim):
    """Call func with full coordinate arrays. Returns (result, success).

    If *coords* is a torch.Tensor, *result* is a torch.Tensor (float64).
    If *coords* is a numpy array, *result* is a numpy array (float64).
    Returns ``(None, False)`` if the function doesn't support array args.
    """
    if isinstance(coords, torch.Tensor):
        try:
            if dim == 2:
                result = func(coords[:, 0], coords[:, 1])
            else:
                result = func(coords[:, 0], coords[:, 1], coords[:, 2])
            if isinstance(result, torch.Tensor):
                return result.to(dtype=torch.float64), True
            # Scalar result from arithmetic lambda
            return (
                torch.full(
                    (coords.shape[0],),
                    float(result),
                    dtype=torch.float64,
                    device=coords.device,
                ),
                True,
            )
        except Exception:
            pass
        return None, False

    try:
        if dim == 2:
            result = func(coords[:, 0], coords[:, 1])
        else:
            result = func(coords[:, 0], coords[:, 1], coords[:, 2])
        return np.asarray(result, dtype=float), True
    except Exception:
        return None, False


def evaluate_bc_values(
    value_spec: Union[float, tuple, Callable, List],
    node_indices,
    node_positions,
    dim: int,
) -> Tuple:
    """
    Evaluate BC values for specified nodes (2D or 3D).

    Accepts torch.Tensor or np.ndarray for node_indices/node_positions.
    Returns (values, components) where values matches the input type.

    Args:
        value_spec: Value specification (scalar, tuple, callable, or [components, value])
        node_indices: Indices of nodes where BC applies (torch.LongTensor or np.ndarray)
        node_positions: (N_total, dim) array of all node coordinates
        dim: 2 or 3

    Returns:
        (values_array, components_list)
        values_array: shape (n_nodes, dim) — torch.Tensor on GPU or np.ndarray
        components_list: None (all components) or list of constrained components
    """
    # --- Torch path (GPU) ---
    if isinstance(node_positions, torch.Tensor):
        n_nodes = node_indices.shape[0]
        coords = node_positions[node_indices]
        dev = coords.device
        return _evaluate_bc_values_torch(value_spec, coords, n_nodes, dim, dev)

    # --- Numpy path (backward compat) ---
    n_nodes = len(node_indices)
    coords = np.asarray(node_positions, dtype=float)[node_indices]
    return _evaluate_bc_values_numpy(value_spec, coords, n_nodes, dim)


def _evaluate_bc_values_torch(value_spec, coords, n_nodes, dim, dev):
    """Torch GPU implementation of evaluate_bc_values."""
    # Case 1: Scalar
    if isinstance(value_spec, (int, float)):
        return (
            torch.full((n_nodes, dim), value_spec, dtype=torch.float64, device=dev),
            None,
        )

    # Case 2: Tuple matching dimension
    if isinstance(value_spec, tuple) and len(value_spec) == dim:
        if all(isinstance(v, (int, float)) for v in value_spec):
            return (
                torch.tensor(value_spec, dtype=torch.float64, device=dev)
                .unsqueeze(0)
                .expand(n_nodes, -1)
                .clone(),
                None,
            )
        if all(callable(v) for v in value_spec):
            values = torch.zeros(n_nodes, dim, dtype=torch.float64, device=dev)
            for c in range(dim):
                vec_result, ok = _call_coord_func_vectorized(value_spec[c], coords, dim)
                if (
                    ok
                    and isinstance(vec_result, torch.Tensor)
                    and vec_result.view(-1).shape[0] >= n_nodes
                ):
                    values[:, c] = vec_result.view(-1)[:n_nodes]
                else:
                    coords_np = coords.cpu().numpy()
                    for idx in range(n_nodes):
                        values[idx, c] = _call_coord_func(
                            value_spec[c], coords_np[idx], dim
                        )
            return values, None

    # Case 3: ([components], scalar_or_callable)
    if (
        isinstance(value_spec, tuple)
        and len(value_spec) == 2
        and isinstance(value_spec[0], list)
    ):
        components = value_spec[0]
        val = value_spec[1]
        values = torch.zeros(n_nodes, dim, dtype=torch.float64, device=dev)
        if isinstance(val, (int, float)):
            for comp in components:
                values[:, comp] = val
        elif callable(val):
            vec_result, ok = _call_coord_func_vectorized(val, coords, dim)
            if ok and isinstance(vec_result, torch.Tensor):
                r = (
                    vec_result.view(-1)[:n_nodes]
                    if vec_result.dim() == 1
                    else vec_result
                )
                if r.dim() == 1:
                    for comp in components:
                        values[:, comp] = r
                else:
                    for ci, comp in enumerate(components):
                        values[:, comp] = r[:, ci]
            else:
                coords_np = coords.cpu().numpy()
                for idx in range(n_nodes):
                    v = _call_coord_func(val, coords_np[idx], dim)
                    if isinstance(v, (tuple, list, np.ndarray)):
                        for ci, comp in enumerate(components):
                            values[idx, comp] = v[ci]
                    else:
                        for comp in components:
                            values[idx, comp] = v
        else:
            raise ValueError(f"Invalid value in (components, value) format: {val}")
        return values, components

    # Case 4: Single callable
    if callable(value_spec):
        vec_result, ok = _call_coord_func_vectorized(value_spec, coords, dim)
        if ok and isinstance(vec_result, torch.Tensor):
            if vec_result.dim() == 1 and vec_result.shape[0] == n_nodes:
                return vec_result.view(-1, 1).expand(-1, dim).clone(), None
            elif vec_result.dim() == 2 and vec_result.shape == (n_nodes, dim):
                return vec_result.clone(), None
        coords_np = coords.cpu().numpy()
        values = torch.zeros(n_nodes, dim, dtype=torch.float64, device=dev)
        for idx in range(n_nodes):
            val = _call_coord_func(value_spec, coords_np[idx], dim)
            if isinstance(val, (tuple, list, np.ndarray)):
                for d in range(dim):
                    values[idx, d] = float(val[d])
            else:
                values[idx, :] = float(val)
        return values, None

    raise TypeError(f"Invalid value_spec type: {type(value_spec)}")


def _evaluate_bc_values_numpy(value_spec, coords, n_nodes, dim):
    """Numpy implementation of evaluate_bc_values (backward compat)."""
    # Case 1: Scalar
    if isinstance(value_spec, (int, float)):
        return np.full((n_nodes, dim), value_spec), None

    # Case 2: Tuple matching dimension
    if isinstance(value_spec, tuple) and len(value_spec) == dim:
        if all(isinstance(v, (int, float)) for v in value_spec):
            return np.tile(value_spec, (n_nodes, 1)), None
        if all(callable(v) for v in value_spec):
            values = np.zeros((n_nodes, dim))
            for c in range(dim):
                vec_result, ok = _call_coord_func_vectorized(value_spec[c], coords, dim)
                if ok and vec_result.shape == (n_nodes,):
                    values[:, c] = vec_result
                else:
                    for idx in range(n_nodes):
                        values[idx, c] = _call_coord_func(
                            value_spec[c], coords[idx], dim
                        )
            return values, None

    # Case 3: ([components], scalar_or_callable)
    if (
        isinstance(value_spec, tuple)
        and len(value_spec) == 2
        and isinstance(value_spec[0], list)
    ):
        components = value_spec[0]
        val = value_spec[1]
        values = np.zeros((n_nodes, dim))
        if isinstance(val, (int, float)):
            for comp in components:
                values[:, comp] = val
        elif callable(val):
            vec_result, ok = _call_coord_func_vectorized(val, coords, dim)
            if ok:
                if vec_result.ndim == 1 and vec_result.shape[0] == n_nodes:
                    for comp in components:
                        values[:, comp] = vec_result
                elif vec_result.ndim == 2 and vec_result.shape[0] == n_nodes:
                    for ci, comp in enumerate(components):
                        values[:, comp] = vec_result[:, ci]
                else:
                    ok = False
            if not ok:
                for idx in range(n_nodes):
                    v = _call_coord_func(val, coords[idx], dim)
                    if isinstance(v, (tuple, list, np.ndarray)):
                        for ci, comp in enumerate(components):
                            values[idx, comp] = v[ci]
                    else:
                        for comp in components:
                            values[idx, comp] = v
        else:
            raise ValueError(f"Invalid value in (components, value) format: {val}")
        return values, components

    # Case 4: Single callable
    if callable(value_spec):
        vec_result, ok = _call_coord_func_vectorized(value_spec, coords, dim)
        if ok:
            if vec_result.ndim == 1 and vec_result.shape[0] == n_nodes:
                return np.broadcast_to(vec_result[:, None], (n_nodes, dim)).copy(), None
            elif vec_result.ndim == 2 and vec_result.shape == (n_nodes, dim):
                return vec_result.copy(), None
        values = np.zeros((n_nodes, dim))
        for idx in range(n_nodes):
            val = _call_coord_func(value_spec, coords[idx], dim)
            if isinstance(val, (tuple, list, np.ndarray)):
                values[idx, :] = val
            else:
                values[idx, :] = val
        return values, None

    raise TypeError(f"Invalid value_spec type: {type(value_spec)}")


# Keep old names as thin wrappers for backward compatibility
def evaluate_bc_values_2d(value_spec, node_indices, node_positions):
    return evaluate_bc_values(value_spec, node_indices, node_positions, 2)


def evaluate_bc_values_3d(value_spec, node_indices, node_positions):
    return evaluate_bc_values(value_spec, node_indices, node_positions, 3)


def process_flexible_bc(
    bc_spec: FlexibleBC,
    dim: int,
    params: dict,
    node_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]]]:
    """
    Process a flexible BC specification.

    Args:
        bc_spec: FlexibleBC object
        dim: Problem dimension (2 or 3)
        params: Parameters dict with mesh info
        node_positions: (N, dim) array of node coordinates

    Returns:
        (node_indices, values, components)
    """
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dimension: {dim}")

    pos = node_positions[:, :dim] if node_positions.shape[1] > dim else node_positions
    nodes = identify_boundary_nodes(bc_spec.boundary_spec, pos, dim)
    values, components = evaluate_bc_values(bc_spec.value_spec, nodes, pos, dim)

    # Override components if specified in BC
    if bc_spec.components is not None:
        components = bc_spec.components

    return nodes, values, components


def bsr_upcast_f32_to_f64(K):
    """Upcast a float32 BSR matrix to float64 (copies structure + values).

    Returns a new BsrMatrix with mat22d/mat33d blocks. Used for AMGX dDDI mode
    when K is assembled in float32 to save memory.
    """
    from warp.sparse import bsr_copy

    return bsr_copy(K, scalar_type=wp.float64)


def bsr_to_dense(matrix) -> np.ndarray:
    """
    Convert a Warp BSR matrix to a dense NumPy array on the host.

    Args:
        matrix: Warp sparse BSR matrix object.

    Returns:
        Dense NumPy array with shape (n_dofs, n_dofs) in float64.
    """
    block_rows, block_cols = matrix.block_shape
    n_block_rows = matrix.nrow
    n_block_cols = matrix.ncol

    dense = np.zeros(
        (n_block_rows * block_rows, n_block_cols * block_cols),
        dtype=np.float64,
    )

    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()
    blocks = matrix.scalar_values.numpy()

    for row in range(n_block_rows):
        start = offsets[row]
        end = offsets[row + 1]
        row_slice = slice(row * block_rows, (row + 1) * block_rows)
        for idx in range(start, end):
            col = columns[idx]
            col_slice = slice(col * block_cols, (col + 1) * block_cols)
            dense[row_slice, col_slice] = blocks[idx]

    return dense


# -----------------------------------------------------------------------------
# Array packing and BC helpers (reduce boilerplate in neoHookean)
# -----------------------------------------------------------------------------


def make_vecd_array(vec_np: np.ndarray, dim: int, device: str):
    """Pack (N, dim) float64 numpy array into Warp vec2d/vec3d array.

    Args:
        vec_np: numpy array of shape (N, dim), float-like
        dim: 2 or 3
        device: 'cuda' or 'cpu'
    Returns:
        wp.array of vec2d/vec3d on device
    """
    if vec_np.size == 0:
        return wp.zeros(0, dtype=wp.types.vector(length=dim, dtype=wp.float64), device=device)

    vec_np = np.ascontiguousarray(vec_np, dtype=np.float64)
    dtype = wp.vec2d if dim == 2 else wp.vec3d
    return wp.array(vec_np, dtype=dtype, device=device)


def build_bc_projector(
    space: fem.FunctionSpace, bc_mask, device: str, scalar_type=None
):
    """Build BSR projector matrix encoding Dirichlet constraints on specific components.

    The matrix has identity blocks at constrained components, zeros elsewhere,
    suitable for use with fem.project_linear_system.

    Args:
        bc_mask: torch.BoolTensor (N, dim) on GPU or np.ndarray (N, dim) bool.
    """
    from warp.sparse import bsr_zeros, bsr_set_from_triplets

    n_nodes = space.node_count()
    dim = space.dimension
    if scalar_type is None:
        scalar_type = wp.float64
    block_type = wp.types.matrix(shape=(dim, dim), dtype=scalar_type)
    bd_matrix = bsr_zeros(n_nodes, n_nodes, block_type)

    if isinstance(bc_mask, torch.Tensor):
        bc_node_mask = bc_mask.any(dim=1)
        bc_nodes = torch.where(bc_node_mask)[0]
        if bc_nodes.shape[0] == 0:
            return bd_matrix

        bc_nodes_i32 = bc_nodes.to(dtype=torch.int32)
        rows = wp.from_torch(bc_nodes_i32.contiguous(), dtype=wp.int32)
        cols = wp.from_torch(bc_nodes_i32.contiguous(), dtype=wp.int32)

        n_bc = bc_nodes.shape[0]
        torch_dtype = torch.float64 if scalar_type == wp.float64 else torch.float32
        blocks_t = torch.zeros(n_bc, dim, dim, dtype=torch_dtype, device=device)
        for d in range(dim):
            blocks_t[:, d, d] = bc_mask[bc_nodes, d].to(dtype=torch_dtype)
        blocks_wp = wp.from_torch(blocks_t.contiguous(), dtype=block_type)

        bsr_set_from_triplets(bd_matrix, rows, cols, blocks_wp)
        return bd_matrix

    # Numpy path (backward compat)
    bc_node_mask = np.any(bc_mask, axis=1)
    bc_nodes = np.where(bc_node_mask)[0]
    if bc_nodes.size == 0:
        return bd_matrix

    rows = wp.array(bc_nodes, dtype=wp.int32, device=device)
    cols = wp.array(bc_nodes, dtype=wp.int32, device=device)

    n_bc = len(bc_nodes)
    np_dtype = np.float64 if scalar_type == wp.float64 else np.float32
    blocks_np = np.zeros((n_bc, dim, dim), dtype=np_dtype)
    for d in range(dim):
        blocks_np[:, d, d] = bc_mask[bc_nodes, d].astype(np_dtype)
    blocks_np = np.ascontiguousarray(blocks_np)
    blocks = wp.array(blocks_np, dtype=block_type, device=device)

    bsr_set_from_triplets(bd_matrix, rows, cols, blocks)
    return bd_matrix


def build_bc_rhs(
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    u_current: np.ndarray,
    dim: int,
    device: str,
):
    """Build the right-hand side vector for Dirichlet projection: (bc - u) on constrained dofs.

    Returns wp.array of vec2d/vec3d with zeros on free dofs.
    """
    n_nodes = u_current.shape[0]
    bd_rhs_np = np.zeros((n_nodes, dim))  # dtype follows make_vecd_array
    idx = np.where(bc_mask)
    diff = bc_values - u_current
    bd_rhs_np[idx] = diff[idx]
    return make_vecd_array(bd_rhs_np, dim, device)


# -----------------------------------------------------------------------------
# Neumann traction lumping helpers
# -----------------------------------------------------------------------------


def lump_boundary_tractions_2d(traction, nodes, coords, values):
    """Lump line (edge) tractions onto end nodes using linear shape functions.

    All inputs are torch tensors on GPU. traction is (N_total, dim) float64,
    modified in-place via scatter_add_.
    """
    if nodes.shape[0] < 2:
        return
    span = coords.max(dim=0).values - coords.min(dim=0).values
    order_axis = 0 if span[0] >= span[1] else 1
    order = torch.argsort(coords[:, order_axis])
    ordered_nodes = nodes[order]
    ordered_coords = coords[order]
    ordered_values = values[order]

    p1, p2 = ordered_coords[:-1], ordered_coords[1:]
    lengths = torch.linalg.norm(p2 - p1, dim=1)
    t1, t2 = ordered_values[:-1], ordered_values[1:]
    valid = lengths > 0
    L = lengths[valid].unsqueeze(-1) / 6.0
    contrib_n1 = L * (2.0 * t1[valid] + t2[valid])
    contrib_n2 = L * (t1[valid] + 2.0 * t2[valid])
    idx1 = ordered_nodes[:-1][valid].long().unsqueeze(-1).expand_as(contrib_n1)
    idx2 = ordered_nodes[1:][valid].long().unsqueeze(-1).expand_as(contrib_n2)
    traction.scatter_add_(0, idx1, contrib_n1)
    traction.scatter_add_(0, idx2, contrib_n2)


def lump_boundary_tractions_3d(traction, nodes, node_positions, values, triangulation):
    """Lump surface tractions on triangulated faces to nodes using linear shape functions.

    All inputs are torch tensors on GPU. triangulation is (T, 3) global node
    index tensor. traction is (N_total, dim) float64, modified in-place.
    """
    if triangulation is None or nodes.shape[0] == 0:
        return

    p1 = node_positions[triangulation[:, 0].long()]
    p2 = node_positions[triangulation[:, 1].long()]
    p3 = node_positions[triangulation[:, 2].long()]
    areas = 0.5 * torch.linalg.norm(torch.linalg.cross(p2 - p1, p3 - p1), dim=1)

    # Map global node index → local index into values
    max_idx = int(nodes.max()) + 1
    local_idx = torch.full((max_idx,), -1, dtype=torch.long, device=nodes.device)
    local_idx[nodes.long()] = torch.arange(nodes.shape[0], device=nodes.device)
    t1 = values[local_idx[triangulation[:, 0].long()]]
    t2 = values[local_idx[triangulation[:, 1].long()]]
    t3 = values[local_idx[triangulation[:, 2].long()]]

    valid = areas > 0
    A = areas[valid].unsqueeze(-1) / 12.0
    c1 = A * (2.0 * t1[valid] + t2[valid] + t3[valid])
    c2 = A * (t1[valid] + 2.0 * t2[valid] + t3[valid])
    c3 = A * (t1[valid] + t2[valid] + 2.0 * t3[valid])
    tri_v = triangulation[valid]
    traction.scatter_add_(0, tri_v[:, 0:1].long().expand_as(c1), c1)
    traction.scatter_add_(0, tri_v[:, 1:2].long().expand_as(c2), c2)
    traction.scatter_add_(0, tri_v[:, 2:3].long().expand_as(c3), c3)


## Removed: array_to_string (unused)


# -----------------------------------------------------------------------------
# Nonlinear solver utilities
# -----------------------------------------------------------------------------


def backtracking_line_search(
    residual_norm,
    relative_residual,
    abs_tol,
    params,
    apply_step,
    revert_step,
    residual_norm_0,
    debug_log=None,
):
    """Backtracking line search with simple Armijo-like condition.

    apply_step(alpha) must return (residual_norm, relative_residual) and perform the update.
    revert_step(alpha) undoes the update.
    """
    max_iters = int(params.get("line_search_max_iters", 8))
    armijo = float(params.get("line_search_armijo", 1e-2))
    min_alpha = float(params.get("line_search_min_alpha", 1e-6))
    rel_tol = float(params.get("relative_tol", 1e-5))
    target_residual = (
        max(abs_tol, rel_tol * residual_norm_0)
        if residual_norm_0 is not None
        else abs_tol
    )

    def log(msg):
        if debug_log is not None:
            debug_log(msg)

    if max_iters <= 0:
        res_trial, rel_trial = apply_step(1.0)
        if not math.isfinite(res_trial):
            revert_step(1.0)
            return False, residual_norm, relative_residual, 0.0
        return True, res_trial, rel_trial, 1.0

    best_alpha = 0.0
    best_residual = residual_norm
    best_relative = relative_residual

    alpha = 1.0
    for _ in range(max_iters):
        res_trial, rel_trial = apply_step(alpha)

        if not math.isfinite(res_trial):
            log(f"      trial alpha={alpha:.3e}, ||R|| = NaN/Inf -> reducing step")
            revert_step(alpha)
            alpha *= 0.5
            if alpha < min_alpha:
                break
            continue

        if res_trial < best_residual:
            best_residual = res_trial
            best_relative = rel_trial
            best_alpha = alpha

        log(
            f"      trial alpha={alpha:.3e}, ||R|| = {res_trial:.3e}, rel = {rel_trial:.3e}"
        )

        sufficient = res_trial < target_residual or res_trial <= residual_norm * (
            1.0 - armijo * alpha
        )
        if sufficient:
            return True, res_trial, rel_trial, alpha

        revert_step(alpha)
        alpha *= 0.5
        if alpha < min_alpha:
            break

    if best_alpha > 0.0 and best_residual < residual_norm:
        res_trial, rel_trial = apply_step(best_alpha)
        log(
            f"      fallback alpha={best_alpha:.3e}, ||R|| = {res_trial:.3e}, rel = {rel_trial:.3e}"
        )
        return True, res_trial, rel_trial, best_alpha

    return False, residual_norm, relative_residual, 0.0


# -----------------------------------------------------------------------------
# Linear system solvers
# -----------------------------------------------------------------------------


def solve_linear_system(
    K,
    b,
    method: str = "cg",
    cg_atol: float = None,
    cg_max_iters: int = 1000,
    device: str = "cuda",
    dim: int = 2,
    mv_routine=None,
    debug_log=None,
    cg_backend: str = "amgx",
    _cache: dict = None,
    check_every: int = 10,
    amgx_rebuild_every_iter: bool = False,
    use_mixed_precision: bool = False,
    scale_system_to_unit_diagonal: bool = False,
    jaxamg_max_iters: int = None,
):
    """Solve K x = b and return a precision-matched Warp vector array.

    Dispatch logic:
    - ``method='dense'``: numpy dense solve (debug only).
    - ``mv_routine is not None``: matrix-free path — warp CG with no
      preconditioner.
    - Otherwise (assembled BSR), controlled by ``cg_backend``:
      - ``cg_backend='native'``: warp CG with block-diagonal Jacobi
        preconditioner.  Eisenstat-Walker adaptive tolerance is passed
        via ``cg_atol``.
      - ``cg_backend='amgx'``: prefer JAX-AMG, falling back to pyamgx only
        when JAX-AMG is unavailable.
      - ``cg_backend='jaxamg'`` / ``'pyamgx'``: require that binding.
        Both run AMGX PCG with AGGREGATION AMG preconditioning. ``cg_atol``
        is ignored: pyamgx uses its fixed config tolerance; JAX-AMG uses its
        precision-specific fixed tolerance (``jaxamg_config``: 5e-6 for f32,
        2e-4 for 2-D f32, 1e-6 for f64).

    Parameters
    ----------
    K : BsrMatrix or None
        Assembled stiffness matrix.  ``None`` for matrix-free paths.
    b : wp.array
        Right-hand side vector.
    method : str
        ``"cg"`` (default) or ``"dense"``.
    cg_atol : float or None
        Absolute CG tolerance for warp CG paths (matrix-free and native
        assembled).  ``None`` defaults to 1e-8. Ignored for AMGX paths.
    cg_max_iters : int
        CG max iterations (warp CG paths only).
    device : str
        Warp device string.
    dim : int
        Spatial dimension (2 or 3).
    mv_routine : callable or None
        Matrix-free matvec ``(x, y, z, alpha, beta)`` implementing
        ``z = alpha*A*x + beta*y``.
    debug_log : callable or None
        Optional logging function.
    cg_backend : str
        ``"native"`` or ``"amgx"`` — selects assembled CG backend.
        Ignored for matrix-free paths.
    _cache : dict or None
        Optional cache for buffer reuse across Newton iterations.
    check_every : int
        CG convergence check interval (warp CG paths only).
    use_mixed_precision : bool
        Select the AMGX solve precision. JAX-AMG uses float32 throughout when
        false and float64 when true. pyamgx remains on its supported dDDI path.
    jaxamg_max_iters : int or None
        JAX-AMG Krylov iteration cap; ``None`` keeps ``jaxamg_config``'s 1000.
    """
    import numpy as _np

    # --- Dense path (debug only) ---
    if method == "dense":
        K_dense = bsr_to_dense(K)
        b_vec = b.numpy().reshape(-1)
        try:
            du_flat = _np.linalg.solve(K_dense, b_vec)
        except _np.linalg.LinAlgError as e:
            if debug_log:
                debug_log(f"[DEBUG] Dense solver failed: {e}")
            raise
        du_vec = du_flat.reshape(-1, dim)
        return make_vecd_array(du_vec, dim, device)

    from warp.sparse import BsrMatrix

    n_nodes = len(b) if b is not None else 0

    # --- Helper: get or allocate a cached buffer ---
    def _get_buf(key):
        """Return a cached zero buffer, or allocate and cache a new one."""
        if _cache is not None and key in _cache:
            buf = _cache[key]
            buf.zero_()
            return buf
        buf = wp.zeros(n_nodes, dtype=b.dtype, device=wp.get_device(device))
        if _cache is not None:
            _cache[key] = buf
        return buf

    effective_atol = float(cg_atol) if cg_atol is not None else 1e-8

    # --- Matrix-free path: warp CG (no preconditioner) ---
    if mv_routine is not None:
        from warp.optim.linear import (
            cg as wp_cg,
            LinearOperator,
        )

        scalar_dofs = n_nodes * dim
        A_op = LinearOperator(
            shape=(scalar_dofs, scalar_dofs),
            dtype=b.dtype,
            device=wp.get_device(device),
            matvec=mv_routine,
        )

        du = _get_buf("du")
        niter, res_norm, _ = wp_cg(
            A=A_op,
            b=b,
            x=du,
            atol=effective_atol,
            maxiter=cg_max_iters,
            M=None,
            check_every=check_every,
            use_cuda_graph=False,
        )
        if debug_log is not None:
            debug_log(
                f"[DEBUG] CG (matrix-free): iters={niter}, "
                f"res={res_norm:.3e}, atol={effective_atol:.3e}"
            )
        return du

    # --- Assembled path ---
    if isinstance(K, BsrMatrix):
        from .jax_amg_solver import resolve_amgx_backend

        backend = resolve_amgx_backend(cg_backend)

        # Determine K's scalar type.
        if hasattr(K.values.dtype, "_wp_scalar_type_"):
            k_scalar = K.values.dtype._wp_scalar_type_
        else:
            k_scalar = K.values.dtype  # already a scalar type (e.g. float32)

        k_vec_type = wp.types.vector(length=K.block_shape[0], dtype=k_scalar)

        # Scalar (block_dim=1) systems must not use either AMGX adapter.
        # pyamgx's AMG V-cycle smoother runs float32 internally even in dDDI
        # mode, while JAX-AMG's Warp adapter requires a block matrix.  The
        # topology Helmholtz filter is scalar, so redirect it to native CG.
        if backend in {"jaxamg", "pyamgx"} and K.block_shape[0] == 1:
            backend = "native"

        # --- Native: warp CG + Jacobi preconditioner ---
        if backend == "native":
            from warp.optim.linear import (
                cg as wp_cg,
                aslinearoperator,
            )
            from .preconditioners import build_jacobi_preconditioner

            # For native CG, K and b must have matching scalar types.
            # If K is float32 and b is float64, cast b to float32.
            if b is not None and b.dtype != k_vec_type:
                b_cast = wp.zeros(
                    len(b), dtype=k_vec_type, device=wp.get_device(device)
                )
                wp.utils.array_cast(b, b_cast)
                b = b_cast
                n_nodes = len(b)

            A_op = aslinearoperator(K)
            # Use b's scalar type (b may have been cast to float32 above)
            _b_scalar = (
                b.dtype._wp_scalar_type_
                if hasattr(b.dtype, "_wp_scalar_type_")
                else b.dtype
            )
            M_jac, _ = build_jacobi_preconditioner(
                K, device=device, _cache=_cache, solver_scalar=_b_scalar
            )

            du = _get_buf("du")
            niter, res_norm, _ = wp_cg(
                A=A_op,
                b=b,
                x=du,
                atol=effective_atol,
                maxiter=cg_max_iters,
                M=M_jac,
                check_every=check_every,
                use_cuda_graph=False,
            )
            if debug_log is not None:
                debug_log(
                    f"[DEBUG] CG+Jacobi: iters={niter}, "
                    f"res={res_norm:.3e}, atol={effective_atol:.3e}"
                )
            return du

        # --- JAX-AMG: GPU solve with zero-copy Warp/JAX value transfer ---
        if backend == "jaxamg":
            from .jax_amg_solver import solve_jaxamg

            du, niter, residual = solve_jaxamg(
                K,
                b,
                use_mixed_precision=use_mixed_precision,
                cache=_cache,
                device=device,
                rebuild_every_iter=amgx_rebuild_every_iter,
                scale_to_unit_diagonal=scale_system_to_unit_diagonal,
                max_iters=jaxamg_max_iters,
            )
            if debug_log is not None:
                precision = "float64" if use_mixed_precision else "float32"
                debug_log(
                    f"[DEBUG] JAX-AMG Krylov+AMG ({precision}): "
                    f"iters={niter}, res={residual:.3e}"
                )
            return du

        # --- pyamgx: full GPU solve (supported dDDI path) ---
        if backend == "pyamgx":
            from .amgx_solver import select_amgx_config, _get_amgx_solver

            block_dim = K.block_shape[0]

            # pyamgx stays on dDDI. Its dDFI path is not implemented by AMGX,
            # while dFFI is not reliable for the nonlinear systems solved here.
            K_amgx = bsr_upcast_f32_to_f64(K) if k_scalar == wp.float32 else K
            vecd_type = wp.types.vector(length=block_dim, dtype=wp.float64)
            if b.dtype != vecd_type:
                b_amgx = wp.empty(
                    len(b), dtype=vecd_type, device=wp.get_device(device)
                )
                wp.utils.array_cast(b, b_amgx)
            else:
                b_amgx = b

            config_type = select_amgx_config(block_dim, mode="dDDI")
            amgx_solver = _get_amgx_solver(block_dim, config_type, mode="dDDI")

            # Force hierarchy rebuild on first call per forward() pass,
            # or every iteration when amgx_rebuild_every_iter is set.
            if _cache is not None and (
                "amgx_fresh" not in _cache or amgx_rebuild_every_iter
            ):
                if amgx_rebuild_every_iter:
                    # Reset to force clean first-time upload+setup path
                    amgx_solver.reset()
                else:
                    amgx_solver._needs_rebuild = True
                _cache["amgx_fresh"] = True

            amgx_solver.setup(K_amgx)

            du_amgx = wp.zeros(
                len(b_amgx), dtype=b_amgx.dtype, device=wp.get_device(device)
            )
            niter = amgx_solver.solve(b_amgx, du_amgx)
            du = du_amgx

            # Report AMGX PCG iterations for adaptive hierarchy rebuild.
            amgx_solver.report_cg_iterations(niter)

            return du

        raise ValueError(
            f"solve_linear_system: unknown cg_backend={backend!r} "
            f"(expected 'native', 'amgx', 'jaxamg', or 'pyamgx')"
        )

    raise ValueError(
        f"solve_linear_system: unsupported configuration — "
        f"method={method!r}, K type={type(K).__name__}, mv_routine={mv_routine}"
    )


# -----------------------------------------------------------------------------
# GPU-native Newton-loop helpers (no numpy in hot path)
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _masked_squared_norm(
    r: wp.array(dtype=Any),
    mask: wp.array(dtype=Any),
    result: wp.array(dtype=wp.float64),
):
    """Compute ||r ⊙ mask||² via atomic reduction."""
    i = wp.tid()
    v = wp.cw_mul(r[i], mask[i])
    wp.atomic_add(result, 0, wp.float64(wp.dot(v, v)))


@wp.kernel(enable_backward=False)
def _scale_inplace(arr: wp.array(dtype=Any), s: Any):
    """In-place scalar multiply: arr[i] *= s."""
    i = wp.tid()
    arr[i] = s * arr[i]


@wp.kernel(enable_backward=False)
def _gradient_descent_step(
    val: wp.array(dtype=Any),
    grad: wp.array(dtype=Any),
    mask: wp.array(dtype=Any),
    lr: wp.float32,
):
    """val[i] -= lr * mask[i] * grad[i]  (all on GPU)."""
    i = wp.tid()
    val[i] = val[i] - lr * wp.cw_mul(mask[i], grad[i])


@wp.kernel(enable_backward=False)
def _clamp_inplace(
    arr: wp.array(dtype=Any),
    lo: Any,
    hi: Any,
):
    """Clamp arr[i] to [lo, hi]."""
    i = wp.tid()
    arr[i] = wp.clamp(arr[i], lo, hi)


@wp.kernel
def _simp_projection_kernel(
    inp: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    beta: wp.float32,
    eta: wp.float32,
):
    """Tanh SIMP projection: out = (tanh(β·η) + tanh(β·(x-η))) / (tanh(β·η) + tanh(β·(1-η)))."""
    i = wp.tid()
    x = inp[i]
    t_eta = wp.tanh(beta * eta)
    out[i] = (t_eta + wp.tanh(beta * (x - eta))) / (t_eta + wp.tanh(beta * (1.0 - eta)))


@wp.kernel
def _simp_projection_kernel_f64(
    inp: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
    beta: wp.float64,
    eta: wp.float64,
):
    """Tanh SIMP projection (float64): out = (tanh(β·η) + tanh(β·(x-η))) / (tanh(β·η) + tanh(β·(1-η)))."""
    i = wp.tid()
    x = inp[i]
    t_eta = wp.tanh(beta * eta)
    out[i] = (t_eta + wp.tanh(beta * (x - eta))) / (
        t_eta + wp.tanh(beta * (wp.float64(1.0) - eta))
    )


@wp.kernel(enable_backward=False)
def _axpy_f32(
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    alpha: wp.float32,
):
    """y[i] += alpha * x[i]  (f32, in-place)."""
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]


@wp.kernel(enable_backward=False)
def _axpy_f64(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    alpha: wp.float64,
):
    """y[i] += alpha * x[i]  (f64, in-place)."""
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]


def masked_norm(arr, mask, n, device, result_buf=None):
    """Compute ||arr ⊙ mask||₂ on GPU.  Returns a Python float (single scalar transfer)."""
    if result_buf is not None:
        result_buf.zero_()
        result = result_buf
    else:
        result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(_masked_squared_norm, dim=n, inputs=[arr, mask, result], device=device)
    return math.sqrt(float(result.numpy()[0]))


# -----------------------------------------------------------------------------
# GPU kernels and helpers (single-use, colocated with call sites)
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _mask_inplace(arr: wp.array(dtype=Any), mask: wp.array(dtype=Any)):
    i = wp.tid()
    arr[i] = wp.cw_mul(arr[i], mask[i])


@wp.kernel(enable_backward=False)
def _copy_masked(
    src: wp.array(dtype=Any), dst: wp.array(dtype=Any), mask: wp.array(dtype=Any)
):
    i = wp.tid()
    dst[i] = wp.cw_mul(src[i], mask[i])


@wp.kernel(enable_backward=False)
def _matvec_bc_combine(
    Kx: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    free_mask: wp.array(dtype=Any),
    bc_mask: wp.array(dtype=Any),
    alpha: Any,
    beta: Any,
):
    i = wp.tid()
    Kx_bc = wp.cw_mul(Kx[i], free_mask[i]) + wp.cw_mul(x[i], bc_mask[i])
    z[i] = alpha * Kx_bc + beta * y[i]


@wp.kernel(enable_backward=False)
def _apply_bc_to_field(
    field: wp.array(dtype=Any),
    bc_values: wp.array(dtype=Any),
    free_mask: wp.array(dtype=Any),
    bc_mask: wp.array(dtype=Any),
):
    i = wp.tid()
    field[i] = wp.cw_mul(field[i], free_mask[i]) + wp.cw_mul(bc_values[i], bc_mask[i])


@wp.kernel(enable_backward=False)
def _enforce_edge_hanging_3d(
    u: wp.array(dtype=Any),
    bc_vals: wp.array(dtype=Any),
    slave: wp.array(dtype=int),
    m1: wp.array(dtype=int),
    m2: wp.array(dtype=int),
    w: Any,
):
    i = wp.tid()
    avg = w * (u[m1[i]] + u[m2[i]])
    u[slave[i]] = avg
    bc_vals[slave[i]] = avg


@wp.kernel(enable_backward=False)
def _enforce_face_hanging_3d(
    u: wp.array(dtype=Any),
    bc_vals: wp.array(dtype=Any),
    slave: wp.array(dtype=int),
    m1: wp.array(dtype=int),
    m2: wp.array(dtype=int),
    m3: wp.array(dtype=int),
    m4: wp.array(dtype=int),
    w: Any,
):
    i = wp.tid()
    avg = w * (u[m1[i]] + u[m2[i]] + u[m3[i]] + u[m4[i]])
    u[slave[i]] = avg
    bc_vals[slave[i]] = avg


# ---------------------------------------------------------------------------
# Constraint condensation kernels: prolongation (C*v) and restriction (C^T*v)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _prolongate_edge_hanging(
    v: wp.array(dtype=Any),
    slave: wp.array(dtype=int),
    m1: wp.array(dtype=int),
    m2: wp.array(dtype=int),
    w: Any,
):
    i = wp.tid()
    v[slave[i]] = w * (v[m1[i]] + v[m2[i]])


@wp.kernel(enable_backward=False)
def _prolongate_face_hanging(
    v: wp.array(dtype=Any),
    slave: wp.array(dtype=int),
    m1: wp.array(dtype=int),
    m2: wp.array(dtype=int),
    m3: wp.array(dtype=int),
    m4: wp.array(dtype=int),
    w: Any,
):
    i = wp.tid()
    v[slave[i]] = w * (v[m1[i]] + v[m2[i]] + v[m3[i]] + v[m4[i]])


@wp.kernel(enable_backward=False)
def _restrict_edge_hanging(
    v_in: wp.array(dtype=Any),
    v_out: wp.array(dtype=Any),
    slave: wp.array(dtype=int),
    m1: wp.array(dtype=int),
    m2: wp.array(dtype=int),
    w: Any,
):
    i = wp.tid()
    val = w * v_in[slave[i]]
    wp.atomic_add(v_out, m1[i], val)
    wp.atomic_add(v_out, m2[i], val)


@wp.kernel(enable_backward=False)
def _restrict_face_hanging(
    v_in: wp.array(dtype=Any),
    v_out: wp.array(dtype=Any),
    slave: wp.array(dtype=int),
    m1: wp.array(dtype=int),
    m2: wp.array(dtype=int),
    m3: wp.array(dtype=int),
    m4: wp.array(dtype=int),
    w: Any,
):
    i = wp.tid()
    val = w * v_in[slave[i]]
    wp.atomic_add(v_out, m1[i], val)
    wp.atomic_add(v_out, m2[i], val)
    wp.atomic_add(v_out, m3[i], val)
    wp.atomic_add(v_out, m4[i], val)


@wp.kernel(enable_backward=False)
def _zero_slave_entries(
    v: wp.array(dtype=Any),
    slaves: wp.array(dtype=int),
):
    # Self-subtraction yields the additive identity for both scalar
    # (e.g. wp.float64) and vector (e.g. wp.vec3d) dtypes, sidestepping
    # Warp's lack of a generic ``type(v[0])()`` zero constructor for
    # scalar types.
    i = wp.tid()
    v[slaves[i]] = v[slaves[i]] - v[slaves[i]]


@wp.kernel(enable_backward=False)
def _squared_norm(arr: wp.array(dtype=Any), result: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_add(result, 0, wp.float64(wp.dot(arr[i], arr[i])))


@wp.kernel(enable_backward=False)
def _max_node_norm_sq(arr: wp.array(dtype=Any), result: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_max(result, 0, wp.float64(wp.dot(arr[i], arr[i])))


def check_finite(arr, n, device, result_buf=None):
    if result_buf is not None:
        result_buf.zero_()
        result = result_buf
    else:
        result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(_squared_norm, dim=n, inputs=[arr, result], device=device)
    return math.isfinite(float(result.numpy()[0]))


def max_node_norm(arr, n, device, result_buf=None):
    if result_buf is not None:
        result_buf.zero_()
        result = result_buf
    else:
        result = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(_max_node_norm_sq, dim=n, inputs=[arr, result], device=device)
    return math.sqrt(float(result.numpy()[0]))


def make_vec_array(vec_np, dim, device, dtype=None):
    if dtype is None:
        dtype = wp.vec2 if dim == 2 else wp.vec3
    if vec_np.size == 0:
        return wp.zeros(0, dtype=dtype, device=device)
    scalar = getattr(dtype, "_wp_scalar_type_", dtype)
    np_dtype = np.float64 if scalar == wp.float64 else np.float32
    vec_np = np.ascontiguousarray(vec_np, dtype=np_dtype)
    return wp.array(vec_np, dtype=dtype, device=device)


def build_bc_mask_warp(bc_mask_np, dim, device, dtype=None):
    if dtype is None:
        dtype = wp.vec2d if dim == 2 else wp.vec3d
    bc_float = bc_mask_np.astype(np.float64)
    free_float = 1.0 - bc_float
    free_mask_wp = make_vec_array(free_float, dim, device, dtype=dtype)
    bc_mask_wp = make_vec_array(bc_float, dim, device, dtype=dtype)
    return free_mask_wp, bc_mask_wp
