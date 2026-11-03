from typing import Callable, List, Tuple, Union

import numpy as np
import torch


# ----- Note ----- #
# Copied from TripoSG inference_utils.py but modified
# In TripoSG, the dimensions represent the number of cells. To make the output compatible
# with finite element computations, the dimensions are now the number of nodes.
# For precision of generated grid, I changed the precision from float16 to float32
# The precision is returned to float16 when being passed to the decoder
def _generate_dense_grid_points_gpu(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    octree_depth: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = 2**octree_depth + 1
    device = bbox_min.device

    x = torch.linspace(
        bbox_min[0], bbox_max[0], int(num_cells), dtype=torch.float32, device=device
    )
    y = torch.linspace(
        bbox_min[1], bbox_max[1], int(num_cells), dtype=torch.float32, device=device
    )
    z = torch.linspace(
        bbox_min[2], bbox_max[2], int(num_cells), dtype=torch.float32, device=device
    )

    xs, ys, zs = torch.meshgrid(x, y, z, indexing=indexing)
    xyz = torch.stack((xs, ys, zs), dim=-1)
    xyz = xyz.view(-1, 3)
    grid_size = [int(num_cells), int(num_cells), int(num_cells)]

    return xyz, grid_size, length


# Copied from TripoSG inference_utils.py but modified
def _parallel_zoom(occupancy_grid, size):
    result = torch.nn.functional.interpolate(
        occupancy_grid.unsqueeze(0).unsqueeze(0), size=size
    )
    return result.squeeze(0).squeeze(0)


# ----- Note ----- #
# Copied from TripoSG inference_utils.py but modified
# This function is originally to mark 8 cells in the denser grid around an identified cell in the coarser grid
# Now we are node-centered, and the objective of this function is to expand the identified cells to refine,
# find elements to be refined following hp-adaptive, store the refinement level of each cell,
# and find all nodes that need to be queried.
def _expand_edge_region_fast(
    edge_anchors, grid_size, existing_anchors, skip_pool=False
):
    #### Use maxpooling to expand the selected cells to refine
    if skip_pool:
        active_element_anchors = edge_anchors
    else:
        expanded_tensor = torch.zeros(
            grid_size - 1,
            grid_size - 1,
            grid_size - 1,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
        expanded_tensor[edge_anchors[:, 0], edge_anchors[:, 1], edge_anchors[:, 2]] = 1

        pooled_tensor = torch.nn.functional.max_pool3d(
            expanded_tensor.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze()

        active_element_anchors = torch.nonzero(pooled_tensor, as_tuple=False)

    final_depth = int(np.log2(existing_anchors.shape[0]))
    current_depth = int(np.log2(grid_size - 1)) + 1

    # Filter out anchors that are not cells
    active_element_anchors_final = active_element_anchors * 2 ** (
        final_depth - current_depth + 1
    )
    active_element_anchors = active_element_anchors[
        existing_anchors[
            active_element_anchors_final[:, 0],
            active_element_anchors_final[:, 1],
            active_element_anchors_final[:, 2],
        ]
        > 0
    ]

    #### Find elements to refine using hp-adaptive
    device = active_element_anchors.device

    # 1. Map fine_anchors in current grid to the final grid to query true refinement levels
    active_element_anchors_final = active_element_anchors * 2 ** (
        final_depth - current_depth + 1
    )
    levels = existing_anchors[
        active_element_anchors_final[:, 0],
        active_element_anchors_final[:, 1],
        active_element_anchors_final[:, 2],
    ]

    # 2. Initialize the dictionary: mapping 'depth' -> 'indices of cells to split in that depth's grid'
    refinements = {}
    unique_levels = torch.unique(levels)

    for lvl_tensor in unique_levels:
        lvl = lvl_tensor.item()
        mask = levels == lvl
        # Map the final coords strictly back to the cell's native 'lvl' grid
        anchors_lvl = active_element_anchors_final[mask] // (2 ** (final_depth - lvl))
        refinements[lvl] = torch.unique(anchors_lvl, dim=0)

    # 3. Enforce the 2:1 rule (1-irregular mesh) via a Ripple Refinement Cascade
    if refinements:
        max_d = max(refinements.keys())

        # Cascade downwards. Refining a cell at level d creates cells at d+1.
        # To prevent a 2-level jump, all neighbors of the level d cell must be at least level d.
        for d in range(max_d, 0, -1):
            if d not in refinements or refinements[d].shape[0] == 0:
                continue

            anchors_d = refinements[d]

            # Generate the 26 spatial neighbor offsets in the d-level grid
            offsets = torch.tensor([-1, 0, 1], device=device)
            dx, dy, dz = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
            neighbor_offsets = torch.stack([dx, dy, dz], dim=-1).reshape(-1, 3)
            neighbor_offsets = neighbor_offsets[~((neighbor_offsets == 0).all(dim=1))]

            # Broadcast to get all spatial neighbors
            neighbors_d = anchors_d.unsqueeze(1) + neighbor_offsets.unsqueeze(0)
            neighbors_d = torch.unique(neighbors_d.reshape(-1, 3), dim=0)

            # Filter out-of-bounds domain queries
            grid_size_d = 2**d
            in_bounds = (neighbors_d >= 0).all(dim=1) & (neighbors_d < grid_size_d).all(
                dim=1
            )
            neighbors_d = neighbors_d[in_bounds]

            # Instead of querying the neighbor directly, find its parent in the (d-1) grid.
            # Integer division ensures any point inside a (d-1) cell maps perfectly to its anchor.
            parents_d_minus_1 = neighbors_d // 2

            # Map these parent anchors to the final grid to query the tensor
            scale_to_final_parent = 2 ** (final_depth - (d - 1))
            parents_final = parents_d_minus_1 * scale_to_final_parent

            p_x, p_y, p_z = (
                parents_final[:, 0],
                parents_final[:, 1],
                parents_final[:, 2],
            )

            # Query the level of the parent anchors
            parent_levels = existing_anchors[p_x, p_y, p_z]

            # A neighbor violates the 2:1 rule IF AND ONLY IF its parent is a leaf cell at level (d - 1).
            # (Assuming the mesh was 1-irregular before this step, we only need to check for d-1)
            violating_parents = parents_d_minus_1[parent_levels == (d - 1)]

            if violating_parents.shape[0] > 0:
                coarse_anchors = torch.unique(violating_parents, dim=0)

                # Append the offending large cells to the refinement queue for the next loop iteration
                if (d - 1) in refinements:
                    refinements[d - 1] = torch.unique(
                        torch.cat([refinements[d - 1], coarse_anchors], dim=0), dim=0
                    )
                else:
                    refinements[d - 1] = coarse_anchors

    #### Get anchors of new cells to update existing anchors

    # The 8 offsets for the child elements (0 and 1 in each dimension)
    offsets = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    def get_new_cells_anchors(anchors, to_depth):
        # Anchors are in the grid of depth to_depth
        # Broadcast to create the 8 children for every valid parent
        # (N, 1, 3) + (1, 8, 3) -> (N, 8, 3)
        new_anchors = anchors.unsqueeze(1) + offsets.unsqueeze(0)
        # Flatten to a single list of new element anchors
        new_anchors = new_anchors.reshape(-1, 3) * 2 ** (final_depth - to_depth)
        return new_anchors

    for depth, anchors in refinements.items():
        new_anchors = get_new_cells_anchors(anchors * 2, depth + 1)
        existing_anchors[new_anchors[:, 0], new_anchors[:, 1], new_anchors[:, 2]] = (
            depth + 1
        )

    #### Get the new nodes indices that need to be queried
    # Generate the 3x3x3 local offsets for a SINGLE element (0 to 2)
    d = torch.arange(0, 3, device=device)
    gx, gy, gz = torch.meshgrid(d, d, d, indexing="ij")
    offsets = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # Shape: (27, 3)

    # Filter out "old" coarse grid nodes.
    # An offset points to an old coarse corner if ALL its components are even (0 or 2).
    is_even = offsets % 2 == 0
    all_even = is_even.all(dim=1)  # True for the 8 original corners
    new_offsets = offsets[~all_even]  # Shape: (19, 3)

    def get_new_nodes(anchors, to_depth):
        # The output anchors should be in the current_depth grid
        # Broadcast offsets to all active elements simultaneously
        # (N, 1, 3) + (1, 19, 3) -> (N, 19, 3)
        new_nodes = anchors.unsqueeze(1) + new_offsets.unsqueeze(0)

        # Flatten to a single list of coordinates
        new_nodes_flat = new_nodes.reshape(-1, 3)  # Shape: (N * 19, 3)

        # Remove duplicates (shared faces/edges between adjacent active elements)
        unique_new_nodes = torch.unique(new_nodes_flat, dim=0) * 2 ** (
            current_depth - to_depth
        )

        # Handle Boundary Conditions (if provided)
        if grid_size is not None:
            shape_tensor = torch.tensor([(grid_size - 1) * 2 + 1], device=device)
            in_bounds = (unique_new_nodes >= 0).all(dim=1) & (
                unique_new_nodes < shape_tensor
            ).all(dim=1)
            unique_new_nodes = unique_new_nodes[in_bounds]
        return unique_new_nodes

    unique_new_nodes = torch.unique(
        torch.cat(
            [
                get_new_nodes(anchors * 2, depth + 1)
                for depth, anchors in refinements.items()
            ],
            dim=0,
        ),
        dim=0,
    )

    return (unique_new_nodes, existing_anchors)


# ----- Note ----- #
# Copied from TripoSG inference_utils.py but modified
# I added the functionality of refining cells whose corners have mixed sdf signs
# borrowed from MISE https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/cython/ops/conversions/mise.pyx
# Also the logic changes from finding active nodes to finding active cells
def _find_candidates_band(
    sdf_grid: torch.Tensor, band_threshold: float, n_limits: int = -1
) -> torch.Tensor:
    """
    Returns the coordinates of all voxels in the sdf_grid where |value| < band_threshold.

    Args:
        sdf_grid (torch.Tensor): A 3D tensor of SDF values.
        band_threshold (float): The threshold below which |SDF| must be to include the voxel.
        n_limits (int): Maximum number of points to return (-1 for no limit)

    Returns:
        torch.Tensor: A 2D tensor of coordinates (N x 3) where each row is [x, y, z].
    """
    #### In-the-band nodes
    core_grid = sdf_grid
    # Create a boolean mask for all cells in the band
    in_band = torch.abs(core_grid) < band_threshold
    in_band = (
        in_band[:-1, :-1, :-1]
        + in_band[1:, :-1, :-1]
        + in_band[:-1, 1:, :-1]
        + in_band[1:, 1:, :-1]
        + in_band[:-1, :-1, 1:]
        + in_band[1:, :-1, 1:]
        + in_band[:-1, 1:, 1:]
        + in_band[1:, 1:, 1:]
    ) > 0

    # Get coordinates of all voxels in the band
    core_mesh_coords = torch.nonzero(in_band, as_tuple=False)

    if n_limits != -1 and core_mesh_coords.shape[0] > n_limits:
        print(
            f"core mesh coords {core_mesh_coords.shape[0]} is too large, limited to {n_limits}"
        )
        ind = np.random.choice(core_mesh_coords.shape[0], n_limits, True)
        core_mesh_coords = core_mesh_coords[ind]

    #### Cells with mixed signs
    # 1. Binarize the grid
    is_pos = (sdf_grid > 0).to(torch.int8)

    # 2. Sum the 8 corners for each voxel using zero-copy slicing
    corner_sums = (
        is_pos[:-1, :-1, :-1]
        + is_pos[1:, :-1, :-1]
        + is_pos[:-1, 1:, :-1]
        + is_pos[1:, 1:, :-1]
        + is_pos[:-1, :-1, 1:]
        + is_pos[1:, :-1, 1:]
        + is_pos[:-1, 1:, 1:]
        + is_pos[1:, 1:, 1:]
    )

    # 3. Active elements have mixed signs
    active_mask = (corner_sums > 0) & (corner_sums < 8)
    active_anchors = torch.nonzero(active_mask, as_tuple=False)  # Shape: (N, 3)

    # Handle the edge case where no active elements exist
    if active_anchors.shape[0] == 0:
        if core_mesh_coords.shape[0] == 0:
            print("Extremely coarse mesh. Refine everywhere")
            return torch.nonzero(
                torch.ones_like(sdf_grid[:-1, :-1, :-1], device=sdf_grid.device),
                as_tuple=False,
            )
        else:
            return core_mesh_coords
    else:
        unique_active_anchors = torch.unique(
            torch.cat([active_anchors, core_mesh_coords], dim=0), dim=0
        )
        return unique_active_anchors


def hierarchical_extraction(
    geometric_func: Callable,
    bounds: Union[Tuple[float], List[float], float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    ),
    init_octree_depth: int = 8,
    octree_depth: int = 9,
    device="cpu",
    verbose=False,
    must_query_positions: torch.Tensor = None,
    sdf_truncation: float = 0.01,
):
    """
    Args:
        sdf_truncation: SDF clamp bound of the geometric_func's decoder
            (must match ``pipe.sdf_truncation_threshold``); sets the
            refinement band width ``0.9 * sdf_truncation``.
        must_query_positions: optional ``(K, 3)`` float tensor in
            normalized-cube coordinates.  Each position should lie on
            some octree lattice (multiple of ``2/2^d`` for some
            ``d <= octree_depth``).  When refinement reaches that depth,
            every depth-``d-1`` parent touching the position is added to the
            ordinary refinement candidates.  The normal expansion path then
            creates the complete child group, queries its new nodes, and
            applies the same 2:1 ripple balancing as SDF-band refinement.
            Each position is carried through all preceding transitions, so a
            finest-lattice-only point also receives a complete, balanced
            ancestry through the requested final depth.

            This patches a defect of ``_find_candidates_band`` +
            ``_parallel_zoom``: a feature smaller than the init-depth
            lattice spacing is invisible to the band/sign detector
            because its supporting lattice point doesn't exist at init
            depth, and nearest-neighbor upsampling doesn't introduce
            it at deeper depths.  Knowing in advance where the small
            features sit (e.g. dip-node positions in phase-field
            mode), we can ensure they get directly queried at the
            correct depth.
    """
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

    bbox_min = torch.tensor(bounds[0:3]).to(device)
    bbox_max = torch.tensor(bounds[3:6]).to(device)

    # It generates the initial grid of nodes
    # We have (2**octree_depth)^3 cells, and (2**octree_depth+1)^3 nodes in a grid at most
    xyz_samples, grid_size, length = _generate_dense_grid_points_gpu(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=init_octree_depth,
        indexing="ij",
    )

    if verbose:
        print(f"Depth {init_octree_depth} query num: {xyz_samples.shape[0]}")
    grid_sdf = (
        geometric_func(xyz_samples.unsqueeze(0).to(torch.float16))
        .to(torch.float16)
        .view(grid_size[0], grid_size[1], grid_size[2])
    )

    # This is a tensor of shape (2**octree_depth,2**octree_depth,2**octree_depth)
    # Each value corresponds to the level of refinement of the cell
    existing_anchors = torch.sparse_coo_tensor(
        torch.nonzero(torch.ones_like(grid_sdf[:-1, :-1, :-1], device=device)).T
        * 2 ** (octree_depth - init_octree_depth),
        values=[init_octree_depth] * (2**init_octree_depth) ** 3,
        size=(2**octree_depth,) * 3,
        dtype=torch.int8,
        device=device,
    ).to_dense()

    # print(f'step 1 grid_sdf shape: {grid_sdf.shape}')
    # Track must-query positions and the depth at which each first appears on
    # the lattice.  At that transition, all coarser parents touching the point
    # are unioned with the SDF-band candidates.  Keeping this work inside
    # ``_expand_edge_region_fast`` is important: it creates complete sibling
    # groups and preserves the octree's 2:1 balance through its ripple cascade.
    bounds_min = float(bounds[0])
    bounds_max = float(bounds[3])
    span = bounds_max - bounds_min
    mq_active = None
    mq_depths = None
    if must_query_positions is not None and must_query_positions.shape[0] > 0:
        mq_active = must_query_positions.to(device=device, dtype=torch.float64)
        # Vectorised first-appearance depth: for each candidate depth ``d``,
        # check on-lattice via integrality of ``(pos - bounds_min)/spacing_d``.
        # Take the smallest ``d`` per position.  ~``octree_depth - init`` iters
        # of pure tensor ops (4 in the default config).
        K = mq_active.shape[0]
        mq_depths = torch.full(
            (K,), -1, dtype=torch.long, device=device
        )
        for d in range(init_octree_depth, octree_depth + 1):
            spacing_d = span / (2 ** d)
            idx_f = (mq_active - bounds_min) / spacing_d
            on_lattice = ((idx_f - idx_f.round()).abs() < 1e-6).all(dim=-1)
            update = on_lattice & (mq_depths == -1)
            mq_depths = torch.where(update, torch.full_like(mq_depths, d), mq_depths)
        # Anything still unassigned (shouldn't happen for valid positions
        # — would mean position is not on any lattice up to octree_depth):
        # default to octree_depth so the loop drops them at the deepest iter.
        mq_depths = torch.where(
            mq_depths == -1, torch.full_like(mq_depths, octree_depth), mq_depths
        )

    # A lattice node can touch up to eight cells.  Those cells can belong to
    # multiple parents at the preceding depth when an index is even.
    if mq_active is not None:
        _coff1d = torch.tensor([-1, 0], device=device, dtype=torch.long)
        _cox, _coy, _coz = torch.meshgrid(_coff1d, _coff1d, _coff1d, indexing="ij")
        _corner_cell_offsets = torch.stack([_cox, _coy, _coz], dim=-1).reshape(-1, 3)

    def _take_must_query_parents(target_depth):
        """Return depth-(target_depth-1) parents forced at this transition."""
        nonlocal mq_active, mq_depths
        if mq_active is None or mq_active.shape[0] == 0:
            return None
        # Positions that first appear deeper still need their containing-cell
        # ancestry refined now.  They remain active until their exact lattice
        # node has been created and queried at ``mq_depths``.
        sel = mq_depths >= target_depth
        if not bool(sel.any()):
            return None
        spacing_d = span / (2**target_depth)
        positions = mq_active[sel]  # (K_sel, 3)
        idx_f = (positions - bounds_min) / spacing_d
        cell = torch.floor(idx_f).long()
        on_lattice = (idx_f - idx_f.round()).abs() < 1e-6

        # A non-lattice point has one containing cell.  On a lattice plane it
        # touches the cells on both sides of that plane.  Broadcasting the
        # {-1, 0} corner offsets only along on-lattice axes covers both cases.
        corner_cells = cell.unsqueeze(1) + (
            _corner_cell_offsets.unsqueeze(0) * on_lattice.unsqueeze(1)
        )
        parents = torch.div(
            corner_cells.reshape(-1, 3), 2, rounding_mode="floor"
        )
        parent_n = 2 ** (target_depth - 1)
        valid = ((parents >= 0) & (parents < parent_n)).all(dim=-1)
        parents = torch.unique(parents[valid], dim=0)

        if verbose:
            print(
                f"[must_query] depth={target_depth}: forcing "
                f"{parents.shape[0]} parent(s) around {int(sel.sum())} active point(s)"
            )

        keep = mq_depths != target_depth
        mq_active = mq_active[keep]
        mq_depths = mq_depths[keep]
        return parents

    # Init-depth lattice positions were included in the dense initial query;
    # there is no coarser-to-finer transition to force for them.
    if mq_active is not None:
        keep = mq_depths > init_octree_depth
        mq_active = mq_active[keep]
        mq_depths = mq_depths[keep]

    for i in range(octree_depth - init_octree_depth):
        curr_octree_depth = init_octree_depth + i + 1
        grid_size = 2**curr_octree_depth + 1
        normalize_offset = 2 ** (curr_octree_depth - 1)

        # The following line expands the previous grid to the current depth and does linear interpolation
        # to fill the new nodal sdf values.
        high_res_occupancy = _parallel_zoom(grid_sdf, size=grid_size)

        # Then it identify active cells to be refined/split by finding cells whose nodal sdfs are within
        # (-0.01,0.01), or whose nodal sdfs don't have consistent signs.
        band_threshold = 0.9 * sdf_truncation
        edge_anchors = _find_candidates_band(grid_sdf, band_threshold)

        # Force known feature parents through the exact same complete-split
        # and 2:1-balanced expansion used by ordinary band candidates.
        forced_parents = _take_must_query_parents(curr_octree_depth)
        if forced_parents is not None and forced_parents.shape[0] > 0:
            edge_anchors = torch.unique(
                torch.cat([edge_anchors, forced_parents], dim=0), dim=0
            )

        # This line dilates the active cells regions (skip_pool=False), uses hp-adaptive to find cells to refine,
        # find new nodes, and update records of refinement levels
        expanded_coords, existing_anchors = _expand_edge_region_fast(
            edge_anchors,
            grid_size=int(2 ** (curr_octree_depth - 1) + 1),
            skip_pool=True,
            existing_anchors=existing_anchors,
        )

        if verbose:
            print(f"Depth {curr_octree_depth} query num: {len(expanded_coords)}")

        # Transform indices to coordinates
        expanded_coords_norm = (expanded_coords - normalize_offset) * (
            abs(bounds[0]) / normalize_offset
        )

        # Inquire SDF values of new nodes.
        query_sdf = None
        # If computing gradients in the futher, make sure to chunk the coords so that gradients are not overwritten
        query_sdf = geometric_func(
            expanded_coords_norm.unsqueeze(0).to(torch.float16)
        ).to(torch.float16)

        expanded_coords = expanded_coords.int()
        high_res_occupancy[
            expanded_coords[:, 0], expanded_coords[:, 1], expanded_coords[:, 2]
        ] = query_sdf.flatten()
        grid_sdf = high_res_occupancy

        torch.cuda.empty_cache()

    return grid_sdf, existing_anchors
