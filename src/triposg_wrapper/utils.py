import torch
from ..warp_wrapper.core.vtu_writer import VTUWriter
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warp as wp
import trimesh
from ..dependency.TripoSG.scripts.inference_triposg import simplify_mesh

import torch


@torch.no_grad()
def sdf2surfacemesh(
    sdf: torch.FloatTensor,
    n_faces: int = -1,
    octree_depth: int = 9,
    bounds: Union[Tuple[float], List[float], float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    ),
):
    """_summary_

    Args:
        sdf (torch.FloatTensor): SDF values on a grid (2**octree_depth+1, 2**octree_depth+1, 2**octree_depth+1)
        n_faces (int, optional): Number of faces in the mesh. Defaults to -1.
        bounds (Union[Tuple[float], List[float], float], optional): Defaults to ( -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, ).

    Returns:
        trimesh.Trimesh
    """
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

    bbox_min = torch.tensor(bounds[0:3]).to(sdf.device)
    bbox_max = torch.tensor(bounds[3:6]).to(sdf.device)
    bbox_size = bbox_max - bbox_min

    sdf_f32 = sdf.detach().to(dtype=torch.float32, device="cuda").contiguous()
    Nx, Ny, Nz = int(sdf_f32.shape[0]), int(sdf_f32.shape[1]), int(sdf_f32.shape[2])
    max_verts = max(8 * Nx * Ny, 1 << 16)
    max_tris = max(16 * Nx * Ny, 1 << 16)
    field_wp = wp.from_torch(sdf_f32, dtype=wp.float32)
    mc = wp.MarchingCubes(
        nx=Nx,
        ny=Ny,
        nz=Nz,
        max_verts=max_verts,
        max_tris=max_tris,
        device="cuda",
    )
    mc.surface(field=field_wp, threshold=0.0)
    vertices = wp.to_torch(mc.verts).cpu().numpy().astype(np.float32)
    faces = wp.to_torch(mc.indices).cpu().numpy().reshape(-1, 3).astype(np.int32)

    # wp.MarchingCubes returns vertices in voxel-index space [0, N-1].
    # Map to physical space defined by ``bounds``.
    bbox_min_np = bbox_min.cpu().numpy().astype(np.float32)
    bbox_size_np = bbox_size.cpu().numpy().astype(np.float32)
    n_minus_1 = np.array([Nx - 1, Ny - 1, Nz - 1], dtype=np.float32)
    vertices = vertices / n_minus_1 * bbox_size_np + bbox_min_np

    mesh_v_f = (vertices.astype(np.float32), np.ascontiguousarray(faces))
    mesh = trimesh.Trimesh(
        mesh_v_f[0].astype(np.float32), np.ascontiguousarray(mesh_v_f[1])
    )

    if n_faces > 0:
        mesh = simplify_mesh(mesh, n_faces)

    torch.cuda.empty_cache()
    return mesh


def _flood_fill_deepest(mask, sdf_grid):
    """Flood-fill the component of ``mask`` holding the most-negative SDF voxel.

    Args:
        mask: (Nx, Ny, Nz) bool tensor to fill within.
        sdf_grid: (Nx, Ny, Nz) SDF tensor used to place the seed.

    Returns:
        (Nx, Ny, Nz) bool tensor: the 6-connected component of the seed.
    """
    sdf_masked = sdf_grid.clone()
    sdf_masked[~mask] = 0.0  # exclude non-mask voxels from seed search
    reached = torch.zeros_like(mask)
    reached.view(-1)[sdf_masked.view(-1).argmin()] = True

    while True:
        expanded = reached.clone()
        for dim in range(3):
            fwd = torch.roll(reached, 1, dims=dim)
            fwd.select(dim, 0).fill_(False)
            bwd = torch.roll(reached, -1, dims=dim)
            bwd.select(dim, reached.shape[dim] - 1).fill_(False)
            expanded |= fwd | bwd
        expanded &= mask
        if torch.equal(expanded, reached):
            return reached
        reached = expanded


def _label_connected_components(mask):
    """Label 6-connected components of a boolean mask.

    Each voxel converges to the largest flat index within its component,
    which serves as that component's id.  Iterative max-propagation: one
    pass labels every component at once, with no per-component loop.

    Propagation runs on the mask's bounding box, since the sweep count is
    the geodesic diameter and every sweep costs the size of the grid it
    runs on -- both shrink with the crop.

    Args:
        mask: (Nx, Ny, Nz) bool tensor.

    Returns:
        (Nx, Ny, Nz) int32 tensor of component ids, ``-1`` outside ``mask``.
    """
    empty = torch.tensor(-1, dtype=torch.int32, device=mask.device)
    occupied = torch.nonzero(mask)
    lo = occupied.amin(dim=0)
    hi = occupied.amax(dim=0) + 1
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    cropped = mask[box]

    flat_index = torch.arange(
        cropped.numel(), dtype=torch.int32, device=mask.device
    ).view(cropped.shape)
    labels = torch.where(cropped, flat_index, empty)

    while True:
        expanded = labels.clone()
        for dim in range(3):
            fwd = torch.roll(labels, 1, dims=dim)
            fwd.select(dim, 0).fill_(-1)
            bwd = torch.roll(labels, -1, dims=dim)
            bwd.select(dim, labels.shape[dim] - 1).fill_(-1)
            expanded = torch.maximum(expanded, torch.maximum(fwd, bwd))
        expanded = torch.where(cropped, expanded, empty)
        if torch.equal(expanded, labels):
            break
        labels = expanded

    full = torch.full(mask.shape, -1, dtype=torch.int32, device=mask.device)
    full[box] = labels
    return full


def _select_component(labels, mask, min_volume_ratio):
    """Pick the component to keep: largest, or center-most if warranted.

    The largest component is kept unless another component's centroid sits
    closer to the grid center *and* that component has a volume of at least
    ``min_volume_ratio`` of the largest.  This rescues the real body when
    the decoder emits a larger spurious blob near the domain boundary.

    Centroid distances use per-axis normalized coordinates, so the metric
    is independent of grid resolution and aspect ratio.

    Args:
        labels: (Nx, Ny, Nz) int32 component ids from
            ``_label_connected_components``.
        mask: (Nx, Ny, Nz) bool tensor the labels were computed on.
        min_volume_ratio: Minimum volume of the center-most component
            relative to the largest for it to be preferred.

    Returns:
        (kept_mask, swapped, stats) where ``kept_mask`` is a bool tensor,
        ``swapped`` says whether the center-most component won, and
        ``stats`` is a dict of the two candidates' volumes and centroid
        distances for logging.
    """
    comp_ids, inverse = torch.unique(labels[mask], return_inverse=True)
    volumes = torch.bincount(inverse, minlength=comp_ids.numel())

    coords = torch.nonzero(mask).to(dtype=torch.float32)
    extent = torch.tensor(
        [max(n - 1, 1) for n in mask.shape], dtype=torch.float32, device=mask.device
    )
    coords /= extent
    sums = torch.zeros(
        (comp_ids.numel(), 3), dtype=torch.float32, device=mask.device
    ).index_add_(0, inverse, coords)
    centroids = sums / volumes.unsqueeze(1).to(dtype=torch.float32)
    distances = (centroids - 0.5).norm(dim=1)

    largest = int(volumes.argmax())
    nearest = int(distances.argmin())
    swapped = nearest != largest and bool(
        volumes[nearest] >= min_volume_ratio * volumes[largest]
    )
    keep = nearest if swapped else largest

    stats = {
        "n_components": int(comp_ids.numel()),
        "largest_volume": int(volumes[largest]),
        "largest_distance": float(distances[largest]),
        "nearest_volume": int(volumes[nearest]),
        "nearest_distance": float(distances[nearest]),
    }
    return labels == comp_ids[keep], swapped, stats


def filter_largest_interior_component(
    sdf_grid,
    erosion_radius=2,
    dilation_margin=None,
    domain_length=2.0,
    min_volume_ratio=0.2,
    log=None,
):
    """Remove noisy interior regions via morphological opening + labeling.

    Uses erosion to disconnect thin noise protrusions from the main body,
    labels the connected cores, keeps one of them, and finally dilates back
    to recover the true surface plus a small margin.  All voxels outside
    the dilated core are reset to ``max(sdf)``.

    The kept core is the largest one, except when a smaller component sits
    closer to the grid center and holds at least ``min_volume_ratio`` of the
    largest one's volume -- then that center-most component wins.  This
    rescues the real body when the decoder emits a larger spurious blob near
    the domain boundary.

    Erosion/dilation use 3D max-pooling (single GPU kernel each).

    The dilation margin (extra beyond erosion recovery) defaults to
    ``ceil(min(max(sdf), 1.5 * eps) / dx)`` where ``eps = 0.5 * dx``.
    Nodes farther from the core than this are likely noise.

    Args:
        sdf_grid: (Nx, Ny, Nz) SDF tensor (any dtype/device).
        erosion_radius: Voxels to erode before labeling.  Removes
            features thinner than ``2 * erosion_radius`` voxels.
        dilation_margin: Extra voxels beyond erosion recovery.  If None,
            computed from ``min(max(sdf), 1.5 * eps) / dx``.
        domain_length: Physical size of the cubic domain (for dx).
        min_volume_ratio: Volume the center-most component must hold,
            relative to the largest, to be kept in its place.  Volumes are
            counted after erosion, which costs a thin component a larger
            share of its voxels than a bulky one.
        log: Callable used for status/diagnostic messages (e.g. a project
            logger).  Defaults to ``print`` when None.

    Returns:
        Filtered SDF tensor, same shape/dtype/device.
    """
    import math

    if log is None:
        log = print

    F = torch.nn.functional
    interior = sdf_grid < 0
    if not interior.any():
        return sdf_grid

    # Compute dilation margin from SDF range and phase field epsilon
    if dilation_margin is None:
        grid_size = sdf_grid.shape[0]
        dx = domain_length / (grid_size - 1)
        eps = 0.5 * dx
        sdf_max = sdf_grid.max().item()
        dilation_distance = min(sdf_max, 1.5 * eps)
        dilation_margin = max(1, math.ceil(dilation_distance / dx))
        # print(f"  Filter: dx={dx:.6f}, eps={eps:.6f}, sdf_max={sdf_max:.6f}, "
        #   f"dilation_margin={dilation_margin} voxels")

    # 1. Morphological erosion via -max_pool3d(-x)
    interior_5d = interior.unsqueeze(0).unsqueeze(0).float()
    ek = 2 * erosion_radius + 1
    eroded = (
        (-F.max_pool3d(-interior_5d, kernel_size=ek, stride=1, padding=erosion_radius))
        .squeeze(0)
        .squeeze(0)
        .bool()
    )

    # Fallback: if erosion empties everything, skip erosion
    if not eroded.any():
        log("  WARNING: erosion emptied interior, falling back to raw interior")
        eroded = interior

    # 2. Pick the core to keep.  Flood-filling one component as a bool mask is
    # far cheaper than labeling the whole grid as int32, and it leaves only the
    # stray components to label -- which crop to a small bounding box.  Nothing
    # left over means a single component and no choice to make.
    body = _flood_fill_deepest(eroded, sdf_grid)
    leftover = eroded & ~body
    reached = body
    if leftover.any():
        labels = _label_connected_components(leftover)
        # Any id beyond the flat-index range of ``leftover`` is free for ``body``.
        body_id = torch.tensor(eroded.numel(), dtype=torch.int32, device=eroded.device)
        labels = torch.where(body, body_id, labels)
        reached, swapped, stats = _select_component(labels, eroded, min_volume_ratio)
        if swapped:
            log(
                f"  Kept center-most component over the largest of "
                f"{stats['n_components']}: volume {stats['nearest_volume']} vs "
                f"{stats['largest_volume']}, centroid distance "
                f"{stats['nearest_distance']:.4f} vs {stats['largest_distance']:.4f}"
            )

    # 3. Dilate core back by erosion_radius + dilation_margin
    total_dilate = erosion_radius + dilation_margin
    dk = 2 * total_dilate + 1
    neighborhood = (
        F.max_pool3d(
            reached.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=dk,
            stride=1,
            padding=total_dilate,
        )
        .squeeze(0)
        .squeeze(0)
        .bool()
    )

    # 4. Reset everything outside the neighborhood to max(sdf)
    far_outside = ~neighborhood
    n_noise = int((interior & ~neighborhood).sum())
    n_reset = int(far_outside.sum())

    result = sdf_grid.clone()
    sdf_max = sdf_grid.max()
    result[far_outside] = sdf_max
    log(
        f"  Filtered {n_noise} noisy interior voxels, "
        f"reset {n_reset} far-field voxels to {sdf_max.item():.6f}"
    )
    return result


def sdf2octree(cell_tensor, sdf_tensor, domain_length=1.0):
    """
    Generates a Hex mesh directly from an explicit octree cell tracking tensor,
    including hanging node constraints for non-conforming interfaces.

    Assumes a 2:1 balanced octree.

    Args:
        cell_tensor (Tensor): (2**depth, 2**depth, 2**depth) tensor where non-zero
                              values represent the refinement level 'm' of an element anchor.
        sdf_tensor (Tensor): (2**depth+1, 2**depth+1, 2**depth+1) tensor of nodal SDF values.
        domain_length (float): The physical size of the cubic domain.

    Returns:
        nodes (Tensor): (V, 3) Unique vertices in physical space.
        node_sdf (Tensor): (V,) SDF values strictly for the active nodes.
        elements (Tensor): (M, 8) Hex connectivity indices.
        edge_hanging (Tensor): (H_e, 3) Edge hanging node constraints. Each row is
            [hanging_node_idx, neighbor1_idx, neighbor2_idx] where the hanging node
            sits at the midpoint of the parent edge. Neighbors are sorted.
        face_hanging (Tensor): (H_f, 5) Face hanging node constraints. Each row is
            [hanging_node_idx, n1, n2, n3, n4] where the hanging node sits at the
            center of the parent face. Neighbors are sorted.
    """
    device = cell_tensor.device

    # 1. Extract valid elements
    # Find all non-zero entries (the anchors) and their corresponding refinement levels
    active_mask = cell_tensor > 0
    anchors = torch.nonzero(active_mask)  # Shape: (N, 3) grid indices
    levels = cell_tensor[active_mask]  # Shape: (N,) refinement levels (m)
    depth = int(np.log2(cell_tensor.shape[0]))

    # 2. Determine the physical footprint of each element in the grid
    # An element at level m spans 2**(depth - m) units in the finest grid
    scales = 2 ** (depth - levels)  # Shape: (N,)

    # 3. Generate the 8 corners for every element
    # Standard VTK/ParaView Hex Node Ordering offsets
    offsets = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    # Scale the local [0,1] offsets by the cell's grid scale
    # anchors: (N, 1, 3) | offsets: (1, 8, 3) | scales: (N, 1, 1)
    scaled_offsets = offsets.unsqueeze(0) * scales.view(-1, 1, 1)
    element_corners = anchors.unsqueeze(1) + scaled_offsets  # Shape: (N, 8, 3)

    # 4. Extract unique nodes and build the connectivity array
    # Flatten all generated corners into a single list
    flat_corners = element_corners.reshape(-1, 3)

    # return_inverse natively provides the element connectivity mapping
    unique_grid_coords, inverse_indices = torch.unique(
        flat_corners, dim=0, return_inverse=True
    )
    elements = inverse_indices.reshape(-1, 8)

    # 5. Calculate physical coordinates for the nodes
    # The grid ranges from 0 to 2**depth, so spacing is domain_length / 2**depth
    spacing = domain_length / (2**depth)
    nodes = unique_grid_coords.float() * spacing - domain_length / 2

    # 6. Extract the exact SDF values for these active nodes
    # Since sdf_tensor is (2**depth+1)^3, the grid coordinates map to it perfectly
    i, j, k = (
        unique_grid_coords[:, 0],
        unique_grid_coords[:, 1],
        unique_grid_coords[:, 2],
    )
    node_sdf = sdf_tensor[i, j, k]

    # 7. Detect hanging nodes at non-conforming octree interfaces
    # Build a 3D lookup table: grid coordinate -> node index (-1 if absent)
    grid_size = 2**depth + 1
    node_lookup = torch.full(
        (grid_size, grid_size, grid_size), -1, device=device, dtype=torch.long
    )
    node_lookup[i, j, k] = torch.arange(len(unique_grid_coords), device=device)

    # --- 7a. Edge hanging nodes ---
    # 12 edges of a hexahedron defined by pairs of local corner indices
    hex_edges = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom face edges
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top face edges
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # vertical edges
        ],
        device=device,
        dtype=torch.long,
    )

    # Gather edge endpoint grid coordinates: (N, 12, 3)
    ep1 = element_corners[:, hex_edges[:, 0], :]
    ep2 = element_corners[:, hex_edges[:, 1], :]

    # Midpoint = (ep1 + ep2) / 2; only valid when the sum is even (integer midpoint).
    # Elements at the finest level (scale=1) produce odd sums -> filtered out.
    mid_sum = ep1 + ep2  # (N, 12, 3)
    is_even = (mid_sum % 2 == 0).all(dim=-1)  # (N, 12)
    edge_mids = (mid_sum // 2).clamp(0, 2**depth)  # (N, 12, 3)

    # Lookup midpoint node indices
    flat_em = edge_mids.reshape(-1, 3)
    mid_nidx = node_lookup[flat_em[:, 0], flat_em[:, 1], flat_em[:, 2]].reshape(-1, 12)

    # Lookup endpoint node indices
    flat_ep1 = ep1.reshape(-1, 3)
    flat_ep2 = ep2.reshape(-1, 3)
    ep1_nidx = node_lookup[flat_ep1[:, 0], flat_ep1[:, 1], flat_ep1[:, 2]].reshape(
        -1, 12
    )
    ep2_nidx = node_lookup[flat_ep2[:, 0], flat_ep2[:, 1], flat_ep2[:, 2]].reshape(
        -1, 12
    )

    # Hanging node exists where the midpoint is integer AND is a mesh node
    edge_hang_mask = is_even & (mid_nidx >= 0)  # (N, 12)

    if edge_hang_mask.any():
        hang_idx = mid_nidx[edge_hang_mask]
        nbr1 = ep1_nidx[edge_hang_mask]
        nbr2 = ep2_nidx[edge_hang_mask]
        # Sort the 2 neighbors for consistent deduplication
        nbrs = torch.stack([nbr1, nbr2], dim=1)
        nbrs, _ = nbrs.sort(dim=1)
        edge_hang_raw = torch.cat([hang_idx.unsqueeze(1), nbrs], dim=1)  # (H, 3)
        edge_hanging = torch.unique(edge_hang_raw, dim=0)
    else:
        edge_hanging = torch.empty((0, 3), device=device, dtype=torch.long)

    # --- 7b. Face hanging nodes ---
    # 6 faces of a hexahedron defined by 4 local corner indices each
    hex_faces = torch.tensor(
        [
            [0, 1, 2, 3],  # bottom (z-)
            [4, 5, 6, 7],  # top (z+)
            [0, 1, 5, 4],  # front (y-)
            [1, 2, 6, 5],  # right (x+)
            [2, 3, 7, 6],  # back (y+)
            [3, 0, 4, 7],  # left (x-)
        ],
        device=device,
        dtype=torch.long,
    )

    # Gather face corner grid coordinates: (N, 6, 4, 3)
    fc = element_corners[:, hex_faces.reshape(-1), :].reshape(-1, 6, 4, 3)

    # Face center = sum of 4 corners / 4; only valid when divisible by 4.
    # Elements at scale <= 2 with non-divisible sums are filtered out.
    fc_sum = fc.sum(dim=2)  # (N, 6, 3)
    is_div4 = (fc_sum % 4 == 0).all(dim=-1)  # (N, 6)
    face_ctrs = (fc_sum // 4).clamp(0, 2**depth)  # (N, 6, 3)

    # Lookup face center node indices
    flat_fc = face_ctrs.reshape(-1, 3)
    fc_nidx = node_lookup[flat_fc[:, 0], flat_fc[:, 1], flat_fc[:, 2]].reshape(-1, 6)

    # Lookup face corner node indices: (N, 6, 4)
    flat_fcc = fc.reshape(-1, 3)
    fcc_nidx = node_lookup[flat_fcc[:, 0], flat_fcc[:, 1], flat_fcc[:, 2]].reshape(
        -1, 6, 4
    )

    # Hanging node exists where the face center is integer AND is a mesh node
    face_hang_mask = is_div4 & (fc_nidx >= 0)  # (N, 6)

    if face_hang_mask.any():
        fh_idx = fc_nidx[face_hang_mask]  # (H,)
        fh_nbrs = fcc_nidx[face_hang_mask]  # (H, 4)
        # Sort the 4 neighbors for consistent deduplication
        fh_nbrs, _ = fh_nbrs.sort(dim=1)
        face_hang_raw = torch.cat([fh_idx.unsqueeze(1), fh_nbrs], dim=1)  # (H, 5)
        face_hanging = torch.unique(face_hang_raw, dim=0)
    else:
        face_hanging = torch.empty((0, 5), device=device, dtype=torch.long)

    return nodes, node_sdf, elements, edge_hanging, face_hanging


def octree2vtu(nodes, node_sdf, elements, output_dir="octree_output"):
    """
    Exports the generated PyTorch octree mesh to ParaView VTU format
    using the custom VTUWriter class.

    Args:
        nodes (torch.Tensor): (V, 3) tensor of vertices.
        node_sdf (torch.Tensor): (V,) tensor of SDF values.
        elements (torch.Tensor): (M, 8) tensor of hexahedral connectivity.
        output_dir (str): Folder to save the .vtu and .pvd files.
    """
    print(
        f"Preparing mesh for export ({nodes.shape[0]} nodes, {elements.shape[0]} elements)..."
    )

    # 1. Safely move PyTorch tensors from CUDA to CPU NumPy arrays
    points_np = nodes.detach().cpu().numpy()
    cells_np = elements.detach().cpu().numpy()
    sdf_np = node_sdf.detach().cpu().numpy()

    # 2. Setup the VTK cell types
    # 12 is the universal VTK identifier for VTK_HEXAHEDRON
    n_elements = cells_np.shape[0]
    cell_types_np = np.full(n_elements, 12, dtype=np.uint8)

    # 3. Construct the mesh_info dictionary expected by VTUWriter
    mesh_info = {
        "dim": 3,
        "points": points_np,
        "cells": cells_np,
        "cell_types": cell_types_np,
    }

    # 4. Initialize the writer
    # Note: VTUWriter will automatically call _fix_hex_orientation()
    # ensuring your elements don't look twisted in ParaView!
    writer = VTUWriter(
        output_dir=output_dir, mesh_info=mesh_info, fix_orientation=False
    )

    # 5. Define the Nodal Fields (PointData)
    fields = {"SDF": sdf_np}

    # 6. Save the data
    # We use time=0.0 since this is a static mesh
    writer.save_timestep(time=0.0, fields=fields)

    # 7. Write the PVD collection file
    pvd_path = writer.write_pvd()

    print(f"Export successful! You can now open {pvd_path} in ParaView.")
