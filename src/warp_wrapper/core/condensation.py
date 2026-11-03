"""BSR condensation for hanging node constraints (C^T K C).

Provides:
- extend_bsr_pattern: extend BSR sparsity to include all entries needed
  for condensation (analogous to deal.ii AffineConstraints::condense on
  the sparsity pattern).  Called once per mesh topology, cached.
- scatter_to_extended: GPU kernel that copies K_standard values into
  K_extended positions (zero-fills extension entries).
- condense_bsr_inplace: two-pass element-local redistribution of hanging
  node contributions on the extended BSR values array (analogous to
  deal.ii distribute_local_to_global).
- Warp BSR ↔ scipy BSR conversion utilities.
"""

import numpy as np
import warp as wp
from warp.sparse import BsrMatrix, bsr_zeros, bsr_set_from_triplets
from warp._src.types import Any


# ---------------------------------------------------------------------------
# BSR binary search helper
# ---------------------------------------------------------------------------


@wp.func
def _bsr_find_col(
    columns: wp.array(dtype=wp.int32),
    lo: int,
    hi: int,
    target: int,
) -> int:
    """Binary search for column `target` in sorted columns[lo:hi).

    Returns the block index if found, -1 otherwise.
    """
    left = lo
    right = hi
    while left < right:
        mid = (left + right) / 2
        val = columns[mid]
        if val < target:
            left = mid + 1
        elif val > target:
            right = mid
        else:
            return mid
    return -1


# ---------------------------------------------------------------------------
# Column condensation kernels (Pass 1: Y = K * C)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _condense_col_2m(
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    sv: wp.array(dtype=Any),
    slave: wp.array(dtype=wp.int32),
    m1: wp.array(dtype=wp.int32),
    m2: wp.array(dtype=wp.int32),
    w: Any,
    bd2: int,
):
    """Column condensation for edge hanging (2 masters).

    For each slave s, for each nonzero column j in row s of K:
    find K[j, s] in row j and add w*K[j, s] to K[j, mk] for each master mk.
    """
    i = wp.tid()
    s = slave[i]
    mk1 = m1[i]
    mk2 = m2[i]
    row_lo = offsets[s]
    row_hi = offsets[s + 1]
    for entry in range(row_lo, row_hi):
        j = columns[entry]
        j_lo = offsets[j]
        j_hi = offsets[j + 1]
        pos_js = _bsr_find_col(columns, j_lo, j_hi, s)
        if pos_js < 0:
            continue
        src_base = pos_js * bd2
        # Master 1
        pos = _bsr_find_col(columns, j_lo, j_hi, mk1)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])
        # Master 2
        pos = _bsr_find_col(columns, j_lo, j_hi, mk2)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])


@wp.kernel(enable_backward=False)
def _condense_col_4m(
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    sv: wp.array(dtype=Any),
    slave: wp.array(dtype=wp.int32),
    m1: wp.array(dtype=wp.int32),
    m2: wp.array(dtype=wp.int32),
    m3: wp.array(dtype=wp.int32),
    m4: wp.array(dtype=wp.int32),
    w: Any,
    bd2: int,
):
    """Column condensation for face hanging (4 masters)."""
    i = wp.tid()
    s = slave[i]
    mk1 = m1[i]
    mk2 = m2[i]
    mk3 = m3[i]
    mk4 = m4[i]
    row_lo = offsets[s]
    row_hi = offsets[s + 1]
    for entry in range(row_lo, row_hi):
        j = columns[entry]
        j_lo = offsets[j]
        j_hi = offsets[j + 1]
        pos_js = _bsr_find_col(columns, j_lo, j_hi, s)
        if pos_js < 0:
            continue
        src_base = pos_js * bd2
        pos = _bsr_find_col(columns, j_lo, j_hi, mk1)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])
        pos = _bsr_find_col(columns, j_lo, j_hi, mk2)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])
        pos = _bsr_find_col(columns, j_lo, j_hi, mk3)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])
        pos = _bsr_find_col(columns, j_lo, j_hi, mk4)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])


# ---------------------------------------------------------------------------
# Row condensation kernels (Pass 2: K_cond = C^T * Y)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _condense_row_2m(
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    sv: wp.array(dtype=Any),
    slave: wp.array(dtype=wp.int32),
    m1: wp.array(dtype=wp.int32),
    m2: wp.array(dtype=wp.int32),
    w: Any,
    bd2: int,
):
    """Row condensation for edge hanging (2 masters).

    For each slave s, for each nonzero column j in row s of K:
    add w*K[s, j] to K[mk, j] for each master mk.
    """
    i = wp.tid()
    s = slave[i]
    mk1 = m1[i]
    mk2 = m2[i]
    row_lo = offsets[s]
    row_hi = offsets[s + 1]
    mk1_lo = offsets[mk1]
    mk1_hi = offsets[mk1 + 1]
    mk2_lo = offsets[mk2]
    mk2_hi = offsets[mk2 + 1]
    for entry in range(row_lo, row_hi):
        j = columns[entry]
        src_base = entry * bd2
        pos = _bsr_find_col(columns, mk1_lo, mk1_hi, j)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])
        pos = _bsr_find_col(columns, mk2_lo, mk2_hi, j)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])


@wp.kernel(enable_backward=False)
def _condense_row_4m(
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    sv: wp.array(dtype=Any),
    slave: wp.array(dtype=wp.int32),
    m1: wp.array(dtype=wp.int32),
    m2: wp.array(dtype=wp.int32),
    m3: wp.array(dtype=wp.int32),
    m4: wp.array(dtype=wp.int32),
    w: Any,
    bd2: int,
):
    """Row condensation for face hanging (4 masters)."""
    i = wp.tid()
    s = slave[i]
    mk1 = m1[i]
    mk2 = m2[i]
    mk3 = m3[i]
    mk4 = m4[i]
    row_lo = offsets[s]
    row_hi = offsets[s + 1]
    mk1_lo = offsets[mk1]
    mk1_hi = offsets[mk1 + 1]
    mk2_lo = offsets[mk2]
    mk2_hi = offsets[mk2 + 1]
    mk3_lo = offsets[mk3]
    mk3_hi = offsets[mk3 + 1]
    mk4_lo = offsets[mk4]
    mk4_hi = offsets[mk4 + 1]
    for entry in range(row_lo, row_hi):
        j = columns[entry]
        src_base = entry * bd2
        pos = _bsr_find_col(columns, mk1_lo, mk1_hi, j)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])
        pos = _bsr_find_col(columns, mk2_lo, mk2_hi, j)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])
        pos = _bsr_find_col(columns, mk3_lo, mk3_hi, j)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])
        pos = _bsr_find_col(columns, mk4_lo, mk4_hi, j)
        if pos >= 0:
            dst = pos * bd2
            for c in range(bd2):
                wp.atomic_add(sv, dst + c, w * sv[src_base + c])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def condense_bsr_inplace(K, constraint_groups, device):
    """Condense BSR matrix K in-place: K <- C^T K C.

    Two-pass redistribution of hanging node contributions directly on BSR
    values array.  No additional GPU allocations.

    Pass 1 (column condensation Y = K*C): for each slave s and master mk,
    redistribute K[j, s] -> K[j, mk] for all nonzero j in row s.

    Pass 2 (row condensation K_cond = C^T*Y): for each slave s and master mk,
    redistribute K[s, j] -> K[mk, j] for all nonzero j in row s.

    Safe in-place because masters are never slaves (octree invariant),
    so slave columns/rows are only read (never written) while master
    columns/rows are only written (via atomics).

    Parameters
    ----------
    K : BsrMatrix
        Assembled tangent matrix.  Modified in-place.
    constraint_groups : list of dict
        Constraint group descriptors (from _build_affine_constraint_gpu_data).
    device : str
        Warp device string.
    """
    if not constraint_groups:
        return

    bd = K.block_shape[0]
    bd2 = bd * bd
    # Determine K's scalar type (float32 or float64)
    k_scalar = (
        K.values.dtype._wp_scalar_type_
        if hasattr(K.values.dtype, "_wp_scalar_type_")
        else K.values.dtype
    )
    sv = wp.array(
        ptr=K.values.ptr,
        dtype=k_scalar,
        shape=(K.nnz * bd2,),
        device=K.values.device,
        copy=False,
    )

    col_kernels = {2: _condense_col_2m, 4: _condense_col_4m}
    row_kernels = {2: _condense_row_2m, 4: _condense_row_4m}

    # Pass 1: column condensation
    for g in constraint_groups:
        k = g["k"]
        inputs = (
            [K.offsets, K.columns, sv, g["slave"]]
            + g["masters"]
            + [k_scalar(g["weight"]), bd2]
        )
        wp.launch(col_kernels[k], dim=g["n"], inputs=inputs, device=device)

    wp.synchronize()

    # Pass 2: row condensation
    for g in constraint_groups:
        k = g["k"]
        inputs = (
            [K.offsets, K.columns, sv, g["slave"]]
            + g["masters"]
            + [k_scalar(g["weight"]), bd2]
        )
        wp.launch(row_kernels[k], dim=g["n"], inputs=inputs, device=device)

    wp.synchronize()


# ---------------------------------------------------------------------------
# Sparsity extension for condensation (deal.ii-style pattern augmentation)
# ---------------------------------------------------------------------------


def _find_missing_condensation_entries(K, constraint_groups):
    """Identify (row, col) block entries required for C^T*K*C but absent from K.

    Analogous to deal.ii ``AffineConstraints::condense(SparsityPattern)`` —
    for each slave ``s`` with masters ``{mk}``, every neighbour ``j`` that
    couples to ``s`` must also have entries ``K[j, mk]`` and ``K[mk, j]``
    for the redistribution to land.

    Also finds **second-order fill-in**: when two neighbouring slaves ``s1``
    and ``s2`` (``K[s1, s2] != 0``) are both eliminated, ``C^T K C`` has
    entries at ``(mk_of_s1, mk_of_s2)`` for every pair of their masters.
    Without these entries the condensation kernels silently drop
    contributions via ``_bsr_find_col`` returning -1.

    Fully GPU-vectorized via torch: builds (row, col) keys as int64 on-device
    and uses ``torch.searchsorted`` against the already-sorted ``k_keys`` for
    set-membership queries. All inputs (K.offsets, K.columns, constraint
    group slave/master arrays) are already warp arrays on GPU — we zero-copy
    them to torch and keep everything on-device.

    Returns a torch int32 tensor of shape (n_missing, 2) on the K device, or
    an empty such tensor if nothing is missing.
    """
    import torch

    device_wp = K.values.device
    torch_device = wp.device_to_torch(device_wp)

    n = K.nrow
    n_i64 = int(n)

    if not constraint_groups:
        return torch.empty((0, 2), dtype=torch.int32, device=torch_device)

    # Zero-copy: warp arrays → torch tensors on the same GPU.
    offsets_t = wp.to_torch(K.offsets).to(torch.int64)  # (nnz_alloc+1,)
    columns_t = wp.to_torch(K.columns).to(torch.int64)
    actual_nnz = int(offsets_t[n].item())

    # Global keys for all existing K entries (row * n + col, int64).
    # BSR rows are in order; within each row columns are sorted, so the
    # concatenated key array is globally sorted — exploited below via
    # torch.searchsorted.
    row_counts = offsets_t[1 : n + 1] - offsets_t[:n]  # (n,) int64
    arange_n = torch.arange(n, dtype=torch.int64, device=torch_device)
    existing_rows_i64 = torch.repeat_interleave(arange_n, row_counts)
    existing_cols_i64 = columns_t[:actual_nnz]
    k_keys = existing_rows_i64 * n_i64 + existing_cols_i64  # sorted

    # Flatten all (slave, master_k) across constraint groups on-device.
    slave_list = []
    master_list = []
    for g in constraint_groups:
        slaves_t = wp.to_torch(g["slave"]).to(torch.int64)
        masters_t = [wp.to_torch(m).to(torch.int64) for m in g["masters"]]
        k = g["k"]
        # Expand: for each slave i, emit k rows (s, mk_0), ..., (s, mk_{k-1})
        s_expanded = torch.repeat_interleave(slaves_t, k)
        # Interleave masters column-wise: shape (n_g, k) → flat (n_g*k,)
        m_stacked = torch.stack(masters_t, dim=1).reshape(-1)
        slave_list.append(s_expanded)
        master_list.append(m_stacked)
    slave_flat = (
        torch.cat(slave_list) if slave_list else
        torch.empty(0, dtype=torch.int64, device=torch_device)
    )
    master_flat = (
        torch.cat(master_list) if master_list else
        torch.empty(0, dtype=torch.int64, device=torch_device)
    )

    # --- First-order fill ---
    # For each (slave s, master mk), every neighbour j of s couples:
    #   missing candidates: (j, mk) and (mk, j)
    slave_col_starts = offsets_t[slave_flat]  # (n_sm,) int64
    slave_col_ends = offsets_t[slave_flat + 1]
    slave_col_counts = slave_col_ends - slave_col_starts
    total_s_cols = int(slave_col_counts.sum().item())

    if total_s_cols > 0:
        sm_idx_expanded = torch.repeat_interleave(
            torch.arange(slave_flat.shape[0], dtype=torch.int64, device=torch_device),
            slave_col_counts,
        )
        # For each expanded slot, the position within its slave's column list.
        cum = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=torch_device),
                slave_col_counts[:-1].cumsum(0),
            ]
        )
        local_offsets = torch.arange(
            total_s_cols, dtype=torch.int64, device=torch_device
        ) - torch.repeat_interleave(cum, slave_col_counts)
        global_col_idx = slave_col_starts[sm_idx_expanded] + local_offsets
        j_flat = columns_t[global_col_idx]
        mk_flat_expanded = master_flat[sm_idx_expanded]

        # Candidate missing keys: (j, mk) and (mk, j)
        cand_fwd = j_flat * n_i64 + mk_flat_expanded
        cand_bwd = mk_flat_expanded * n_i64 + j_flat
        cand_first = torch.unique(torch.cat([cand_fwd, cand_bwd]))  # sorted
    else:
        cand_first = torch.empty(0, dtype=torch.int64, device=torch_device)
        sm_idx_expanded = torch.empty(0, dtype=torch.int64, device=torch_device)
        j_flat = torch.empty(0, dtype=torch.int64, device=torch_device)

    # Subtract existing K entries → missing from first-order.
    if cand_first.shape[0] > 0 and k_keys.shape[0] > 0:
        idx = torch.searchsorted(k_keys, cand_first)
        idx_clamped = torch.clamp(idx, max=k_keys.shape[0] - 1)
        found = k_keys[idx_clamped] == cand_first
        missing_first = cand_first[~found]
    else:
        missing_first = cand_first

    # --- Second-order fill ---
    # When two neighbouring slaves s1, s2 are both eliminated, C^T K C creates
    # entries at (mk_s1, mk_s2) for each master pair.
    if total_s_cols > 0:
        max_k = max(g["k"] for g in constraint_groups)
        # Dense per-slave master table. n is ~O(50k), max_k ≤ 4 → small.
        slave_master_table = torch.full(
            (n, max_k), -1, dtype=torch.int64, device=torch_device
        )
        slave_k_arr = torch.zeros(n, dtype=torch.int32, device=torch_device)
        for g in constraint_groups:
            slaves_g = wp.to_torch(g["slave"]).to(torch.int64)
            k_g = g["k"]
            for col in range(k_g):
                mg = wp.to_torch(g["masters"][col]).to(torch.int64)
                slave_master_table[slaves_g, col] = mg
            slave_k_arr[slaves_g] = k_g
    
        # For each (s, mk_s) entry, over all cols j of s, check if j is a slave.
        j_is_slave = slave_k_arr[j_flat] > 0  # (total_s_cols,) bool
        if bool(j_is_slave.any().item()):
            valid = j_is_slave
            j_sub = j_flat[valid]
            sm_idx_sub = sm_idx_expanded[valid]
            mk_s_sub = master_flat[sm_idx_sub]
            # Skip self-loops (j == s).
            not_self = j_sub != slave_flat[sm_idx_sub]
            j_sub = j_sub[not_self]
            mk_s_sub = mk_s_sub[not_self]

            j_k_sub = slave_k_arr[j_sub].to(torch.int64)  # masters count per j
            total_pairs = int(j_k_sub.sum().item())
            if total_pairs > 0:
                expand_idx = torch.repeat_interleave(
                    torch.arange(j_sub.shape[0], dtype=torch.int64, device=torch_device),
                    j_k_sub,
                )
                cum_k = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int64, device=torch_device),
                        j_k_sub[:-1].cumsum(0),
                    ]
                )
                local_k_idx = torch.arange(
                    total_pairs, dtype=torch.int64, device=torch_device
                ) - torch.repeat_interleave(cum_k, j_k_sub)
                j_exp = j_sub[expand_idx]
                mk_s_exp = mk_s_sub[expand_idx]
                mk_j_exp = slave_master_table[j_exp, local_k_idx]
                cand_fwd2 = mk_s_exp * n_i64 + mk_j_exp
                cand_bwd2 = mk_j_exp * n_i64 + mk_s_exp
                cand_second = torch.unique(torch.cat([cand_fwd2, cand_bwd2]))
            else:
                cand_second = torch.empty(0, dtype=torch.int64, device=torch_device)
        else:
            cand_second = torch.empty(0, dtype=torch.int64, device=torch_device)
    else:
        cand_second = torch.empty(0, dtype=torch.int64, device=torch_device)

    # Second-order missing: exclude both K entries AND first-order additions.
    # Both k_keys and missing_first are sorted — use two independent
    # searchsorted queries instead of union.
    if cand_second.shape[0] > 0:
        if k_keys.shape[0] > 0:
            idx_k = torch.searchsorted(k_keys, cand_second)
            idx_k_c = torch.clamp(idx_k, max=k_keys.shape[0] - 1)
            in_k = k_keys[idx_k_c] == cand_second
        else:
            in_k = torch.zeros(
                cand_second.shape[0], dtype=torch.bool, device=torch_device
            )
        if missing_first.shape[0] > 0:
            idx_m = torch.searchsorted(missing_first, cand_second)
            idx_m_c = torch.clamp(idx_m, max=missing_first.shape[0] - 1)
            in_m = missing_first[idx_m_c] == cand_second
        else:
            in_m = torch.zeros(
                cand_second.shape[0], dtype=torch.bool, device=torch_device
            )
        missing_second = cand_second[~(in_k | in_m)]
    else:
        missing_second = torch.empty(0, dtype=torch.int64, device=torch_device)

    # Combine and decode to (row, col) int32 pairs.
    all_missing = torch.unique(torch.cat([missing_first, missing_second]))
    missing_rows = (all_missing // n_i64).to(torch.int32)
    missing_cols = (all_missing % n_i64).to(torch.int32)
    return torch.stack([missing_rows, missing_cols], dim=1)


def _build_scatter_mapping(K_std, K_ext):
    """Build a block-index mapping from K_std positions to K_ext positions.

    Both matrices have sorted columns per row.  K_ext is a superset of
    K_std's sparsity, so every (r, c) in K_std exists in K_ext.

    Fully GPU: zero-copies K_std/K_ext offsets and columns to torch, builds
    global (row*n + col) int64 keys on-device, and uses a single
    ``torch.searchsorted`` to locate every K_std entry in K_ext.

    Returns ``(mapping_wp, actual_nnz_std)`` where ``mapping_wp`` is a
    ``wp.array(int32)`` on the same device as ``K_std``.
    """
    import torch

    device_wp = K_std.values.device
    torch_device = wp.device_to_torch(device_wp)
    n = K_std.nrow
    n_i64 = int(n)

    std_off = wp.to_torch(K_std.offsets).to(torch.int64)
    std_col = wp.to_torch(K_std.columns).to(torch.int64)
    ext_off = wp.to_torch(K_ext.offsets).to(torch.int64)
    ext_col = wp.to_torch(K_ext.columns).to(torch.int64)

    actual_nnz_std = int(std_off[n].item())
    actual_nnz_ext = int(ext_off[n].item())

    arange_n = torch.arange(n, dtype=torch.int64, device=torch_device)

    std_row_counts = std_off[1 : n + 1] - std_off[:n]
    std_rows = torch.repeat_interleave(arange_n, std_row_counts)
    std_keys = std_rows * n_i64 + std_col[:actual_nnz_std]

    ext_row_counts = ext_off[1 : n + 1] - ext_off[:n]
    ext_rows = torch.repeat_interleave(arange_n, ext_row_counts)
    ext_keys = ext_rows * n_i64 + ext_col[:actual_nnz_ext]

    mapping = torch.searchsorted(ext_keys, std_keys).to(torch.int32).contiguous()
    mapping_wp = wp.from_torch(mapping, dtype=wp.int32)
    # Keep the backing torch tensor alive for the lifetime of mapping_wp
    # (wp.from_torch is zero-copy but does not take a reference).
    mapping_wp._torch_keepalive = mapping
    return mapping_wp, actual_nnz_std


def extend_bsr_pattern(K, constraint_groups, block_type, device):
    """Extend BSR sparsity to include all entries needed for C^T*K*C.

    Analogous to deal.ii ``AffineConstraints::condense(SparsityPattern)``.
    Called once per mesh topology; the returned objects are cached and
    reused for every Newton iteration.

    If no entries are missing, returns ``(K, None, 0)`` unchanged.

    Returns
    -------
    K_ext : BsrMatrix
        Extended matrix (same object as *K* if no extension needed).
    scatter_mapping_wp : wp.array(int32) or None
        GPU array mapping each K block index to its K_ext position.
    actual_nnz_std : int
        Exact nnz of K (launch dimension for the scatter kernel).
    """
    import torch

    missing = _find_missing_condensation_entries(K, constraint_groups)

    if missing.shape[0] == 0:
        return K, None, 0

    # -- Build existing (row, col) tensors on-GPU from K.offsets/K.columns --
    torch_device = wp.device_to_torch(K.values.device)
    n = K.nrow
    bd = K.block_shape[0]

    offsets_t = wp.to_torch(K.offsets).to(torch.int64)
    columns_t = wp.to_torch(K.columns)  # int32 zero-copy
    actual_nnz = int(offsets_t[n].item())

    row_counts = (offsets_t[1 : n + 1] - offsets_t[:n]).to(torch.int32)
    arange_n32 = torch.arange(n, dtype=torch.int32, device=torch_device)
    existing_rows = torch.repeat_interleave(arange_n32, row_counts)
    existing_cols = columns_t[:actual_nnz].to(torch.int32)

    # -- Append missing entries (zero-valued) on-GPU --
    n_missing = int(missing.shape[0])
    all_rows = torch.cat([existing_rows, missing[:, 0].contiguous()]).contiguous()
    all_cols = torch.cat([existing_cols, missing[:, 1].contiguous()]).contiguous()
    total_nnz = actual_nnz + n_missing

    # -- Build K_ext via bsr_set_from_triplets (zero-copy torch→warp for
    #    int32 rows/cols; allocate zero-valued blocks directly as warp). --
    rows_wp = wp.from_torch(all_rows, dtype=wp.int32)
    cols_wp = wp.from_torch(all_cols, dtype=wp.int32)
    vals_wp = wp.zeros(total_nnz, dtype=block_type, device=device)
    K_ext = bsr_zeros(n, K.ncol, block_type)
    bsr_set_from_triplets(K_ext, rows_wp, cols_wp, vals_wp, prune_numerical_zeros=False)

    # -- Build scatter mapping (K_std block i → K_ext block mapping[i]) --
    scatter_mapping_wp, actual_nnz_std = _build_scatter_mapping(K, K_ext)

    # Initial population: copy K_std values into K_ext.
    scatter_to_extended(K, K_ext, scatter_mapping_wp, actual_nnz_std, device)

    return K_ext, scatter_mapping_wp, actual_nnz_std


# ---------------------------------------------------------------------------
# GPU scatter kernel: K_standard values → K_extended positions
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _scatter_bsr_values_kernel(
    src_sv: wp.array(dtype=Any),
    dst_sv: wp.array(dtype=Any),
    mapping: wp.array(dtype=wp.int32),
    bd2: int,
):
    """Copy block *i* of src to block mapping[i] of dst (injective)."""
    i = wp.tid()
    src_base = i * bd2
    dst_base = mapping[i] * bd2
    for c in range(bd2):
        dst_sv[dst_base + c] = src_sv[src_base + c]


def scatter_to_extended(K_std, K_ext, scatter_mapping_wp, actual_nnz_std, device):
    """Zero K_ext values, then copy K_std blocks to their K_ext positions.

    Extension entries (those added by ``extend_bsr_pattern``) remain zero,
    ready for the subsequent ``condense_bsr_inplace`` to fill them.
    """
    bd = K_std.block_shape[0]
    bd2 = bd * bd

    # Zero all K_ext values (clears stale condensation values from last iter)
    K_ext.values.zero_()

    k_scalar = (
        K_std.values.dtype._wp_scalar_type_
        if hasattr(K_std.values.dtype, "_wp_scalar_type_")
        else K_std.values.dtype
    )
    src_sv = wp.array(
        ptr=K_std.values.ptr,
        dtype=k_scalar,
        shape=(K_std.nnz * bd2,),
        device=K_std.values.device,
        copy=False,
    )
    dst_sv = wp.array(
        ptr=K_ext.values.ptr,
        dtype=k_scalar,
        shape=(K_ext.nnz * bd2,),
        device=K_ext.values.device,
        copy=False,
    )

    wp.launch(
        _scatter_bsr_values_kernel,
        dim=actual_nnz_std,
        inputs=[src_sv, dst_sv, scatter_mapping_wp, bd2],
        device=device,
    )


# ---------------------------------------------------------------------------
# Warp BSR ↔ scipy BSR conversion
# ---------------------------------------------------------------------------


def warp_bsr_to_scipy(K):
    """Convert a warp ``BsrMatrix`` to a ``scipy.sparse.bsr_matrix``.

    Transfers data to CPU.  Returns a scipy BSR matrix.
    """
    from scipy.sparse import bsr_matrix

    offsets = K.offsets.numpy().copy()
    columns = K.columns.numpy().copy()
    values = K.values.numpy().copy()  # (nnz, block_dim, block_dim)
    bd = K.block_shape[0]
    return bsr_matrix(
        (values, columns, offsets),
        shape=(K.nrow * bd, K.ncol * bd),
    )


def scipy_bsr_to_warp(sp_mat, block_type, device="cuda"):
    """Convert a scipy sparse matrix to a warp ``BsrMatrix``.

    The matrix is converted to BSR format with blocksize matching
    *block_type*, then transferred to GPU.
    """
    from scipy.sparse import bsr_matrix as sp_bsr

    if block_type == wp.mat33d:
        bd = 3
    elif block_type == wp.mat22d:
        bd = 2
    else:
        raise ValueError(f"Unsupported block_type: {block_type}")

    sp_bsr_mat = sp_mat.tobsr(blocksize=(bd, bd))
    n_block_rows = sp_bsr_mat.shape[0] // bd
    n_block_cols = sp_bsr_mat.shape[1] // bd

    indptr = sp_bsr_mat.indptr
    indices = sp_bsr_mat.indices
    data = sp_bsr_mat.data  # (nnz, bd, bd)

    # Build row indices from indptr
    rows_list = []
    for i in range(len(indptr) - 1):
        count = indptr[i + 1] - indptr[i]
        rows_list.extend([i] * count)
    rows_np = np.array(rows_list, dtype=np.int32)

    rows_wp = wp.array(rows_np, dtype=wp.int32, device=device)
    cols_wp = wp.array(indices.astype(np.int32).copy(), dtype=wp.int32, device=device)
    blocks_wp = wp.array(
        np.ascontiguousarray(data.astype(np.float64)), dtype=block_type, device=device
    )

    K_out = bsr_zeros(n_block_rows, n_block_cols, block_type)
    bsr_set_from_triplets(K_out, rows_wp, cols_wp, blocks_wp)
    return K_out
