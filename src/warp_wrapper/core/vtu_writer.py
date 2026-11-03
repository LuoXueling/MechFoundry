"""
VTU/PVTU Writer for ParaView Visualization

Writes VTK UnstructuredGrid format (.vtu files) with multiple fields
and creates a ParaView Collection file (.pvd) linking all timesteps.
"""

import os
import numpy as np
import torch
from xml.etree import ElementTree as ET
import base64
import threading
import queue as queue_mod


def _encode_array_base64(arr):
    """Encode a contiguous numpy array to VTK Base64 with a UInt64 size header."""
    arr = np.ascontiguousarray(arr)
    data_bytes = arr.tobytes()
    header = np.array([len(data_bytes)], dtype=np.uint64).tobytes()
    return base64.b64encode(header + data_bytes).decode("ascii")


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from a triangle soup.

    ``wp.MarchingCubes`` does not emit vertex normals, so we compute them
    here for downstream VTU/GLB writers.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]  # (F, 3, 3)
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    face_n = np.cross(edge1, edge2)  # (F, 3) — magnitude = 2 * area
    normals = np.zeros_like(v)
    np.add.at(normals, f[:, 0], face_n)
    np.add.at(normals, f[:, 1], face_n)
    np.add.at(normals, f[:, 2], face_n)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (normals / norm).astype(np.float64)


class VTUWriter:
    """
    Write VTU files (VTK UnstructuredGrid XML format) for ParaView visualization

    Features:
    - Saves multiple fields per timestep
    - Creates subdirectory for organized output
    - Generates PVTU collection file for time series
    - Supports 2D and 3D meshes
    """

    def __init__(
        self, output_dir, mesh_info, fix_orientation=True, async_io=False, log=None
    ):
        """
        Initialize VTU writer

        Args:
            output_dir: Base output directory
            mesh_info: Dictionary with:
                - dim: dimension (2 or 3)
                - points: node coordinates array (n_nodes, dim)
                - cells: element connectivity array (n_elements, nodes_per_elem)
                - cell_types: VTK cell type for each element
            fix_orientation: If True, ensure hex cells have right-handed orientation
            async_io: If True, reinit and save_timestep run in a background thread
        """
        self.output_dir = output_dir
        self.vtu_dir = os.path.join(output_dir, "vtu")
        os.makedirs(self.vtu_dir, exist_ok=True)

        self.mesh_info = mesh_info
        self.dim = mesh_info["dim"]
        self.points = np.asarray(mesh_info["points"], dtype=np.float64)
        self.cells = np.asarray(mesh_info["cells"], dtype=np.int32)
        self.cell_types = np.asarray(mesh_info["cell_types"], dtype=np.uint8)

        if self.dim == 3 and fix_orientation:
            self._fix_hex_orientation()

        self.n_points = len(self.points)
        self.n_cells = len(self.cells)

        # Store timesteps for PVTU file
        self.timesteps = []
        self.vtu_files = []
        self._log = log if log is not None else print

        # Async IO support
        self._async = async_io
        if self._async:
            self._queue = queue_mod.Queue()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    def reinit(self, mesh_info, fix_orientation=True):
        """Update mesh info so this writer can be reused with a new mesh.

        Timestep history is preserved for PVD continuity across mesh changes.
        """
        if self._async:
            self._queue.put(("reinit", mesh_info, fix_orientation))
        else:
            self._do_reinit(mesh_info, fix_orientation)

    def _do_reinit(self, mesh_info, fix_orientation):
        """Apply mesh info update (runs on caller or worker thread)."""
        self.mesh_info = mesh_info
        self.dim = mesh_info["dim"]
        self.points = np.asarray(mesh_info["points"], dtype=np.float64)
        self.cells = np.asarray(mesh_info["cells"], dtype=np.int32)
        self.cell_types = np.asarray(mesh_info["cell_types"], dtype=np.uint8)
        if self.dim == 3 and fix_orientation:
            self._fix_hex_orientation()
        self.n_points = len(self.points)
        self.n_cells = len(self.cells)

    def shutdown(self):
        """Wait for pending jobs and stop the background worker."""
        if self._async:
            self._queue.join()
            self._queue.put(None)  # sentinel to stop the worker
            self._worker.join()
            self._async = False

    def _worker_loop(self):
        """Background thread: process reinit and save jobs in FIFO order."""
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            try:
                if job[0] == "reinit":
                    _, mesh_info, fix_orientation = job
                    self._do_reinit(mesh_info, fix_orientation)
                elif job[0] == "compute_save":
                    _, filepath, fields_fn = job
                    fields, cell_data = fields_fn()
                    self._write_vtu(filepath, fields, cell_data)
                elif job[0] == "save":
                    _, filepath, fields, cell_data = job
                    self._write_vtu(filepath, fields, cell_data)
            except Exception as e:
                import traceback

                self._log(f"VTUWriter async error: {e}\n{traceback.format_exc()}")
            finally:
                self._queue.task_done()

    def save_timestep(self, time, fields, cell_data=None):
        """
        Save a single timestep with multiple fields

        Args:
            time: Time value
            fields: Dictionary of field_name -> numpy array
                   Each array should have shape (n_points, n_components)
            cell_data: Optional dictionary of cell-level fields

        Example:
            writer.save_timestep(0.5, {
                'displacement': u_array,  # (n_points, 3)
                'strain': strain_array,    # (n_points, 6)
            })
        """
        # Create filename
        filename = f"solution_t{len(self.timesteps):04d}.vtu"
        filepath = os.path.join(self.vtu_dir, filename)

        # Record timestep
        self.timesteps.append(time)
        self.vtu_files.append(filename)

        if self._async:
            if callable(fields):
                # fields is a 0-arg callable returning (point_data_dict, cell_data_dict).
                # The worker calls it to compute fields on the background thread.
                self._queue.put(("compute_save", filepath, fields))
            else:
                self._queue.put(("save", filepath, fields, cell_data))
            pending = self._queue.qsize()
            self._log(f"Queued VTU save: {filepath} ({pending} job(s) in queue)")
        else:
            if callable(fields):
                fields, cell_data = fields()
            self._write_vtu(filepath, fields, cell_data)
            self._log(f"Saved VTU: {filepath}")

    def _encode_array_base64(self, arr):
        return _encode_array_base64(arr)

    def _fix_hex_orientation(self):
        """Ensure hexahedral cells follow VTK right-handed orientation (GPU-parallel)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        points_t = torch.as_tensor(self.points, dtype=torch.float64, device=device)
        cells_t = torch.as_tensor(self.cells, dtype=torch.long, device=device)

        p0 = points_t[cells_t[:, 0]]
        p1 = points_t[cells_t[:, 1]]
        p3 = points_t[cells_t[:, 3]]
        p4 = points_t[cells_t[:, 4]]

        v1 = p1 - p0
        v3 = p3 - p0
        v4 = p4 - p0
        volume = torch.sum(torch.linalg.cross(v1, v3) * v4, dim=1)

        flip_mask = volume < 0.0
        flips = int(flip_mask.sum().item())

        if flips:
            cells_t[flip_mask, 2], cells_t[flip_mask, 3] = (
                cells_t[flip_mask, 3].clone(),
                cells_t[flip_mask, 2].clone(),
            )
            cells_t[flip_mask, 6], cells_t[flip_mask, 7] = (
                cells_t[flip_mask, 7].clone(),
                cells_t[flip_mask, 6].clone(),
            )
            self._log(
                f"VTUWriter: corrected orientation for {flips} hex cells to avoid twists."
            )

        self.cells = cells_t.cpu().numpy().astype(np.int32)

    def _write_vtu(self, filepath, fields, cell_data=None):
        """Write a single VTU file using Base64 Binary encoding."""
        root = ET.Element("VTKFile")
        root.set("type", "UnstructuredGrid")
        root.set("version", "1.0")
        root.set("byte_order", "LittleEndian")
        root.set("header_type", "UInt64")

        unstructured_grid = ET.SubElement(root, "UnstructuredGrid")
        piece = ET.SubElement(unstructured_grid, "Piece")
        piece.set("NumberOfPoints", str(self.n_points))
        piece.set("NumberOfCells", str(self.n_cells))

        # --- Points ---
        points_elem = ET.SubElement(piece, "Points")
        data_array = ET.SubElement(points_elem, "DataArray")
        data_array.set("type", "Float32")  # Float32 cuts size in half vs Float64
        data_array.set("Name", "Points")
        data_array.set("NumberOfComponents", "3")
        data_array.set("format", "binary")

        if self.dim == 2:
            points_3d = np.column_stack(
                [self.points[:, 0], self.points[:, 1], np.zeros(self.n_points)]
            )
        else:
            points_3d = self.points

        data_array.text = self._encode_array_base64(points_3d.astype(np.float32))

        # --- Cells ---
        cells_elem = ET.SubElement(piece, "Cells")

        # Connectivity
        connectivity = ET.SubElement(cells_elem, "DataArray")
        connectivity.set("type", "Int32")
        connectivity.set("Name", "connectivity")
        connectivity.set("format", "binary")
        connectivity.text = self._encode_array_base64(
            self.cells.ravel().astype(np.int32)
        )

        # Offsets
        offsets = ET.SubElement(cells_elem, "DataArray")
        offsets.set("type", "Int32")
        offsets.set("Name", "offsets")
        offsets.set("format", "binary")
        # Efficient cumulative sum for offsets
        offset_value = 8 if self.dim == 3 else 4
        offset_values = np.arange(
            offset_value, offset_value * self.n_cells + 1, offset_value, dtype=np.int32
        )
        offsets.text = self._encode_array_base64(offset_values)

        # Cell types
        types = ET.SubElement(cells_elem, "DataArray")
        types.set("type", "UInt8")
        types.set("Name", "types")
        types.set("format", "binary")
        types.text = self._encode_array_base64(self.cell_types.astype(np.uint8))

        # --- Point Data (Fields) ---
        point_data = ET.SubElement(piece, "PointData")

        for field_name, field_data in fields.items():
            field_array = ET.SubElement(point_data, "DataArray")
            # Using Float32 for output fields usually provides plenty of precision for viz
            field_array.set("type", "Float32")
            field_array.set("Name", field_name)

            n_components = 1 if len(field_data.shape) == 1 else field_data.shape[1]
            field_array.set("NumberOfComponents", str(n_components))
            field_array.set("format", "binary")

            field_array.text = self._encode_array_base64(
                field_data.astype(np.float32).ravel()
            )

        # --- Cell Data ---
        if cell_data:
            cell_data_elem = ET.SubElement(piece, "CellData")
            for name, data in cell_data.items():
                array = ET.SubElement(cell_data_elem, "DataArray")
                array.set("type", "Float32")
                array.set("Name", name)

                n_components = 1 if data.ndim == 1 else data.shape[1]
                array.set("NumberOfComponents", str(n_components))
                array.set("format", "binary")
                array.text = self._encode_array_base64(data.astype(np.float32).ravel())

        # --- Write to file ---
        # Using ElementTree directly without minidom.toprettyxml()
        # minidom is extremely slow and memory-heavy for large XML trees
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)  # Python 3.9+ native indent
        tree.write(filepath, encoding="utf-8", xml_declaration=True)

    def write_pvd(self, pvd_name="solution.pvd"):
        """
        Write PVD collection file linking all timesteps

        This file can be opened directly in ParaView to load the entire
        time series.
        """
        if self._async:
            self._queue.join()

        pvd_path = os.path.join(self.output_dir, pvd_name)

        root = ET.Element("VTKFile")
        root.set("type", "Collection")
        root.set("version", "1.0")
        root.set("byte_order", "LittleEndian")

        collection = ET.SubElement(root, "Collection")

        vtu_rel = os.path.relpath(self.vtu_dir, self.output_dir)
        for time, vtu_file in zip(self.timesteps, self.vtu_files):
            dataset = ET.SubElement(collection, "DataSet")
            dataset.set("timestep", str(time))
            dataset.set("file", f"{vtu_rel}/{vtu_file}")

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(pvd_path, encoding="unicode", xml_declaration=True)

        self._log(f"Saved PVD collection: {pvd_path}")
        return pvd_path


class VTUSurfaceWriter:
    """Async marching-cubes surface extraction with VTU + GLB output.

    Given a flat SDF array on a structured grid, extracts the zero level set
    via ``wp.MarchingCubes``, writes the triangle surface to VTU (ParaView)
    and GLB, and collects timesteps into a PVD file.

    Callers that already have a pre-extracted surface mesh (e.g.
    :class:`LatentShapeOpt`, which extracts the mesh once per latent
    rebuild for analytical SDF computation) should use
    :meth:`save_from_mesh` to skip the redundant marching cubes step.
    """

    def __init__(
        self,
        output_dir,
        grid_shape=None,
        bounds_lo=None,
        bounds_hi=None,
        write_vtu=True,
        write_glb=True,
        write_glb_deformed=True,
        simplify_n_faces=-1,
        async_io=False,
        log=None,
    ):
        """
        Args:
            output_dir: Base output directory.
            grid_shape: (Nx, Ny, Nz) number of nodes along each axis. Required
                only for :meth:`save_timestep` (grid-based marching cubes).
                Pass ``None`` when only :meth:`save_from_mesh` will be used
                (e.g. :class:`LatentShapeOpt` which has an octree, not a
                structured grid).
            bounds_lo: (3,) physical coordinates of the grid origin. Required
                only when ``grid_shape`` is provided.
            bounds_hi: (3,) physical coordinates of the grid far corner.
                Required only when ``grid_shape`` is provided.
            write_vtu: Write triangle surface VTU files.
            write_glb: Write undeformed GLB files.
            write_glb_deformed: Write deformed GLB files.
            simplify_n_faces: Simplify to N faces (-1 = no simplification).
            async_io: If True, extraction + writing run on a background thread.
            log: Logging callable; defaults to ``print``.
        """
        self.output_dir = output_dir
        if grid_shape is not None:
            self.grid_shape = tuple(grid_shape)
            self.bounds_lo = np.asarray(bounds_lo, dtype=np.float64)
            self.bounds_hi = np.asarray(bounds_hi, dtype=np.float64)
            self.spacing = (self.bounds_hi - self.bounds_lo) / (
                np.array(self.grid_shape) - 1
            )
        else:
            self.grid_shape = None
            self.bounds_lo = None
            self.bounds_hi = None
            self.spacing = None

        self.write_vtu = write_vtu
        self.write_glb = write_glb
        self.write_glb_deformed = write_glb_deformed
        self.simplify_n_faces = simplify_n_faces

        self.vtu_dir = os.path.join(output_dir, "vtu_surface")
        self.glb_dir = os.path.join(output_dir, "glb")
        self.glb_deformed_dir = os.path.join(output_dir, "glb_deformed")
        if write_vtu:
            os.makedirs(self.vtu_dir, exist_ok=True)
        if write_glb:
            os.makedirs(self.glb_dir, exist_ok=True)
        if write_glb_deformed:
            os.makedirs(self.glb_deformed_dir, exist_ok=True)

        self.timesteps = []
        self.vtu_files = []
        self._log = log if log is not None else print

        self._async = async_io
        if self._async:
            self._queue = queue_mod.Queue()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_timestep(self, time, sdf_flat, grid_fields=None):
        """Extract the zero level set from *sdf_flat* and save VTU + GLB.

        Args:
            time: Timestep value for PVD indexing.
            sdf_flat: 1-D array (or warp/torch tensor) of nodal SDF values,
                length must equal ``prod(grid_shape)``.
            grid_fields: Optional dict of ``name -> array`` where each array
                has shape ``(prod(grid_shape),)`` or ``(prod(grid_shape), C)``.
                Fields are trilinearly interpolated at the surface vertices
                and included in the VTU output.
        """
        if self.grid_shape is None:
            raise RuntimeError(
                "VTUSurfaceWriter.save_timestep requires a structured grid; "
                "this writer was constructed without grid_shape. Use "
                "save_from_mesh() instead."
            )
        filename = f"surface_t{len(self.timesteps):04d}"
        self.timesteps.append(time)
        self.vtu_files.append(filename + ".vtu")

        sdf_np = np.asarray(sdf_flat, dtype=np.float64).ravel().copy()
        # Snapshot grid fields
        gf = {}
        if grid_fields:
            for name, arr in grid_fields.items():
                gf[name] = np.asarray(arr, dtype=np.float64).copy()

        if self._async:
            self._queue.put(("extract_save", filename, sdf_np, gf))
            pending = self._queue.qsize()
            self._log(
                f"Queued surface extraction: {filename} " f"({pending} job(s) in queue)"
            )
        else:
            self._do_extract_save(filename, sdf_np, gf)

    def save_from_mesh(self, time, vertices, faces, point_data=None):
        """Save VTU + GLB from a pre-extracted surface mesh.

        Used by callers that already ran marching cubes (e.g.
        :class:`LatentShapeOpt` extracts the mesh once for analytical
        SDF computation and reuses it here).

        Args:
            time: Timestep value for PVD indexing.
            vertices: ``(V, 3)`` numpy array of vertex positions in
                physical coordinates.
            faces: ``(F, 3)`` numpy array of int triangle indices.
            point_data: Optional dict of ``name -> array`` where each
                array has shape ``(V,)`` or ``(V, C)`` — values already
                sampled at the surface vertices. No interpolation is
                performed. If ``"displacement"`` is present it will also
                be used for the deformed GLB.
        """
        filename = f"surface_t{len(self.timesteps):04d}"
        self.timesteps.append(time)
        self.vtu_files.append(filename + ".vtu")

        verts_np = np.asarray(vertices, dtype=np.float64).copy()
        faces_np = np.asarray(faces, dtype=np.int32).copy()
        pd = {}
        if point_data:
            for name, arr in point_data.items():
                pd[name] = np.asarray(arr, dtype=np.float64).copy()

        if self._async:
            self._queue.put(("save_mesh", filename, verts_np, faces_np, pd))
            pending = self._queue.qsize()
            self._log(
                f"Queued surface save: {filename} " f"({pending} job(s) in queue)"
            )
        else:
            self._do_save_from_mesh(filename, verts_np, faces_np, pd)

    def save_from_mesh_lazy(self, time, compute_fn):
        """Queue a surface save whose heavy work is deferred to the worker.

        ``compute_fn`` is called inside the async worker (or inline if the
        writer was constructed with ``async_io=False``) and must return
        ``(vertices, faces, point_data)`` — numpy arrays / dict as in
        :meth:`save_from_mesh`.  This lets the caller log a ``Queued surface
        save`` message immediately while an expensive computation (e.g.
        octree interpolation of the displacement field) runs asynchronously.
        """
        filename = f"surface_t{len(self.timesteps):04d}"
        self.timesteps.append(time)
        self.vtu_files.append(filename + ".vtu")

        if self._async:
            self._queue.put(("save_mesh_lazy", filename, compute_fn))
            pending = self._queue.qsize()
            self._log(
                f"Queued surface save: {filename} ({pending} job(s) in queue)"
            )
        else:
            v, f, pd = compute_fn()
            self._do_save_from_mesh(
                filename,
                np.asarray(v, dtype=np.float64).copy(),
                np.asarray(f, dtype=np.int32).copy(),
                {k: np.asarray(a, dtype=np.float64).copy() for k, a in (pd or {}).items()},
            )

    def write_pvd(self, pvd_name="surface.pvd"):
        """Write PVD collection file linking all surface timesteps."""
        if self._async:
            self._queue.join()

        pvd_path = os.path.join(self.output_dir, pvd_name)
        root = ET.Element("VTKFile")
        root.set("type", "Collection")
        root.set("version", "1.0")
        root.set("byte_order", "LittleEndian")

        collection = ET.SubElement(root, "Collection")
        vtu_rel = os.path.relpath(self.vtu_dir, self.output_dir)
        for time, vtu_file in zip(self.timesteps, self.vtu_files):
            dataset = ET.SubElement(collection, "DataSet")
            dataset.set("timestep", str(time))
            dataset.set("file", f"{vtu_rel}/{vtu_file}")

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(pvd_path, encoding="unicode", xml_declaration=True)
        self._log(f"Saved PVD collection: {pvd_path}")
        return pvd_path

    def shutdown(self):
        """Wait for pending jobs and stop the background worker."""
        if self._async:
            self._queue.join()
            self._queue.put(None)
            self._worker.join()
            self._async = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _worker_loop(self):
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            try:
                kind = job[0]
                if kind == "extract_save":
                    _, filename, sdf_np, gf = job
                    self._do_extract_save(filename, sdf_np, gf)
                elif kind == "save_mesh":
                    _, filename, verts_np, faces_np, pd = job
                    self._do_save_from_mesh(filename, verts_np, faces_np, pd)
                elif kind == "save_mesh_lazy":
                    _, filename, compute_fn = job
                    v, f, pd = compute_fn()
                    self._do_save_from_mesh(
                        filename,
                        np.asarray(v, dtype=np.float64).copy(),
                        np.asarray(f, dtype=np.int32).copy(),
                        {k: np.asarray(a, dtype=np.float64).copy()
                         for k, a in (pd or {}).items()},
                    )
            except Exception as e:
                self._log(f"VTUSurfaceWriter async error: {e}")
            finally:
                self._queue.task_done()

    def _do_extract_save(self, filename, sdf_np, grid_fields):
        import warp as wp

        sdf_3d = sdf_np.reshape(self.grid_shape).astype(np.float32)
        Nx, Ny, Nz = self.grid_shape
        max_verts = max(8 * Nx * Ny, 1 << 16)
        max_tris = max(16 * Nx * Ny, 1 << 16)
        try:
            sdf_wp = wp.from_numpy(sdf_3d, dtype=wp.float32, device="cuda")
            mc = wp.MarchingCubes(
                nx=Nx,
                ny=Ny,
                nz=Nz,
                max_verts=max_verts,
                max_tris=max_tris,
                device="cuda",
            )
            mc.surface(field=sdf_wp, threshold=0.0)
            verts_idx = wp.to_torch(mc.verts).cpu().numpy().astype(np.float64)
            faces = (
                wp.to_torch(mc.indices).cpu().numpy().reshape(-1, 3).astype(np.int32)
            )
        except Exception as exc:
            self._log(f"Surface extraction failed for {filename}: {exc}")
            return

        if len(verts_idx) == 0 or len(faces) == 0:
            self._log(f"Surface extraction empty for {filename} — skipping")
            return

        # Index space [0, N-1] → physical space.
        vertices = verts_idx * self.spacing + self.bounds_lo

        # Interpolate grid fields at surface vertices before handing the
        # mesh to the shared writer (which now only understands
        # pre-sampled per-vertex data).
        point_data = {}
        if grid_fields:
            for name, arr in grid_fields.items():
                point_data[name] = self._trilinear_interpolate(arr, vertices)
        self._do_save_from_mesh(filename, vertices, faces, point_data)

    def _do_save_from_mesh(self, filename, vertices, faces, point_data):
        # Optional mesh simplification. NOTE: simplification drops the
        # vertex ↔ point_data correspondence, so it is not supported when
        # point_data is provided.
        if self.simplify_n_faces > 0 and len(faces) > self.simplify_n_faces:
            if point_data:
                raise RuntimeError(
                    "save_from_mesh: simplify_n_faces is incompatible with "
                    "point_data — simplification would break vertex↔value "
                    "correspondence."
                )
            import trimesh

            mesh_raw = trimesh.Trimesh(
                vertices=vertices.astype(np.float32),
                faces=np.ascontiguousarray(faces),
            )
            mesh_raw = mesh_raw.simplify_quadric_decimation(self.simplify_n_faces)
            vertices = np.asarray(mesh_raw.vertices, dtype=np.float64)
            faces = np.asarray(mesh_raw.faces, dtype=np.int32)

        # Compute vertex normals (area-weighted average of face normals).
        normals = _compute_vertex_normals(vertices, faces)

        if point_data is None:
            point_data = {}

        # Write VTU (triangle surface mesh with interpolated fields)
        if self.write_vtu:
            vtu_path = os.path.join(self.vtu_dir, filename + ".vtu")
            self._write_surface_vtu(
                vtu_path,
                vertices,
                faces,
                normals,
                point_data,
            )
            # self._log(f"Saved surface VTU: {vtu_path}")

        import trimesh

        # Write undeformed GLB
        if self.write_glb:
            glb_path = os.path.join(self.glb_dir, filename + ".glb")
            mesh = trimesh.Trimesh(
                vertices=vertices.astype(np.float32),
                faces=np.ascontiguousarray(faces),
                vertex_normals=normals.astype(np.float32),
            )
            mesh.export(glb_path)
            # self._log(f"Saved GLB: {glb_path}")

        # Write deformed GLB (vertices + displacement)
        if self.write_glb_deformed and "displacement" in point_data:
            glb_def_path = os.path.join(
                self.glb_deformed_dir,
                filename + ".glb",
            )
            mesh_def = trimesh.Trimesh(
                vertices=(vertices + point_data["displacement"]).astype(np.float32),
                faces=np.ascontiguousarray(faces),
            )
            mesh_def.export(glb_def_path)
            # self._log(f"Saved deformed GLB: {glb_def_path}")

    def _trilinear_interpolate(self, grid_data, query_points):
        """Trilinear interpolation of grid data at arbitrary query positions.

        Args:
            grid_data: Flat array, shape ``(N,)`` or ``(N, C)`` where
                ``N = prod(grid_shape)``.
            query_points: ``(M, 3)`` physical coordinates.

        Returns:
            ``(M,)`` or ``(M, C)`` interpolated values.
        """
        Nx, Ny, Nz = self.grid_shape
        gc = (query_points - self.bounds_lo) / self.spacing  # (M, 3)
        gc[:, 0] = np.clip(gc[:, 0], 0, Nx - 1 - 1e-8)
        gc[:, 1] = np.clip(gc[:, 1], 0, Ny - 1 - 1e-8)
        gc[:, 2] = np.clip(gc[:, 2], 0, Nz - 1 - 1e-8)

        i0 = np.floor(gc[:, 0]).astype(np.intp)
        j0 = np.floor(gc[:, 1]).astype(np.intp)
        k0 = np.floor(gc[:, 2]).astype(np.intp)
        i1 = np.minimum(i0 + 1, Nx - 1)
        j1 = np.minimum(j0 + 1, Ny - 1)
        k1 = np.minimum(k0 + 1, Nz - 1)

        s = gc[:, 0] - i0
        t = gc[:, 1] - j0
        u = gc[:, 2] - k0

        scalar = grid_data.ndim == 1
        if scalar:
            data = grid_data.reshape(Nx, Ny, Nz)
        else:
            data = grid_data.reshape(Nx, Ny, Nz, -1)
            s, t, u = s[:, None], t[:, None], u[:, None]

        result = (
            data[i0, j0, k0] * (1 - s) * (1 - t) * (1 - u)
            + data[i1, j0, k0] * s * (1 - t) * (1 - u)
            + data[i0, j1, k0] * (1 - s) * t * (1 - u)
            + data[i1, j1, k0] * s * t * (1 - u)
            + data[i0, j0, k1] * (1 - s) * (1 - t) * u
            + data[i1, j0, k1] * s * (1 - t) * u
            + data[i0, j1, k1] * (1 - s) * t * u
            + data[i1, j1, k1] * s * t * u
        )
        return result

    @staticmethod
    def _write_surface_vtu(filepath, vertices, faces, normals=None, point_data=None):
        """Write a triangle surface mesh as VTU."""
        n_points = len(vertices)
        n_cells = len(faces)

        root = ET.Element("VTKFile")
        root.set("type", "UnstructuredGrid")
        root.set("version", "1.0")
        root.set("byte_order", "LittleEndian")
        root.set("header_type", "UInt64")

        grid = ET.SubElement(root, "UnstructuredGrid")
        piece = ET.SubElement(grid, "Piece")
        piece.set("NumberOfPoints", str(n_points))
        piece.set("NumberOfCells", str(n_cells))

        # Points
        pts_elem = ET.SubElement(ET.SubElement(piece, "Points"), "DataArray")
        pts_elem.set("type", "Float32")
        pts_elem.set("Name", "Points")
        pts_elem.set("NumberOfComponents", "3")
        pts_elem.set("format", "binary")
        pts_elem.text = _encode_array_base64(vertices.astype(np.float32))

        # Cells
        cells_elem = ET.SubElement(piece, "Cells")

        conn = ET.SubElement(cells_elem, "DataArray")
        conn.set("type", "Int32")
        conn.set("Name", "connectivity")
        conn.set("format", "binary")
        conn.text = _encode_array_base64(faces.ravel().astype(np.int32))

        offsets_da = ET.SubElement(cells_elem, "DataArray")
        offsets_da.set("type", "Int32")
        offsets_da.set("Name", "offsets")
        offsets_da.set("format", "binary")
        offsets_da.text = _encode_array_base64(
            np.arange(3, 3 * n_cells + 1, 3, dtype=np.int32)
        )

        types_da = ET.SubElement(cells_elem, "DataArray")
        types_da.set("type", "UInt8")
        types_da.set("Name", "types")
        types_da.set("format", "binary")
        types_da.text = _encode_array_base64(
            np.full(n_cells, 5, dtype=np.uint8)  # VTK_TRIANGLE
        )

        # Point data
        pd = ET.SubElement(piece, "PointData")
        if normals is not None:
            n_da = ET.SubElement(pd, "DataArray")
            n_da.set("type", "Float32")
            n_da.set("Name", "Normals")
            n_da.set("NumberOfComponents", "3")
            n_da.set("format", "binary")
            n_da.text = _encode_array_base64(normals.astype(np.float32))
        if point_data:
            for name, arr in point_data.items():
                da = ET.SubElement(pd, "DataArray")
                da.set("type", "Float32")
                da.set("Name", name)
                n_comp = 1 if arr.ndim == 1 else arr.shape[1]
                da.set("NumberOfComponents", str(n_comp))
                da.set("format", "binary")
                da.text = _encode_array_base64(arr.astype(np.float32).ravel())

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(filepath, encoding="utf-8", xml_declaration=True)


def extract_mesh_info_from_warp(geo, space, dim):
    """
    Extract mesh information from Warp FEM geometry

    Args:
        geo: Warp geometry object
        space: Warp function space
        dim: dimension (2 or 3)

    Returns:
        Dictionary with mesh information for VTUWriter
    """
    # Get node positions — .numpy() on vec arrays already returns (N, dim)
    # For float64 quad meshes, node_positions() fails (float32 kernel);
    # use geometry positions array directly.
    _geo_positions = getattr(geo, "positions", None)
    _pos_dtype_name = getattr(getattr(_geo_positions, "dtype", None), "__name__", "")
    if _geo_positions is not None and _pos_dtype_name in ("vec2d", "vec3d"):
        points = np.asarray(_geo_positions.numpy(), dtype=np.float64).copy()
    else:
        points = space.node_positions().numpy().copy()

    # Build cell connectivity via vectorized meshgrid arithmetic
    if dim == 2:
        nx, ny = geo.res[0], geo.res[1]
        vtk_cell_type = 9  # VTK_QUAD
        s = ny + 1

        i_idx, j_idx = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        i_f, j_f = i_idx.ravel(), j_idx.ravel()

        n0 = j_f + i_f * s
        n1 = (j_f + 1) + i_f * s
        n2 = (j_f + 1) + (i_f + 1) * s
        n3 = j_f + (i_f + 1) * s
        # Order: (p1-p0) aligns +x, (p3-p0) aligns +y
        cells = np.column_stack([n0, n3, n2, n1]).astype(np.int32)
    else:  # 3D
        nx, ny, nz = geo.res[0], geo.res[1], geo.res[2]
        vtk_cell_type = 12  # VTK_HEXAHEDRON
        stride = (ny + 1) * (nz + 1)

        i_idx, j_idx, k_idx = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        i_f, j_f, k_f = i_idx.ravel(), j_idx.ravel(), k_idx.ravel()

        def ni(i, j, k):
            return k + j * (nz + 1) + i * stride

        n0 = ni(i_f, j_f, k_f)
        n1 = ni(i_f + 1, j_f, k_f)
        n2 = ni(i_f + 1, j_f + 1, k_f)
        n3 = ni(i_f, j_f + 1, k_f)
        n4 = ni(i_f, j_f, k_f + 1)
        n5 = ni(i_f + 1, j_f, k_f + 1)
        n6 = ni(i_f + 1, j_f + 1, k_f + 1)
        n7 = ni(i_f, j_f + 1, k_f + 1)
        cells = np.column_stack([n0, n1, n2, n3, n4, n5, n6, n7]).astype(np.int32)

    n_cells = cells.shape[0]
    cell_types = np.full(n_cells, vtk_cell_type, dtype=np.uint8)

    return {
        "dim": dim,
        "points": points,
        "cells": cells,
        "cell_types": cell_types,
    }


def _solve_affine_gradient(
    points_elem: np.ndarray, disp_elem: np.ndarray
) -> np.ndarray:
    """
    Solve for the best-fit affine displacement gradient over an element using least squares.

    Args:
        points_elem: (n_nodes_per_elem, dim) array of element node coordinates
        disp_elem: (n_nodes_per_elem, dim) array of nodal displacements

    Returns:
        Gradient matrix of shape (dim, dim)
    """
    dim = points_elem.shape[1]
    ones = np.ones((points_elem.shape[0], 1), dtype=np.float64)
    design = np.hstack((points_elem, ones))
    grad = np.zeros((dim, dim), dtype=np.float64)

    for comp in range(dim):
        coeffs, _, _, _ = np.linalg.lstsq(design, disp_elem[:, comp], rcond=None)
        grad[comp, :] = coeffs[:dim]

    return grad


def compute_node_gradients(u_field, geo, space, dim, mesh_cache=None, disp_tensor=None):
    """
    Compute displacement gradients at nodes using per-element affine reconstruction.
    GPU-accelerated via batched torch.linalg.lstsq.

    Args:
        u_field: Displacement field (Warp field). Ignored when disp_tensor is provided.
        geo: Warp geometry. Used only when mesh_cache is not yet populated.
        space: Warp function space. Used only when mesh_cache is not yet populated.
        dim: spatial dimension
        mesh_cache: Optional dict with pre-computed 'points' and 'cells' torch tensors.
                    If provided, skips extract_mesh_info_from_warp. The dict is populated
                    on first call when passed as an empty dict {}.
        disp_tensor: Optional torch.Tensor (n_nodes, dim) with displacement DOF values.
                     When provided, bypasses the warp field read — use this when calling
                     from a background thread with a pre-snapshotted array.

    Returns:
        torch.Tensor of shape (n_nodes, dim, dim) on GPU with averaged nodal gradients.
    """
    if mesh_cache is not None and "points" in mesh_cache:
        points = mesh_cache["points"]
        cells = mesh_cache["cells"]
        # Derive device from cached tensors so CPU-cache callers stay on CPU
        device = points.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mesh_info = extract_mesh_info_from_warp(geo, space, dim)
        points = torch.as_tensor(
            np.asarray(mesh_info["points"], dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        cells = torch.as_tensor(
            np.asarray(mesh_info["cells"], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        if mesh_cache is not None:
            mesh_cache["points"] = points
            mesh_cache["cells"] = cells

    if disp_tensor is not None:
        disp = disp_tensor.to(device=device, dtype=torch.float64)
    else:
        import warp as wp
        disp = wp.to_torch(u_field.dof_values).to(dtype=torch.float64)

    # Gather per-element points: (n_cells, npe, dim)
    elem_pts = points[cells]

    # Build or reuse the per-element pseudo-inverse of the design matrix.
    # The design matrix [coords | 1] depends only on mesh geometry (fixed), so we
    # compute design_pinv = (D^T D)^{-1} D^T once and cache it in mesh_cache.
    if mesh_cache is not None and "design_pinv" not in mesh_cache:
        ones_d = torch.ones(*elem_pts.shape[:2], 1, dtype=torch.float64, device=device)
        design = torch.cat([elem_pts, ones_d], dim=2)  # (n_cells, npe, dim+1)
        DTD = torch.bmm(design.mT, design)  # (n_cells, dim+1, dim+1)
        mesh_cache["design_pinv"] = torch.bmm(
            torch.linalg.inv(DTD), design.mT
        )  # (n_cells, dim+1, npe)

    elem_disp = disp[cells]  # (n_cells, npe, dim)

    if mesh_cache is not None and "design_pinv" in mesh_cache:
        # Fast path: one batched matmul (replaces lstsq, ~60× faster on CPU)
        coeffs = torch.bmm(
            mesh_cache["design_pinv"], elem_disp
        )  # (n_cells, dim+1, dim)
    else:
        # Slow path: lstsq — used when mesh_cache is None (one-shot callers)
        ones_d = torch.ones(*elem_pts.shape[:2], 1, dtype=torch.float64, device=device)
        design = torch.cat([elem_pts, ones_d], dim=2)
        coeffs = torch.linalg.lstsq(design, elem_disp).solution

    # Extract spatial gradient: rows = spatial coords, cols = displacement components
    elem_grads = coeffs[:, :dim, :].permute(0, 2, 1)  # (n_cells, dim, dim)

    # Scatter-average to nodes
    npe = cells.shape[1]
    n_nodes = points.shape[0]
    flat_nodes = cells.reshape(-1)  # (n_cells * npe,)
    flat_grads = elem_grads.unsqueeze(1).expand(-1, npe, -1, -1).reshape(-1, dim, dim)

    node_sum = torch.zeros(n_nodes, dim, dim, dtype=torch.float64, device=device)
    counts = torch.zeros(n_nodes, dtype=torch.float64, device=device)

    idx = flat_nodes.unsqueeze(1).unsqueeze(2).expand(-1, dim, dim)
    node_sum.scatter_add_(0, idx, flat_grads)
    counts.scatter_add_(0, flat_nodes, torch.ones_like(flat_nodes, dtype=torch.float64))

    grads = node_sum / counts.unsqueeze(1).unsqueeze(2)
    grads = torch.where(grads.abs() < 5e-8, torch.zeros_like(grads), grads)
    return grads  # torch (N, dim, dim) on GPU


def compute_strain_tensor(u_field, geo, space, dim, gradients=None):
    """
    Compute small-strain tensor in Voigt notation at mesh nodes.
    GPU-accelerated via torch indexing.

    Args:
        gradients: torch.Tensor (N, dim, dim) on GPU, or None to compute from scratch

    Returns:
        numpy array — (N, 3) for 2D or (N, 6) for 3D
    """
    if gradients is None:
        gradients = compute_node_gradients(u_field, geo, space, dim)

    if dim == 2:
        strain = torch.stack(
            [
                gradients[:, 0, 0],
                gradients[:, 1, 1],
                0.5 * (gradients[:, 0, 1] + gradients[:, 1, 0]),
            ],
            dim=1,
        )
    else:
        strain = torch.stack(
            [
                gradients[:, 0, 0],
                gradients[:, 1, 1],
                gradients[:, 2, 2],
                0.5 * (gradients[:, 0, 1] + gradients[:, 1, 0]),
                0.5 * (gradients[:, 1, 2] + gradients[:, 2, 1]),
                0.5 * (gradients[:, 0, 2] + gradients[:, 2, 0]),
            ],
            dim=1,
        )

    strain = torch.where(strain.abs() < 5e-8, torch.zeros_like(strain), strain)
    return strain.detach().cpu().numpy()


def compute_stress_tensor(u_field, geo, space, dim, mu, kappa, gradients=None):
    """
    Compute Cauchy stress tensor in Voigt notation at mesh nodes using the
    Neo-Hookean constitutive law. GPU-accelerated via batched torch linalg.

    Args:
        gradients: torch.Tensor (N, dim, dim) on GPU, or None to compute from scratch

    Returns:
        numpy array — (N, 3) for 2D or (N, 6) for 3D
    """
    if gradients is None:
        gradients = compute_node_gradients(u_field, geo, space, dim)

    device = gradients.device
    N = gradients.shape[0]
    I = (
        torch.eye(dim, dtype=torch.float64, device=device)
        .unsqueeze(0)
        .expand(N, -1, -1)
    )

    F = I + gradients  # (N, dim, dim)
    J = torch.linalg.det(F)  # (N,)

    # Inverted-element fallback: instead of raising, mark bad nodes with a
    # sentinel and continue.  The bad nodes' stress entries are overwritten
    # at the end of this function with the original (signed) J value
    # broadcast to all components, so the writer can still emit a VTU and
    # the inversion shows up as a glaring outlier in visualization rather
    # than killing the run.
    bad_mask = J <= 0.0
    bad_J = None
    if torch.any(bad_mask):
        n_bad = int(bad_mask.sum().item())
        worst = float(J[bad_mask].min().item())
        print(
            f"  [VTUWriter] WARNING: {n_bad} inverted nodes (worst J={worst:.3e}); "
            f"sentinel-filling stress at those nodes"
        )
        bad_J = J.clone()
        # Replace J at bad nodes with 1.0 so the math below doesn't NaN.
        J = torch.where(bad_mask, torch.ones_like(J), J)
        # Replace F at bad nodes with identity so inv/det don't NaN.
        F = torch.where(bad_mask[:, None, None], I, F)

    F_inv_T = torch.linalg.inv(F).mT  # (N, dim, dim)
    C = torch.bmm(F.mT, F)  # (N, dim, dim)
    I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1)  # (N,)

    if dim == 2:
        I1 = I1 + 1.0  # plane-strain convention: trace(C) + 1

    J_m23 = J.pow(-2.0 / 3.0)  # (N,)

    # Scalars preserve the uniform-material fast path; vectors allow the
    # adaptive solver's already sampled nodal material to drive output too.
    mu_t = torch.as_tensor(mu, dtype=gradients.dtype, device=device)
    kappa_t = torch.as_tensor(kappa, dtype=gradients.dtype, device=device)
    if mu_t.ndim:
        mu_t = mu_t[:, None, None]
    if kappa_t.ndim:
        kappa_t = kappa_t[:, None, None]
    P = (
        mu_t * J_m23[:, None, None] * (F - (I1 / 3.0)[:, None, None] * F_inv_T)
        + kappa_t * J.log()[:, None, None] * F_inv_T
    )  # (N, dim, dim)

    sigma = torch.bmm(P, F.mT) / J[:, None, None]  # (N, dim, dim)
    sigma_sym = 0.5 * (sigma + sigma.mT)

    if dim == 2:
        stress = torch.stack(
            [
                sigma_sym[:, 0, 0],
                sigma_sym[:, 1, 1],
                sigma_sym[:, 0, 1],
            ],
            dim=1,
        )
    else:
        stress = torch.stack(
            [
                sigma_sym[:, 0, 0],
                sigma_sym[:, 1, 1],
                sigma_sym[:, 2, 2],
                sigma_sym[:, 0, 1],
                sigma_sym[:, 1, 2],
                sigma_sym[:, 0, 2],
            ],
            dim=1,
        )

    stress = torch.where(stress.abs() < 1e-6, torch.zeros_like(stress), stress)
    if bad_J is not None:
        # Sentinel-fill all stress components at inverted nodes with the
        # original (signed) J value so they're obvious outliers in ParaView.
        stress = torch.where(
            bad_mask[:, None],
            bad_J[:, None].to(stress.dtype).expand_as(stress),
            stress,
        )
    return stress.detach().cpu().numpy()
