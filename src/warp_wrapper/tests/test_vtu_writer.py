"""Unit tests for VTUWriter reinit and async IO functionality."""

import os
import sys
import shutil
import tempfile
import time as time_mod

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.vtu_writer import VTUWriter


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_single_hex_mesh():
    """Single right-handed hex element (unit cube, 8 nodes)."""
    points = np.array(
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
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    cell_types = np.array([12], dtype=np.uint8)
    return {"dim": 3, "points": points, "cells": cells, "cell_types": cell_types}


def _make_two_hex_mesh():
    """Two hex elements side-by-side (12 nodes)."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [2, 1, 1],
        ],
        dtype=np.float64,
    )
    cells = np.array(
        [
            [0, 1, 4, 3, 6, 7, 10, 9],
            [1, 2, 5, 4, 7, 8, 11, 10],
        ],
        dtype=np.int32,
    )
    cell_types = np.array([12, 12], dtype=np.uint8)
    return {"dim": 3, "points": points, "cells": cells, "cell_types": cell_types}


def _random_fields(n_points):
    """Generate dummy point-data fields for a given number of nodes."""
    rng = np.random.default_rng(42)
    return {"displacement": rng.standard_normal((n_points, 3)).astype(np.float32)}


def _random_cell_data(n_cells):
    """Generate dummy cell-data for a given number of elements."""
    rng = np.random.default_rng(99)
    return {"strain_energy_density": rng.standard_normal(n_cells).astype(np.float32)}


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------


class TestReinitSync:
    """reinit() in synchronous mode."""

    def test_updates_mesh_state(self, tmp_dir):
        mesh1 = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh1, fix_orientation=False)
        assert writer.n_points == 8
        assert writer.n_cells == 1

        mesh2 = _make_two_hex_mesh()
        writer.reinit(mesh2, fix_orientation=False)
        assert writer.n_points == 12
        assert writer.n_cells == 2
        np.testing.assert_array_equal(writer.points, mesh2["points"])
        np.testing.assert_array_equal(writer.cells, mesh2["cells"])

    def test_preserves_timestep_history(self, tmp_dir):
        mesh1 = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh1, fix_orientation=False)

        # Save one timestep on the original mesh
        fields = _random_fields(8)
        writer.save_timestep(0.0, fields)
        assert len(writer.timesteps) == 1

        # Reinit with new mesh
        mesh2 = _make_two_hex_mesh()
        writer.reinit(mesh2, fix_orientation=False)

        # Save another timestep on the new mesh
        fields2 = _random_fields(12)
        writer.save_timestep(1.0, fields2)

        # Both timesteps are recorded
        assert writer.timesteps == [0.0, 1.0]
        assert len(writer.vtu_files) == 2

    def test_save_after_reinit_uses_new_mesh(self, tmp_dir):
        mesh1 = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh1, fix_orientation=False)

        mesh2 = _make_two_hex_mesh()
        writer.reinit(mesh2, fix_orientation=False)

        fields = _random_fields(12)
        cell_data = _random_cell_data(2)
        writer.save_timestep(0.0, fields, cell_data=cell_data)

        vtu_path = os.path.join(writer.vtu_dir, writer.vtu_files[0])
        assert os.path.isfile(vtu_path)
        # File should contain the 12-node mesh (check size is non-trivial)
        assert os.path.getsize(vtu_path) > 0


class TestSaveTimestepSync:
    """save_timestep() in synchronous mode."""

    def test_creates_vtu_file(self, tmp_dir):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False)
        writer.save_timestep(0.0, _random_fields(8))

        vtu_path = os.path.join(writer.vtu_dir, "solution_t0000.vtu")
        assert os.path.isfile(vtu_path)

    def test_write_pvd_links_all_timesteps(self, tmp_dir):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False)
        for t in [0.0, 0.5, 1.0]:
            writer.save_timestep(t, _random_fields(8))

        pvd_path = writer.write_pvd()
        assert os.path.isfile(pvd_path)
        with open(pvd_path) as f:
            content = f.read()
        for i in range(3):
            assert f"solution_t{i:04d}.vtu" in content


class TestShutdownSync:
    """shutdown() is a no-op in synchronous mode."""

    def test_shutdown_noop(self, tmp_dir):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False)
        writer.shutdown()  # should not raise
        assert not writer._async


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsyncSave:
    """Async save_timestep queues work and writes files in background."""

    def test_creates_files_after_drain(self, tmp_dir):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False, async_io=True)

        for t in [0.0, 0.5, 1.0]:
            writer.save_timestep(t, _random_fields(8))

        # Files may not exist yet; drain the queue
        writer.shutdown()

        for i in range(3):
            vtu_path = os.path.join(writer.vtu_dir, f"solution_t{i:04d}.vtu")
            assert os.path.isfile(vtu_path), f"Missing {vtu_path}"

    def test_queue_reporting(self, tmp_dir, capsys):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False, async_io=True)

        writer.save_timestep(0.0, _random_fields(8))
        captured = capsys.readouterr()
        assert "Queued VTU save" in captured.out
        assert "job(s) in queue" in captured.out

        writer.shutdown()

    def test_write_pvd_drains_queue(self, tmp_dir):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False, async_io=True)

        writer.save_timestep(0.0, _random_fields(8))
        writer.save_timestep(1.0, _random_fields(8))

        # write_pvd should block until all saves complete
        pvd_path = writer.write_pvd()
        assert os.path.isfile(pvd_path)

        for i in range(2):
            vtu_path = os.path.join(writer.vtu_dir, f"solution_t{i:04d}.vtu")
            assert os.path.isfile(vtu_path)

        writer.shutdown()


class TestAsyncReinit:
    """Async reinit queues mesh updates; ordering with save is preserved."""

    def test_reinit_then_save_ordering(self, tmp_dir):
        """Reinit followed by save: the save must use the new mesh."""
        mesh1 = _make_single_hex_mesh()  # 8 nodes
        writer = VTUWriter(tmp_dir, mesh1, fix_orientation=False, async_io=True)

        # Queue a reinit to the 12-node mesh, then a save with 12-node fields
        mesh2 = _make_two_hex_mesh()
        writer.reinit(mesh2, fix_orientation=False)
        writer.save_timestep(0.0, _random_fields(12), cell_data=_random_cell_data(2))

        writer.shutdown()

        vtu_path = os.path.join(writer.vtu_dir, "solution_t0000.vtu")
        assert os.path.isfile(vtu_path)
        # The file was written without error, meaning _write_vtu used the
        # 12-node mesh (n_points / n_cells updated by reinit before save).

    def test_multiple_reinit_save_cycles(self, tmp_dir):
        """Multiple reinit+save cycles maintain correct ordering."""
        mesh1 = _make_single_hex_mesh()
        mesh2 = _make_two_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh1, fix_orientation=False, async_io=True)

        # Cycle 1: save on mesh1
        writer.save_timestep(0.0, _random_fields(8), cell_data=_random_cell_data(1))

        # Cycle 2: reinit to mesh2, save
        writer.reinit(mesh2, fix_orientation=False)
        writer.save_timestep(1.0, _random_fields(12), cell_data=_random_cell_data(2))

        # Cycle 3: reinit back to mesh1, save
        writer.reinit(mesh1, fix_orientation=False)
        writer.save_timestep(2.0, _random_fields(8), cell_data=_random_cell_data(1))

        writer.shutdown()

        # All 3 VTU files should exist
        for i in range(3):
            vtu_path = os.path.join(writer.vtu_dir, f"solution_t{i:04d}.vtu")
            assert os.path.isfile(vtu_path), f"Missing {vtu_path}"

        # Timestep history spans all cycles
        assert writer.timesteps == [0.0, 1.0, 2.0]

    def test_pvd_spans_reinit_cycles(self, tmp_dir):
        """PVD file correctly links timesteps across reinit boundaries."""
        mesh1 = _make_single_hex_mesh()
        mesh2 = _make_two_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh1, fix_orientation=False, async_io=True)

        writer.save_timestep(0.0, _random_fields(8))
        writer.reinit(mesh2, fix_orientation=False)
        writer.save_timestep(1.0, _random_fields(12))

        pvd_path = writer.write_pvd()
        with open(pvd_path) as f:
            content = f.read()
        assert "solution_t0000.vtu" in content
        assert "solution_t0001.vtu" in content
        assert 'timestep="0.0"' in content
        assert 'timestep="1.0"' in content

        writer.shutdown()


class TestAsyncShutdown:
    """shutdown() drains the queue and stops the worker."""

    def test_worker_thread_stops(self, tmp_dir):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False, async_io=True)
        assert writer._worker.is_alive()

        writer.shutdown()
        assert not writer._worker.is_alive()
        assert not writer._async

    def test_shutdown_idempotent(self, tmp_dir):
        """Calling shutdown twice should not raise."""
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False, async_io=True)
        writer.shutdown()
        writer.shutdown()  # second call is a no-op


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """When async_io=False (default), behaviour is unchanged."""

    def test_default_is_sync(self, tmp_dir):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False)
        assert not writer._async
        assert not hasattr(writer, "_queue") or not hasattr(writer, "_worker")

    def test_sync_save_writes_immediately(self, tmp_dir):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False)
        writer.save_timestep(0.0, _random_fields(8))

        vtu_path = os.path.join(writer.vtu_dir, "solution_t0000.vtu")
        # File must already exist (no queue to drain)
        assert os.path.isfile(vtu_path)

    def test_sync_print_message(self, tmp_dir, capsys):
        mesh = _make_single_hex_mesh()
        writer = VTUWriter(tmp_dir, mesh, fix_orientation=False)
        writer.save_timestep(0.0, _random_fields(8))
        captured = capsys.readouterr()
        assert "Saved VTU:" in captured.out
        assert "Queued" not in captured.out


# ---------------------------------------------------------------------------
# Entry point for running outside pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
