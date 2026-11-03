import json

import numpy as np
import pytest


def test_partfield_is_optional_when_its_environment_is_absent(monkeypatch, tmp_path):
    from src.partfield_wrapper import availability

    monkeypatch.setenv("PARTFIELD_CONDA_ENV", "missing-partfield-test")
    monkeypatch.setenv("PARTFIELD_ROOT", str(tmp_path / "source"))
    monkeypatch.setattr(availability, "conda_interpreter", lambda _name: None)
    availability.partfield_status.cache_clear()
    try:
        status = availability.partfield_status()
    finally:
        availability.partfield_status.cache_clear()
    assert not status.ready
    assert "not installed" in status.summary


def test_a_pre_rename_override_finds_the_renamed_environment(monkeypatch, tmp_path):
    """``*_CONDA_ENV=implicit-*`` set before the MechFoundry rename resolves to ``mechfoundry-*``
    once the old environment is gone, as a stored compute environment does."""
    pytest.importorskip("langgraph")
    from src.agents_wrapper.fem_workflow import runtime_files
    from src.partfield_wrapper import availability

    monkeypatch.setitem(runtime_files.RENAMED_ENVIRONMENTS, "implicit-zz-test", "mechfoundry-zz-test")
    for variable in ("CONDA_ROOT", "CONDA_PREFIX_1"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("CONDA_EXE", str(tmp_path / "bin" / "conda"))
    new = tmp_path / "envs" / "mechfoundry-zz-test" / "bin" / "python"
    new.parent.mkdir(parents=True)
    new.write_text("")
    assert availability.conda_interpreter("implicit-zz-test") == new
    old = tmp_path / "envs" / "implicit-zz-test" / "bin" / "python"
    old.parent.mkdir(parents=True)
    old.write_text("")
    assert availability.conda_interpreter("implicit-zz-test") == old  # still installed: kept


def test_the_old_partfield_env_name_finds_mechfoundry_partfield(monkeypatch, tmp_path):
    """``PARTFIELD_CONDA_ENV=partfield`` (the env's name before it took the mechfoundry- prefix)
    still reaches the renamed env once ``partfield`` is gone; the default is the new name."""
    pytest.importorskip("langgraph")
    from src.partfield_wrapper import availability

    for variable in ("CONDA_ROOT", "CONDA_PREFIX_1", "PARTFIELD_CONDA_ENV"):
        monkeypatch.delenv(variable, raising=False)
    assert availability.settings()[0] == "mechfoundry-partfield"
    monkeypatch.setenv("CONDA_EXE", str(tmp_path / "bin" / "conda"))
    new = tmp_path / "envs" / "mechfoundry-partfield" / "bin" / "python"
    new.parent.mkdir(parents=True)
    new.write_text("")
    monkeypatch.setenv("PARTFIELD_CONDA_ENV", "partfield")
    assert availability.conda_interpreter(availability.settings()[0]) == new


def test_availability_probe_does_not_import_the_model_or_initialize_cuda():
    from src.partfield_wrapper import availability

    assert "import torch\n" not in availability._PROBE
    assert "import lightning\n" not in availability._PROBE
    assert "model_trainer" not in availability._PROBE
    assert "torch.cuda" not in availability._PROBE


def test_common_artifacts_do_not_require_shape_diameter(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from src.agents_wrapper.fem_workflow.segmentation_artifacts import write_segmentation

    mesh = trimesh.creation.box()
    labels = np.arange(len(mesh.faces)) % 3
    outputs = [tmp_path / "segments.npz", tmp_path / "segments.json",
               tmp_path / "segments.glb"]
    write_segmentation(
        mesh, labels, method="partfield", num_clusters=3,
        output_npz=outputs[0], output_json=outputs[1], output_glb=outputs[2],
        method_settings={"clustering": "gpu_kmeans"},
    )
    arrays = np.load(outputs[0])
    metadata = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert set(arrays.files) == {"segment_ids"}
    assert metadata["method"] == "partfield"
    assert metadata["clustering"] == "gpu_kmeans"
    assert len(metadata["segments"]) == 3
    assert outputs[2].is_file()


def test_diagnostic_ply_is_skipped_after_features_are_saved(tmp_path):
    from src.partfield_wrapper.segment import _without_diagnostic_ply

    feature_path = tmp_path / "part_feat_shape_0_batch.npy"
    diagnostic = tmp_path / "feat_pca_shape_0.ply"

    def upstream_predict_step():
        np.save(feature_path, np.ones((3, 4), dtype=np.float32))
        diagnostic.write_text("unneeded", encoding="utf-8")

    assert _without_diagnostic_ply(upstream_predict_step)() is None
    assert np.load(feature_path).shape == (3, 4)
    assert not diagnostic.exists()


def test_canonical_input_ply_is_not_written(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from src.partfield_wrapper.segment import _without_input_ply

    mesh = trimesh.creation.box()
    suppressed = tmp_path / "input_shape_0.ply"
    retained = tmp_path / "user_result.ply"
    with _without_input_ply():
        assert mesh.export(suppressed) is None
        mesh.export(retained)
    assert not suppressed.exists()
    assert retained.is_file()


def test_gpu_kmeans_accepts_multichannel_features_on_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from src.warp_wrapper.core.bc_surface_segmentation import gpu_kmeans

    features = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
    ])
    labels, centers, distances = gpu_kmeans(features, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert centers.shape == (2, 2)
    assert distances.shape == (4, 2)


def test_gpu_agglomerative_reuses_512_microclusters(monkeypatch, tmp_path):
    from src.partfield_wrapper import gpu_cluster

    features = np.arange(600 * 4, dtype=np.float32).reshape(600, 4)
    calls = []
    ward_calls = []

    def microcluster(values, count):
        calls.append((values.shape, count))
        return (np.arange(len(values), dtype=np.int32) % count,
                np.ones((count, values.shape[1]), dtype=np.float32))

    def ward_cuts(centers, labels, max_clusters):
        ward_calls.append(max_clusters)
        return np.stack([
            np.arange(len(centers), dtype=np.int16) % clusters
            for clusters in range(2, max_clusters + 1)
        ])

    monkeypatch.setattr(gpu_cluster, "_cuml_microclusters", microcluster)
    monkeypatch.setattr(gpu_cluster, "_weighted_gpu_ward_cuts", ward_cuts)
    cache = tmp_path / "microclusters.npz"
    first, metadata = gpu_cluster.gpu_agglomerative(
        features, 5, cache_path=cache
    )
    second, _ = gpu_cluster.gpu_agglomerative(features, 7, cache_path=cache)

    assert calls == [((600, 4), 512)]
    assert ward_calls == [20]
    assert np.unique(first).size == 5
    assert np.unique(second).size == 7
    assert metadata["clustering"] == "gpu_weighted_ward"
    assert metadata["microclusters"] == 512


def test_weighted_gpu_ward_matches_expanded_cpu_ward():
    torch = pytest.importorskip("torch")
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from src.partfield_wrapper.gpu_cluster import _weighted_gpu_ward

    centers = np.asarray([
        [0.0, 0.0], [0.2, 0.0], [4.0, 0.0],
        [4.2, 0.0], [9.0, 0.0], [9.2, 0.0],
    ], dtype=np.float32)
    weights = np.asarray([3, 2, 4, 2, 3, 2])
    leaves = np.repeat(np.arange(len(centers)), weights)
    actual = _weighted_gpu_ward(centers, leaves, 3)
    expected = sklearn_cluster.AgglomerativeClustering(
        n_clusters=3, linkage="ward"
    ).fit_predict(np.repeat(centers, weights, axis=0))
    assert sklearn_metrics.adjusted_rand_score(expected, actual) == 1.0


def test_weighted_gpu_ward_leaves_empty_microclusters_out():
    torch = pytest.importorskip("torch")
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from src.partfield_wrapper.gpu_cluster import _weighted_gpu_ward_cuts

    # Leaves 1 and 4 are empty; their stale centres sit between the groups.
    centers = np.asarray([
        [0.0, 0.0], [2.0, 0.0], [0.2, 0.0], [4.0, 0.0],
        [6.0, 0.0], [4.2, 0.0], [9.0, 0.0],
    ], dtype=np.float32)
    weights = np.asarray([3, 0, 2, 4, 0, 2, 3])
    leaves = np.repeat(np.arange(len(centers)), weights)
    cuts = _weighted_gpu_ward_cuts(centers, leaves, 5).cpu().numpy()
    assert (cuts[:, [1, 4]] == -1).all()
    for clusters in range(2, 6):
        actual = cuts[clusters - 2][leaves]
        expected = sklearn_cluster.AgglomerativeClustering(
            n_clusters=clusters, linkage="ward"
        ).fit_predict(np.repeat(centers, weights, axis=0))
        assert np.isfinite(actual).all()
        assert sklearn_metrics.adjusted_rand_score(expected, actual) == 1.0
