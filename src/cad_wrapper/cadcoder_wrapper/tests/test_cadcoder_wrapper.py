import numpy as np
import pytest

from src.cad_wrapper.cadcoder_wrapper.availability import cadcoder_status

PLATE = """import cadquery as cq
r = cq.Workplane("XY").box(2.0, 2.0, 0.2).faces(">Z").workplane().hole(0.4)
"""


def test_missing_environment_is_not_ready(monkeypatch, tmp_path):
    cadcoder_status.cache_clear()
    monkeypatch.setenv("CADCODER_CONDA_ENV", "mechfoundry-cadcoder-missing")
    monkeypatch.setenv("CADCODER_ROOT", str(tmp_path / "missing"))
    try:
        status = cadcoder_status()
        assert not status.ready
        assert status.model_dump()["environment"] == "mechfoundry-cadcoder-missing"
    finally:
        cadcoder_status.cache_clear()


def test_answers_are_parsed_as_upstream_does():
    generate = pytest.importorskip("src.cad_wrapper.cadcoder_wrapper.generate")
    boxed = "<think>plan</think>\\boxed{```python\nr = 1\n```}"
    assert generate.extract_code(boxed) == "r = 1"
    assert generate.extract_code("text\n```python\nr = 2\n```") == "r = 2"
    with pytest.raises(ValueError):
        generate.extract_code("no code at all")


@pytest.mark.parametrize("code", [
    "import os\nr = os.getcwd()",
    "import subprocess",
    "from socket import socket",
    "r = open('/etc/passwd').read()",
    "r = ().__class__.__bases__[0].__subclasses__()",
    "r = eval('1')",
    "r = getattr(object, 'x')",
    "import cadquery as cq\ncq.exporters.export(None, '/tmp/x.step')",
    "import numpy as np\nnp.save('/tmp/x.npy', [1])",
    "import cadquery as cq\nr = cq.importers.importStep('/tmp/x.step')",
    "__import__('os')",
])
def test_unsafe_scripts_are_refused_before_running(code):
    sandbox = pytest.importorskip("src.cad_wrapper.cadcoder_wrapper.sandbox")
    with pytest.raises(sandbox.UnsafeCode):
        sandbox.check_source(code)


def test_the_sandbox_builds_a_part_and_labels_its_faces():
    pytest.importorskip("cadquery")
    from src.cad_wrapper.cad_common.mesh import finish_mesh
    from src.cad_wrapper.cadcoder_wrapper.sandbox import run_sandboxed

    vertices, triangles, labels, surface_types = run_sandboxed(PLATE)
    mesh, labels, discarded = finish_mesh(vertices, triangles, labels)
    assert mesh.is_watertight and discarded == 0
    assert mesh.euler_number == 0  # one through-hole
    assert sorted(surface_types) == ["cylinder"] + ["plane"] * 6
    bore = labels == surface_types.index("cylinder")
    radii = np.linalg.norm(mesh.triangles_center[bore][:, :2], axis=1)
    assert np.allclose(radii, radii.mean(), rtol=0.05)


@pytest.mark.parametrize("code, message", [
    ("import cadquery as cq\nr = cq.Workplane('XY').box(1, 1, 1)\nwhile True:\n    pass", "ran past"),
    ("import math\nr = math.sqrt(-1)", "ValueError"),
    ("import cadquery as cq\nx = 1", "did not assign"),
])
def test_the_sandbox_reports_hangs_and_failures(code, message):
    pytest.importorskip("cadquery")
    from src.cad_wrapper.cadcoder_wrapper import sandbox

    with pytest.raises(RuntimeError, match=message):
        sandbox.run_sandboxed(code, timeout=15)


def test_the_sandbox_has_no_network_or_credentials(monkeypatch):
    """Even code that got past the checks sees no network and no secrets."""
    pytest.importorskip("cadquery")
    import shutil
    import subprocess
    import sys

    from src.cad_wrapper.cadcoder_wrapper import sandbox

    if not shutil.which("unshare") or subprocess.run(
            ["unshare", "-rn", "true"], capture_output=True).returncode:
        pytest.skip("unprivileged network namespaces are unavailable here")
    monkeypatch.setenv("HF_TOKEN", "secret")
    probe = ("import os, socket\n"
             "print(os.environ.get('HF_TOKEN'))\n"
             "try:\n    socket.create_connection(('1.1.1.1', 80), timeout=2); print('online')\n"
             "except OSError as exc:\n    print('offline', exc.errno)\n")
    result = subprocess.run(
        ["unshare", "-rn", sys.executable, "-I", "-c", probe],
        env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=30,
    )
    assert result.stdout.split("\n")[0] == "None"
    assert result.stdout.split("\n")[1].startswith("offline")
