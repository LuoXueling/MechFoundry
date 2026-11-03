"""Stage 3 (pvbatch + EGL, GPU): the gallery scene, in five passes, one camera.

    pvbatch render_gl.py WORK_DIR [SCENE_JSON]      # default am_scene.json (the amphitheater)

The camera is the layout's, set verbatim (no ResetCamera): the amphitheater's perspective
eye on the stage (vertical view angle from the fitted horizontal FOV), or the earlier
orthographic gallery's parallel camera.  Passes (all PAGE size, 2x of 600 dpi at 180 mm):

  scene.png    the deformed bodies (already magnified by their own x N in gl_layout's
               world file) coloured by u_rel = |u| / own |u|max through
               ar_style.solve_rgb_points(1.0): the agentic-results last column's ramp
               per object; its light kit (SOLVE_VIEW) and finish (SOLVE_FINISH), normals
               at the 50 deg feature angle; transparent background
  ground.png   the key-light planar shadows alone (grey 0.88, unlit, hard; review Q4),
               transparent: the compose lays them UNDER the bodies (render_ar's rule)
  id_def.png   the deformed bodies unlit in their id colour (id, 255, 0), opaque white
  id_und.png   the undeformed bodies likewise: the outline and the gates
  urel.png     the deformed bodies unlit, u_rel as grey (0 black .. 1 white): the
               |u| > 0.7 balance gate
"""

import json
import os
import sys

from paraview.simple import *  # noqa: F403

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
import ar_style as A  # noqa: E402

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405
WORK = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_work"))
SCENE_FILE = sys.argv[2] if len(sys.argv) > 2 else "am_scene.json"
SCENE = json.load(open(os.path.join(WORK, SCENE_FILE)))
PX_W = int(os.environ.get("GL_PX_W", "8504"))
PX_H = int(round(PX_W * SCENE["camera"]["page_mm"][1] / SCENE["camera"]["page_mm"][0]))
SIZE = [PX_W, PX_H]
SHADOW_GREY = [0.88, 0.88, 0.88]


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


def new_view(kit):
    rv = CreateRenderView()  # noqa: F405
    rv.ViewSize = SIZE
    rv.OrientationAxesVisibility = 0
    for attr, val in kit:
        _try(lambda a=attr, v=val: setattr(rv, a, v))
    cam = SCENE["camera"]
    rv.CameraParallelProjection = 1 if cam.get("parallel", True) else 0
    rv.CameraFocalPoint = cam["focal"]
    rv.CameraPosition = cam["position"]
    rv.CameraViewUp = cam["view_up"]
    rv.CameraParallelScale = cam["parallel_scale"]
    if not cam.get("parallel", True):
        rv.CameraViewAngle = cam["view_angle_deg"]      # vertical, as vtkCamera takes it
    rv.Background = [1.0, 1.0, 1.0]
    _try(lambda: setattr(rv, "UseColorPaletteForBackground", 0))
    return rv


FLAT = tuple((a, v) for a, v in A.SURF_VIEW if a not in ("UseFXAA", "UseToneMapping", "UseSSAO"))


def flat_view():
    rv = new_view(FLAT)
    _try(lambda: setattr(rv, "UseFXAA", 0))
    _try(lambda: setattr(rv, "UseToneMapping", 0))
    return rv


def save(rv, name, transparent):
    Render(rv)  # noqa: F405
    SaveScreenshot(os.path.join(WORK, name), rv, ImageResolution=SIZE,  # noqa: F405
                   TransparentBackground=1 if transparent else 0)
    print("wrote", name)
    Delete(rv)  # noqa: F405


def scene_pass():
    rv = new_view(A.SOLVE_VIEW)
    src = LegacyVTKReader(FileNames=[os.path.join(WORK, "scene_def.vtk")])  # noqa: F405
    nrm = GenerateSurfaceNormals(Input=src)  # noqa: F405
    _try(lambda: setattr(nrm, "FeatureAngle", A.FEATURE_ANGLE))
    d = Show(nrm, rv)  # noqa: F405
    d.Representation = "Surface"
    ColorBy(d, ("POINTS", "u_rel"))  # noqa: F405
    lut = GetColorTransferFunction("u_rel")  # noqa: F405
    lut.ColorSpace = "Lab"
    lut.RGBPoints = A.solve_rgb_points(1.0)
    d.SetScalarBarVisibility(rv, False)
    for k, v in A.SOLVE_FINISH.items():
        setattr(d, k, v)
    save(rv, "scene.png", True)


def ground_pass():
    rv = new_view(A.SURF_VIEW)
    d = Show(LegacyVTKReader(FileNames=[os.path.join(WORK, "scene_shadow.vtk")]), rv)  # noqa: F405
    d.Representation = "Surface"
    d.DiffuseColor = SHADOW_GREY
    d.AmbientColor = SHADOW_GREY
    d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    save(rv, "ground.png", True)


def id_pass(vtk, name):
    rv = flat_view()
    d = Show(LegacyVTKReader(FileNames=[os.path.join(WORK, vtk)]), rv)  # noqa: F405
    d.Representation = "Surface"
    d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    ColorBy(d, ("CELLS", "idrgb"))  # noqa: F405
    d.MapScalars = 0
    d.SetScalarBarVisibility(rv, False)
    save(rv, name, False)


def urel_pass():
    rv = flat_view()
    d = Show(LegacyVTKReader(FileNames=[os.path.join(WORK, "scene_def.vtk")]), rv)  # noqa: F405
    d.Representation = "Surface"
    d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    ColorBy(d, ("POINTS", "u_rel"))  # noqa: F405
    lut = GetColorTransferFunction("u_rel")  # noqa: F405
    lut.ColorSpace = "RGB"
    lut.RGBPoints = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    d.SetScalarBarVisibility(rv, False)
    _try(lambda: setattr(d, "InterpolateScalarsBeforeMapping", 1))
    save(rv, "urel.png", False)


if __name__ == "__main__":
    _probe = CreateRenderView()  # noqa: F405
    Render(_probe)  # noqa: F405
    renderer = GetOpenGLInformation().GetRenderer()  # noqa: F405
    Delete(_probe)  # noqa: F405
    print("RENDERER:", renderer)
    assert "llvmpipe" not in renderer.lower(), "software rendering: set the EGL env (AGENTS.md §5)"
    json.dump({"renderer": renderer, "size": SIZE}, open(os.path.join(WORK, "render_info.json"), "w"))
    scene_pass()
    if os.path.exists(os.path.join(WORK, "scene_shadow.vtk")):     # the row gallery only:
        ground_pass()                                               # the amphitheater has none
    elif os.path.exists(os.path.join(WORK, "ground.png")):
        os.remove(os.path.join(WORK, "ground.png"))
    id_pass("scene_def.vtk", "id_def.png")
    id_pass("scene_und.vtk", "id_und.png")
    urel_pass()
