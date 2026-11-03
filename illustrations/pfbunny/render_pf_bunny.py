"""ParaView renderer for the two PartField bunny panels.

View direction, view angle, material and lighting are copied verbatim from
``../bcautobunny/render_auto_bunny.py`` so these panels can be placed beside
its shape-diameter and segment panels without the bunny moving or changing
finish.  Both panels here share one camera, so the compose step can crop them
to a common ink box.
"""

from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

VIEW_DIR = [1.5, -3.0, 1.25]
SIZE = [1800, 1800]

def view():
    rv = CreateRenderView()  # noqa: F405
    rv.ViewSize = SIZE
    rv.Background = [1.0, 1.0, 1.0]
    rv.OrientationAxesVisibility = 0
    rv.CameraViewAngle = 25
    for name, value in (("UseColorPaletteForBackground", 0),
                        ("UseFXAA", 1), ("UseSSAO", 1),
                        ("UseToneMapping", 0),
                        ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
                        ("KeyLightIntensity", 0.85), ("KeyLightWarmth", 0.55),
                        ("FillLightKFRatio", 2.6), ("BackLightKBRatio", 3.0),
                        ("HeadLightKHRatio", 3.0)):
        try:
            setattr(rv, name, value)
        except Exception:
            pass
    return rv


def surface_actor(rv):
    """bcregion's identical smooth material, on the shared PartField surface."""
    src = LegacyVTKReader(FileNames=["pf_surface.vtk"])  # noqa: F405
    normals = GenerateSurfaceNormals(Input=src)  # noqa: F405
    try:
        normals.FeatureAngle = 50
    except Exception:
        pass
    disp = Show(normals, rv)  # noqa: F405
    disp.Representation = "Surface"
    disp.Ambient = 0.12
    disp.Diffuse = 1.0
    disp.Specular = 0.35
    disp.SpecularPower = 30
    return normals, disp


def fit(rv):
    Render(rv)  # noqa: F405
    rv.CameraFocalPoint = [0, 0, 0]
    rv.CameraPosition = [3 * q for q in VIEW_DIR]
    rv.CameraViewUp = [0, 0, 1]
    rv.CameraParallelProjection = 0
    ResetCamera(rv)  # noqa: F405
    fp = list(rv.CameraFocalPoint)
    pos = list(rv.CameraPosition)
    dist = sum((pos[i] - fp[i]) ** 2 for i in range(3)) ** 0.5
    unit = [q / sum(c * c for c in VIEW_DIR) ** 0.5 for q in VIEW_DIR]
    rv.CameraPosition = [fp[i] + unit[i] * dist * 1.02 for i in range(3)]
    Render(rv)  # noqa: F405


def save(tag, rv):
    fit(rv)
    SaveScreenshot(f"pf_panel_{tag}.png", rv, ImageResolution=SIZE,
                   OverrideColorPalette="WhiteBackground")  # noqa: F405
    Delete(rv)  # noqa: F405
    print(f"wrote pf_panel_{tag}.png")


# (1) The learned feature field, projected onto its 3 leading principal
# components.  Left per-FACET: render_auto_bunny.py runs its shape diameter
# through CellDatatoPointData for a visually continuous surface, but that filter
# will not take the unsigned-char array direct colouring needs (the chained
# normals filter then has no representation at all).  A PartField feature is a
# per-facet quantity anyway, and at print size the facets are not resolvable.  The colours ARE the data -- MapScalars=0 paints the unsigned-char
# array the legacy COLOR_SCALARS block carries, with no transfer function in
# between, so no LUT choice can bias what the projection shows.
rv = view()
normals, disp = surface_actor(rv)
ColorBy(disp, ("CELLS", "feature_rgb"))  # noqa: F405
disp.MapScalars = 0
disp.SetScalarBarVisibility(rv, False)
save("feature", rv)

# (2) The PartField parts, in the tints Figure 1's segmentation panel gives
# them.  Coloured the same way, and for the same reason: pf_prep.py owns the
# palette (and the gate that cleared it), so no transfer function here can
# quietly recolour the parts.
rv = view()
normals, disp = surface_actor(rv)
ColorBy(disp, ("CELLS", "segment_rgb"))  # noqa: F405
disp.MapScalars = 0
disp.SetScalarBarVisibility(rv, False)
save("segments", rv)

print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
