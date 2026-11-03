"""ParaView renderer for the four automatic bunny BC workflow panels."""

from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

BODY = [0.92, 0.90, 0.86]
ORANGE = [0.867, 0.518, 0.322]
GREEN = [0.333, 0.659, 0.408]
VIEW_DIR = [1.5, -3.0, 1.25]
SIZE = [1800, 1800]
stats = {}
for line in open("auto_stats.txt"):
    key, *values = line.split()
    stats[key] = values
SELECTED_CENTER = [float(v) for v in stats["selected_center"]]
SELECTED_RADIUS = float(stats["selected_radius"][0])


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


def actor(src, rv, color=BODY, opacity=1.0):
    normals = GenerateSurfaceNormals(Input=src)  # noqa: F405
    try:
        normals.FeatureAngle = 50
    except Exception:
        pass
    disp = Show(normals, rv)  # noqa: F405
    disp.Representation = "Surface"
    disp.DiffuseColor = color
    disp.Opacity = opacity
    disp.Ambient = 0.12
    disp.Diffuse = 1.0
    disp.Specular = 0.35
    disp.SpecularPower = 30
    return disp


def scalar_actor(src, rv, array_name, preset, association="CELLS"):
    """Scalar-coloured surface with bcregion's identical smooth material."""
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
    ColorBy(disp, (association, array_name))  # noqa: F405
    lut = GetColorTransferFunction(array_name)  # noqa: F405
    lut.ApplyPreset(preset, True)
    disp.SetScalarBarVisibility(rv, False)
    return disp, lut


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
    SaveScreenshot(f"auto_panel_{tag}.png", rv, ImageResolution=SIZE,
                   OverrideColorPalette="WhiteBackground")  # noqa: F405
    Delete(rv)  # noqa: F405
    print(f"wrote auto_panel_{tag}.png")


# (a) Explicit seed.
rv = view()
actor(STLReader(FileNames=["auto_seed_rest.stl"]), rv)  # noqa: F405
actor(STLReader(FileNames=["auto_seed.stl"]), rv, ORANGE)  # noqa: F405
save("a", rv)

# (b) Shape diameter computed by Warp rays against the exact displayed surface.
rv = view()
surface = LegacyVTKReader(FileNames=["auto_surface.vtk"])  # noqa: F405
# Shape diameter is computed per facet, as in CGAL.  Interpolate those exact
# values to shared vertices for a visually continuous scalar surface; this is
# render-only and does not alter clustering or segment selection.
continuous_sdf = CellDatatoPointData(Input=surface)  # noqa: F405
d, lut = scalar_actor(continuous_sdf, rv, "shape_diameter",
                      "Viridis (matplotlib)", association="POINTS")
save("b", rv)

# (c) All connected automatic segments.
rv = view()
surface = LegacyVTKReader(FileNames=["auto_surface.vtk"])  # noqa: F405
d, lut = scalar_actor(surface, rv, "segment_color", "Viridis (matplotlib)")
# Restrained qualitative palette inspired by Nature's preference for muted,
# colorblind-safe categorical colours; avoid rainbow/heat-map semantics.
lut.ColorSpace = "Step"
lut.Discretize = 1
lut.NumberOfTableValues = 10
lut.RGBPoints = [
    0.000, 0.180, 0.310, 0.420,
    0.111, 0.310, 0.510, 0.640,
    0.222, 0.340, 0.620, 0.560,
    0.333, 0.510, 0.650, 0.390,
    0.444, 0.760, 0.690, 0.390,
    0.556, 0.860, 0.660, 0.330,
    0.667, 0.800, 0.470, 0.280,
    0.778, 0.650, 0.350, 0.390,
    0.889, 0.500, 0.390, 0.560,
    1.000, 0.390, 0.410, 0.440,
]
lut.RescaleTransferFunction(0.0, 1.0)
save("c", rv)

# (d) Close-up styled like bc_region_id panel (c).
rv = view()
context = LegacyVTKReader(FileNames=["auto_selected_context.vtk"])  # noqa: F405
dc = actor(context, rv, [0.940, 0.928, 0.906])
dc.Ambient, dc.Diffuse, dc.Specular = 0.65, 0.45, 0.0
ColorBy(dc, ("CELLS", "fade"))  # noqa: F405
fade_lut = GetColorTransferFunction("fade")  # noqa: F405
fade_lut.ColorSpace = "RGB"
fade_lut.RGBPoints = [0.0, 0.940, 0.928, 0.906,
                      0.5, 0.940, 0.928, 0.906,
                      0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
fade_pwf = GetOpacityTransferFunction("fade")  # noqa: F405
fade_pwf.Points = [0.0, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0,
                   0.85, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
fade_lut.EnableOpacityMapping = 1
dc.SetScalarBarVisibility(rv, False)
actor(STLReader(FileNames=["auto_selected.stl"]), rv, ORANGE)  # noqa: F405
Render(rv)  # noqa: F405
norm = sum(q * q for q in VIEW_DIR) ** 0.5
unit = [q / norm for q in VIEW_DIR]
rv.CameraFocalPoint = SELECTED_CENTER
rv.CameraPosition = [SELECTED_CENTER[i] + unit[i] * SELECTED_RADIUS * 4.7
                     for i in range(3)]
rv.CameraViewUp = [0, 0, 1]
Render(rv)  # noqa: F405
SaveScreenshot("auto_panel_d.png", rv, ImageResolution=SIZE,
               OverrideColorPalette="WhiteBackground")  # noqa: F405
Delete(rv)  # noqa: F405
print("wrote auto_panel_d.png")

print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
