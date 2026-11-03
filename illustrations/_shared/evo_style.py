"""shape_opt_merged b's result frames (round 11, 2026-10-16): ONE place for the look both
runs share, imported by shapeopt/render_so.py (`m_evo` slots) and hiddenphi/render_hp.py
(`m_evo_*` slots).  The standalone loop figures keep their own `evo_*` / `l_evo_*` frames
(GHOST_OPACITY / L_GHOST = 0.34, voids in the body colour) untouched.

- The near half (the translucent ghost) is FAINTER: M_GHOST_OPACITY 0.34 -> 0.12, one
  value for both runs (0.20 was tried first and read as barely changed at print size).
  0.20 had been rejected for the old strip because frame 0's cylinder blank nearly
  vanished; here the user asked for the front half to recede, and the far half's opaque
  body + section face carry each frame's outline, so the table still reads.
- The voids are DARKER: every vertex carries `open` (../_shared/void_open.py: the fraction
  of cosine-weighted hemisphere rays that escape the uncut design surface), and the
  surface is coloured from the body colour at open >= VOID_OPEN[1] down to VOID_RGB at
  open <= VOID_OPEN[0], linear in between.  A rule, not a per-frame pick: an enclosed
  cavity scores 0, the hidden-phi run's tunnels a few hundredths, the open outside well
  above VOID_OPEN[1] (measured 2026-10-16, shell minimum: shape 0 / 40 = 0.47 / 0.32,
  hidden 20 = 0.44).  The shape run has no void; at 100 only 21 vertices (0.2 %), a
  pinched crevice under the top's rim at z ~ 0.33, fall below VOID_OPEN[0] -- sub-pixel
  in print.  Below 0.22: 10 % of the hidden run's shell at 43, 22 % at 90.
  The ghost uses the same ramp from its own colour, so a void cut through by the
  section shows its near wall too, faintly.
"""

M_GHOST_OPACITY = 0.12
VOID_OPEN = (0.08, 0.22)          # open <= lo: full void colour; >= hi: the body colour
VOID_RGB = [0.40, 0.385, 0.365]   # a dark warm grey, the body's own hue (lifted by gamma
#                                   0.72 at compose to ~0.52, still half the body's value)


def open_actor(rv, path, rgb, opacity=1.0, clip_y=None, name="open"):
    """A surface coloured by its `open` array: `rgb` where open, VOID_RGB where enclosed.
    The material is render_so._evo_body / render_hp.body_actor's plain one (pbr=False).
    clip_y keeps y < clip_y (render_so's open_cut for its capped ghosts)."""
    from paraview.simple import (Clip, ExtractSurface, GenerateSurfaceNormals,  # noqa: F401
                                 GetColorTransferFunction, LegacyVTKReader, Show)
    src = LegacyVTKReader(FileNames=[path])
    if clip_y is not None:
        c = Clip(Input=src)
        c.ClipType = "Plane"
        c.ClipType.Origin = [0.0, clip_y, 0.0]
        c.ClipType.Normal = [0.0, 1.0, 0.0]
        c.Invert = 1
        src = ExtractSurface(Input=c)
    n = GenerateSurfaceNormals(Input=src)
    n.FeatureAngle = 50
    d = Show(n, rv)
    d.Representation = "Surface"
    d.ColorArrayName = ["POINTS", "open"]
    lut = GetColorTransferFunction(name)          # one LUT per base colour
    lut.ColorSpace = "RGB"
    lut.RGBPoints = [0.0, *VOID_RGB, VOID_OPEN[0], *VOID_RGB,
                     VOID_OPEN[1], *rgb, 1.0, *rgb]
    try:
        lut.AutomaticRescaleRangeMode = "Never"
    except Exception:  # noqa: BLE001
        pass
    d.LookupTable = lut
    d.MapScalars = 1
    d.Opacity = opacity
    d.Ambient, d.Diffuse, d.Specular, d.SpecularPower = 0.14, 1.0, 0.30, 30
    d.SetScalarBarVisibility(rv, False)
    return d
