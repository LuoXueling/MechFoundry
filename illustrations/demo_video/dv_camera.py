"""The demo video's camera in numpy: ParaView's perspective projection, written out.

Shared by dv_prep (the orbit fits) and dv_compose (the octree section it draws over
the pvbatch render), so a point lands on the same pixel in both.  render_dv sets the
same camera on ParaView: focal point c, position c + D * view_dir, view-up +y,
vertical view angle ar_style.VIEW_ANGLE.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agentic_results"))
import ar_style as A  # noqa: E402


def view_dir(heading_deg, elev_deg):
    """Unit object -> camera direction; heading = atan2(x, z), y up."""
    a, e = math.radians(heading_deg), math.radians(elev_deg)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def basis(heading, elev):
    """(forward, right, up) of the camera."""
    f = -view_dir(heading, elev)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    return f, r, np.cross(r, f)


def project(P, c, heading, elev, D, w_px, h_px, fov=A.VIEW_ANGLE):
    """ParaView's perspective projection of points P (n x 3) -> pixel (x, y), y down."""
    f, r, u = basis(heading, elev)
    q = P - (np.asarray(c, float) - D * f)
    z = q @ f
    t = math.tan(math.radians(fov) / 2)
    x = (q @ r) / (z * t) * (h_px / 2)
    y = (q @ u) / (z * t) * (h_px / 2)
    return w_px / 2 + x, h_px / 2 - y
