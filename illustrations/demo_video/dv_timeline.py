"""The demo video's timeline: every time-dependent quantity as a function of t (s).

Plain Python (math only), imported by the pvbatch renderer and by the PIL compose,
so the two cannot disagree on which pass a frame needs or where the camera is.

The director's 20.0 s table, as revised in round 2 (review2_*.md) and by the user's
later changes (the full octree held ~2.5 s, BCs seen from the front, the bench one
ordinary tile in the finale; then a 3 s longer finale, spin 20 % faster, the bench
framed lower), now 23.0 s:

  1  0.0-2.2   CLI, opaque from frame 0 with "> " and a blinking cursor: typing
               0.15-1.75, "enter" at 1.75
  2  2.2-4.2   CLI up 300 px 2.2-2.5, faded by 2.4, under the bench, which is opaque by
               2.35; bench + grid rise 2.2-3.1; the bench spins from 2.2 at 11.4 deg/s
               from heading +25.55, chosen so the middle of the BC scene (7.95 s) is a
               front three-quarter at -40 deg (the backrest's backward load arrows seen
               along their length) and the octree hold is centred face-on to the cut
               (the bench's lateral mid-plane) at -90 deg
  3  4.2-5.7   clay -> paint 4.2-4.8; " with texture" joins the title
  4  5.7-7.2   paint -> segmentation 5.7-6.3
  5  7.2-8.7   segmentation -> supports and loads 7.2-7.8 (heading 0 at 7.95)
  6  8.7-10.3  BC -> Young's modulus 8.7-9.3; colour bar 8.9-9.3
  7  10.3-17.0 E -> |u| 10.3-10.8 (at s = 1); the solve's full octree (its far half,
               cut by the camera-facing plane through the body) over the |u| surface,
               in 10.8-11.1, held 2.5 s to 13.6 (headings -76 .. -104, the cut face
               seen face-on mid-hold), out 13.6-13.9; ghost and "x1" 13.9-14.1; s(t)
               oscillates 1 -> k -> 1 with T = 2.0 s from t0 = 14.0 (peaks 15.0, 17.0)
  8  17.0-23.0 the bench shrinks into the top-left cell, an ordinary tile; its title
               morphs into its prompt label; label leaves at the 17.0 peak;
               grid fades with the move 17.0-17.55; the 20 tiles pop from 17.6 as a
               diagonal wave (50 ms a step, 0.3 s each; last lands 18.25); two-line
               title 17.3-17.7; hold to 23.0 (the user: "the spinning objects can
               stay for a few more seconds")

Title changes dip rather than dissolve: the old title out over 0.18 s, then the new
one in over 0.22 s (two-line over one-line dissolves read as mush on a phone).
"""

import math

FPS = 30
DURATION = 23.0
NFRAMES = int(round(FPS * DURATION))
W, H = 1920, 1080

SPIN_DPS = 11.4          # the bench (UI viewer: 7.5 deg/s); CCW from above
TILE_SPIN_DPS = 17.1     # the scene-8 tiles, each within +-10 %
SPIN_START = 2.2
# heading0 (the user: "the backward facing arrows in the BC clearly visible"): the BC
# scene's middle (7.95 s) sits at -40 deg, a front three-quarter from the -x side.  The
# spin turns 51 deg more by the octree hold's middle (12.35 s), -90 deg: face-on to the
# lateral cut, whose far (+x) half is kept.  The +40 side would put the hold near 0 deg,
# with the cut seen edge-on.  The cost: the bench turns past -90 deg at 12.34 s, so the
# rest of scene 7 is a rear three-quarter view.
BC_MID_T, BC_MID_HEADING = 7.95, -40.0
BENCH_HEADING0 = BC_MID_HEADING + SPIN_DPS * (BC_MID_T - SPIN_START)   # +25.55
T_OSC = 2.0              # s(t) period, every object
T0 = 14.0                # the bench's oscillation starts (s = 1 here, rising)
VIEWPORT = (220, 1000)   # the body's viewport in scenes 2-7 (y, px); centre 610,
                         # 70 px below the frame centre (the user: "a little bit lower")
FIT_REST = 0.92          # the bench's at-rest extent over the clip's headings / viewport height

# looks, in order, and the crossfade from each to the next
SEQ = ("clay", "paint", "seg", "bc", "E", "solve", "octree", "solve")
LOOKS = tuple(dict.fromkeys(SEQ))
FADES = ((4.2, 4.8), (5.7, 6.3), (7.2, 7.8), (8.7, 9.3), (10.3, 10.8), (10.8, 11.1), (13.6, 13.9))
GHOST_IN = (13.9, 14.1)      # after the mesh is gone
LABEL_IN = (13.9, 14.1)
LABEL_OUT = (17.0, 17.3)
BAR_IN, BAR_OUT = (8.9, 9.3), (10.3, 10.6)
TITLE_OUT, TITLE_IN = 0.18, 0.22   # the dip between two titles
TITLES = (            # (start of the change, text); scene 3 adds " with texture"
    (2.6, "Shape generation"), (4.2, "Shape generation with texture"),
    (5.7, "Surface segmentation"), (7.2, "Boundary conditions"), (8.7, "Young's modulus"),
    (10.3, "Diffuse Nitsche FEA in NVIDIA Warp\non non-conforming octree mesh"))
# two lines, the user's wording; 6.5 min = the mean wall clock of the 50 Opus runs
# (6.46 min; dv_prep asserts it from the Opus category records' metrics.json values)
NEW_TITLE = (17.3, 17.7, "AI operator does all these fully autonomously\neach in 6.5 minutes on average")
MEAN_WALL_MIN = 6.5

# scene 8 grid: 7 x 3 uniform cells under a two-line title; the bench is cell (0, 0)
COLS, ROWS = 7, 3
GRID_TOP = 176
GRID_X0, GRID_X1 = 64, W - 64     # inset from the frame edges (safe margin)
CELL_W = (GRID_X1 - GRID_X0) / COLS
CELL_H = (H - GRID_TOP) / ROWS
NAME_BAND = 76           # the object's prompt above it (up to two lines), inside the cell
TILE_BODY_W, TILE_BODY_H = CELL_W, CELL_H - NAME_BAND - 4      # the body's area in a cell
TILE_FILL = 0.88         # max extent over all azimuths at peak k / the body area's short side
MOVE = (17.0, 17.55)     # bench into its cell
TILE_POP0, TILE_STAGGER, TILE_POP = 17.6, 0.05, 0.3
SPIN_JITTER = 0.10       # per-tile spin rate within +-10 % of TILE_SPIN_DPS


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ramp(t, a, b):
    return clamp((t - a) / (b - a))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_o(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_ob(x, c=1.70158):
    x = clamp(x)
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def look_weights(t):
    """{look: weight} with weight > 0; crossfades are smoothsteps (blended in sRGB)."""
    for i, (a, b) in enumerate(FADES):
        if t < a:
            return {SEQ[i]: 1.0}
        if t < b:
            w = smooth((t - a) / (b - a))
            return {SEQ[i]: 1.0 - w, SEQ[i + 1]: w} if w > 0 else {SEQ[i]: 1.0}
    return {SEQ[-1]: 1.0}


def bench_visible(t):
    return t >= SPIN_START


def bench_heading(t, h0):
    """Camera heading (deg, atan2(x, z) of the object -> camera direction).  The object
    turns counter-clockwise seen from above, so the camera's heading decreases."""
    return h0 - SPIN_DPS * max(0.0, t - SPIN_START)


def osc(t, k, phase=0.0):
    """1 -> k -> 1 sinusoid, period T_OSC, from T0 (1 before it)."""
    if t < T0:
        return 1.0
    return 1.0 + (k - 1.0) * 0.5 * (1.0 - math.cos(2 * math.pi * ((t - T0) / T_OSC + phase)))


def ghost_opacity(t):
    return smooth(ramp(t, *GHOST_IN))


def grid_opacity(t):
    if t < SPIN_START:
        return 0.0
    return 1.0 - smooth(ramp(t, *MOVE))


def rise(t):
    """(y offset px, opacity) of the bench layer as it rises from the bottom."""
    return 700.0 * (1.0 - ease_o(ramp(t, 2.2, 3.1))), smooth(ramp(t, 2.2, 2.35))   # opaque before it reaches the leaving CLI


def move(t):
    """0 -> 1 progress of the move into the top-left block."""
    return ease_io(ramp(t, *MOVE))


# -- scene 8 tiles ---------------------------------------------------------------------
def cells():
    """The 20 cells (col, row) after the bench's own (0, 0), in reading order."""
    return [(c, r) for r in range(ROWS) for c in range(COLS) if (c, r) != (0, 0)]


def cell_box(c, r):
    x0, y0 = GRID_X0 + c * CELL_W, GRID_TOP + r * CELL_H
    return x0, y0, x0 + CELL_W, y0 + CELL_H


def body_box(c, r):
    """The part of a cell under its name band, where the body is fitted."""
    x0, y0, x1, y1 = cell_box(c, r)
    return x0, y0 + NAME_BAND, x1, y1 - 4


def pop_rank():
    """cell index -> wave step: a diagonal wave out from the bench's cell, the cells
    touching it first (step = column + row - 1)."""
    return {i: c + r - 1 for i, (c, r) in enumerate(cells())}


def tile_pop_time(i):
    return TILE_POP0 + TILE_STAGGER * pop_rank()[i]


def tile_phase(i):
    """Golden-ratio phase offsets, so no two tiles breathe in step."""
    return (0.618 * (i + 1)) % 1.0


def tile_rate(i):
    """SPIN_DPS within +-SPIN_JITTER, a fixed low-discrepancy pick per tile."""
    j = ((0.7548776662 * (i + 1)) % 1.0) * 2.0 - 1.0
    return TILE_SPIN_DPS * (1.0 + SPIN_JITTER * j)


def tile_heading(t, i, h0):
    return h0 - tile_rate(i) * (t - TILE_POP0)


def tile_visible(t, i):
    return t >= tile_pop_time(i)


def frame_times(step=1):
    return [(f, f / FPS) for f in range(0, NFRAMES, step)]
