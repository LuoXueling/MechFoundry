"""The demo video's timeline: every time-dependent quantity as a function of t (s).

Plain Python (math only), imported by the pvbatch renderer and by the PIL compose,
so the two cannot disagree on which pass a frame needs or where the camera is.

The director's 20.0 s table, as revised in round 2 (review2_*.md) and by the user's
later changes (the full octree held ~2.5 s, BCs seen from the front, the bench one
ordinary tile in the finale; then a 3 s longer finale, spin 20 % faster, the bench
framed lower; then the optimisation beat after one |u| cycle), now 28.8 s:

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
               oscillates 1 -> k -> 1 with T = 2.0 s from t0 = 14.0 (peaks 15.0, 17.0,
               ...), and the bench spins on at 11.4 deg/s, never slowed, to the end
  7b 16.0-22.8 the optimisation beat (the user, 2026-11-07): at 16.0 (s back to 1) one
               eased move 16.0-16.7 takes the camera into a close-up (2.05x) of the
               bench end whose change is largest (-x), in the left half, while the spin
               eases to 3.5 deg/s, |u| fades to the surface-change colouring (16.0-16.6;
               ghost and "x1" out 16.0-16.2) and the title dips to "Optionally, it can
               be optimized / for load-carrying ability"; the plot (Iteration vs strain
               energy vs. initial) fades in on the right 16.3-16.8 with its live "-NN %"
               and bracket; the 41 iterates play 16.8-21.8 (8 a second) in the slowly
               turning close-up, the curve drawing in step, then a 1.0 s hold on it_40
               (21.8-22.8, frames 654-683), still turning, arrows and plot up; change arrows (dense where
               the change is large, each fading in over 0.3 s once its site passes
               0.5 h) 16.7-16.9, full through the hold, out linearly 22.8-23.3 with
               the shrink's iterate body (1/15 alpha a frame); plot out 22.75-23.05
  8  22.8-28.8 the OPTIMISED bench shrinks into the top-left cell (22.8-23.35) while
               the camera pulls back out of the close-up and the spin eases back to
               11.4 deg/s; its colour back to |u| (22.8-23.3, its own solve with the
               session's per-part materials) and s(t) restarting from 1 with the same
               k; its title (the optimisation one) morphs into its prompt label; grid
               fades with the move; the 20 tiles pop from 23.4 as a diagonal wave
               (50 ms a step, 0.3 s each; last lands 24.05); two-line title
               23.1-23.5; hold to 28.8 (poster: frame 747, 24.9 s) (the user: "the spinning objects can stay for
               a few more seconds")

Title changes dip rather than dissolve: the old title out over 0.18 s, then the new
one in over 0.22 s (two-line over one-line dissolves read as mush on a phone).
"""

import math

FPS = 30
# -- the optimisation beat (2026-11-07), after the first full |u| cycle; every scene-8
# time is the 23.0 s cut's + OPT_DT -------------------------------------------------------
OPT_T = 16.0              # s back to 1 after one cycle: the beat starts
MOVE_IN = (16.0, 16.7)    # camera into the close-up of one end (one eased motion)
CLOSE_SPAN_M = 0.85       # the close-up frames this much height (m) in the 780 px viewport
CLOSE_CENTRE = (470.0, 620.0)   # where the focused end's change sites centre (px)
BODY_CLEAR_X = (840.0, 960.0)   # the body fades to 0 across this x band while the plot is up
OPT_SPIN_DPS = 3.5        # the spin through the beat (the user: ~3-4 deg/s), eased in/out
N_ITERS = 40              # the clean-up's updates (it_0 .. it_40)
ITER_RATE = 8.0           # iterates a second (the user: 15/s, then 10/s = 4.0 s, then one
                          # second more: 8/s = 5.0 s)
ITERS = (16.8, 16.8 + N_ITERS / ITER_RATE)   # 16.8 - 21.8
END_PAUSE = 1.0           # the hold on it_40 before the shrink, the bench still turning, the
                          # arrows and the plot up (the user: 0.3 s, then 1.0 s to read it)
ARROWS_IN = (16.7, 16.9)
ARROW_FADE_S = 0.3          # an arrow fades in over 0.3 s once its site passes the threshold
PLOT_IN = (16.3, 16.8)
SHRINK0 = ITERS[1] + END_PAUSE   # 22.8: the move into the cell after the pause on it_40 (the
                          # user; the 1.2 s "-42 %" hold and its pop are gone, the label is live)
PLOT_OUT = (SHRINK0 - 0.05, SHRINK0 + 0.25)
ARROWS_OUT = (SHRINK0, SHRINK0 + 0.5)   # 22.8 - 23.3: full through the last iterate AND the
                          # whole end pause (the user: that is where the result is read; a fade
                          # from 20.55 left them faint there), then out with the iterate body they
                          # sit on (OPT_FADE: the iter look is gone at 23.3, and the arrows are
                          # rendered in its view).  LINEAR: 0.5 s at 1/15 alpha a frame, the
                          # per-frame limit (the user: arrows must not vanish suddenly); a
                          # smoothstep that short would peak at 0.1 a frame
OPT_DT = SHRINK0 - 17.0   # 5.8 s inserted: every scene-8 time is the 23.0 s cut's + OPT_DT
UNI_FADE = (16.0, 16.6)   # |u| -> the uniform zero-|u| colour (on it_0's shape)
OPT_FADE = (SHRINK0, SHRINK0 + 0.5)   # uniform -> the optimised bench's |u|
OPT_TITLE = "Optionally, it can be optimized\nfor load-carrying ability"

DURATION = 23.0 + OPT_DT
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

# looks, in order, and the crossfade from each to the next; "iter" = the clean-up's
# iterates in the uniform zero-|u| colour (two consecutive ones blended, vm_iterate),
# "opt" = the optimised bench's |u| (its own per-part solve), also its finale tile
SEQ = ("clay", "paint", "seg", "bc", "E", "solve", "octree", "solve", "iter", "opt")
LOOKS = tuple(dict.fromkeys(SEQ))
FADES = ((4.2, 4.8), (5.7, 6.3), (7.2, 7.8), (8.7, 9.3), (10.3, 10.8), (10.8, 11.1), (13.6, 13.9),
         UNI_FADE, OPT_FADE)
GHOSTS = (("clay", (13.9, 14.1), (OPT_T, OPT_T + 0.2)),          # (undeformed pass, in, out)
          ("optclay", (SHRINK0 + 0.1, SHRINK0 + 0.4), (1e9, 1e9 + 1)))
LABEL_IN = (13.9, 14.1)
LABEL_OUT = (OPT_T, OPT_T + 0.2)   # the deformation is off from here
BAR_IN, BAR_OUT = (8.9, 9.3), (10.3, 10.6)
TITLE_OUT, TITLE_IN = 0.18, 0.22   # the dip between two titles
TITLES = (            # (start of the change, text); scene 3 adds " with texture"
    (2.6, "Shape generation"), (4.2, "Shape generation with texture"),
    (5.7, "Surface segmentation"), (7.2, "Boundary conditions"), (8.7, "Young's modulus"),
    (10.3, "Diffuse Nitsche FEA in NVIDIA Warp\non non-conforming octree mesh"),
    # the user's wording; one line at 72 px is 1860 px, so two (1040 / 800 px); the last
    # title shrinks into the bench's cell and gives way to its prompt
    (OPT_T, OPT_TITLE))
# two lines, the user's wording; 6.5 min = the mean wall clock of the 50 Opus runs
# (6.46 min; dv_prep asserts it from the Opus category records' metrics.json values)
NEW_TITLE = (17.3 + OPT_DT, 17.7 + OPT_DT,
             "AI operator does all these fully autonomously\neach in 6.5 minutes on average")
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
MOVE = (SHRINK0, SHRINK0 + 0.55)     # bench into its cell
TILE_POP0, TILE_STAGGER, TILE_POP = 17.6 + OPT_DT, 0.05, 0.3
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
    turns counter-clockwise seen from above, so the camera's heading decreases; through
    the beat the rate eases from SPIN_DPS to OPT_SPIN_DPS and back (the user), its
    integral taken in closed form so the heading never jumps."""
    return (h0 - SPIN_DPS * max(0.0, t - SPIN_START)
            + (SPIN_DPS - OPT_SPIN_DPS) * _slowdown(t))


def osc(t, k, phase=0.0):
    """1 -> k -> 1 sinusoid, period T_OSC, from T0 (1 before it)."""
    if t < T0:
        return 1.0
    return 1.0 + (k - 1.0) * 0.5 * (1.0 - math.cos(2 * math.pi * ((t - T0) / T_OSC + phase)))


def closeup(t):
    """0 -> 1 progress of the move into the beat's close-up, back to 0 during the shrink
    (the camera pulls out while the compose shrinks the layer into the cell, so the
    tile ends as the whole bench under the orbit camera, the tiles' own rule)."""
    return ease_io(ramp(t, *MOVE_IN)) * (1.0 - ease_io(ramp(t, *MOVE)))


def _slowdown(t):
    """Integral over [-inf, t] of the spin's slow-down weight (0 -> 1 smoothstep over
    MOVE_IN, 1 to the shrink, 1 -> 0 over MOVE), in s: closed form, so the heading is
    continuous and smooth."""
    S = lambda x: x ** 3 - x ** 4 / 2          # integral of smooth over [0, x]  # noqa: E731
    l1, l2 = MOVE_IN[1] - MOVE_IN[0], MOVE[1] - MOVE[0]
    i = l1 * S(ramp(t, *MOVE_IN))
    i += max(0.0, min(t, MOVE[0]) - MOVE_IN[1])
    x = ramp(t, *MOVE)
    return i + l2 * (x - S(x))


SHRINK_OVERSIZE = 2.0     # the shrink's bench passes: canvas x2 each way, same pixel scale


def oversize(t):
    """The bench passes' canvas factor at t: SHRINK_OVERSIZE while the shrink scales the
    layer (0 < move < 1), else 1.  The close-up's bench runs up to 770 px off the frame's
    left and below its bottom; scaled down, a frame-sized layer showed its own border as
    a straight cut through the legs and the grid (f0636).  On a 2x canvas (same camera,
    same pixel scale, wider view angle) the whole bench is inside the layer."""
    m = move(t)
    return SHRINK_OVERSIZE if 0.0 < m < 1.0 else 1.0


def bench_osc(t, k):
    """The bench's s(t): 1 -> k -> 1 from T0, held at 1 through the beat (the uniform
    look shows no deformation), restarting from 1 at the shrink (the user)."""
    if t < OPT_T:
        return osc(t, k)
    if t < SHRINK0:
        return 1.0
    return osc(t - SHRINK0 + T0, k)


def plot_opacity(t):
    return smooth(ramp(t, *PLOT_IN)) * (1.0 - smooth(ramp(t, *PLOT_OUT)))


def arrows_opacity(t):
    return smooth(ramp(t, *ARROWS_IN)) * (1.0 - ramp(t, *ARROWS_OUT))   # out: linear (above)


def arrows_visible(t):
    return ARROWS_IN[0] <= t < ARROWS_OUT[1]


def _in_out(t, a, b):
    return smooth(ramp(t, *a)) * (1.0 - smooth(ramp(t, *b)))


def ghosts(t):
    """{undeformed pass: opacity} of the ghost outline(s) at t."""
    return {g: o for g, a, b in GHOSTS if (o := _in_out(t, a, b)) > 0}


def iter_pos(t):
    """The iterate index on screen, 0 -> N_ITERS linearly over ITERS (15 a second)."""
    return N_ITERS * ramp(t, *ITERS)


def vm_iterate(t):
    """(i0, i1, w): the two consecutive iterates on screen at t and i1's weight."""
    p = iter_pos(t)
    i0 = min(int(math.floor(p)), N_ITERS - 1)
    return i0, i0 + 1, p - i0


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
