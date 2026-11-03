"""Stage 1 of method_overview: the generative assets, made once and cached.

    python ov_assets.py [step ...]        # run in overview/, env mechfoundry-warp117
    OV_ASSETS_FORCE=1 python ov_assets.py paint   # redo one step

Every picture on the overview except these is recomputed deterministically on
each build.  These are not: TripoSG, SDXL, Hunyuan3D, PartField and the vision
model are generative, their GPU kernels are not bit-reproducible, and three of
them live in conda envs the figure build otherwise never touches.  So they run
here, once, with their seeds and prompts recorded, and ``./build.sh overview``
only READS ``assets/``.  ``./build.sh ovassets`` is the (opt-in, never in
``all``) way back in.  A step whose outputs exist is skipped; ``OV_ASSETS_FORCE=1``
reruns it, moving the old outputs aside to /tmp first.

Steps, in dependency order (all GPU work on ``OV_GPU``, default 1 — one job at a
time, AGENTS.md §2):

  input      the clay render of the display bunny from the paper's camera, and
             a scribble traced from its silhouette — the two TripoSG input cards
             of the first front end (unused since the figure starts from text
             alone, 2026-09-29; kept, cached)
  paint      Hunyuan-Paint on the display bunny: SDXL draws a depth-conditioned
             reference from PAINT_PROMPT, the paint stage follows it.  The
             reference is kept — it is ALSO the figure's SDXL drawing (standing in
             for draw_image's free one, whose pose would not match the bunny)
  segments   PartField (k = 5..10, one feature cache — ov_bunny_loop picks the
             k that cuts both ears off the rest), on the display bunny
  cad        the downloaded STEP (assets/cad/*.step|stp), tessellated with its
             B-rep face ids through the workflow's CAD tessellator
  cadmodel   the CAD lane's part, a real B-rep: ov_cad_fit.py measures the
             display bunny (its radius on latitude rings, the ears' slices) and
             writes a CAD-Coder-style CadQuery script from them (assets/cad/
             bunny_cadcoder.py); ov_cad_build.py builds it with the CAD-Coder
             sandbox's own check / execute / tessellate into cad_bunny.npz
  cadimage   (superseded by cadmodel, kept) a Gemini-generated image of a bunny drawn
             as a B-rep surface (patches and their edge curves), supplied by the
             user (2026-09-29) in place of the downloaded STEP, which GrabCAD
             gates.  Cut out along its own outer outline, so the soft floor
             shadow under it goes (it lies outside the outline)
  cutout     the three input cards on a transparent background, so the
             generation tint shows through them: the clay render and the SDXL
             image matted by RMBG-1.4 — TripoSG's own background remover, run as
             its pipeline runs it — and colour-decontaminated against their
             backgrounds; the scribble's ink keyed off its white paper

ONE MESH FOR EVERY SOURCE (the user's decision, 2026-09-29): the figure shows
the TripoSG, Hunyuan3D and painting outputs as the same display bunny — grey,
painted, painted — so the lanes differ only in what goes in.  No TripoSG or
Hunyuan3D shape is generated here; the CAD lane keeps its own downloaded part.

THE DISPLAY BUNNY.  Every asset is made on ``assets/bunny_display.ply``: bunny.stl
mirrored in y exactly as bcregion / bcautobunny / pfbunny mirror it (the paper's
camera then sees its face), same vertices order, same face order.  The
reflection also turns bunny.stl's inward winding outward (see ``bunny()``).

Every file written lands in ``assets/MANIFEST.json`` with its sha256 and the
command, env, seed and prompt that made it; ``ov_surfaces.py`` refuses to read
an asset whose hash no longer matches.
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ASSETS = HERE / "assets"
MANIFEST = ASSETS / "MANIFEST.json"
GPU = os.environ.get("OV_GPU", "1")
FORCE = os.environ.get("OV_ASSETS_FORCE") == "1"
ENVS = Path(os.environ.get("CONDA_ENVS", "/home/xl2254/anaconda3/envs"))
WARP_PY = ENVS / "mechfoundry-warp117" / "bin" / "python"
HUNYUAN_PY = ENVS / "mechfoundry-hunyuan3d" / "bin" / "python"
WORKFLOW_TOOL = REPO / "src/agents_wrapper/fem_workflow/scripts/workflow_tool.py"
GENERATE_SURFACE = REPO / "src/agents_wrapper/fem_workflow/scripts/generate_surface.py"
HUNYUAN = REPO / "src/hunyuan3d_wrapper/generate.py"
PV_ENV = {"__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
          "VTK_DEFAULT_EGL_DEVICE_INDEX": GPU}

SEED = 42
TEXT_PROMPT = "a sitting rabbit"
# What SDXL is asked to draw over the bunny's depth.  Several visibly different
# materials on purpose: the per-part stage reads a material off each part's
# colour and texture, and a one-material figurine would give it nothing to do.
# What SDXL is asked to draw over the bunny's depth, and the seed that drew it.
# Chosen by the user's brief (2026-09-29): a real-looking rabbit, not a toy or a
# figurine, with no dark patch round the eyes — picked from a screen of 4
# prompts x 3 seeds for an even coat the paint stage can carry cleanly (a
# patchy brown-and-white reference painted into blotches).
PAINT_PROMPT = os.environ.get(
    "OV_PAINT_PROMPT", "a realistic brown wild rabbit with soft natural fur, photograph")
PAINT_SEED = int(os.environ.get("OV_PAINT_SEED", 1))
# The paint stage at its better settings (the workflow's defaults are 6 views at
# 512 px into a 1024 texture): at the defaults the fur came out smeared.
PAINT_QUALITY = ["--views", "9", "--paint-resolution", "768", "--texture-size", "2048"]
PART_CLUSTERS = (5, 6, 7, 8, 9, 10)   # ov_surfaces picks the k that separates both ears

# The display bunny is z-up; Hunyuan is y-up and its paint stage draws the SDXL
# reference from its front camera, which looks down -z.  The paper's camera sits
# on -y, so -y must become +z: (x, y, z) -> (x, z, -y).  A proper rotation
# (det +1): a mirror here would paint, and then show, a mirrored bunny.
TO_HUNYUAN = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
assert abs(np.linalg.det(TO_HUNYUAN) - 1.0) < 1e-12
MIRROR = np.array([1.0, -1.0, 1.0])            # == pfbunny/pf_surface.MIRROR


# --------------------------------------------------------------------------
# manifest + running
# --------------------------------------------------------------------------
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def record(paths, **how):
    """Hash each output and file it under how it was made."""
    manifest = _load_manifest()
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    for p in paths:
        p = Path(p)
        manifest[str(p.relative_to(ASSETS))] = {"sha256": _sha(p), "date": stamp, **how}
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))


def run(cmd, env=None, cwd=None, gpu=True):
    full = dict(os.environ)
    if gpu:
        full["CUDA_VISIBLE_DEVICES"] = GPU
    full.update(env or {})
    print("$", " ".join(str(c) for c in cmd), flush=True)
    t = time.perf_counter()
    subprocess.run([str(c) for c in cmd], check=True, env=full, cwd=cwd or HERE)
    print(f"  ({time.perf_counter() - t:.0f} s)", flush=True)


def fresh(*outputs):
    """True when every output exists and this step may be skipped."""
    if all(Path(o).exists() for o in outputs) and not FORCE:
        print("  cached:", ", ".join(Path(o).name for o in outputs))
        return True
    if FORCE:
        aside = Path("/tmp") / f"ov_assets_old_{int(time.time())}"
        for o in outputs:
            if Path(o).exists():
                aside.mkdir(parents=True, exist_ok=True)
                shutil.move(str(o), aside / Path(o).name)
        if aside.exists():
            print(f"  moved old outputs to {aside}")
    return False


def bunny():
    """The display bunny: bunny.stl mirrored in y, wound OUTWARD.

    ``prep_bunny_sdf`` runs ``fix_normals`` and THEN swaps two axes (itself a
    mirror), so bunny.stl ships wound inward: its volume is -0.198.  Every
    consumer that trusts the winding breaks on that, silently — Hunyuan-Paint
    bakes only the faces it sees from the front and returned an all-black
    texture, and shape diameter casts its rays against the normal, i.e. out of
    the body.  The y-mirror pfbunny applies (without its winding flip) turns it
    outward again.  Face i stays face i, so labels and colours map one to one
    onto bunny.stl and onto every mirrored copy of it.
    """
    import trimesh
    src = HERE / "bunny.stl"
    if not src.exists():
        raise SystemExit("bunny.stl missing: run ../bunny/prep_bunny_sdf.py here first")
    b = trimesh.load(src)             # merged vertices: 80,786 V / 161,568 F
    m = trimesh.Trimesh(np.asarray(b.vertices) * MIRROR, np.asarray(b.faces), process=False)
    assert m.volume > 0 and m.is_winding_consistent, "the display bunny is not outward"
    return m


def bunny_display_stl():
    """The display bunny as STL, for ParaView — which reads a trimesh .ply as a
    bare point cloud (AGENTS.md §5) and rendered the first input card as dots."""
    import trimesh
    path = ASSETS / "bunny_display.stl"
    b = bunny()
    if not path.exists():
        b.export(path)
        record([path], step="display", command="ov_assets.bunny() as STL")
    soup = trimesh.load(path, force="mesh", process=False)
    drift = np.abs(soup.triangles.mean(axis=1) - b.triangles.mean(axis=1)).max()
    assert len(soup.faces) == len(b.faces) and drift < 1e-5, "STL re-ordered faces"
    return path


def bunny_display_file():
    """The display bunny on disk, for the tools that take a path."""
    import trimesh
    path = ASSETS / "bunny_display.ply"
    b = bunny()
    if not path.exists():
        b.export(path)
        record([path], step="display", command="ov_assets.bunny(): bunny.stl * (1,-1,1)")
    back = trimesh.load(path, process=False)
    assert np.array_equal(np.asarray(back.faces), np.asarray(b.faces)), "PLY re-ordered faces"
    return path


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------
def step_input():
    out = ASSETS / "input"
    image, scribble = out / "bunny_image.png", out / "scribble.png"
    if fresh(image, scribble):
        return
    out.mkdir(parents=True, exist_ok=True)
    run(["pvbatch", HERE / "render_ov_input.py"],
        env={**PV_ENV, "OV_INPUT_OUT": str(image), "OV_INPUT_SURFACE": str(bunny_display_stl())})
    _trace_scribble(image, scribble)
    record([image], step="input", command="pvbatch render_ov_input.py", env="paraview")
    record([scribble], step="input", command="ov_assets._trace_scribble(bunny_image.png)",
           env="mechfoundry-warp117")


def _trace_scribble(image_path, out_path, size=1216):
    """A scribble of the bunny, the way a person draws one, like TripoSG's own
    examples: the clay render's silhouette simplified hard (a hand does not
    trace), drawn as a smooth thick marker stroke that overshoots where it
    started, a quicker second pass over the back and the ears, and the few
    marks a hand adds — the split between the ears, an eye, a front leg, a tail.
    A traced outline read as a silhouette, not a scribble.
    """
    from PIL import Image, ImageDraw
    from scipy import interpolate, ndimage
    from skimage import measure

    rgb = np.asarray(Image.open(image_path).convert("RGB")).astype(float)
    mask = (255.0 - rgb).max(axis=2) > 24.0
    mask = ndimage.binary_fill_holes(ndimage.binary_opening(mask, iterations=2))
    ys, xs = np.where(mask)
    lo = np.array([xs.min(), ys.min()])
    span = float(max(np.ptp(xs), np.ptp(ys)))
    rng = np.random.default_rng(7)
    ink = (25, 25, 25)

    def canvas(p):
        return (p - lo) / span * (0.72 * size) + 0.14 * size

    def jitter(q, amp, corr):
        tan = np.gradient(q, axis=0)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        w = ndimage.gaussian_filter1d(rng.normal(0, 1, len(q)), corr)
        return q + nrm * (w / (np.abs(w).max() + 1e-9) * amp)[:, None]

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    def stroke(q, width):
        draw.line([tuple(p) for p in q], fill=ink, width=width, joint="curve")
        for p in (q[0], q[-1]):
            r = width / 2
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=ink)

    def line(a, b, amp, width):
        stroke(jitter(np.linspace(a, b, 70), amp, 10), width)

    contour = max(measure.find_contours(mask.astype(float), 0.5), key=len)
    poly = measure.approximate_polygon(canvas(np.stack([contour[:, 1], contour[:, 0]], 1)),
                                       tolerance=0.022 * size)[:-1]
    closed = np.vstack([poly, poly[:1]])
    tck, _ = interpolate.splprep([closed[:, 0], closed[:, 1]], s=0.0, per=True, k=3)
    base = np.stack(interpolate.splev(np.linspace(0, 1, 900), tck), 1)
    n = len(base)
    # the outline: once round and past the start, then a quick second pass
    q = jitter(base[(np.arange(int(1.04 * n)) + int(0.1 * n)) % n], 0.012 * size, 35)
    q[-40:] += np.linspace(0, 1, 40)[:, None] * np.array([0.02 * size, 0.012 * size])
    stroke(q, 17)
    stroke(jitter(base[(np.arange(int(0.45 * n)) + int(0.55 * n)) % n] + np.array([6, -4]),
                  0.018 * size, 25), 9)
    # the ear split: from the notch between the two ear tips down into the head
    top = base[:, 1]
    tips = sorted((i for i in range(n) if top[i] <= top[(i - 1) % n]
                   and top[i] <= top[(i + 1) % n]), key=lambda i: top[i])[:2]
    a, b = sorted(tips)
    between = np.arange(a, b + 1) if (b - a) < n / 2 else np.r_[b:n, 0:a + 1]
    notch = base[between[np.argmax(top[between])]]
    line(notch, notch + np.array([-0.035, 0.13]) * size, 0.004 * size, 11)
    # an eye, a front leg, a tail
    head = base[np.argmin(base[:, 0])]
    eye = head + np.array([0.11, -0.03]) * size
    r = 0.012 * size
    draw.ellipse([eye[0] - r, eye[1] - r, eye[0] + r, eye[1] + r], fill=ink)
    low = base[np.argmax(base[:, 1])]
    near_floor = base[:, 1] > low[1] - 0.06 * size
    front = base[np.argmin(np.where(near_floor, base[:, 0], 1e9))]
    line(front + np.array([0.09, -0.005]) * size, front + np.array([0.12, -0.13]) * size,
         0.005 * size, 11)
    back = base[np.argmax(np.where(base[:, 1] > low[1] - 0.2 * size, base[:, 0], -1e9))]
    t = np.linspace(0.2, 2.1 * np.pi, 80)
    c = back + np.array([-0.035, -0.02]) * size
    stroke(jitter(np.stack([c[0] + 0.03 * size * np.cos(t), c[1] + 0.03 * size * np.sin(t)], 1),
                  0.003 * size, 8), 9)
    img.save(out_path)
    print(f"wrote {out_path}")

def step_paint():
    import trimesh
    out = ASSETS / "paint"
    painted, reference = out / "painted_yup.glb", out / "sdxl.png"
    if fresh(painted, reference):
        return
    out.mkdir(parents=True, exist_ok=True)
    b = bunny()
    yup = trimesh.Trimesh(b.vertices @ TO_HUNYUAN.T, b.faces, process=False)
    src = out / "bunny_yup.glb"
    yup.export(src)
    run([HUNYUAN_PY, HUNYUAN, "--surface", src, "--prompt", PAINT_PROMPT,
         "--reference", reference, "--output", painted, "--seed", PAINT_SEED,
         *PAINT_QUALITY], cwd=REPO)
    back = trimesh.load(painted, force="mesh", process=False)
    assert len(back.vertices) == len(b.vertices) and len(back.faces) == len(b.faces), \
        "the paint stage changed the surface"
    record([painted, reference], step="paint", env="mechfoundry-hunyuan3d",
           model="Hunyuan3D-2.1 paint + SDXL-base/controlnet-depth-sdxl-1.0",
           command="hunyuan3d_wrapper/generate.py --surface " + " ".join(PAINT_QUALITY),
           seed=PAINT_SEED,
           prompt=PAINT_PROMPT, frame="y-up: (x, y, z)_display -> (x, z, -y)")


def _partfield_python():
    sys.path.insert(0, str(REPO))
    from src.partfield_wrapper.availability import partfield_status
    status = partfield_status()
    if not status.ready:
        raise SystemExit("PartField is not ready: " + status.summary)
    return status.interpreter


def step_segments():
    out = ASSETS / "seg"
    wanted = [out / f"pf_k{k}.{ext}" for k in PART_CLUSTERS for ext in ("npz", "json")]
    if fresh(*wanted):
        return
    out.mkdir(parents=True, exist_ok=True)
    surface = bunny_display_file()
    pf_py = _partfield_python()
    for k in PART_CLUSTERS:
        run([pf_py, REPO / "src/partfield_wrapper/segment.py", "--surface", surface,
             "--output-npz", out / f"pf_k{k}.npz", "--output-json", out / f"pf_k{k}.json",
             "--output-glb", out / f"pf_k{k}.glb", "--feature-cache", out / "pf_cache",
             "--clusters", k, "--cluster-interpreter", WARP_PY], cwd=REPO)
        record([out / f"pf_k{k}.npz", out / f"pf_k{k}.json"], step="segments",
               env="mechfoundry-partfield", model="PartField", command="partfield_wrapper/segment.py",
               clusters=k)
    # the one inference every k was clustered from: pfbunny's feature panel reads it
    record([out / "pf_cache" / "features.npy"], step="segments", env="mechfoundry-partfield",
           model="PartField", command="partfield_wrapper/segment.py (feature cache)")


def step_cad():
    out = ASSETS / "cad"
    found = sorted(p for p in out.glob("*") if p.suffix.lower() in (".step", ".stp"))
    if not found:
        raise SystemExit(f"no STEP file in {out}: download the CAD bunny there first")
    tess = out / "cad_bunny.npz"
    if fresh(tess):
        return
    cad_py = ENVS / "mechfoundry-cadcoder" / "bin" / "python"
    run([cad_py, HERE / "ov_cad_tessellate.py", found[0], tess], cwd=REPO, gpu=False)
    record([found[0]], step="cad", command="downloaded by the user (see SOURCE.txt)")
    record([tess], step="cad", env="mechfoundry-cadcoder", command="ov_cad_tessellate.py")


def _rmbg(net, rgb):
    """TripoSG's matte (scripts/image_process.load_image): RMBG-1.4 at 1024^2,
    normalised as it normalises, upsampled bilinear, min-max stretched."""
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(rgb, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None] / 255.0
    x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
    with torch.no_grad():
        a = net((x - 0.5) / 1.0)[0][0]
    a = F.interpolate(a, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    return ((a - a.min()) / (a.max() - a.min())).cpu().numpy()


def _decontaminate(rgb, alpha, sigma=12.0):
    """Take the old background out of the edge pixels, so the cut-out does not
    carry a grey rim onto the tint: I = a F + (1 - a) B, with B the background
    smoothed in from the pixels RMBG calls background."""
    from scipy import ndimage
    img = rgb.astype(float) / 255.0
    w = (alpha < 0.02).astype(float)
    wb = ndimage.gaussian_filter(w, sigma)
    B = np.stack([ndimage.gaussian_filter(img[..., c] * w, sigma) for c in range(3)], -1)
    B = B / np.maximum(wb, 1e-6)[..., None]
    B[wb < 1e-3] = img[wb < 1e-3]                 # deep inside: no background to take out
    a = alpha[..., None]
    fg = np.where(a > 1e-3, (img - (1.0 - a) * B) / np.maximum(a, 1e-3), img)
    return np.clip(fg, 0.0, 1.0)


def step_cutout():
    from PIL import Image
    out = ASSETS / "input"
    targets = {"bunny_image_cut.png": ASSETS / "input/bunny_image.png",
               "sdxl_cut.png": ASSETS / "paint/sdxl.png"}
    scribble = out / "scribble_cut.png"
    if fresh(*(out / k for k in targets), scribble):
        return
    import torch
    sys.path.insert(0, str(REPO / "src/dependency/TripoSG/scripts"))
    from briarmbg import BriaRMBG
    weights = REPO / "src/dependency/TripoSG/pretrained_weights/RMBG-1.4"
    net = BriaRMBG.from_pretrained(str(weights)).to("cuda").eval()
    for name, src in targets.items():
        rgb = np.asarray(Image.open(src).convert("RGB"))
        alpha = _rmbg(net, rgb)
        fg = _decontaminate(rgb, alpha)
        rgba = np.concatenate([fg, alpha[..., None]], -1)
        Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), "RGBA").save(out / name)
        print(f"wrote {out / name}  (alpha: {np.mean(alpha > 0.5):.3f} opaque)")
    del net
    torch.cuda.empty_cache()
    grey = np.asarray(Image.open(ASSETS / "input/scribble.png").convert("L")).astype(float)
    ink = 25.0                                        # _trace_scribble's ink
    a = np.clip((255.0 - grey) / (255.0 - ink), 0.0, 1.0)
    rgba = np.zeros(grey.shape + (4,), np.uint8)
    rgba[..., :3] = int(ink)
    rgba[..., 3] = (a * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(scribble)
    print(f"wrote {scribble}")
    record([out / k for k in targets], step="cutout", env="mechfoundry-warp117",
           command="ov_assets.step_cutout: RMBG-1.4 (TripoSG pretrained_weights) + decontaminate",
           model="briaai/RMBG-1.4", gpu=GPU)
    record([scribble], step="cutout", env="mechfoundry-warp117",
           command="ov_assets.step_cutout: scribble ink keyed off white")


def step_cadimage():
    from PIL import Image
    from scipy import ndimage
    src = ASSETS / "cad" / "brep_bunny_gemini.jpeg"
    out = ASSETS / "cad" / "brep_bunny_cut.png"
    if not src.exists():
        raise SystemExit(f"no {src}: put the CAD lane's image there first")
    if fresh(out):
        return
    rgb = np.asarray(Image.open(src).convert("RGB")).astype(float)
    grey = rgb.mean(axis=2)
    # The drawing is a white body on white paper, so no colour key separates it
    # (RMBG was not tried: the outline is exact and needs no model).  Its OUTLINE does:
    # a closed dark stroke round the silhouette, filled, is the object; the floor
    # shadow, grey but outside the stroke, is not.
    ink = grey < 120.0
    body = ndimage.binary_fill_holes(ndimage.binary_closing(ink, iterations=3))
    body = ndimage.binary_opening(body, iterations=2)
    labels, n = ndimage.label(body)
    sizes = ndimage.sum(body, labels, index=np.arange(1, n + 1))
    body = labels == 1 + int(np.argmax(sizes))       # the bunny, not a stray stroke
    frac = body.mean()
    assert 0.05 < frac < 0.6, f"the outline did not close round the bunny ({frac:.3f})"
    # the stroke's anti-aliased outer rim: ink over white paper, alpha by darkness
    rim = ndimage.binary_dilation(body, iterations=3) & ~body
    ink_rgb = np.array([38.0, 44.0, 56.0])           # the drawing's stroke colour
    a_rim = np.clip((255.0 - grey) / (255.0 - ink_rgb.mean()), 0.0, 1.0)
    alpha = np.where(body, 1.0, np.where(rim, a_rim, 0.0))
    colour = np.where(body[..., None], rgb, ink_rgb)
    rgba = np.concatenate([colour, alpha[..., None] * 255.0], axis=-1)
    Image.fromarray((rgba + 0.5).clip(0, 255).astype(np.uint8), "RGBA").save(out)
    print(f"wrote {out} ({frac:.1%} of the image is the bunny)")
    record([src], step="cadimage", model="Gemini (Google) image generation",
           command="supplied by the user: a B-rep-surface bunny, stand-in for a Text2CAD / CAD-Coder output")
    record([out], step="cadimage", env="mechfoundry-warp117",
           command="ov_assets.step_cadimage: filled outer outline, stroke rim keyed off white")


def step_cadmodel():
    out = ASSETS / "cad"
    script, tess = out / "bunny_cadcoder.py", out / "cad_bunny.npz"
    if fresh(script, tess):
        return
    out.mkdir(parents=True, exist_ok=True)
    run([WARP_PY, HERE / "ov_cad_fit.py"], gpu=False)
    cad_py = ENVS / "mechfoundry-cadcoder" / "bin" / "python"
    run([cad_py, HERE / "ov_cad_build.py", script, tess], cwd=REPO, gpu=False)
    record([script], step="cadmodel", env="mechfoundry-warp117",
           command="ov_cad_fit.py: sections and ear axes measured on bunny_display + pf_k8")
    record([tess], step="cadmodel", env="mechfoundry-cadcoder",
           command="ov_cad_build.py: cadcoder_wrapper.sandbox.run_sandboxed + finish_mesh")


STEPS = {"input": step_input, "paint": step_paint, "segments": step_segments,
         "cad": step_cad, "cadmodel": step_cadmodel, "cadimage": step_cadimage,
         "cutout": step_cutout}

if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    chosen = sys.argv[1:] or list(STEPS)
    unknown = [s for s in chosen if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    for name in chosen:
        print(f"== {name}", flush=True)
        STEPS[name]()
