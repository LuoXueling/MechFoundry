"""Blender's view renderer, rasterised on the GPU.

VoMP's mesh path renders its 150 views with Blender Cycles at 128 samples,
which is 159s of a 207s prediction here (71s across four processes).  The
paper's own breakdown spends 2.11s of 3.59s on rendering, because the method
was benchmarked on Gaussian splats, which it rasterises with
``diff_gaussian_rasterization``.  A mesh has no such path upstream -- Blender
is the only renderer VoMP offers it.

What Blender is asked to produce, though, is not obviously a path-tracing
problem: the scene is one object under a material override of a plain diffuse
BSDF at 0.5 grey, lit by three fixed lights, on a transparent film.  Every
view is a shaded silhouette.  nvdiffrast rasterises all 150 of them in 0.042s,
and this module puts the same scene through it: the same normalisation, the
same camera, the same lights.

Measured against Blender's renders of the same eight views of the same mesh:

  geometry   silhouette IoU 0.9999, once the quarter turn Blender's glTF
             importer applies is applied here too.  Without that turn the
             object is correctly framed and wrongly oriented -- 0.93, which
             looks right and is not.
  shading    per-pixel correlation 0.954, mean object luminance 67.3 against
             68.1, spread 28.1 against 25.0, mean error 3.08/255.  What earned
             that was visibility: the same scene without shadow or occlusion
             correlates 0.17, because Blender's contrast is mostly what the
             object hides from its own lights.
  speed      4.12s for 150 views against 71s for Blender across four
             processes.  The rasterisation is 0.042s of it; the rest is the 65
             shadow maps, which are paid once per mesh, and PNG encoding.

The bounce Cycles is allowed (diffuse_bounces = 1) is not reproduced, and the
area lights are treated as domes rather than as the finite panels they are.
Both show up in the 0.046 of correlation that is missing.

Only ``--geo_mode`` applies that grey override, though, and VoMP's inference
does not pass it: Blender renders a mesh with its own material.  An untextured
surface has none and is drawn grey as above.  A coloured one -- a UV texture,
or glTF ``COLOR_0`` -- is drawn with its base colour through Blender's Filmic
view transform (see ``COLOUR_EXPOSURE``); drawn grey instead, VoMP's field for
a textured penguin correlated 0.05 with the one from Blender's renders.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Blender's material override: a diffuse BSDF at 0.5 grey, roughness 1.
ALBEDO = 0.5

# init_lighting(): a point light, and area lights above and below.  Power is
# in watts, so a point light of power P has radiant intensity P / 4pi, and an
# area light of power P over area A has radiance P / (A * pi).
KEY_LIGHT = ((4.0, 1.0, 6.0), 1000.0)
TOP_LIGHT_RADIANCE = 10000.0 / (100.0 * 100.0 * math.pi)
BOTTOM_LIGHT_RADIANCE = 1000.0 / (100.0 * 100.0 * math.pi)

# Blender 3.0 leaves colour management at its default, which is the Filmic view
# transform -- a log encoding and contrast curve that no linear shading model
# reproduces on its own.  Rather than reimplement it, these two numbers were
# fitted by least absolute error against Blender's own renders of eight views
# of a 150K-face mesh, over the 1.3M pixels where both renderers report full
# coverage.  They bring the mean object luminance to 67.3 against Blender's
# 68.1, the spread to 28.1 against 25.0, and the per-pixel error to 3.08/255.
TONE_GAIN = 8.2161
TONE_GAMMA = 1.450


def ensure_toolchain_on_path() -> None:
    """Put this interpreter's own bin on PATH, because the build needs ninja.

    nvdiffrast builds its plugin through ``torch.utils.cpp_extension`` on first
    use, and torch writes and runs a ninja file every time -- cached build or
    not.  pip puts ninja beside the interpreter, and the workflow launches this
    environment's python directly rather than activating the environment,
    because activation costs ~1.8s on every tool call.  So PATH does not carry
    that directory, and the plugin build fails with "Ninja is required to load
    C++ extensions" in the middle of a prediction, which is where it was found.
    """
    directory = str(Path(sys.executable).resolve().parent)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if directory not in entries:
        os.environ["PATH"] = os.pathsep.join([directory, *entries])


def build_context(device: str = "cuda"):
    """The rasteriser's context, building its plugin if that has not happened.

    Worth doing while a worker loads: the build is the one part of rendering
    that can fail for reasons that have nothing to do with the mesh, and a
    worker that cannot rasterise should say so before a prediction is waiting.
    """
    import nvdiffrast.torch as dr

    ensure_toolchain_on_path()
    return dr.RasterizeCudaContext(device=device)


def normalized_mesh(mesh, turn: bool = True):
    """The vertices Blender renders: unit largest extent, centred on origin.

    glTF is Y-up and Blender is Z-up, so Blender's importer turns the mesh a
    quarter turn about X on the way in and trimesh does not.  Skipping it
    leaves the object correctly framed and wrongly oriented, which costs the
    silhouette about 7% of its area against Blender's -- enough to change what
    DINO sees and little enough to look right.  With the turn the two agree to
    an IoU of 0.9999.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    scale = 1.0 / float((high - low).max())
    centred = (vertices - 0.5 * (low + high)) * scale
    if not turn:
        return centred
    return np.stack([centred[:, 0], -centred[:, 2], centred[:, 1]], axis=1)


def camera_to_world(view) -> np.ndarray:
    """Blender's camera for one view: -Z at the origin, +Y up, world Z up.

    ``init_camera`` tracks an empty at the origin with TRACK_NEGATIVE_Z and
    UP_Y, and the render loop places the camera by spherical coordinates, so
    this is a look-at whose up vector is the world's.
    """
    yaw, pitch, radius = float(view["yaw"]), float(view["pitch"]), float(view["radius"])
    position = np.array([
        radius * math.cos(yaw) * math.cos(pitch),
        radius * math.sin(yaw) * math.cos(pitch),
        radius * math.sin(pitch),
    ])
    back = position / max(np.linalg.norm(position), 1e-12)
    world_up = np.array([0.0, 0.0, 1.0])
    # Straight up or down leaves the up vector undefined; Blender's constraint
    # falls back on another axis, and so does this.
    if abs(float(np.dot(back, world_up))) > 1.0 - 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, back)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(back, right)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = right, up, back
    matrix[:3, 3] = position
    return matrix


def _projection(fov: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    """Square-sensor perspective, matching lens = 16 / tan(fov / 2) at 32mm."""
    focal = 1.0 / math.tan(0.5 * fov)
    matrix = np.zeros((4, 4))
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -2.0 * far * near / (far - near)
    matrix[3, 2] = -1.0
    return matrix



# How many directions the dome lights are sampled over, and how far a surface
# may sit behind the nearest one toward a light before it counts as shadowed.
# The bias is in object units, where the object spans one.
DOME_DIRECTIONS = 64
SHADOW_BIAS = 2.0e-3

# The area lights are 100m across at +/-10m, so from anywhere on the object
# they fill everything above (or below) about 11 degrees from the horizon.
DOME_HORIZON = 0.2


def _sphere_directions(count: int, device):
    """Directions spread evenly over the sphere, by the Fibonacci lattice."""
    import torch

    index = torch.arange(count, dtype=torch.float32, device=device) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    angle = index * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle), z], 1)


def _basis(direction):
    """Any two axes across a direction, for an orthographic map along it."""
    import torch

    reference = torch.tensor(
        [0.0, 0.0, 1.0] if abs(float(direction[2])) < 0.9 else [1.0, 0.0, 0.0],
        device=direction.device,
    )
    x_axis = torch.linalg.cross(reference, direction)
    x_axis = x_axis / x_axis.norm().clamp_min(1e-12)
    return x_axis, torch.linalg.cross(direction, x_axis)


def _visible_from(context, vertices, faces, direction, resolution: int = 1024):
    """Which vertices the surface does not hide from a far light along ``direction``.

    An orthographic depth map along the direction, which is what a light 100m
    across at 10m is: the shadow it casts is the object's own occlusion, and
    the alternative -- no shadow at all -- is what left this renderer's images
    correlating 0.17 with Blender's.
    """
    import nvdiffrast.torch as dr
    import torch

    x_axis, y_axis = _basis(direction)
    span = float(vertices.norm(dim=1).max()) * 1.05 + 1e-6
    toward = (vertices * direction).sum(1, keepdim=True)
    clip = torch.cat([
        (vertices * x_axis).sum(1, keepdim=True) / span,
        (vertices * y_axis).sum(1, keepdim=True) / span,
        -toward / span,
        torch.ones_like(toward),
    ], dim=1).unsqueeze(0).contiguous()
    raster, _ = dr.rasterize(context, clip, faces, resolution=[resolution, resolution])
    nearest, _ = dr.interpolate(toward.unsqueeze(0).contiguous(), raster, faces)
    # Sample the map where each vertex lands.  grid_sample and nvdiffrast agree
    # on where -1 is, so the coordinates go straight across.
    grid = clip[:, :, :2].unsqueeze(1)
    sampled = torch.nn.functional.grid_sample(
        nearest.permute(0, 3, 1, 2), grid, mode="nearest",
        padding_mode="border", align_corners=False,
    )
    return toward.squeeze(1) >= sampled.reshape(-1) - SHADOW_BIAS


def vertex_irradiance(vertices, faces, normals, context):
    """Light arriving at each vertex from the three fixed lights, shadows included.

    Every light in ``init_lighting`` is fixed and every surface is matte, so
    what a vertex receives does not depend on where it is looked at from.
    Computing it once per mesh rather than once per pixel per view is what
    makes 150 views cost one rasterisation each.
    """
    import torch

    position, power = KEY_LIGHT
    key_position = torch.tensor(position, dtype=torch.float32, device=vertices.device)
    delta = key_position - vertices
    distance = delta.norm(dim=1, keepdim=True).clamp_min(1e-6)
    key_direction = delta / distance
    lit = (normals * key_direction).sum(1).clamp_min(0.0)
    key_visible = _visible_from(
        context, vertices, faces,
        key_position / key_position.norm(),
    ).float()
    irradiance = key_visible * lit * (power / (4.0 * math.pi)) / distance.squeeze(1).square()

    directions = _sphere_directions(DOME_DIRECTIONS, vertices.device)
    weight = 4.0 * math.pi / DOME_DIRECTIONS
    for direction in directions:
        radiance = (
            TOP_LIGHT_RADIANCE if float(direction[2]) > DOME_HORIZON
            else BOTTOM_LIGHT_RADIANCE if float(direction[2]) < -DOME_HORIZON
            else 0.0
        )
        if radiance == 0.0:
            continue
        facing = (normals * direction).sum(1).clamp_min(0.0)
        if not bool((facing > 0).any()):
            continue
        visible = _visible_from(context, vertices, faces, direction).float()
        irradiance = irradiance + weight * radiance * facing * visible
    return irradiance


def tone(radiance, tone_gain, tone_gamma):
    """Blender's Filmic view transform, as fitted (see TONE_GAIN)."""
    return (tone_gain * radiance).clamp(0.0, 1.0) ** (1.0 / tone_gamma)


# Blender 3.0's colour management, which VoMP's renders go through at its
# defaults: the Filmic view, no look, exposure 0.  Its OCIO config defines the
# view as a log2 shaper over [-12.47, 12.53] stops, a 65^3 desaturation cube,
# a rescale by 1 / 0.66, and the base-contrast curve.
FILMIC_LOG2_RANGE = (-12.473931188, 12.526068812)
FILMIC_CUBE_SCALE = 0.66

# A coloured surface goes through that transform rather than the grey scene's
# fitted power curve, which cannot follow Filmic's toe and shoulder across dark
# and bright albedos.  Blender's glTF material is a Principled BSDF, whose
# specular (F0 about 0.04) is most of what a near-black albedo sends back; it
# is folded in as a view-independent floor on the albedo.  Both numbers were
# fitted by least absolute error against Blender's renders of 30 of the 150
# views of a Hunyuan3D-2.1 textured mesh: per-pixel correlation 0.942 (0.911
# with the grey curve and no floor), mean object luminance 92.3 against 94.0,
# spread 57.3 against 60.0, mean error 14.2/255 (34.3 before).
COLOUR_EXPOSURE = 4.5
SPECULAR_FLOOR = 0.02


def _filmic_tables(device):
    """Blender's own Filmic LUTs, read from the Blender VoMP installs."""
    from src.vomp_wrapper.availability import settings

    import torch

    root = Path(settings()[1])
    folders = sorted(root.glob("blender-*/*/datafiles/colormanagement/filmic"))
    if not folders:
        raise FileNotFoundError(f"no Blender colour management under {root}")
    folder = folders[-1]
    lines = (folder / "filmic_desat65cube.spi3d").read_text().split("\n")
    size = int(lines[2].split()[0])
    table = np.loadtxt(lines[3:3 + size ** 3], dtype=np.float32)
    # Rows are (r, g, b) indices then the output; store the cube as [c, r, g, b].
    cube = np.zeros((3, size, size, size), dtype=np.float32)
    index = table[:, :3].astype(int)
    cube[:, index[:, 0], index[:, 1], index[:, 2]] = table[:, 3:].T
    curve_lines = (folder / "filmic_to_0-70_1-03.spi1d").read_text().split("\n")
    length = int(next(line for line in curve_lines if line.startswith("Length")).split()[1])
    start = next(i for i, line in enumerate(curve_lines) if line.strip() == "{") + 1
    curve = np.array([float(value) for value in curve_lines[start:start + length]], dtype=np.float32)
    return (torch.tensor(cube, device=device).unsqueeze(0),
            torch.tensor(curve, device=device))


_FILMIC: dict = {}


def filmic(radiance):
    """Blender's Filmic sRGB view transform of linear RGB ``radiance`` (..., 3)."""
    import torch

    device = radiance.device
    if device not in _FILMIC:
        _FILMIC[device] = _filmic_tables(device)
    cube, curve = _FILMIC[device]
    low, high = FILMIC_LOG2_RANGE
    shaped = ((torch.log2(radiance.clamp_min(2.0 ** low)) - low) / (high - low)).clamp(0.0, 1.0)
    shape = shaped.shape
    # grid_sample's (x, y, z) index (W, H, D), which is (b, g, r) here.
    grid = (shaped.reshape(1, -1, 1, 1, 3).flip(-1) * 2.0 - 1.0)
    desaturated = torch.nn.functional.grid_sample(
        cube, grid, mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(3, -1).T.reshape(shape)
    position = (desaturated / FILMIC_CUBE_SCALE).clamp(0.0, 1.0) * (len(curve) - 1)
    lower = position.floor().long().clamp(max=len(curve) - 2)
    fraction = position - lower
    return curve[lower] * (1.0 - fraction) + curve[lower + 1] * fraction


def vertex_radiance(vertices, faces, normals, context, tone_gain, tone_gamma):
    """Outgoing radiance at each vertex of a uniformly grey surface, tone mapped."""
    irradiance = vertex_irradiance(vertices, faces, normals, context)
    return tone((ALBEDO / math.pi) * irradiance, tone_gain, tone_gamma)


def srgb_to_linear(values):
    """Decode sRGB-encoded colour in [0, 1], as Blender does for a colour texture."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(values <= 0.04045, values / 12.92,
                    ((values + 0.055) / 1.055) ** 2.4).astype(np.float32)


# Views are 512 px, so detail beyond this only costs memory: a 16k texture
# (GVM's Jobba sofa) is 3 GB as linear floats.
MAX_TEXTURE_SIZE = 4096


def surface_albedo(mesh):
    """The base colour Blender would shade the mesh with, or None for none.

    Upstream renders a mesh with its own material -- only ``--geo_mode``
    overrides it with grey, and inference does not pass it -- so a textured
    surface reaches DINO in colour.  Returns ``("texture", uv, image)`` with the
    image linear and its first row at v = 0, ``("vertex", colours)`` with
    ``COLOR_0`` decoded to linear, or None, which keeps the grey scene the tone
    curve was fitted on.

    ``COLOR_0`` is read as sRGB although glTF defines it as linear: the
    Hunyuan3D wrapper samples it straight from an sRGB texture, as trimesh and
    most tools write it, and read as linear its near-black renders teal.
    Blender reads none of it -- trimesh writes vertex colours with no
    material, and Blender's importer only applies them through one -- so here
    the renderer draws what the texture looked like, not what Blender would.
    """
    visual = getattr(mesh, "visual", None)
    kind = getattr(visual, "kind", None)
    if kind == "texture" and getattr(visual, "uv", None) is not None:
        material = visual.material
        image = getattr(material, "baseColorTexture", None)
        # glTF's base-colour factor tints its texture; an OBJ's Kd does not
        # tint its map_Kd -- Blender's importer draws the image alone.
        factor = getattr(material, "baseColorFactor", None)
        if image is None:
            image = getattr(material, "image", None)
        if image is not None:
            if max(image.size) > MAX_TEXTURE_SIZE:
                from PIL import Image
                ratio = MAX_TEXTURE_SIZE / max(image.size)
                image = image.convert("RGB").resize(
                    (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                    Image.Resampling.BOX)
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            # trimesh puts v = 0 at the bottom of the image, and nvdiffrast
            # reads the first row at v = 0.
            linear = srgb_to_linear(pixels[::-1])
            if factor is not None:
                factor = np.asarray(factor, dtype=np.float32)[:3]
                linear = linear * (factor / 255.0 if factor.max() > 1.0 else factor)
            return "texture", np.asarray(visual.uv, dtype=np.float32), np.ascontiguousarray(linear)
    if kind == "vertex":
        colours = np.asarray(visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        return "vertex", srgb_to_linear(colours)
    return None


# Blender's default surface for a mesh with no material: a Principled BSDF
# whose base colour is 0.8 grey.
DEFAULT_BASE_COLOUR = 0.8


def material_colour(visual):
    """A material's plain base colour, linear, or None: glTF base colour or OBJ Kd."""
    material = getattr(visual, "material", None)
    for name in ("baseColorFactor", "diffuse", "main_color"):
        value = getattr(material, name, None)
        if value is not None:
            value = np.asarray(value, dtype=np.float32)[:3]
            return value / 255.0 if value.max() > 1.0 else value
    return None


class SceneSurface:
    """Every part of a mesh file with its own base colour, as Blender draws them.

    A file with several materials -- an OBJ with an MTL, a multi-primitive glTF
    -- is several geometries, each with a texture, a plain material colour or
    vertex colours.  Merged into one mesh, only one of those survives, and a
    GVM test table rendered with its teal legs black and its wooden top grey.
    Here the parts are concatenated and each face keeps its source: per-vertex
    ``rgb`` (linear) for plain and vertex colours, or an index into
    ``textures`` (linear images, first row at v = 0) looked up through ``uv``.
    ``coloured`` is False when no part has any colour, which keeps the grey
    scene the tone curve was fitted on.
    """

    def __init__(self, mesh_path):
        import trimesh

        from PIL import Image

        # PIL refuses images over ~179 MP as decompression bombs, and trimesh
        # then drops the texture without a word: the part renders in its plain
        # Kd.  Blender loads them, so lift the limit for this trusted file.
        limit, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None
        try:
            loaded = trimesh.load(mesh_path, force="scene", process=False)
        finally:
            Image.MAX_IMAGE_PIXELS = limit
        parts = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not parts:
            raise ValueError(f"no triangle geometry in {mesh_path}")
        self.mesh = trimesh.util.concatenate(
            [trimesh.Trimesh(g.vertices, g.faces, process=False) for g in parts]
        )
        rgb, uv, face_texture, self.textures = [], [], [], []
        self.coloured = False
        for g in parts:
            n = len(g.vertices)
            albedo = surface_albedo(g)
            texture = -1
            if albedo is not None and albedo[0] == "texture":
                texture = len(self.textures)
                self.textures.append(albedo[2])
                rgb.append(np.ones((n, 3), dtype=np.float32)); uv.append(albedo[1])
                self.coloured = True
            elif albedo is not None:
                rgb.append(albedo[1]); uv.append(np.zeros((n, 2), dtype=np.float32))
                self.coloured = True
            else:
                colour = material_colour(g.visual)
                self.coloured |= colour is not None
                rgb.append(np.tile(colour if colour is not None else DEFAULT_BASE_COLOUR, (n, 1)).astype(np.float32))
                uv.append(np.zeros((n, 2), dtype=np.float32))
            face_texture.append(np.full(len(g.faces), texture, dtype=np.int64))
        self.rgb = np.concatenate(rgb).astype(np.float32)
        self.uv = np.concatenate(uv).astype(np.float32)
        self.face_texture = np.concatenate(face_texture)


def welded_normals(mesh) -> np.ndarray:
    """Vertex normals averaged across UV seams, as Blender's importer welds them.

    A textured mesh repeats a vertex wherever its UVs split; trimesh's own
    normals then stop at every seam and the seams show up as creases in the
    shading.  Blender imports glTF with ``merge_vertices=True``.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    _, inverse = np.unique(vertices.round(7), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    welded = inverse[faces]
    corners = vertices[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    summed = np.zeros((inverse.max() + 1, 3))
    for corner in range(3):
        np.add.at(summed, welded[:, corner], face_normals)
    summed /= np.maximum(np.linalg.norm(summed, axis=1, keepdims=True), 1e-20)
    return summed[inverse]


# The renders this process last wrote, by resolved path.  The feature stage
# opens each one and decodes it -- 0.44s of a 4.90s prediction -- immediately
# after this module encoded it.  Holding them costs 161 MiB for a 150-view
# render and is emptied at the start of the next one, so only the current
# render is ever in it.  ``patches.patch_image_cache`` is what reads it; on its
# own this dict changes nothing.
RENDERED: dict[str, object] = {}


def render_views(mesh_path, views, output_dir, image_size: int = 512,
                 output_size: int | None = None,
                 tone_gain: float | None = None, tone_gamma: float | None = None,
                 batch_size: int = 25, device: str = "cuda") -> list[dict]:
    """Render ``views`` of a mesh and return the frames Blender would have.

    Writes ``NNN.png`` and ``transforms.json`` into ``output_dir`` in the
    layout ``_render_views_sequential`` produces, so the caller cannot tell the
    two apart.  Images are RGBA over a transparent film, as Blender's are.

    Views are rasterised in batches and the shading is already done, so the GPU
    work per view is one rasterisation and two interpolations.  Encoding the
    PNGs is then the slowest part left, and it happens on worker threads
    because PIL gives up the interpreter lock while it compresses.
    """
    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    tone_gain = TONE_GAIN if tone_gain is None else tone_gain
    tone_gamma = TONE_GAMMA if tone_gamma is None else tone_gamma
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    RENDERED.clear()
    surface = SceneSurface(mesh_path)
    mesh = surface.mesh
    from src.vomp_wrapper.patches import TURNED_FORMATS

    # Blender turns only glTF and OBJ; a PLY or STL goes in as it is.
    turn = Path(mesh_path).suffix.lower() in TURNED_FORMATS
    placed = normalized_mesh(mesh, turn=turn)
    vertices = torch.tensor(placed, dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
    normals = torch.tensor(
        welded_normals(mesh) if surface.coloured else np.asarray(mesh.vertex_normals),
        dtype=torch.float32, device=device,
    )
    # The mesh is turned on the way in, so its normals turn with it.
    if turn:
        normals = torch.stack([normals[:, 0], -normals[:, 2], normals[:, 1]], dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1)

    context = build_context(device)
    # Cycles shades whichever side of a face the camera sees, flipping the
    # normal for a face seen from behind.  Light is computed for both sides of
    # every vertex and each pixel takes the side its face turns to the camera:
    # with one side only, a GVM test table's leg rods, whose normals point
    # inward, rendered black where Blender draws them teal.
    irradiance = vertex_irradiance(vertices, faces, normals, context)
    irradiance_back = vertex_irradiance(vertices, faces, -normals, context)
    corners = vertices[faces.long()]
    face_normals = torch.nn.functional.normalize(
        torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), dim=1
    )
    textures = []
    if not surface.coloured:
        # Grey: the tone curve is applied per vertex, exactly as it was fitted.
        shade = tone((ALBEDO / math.pi) * irradiance, tone_gain, tone_gamma)
        shade_back = tone((ALBEDO / math.pi) * irradiance_back, tone_gain, tone_gamma)
        attributes = torch.cat([
            shade.unsqueeze(1), shade_back.unsqueeze(1), vertices,
        ], dim=1).contiguous()
    else:
        # Irradiance, UV and plain base colour go through the rasteriser; each
        # face's texture is looked up and multiplied in per pixel, so it is not
        # blurred to vertex spacing.
        attributes = torch.cat([
            irradiance.unsqueeze(1) / math.pi,
            torch.tensor(surface.uv, device=device),
            torch.tensor(surface.rgb, device=device),
            irradiance_back.unsqueeze(1) / math.pi,
            vertices,
        ], dim=1).contiguous()
        textures = [torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0)
                    for t in surface.textures]
        face_texture = torch.tensor(surface.face_texture, device=device)
    homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)

    poses = [camera_to_world(view) for view in views]
    frames = [
        {"file_path": f"{index:03d}.png", "camera_angle_x": float(view["fov"]),
         "transform_matrix": pose.tolist()}
        for index, (view, pose) in enumerate(zip(views, poses))
    ]

    def write(index, image):
        picture = Image.fromarray(image, mode="RGBA")
        # The feature stage opens every render and resamples it from the size
        # the training pipeline rendered at to the size DINO reads, which cost
        # 0.88s of a 5.73s prediction on one thread.  Doing it here is the same
        # filter on the same pixels, on eight threads, behind the next batch's
        # rasterisation -- and Pillow's resize returns early when the size it
        # is asked for is the size it already has.
        if output_size is not None and output_size != image_size:
            picture = picture.resize(
                (output_size, output_size), Image.Resampling.LANCZOS
            )
        # Level 1 costs a few percent of size against level 6 and a third of
        # the time, and nothing downstream reads these twice.
        path = output_dir / f"{index:03d}.png"
        picture.save(path, compress_level=1)
        RENDERED[str(path.resolve())] = picture

    matrices = torch.tensor(
        np.stack([_projection(float(view["fov"])) @ np.linalg.inv(pose)
                  for view, pose in zip(views, poses)]),
        dtype=torch.float32, device=device,
    )
    # Encoding is what is left to pay: the whole GPU pass is 0.16s for 150
    # views and one thread spends 1.32s turning them into PNGs.  Each batch is
    # handed to the pool and not waited on, so the next batch's rasterisation
    # runs while the last one compresses.
    with ThreadPoolExecutor(max_workers=8) as writers:
        pending = []
        for start in range(0, len(views), batch_size):
            stop = min(start + batch_size, len(views))
            clip = torch.einsum(
                "vc,bdc->bvd", homogeneous, matrices[start:stop]
            ).contiguous()
            raster, derivatives = dr.rasterize(
                context, clip, faces, resolution=[image_size, image_size]
            )
            batch = attributes.unsqueeze(0).expand(stop - start, -1, -1).contiguous()
            # Which side of its face each pixel sees: towards the camera or away.
            triangle = raster[..., 3].long() - 1
            eye = torch.tensor(np.stack([pose[:3, 3] for pose in poses[start:stop]]),
                               dtype=torch.float32, device=device)[:, None, None, :]
            if not surface.coloured:
                values, _ = dr.interpolate(batch, raster, faces)
                seen_from_front = ((face_normals[triangle.clamp_min(0)]
                                    * (eye - values[..., 2:5])).sum(-1, keepdim=True) >= 0)
                shade = torch.where(seen_from_front, values[..., :1], values[..., 1:2])
                colour = shade.expand(-1, -1, -1, 3)
            else:
                values, uv_derivatives = dr.interpolate(
                    batch, raster, faces, rast_db=derivatives, diff_attrs=[1, 2],
                )
                seen_from_front = ((face_normals[triangle.clamp_min(0)]
                                    * (eye - values[..., 7:10])).sum(-1, keepdim=True) >= 0)
                light = torch.where(seen_from_front, values[..., :1], values[..., 6:7])
                base = values[..., 3:6]
                if textures:
                    # Which texture each pixel's face uses (-1: none, or background).
                    source = torch.where(
                        triangle >= 0, face_texture[triangle.clamp_min(0)],
                        torch.full_like(triangle, -1),
                    )
                    for index, texture in enumerate(textures):
                        where = source == index
                        if not bool(where.any()):
                            continue
                        looked_up = dr.texture(
                            texture, values[..., 1:3].contiguous(),
                            uv_da=uv_derivatives.contiguous(),
                            filter_mode="linear-mipmap-linear", boundary_mode="clamp",
                        )
                        base = torch.where(where.unsqueeze(-1), base * looked_up, base)
                base = base * (1.0 - SPECULAR_FLOOR) + SPECULAR_FLOOR
                colour = filmic(COLOUR_EXPOSURE * base * light)
            colour = torch.cat([
                colour[..., :3], (raster[..., 3:4] > 0).float()
            ], dim=-1).contiguous()
            colour = dr.antialias(colour, raster, clip, faces)
            # nvdiffrast rasterises with the origin at the bottom left, so the
            # flip happens here rather than as a reversed host view the encoder
            # would have to copy anyway.
            images = (
                colour.clamp(0.0, 1.0).flip(1) * 255.0
            ).round().to(torch.uint8).cpu().numpy()
            pending.extend(
                writers.submit(write, index, images[index - start])
                for index in range(start, stop)
            )
        for future in pending:
            future.result()

    (output_dir / "transforms.json").write_text(
        json.dumps({"frames": frames}, indent=4), encoding="utf-8"
    )
    return frames
