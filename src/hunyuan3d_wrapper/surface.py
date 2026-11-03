"""Mesh steps around Hunyuan3D-2.1 that need only numpy, scipy, trimesh and PIL.

Kept apart from ``generate.py`` so they can be tested without the models.
"""

from __future__ import annotations

# Largest half-extent of the fitted surface, inside TripoSG's +-1.005 box -- the
# same frame the other generators hand to the workflow.
HALF_EXTENT = 0.95
# Face budget when --faces is -1: upstream's own FaceReducer default.
DEFAULT_FACES = 40000
# A textured vertex farther than this (relative to the surface's size) from the
# surface it is meant to colour means the paint stage moved the geometry.
MATCH_TOLERANCE = 1e-4


def clean_surface(mesh, source: str = "Hunyuan3D-2.1"):
    """Weld, keep the largest component by volume, close single-triangle holes, orient outward.

    Only holes bounded by three edges are closed (trimesh's ``fill_holes``);
    any larger opening leaves the surface open, and it is refused.

    Returns the watertight mesh and how many loose components were dropped.
    """
    import numpy as np
    import trimesh

    mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise RuntimeError(f"{source} produced an empty surface")
    components = mesh.split(only_watertight=False)
    discarded = max(len(components) - 1, 0)
    if discarded:
        mesh = max(components, key=lambda part: abs(part.volume))
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    if not mesh.is_watertight:
        raise RuntimeError(f"{source} produced no usable watertight surface component")
    mesh.fix_normals()
    if mesh.volume <= 0:
        raise RuntimeError(f"{source} produced an empty solid")
    return mesh, discarded


def fit_to_frame(mesh):
    """Centre the bounding box and scale its largest half-extent to HALF_EXTENT.

    Hunyuan3D-2.1, like TripoSG, is y-up already, so no rotation is applied.
    """
    mesh = mesh.copy()
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2.0)
    mesh.apply_scale(HALF_EXTENT / (float((hi - lo).max()) / 2.0))
    return mesh


def sample_texture(image, uv):
    """Bilinearly sample an RGB(A) image at glTF/OBJ texture coordinates (v up)."""
    import numpy as np

    pixels = np.asarray(image.convert("RGBA"), dtype=np.float64)
    height, width = pixels.shape[:2]
    x = np.clip(np.asarray(uv[:, 0], dtype=np.float64) % 1.0, 0.0, 1.0) * (width - 1)
    y = (1.0 - np.clip(np.asarray(uv[:, 1], dtype=np.float64) % 1.0, 0.0, 1.0)) * (height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = pixels[y0, x0] * (1 - fx) + pixels[y0, x1] * fx
    bottom = pixels[y1, x0] * (1 - fx) + pixels[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def transfer_colours(surface, textured_vertices, textured_uv, texture):
    """Per-vertex RGBA for ``surface`` from a UV-textured copy of the same geometry.

    The paint stage keeps every vertex position and only duplicates vertices
    along its UV seams, so each surface vertex has one or more textured
    vertices at the same place.  Their texture samples are averaged, which
    gives a seam vertex one colour instead of two.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    vertices = np.asarray(surface.vertices, dtype=np.float64)
    textured_vertices = np.asarray(textured_vertices, dtype=np.float64)
    size = float(np.ptp(vertices, axis=0).max())
    distance, index = cKDTree(vertices).query(textured_vertices)
    if len(distance) and float(distance.max()) > MATCH_TOLERANCE * size:
        raise RuntimeError(
            "the painted mesh does not match the generated surface "
            f"(largest vertex offset {distance.max():.3g})"
        )
    samples = sample_texture(texture, np.asarray(textured_uv))
    total = np.zeros((len(vertices), 4))
    count = np.zeros(len(vertices))
    np.add.at(total, index, samples)
    np.add.at(count, index, 1.0)
    colours = np.full((len(vertices), 4), 255.0)
    painted = count > 0
    colours[painted] = total[painted] / count[painted, None]
    if not painted.all():
        # Any vertex the paint stage dropped takes its nearest painted neighbour's colour.
        _, nearest = cKDTree(vertices[painted]).query(vertices[~painted])
        colours[~painted] = colours[painted][nearest]
    colours[:, 3] = 255.0
    return np.clip(np.rint(colours), 0, 255).astype(np.uint8)


def tilt_rotation(degrees: float):
    """A rotation (3x3) by ``degrees`` about x, then by ``degrees`` about y.

    Every direction along a coordinate axis is moved off all three axes by it: x by the turn
    about y, y by the turn about x, z by both.  Row-vector points are turned by ``p @ R.T`` and
    turned back by ``p @ R``.
    """
    import numpy as np

    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    about_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    about_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return about_y @ about_x


def paint_frame(vertices):
    """Centre and half-size that take ``vertices`` into a frame of half-extent 1: ``(p - c) / s``.

    The paint stage unwraps the surface with xatlas on the coordinates it is given, and xatlas
    measures face areas against absolute float32 epsilons: a face smaller than about 1.2e-7 in
    the file's units counts as degenerate.  A small body in metres -- a few centimetres across,
    remeshed to the workflow's 34,000-face budget -- has every face below that (median 6.6e-8
    m^2 in a held-out run, 2026-10-10), xatlas then returns NaN for every UV, and upstream's
    vertex inpainting indexes the texture with them and dies with SIGSEGV.  The same surface
    scaled by two or more unwraps cleanly; in this frame any surface does.  Positions are what
    change, never shapes: the renderer re-centres and re-scales every mesh it loads.
    """
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    half = float((hi - lo).max()) / 2.0
    return (lo + hi) / 2.0, (half if half > 0 else 1.0)


# The plain colour a surface is given when the painter cannot paint it (plain_colour): sRGB of
# colour words first, then of material and finish words, which say a colour when no colour word
# does.  A generic vocabulary, as the words of a texture prompt use it.
PLAIN_COLOUR_WORDS = {
    "red": (190, 40, 40), "crimson": (170, 25, 50), "scarlet": (215, 45, 30), "maroon": (120, 30, 40),
    "burgundy": (125, 25, 45), "pink": (230, 150, 170), "orange": (230, 120, 40),
    "ochre": (200, 140, 50), "amber": (225, 160, 40), "yellow": (230, 200, 60), "gold": (210, 170, 70),
    "golden": (210, 170, 70), "olive": (128, 128, 60), "lime": (150, 200, 70), "green": (60, 140, 70),
    "teal": (40, 130, 130), "turquoise": (60, 190, 180), "cyan": (60, 180, 200), "blue": (50, 90, 180),
    "navy": (30, 40, 90), "violet": (140, 90, 190), "purple": (120, 60, 150), "magenta": (190, 50, 150),
    "brown": (130, 85, 50), "tan": (200, 165, 120), "beige": (215, 200, 170), "cream": (235, 225, 195),
    "ivory": (235, 230, 210), "khaki": (190, 175, 130), "white": (235, 235, 235),
    "silver": (195, 195, 200), "grey": (150, 150, 150), "gray": (150, 150, 150),
    "charcoal": (65, 65, 70), "black": (40, 40, 40),
}
PLAIN_MATERIAL_WORDS = {
    "brass": (200, 170, 100), "bronze": (170, 120, 70), "copper": (190, 110, 70),
    "steel": (165, 168, 172), "aluminium": (190, 192, 196), "aluminum": (190, 192, 196),
    "iron": (110, 110, 112), "chrome": (200, 200, 205), "titanium": (160, 160, 165),
    "wood": (165, 120, 75), "wooden": (165, 120, 75), "oak": (175, 135, 85), "pine": (205, 170, 115),
    "maple": (210, 175, 125), "birch": (215, 190, 150), "walnut": (110, 75, 50),
    "mahogany": (115, 55, 40), "bamboo": (210, 185, 120), "cork": (190, 150, 105),
    "concrete": (160, 160, 155), "stone": (150, 145, 135), "granite": (140, 135, 135),
    "marble": (225, 222, 215), "brick": (165, 80, 60), "terracotta": (190, 100, 70),
    "clay": (180, 110, 80), "ceramic": (225, 225, 220), "porcelain": (235, 235, 230),
    "glass": (200, 215, 220), "rubber": (60, 60, 60), "leather": (130, 80, 50), "bone": (225, 215, 190),
}
# Shade words: a light shade is the colour mixed a third of the way to white, a dark one at 60 %.
PLAIN_LIGHT_WORDS = frozenset(("light", "pale", "pastel", "bright", "soft"))
PLAIN_DARK_WORDS = frozenset(("dark", "deep"))
PLAIN_DEFAULT = (160, 160, 160)


def plain_colour(words: str) -> tuple[tuple[int, int, int], str]:
    """One sRGB colour for a texture prompt, and the word it was read from.

    The first colour word (with the shade word just before it), else the first material word,
    else a mid grey.  Of a prompt of several parts ("grey steel plate, brown wood rail") it is the
    first part's colour.  A neutral colour (equal channels) has its blue moved by one: a surface
    of one exactly neutral colour reads as unpainted (``has_colour``), and this one is painted.
    """
    import re

    tokens = re.findall(r"[a-z]+", str(words or "").lower())
    colour, word = None, ""
    for table in (PLAIN_COLOUR_WORDS, PLAIN_MATERIAL_WORDS):
        index = next((k for k, token in enumerate(tokens) if token in table), None)
        if index is not None:
            colour, word = table[tokens[index]], tokens[index]
            shade = tokens[index - 1] if index > 0 else ""
            if shade in PLAIN_LIGHT_WORDS:
                colour = tuple(round(c + (255 - c) / 3.0) for c in colour)
            elif shade in PLAIN_DARK_WORDS:
                colour = tuple(round(0.6 * c) for c in colour)
            word = f"{shade} {word}" if shade in PLAIN_LIGHT_WORDS | PLAIN_DARK_WORDS else word
            break
    if colour is None:
        colour, word = PLAIN_DEFAULT, "mid grey (no colour word)"
    r, g, b = (int(c) for c in colour)
    if r == g == b:
        b = b + 1 if b < 255 else b - 1
    return (r, g, b), word


def texture_of(textured):
    """The base-colour image of a textured trimesh, from a simple or PBR material."""
    material = textured.visual.material
    image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
    if image is None:
        raise RuntimeError("the painted mesh carries no colour texture")
    return image


def transfer_colours_nearest(surface, painted, texture):
    """Per-vertex RGBA for ``surface`` from a UV-textured *simplified* copy of it.

    Used when the paint stage ran on a decimated copy, so vertices no longer
    coincide: each surface vertex takes the texture at the nearest of a dense
    set of points spread over the painted copy, whose UVs are interpolated
    within their faces.  (A KD-tree over samples, as trimesh's exact
    closest-point query needs rtree, which the environment lacks.)
    """
    import numpy as np
    from scipy.spatial import cKDTree

    vertices = np.asarray(surface.vertices, dtype=np.float64)
    corners = np.asarray(painted.vertices, dtype=np.float64)[painted.faces]
    uv = np.asarray(painted.visual.uv, dtype=np.float64)[painted.faces]
    # Every face's corners and centroid, plus area-weighted extra points so
    # the spacing stays well below the copy's edge length.
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2
    extra = min(max(4 * len(vertices), 8 * len(painted.faces)), 4_000_000)
    rng = np.random.default_rng(0)
    face = np.concatenate([np.repeat(np.arange(len(corners)), 4),
                           rng.choice(len(corners), extra, p=area / area.sum())])
    fixed = np.tile(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]]), (len(corners), 1))
    r = rng.random((extra, 2))
    r = np.where(r.sum(1, keepdims=True) > 1, 1 - r, r)
    weights = np.concatenate([fixed, np.column_stack([1 - r.sum(1), r])])
    points = np.einsum("ij,ijk->ik", weights, corners[face])
    _, nearest = cKDTree(points).query(vertices, workers=-1)
    samples = sample_texture(texture, np.einsum("ij,ijk->ik", weights[nearest], uv[face[nearest]]))
    samples[:, 3] = 255.0
    return np.clip(np.rint(samples), 0, 255).astype(np.uint8)


def has_colour(path) -> bool:
    """Whether a surface file carries colour VoMP can see: vertex colours or a texture.

    Plain white or grey (every vertex the same neutral value, as trimesh
    reports a mesh with no colour at all) counts as none.
    """
    import numpy as np
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    for geometry in scene.geometry.values():
        visual = getattr(geometry, "visual", None)
        if visual is None:
            continue
        if visual.kind == "texture":
            material = visual.material
            if (getattr(material, "image", None) is not None
                    or getattr(material, "baseColorTexture", None) is not None):
                return True
            continue
        if visual.kind == "vertex":
            colours = np.asarray(visual.vertex_colors)[:, :3].astype(int)
            if np.ptp(colours, axis=0).max() > 0 or np.ptp(colours[0]) > 0:
                return True
    return False
