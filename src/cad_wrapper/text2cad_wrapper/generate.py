#!/usr/bin/env python
"""Text -> CAD sequence -> watertight surface mesh with Text2CAD.

Text2CAD decodes a prompt into a quantized sketch-and-extrude sequence, which
the upstream ``CadSeqProc`` package builds into a B-rep.  That solid is fused,
tessellated with its B-rep faces recorded per triangle, and normalised into
the same [-1, 1] box a TripoSG surface occupies, so everything downstream
treats the two alike.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from src.cad_wrapper.cad_common.mesh import (
    SURFACE_TYPES, face_labels_path, finish_mesh, normalize, write_face_labels,
)
from src.cad_wrapper.text2cad_wrapper.availability import settings, text_encoder_cache

# Nucleus probability for the sampled retries after a failed greedy decode.
RETRY_NUCLEUS = 0.9


def fuse_solids(shape):
    """One solid from the builder's result.

    A sketch with several faces is extruded into one prism per face and the
    prisms are returned side by side, overlapping wherever the faces did; their
    surfaces together are not the boundary of anything.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer

    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    if len(solids) < 2:
        return shape
    fused = solids[0]
    for solid in solids[1:]:
        operation = BRepAlgoAPI_Fuse(fused, solid)
        if not operation.IsDone():
            raise ValueError("could not fuse the CAD bodies")
        fused = operation.Shape()
    unify = ShapeUpgrade_UnifySameDomain(fused, True, True, True)
    unify.Build()
    return unify.Shape()


def tessellate(shape, relative_deflection: float = 1e-3):
    """Triangulate a B-rep in double precision, one node set per face.

    Also returns which B-rep face each triangle came from, and each face's
    surface type: the part's own partition into faces, kept for segmentation.
    """
    import numpy as np
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    size = max(x1 - x0, y1 - y0, z1 - z0)
    BRepMesh_IncrementalMesh(shape, relative_deflection * size, False, 0.5, True).Perform()
    vertices, faces, face_ids, surface_types = [], [], [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None:
            continue
        offset = len(vertices)
        transform = location.Transformation()
        for i in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(i).Transformed(transform)
            vertices.append((point.X(), point.Y(), point.Z()))
        reverse = face.Orientation() == TopAbs_REVERSED
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = (offset + index - 1 for index in triangulation.Triangle(i).Get())
            faces.append((a, c, b) if reverse else (a, b, c))
        face_ids.extend([len(surface_types)] * triangulation.NbTriangles())
        kind = int(BRepAdaptor_Surface(face).GetType())
        surface_types.append(SURFACE_TYPES[kind] if kind < len(SURFACE_TYPES) else "other")
    return (np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64),
            np.asarray(face_ids, dtype=np.int64), surface_types)


def build_solid(cad_vec, root: Path):
    """The B-rep a decoded CAD vector describes, built by upstream CadSeqProc."""
    if os.fspath(root) not in sys.path:
        sys.path.insert(0, os.fspath(root))
    from CadSeqProc.cad_sequence import CADSequence
    from CadSeqProc.utility.macro import N_BIT

    sequence = CADSequence.from_vec(cad_vec, bit=N_BIT, post_processing=True)
    sequence.create_cad_model()
    return sequence.cad_model


def cad_to_mesh(shape):
    """Fuse a B-rep into one solid and tessellate it watertight.

    Returns the mesh, the B-rep face of each of its triangles, each B-rep
    face's surface type, and how many loose components were dropped.
    """
    vertices, triangles, labels, surface_types = tessellate(fuse_solids(shape))
    mesh, labels, discarded = finish_mesh(vertices, triangles, labels, "Text2CAD sequence")
    return mesh, labels, surface_types, discarded


class Text2CADRuntime:
    """The Text2CAD model with its checkpoint loaded, kept resident by the server."""

    def __init__(self, checkpoint: Path, device: str = "cuda"):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.root = settings()[1]
        self.model = None

    def preload(self):
        if self.model is not None:
            return self
        import torch
        import yaml

        if os.fspath(self.root) not in sys.path:
            sys.path.insert(0, os.fspath(self.root))
        from Cad_VLM.models.text2cad import Text2CAD
        from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH

        config = yaml.safe_load(
            (self.root / "Cad_VLM" / "config" / "inference_user_input.yaml").read_text()
        )
        config["text_encoder"]["text_embedder"]["cache_dir"] = os.fspath(
            text_encoder_cache(self.root)
        )
        cad_config = {**config["cad_decoder"], "cad_seq_len": MAX_CAD_SEQUENCE_LENGTH}
        model = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value
                 for key, value in state["model_state_dict"].items()}
        # Upstream loads with strict=False.  The frozen BERT encoder is not in
        # the checkpoint, but everything else must be, or the decoder would
        # silently run on its random initial weights.
        result = model.load_state_dict(state, strict=False)
        missing = [key for key in result.missing_keys
                   if not key.startswith("base_text_embedder.")]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"Text2CAD checkpoint does not fit the model: missing {missing[:5]}, "
                f"unexpected {result.unexpected_keys[:5]}"
            )
        self.model = model.to(self.device).eval()
        self.max_length = MAX_CAD_SEQUENCE_LENGTH
        return self

    def complete(self, prompt: str, sample: bool, seed: int):
        """One decoded CAD vector: greedy, or nucleus-sampled for a retry."""
        import torch

        self.preload()
        torch.manual_seed(seed)
        with torch.no_grad():
            decoded = self.model.test_decode(
                texts=[prompt], maxlen=self.max_length,
                nucleus_prob=RETRY_NUCLEUS if sample else 0, topk_index=1,
                device=self.device,
            )
        return decoded["cad_vec"][0].cpu().numpy()


def parser() -> argparse.ArgumentParser:
    _, _, checkpoint = settings()
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--prompt", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--faces", type=int, default=-1)
    result.add_argument("--attempts", type=int, default=3,
                        help="the first attempt decodes greedily; retries sample")
    result.add_argument("--checkpoint", type=Path, default=checkpoint)
    result.add_argument("--device", default="cuda")
    return result


def run(args, runtime: Text2CADRuntime | None = None) -> list[str]:
    import json

    import numpy as np

    if not args.prompt.strip():
        raise ValueError("Text2CAD needs a text prompt")
    if args.faces != -1 and args.faces < 1000:
        raise ValueError("--faces must be -1 or at least 1000")
    runtime = runtime or Text2CADRuntime(args.checkpoint, args.device)
    errors = []
    for attempt in range(max(args.attempts, 1)):
        cad_vec = runtime.complete(args.prompt, sample=attempt > 0, seed=args.seed + attempt)
        try:
            shape = build_solid(cad_vec, runtime.root)
            mesh, labels, surface_types, discarded = cad_to_mesh(shape)
            break
        except Exception as exc:  # the B-rep kernel raises bare Exceptions
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            print(errors[-1], file=sys.stderr)
    else:
        raise RuntimeError("Text2CAD produced no valid solid; " + "; ".join(errors))
    surface, labels = normalize(mesh, args.faces, labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    surface.export(output)
    sequence = output.with_suffix(output.suffix + ".cad_vec.npy")
    np.save(sequence, cad_vec)
    face_labels = face_labels_path(output)
    write_face_labels(face_labels, labels, surface_types)
    artifacts = [str(output), str(sequence), str(face_labels)]
    print(json.dumps({
        "artifacts": artifacts, "vertices": len(surface.vertices),
        "faces": len(surface.faces), "cad_faces": len(set(labels.tolist())),
        "attempts": attempt + 1, "discarded_components": discarded,
    }))
    return artifacts


def main():
    run(parser().parse_args())
    sys.stdout.flush(); sys.stderr.flush()
    # Release the GPU at once instead of waiting on interpreter teardown.
    os._exit(0)


if __name__ == "__main__":
    main()
