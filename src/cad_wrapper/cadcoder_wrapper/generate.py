#!/usr/bin/env python
"""Text -> CadQuery script -> watertight surface mesh with CAD-Coder.

CAD-Coder (a Qwen2.5-7B tuned with geometric rewards) reasons about the
description and answers with a CadQuery script.  The script is built only in
the sandbox of ``sandbox.py``; what comes back -- the solid tessellated face by
face -- goes through the same post-processing as every CAD generator here:
welded, checked closed, refined to a face budget, normalised into TripoSG's
[-1, 1] box, and saved with the B-rep face of every triangle.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from src.cad_wrapper.cad_common.mesh import face_labels_path, finish_mesh, normalize, write_face_labels
from src.cad_wrapper.cadcoder_wrapper.availability import settings
from src.cad_wrapper.cadcoder_wrapper.sandbox import run_sandboxed

# Room for the reasoning and the script.  Upstream's example allows 2048; a
# description of several parts (a cup with a handle, a table) needs more.
MAX_NEW_TOKENS = 3072
# Upstream's batch-inference sampling, used for the retries.
RETRY_TEMPERATURE = 0.7
RETRY_TOP_P = 0.7


def prompt_template(root: Path) -> str:
    """The instruction CAD-Coder was trained with, read from upstream's example.

    Upstream's ``inference.py`` holds it as one literal ending in a sample
    description; everything up to that description is the template.
    """
    tree = ast.parse((root / "inference.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(target, "id", None) == "prompt" for target in node.targets):
            text = ast.literal_eval(node.value)
            marker = "description:\n"
            return text[:text.index(marker) + len(marker)]
    raise RuntimeError("the CAD-Coder prompt template was not found in inference.py")


def extract_code(response: str) -> str:
    """The script from the answer, as upstream extracts it."""
    for pattern in (r"\\boxed\{```python\n(.*?)```\}", r"```python\n(.*?)```"):
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    raise ValueError("the answer contains no Python code block")


class CADCoderRuntime:
    """CAD-Coder loaded on the GPU, kept resident by the server."""

    def __init__(self, weights: Path, device: str = "cuda"):
        self.weights = Path(weights)
        self.device = device
        self.root = settings()[1]
        self.model = None

    def preload(self):
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.weights)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights, dtype=torch.bfloat16
        ).to(self.device).eval()
        self.template = prompt_template(self.root)
        return self

    def complete(self, prompt: str, sample: bool, seed: int) -> str:
        import torch

        self.preload()
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.template + prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        torch.manual_seed(seed)
        options = ({"do_sample": True, "temperature": RETRY_TEMPERATURE, "top_p": RETRY_TOP_P}
                   if sample else {"do_sample": False, "temperature": None, "top_p": None,
                                   "top_k": None})
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **options)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


def parser() -> argparse.ArgumentParser:
    _, _, weights = settings()
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--prompt", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--faces", type=int, default=-1)
    result.add_argument("--attempts", type=int, default=3,
                        help="the first attempt decodes greedily; retries sample")
    result.add_argument("--weights", type=Path, default=weights)
    result.add_argument("--device", default="cuda")
    return result


def run(args, runtime: CADCoderRuntime | None = None) -> list[str]:
    import json

    if not args.prompt.strip():
        raise ValueError("CAD-Coder needs a text prompt")
    if args.faces != -1 and args.faces < 1000:
        raise ValueError("--faces must be -1 or at least 1000")
    runtime = runtime or CADCoderRuntime(args.weights, args.device)
    errors = []
    for attempt in range(max(args.attempts, 1)):
        response = runtime.complete(args.prompt, sample=attempt > 0, seed=args.seed + attempt)
        try:
            code = extract_code(response)
            vertices, triangles, labels, surface_types = run_sandboxed(code)
            mesh, labels, discarded = finish_mesh(vertices, triangles, labels, "CAD-Coder script")
            break
        except Exception as exc:
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            print(errors[-1], file=sys.stderr)
    else:
        raise RuntimeError("CAD-Coder produced no valid solid; " + "; ".join(errors))
    surface, labels = normalize(mesh, args.faces, labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    surface.export(output)
    script = output.with_suffix(output.suffix + ".cadquery.py")
    script.write_text(code + "\n", encoding="utf-8")
    face_labels = face_labels_path(output)
    write_face_labels(face_labels, labels, surface_types)
    artifacts = [str(output), str(script), str(face_labels)]
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
