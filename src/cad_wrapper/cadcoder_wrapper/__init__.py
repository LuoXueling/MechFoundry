"""Optional CAD-Coder text-to-CAD surface generator, run in its own environment."""

from .availability import CADCoderStatus, cadcoder_status

__all__ = ["CADCoderStatus", "cadcoder_status"]
