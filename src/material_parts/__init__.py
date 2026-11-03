"""Per-part material fields: a vision model gives each part its own isotropic parameters."""

from .field import field_from_parts, part_grid

__all__ = ["field_from_parts", "part_grid"]
