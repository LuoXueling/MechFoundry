"""Optional Text2CAD text-to-CAD surface generator, run in its own environment."""

from .availability import Text2CADStatus, text2cad_status, face_labels_path

__all__ = ["Text2CADStatus", "text2cad_status", "face_labels_path"]
