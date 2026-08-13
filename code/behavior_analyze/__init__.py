"""Core modules for the mouse behaviour analysis pipeline."""

from .geometry import parse_corners, perspective_geometry
from .segmentation import Detection, segment_mouse
from .head_tracking import HeadTracker, ReflectionTracker

__all__ = ["Detection", "HeadTracker", "ReflectionTracker", "parse_corners", "perspective_geometry", "segment_mouse"]
