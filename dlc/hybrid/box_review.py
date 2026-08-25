"""Dependency-complete facade for the YOLO box review implementation."""

import math

from dlc.hybrid import _box_review_impl as _impl

# The original review implementation intentionally deferred this lightweight
# import. Keep its function globals self-contained after package modularization.
_impl.math = math

from dlc.hybrid._box_review_impl import *  # noqa: E402,F401,F403
