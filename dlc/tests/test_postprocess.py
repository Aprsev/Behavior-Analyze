"""Dependency-aware discovery wrapper for post-processing test cases."""

from importlib.util import find_spec
import unittest

REQUIRED = ("cv2", "numpy", "pandas")
missing = [name for name in REQUIRED if find_spec(name) is None]
if missing:
    raise unittest.SkipTest("optional scientific dependencies missing: " + ", ".join(missing))

from dlc.tests._postprocess_cases import *  # noqa: E402,F401,F403
