"""Package-aware facade for the hybrid pipeline implementation."""

from pathlib import Path

from dlc.hybrid import _hybrid_pipeline_impl as _impl

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_impl.ROOT = REPOSITORY_ROOT
_impl.TRADITIONAL_CODE = REPOSITORY_ROOT / "traditional" / "code"

from dlc.hybrid._hybrid_pipeline_impl import *  # noqa: E402,F401,F403
