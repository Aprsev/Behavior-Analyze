"""Compatibility facade for compact Ultralytics training output."""

from __future__ import annotations

from typing import Any

from dlc.hybrid import _compact_yolo_impl as _impl
from dlc.hybrid._compact_yolo_impl import *  # noqa: F401,F403

_callback_logger = _impl.add_compact_training_logger


def add_compact_training_logger(model: Any, every: int = 5) -> None:
    if not hasattr(model, "add_callback"):
        print("Compact YOLO callback is unavailable; verbose progress remains disabled.", flush=True)
        return
    _callback_logger(model, every)


# The implementation resolves this global when compact_train is called.
_impl.add_compact_training_logger = add_compact_training_logger
compact_train = _impl.compact_train
