"""Public active-learning API with localized pandas compatibility handling."""

from __future__ import annotations

import warnings

from dlc.hybrid import _active_learning_impl as _impl
from dlc.hybrid._active_learning_impl import *  # noqa: F401,F403


def _call_without_concat_future_warning(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        return function(*args, **kwargs)


def import_labeled_dataset(cfg):
    return _call_without_concat_future_warning(_impl.import_labeled_dataset, cfg)


def mine_low_confidence_frames(cfg):
    return _call_without_concat_future_warning(_impl.mine_low_confidence_frames, cfg)
