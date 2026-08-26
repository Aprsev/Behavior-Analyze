#!/usr/bin/env python3
"""Public hybrid worker with SR, DLC, and video-integrity safeguards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _active_jobs_impl as _config
from dlc.hybrid import _hybrid_jobs_impl as _original
from dlc.hybrid import _sr_preflight_jobs as _base
from dlc.hybrid._sr_preflight_jobs import *  # noqa: E402,F401,F403
from dlc.hybrid.dlc_runtime import validate_video_adaptation
from dlc.hybrid.stable_video import prepare_hybrid_video_stable
from dlc.hybrid.tolerant_postprocess import postprocess_hybrid_predictions
from dlc.hybrid.video_preflight import validate_prepared_video

_legacy_prepare = _original.prepare_hybrid_video
_legacy_run_dlc = _original.job_run_hybrid_dlc


def _stable_prepare(cfg: dict, video: Path) -> dict[str, str]:
    return prepare_hybrid_video_stable(_legacy_prepare, cfg, video)


def _validated_run_dlc(cfg: dict) -> None:
    validate_video_adaptation(cfg)
    for value in _original.base_jobs.require_videos(cfg):
        validate_prepared_video(cfg, Path(value))
    _legacy_run_dlc(cfg)


# The train/test wrappers call these original globals at runtime, so the hooks
# cover direct actions, held-out videos, and every full-pipeline variant.
_original.prepare_hybrid_video = _stable_prepare
_original.job_run_hybrid_dlc = _validated_run_dlc
_original.postprocess_hybrid_predictions = postprocess_hybrid_predictions

DLC_ACTIONS = {"run_hybrid_dlc", "full_hybrid", "full_hybrid_train", "full_hybrid_test"}


def _dlc_action(name: str) -> Callable[[dict], None]:
    def run(cfg: dict) -> None:
        validate_video_adaptation(cfg)
        _base.ALL_JOBS[name](cfg)

    run.__name__ = getattr(_base.ALL_JOBS[name], "__name__", f"job_{name}")
    return run


ALL_JOBS = dict(_base.ALL_JOBS)
for _name in DLC_ACTIONS.intersection(ALL_JOBS):
    ALL_JOBS[_name] = _dlc_action(_name)

job_prepare_hybrid = ALL_JOBS["prepare_hybrid"]
job_run_hybrid_dlc = ALL_JOBS["run_hybrid_dlc"]
job_postprocess_hybrid = ALL_JOBS["postprocess_hybrid"]
job_full_hybrid = ALL_JOBS["full_hybrid"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(ALL_JOBS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _config.load_config(args.config)
    print(f"=== {args.action} ===", flush=True)
    ALL_JOBS[args.action](cfg)
    print(f"=== {args.action} completed ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
