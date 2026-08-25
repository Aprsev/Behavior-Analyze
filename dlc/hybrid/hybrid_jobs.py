#!/usr/bin/env python3
"""Hybrid worker with super-resolution model preflight and auto-discovery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlc.hybrid import _active_jobs_impl as _config
from dlc.hybrid import _quiet_validation_jobs as _base
from dlc.hybrid._quiet_validation_jobs import *  # noqa: E402,F401,F403
from dlc.hybrid.sr_model import resolve_super_resolution_model

SR_ACTIONS = {"prepare_hybrid", "full_hybrid", "full_hybrid_train", "full_hybrid_test"}


def _sr_action(name: str) -> Callable[[dict], None]:
    def run(cfg: dict) -> None:
        resolve_super_resolution_model(cfg, repository_root=REPOSITORY_ROOT, require=True)
        _base.ALL_JOBS[name](cfg)
    return run


ALL_JOBS = dict(_base.ALL_JOBS)
for _name in SR_ACTIONS.intersection(ALL_JOBS):
    ALL_JOBS[_name] = _sr_action(_name)

job_prepare_hybrid = ALL_JOBS["prepare_hybrid"]
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
