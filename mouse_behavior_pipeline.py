"""Backward-compatible launcher. Implementation: code/mouse_behavior_pipeline.py."""
from code.mouse_behavior_pipeline import *
from code.mouse_behavior_pipeline import parse_args, run
if __name__ == "__main__":
    try: run(parse_args())
    except Exception: raise SystemExit(1)
