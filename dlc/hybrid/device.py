"""Device selection helpers for Ultralytics jobs.

Ultralytics does not accept ``device="auto"`` in all releases.  The GUI keeps
that convenient, portable value and this module resolves it immediately before
a worker job starts.
"""

from __future__ import annotations

from typing import Any, MutableMapping


def resolve_ultralytics_device(value: Any, torch_module: Any = None) -> Any:
    """Return an Ultralytics-compatible device value.

    Explicit user selections are preserved.  ``auto`` chooses CUDA GPU 0,
    Apple MPS, or CPU in that order.  Numeric single-GPU strings are returned
    as integers, while multi-GPU selections such as ``0,1`` remain strings.
    """

    text = "auto" if value is None else str(value).strip()
    if text and text.lower() != "auto":
        return int(text) if text.isdigit() else text

    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:
            return "cpu"

    try:
        if torch_module.cuda.is_available() and torch_module.cuda.device_count() > 0:
            return 0
    except Exception:
        pass

    try:
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def normalize_yolo_device(config: MutableMapping[str, Any], torch_module: Any = None) -> Any:
    """Resolve ``config['yolo']['device']`` in place and return the result."""

    yolo = config.setdefault("yolo", {})
    requested = yolo.get("device", "auto")
    resolved = resolve_ultralytics_device(requested, torch_module=torch_module)
    yolo["device"] = resolved
    if str(requested).strip().lower() in {"", "auto", "none"}:
        label = f"CUDA GPU {resolved}" if isinstance(resolved, int) else str(resolved).upper()
        print(f"Ultralytics device: auto -> {resolved!r} ({label})")
    return resolved
