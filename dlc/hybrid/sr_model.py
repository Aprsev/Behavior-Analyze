"""Discovery and validation of external OpenCV super-resolution weights."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

NEURAL_METHODS = {"edsr", "espcn", "fsrcnn", "lapsrn"}
MODEL_PREFIX = {
    "edsr": "EDSR", "espcn": "ESPCN", "fsrcnn": "FSRCNN", "lapsrn": "LapSRN",
}
MODEL_NAME = re.compile(r"^(edsr|espcn|fsrcnn|lapsrn)_x(\d+)\.pb$", re.IGNORECASE)


def expected_model_name(method: str, scale: int) -> str:
    key = str(method).strip().lower()
    if key not in NEURAL_METHODS:
        raise ValueError(f"{method} does not use an OpenCV .pb model")
    return f"{MODEL_PREFIX[key]}_x{int(scale)}.pb"


def _resolve_configured(value: str, config_path: Path | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and config_path is not None:
        path = config_path.resolve().parent / path
    return path.resolve()


def _case_insensitive_file(folder: Path, name: str) -> Path | None:
    if not folder.is_dir():
        return None
    expected = name.casefold()
    return next((item.resolve() for item in folder.glob("*.pb") if item.name.casefold() == expected), None)


def resolve_super_resolution_model(
    cfg: dict[str, Any],
    *,
    repository_root: Path,
    config_path: Path | None = None,
    require: bool = False,
) -> Path | None:
    """Resolve a configured model or discover its conventional project path."""

    sr = cfg.setdefault("super_resolution", {})
    method = str(sr.get("method", "bicubic")).strip().lower()
    if method not in NEURAL_METHODS:
        return None
    scale = int(sr.get("scale", 4))
    expected = expected_model_name(method, scale)
    configured = str(sr.get("model_path", "") or "").strip()
    searched: list[Path] = []
    if configured:
        selected = _resolve_configured(configured, config_path)
        searched.append(selected)
        if selected.is_file():
            match = MODEL_NAME.match(selected.name)
            if match and (match.group(1).lower() != method or int(match.group(2)) != scale):
                raise ValueError(
                    f"Selected SR model {selected.name} does not match Method={method} and Native SR scale={scale}. "
                    f"Select {expected} or change the GUI settings."
                )
            sr["model_path"] = str(selected)
            return selected

    folders = [repository_root.resolve() / "models" / "super_resolution"]
    if config_path is not None:
        folders.append(config_path.resolve().parent / "models" / "super_resolution")
    for folder in folders:
        candidate = folder / expected
        if candidate not in searched:
            searched.append(candidate)
        found = _case_insensitive_file(folder, expected)
        if found is not None:
            sr["model_path"] = str(found)
            print(f"Auto-discovered {method.upper()} model: {found}", flush=True)
            return found

    if require:
        locations = "\n- ".join(str(item) for item in searched)
        raise FileNotFoundError(
            f"{method.upper()} requires {expected}. The configured path is empty or invalid.\n"
            f"Searched:\n- {locations}\n"
            "Choose the .pb file in 'OpenCV SR model (.pb)', then save the configuration; "
            "or choose Method=bicubic for a model-free test run."
        )
    return None
