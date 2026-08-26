"""Resolve a supervised DLC project config without silently choosing ambiguity."""

from __future__ import annotations

from pathlib import Path


def resolve_project_config(cfg: dict) -> Path:
    configured = str(cfg.get("project_config", "")).strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            cfg["project_config"] = str(path)
            return path
        raise FileNotFoundError(f"Configured DLC project file does not exist: {path}")

    root_value = str(cfg.get("working_directory", "")).strip()
    root = Path(root_value).expanduser().resolve() if root_value else None
    project = cfg.get("project", {})
    prefix = f"{project.get('task', 'mouse_occlusion')}-{project.get('experimenter', 'researcher')}-"
    candidates = []
    if root is not None and root.is_dir():
        candidates = sorted(
            (child / "config.yaml" for child in root.iterdir() if child.is_dir() and child.name.startswith(prefix)),
            key=lambda path: path.stat().st_mtime if path.is_file() else 0,
            reverse=True,
        )
        candidates = [path.resolve() for path in candidates if path.is_file()]
    if len(candidates) == 1:
        cfg["project_config"] = str(candidates[0])
        return candidates[0]
    if len(candidates) > 1:
        listed = "\n- ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Multiple matching DLC projects were found. Select the intended config.yaml "
            f"in the GUI Setup tab:\n- {listed}"
        )
    location = str(root) if root is not None else "<DLC project root is empty>"
    raise FileNotFoundError(
        "No DLC project config.yaml is selected and no unique matching project was found "
        f"under {location}. Run 'Create DLC project' first, or select an existing config.yaml."
    )
