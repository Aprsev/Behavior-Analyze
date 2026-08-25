"""Compact, stable console reporting for Ultralytics detection training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _number(value: Any) -> str | None:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return f"{float(value):.5g}"
    except (TypeError, ValueError, RuntimeError):
        return None


def _epoch_values(trainer: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    try:
        labeled = trainer.label_loss_items(trainer.tloss, prefix="train")
        if isinstance(labeled, Mapping):
            values.update(labeled)
    except Exception:
        names = tuple(getattr(trainer, "loss_names", ()))
        losses = getattr(trainer, "tloss", ())
        try:
            values.update({f"train/{name}": losses[index] for index, name in enumerate(names)})
        except Exception:
            pass
    metrics = getattr(trainer, "metrics", {})
    if isinstance(metrics, Mapping):
        values.update(metrics)
    return values


def add_compact_training_logger(model: Any, every: int = 5) -> None:
    """Print one train/validation summary every N completed epochs."""

    interval = max(1, int(every))
    state = {"last_epoch": 0}

    def report(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        args = getattr(trainer, "args", None)
        total = int(getattr(trainer, "epochs", 0) or getattr(args, "epochs", 0) or epoch)
        if epoch == state["last_epoch"]:
            return
        if epoch % interval and epoch != total:
            return
        state["last_epoch"] = epoch
        values = _epoch_values(trainer)
        preferred = (
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        )
        parts = []
        for name in preferred:
            if name not in values:
                continue
            rendered = _number(values[name])
            if rendered is not None:
                parts.append(f"{name.split('/')[-1]}={rendered}")
        suffix = " | " + " | ".join(parts) if parts else ""
        print(f"YOLO epoch {epoch}/{total}{suffix}", flush=True)

    model.add_callback("on_fit_epoch_end", report)
    print(f"Compact YOLO log enabled: one summary every {interval} epochs.", flush=True)


def compact_train(model: Any, kwargs: dict[str, Any], every: int = 5) -> Any:
    """Train with Ultralytics progress bars disabled and compact callbacks enabled."""

    add_compact_training_logger(model, every)
    clean_kwargs = dict(kwargs)
    clean_kwargs["verbose"] = False
    return model.train(**clean_kwargs)
