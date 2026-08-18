"""Reflection-first head selection with a conservative learned fallback."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HeadChoice:
    point: tuple[float, float] | None
    confidence: float
    source: str
    disagreement_px: float | None


def choose_head(reflection, reflection_confidence: float,
                learned, learned_confidence: float,
                agreement_px: float = 18.0,
                learned_threshold: float = 0.20) -> HeadChoice:
    """Prefer the physical reflection; learned prediction only nudges/fills.

    A learned point can move a valid reflection by at most 15%, and only when
    the two methods already agree spatially. A confidently wrong heatmap can
    therefore never replace a valid reflection.
    """
    r = np.asarray(reflection, float) if reflection is not None else None
    l = np.asarray(learned, float) if learned is not None else None
    if r is not None and np.isfinite(r).all():
        disagreement = float(np.linalg.norm(l - r)) if l is not None and np.isfinite(l).all() else None
        if disagreement is not None and disagreement <= agreement_px and learned_confidence >= learned_threshold:
            weight = min(0.15, 0.15 * learned_confidence)
            point = (1.0 - weight) * r + weight * l
            return HeadChoice(tuple(map(float, point)),
                              float(max(reflection_confidence, learned_confidence * .5)),
                              "fused_reflection_primary", disagreement)
        return HeadChoice(tuple(map(float, r)), float(reflection_confidence),
                          "reflection", disagreement)
    if l is not None and np.isfinite(l).all() and learned_confidence >= learned_threshold:
        return HeadChoice(tuple(map(float, l)), float(learned_confidence),
                          "learned_fallback", None)
    return HeadChoice(None, 0.0, "missing", None)
