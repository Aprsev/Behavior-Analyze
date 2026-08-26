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


def choose_reflection(heuristic, heuristic_confidence: float,
                      learned, learned_confidence: float,
                      agreement_px: float = 20.0) -> HeadChoice:
    """Fuse the learned reflection heatmap with the legacy bright-spot cue.

    The learned point is preferred when confident.  The heuristic remains a
    backward-compatible fallback for old checkpoints and dim/unlabelled
    appearances. Agreement raises confidence; disagreement is exposed in the
    source/diagnostics instead of being hidden.
    """
    h = np.asarray(heuristic, float) if heuristic is not None else None
    l = np.asarray(learned, float) if learned is not None else None
    h_ok = h is not None and np.isfinite(h).all()
    l_ok = l is not None and np.isfinite(l).all()
    if l_ok and h_ok:
        disagreement = float(np.linalg.norm(l - h))
        if disagreement <= agreement_px:
            weight = float(np.clip(0.55 + learned_confidence, 0.60, 0.85))
            point = weight * l + (1.0 - weight) * h
            return HeadChoice(tuple(map(float, point)),
                              float(max(learned_confidence, heuristic_confidence)),
                              "reflection_model_consensus", disagreement)
        # On a new background the learned branch can be confidently attracted
        # to the tail. A usable physical bright-spot observation is an
        # independent sensor and wins a large disagreement; confidence from
        # two spatially contradictory methods must not make the model primary.
        if heuristic_confidence >= 0.12:
            return HeadChoice(tuple(map(float, h)), float(heuristic_confidence),
                              "reflection_heuristic_rejects_model", disagreement)
        if learned_confidence >= 0.20:
            return HeadChoice(tuple(map(float, l)), float(learned_confidence),
                              "reflection_model_disagrees", disagreement)
        return HeadChoice(tuple(map(float, h)), float(heuristic_confidence),
                          "reflection_heuristic_disagrees", disagreement)
    if l_ok and learned_confidence >= 0.05:
        return HeadChoice(tuple(map(float, l)), float(learned_confidence),
                          "reflection_model", None)
    if h_ok:
        return HeadChoice(tuple(map(float, h)), float(heuristic_confidence),
                          "reflection_heuristic", None)
    return HeadChoice(None, 0.0, "reflection_missing", None)


def choose_head(reflection, reflection_confidence: float,
                learned, learned_confidence: float,
                agreement_px: float = 18.0,
                learned_threshold: float = 0.0,
                reflection_source: str | None = None) -> HeadChoice:
    """Prefer the physical reflection; learned prediction only nudges/fills.

    A learned point can move a valid reflection by at most 15%, and only when
    the two methods already agree spatially. A confidently wrong heatmap can
    therefore never replace a valid reflection.
    """
    r = np.asarray(reflection, float) if reflection is not None else None
    l = np.asarray(learned, float) if learned is not None else None
    if r is not None and np.isfinite(r).all():
        reflection_tag = (str(reflection_source).strip().lower()
                          if reflection_source else "")
        disagreement = float(np.linalg.norm(l - r)) if l is not None and np.isfinite(l).all() else None
        if disagreement is not None and disagreement <= agreement_px and learned_confidence >= learned_threshold:
            weight = min(0.15, 0.15 * learned_confidence)
            point = (1.0 - weight) * r + weight * l
            return HeadChoice(tuple(map(float, point)),
                              float(max(reflection_confidence, learned_confidence * .5)),
                              ("fused_reflection_primary_" + reflection_tag
                               if reflection_tag else "fused_reflection_primary"),
                              disagreement)
        return HeadChoice(tuple(map(float, r)), float(reflection_confidence),
                          ("reflection_" + reflection_tag
                           if reflection_tag else "reflection"),
                          disagreement)
    if l is not None and np.isfinite(l).all() and learned_confidence >= learned_threshold:
        source = ("learned_fallback" if learned_confidence >= 0.20
                  else "learned_low_confidence_fallback")
        return HeadChoice(tuple(map(float, l)), float(learned_confidence),
                          source, None)
    return HeadChoice(None, 0.0, "missing", None)


class HeadTemporalStabilizer:
    """Smooth uncertain head estimates in body-relative coordinates.

    A tethered/miniscope mouse can translate rapidly while its head-to-body
    vector changes much more gradually.  Reflection/manual observations reset
    that vector immediately. Learned fallbacks update it conservatively, and
    a truly missing candidate may reuse it only for a short bounded gap.
    """

    def __init__(self, max_gap_frames: int = 15):
        self.relative: np.ndarray | None = None
        self.gap = 0
        self.max_gap_frames = max(1, int(max_gap_frames))

    def reset(self) -> None:
        self.relative = None
        self.gap = 0

    def update(self, body, choice: HeadChoice) -> HeadChoice:
        if body is None:
            self.gap += 1
            if self.gap > self.max_gap_frames:
                self.relative = None
            return choice
        b = np.asarray(body, dtype=float)
        point = np.asarray(choice.point, dtype=float) if choice.point is not None else None
        source = str(choice.source).lower()
        trusted_reflection = (
            "reflection" in source
            and "confirmed" in source
            and "missing" not in source
            and "absent" not in source
            and "disagrees" not in source
        )
        trusted = choice.source in {"manual_override"} or trusted_reflection
        if point is not None and np.isfinite(point).all():
            observed_relative = point - b
            if trusted or self.relative is None:
                self.relative = observed_relative
            else:
                # Low peaks remain useful as a direction cue, but cannot make
                # a one-frame jump from head to tail/fibre.
                alpha = 0.50 if choice.confidence >= 0.20 else 0.22
                delta = observed_relative - self.relative
                distance = float(np.linalg.norm(delta))
                if distance > 10.0:
                    delta *= 10.0 / distance
                self.relative = self.relative + alpha * delta
                point = b + self.relative
                choice = HeadChoice(tuple(map(float, point)), choice.confidence,
                                    "temporal_" + choice.source,
                                    choice.disagreement_px)
            self.gap = 0
            return choice
        self.gap += 1
        if self.relative is not None and self.gap <= self.max_gap_frames:
            predicted = b + self.relative
            confidence = max(0.02, 0.18 * (1.0 - self.gap / (self.max_gap_frames + 1)))
            return HeadChoice(tuple(map(float, predicted)), confidence,
                              "temporal_short_gap", choice.disagreement_px)
        if self.gap > self.max_gap_frames:
            self.relative = None
        return choice
