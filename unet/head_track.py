#!/usr/bin/env python3
"""Combine the U-Net mask with the miniscope reflection to track body + head.

For every frame:
  - the U-Net supplies a clean mouse+miniscope mask (fibre/tail excluded);
  - the body is the mask centroid (rectified arena coordinates -> cm);
  - new checkpoints predict a learned head heatmap inside the clean mask;
    old checkpoints fall back to the reflection tracker;
  - excluded frames (--exclude-csv) become NaN rows and are marked in the
    overlay video.

Outputs: head_track_trajectory.csv, head_track_overlay.mp4,
mouse_miniscope_mask.mp4, head_track_metadata.json.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "traditional" / "code"
sys.path.insert(0, str(CODE))
from calibrate import detect_floor_bounds  # noqa: E402
from compare_head_methods import ReflectionTracker  # noqa: E402
from mouse_behavior_pipeline import (  # noqa: E402
    perspective_geometry, rectified_to_cm, robust_threshold, sample_frames, video_properties,
)
from model import checkpoint_model, unpack_outputs  # noqa: E402
from anatomical_head import (AnatomicalHeadConstraint, appendage_foreground,
                             clamp_choice_to_mask)  # noqa: E402
from head_fusion import (HeadChoice, HeadTemporalStabilizer, choose_head,
                         choose_reflection)  # noqa: E402
from label_compat import as_bool, video_mask  # noqa: E402
from preprocess import estimate_background  # noqa: E402
from postprocess import TemporalMaskFilter, model_input  # noqa: E402


def rot_pt(p: tuple[float, float], k: int, h: int, w: int) -> tuple[float, float]:
    """(col,row) in an (h,w) frame -> (col,row) in np.rot90(frame, k)."""
    x, y = p
    for _ in range(k % 4):
        x, y = y, w - 1 - x
        h, w = w, h
    return x, y


def inv_rot_pt(p: tuple[float, float], k: int, h: int, w: int) -> tuple[float, float]:
    """inverse of rot_pt (p is in the rotated frame)."""
    x, y = p
    for _ in range(k % 4):
        h, w = w, h
    for _ in range(k % 4):
        x, y = h - 1 - y, x
        h, w = w, h
    return x, y


def load_corners(roi_json: Path) -> np.ndarray:
    data = json.loads(Path(roi_json).read_text(encoding="utf-8"))
    return np.asarray(data["arena_corners_px"], np.float32)


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask), None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8) * 255, (float(cent[label][0]), float(cent[label][1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True); p.add_argument("--model", required=True)
    p.add_argument("--roi-json", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--arena-width-cm", type=float, default=25.0)
    p.add_argument("--arena-height-cm", type=float, default=30.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fibre-opening", type=int, default=5)
    p.add_argument("--reacquire-sec", type=float, default=.35)
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="arena turned vs training: rotate frames before the CNN "
                        "(ROI corners are rotated too; output stays in source space)")
    p.add_argument("--exclude-csv", default="")
    p.add_argument("--head-labels", default="",
                   help="manual head/reflection CSV; exact frames override and reseed tracking")
    a = p.parse_args()

    excluded: set[int] = set()
    if a.exclude_csv and Path(a.exclude_csv).is_file():
        ex = pd.read_csv(a.exclude_csv)
        matches = video_mask(ex.video, Path(a.video).name) if "video" in ex else pd.Series(True, index=ex.index)
        ex = ex.loc[ex.exclude.map(as_bool) & matches]
        excluded = set(int(f) for f in ex.frame)
        print(f"Marking {len(excluded)} screened frames as excluded")

    pack = torch.load(a.model, map_location="cpu")
    size = int(pack["size"])
    in_channels = int(pack.get("in_channels", 2 if pack.get("dual_channel") else 1))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = checkpoint_model(pack, dev); net.eval()
    reflection_validation_error = pack.get("reflection_error_px")
    learned_reflection_enabled = bool(pack.get("reflection_output", False)) and \
        reflection_validation_error is not None and np.isfinite(reflection_validation_error) and \
        float(reflection_validation_error) <= 18.0
    if bool(pack.get("reflection_output", False)):
        if learned_reflection_enabled:
            print(f"Learned reflection branch enabled (validation error "
                  f"{float(reflection_validation_error):.2f} px)")
        else:
            print("WARNING: reflection branch lacks acceptable validation accuracy; "
                  "using legacy bright-spot fallback")
    else:
        print("Legacy checkpoint without reflection branch; using bright-spot tracker")

    corners = load_corners(Path(a.roi_json))
    cap = cv2.VideoCapture(a.video)
    w, h = int(cap.get(3)), int(cap.get(4))
    k = int(a.rotate) // 90 % 4
    if k:
        # Arena turned by a quarter-turn: rotate the frame for the CNN and
        # rotate the ROI corners the same way so perspective stays aligned.
        corners = np.asarray([rot_pt(tuple(c), k, h, w) for c in corners], np.float32)

    # Background-invariant CNN input, identical to train.py: active only
    # when the model was trained with it (old checkpoints keep raw frames).
    # This is a source-space background (the CNN never sees the rectified
    # one below, which serves the reflection tracker).
    bg_small = None
    if bool(pack.get("bg_subtract")) or in_channels > 1:
        bg = estimate_background(Path(a.video))
        if bg is not None:
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            if k:
                bg_gray = np.rot90(bg_gray, k)  # same rotation as the frames
            bg_small = cv2.resize(bg_gray, (size, size), interpolation=cv2.INTER_AREA)
            print(f"Model trained with background subtraction - applying "
                  f"background-centered input (gain {pack.get('bg_gain', 2.0):.1f})")
        else:
            print("WARNING: background estimation failed; using raw frames")
    total, fps, _, _ = video_properties(Path(a.video))
    rw, rh, forward, inverse, _, _ = perspective_geometry(corners, a.arena_width_cm, a.arena_height_cm)
    _, samples = sample_frames(Path(a.video), total, 61)
    rect_samples = np.stack([cv2.warpPerspective(np.rot90(f, k) if k else f, forward, (rw, rh)) for f in samples])
    background = np.percentile(rect_samples, 85, axis=0).astype(np.uint8)
    robust_threshold(rect_samples, background, 0)

    # Wall-band exclusion: the ROI may include the walls (the camera sees
    # the mouse's shadow projected on them). Wall content changes with the
    # mouse position, so everything outside the detected floor rectangle is
    # zeroed in the mask and can never influence body/head positions.
    floor = detect_floor_bounds(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
    wall_mask = None
    if floor is not None:
        x0, y0, x1, y1 = floor
        wall_mask = np.ones((rh, rw), np.uint8) * 255
        wall_mask[y0:y1, x0:x1] = 0
        share = 100.0 * (x1 - x0) * (y1 - y0) / (rw * rh)
        print(f"Floor detected {floor} ({share:.0f}% of rectified arena); "
              f"wall band excluded from segmentation")
    else:
        print("No distinct wall band detected; whole rectified arena is floor")

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ow = cv2.VideoWriter(str(out / "head_track_overlay.mp4"), fourcc, fps, (w, h))
    mw = cv2.VideoWriter(str(out / "mouse_miniscope_mask.mp4"), fourcc, fps, (w, h))
    tracker = ReflectionTracker(fps)
    head_temporal = HeadTemporalStabilizer(max_gap_frames=max(5, round(fps * 0.5)))
    head_anatomy = AnatomicalHeadConstraint()
    temporal = TemporalMaskFilter(fps=fps, opening_px=a.fibre_opening, hold_frames=3,
                                  reacquire_frames=max(8, round(fps * a.reacquire_sec)))
    manual_heads = pd.DataFrame()
    if a.head_labels and Path(a.head_labels).is_file():
        manual_heads = pd.read_csv(a.head_labels)
        if "video" in manual_heads:
            manual_heads = manual_heads.loc[video_mask(manual_heads.video, Path(a.video).name)]
        if "exclude" in manual_heads:
            manual_heads = manual_heads.loc[~manual_heads.exclude.map(as_bool)]
        if len(manual_heads):
            manual_heads = manual_heads.drop_duplicates("frame", keep="last").set_index("frame")
    rows = []; i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k:
            frame = np.rot90(frame, k)  # everything below works in rotated space
        reflection = learned_head = learned_reflection = None
        reflection_source = "excluded" if i in excluded else "reflection_missing"
        reflection_disagreement = np.nan
        reflection_conf = learned_reflection_conf = 0.0
        head_source = "excluded" if i in excluded else "missing"
        disagreement = np.nan
        anatomy_corrected = False; body_elongation = np.nan
        tail_hint_detected = False; head_outside_distance = np.nan
        if i in excluded:
            body = head = None; conf = 0.0; overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), (0, 0, 220), 6)
            cv2.putText(overlay, f"EXCLUDED frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)
            mask_src = None
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (size, size))
            channels = model_input(small, bg_small, in_channels, float(pack.get("bg_gain", 2.0)))
            x = torch.from_numpy(channels[None].copy()).float().to(dev) / 255
            with torch.no_grad():
                output = net(x)
                mask_logits, head_logits, reflection_logits = unpack_outputs(output)
                prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
                head_prob = torch.sigmoid(head_logits)[0, 0].cpu().numpy() if head_logits is not None else None
                reflection_prob = (torch.sigmoid(reflection_logits)[0, 0].cpu().numpy()
                                   if reflection_logits is not None and learned_reflection_enabled else None)
            filtered = temporal.update(prob, a.threshold)
            if filtered.status == "acquired" and i > 0:
                # A released body prior means continuity across the gap is no
                # longer valid for head/tail orientation either.
                head_anatomy.reset()
                head_temporal.reset()
            if filtered.centroid is not None:
                mask_src = cv2.resize(filtered.mask, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                body_src = (filtered.centroid[0] * frame.shape[1] / size,
                            filtered.centroid[1] * frame.shape[0] / size)
            else:
                mask_src = np.zeros(frame.shape[:2], np.uint8); body_src = None
            body = None; head = None; conf = 0.0
            if body_src is not None:
                body_mask = cv2.warpPerspective(mask_src, forward, (rw, rh), flags=cv2.INTER_NEAREST)
                if wall_mask is not None:
                    body_mask[wall_mask > 0] = 0  # wall projections excluded
                # body = centroid of the wall-excluded mask (rectified space)
                m = cv2.moments(body_mask)
                if m["m00"] > 0:
                    body = (m["m10"] / m["m00"], m["m01"] / m["m00"])
                    rect = cv2.warpPerspective(frame, forward, (rw, rh))
                    # New checkpoints learn a head heatmap from the existing
                    # manual red-point labels. Restrict its maximum to the
                    # clean body mask so a highlight on the fibre
                    # cannot become the head. Old checkpoints fall back to
                    # the reflection tracker.
                    learned_conf = 0.0
                    # Network candidates must start inside the clean green
                    # mask. A second rectified-space clamp is applied after
                    # fusion and temporal stabilization.
                    allowed = filtered.mask > 0
                    if head_prob is not None:
                        constrained = np.where(allowed, head_prob, 0.0)
                        learned_conf = float(constrained.max())
                        # Always retain the in-mask argmax as a candidate.
                        # The heatmap loss often produces a useful location
                        # with a peak below 0.20; confidence is preserved for
                        # QA and temporal weighting instead of deleting it.
                        if np.isfinite(learned_conf) and allowed.any():
                            yy, xx = np.unravel_index(int(constrained.argmax()), constrained.shape)
                            src_head = np.asarray([[[xx * frame.shape[1] / size,
                                                    yy * frame.shape[0] / size]]], np.float32)
                            hp = cv2.perspectiveTransform(src_head, forward)[0, 0]
                            learned_head = tuple(float(v) for v in hp)
                    if reflection_prob is not None:
                        constrained_reflection = np.where(allowed, reflection_prob, 0.0)
                        learned_reflection_conf = float(constrained_reflection.max())
                        if np.isfinite(learned_reflection_conf) and allowed.any():
                            ryy, rxx = np.unravel_index(
                                int(constrained_reflection.argmax()), constrained_reflection.shape)
                            src_reflection = np.asarray(
                                [[[rxx * frame.shape[1] / size,
                                   ryy * frame.shape[0] / size]]], np.float32)
                            rp = cv2.perspectiveTransform(src_reflection, forward)[0, 0]
                            learned_reflection = tuple(float(v) for v in rp)
                    heuristic_reflection, heuristic_conf, _ = tracker.update(
                        rect, background, body, body_mask)
                    reflection_choice = choose_reflection(
                        heuristic_reflection, heuristic_conf,
                        learned_reflection, learned_reflection_conf)
                    reflection = reflection_choice.point
                    reflection_conf = reflection_choice.confidence
                    reflection_source = reflection_choice.source
                    reflection_disagreement = (reflection_choice.disagreement_px
                                               if reflection_choice.disagreement_px is not None else np.nan)
                    if reflection is not None and reflection_source.startswith("reflection_model"):
                        tracker.position = np.asarray(reflection, np.float32)
                        tracker.relative = tracker.position - np.asarray(body, np.float32)
                    manual = manual_heads.loc[i] if i in manual_heads.index else None
                    manual_head_absent = False
                    if manual is not None:
                        rx = pd.to_numeric(manual.get("reflection_x_cm"), errors="coerce")
                        ry = pd.to_numeric(manual.get("reflection_y_cm"), errors="coerce")
                        hx = pd.to_numeric(manual.get("head_x_cm"), errors="coerce")
                        hy = pd.to_numeric(manual.get("head_y_cm"), errors="coerce")
                        # Legacy rows without explicit verification are useful
                        # for audit/retraining migration, but must never bypass
                        # current anatomical inference as exact frame truth.
                        head_trusted = as_bool(manual.get("head_verified", False))
                        reflection_verified = as_bool(manual.get("reflection_verified", False))
                        reflection_trusted = reflection_verified
                        if reflection_trusted and as_bool(manual.get("reflection_present", True)):
                            if np.isfinite([rx, ry]).all():
                                reflection = (float(rx) / a.arena_width_cm * (rw - 1),
                                              float(ry) / a.arena_height_cm * (rh - 1))
                                tracker.position = np.asarray(reflection, np.float32)
                                tracker.relative = tracker.position - np.asarray(body, np.float32)
                                reflection_conf = 1.0
                                reflection_source = "manual_reflection"
                        elif reflection_verified:
                            reflection = None; reflection_conf = 0.0
                            reflection_source = "manual_reflection_absent"
                        if head_trusted and as_bool(manual.get("head_present", True)):
                            if np.isfinite([hx, hy]).all():
                                head = (float(hx) / a.arena_width_cm * (rw - 1),
                                        float(hy) / a.arena_height_cm * (rh - 1))
                                manual_choice = clamp_choice_to_mask(
                                    HeadChoice(head, 1.0, "manual_override", None), body_mask)
                                head, conf, head_source = (manual_choice.point,
                                                           manual_choice.confidence,
                                                           manual_choice.source)
                        elif head_trusted:
                            manual_head_absent = True
                            head = None; conf = 0.0; head_source = "manual_head_absent"
                    if head is None and not manual_head_absent:
                        choice = choose_head(reflection, reflection_conf,
                                             learned_head, learned_conf)
                        # Inspect both the raw pre-opening CNN foreground and
                        # dark static-background residual. Opening removes the
                        # thin tail and fibre from the green mask; here they are
                        # restored only as anatomical side cues.
                        raw_model_src = cv2.resize(
                            (prob >= a.threshold).astype(np.uint8) * 255,
                            (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        raw_model_rect = cv2.warpPerspective(
                            raw_model_src, forward, (rw, rh), flags=cv2.INTER_NEAREST)
                        if wall_mask is not None:
                            raw_model_rect[wall_mask > 0] = 0
                        foreground = appendage_foreground(
                            rect, background, body_mask, raw_model_rect)
                        anatomy = head_anatomy.update(body, body_mask, foreground, choice)
                        choice = anatomy.choice
                        anatomy_corrected = anatomy.corrected
                        body_elongation = anatomy.elongation
                        tail_hint_detected = anatomy.tail_detected
                        head_outside_distance = anatomy.outside_distance_px
                        choice = head_temporal.update(body, choice)
                        choice = clamp_choice_to_mask(choice, body_mask)
                        head, conf, head_source = choice.point, choice.confidence, choice.source
                        disagreement = choice.disagreement_px if choice.disagreement_px is not None else np.nan
                    else:
                        choice = head_temporal.update(
                            body, HeadChoice(head, conf, head_source, None))
                        head, conf, head_source = choice.point, choice.confidence, choice.source
                else:
                    body = None; head = None; conf = 0.0
            overlay = frame.copy()
            overlay[mask_src > 0] = (0, 220, 0)
            overlay = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
            if wall_mask is not None:
                # floor rectangle outline (rectified -> source space)
                fpts = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
                src_pts = cv2.perspectiveTransform(fpts[None], inverse)[0]
                cv2.polylines(overlay, [src_pts.astype(np.int32)], True, (255, 128, 0), 2, cv2.LINE_AA)
            if body is not None:
                b_px = cv2.perspectiveTransform(np.asarray([[body]], np.float32), inverse)[0, 0].astype(int)
                cv2.circle(overlay, tuple(b_px), 6, (0, 0, 255), -1, cv2.LINE_AA)
                if head is not None:
                    h_px = cv2.perspectiveTransform(np.asarray([[head]], np.float32), inverse)[0, 0].astype(int)
                    cv2.circle(overlay, tuple(h_px), 7, (0, 220, 255), -1, cv2.LINE_AA)
                    cv2.line(overlay, tuple(b_px), tuple(h_px), (255, 255, 255), 1, cv2.LINE_AA)
                if reflection is not None:
                    r_px = cv2.perspectiveTransform(np.asarray([[reflection]], np.float32), inverse)[0, 0].astype(int)
                    cv2.circle(overlay, tuple(r_px), 4, (255, 0, 255), 1, cv2.LINE_AA)
                if learned_reflection is not None:
                    lr_px = cv2.perspectiveTransform(
                        np.asarray([[learned_reflection]], np.float32), inverse)[0, 0].astype(int)
                    cv2.drawMarker(overlay, tuple(lr_px), (255, 255, 0),
                                   cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
                cv2.putText(overlay, f"R {reflection_source} {reflection_conf:.2f}",
                            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .48,
                            (255, 255, 255), 1, cv2.LINE_AA)
                if np.isfinite(body_elongation):
                    anatomy_text = (f"A {'FIX' if anatomy_corrected else 'KEEP'} "
                                    f"elong={body_elongation:.2f} tail={int(tail_hint_detected)}")
                    cv2.putText(overlay, anatomy_text, (12, 46),
                                cv2.FONT_HERSHEY_SIMPLEX, .43,
                                (0, 220, 255) if anatomy_corrected else (255, 255, 255),
                                1, cv2.LINE_AA)
                cv2.putText(overlay, f"H {head_source}", (12, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 220, 255),
                            1, cv2.LINE_AA)
        overlay_write = np.rot90(overlay, -k) if k else overlay
        if mask_src is None:
            mask_write = np.zeros((h, w), np.uint8)
        else:
            mask_write = np.rot90(mask_src, -k) if k else mask_src
        ow.write(overlay_write)
        mw.write(cv2.cvtColor(mask_write, cv2.COLOR_GRAY2BGR))
        body_cm = rectified_to_cm(body, rw, rh, a.arena_width_cm, a.arena_height_cm) if body else (float("nan"), float("nan"))
        head_cm = rectified_to_cm(head, rw, rh, a.arena_width_cm, a.arena_height_cm) if head else (float("nan"), float("nan"))
        reflection_cm = rectified_to_cm(reflection, rw, rh, a.arena_width_cm, a.arena_height_cm) if reflection else (float("nan"), float("nan"))
        learned_cm = rectified_to_cm(learned_head, rw, rh, a.arena_width_cm, a.arena_height_cm) if learned_head else (float("nan"), float("nan"))
        learned_reflection_cm = rectified_to_cm(
            learned_reflection, rw, rh, a.arena_width_cm, a.arena_height_cm
        ) if learned_reflection else (float("nan"), float("nan"))
        rows.append((i, i / fps, body_cm[0], body_cm[1], head_cm[0], head_cm[1], conf,
                     reflection_cm[0], reflection_cm[1], learned_cm[0], learned_cm[1],
                     learned_reflection_cm[0], learned_reflection_cm[1],
                     learned_reflection_conf, reflection_source, reflection_disagreement,
                     head_source, disagreement, anatomy_corrected, body_elongation,
                     tail_hint_detected, head_outside_distance))
        i += 1
    cap.release(); ow.release(); mw.release()
    df = pd.DataFrame(rows, columns=["frame", "timestamp_sec", "body_x_cm", "body_y_cm",
                                     "head_x_cm", "head_y_cm", "head_confidence",
                                     "reflection_x_cm", "reflection_y_cm",
                                     "learned_head_x_cm", "learned_head_y_cm",
                                     "learned_reflection_x_cm", "learned_reflection_y_cm",
                                     "learned_reflection_confidence", "reflection_source",
                                     "reflection_disagreement_px",
                                     "head_source", "head_disagreement_px",
                                     "head_anatomy_corrected", "body_elongation",
                                     "tail_hint_detected", "head_outside_distance_px"])
    df.to_csv(out / "head_track_trajectory.csv", index=False, float_format="%.5f")
    (out / "head_track_metadata.json").write_text(json.dumps(
        {"input": str(Path(a.video).resolve()), "model": str(Path(a.model).resolve()),
         "device": dev, "frames": i, "threshold": a.threshold, "rotate": a.rotate,
         "bg_subtract": bg_small is not None, "in_channels": in_channels,
         "head_policy": "reflection_primary_anatomical_temporal_fallback",
         "anatomical_head_constraint": {
             "mask_containment": True, "major_axis_endpoints": True,
             "contained_tail_vs_boundary_fibre": True, "flip_confirm_frames": 4},
         "head_source_counts": {str(k): int(v) for k, v in df.head_source.value_counts().items()},
         "reflection_policy": ("learned_primary_heuristic_fallback"
                               if learned_reflection_enabled
                               else "legacy_heuristic"),
         "reflection_model_validation_error_px": reflection_validation_error,
         "reflection_source_counts": {str(k): int(v) for k, v in df.reflection_source.value_counts().items()},
         "fibre_aware_temporal_filter": True,
         "fibre_opening": a.fibre_opening, "reacquire_sec": a.reacquire_sec,
         "floor_bounds": list(floor) if floor is not None else None, "model_size": size,
         "body_valid_percent": round(100 * float(df.body_x_cm.notna().mean()), 2),
         "reflection_valid_percent": round(100 * float(df.reflection_x_cm.notna().mean()), 2),
         "learned_reflection_candidate_percent": round(
             100 * float(df.learned_reflection_x_cm.notna().mean()), 2),
         "learned_candidate_percent": round(100 * float(df.learned_head_x_cm.notna().mean()), 2),
         "head_valid_percent": round(100 * float(df.head_x_cm.notna().mean()), 2),
         "head_anatomy_corrected_percent": round(
             100 * float(df.head_anatomy_corrected.map(as_bool).mean()), 2),
         "tail_hint_detected_percent": round(
             100 * float(df.tail_hint_detected.map(as_bool).mean()), 2),
         "head_reliable_percent": round(100 * float((df.head_x_cm.notna() &
                                                       (df.head_confidence >= 0.20)).mean()), 2),
         "excluded_frames": sorted(excluded & set(range(i)))}, indent=2), encoding="utf-8")
    body_rate = df.body_x_cm.notna().mean()
    reflection_rate = df.reflection_x_cm.notna().mean()
    learned_rate = df.learned_head_x_cm.notna().mean()
    learned_reflection_rate = df.learned_reflection_x_cm.notna().mean()
    head_rate = df.head_x_cm.notna().mean()
    reliable_rate = (df.head_x_cm.notna() & (df.head_confidence >= 0.20)).mean()
    print(f"Wrote {i} frames to {out}; body {body_rate:.1%}; "
          f"reflection {reflection_rate:.1%}; learned reflection {learned_reflection_rate:.1%}; "
          f"learned head {learned_rate:.1%}; "
          f"head valid {head_rate:.1%}; reliable {reliable_rate:.1%}")


if __name__ == "__main__":
    main()
