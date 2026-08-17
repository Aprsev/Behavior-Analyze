#!/usr/bin/env python3
"""Manually calibrate a trained U-Net on its weakest frames, then retrain.

Pass 1 runs the model over the whole video and scores every frame by
mean |p - 0.5| (low = the model is unsure about the mouse); only frames
with a plausible mask (0.1%..20% of the frame) qualify. The ~--n-frames
least confident ones open in one correction window where all three can be
edited at once:

  * polygon      - the model's mask contour: drag vertices, click an edge
                   to add a vertex, T thins back to 8 points, R re-derives
                   it from the model;
  * HEAD         - red dot, initialised from head_method_comparison.csv
                   (silhouette method);
  * REFLECTION   - magenta dot, initialised from the same CSV.

Saved corrections:
  * polygon rows are upserted (per video+frame) into the torso labels CSV
    -- the caller then re-runs prepare + train, so the retrain learns the
    corrected masks;
  * head/reflection anchors are stored in cm in the rectified arena
    (--heads) as ground truth for the head pipeline.

Exit code 0 = at least one correction was saved (retraining is worth it).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from model import UNet
from preprocess import estimate_background, bg_centered

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "traditional" / "code"
sys.path.insert(0, str(CODE))
from mouse_behavior_pipeline import perspective_geometry, video_properties  # noqa: E402

POLY_SIMPLIFY = 8


def rot_pt(p: tuple[float, float], k: int, h: int, w: int) -> tuple[float, float]:
    x, y = p
    for _ in range(k % 4):
        x, y = y, w - 1 - x
        h, w = w, h
    return x, y


def inv_rot_pt(p: tuple[float, float], k: int, h: int, w: int) -> tuple[float, float]:
    x, y = p
    for _ in range(k % 4):
        h, w = w, h
    for _ in range(k % 4):
        x, y = h - 1 - y, x
        h, w = w, h
    return x, y


def simplify(points: np.ndarray, count: int = POLY_SIMPLIFY) -> np.ndarray:
    if len(points) <= count:
        return points.astype(float)
    return points[np.linspace(0, len(points) - 1, count, dtype=int)].astype(float)


def load_labels(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["frame", "polygon_px", "exclude", "video"])
    df = pd.read_csv(path)
    for c in ("frame", "polygon_px", "exclude", "video"):
        if c not in df:
            df[c] = None
    return df


def load_heads(path: Path) -> pd.DataFrame:
    cols = ["frame", "timestamp_sec", "head_x_cm", "head_y_cm", "reflection_x_cm",
            "reflection_y_cm", "exclude", "reflection_present", "head_present", "video"]
    if not path.is_file():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for c in cols:
        if c not in df:
            df[c] = None
    return df[cols]


class Calibrator:
    """One window editing polygon + head + reflection for the picked frames."""

    def __init__(self, args, frames: list[int], caps: dict, source: pd.DataFrame,
                 corners: np.ndarray):
        self.args, self.frames, self.caps, self.source = args, frames, caps, source
        self.cap = cv2.VideoCapture(args.video)  # own capture: draw() seeks freely
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rect_w, self.rect_h, self.forward, self.inverse, _, _ = perspective_geometry(
            corners, args.arena_width_cm, args.arena_height_cm)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.i = 0
        self.drag: tuple[str, int] | None = None
        self.dirty = False
        self.title = "Model calibration: polygon / HEAD / REFLECTION (S save, Q quit)"

    # ---- per-frame editable state ----------------------------------------
    def state(self) -> dict:
        f = self.frames[self.i]
        return self.caps.setdefault(f, {"poly": None, "head": None, "ref": None,
                                        "exclude": False, "reset": None})

    # ---- coordinate helpers ----------------------------------------------
    def cm_to_px(self, x: float, y: float) -> tuple[float, float]:
        rect = np.asarray([[[x / self.args.arena_width_cm * (self.rect_w - 1),
                             y / self.args.arena_height_cm * (self.rect_h - 1)]]], np.float32)
        return tuple(float(v) for v in cv2.perspectiveTransform(rect, self.inverse)[0, 0])

    def px_to_cm(self, p: tuple[float, float]) -> tuple[float, float]:
        rect = cv2.perspectiveTransform(np.asarray([[[p[0], p[1]]]], np.float32), self.forward)[0, 0]
        return (rect[0] * self.args.arena_width_cm / (self.rect_w - 1),
                rect[1] * self.args.arena_height_cm / (self.rect_h - 1))

    # ---- initial values per frame ----------------------------------------
    def initial_head_ref(self, frame: int) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if self.source is None or frame not in self.source.index:
            return None, None
        row = self.source.loc[frame]
        head = ref = None
        try:
            hx = float(pd.to_numeric(row.head_silhouette_x_cm, errors="coerce"))
            hy = float(pd.to_numeric(row.head_silhouette_y_cm, errors="coerce"))
            if np.isfinite(hx) and np.isfinite(hy):
                head = self.cm_to_px(hx, hy)
        except (KeyError, TypeError, ValueError):
            pass
        try:
            rx = float(pd.to_numeric(row.head_reflection_x_cm, errors="coerce"))
            ry = float(pd.to_numeric(row.head_reflection_y_cm, errors="coerce"))
            if np.isfinite(rx) and np.isfinite(ry):
                ref = self.cm_to_px(rx, ry)
        except (KeyError, TypeError, ValueError):
            pass
        return head, ref

    # ---- mouse -----------------------------------------------------------
    def _nearest(self, pts: np.ndarray, x: float, y: float) -> tuple[int, float]:
        d = np.linalg.norm(pts - np.asarray([[x, y]], np.float32), axis=1)
        idx = int(d.argmin())
        return idx, float(d[idx])

    def mouse(self, event, x, y, flags, param) -> None:
        st = self.state()
        if event == cv2.EVENT_LBUTTONDOWN:
            best, kind, idx = 1e18, None, -1
            if st["poly"] is not None and len(st["poly"]) >= 3:
                i, d = self._nearest(st["poly"], x, y)
                if d < best:
                    best, kind, idx = d, "poly", i
            for name in ("head", "ref"):
                p = st.get(name)
                if p is not None:
                    d = np.hypot(p[0] - x, p[1] - y)
                    if d < best:
                        best, kind, idx = d, name, -1
            if kind == "poly":
                self.drag = ("poly", idx)
            elif kind in ("head", "ref"):
                st[kind] = (float(x), float(y)); self.drag = (kind, -1); self.dirty = True
            else:
                # click on empty space: add a polygon vertex on the nearest edge
                poly = st["poly"]
                if poly is not None and len(poly) >= 3:
                    edges = np.roll(poly, -1, axis=0) - poly
                    q = np.asarray([[x, y]], np.float32) - poly
                    t = np.clip(np.sum(q * edges, axis=1) / np.maximum(np.sum(edges * edges, axis=1), 1e-6), 0, 1)
                    d = np.linalg.norm(q - edges * t[:, None], axis=1)
                    st["poly"] = np.insert(poly, int(d.argmin()) + 1, [x, y], axis=0)
                    self.dirty = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            kind, idx = self.drag
            if kind == "poly":
                st["poly"][idx] = [x, y]; self.dirty = True
            else:
                st[kind] = (float(x), float(y)); self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    # ---- rendering -------------------------------------------------------
    def draw(self) -> np.ndarray:
        frame = self.frames[self.i]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame}")
        st = self.state()
        if st["poly"] is not None:
            pts = st["poly"].astype(np.int32)
            cv2.polylines(img, [pts], True, (255, 255, 0), 2, cv2.LINE_AA)
            for px, py in pts:
                cv2.circle(img, (px, py), 4, (255, 255, 0), -1, cv2.LINE_AA)
        for name, color, label in (("head", (0, 0, 255), "HEAD"),
                                   ("ref", (255, 0, 255), "REFLECTION")):
            p = st.get(name)
            if p is not None:
                pt = (int(round(p[0])), int(round(p[1])))
                cv2.circle(img, pt, 7, color, -1, cv2.LINE_AA)
                cv2.putText(img, label, (pt[0] + 9, pt[1] - 9), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 47), (0, 0, 0), -1)
        flag = " [EXCLUDED]" if st["exclude"] else ""
        cv2.putText(img, f'{self.i + 1}/{len(self.frames)} frame={frame}{flag} (model uncertainty sorted)',
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .43,
                    (0, 0, 255) if st["exclude"] else (255, 255, 255), 1)
        cv2.putText(img, 'drag: poly vertex/HEAD/REF  click edge: add  T thin(8)  R reset  E excl  X/V no reflect/head',
                    (8, 39), cv2.FONT_HERSHEY_SIMPLEX, .37, (255, 255, 255), 1)
        return img

    # ---- navigation / save ----------------------------------------------
    def reset_poly(self) -> None:
        f = self.frames[self.i]
        st = self.state()
        base = self.caps.get(f, {}).get("reset")
        if base is not None:
            st["poly"] = base.copy(); self.dirty = True

    def thin(self) -> None:
        st = self.state()
        if st["poly"] is not None and len(st["poly"]) > POLY_SIMPLIFY:
            st["poly"] = simplify(st["poly"], POLY_SIMPLIFY); self.dirty = True

    def save(self) -> int:
        """Write polygon + head corrections; return number of rows saved."""
        video = Path(self.args.video).name
        rows_poly, rows_head = [], []
        for frame, st in self.caps.items():
            if st["poly"] is not None:
                rows_poly.append({"frame": frame,
                                  "polygon_px": json.dumps(st["poly"].round(1).tolist()),
                                  "exclude": bool(st["exclude"]), "video": video})
            head = st.get("head"); ref = st.get("ref")
            if head is not None or ref is not None:
                hx, hy = self.px_to_cm(head) if head is not None else (np.nan, np.nan)
                rx, ry = self.px_to_cm(ref) if ref is not None else (np.nan, np.nan)
                rows_head.append({"frame": frame, "timestamp_sec": frame / self.fps,
                                  "head_x_cm": hx, "head_y_cm": hy,
                                  "reflection_x_cm": rx, "reflection_y_cm": ry,
                                  "exclude": bool(st["exclude"]),
                                  "reflection_present": ref is not None,
                                  "head_present": head is not None, "video": video})
        n = 0
        if rows_poly:
            labels = load_labels(Path(self.args.labels))
            keep = ~((labels.video == video) & labels.frame.isin([r["frame"] for r in rows_poly]))
            out = pd.concat([labels.loc[keep], pd.DataFrame(rows_poly)], ignore_index=True)
            Path(self.args.labels).parent.mkdir(parents=True, exist_ok=True)
            out.sort_values("frame").to_csv(self.args.labels, index=False)
            n += len(rows_poly)
            print(f'Saved {len(rows_poly)} polygon corrections to {self.args.labels}')
        if rows_head:
            heads = load_heads(Path(self.args.heads))
            keep = ~((heads.video == video) & heads.frame.isin([r["frame"] for r in rows_head]))
            out = pd.concat([heads.loc[keep], pd.DataFrame(rows_head)], ignore_index=True)
            Path(self.args.heads).parent.mkdir(parents=True, exist_ok=True)
            out.sort_values("frame").to_csv(self.args.heads, index=False, float_format="%.4f")
            n += len(rows_head)
            print(f'Saved {len(rows_head)} head/reflection anchors to {self.args.heads}')
        return n

    def run(self) -> int:
        n = 0
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.mouse)
        while True:
            # X close saves too - otherwise corrections live only in memory
            # and reappear as "uncalibrated" on the next run.
            if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                n = self.save()
                print("Window closed - corrections saved.")
                break
            cv2.imshow(self.title, self.draw())
            k = cv2.waitKey(20) & 255
            if k in (27, ord("q")):
                n = self.save(); break
            if k == ord("s"):
                self.save()
            elif k == ord("t"):
                self.thin()
            elif k == ord("r"):
                self.reset_poly()
            elif k == ord("e"):
                self.state()["exclude"] = not self.state()["exclude"]
            elif k == ord("x"):
                self.state()["ref"] = None; self.dirty = True
            elif k == ord("v"):
                self.state()["head"] = None; self.dirty = True
            elif k in (81, ord("a")):
                self.i = max(0, self.i - 1)
            elif k in (83, ord("d")):
                self.i = min(len(self.frames) - 1, self.i + 1)
        self.cap.release(); cv2.destroyAllWindows()
        return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True); p.add_argument("--model", required=True)
    p.add_argument("--roi-json", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--heads", required=True)
    p.add_argument("--comparison-csv", default="")
    p.add_argument("--exclude-csv", default="")
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--arena-width-cm", type=float, default=25.0)
    p.add_argument("--arena-height-cm", type=float, default=30.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()

    excluded: set[int] = set()
    if a.exclude_csv and Path(a.exclude_csv).is_file():
        ex = pd.read_csv(a.exclude_csv)
        ex = ex.loc[ex.exclude.fillna(False).astype(bool) & (ex.video == Path(a.video).name)]
        excluded = set(int(f) for f in ex.frame)
    pack = torch.load(a.model, map_location="cpu")
    size = int(pack["size"]); dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet().to(dev); net.load_state_dict(pack["state_dict"]); net.eval()
    corners = np.asarray(json.loads(Path(a.roi_json).read_text(encoding="utf-8"))["arena_corners_px"], np.float32)
    total, fps, w, h = video_properties(Path(a.video))
    k = int(a.rotate) // 90 % 4
    bg_small = None
    if bool(pack.get("bg_subtract")):
        bg = estimate_background(a.video)
        if bg is not None:
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            if k:
                bg_gray = np.rot90(bg_gray, k)
            bg_small = cv2.resize(bg_gray, (size, size), interpolation=cv2.INTER_AREA)

    # Pass 1: score every frame by model uncertainty, keep frames whose mask
    # is plausible (a real mouse occupies 0.1%..20%). The score is the mean
    # |p - 0.5| INSIDE the predicted mask only - background pixels are always
    # confidently 0 and would drown the signal; a low score means the model
    # is hesitant about exactly the region it marks as mouse.
    scores = np.full(total, np.nan)
    cap = cv2.VideoCapture(a.video); i = 0
    print(f"Scoring {total} frames for model uncertainty ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % a.stride == 0 and i not in excluded:
            frame_in = np.rot90(frame, k) if k else frame
            gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (size, size))
            if bg_small is not None:
                small = bg_centered(small, bg_small)
            x = torch.from_numpy(small[None, None].copy()).float().to(dev) / 255
            with torch.no_grad():
                prob = torch.sigmoid(net(x))[0, 0].cpu().numpy()
            p_full = cv2.resize(prob, (frame_in.shape[1], frame_in.shape[0]))
            mask = p_full >= a.threshold
            frac = float(mask.mean())
            if 0.001 <= frac <= 0.20:
                scores[i] = float(np.abs(p_full[mask] - 0.5).mean())
        i += 1
    cap.release()
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) < 2:
        raise SystemExit("Fewer than 2 frames with a plausible mouse mask - nothing to calibrate.")
    order = valid[np.argsort(scores[valid])]
    frames = [int(f) for f in order[: a.n_frames]]
    print(f"Selected {len(frames)} least-confident frames: {frames}")

    # Initial head/reflection anchors from the comparison CSV (cm -> px).
    source = None
    if a.comparison_csv and Path(a.comparison_csv).is_file():
        df = pd.read_csv(a.comparison_csv)
        df = df.loc[df.frame.isin(frames)] if "frame" in df else df
        if len(df):
            source = df.set_index("frame")

    # Pass 2: model contour for the selected frames (source space).
    caps: dict[int, dict] = {}
    cap = cv2.VideoCapture(a.video)
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_in = np.rot90(frame, k) if k else frame
        gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (size, size))
        if bg_small is not None:
            small = bg_centered(small, bg_small)
        x = torch.from_numpy(small[None, None].copy()).float().to(dev) / 255
        with torch.no_grad():
            prob = torch.sigmoid(net(x))[0, 0].cpu().numpy()
        mask = (cv2.resize(prob, (frame_in.shape[1], frame_in.shape[0])) >= a.threshold).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        poly = None
        if cnts:
            c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
            if len(c) >= 3:
                if k:
                    c = np.asarray([inv_rot_pt(tuple(p_), k, h, w) for p_ in c], np.float32)
                poly = simplify(c, POLY_SIMPLIFY)
        caps[f] = {"poly": poly.copy() if poly is not None else None,
                   "head": None, "ref": None, "exclude": False,
                   "reset": poly.copy() if poly is not None else None}
    cap.release()
    frames = [f for f in frames if f in caps]
    cal = Calibrator(a, frames, caps, source, corners)
    for f in frames:  # initial head/reflection anchors (comparison CSV, cm -> px)
        st = caps[f]
        st["head"], st["ref"] = cal.initial_head_ref(f)
    n = cal.run()
    if n == 0:
        print("No corrections saved - retrain skipped.")
        raise SystemExit(1)
    print(f"Calibration complete ({n} rows). Re-run prepare + train to retrain.")


if __name__ == "__main__":
    main()
