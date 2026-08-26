"""Visualize body-to-head orientation from a head-track trajectory CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("body_x_cm", "body_y_cm", "head_x_cm", "head_y_cm")


def add_orientation_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with raw and unwrapped body-to-head angles in degrees."""
    missing = [name for name in REQUIRED_COLUMNS if name not in data.columns]
    if missing:
        raise ValueError("trajectory CSV is missing columns: " + ", ".join(missing))
    result = data.copy()
    numeric = result.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    dx = numeric.head_x_cm - numeric.body_x_cm
    dy = numeric.head_y_cm - numeric.body_y_cm
    valid = np.isfinite(dx) & np.isfinite(dy) & (np.hypot(dx, dy) > 1e-9)
    raw = pd.Series(np.nan, index=result.index, dtype=float)
    raw.loc[valid] = np.degrees(np.arctan2(dy.loc[valid], dx.loc[valid]))
    unwrapped = pd.Series(np.nan, index=result.index, dtype=float)
    groups = (valid != valid.shift(fill_value=False)).cumsum()
    for _, indices in result.index[valid].to_series().groupby(groups[valid]):
        idx = indices.to_numpy()
        unwrapped.loc[idx] = np.degrees(np.unwrap(np.radians(raw.loc[idx].to_numpy())))
    result["head_vector_dx_cm"] = dx
    result["head_vector_dy_cm"] = dy
    result["head_angle_deg"] = raw
    result["head_angle_unwrapped_deg"] = unwrapped
    return result


def _time_axis(data: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "timestamp_sec" in data:
        values = pd.to_numeric(data.timestamp_sec, errors="coerce").to_numpy(float)
        if np.isfinite(values).any():
            return values, "Time (s)"
    if "frame" in data:
        return pd.to_numeric(data.frame, errors="coerce").to_numpy(float), "Frame"
    return np.arange(len(data), dtype=float), "Row"


def make_plots(data: pd.DataFrame, output_dir: Path, prefix: str, show: bool,
               dpi: int) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required for trajectory plots. Install it with: "
            "python -m pip install matplotlib>=3.7") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    time, xlabel = _time_axis(data)
    valid = np.isfinite(data.head_angle_deg.to_numpy(float))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                             constrained_layout=True)
    axes[0].plot(time, data.head_angle_deg, color="#155e75", linewidth=.75)
    axes[0].set_ylabel("Direction angle (deg)")
    axes[0].set_ylim(-185, 185)
    axes[0].set_yticks([-180, -90, 0, 90, 180])
    axes[0].grid(alpha=.22)
    axes[0].set_title("Body centroid to head direction")
    axes[1].plot(time, data.head_angle_unwrapped_deg, color="#9a3412", linewidth=.8)
    axes[1].set_ylabel("Unwrapped angle (deg)")
    axes[1].set_xlabel(xlabel)
    axes[1].grid(alpha=.22)
    fig.text(.995, .005, f"Valid frames: {int(valid.sum())}/{len(data)}",
             ha="right", va="bottom", fontsize=8, color="#555555")
    time_path = output_dir / f"{prefix}_head_angle_time.png"
    fig.savefig(time_path, dpi=dpi, facecolor="white")

    fig3 = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    ax = fig3.add_subplot(111, projection="3d")
    x = pd.to_numeric(data.body_x_cm, errors="coerce").to_numpy(float)
    y = pd.to_numeric(data.body_y_cm, errors="coerce").to_numpy(float)
    angle = data.head_angle_deg.to_numpy(float)
    keep = valid & np.isfinite(x) & np.isfinite(y)
    norm = colors.Normalize(-180, 180)
    points = ax.scatter(x[keep], y[keep], angle[keep], c=angle[keep], cmap="twilight",
                        norm=norm, s=7, alpha=.55, linewidths=0, rasterized=True)
    ax.set_xlabel("Body X (cm)")
    ax.set_ylabel("Body Y (cm)")
    ax.set_zlabel("Head direction angle (deg)")
    ax.set_zlim(-180, 180)
    ax.set_zticks([-180, -90, 0, 90, 180])
    ax.set_title("Head direction by body position")
    fig3.colorbar(points, ax=ax, pad=.10, shrink=.72, label="Direction angle (deg)")
    spatial_path = output_dir / f"{prefix}_head_angle_3d.png"
    fig3.savefig(spatial_path, dpi=dpi, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig); plt.close(fig3)
    return time_path, spatial_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot body-to-head angle over time and against body X/Y position.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", help="Path to head_track_trajectory.csv")
    source.add_argument("--result-dir", help="Analysis result directory containing the CSV")
    parser.add_argument("--output-dir", default="",
                        help="Plot directory (default: trajectory/result directory)")
    parser.add_argument("--prefix", default="trajectory")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true", help="Also open interactive plot windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = (Path(args.trajectory) if args.trajectory else
                Path(args.result_dir) / "head_track_trajectory.csv")
    if not csv_path.is_file():
        raise SystemExit(f"Trajectory CSV not found: {csv_path}")
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    data = add_orientation_columns(pd.read_csv(csv_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{args.prefix}_with_head_angle.csv"
    data.to_csv(table_path, index=False, float_format="%.5f")
    time_path, spatial_path = make_plots(data, output_dir, args.prefix, args.show, args.dpi)
    valid = int(data.head_angle_deg.notna().sum())
    print(f"Computed valid angles for {valid}/{len(data)} frames")
    print(f"Angle table: {table_path.resolve()}")
    print(f"Time plot:   {time_path.resolve()}")
    print(f"3D plot:     {spatial_path.resolve()}")


if __name__ == "__main__":
    main()
