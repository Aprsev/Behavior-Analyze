# Behavior Analyze

`Behavior Analyze` is an automated computer-vision pipeline for tracking one mouse in a static-camera rectangular open-field arena. It detects the arena, removes perspective distortion, constructs a static background model, segments the mouse, estimates body-centroid and head positions, converts positions to centimetres, and writes both numerical trajectories and an annotated video. Explicit input paths support MP4, AVI, MOV, MKV, M4V, and WMV files.

The repository now contains two independent paths: `traditional/` keeps the
classical OpenCV implementation, while `unet/` is the GPU-oriented,
fibre-aware segmentation and learned-head pipeline recommended for recordings
with a tethered miniscope.

## Scope and assumptions

The method is designed for recordings with:

- one mouse in one rectangular arena;
- a fixed overhead or near-overhead camera;
- an arena visible throughout the recording;
- a static background and illumination;
- enough contrast between the mouse and arena floor;
- a known physical arena width and height, 25 x 30 cm by default; and
- an MP4 input readable by the local OpenCV/FFmpeg installation.

The camera background can be static even though raw frame differences are not zero: the mouse moves, video compression introduces small changes, and shadows may vary. The pipeline therefore reports robust difference statistics rather than treating the maximum difference over the whole image as a camera-stability test.

## Installation

Python 3.10 or newer is recommended. From Git Bash on Windows:

```bash
cd /d/Desktop/HKU/behavior_analyze
py -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

From PowerShell, activate the same environment with:

```powershell
Set-Location "D:\Desktop\HKU\behavior_analyze"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The environment and generated analysis artifacts are excluded by `.gitignore`.

## Quick start

The default input path in `mouse_behavior_pipeline.py` is intentionally empty. Place exactly one source MP4 in the project directory and run:

```bash
python mouse_behavior_pipeline.py
```

If there are multiple MP4 files, specify the source explicitly:

```bash
python mouse_behavior_pipeline.py --input recording.mp4
```

To write results to another directory:

```bash
python mouse_behavior_pipeline.py \
  --input recording.mp4 \
  --output-dir results/session_01
```

With the default output directory, `annotated_output.mp4` is ignored during automatic input discovery, so a previous result is not accidentally processed as source data.

## Command-line options

| Option | Default | Purpose |
| --- | ---: | --- |
| `--input PATH` | blank | Source MP4. A blank value auto-selects the sole MP4 in the current directory. |
| `--output-dir DIR` | `.` | Directory for all output artifacts. |
| `--arena-width-cm N` | `25` | Physical arena width in centimetres. |
| `--arena-height-cm N` | `30` | Physical arena height in centimetres. |
| `--threshold N` | `0` | Fixed foreground difference threshold. Zero enables robust automatic estimation. |
| `--corners POINTS` | blank | Manual arena corners formatted as `x,y;x,y;x,y;x,y`. |
| `--start-sec N` | `0` | Start tracking at N seconds while retaining earlier CSV rows as `NaN`. |
| `--max-frames N` | `0` | Process only the first N frames for debugging. Zero processes the full video. |
| `--no-contour` | off | Suppress the mouse-contour overlay in the annotated video. |

Display the authoritative CLI help with:

```bash
python mouse_behavior_pipeline.py --help
```

## Technical method

### 1. Video ingestion and sampling

OpenCV reads the frame count, frame rate, encoded width, and encoded height. Invalid or unavailable metadata causes an immediate failure rather than producing incorrectly timed trajectories.

Up to 31 frame indices are distributed uniformly from the first to the last frame. The decoded sample frames are stacked, and their per-pixel median forms a representative calibration image. Uniform temporal sampling reduces the probability that the mouse occupies the same arena location in most samples.

The log records absolute-difference and temporal-range statistics from these samples. Large raw maxima are expected at mouse locations. A low median and low background quantiles are more informative indicators of a stable camera and floor.

### 2. Arena detection and perspective calibration

The median image is converted to grayscale and Gaussian-smoothed. The automatic detector then evaluates several Canny edge settings: `(30, 90)`, `(50, 150)`, and `(80, 220)`. For each edge map it:

1. closes small gaps in edges;
2. finds contours;
3. approximates each closed contour as a polygon;
4. retains convex quadrilaterals occupying 20-98% of the image;
5. rejects candidates with implausibly short sides; and
6. scores candidates by area while penalizing image-border rectangles.

If no credible quadrilateral is found, the minimum-area rectangle of the dominant edge contour is attempted as a fallback. If that also fails, the pipeline stops and requests explicit corners instead of silently treating the entire image as the arena.

Detected points are ordered as top-left, top-right, bottom-right, bottom-left. A projective homography maps these points to a rectified image with a nominal resolution of 10 pixels/cm. With the default dimensions, the rectified arena is 250 x 300 pixels.

For rectified pixel coordinates `(u, v)`, physical coordinates are:

```text
x_cm = u * arena_width_cm  / (rectified_width_px  - 1)
y_cm = v * arena_height_cm / (rectified_height_px - 1)
```

The origin `(0, 0)` is the arena's top-left corner. `x` increases to the right, and `y` increases downward in the camera view. If an analysis requires a Cartesian upward-positive y-axis, transform it afterward with `y_cartesian = arena_height_cm - y_cm`.

The `cm_per_px_x` and `cm_per_px_y` fields in `calibration.json` describe the approximate source-image scale calculated from opposing side lengths. Actual trajectory conversion uses the perspective transform and the rectified scales, which is the correct procedure when the arena is trapezoidal in the source.

#### Manual corner correction

Inspect `arena_calibration_preview.png`. The green polygon should follow the inner usable arena boundary. If it does not, obtain the four source-image pixel coordinates and rerun:

```bash
python mouse_behavior_pipeline.py \
  --input recording.mp4 \
  --corners "100,60;540,80;520,420;80,400"
```

Corner order must be top-left, top-right, bottom-right, bottom-left. Keep one consistent definition of the physical boundary across every experimental session.

### 3. Static background model

Every sampled frame is perspective-warped into the common arena coordinate system. Their pixelwise median becomes `median_background.png`. A moving mouse is normally rejected by the median, whereas the static arena floor remains.

For dark animals recorded on a bright arena, use a high temporal percentile rather than the median for the background. The default `--background-percentile 85` retains the bright floor when a mouse, tether, or miniscope occupies a pixel in a substantial minority of calibration frames. Foreground is computed one-sided as `background - frame`, so bright reflections are not treated as animal foreground. Set a different percentile only after checking the saved background image.

For each sample, the grayscale one-sided difference from this background is measured. Unless `--threshold` is supplied, the foreground threshold is:

```text
clip(max(12, median_difference + 8 * max(MAD, 1)), 12, 60)
```

Here MAD is the median absolute deviation. This robust rule is less sensitive to mouse pixels and occasional compression artifacts than a mean/standard deviation estimator.

### 4. Mouse segmentation

Each full-resolution video frame is warped into the calibrated arena. The per-channel absolute difference from the median background is calculated, and the maximum channel difference is thresholded into a binary foreground mask.

The pipeline then:

1. clears a narrow rectified-image margin to suppress arena-wall artifacts;
2. applies morphological opening to remove isolated foreground noise;
3. applies morphological closing to fill small holes in the mouse silhouette;
4. labels connected components;
5. keeps components occupying 0.04-20% of the arena; and
6. selects the component using an area score with a temporal-distance penalty.

The selected component is opened to remove thin appendages, then the largest remaining torso is retained. The torso mask is lightly closed and Gaussian-smoothed before extracting the final contour. A predicted torso bounding box from the previous frame constrains the next candidate: compact candidates overlapping the expanded prior box and close to the prior torso centroid are preferred over distant cable fragments or reflections. The final torso center of mass is the body coordinate.

The broad component-area limits allow different image scales and postures. For best scientific accuracy, inspect the contour during wall contact, grooming, and rearing. Persistent shadows or dark cables can merge with the component and should be eliminated during acquisition or handled by adjusting lighting and `--threshold`.

### 5. Head estimation and temporal stabilization

The contour points are centered on the body centroid, and a 2D covariance matrix is calculated. The eigenvector with the largest eigenvalue is the silhouette's major axis. Projecting every contour point onto that axis produces two extreme candidates, corresponding approximately to the two ends of the animal.

Head/tail polarity is resolved temporally:

- during locomotion of at least 0.75 rectified pixels/frame, the leading candidate in the body-centroid movement direction is preferred;
- at low speed, the previous anatomical orientation and predicted head location are preferred to avoid polarity flicker during grooming or rest;
- strong directional evidence can deliberately reverse an initially arbitrary polarity choice; and
- accepted head motion is low-pass filtered and step-limited to reject jumps.

The estimated head is velocity-limited and cannot flip to the opposite torso endpoint in one frame. This substantially reduces tether-induced jumps, at the cost of a short lag during abrupt turns. Missing head observations remain missing and are only jointly interpolated with the body across short internal gaps.

This procedure identifies an estimated head end, not a trained nose keypoint. The first stationary frame is intrinsically ambiguous, and backward walking, tight turns, rearing, grooming, or a visible tail can challenge the heuristic. Experiments requiring anatomical nose accuracy should compare this output with manually labeled frames or replace this stage with a validated animal-pose model.

### 6. Missing data and trajectory assembly

For each decoded frame the pipeline records:

| Column | Unit | Meaning |
| --- | --- | --- |
| `frame` | frame index | Zero-based decoded frame number. |
| `timestamp_sec` | seconds | `frame / FPS`. |
| `body_x_cm` | cm | Body-centroid horizontal position. |
| `body_y_cm` | cm | Body-centroid vertical position. |
| `head_x_cm` | cm | Filtered estimated-head horizontal position. |
| `head_y_cm` | cm | Filtered estimated-head vertical position. |

Internal missing values are linearly interpolated only across gaps of at most 0.5 seconds. Leading, trailing, and longer gaps remain empty (`NaN`) so data loss remains visible to downstream analyses.

### 7. Annotated video

The pipeline performs a second pass over the source and writes an MP4 at the original resolution and frame rate. It draws:

- the calibrated arena polygon in green;
- the cumulative body trail;
- the current estimated head as a red dot;
- the current mouse contour unless `--no-contour` is used; and
- frame number and timestamp.

Segmentation in this pass is computed from the untouched source frame before overlays are drawn, preventing the annotation itself from contaminating the foreground mask.

## Output files

The four primary deliverables are:

| File | Description |
| --- | --- |
| `calibration.json` | Source corners, arena dimensions, source/rectified scales, homography, sampling indices, and calibration diagnostics. |
| `trajectory.csv` | One row per processed frame with timestamps and body/head coordinates. |
| `annotated_output.mp4` | Source-resolution video with calibration and tracking overlays. |
| `pipeline_log.txt` | Metadata, thresholds, progress, warnings, missing rate, QA statistics, and final status. |

Two additional diagnostic images are written:

- `arena_calibration_preview.png`: median source frame with detected corners;
- `median_background.png`: rectified static background used for subtraction.

All artifacts are replaced when the same output directory is reused. Use a separate output directory for each animal/session.

## Validation protocol

Do not use a new camera/arena configuration for scientific analysis without a visual validation pass. A practical procedure is:

1. Run the first few thousand frames with `--max-frames`.
2. Check `arena_calibration_preview.png` for correct boundary placement.
3. Inspect the annotated video at the beginning, middle, and end.
4. Specifically inspect wall contact, corners, rapid turns, immobility, grooming, rearing, and shadowed frames.
5. Confirm that the contour contains the mouse without walls or shadows.
6. Confirm that the body trail follows the animal and remains inside 0-25 cm by 0-30 cm, or the custom arena dimensions.
7. Confirm that the red marker follows the head rather than repeatedly flipping between the two ends.
8. Read `pipeline_log.txt` and investigate high missing rates, unstable mask area, implausible speed, or implausible head-body distance.
9. Adjust corners or threshold and repeat the subset run.
10. Remove `--max-frames` only after the subset is reliable.

Example subset run:

```bash
python mouse_behavior_pipeline.py \
  --input recording.mp4 \
  --output-dir results/calibration_test \
  --max-frames 3000
```

Example full run after validation:

```bash
python mouse_behavior_pipeline.py \
  --input recording.mp4 \
  --output-dir results/session_01
```

## Reading the QA log

`pipeline_log.txt` reports several useful diagnostics:

- **raw missing body detections**: should be low; a high fraction indicates a poor threshold, background, arena mask, or recording;
- **body speed p99**: unusually large values suggest component swaps or centroid jumps;
- **head-body distance median/p99**: should be compatible with mouse body size; sudden increases suggest head-end errors;
- **mouse mask area median and coefficient of variation (CV)**: moderate posture-dependent variation is expected, while large variation suggests shadows, wall merging, or fragmented masks; and
- **background difference statistics**: used to diagnose illumination drift, compression, and an insufficiently sampled background.

These are diagnostics rather than universal pass/fail thresholds. Acceptable values depend on camera resolution, frame rate, mouse size, and experimental conditions.

## Parameter tuning and troubleshooting

### No arena is detected

- Verify that all four arena sides are visible and have contrasting edges.
- Inspect a normal frame for reflections or objects crossing the boundary.
- Supply manual corners with `--corners`.
- Confirm corner ordering and use the same physical boundary definition across sessions.

### The mouse mask is empty or fragmented

- Try a lower fixed threshold, for example `--threshold 12`.
- Confirm that the mouse differs sufficiently from the arena floor.
- Check whether illumination changes over time.
- Inspect `median_background.png` for a residual mouse caused by prolonged immobility at one location.

### Shadows or walls are included

- Try a higher threshold, for example `--threshold 25`.
- Correct the arena corners so wall pixels lie outside the rectified interior.
- Reduce glare and directional shadows during acquisition.
- Prefer diffuse, temporally stable illumination.

### The head marker flips or follows the tail

- Inspect whether body motion represents forward locomotion in the failing segment.
- Check whether a long visible tail dominates the silhouette's major axis.
- Treat head estimates during immobility, rearing, and grooming cautiously.
- For nose-level precision, validate against manual labels or use a trained pose estimator; body-centroid tracking can still remain valid independently.

### The annotated MP4 cannot be written

- Confirm that the output directory is writable.
- Verify that the OpenCV build supports the `mp4v` codec.
- Ensure sufficient free disk space for a full-length re-encoded video.

## Reproducibility and data management

- Record the Git commit used for each analysis.
- Keep raw recordings immutable and outside version control.
- Store each session in its own output directory.
- Archive `calibration.json`, `trajectory.csv`, and `pipeline_log.txt` with the experimental metadata.
- Retain the annotated video for traceable visual quality control.
- Document every manual corner or threshold override.
- Revalidate after changing camera position, zoom, arena geometry, lighting, or video encoding.

The repository intentionally ignores raw and generated MP4 files because they may be large or contain sensitive experimental data. Git should contain the code and documentation; durable research storage should contain recordings and per-session results.

## Known limitations

- Only one connected animal is tracked.
- Background subtraction requires a fixed camera and sufficiently static scene.
- Median modeling can fail when the mouse remains in one position for most sampled frames.
- The body centroid is a silhouette center of mass, not an anatomical landmark.
- The estimated head is a temporally stabilized silhouette endpoint, not a learned nose keypoint.
- Severe occlusion, climbing, rearing, grooming, reflections, bedding motion, cables, and strong shadows can invalidate classical segmentation.
- The pipeline does not calculate behavioral labels such as freezing, center time, grooming, or rearing; these can be derived and validated downstream from the exported trajectory and video.

Always inspect the annotated output before interpreting trajectory-derived behavioral measures.
## Code organization

All maintained implementation code is now under `code/`:

- `code/mouse_behavior_pipeline.py` — standard body/head trajectory pipeline
- `code/extract_mouse_only_video.py` — original-FOV mouse-only extraction
- `code/export_foreground_mask.py` — foreground-mask QA video export
- `code/compare_head_methods.py` — silhouette vs. miniscope-reflection comparison
- `code/annotate_head_anchors.py` — manual anatomical-head/reflection labelling
- `code/train_head_calibrator.py` and `code/finalize_head_trajectory.py` — lightweight manual-calibration model
- `code/behavior_analyze/` — reusable geometry, segmentation, tracking and calibration interfaces

Root-level `.py` files are compatibility launchers only, so existing commands
continue to work. New development should import from `code.behavior_analyze`.
## Project layout

```text
behavior_analyze/
├── .git/                 # repository metadata stays at root
├── data/                 # raw videos stay at root
├── traditional/          # classical/reflection/manual-calibration code + results
└── unet/                 # GPU-oriented CNN segmentation code
```

## Scientific desktop GUI

For tethered-miniscope videos, launch the U-Net production GUI:

```powershell
python unet/run_unet.py ui
```

Its default tab performs one-pass full analysis and writes the clean mask,
head/body overlay, trajectory CSV and metadata. A second tab rebuilds and
trains from existing labels without requiring more annotation; manual tools
are isolated on an advanced tab. Every video has its own ROI and output path,
and all long operations run in cancellable child processes.

In **使用已有标注训练**, click **一键重建数据集并训练** to reuse all saved
annotations. **拼接查看全部已有标注** displays 20 labels per page and supports
legacy path/polygon formats; **查看上次重建样本** separately checks the last exported training masks. Use
**复查/修正已有轮廓** on the advanced tab to overwrite an incorrect old label.
Closing that review window automatically rebuilds the current video's dataset,
so newly excluded frames and corrected masks cannot remain as stale PNGs.
In the contact sheet, double-click a thumbnail to open a large editor. Drag a
point or click an edge to add one; every completed edit is atomically saved to
the CSV and the contact-sheet page refreshes when the editor closes.

U-Net v3 has three supervised outputs: mouse/miniscope mask, anatomical Head
heatmap, and Reflection heatmap. Existing `head_anchor_calibration.csv` rows
provide both point targets, so upgrading does not require relabelling. The
learned Reflection is fused with the legacy bright-spot detector; old mask-only
and Mask+Head checkpoints remain loadable and automatically use the legacy
detector. A learned Reflection branch is enabled for inference only when its
saved validation error is available and no greater than 18 px.

Training never overwrites a checkpoint. Each successful run creates a file
such as `best_unet_reflection_YYYYMMDD_HHMMSS_microseconds.pt`; the model chosen
before training is read only as a warm-start source, and the GUI automatically
selects the newly promoted file. Model selection uses Mask Dice together with
Head and Reflection errors instead of Mask Dice alone.

Head tracking remains reflection-anchored: the fused Reflection is used when
available, the anatomical Head heatmap may make an agreeing correction, and
acts alone when Reflection is missing. After a full analysis, use
**根据结果补充 Head 标记** to correct automatically selected low-confidence or
high-disagreement frames; the accepted green mask is snapshotted alongside the
head label so it can supervise the next training run.

The legacy classical GUI remains available with:

```powershell
python traditional/behavior_analyze_gui.py
```
