# U-Net mouse / miniscope segmentation

This is the GPU-oriented CNN path. It replaces dark-background segmentation
with a supervised pixel classifier. The initial release trains a **binary**
mask: `mouse + head-mounted miniscope = 1`; `fibre + tail + arena = 0`.
This directly solves the current failure mode where a dark fibre is mistaken
for the mouse.

## Fibre-aware v2 workflow (no additional labelling required)

The current training path reuses all existing torso and head labels. Do not
repeat screening/annotation just because the model has been retrained.

The v2 model fixes the tether case at three levels:

- two input channels: raw grayscale keeps a stationary mouse visible, while
  the per-video background residual provides motion contrast;
- hard-negative mining: the highest-probability pixels outside the manual
  mouse mask receive extra loss, so a moving black fibre is learned as a hard
  negative instead of being drowned by millions of easy arena pixels;
- reflection-first head fusion: a valid physical reflection cannot be
  replaced by a confidently wrong heatmap. An agreeing heatmap contributes at
  most 15%; when reflection is missing it becomes the fallback. Per-frame
  `head_source` and both candidate coordinates are exported for auditing;
- fibre-aware temporal filtering: thin components are opened away, remaining
  components are ranked by compactness, CNN probability and overlap with the
  previous body. After about 0.35 s of disagreement the prior is released and
  the tracker automatically reacquires a mouse moved by the experimenter.

Existing `head_anchor_calibration.csv` points are now exported as Gaussian
heatmaps and train an optional head decoder. At inference the learned head is
restricted to the clean body mask; old checkpoints automatically fall back to
the reflection tracker.

Run only these steps on the GPU host after pulling the new code:

```powershell
python unet/run_unet.py prepare
python unet/run_unet.py train
python unet/run_unet.py infer
python unet/run_unet.py head
```

`prepare` now synchronizes the dataset with the CSV and removes stale samples
for each video. `train` reads only `dataset.json`, warm-starts from the current
`best_unet.pt`, archives that checkpoint, and saves a new candidate. The
candidate replaces the best model only when it is not worse on the exact same
contiguous temporal validation split. Use `python unet/train.py ... --fresh`
only when an intentional from-scratch experiment is required.

The synthetic fibre regression test can be run without a GPU:

```powershell
python unet/test_postprocess.py
```

The default opening kernel is 5 pixels at 256×256 model resolution. If the
physical tether is visibly thicker in the mask, try `--fibre-opening 7` on
`infer`/`head`; use an odd value. `--reacquire-sec 0.35` controls how long a
disjoint candidate is rejected before automatic relocation recovery.

The existing polygon labels from `../traditional/annotate_torso_constraints.py`
are accepted as positive masks. Draw them tightly around mouse plus miniscope,
and explicitly exclude fibre/tail.

## Multi-video workflow (recommended)

`unet/run_unet.py` is the one-command launcher. The `VIDEOS` list at the top
of the file defines every recording that will contribute to training; all
labels, screening and dataset export run over that list automatically.

```powershell
# 0. Install CUDA PyTorch once (edit VIDEOS/SCREENING paths in run_unet.py if needed)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r unet/requirements-gpu.txt

# 0b. Production GUI (recommended): one-click analysis, existing-label
#     rebuild/train, and advanced annotation on separate tabs. Every video has
#     its own ROI/output directory; paths and fibre settings survive restarts.
#     Video decoding/training runs in child processes and never blocks Tk.
python unet/run_unet.py ui

# In the GUI, open “使用已有标注训练”. The first, prominent button is
# “一键重建数据集并训练”; it reuses all saved CSV labels and does not start
# another annotation round. “拼接查看全部已有标注” opens a 5 x 4 contact
# sheet from the source-of-truth CSV plus legacy exported samples. It accepts
# old full-path/time-only video IDs and Nx2/Nx1x2/flat polygons of any point
# count. “查看上次重建样本” separately shows the actual exported mask (green)
# and available head heatmap (red). Arrow keys change pages.
# “复查/修正已有轮廓” automatically re-exports the current video after the
# correction window closes, including removal of newly excluded stale PNGs.
# In the contact sheet, double-click a tile and choose body contour or
# Head/Reflection. For keypoints use H/R to select, click/drag to move,
# C to confirm, X/N to mark Reflection/Head absent, and S to save+close.
# Completed edits auto-save to CSV and refresh the page.

# 0b. NEW video, first time: click the 4 arena corners to make its ROI JSON
#     (saved as traditional/basic_rois/{stem}_roi.json), then add the
#     video/roi lines to the VIDEOS list at the top of run_unet.py
python unet/run_unet.py roi --video "data/new.avi"

# 1. Status overview: what exists, what is missing
python unet/run_unet.py check

# 2. (re)generate head_method_comparison.csv for every video (needed by annotator)
python unet/run_unet.py compare

# 3. Screen frames: 10 frames per video (default; adjustable), spread
#    across the whole recording - the video is divided into 10 equal time
#    bins and the most distinct frame of each bin is kept, so no cluster
#    of near-duplicate frames appears. A montage lets you exclude junk
#    (mouse absent, human intervention, motion blur), auto-flagged last.
python unet/run_unet.py screen

# 4. Annotate torso polygons for the screened frames of each video
#    (external OpenCV window; ~8 points per mouse are enough, press T to
#    thin any automatic contour to 8; already labelled frames are skipped)
python unet/run_unet.py annotate

# 5. Export all videos into one dataset (excluded frames are dropped;
#    image names are prefixed per video so nothing collides)
python unet/run_unet.py prepare

# 6. Train on the combined dataset (epochs / learning rate in the GUI;
#    live loss curve is drawn while training runs)
python unet/run_unet.py train

# 7. Manual model calibration: 20 least-confident frames (sorted by mean
#    |p-0.5| inside the predicted mask) open in one window where you can
#    correct the polygon contour + head point + reflection point. Saving
#    automatically re-exports the dataset and retrains.
python unet/run_unet.py calibrate

# 8. Predict a new video; excluded frames become NaN + EXCLUDED overlay
python unet/run_unet.py infer

# 9. Combined body + head tracking: U-Net mask centroid (body) +
#    miniscope reflection inside the clean mask (head)
python unet/run_unet.py head
```

### Running on a brand-new video (no manual path editing)

- Only inference (use the trained model, no retraining):

  ```powershell
  python unet/run_unet.py head --interactive
  ```
  The first dialog asks for the video; the ROI is then resolved
  automatically — existing `{stem}_roi.json`, or any basic_rois JSON whose
  `input` field points at the video, or (if none exists) the corner picker
  opens right there so you can click the 4 arena corners on the spot. Then
  pick the model and the output dir. (`infer --interactive` works the same
  way, minus the ROI dialog.) All dialogs pre-fill with the defaults, so
  plain Enter accepts them.

- To add the video to training, append a dict to the `VIDEOS` list in
  `run_unet.py` (3 lines), then run `compare screen annotate prepare train`
  once each over the whole list.

### Frame selection and screening

- The old low-confidence + uniform picking produced many near-identical
  frames. `scan_candidates` scans the whole video and keeps frames with a
  plausible automatic mouse segmentation; `farthest_pick_binned` then
  divides the recording into `--per-video` equal time bins (default **10**)
  and takes the frame whose normalized arena appearance (position, posture)
  differs most from everything already picked **in each bin**. The 10
  candidates therefore span the entire recording and never cluster in one
  quiet/active period, which removes the near-duplicate frames.
- The GUI exposes the budget as "每视频标注帧数" (③ screening and ④
  annotate use it; default 10). The annotator never shows more than this
  many *new* frames per video per run.
- `unet/screen_frames.py` shows the picked frames as 3x3 montage pages
  (click to toggle EXCLUDED, `n`/`p` pages, `s` save, `q` quit). Frames where
  the automatic segmentation failed — mouse absent, human intervention,
  heavy occlusion — are appended pre-excluded; click to re-include if a
  flag is wrong. Output: `screening.csv` (`video, frame, exclude`).
- The same `screening.csv` is consumed downstream:
  - annotator (`--candidate-csv`): labels only screened, non-excluded frames;
  - `prepare_dataset.py` (`--exclude-csv`): drops excluded frames from training;
  - `infer.py` (`--exclude-csv`): excluded frames keep the video length but
    their trajectory rows are `NaN`, the mask frame is black and the overlay
    frame is marked `EXCLUDED`; `inference.json` reports the count.

### Training on several videos

Run `prepare` once per video (the launcher does this for the whole `VIDEOS`
list). Image names are `{video_stem}_{frame:07d}.png`, so frame numbers never
collide between videos, and `dataset.json` accumulates one entry per video.

### Background-invariant preprocessing (inter-video background differences)

A single recording has an almost static background, but different recordings
may look very different (arena, lamp, camera). To stop the U-Net from
memorizing one arena's appearance without making a resting mouse disappear,
the pipeline supplies both the original image and a background residual:

- `prepare_dataset.py` samples ~61 frames spread over the video and caches
  the per-pixel 85th-percentile background as `<dataset>/backgrounds/<stem>.png`;
- `train.py` builds `[gray, 128 + 2*(gray-bg)]`: the first channel preserves
  mouse/miniscope appearance while the second highlights deviations;
- the checkpoint records `"bg_subtract": true`;
- `infer.py` / `head_track.py` read that flag and apply the identical
  transform at inference (estimating the background with the same function,
  rotated together with `--rotate` frames). Old checkpoints without the flag
  keep the raw-input behaviour, so nothing breaks until you retrain.

One-time migration after pulling: rerun `prepare` (⑤) once so every video
gets its background cache, then `train` (⑥). Training refuses a mixed dataset
when even one video lacks a background cache, preventing train/inference input
mismatch.

### Manual model calibration (step ⑦)

After training, the model's weak spots are usually a handful of hard frames.
`unet/calibrate_model.py` finds them and lets you correct them by hand:

1. scores every frame of the video with the trained model; the uncertainty
   score is the mean `|p-0.5|` **inside the predicted mask** (background
   pixels are confidently 0 and would drown the metric otherwise);
2. keeps only plausible masks (0.1%-20% of the frame) and picks the
   `--n-frames` (default 20) lowest-scoring frames, with a temporal diversity
   gap; frames already saved by an earlier calibration round are skipped;
3. opens them in one window sorted by confidence, where you can drag the
   polygon vertices (or click an edge to insert), drag the **HEAD** dot
   (red, initialized from `head_method_comparison.csv`) and the
   **REFLECTION** dot (magenta) — per-frame `v`/`x` mark the head/reflection
   as absent;
4. `s` saves, `q`/`Esc`/window-X saves and exits, `r` restores the model's
   polygon, `t` thins to 8 points, `e` excludes the frame from training.

Corrections are upserted per (video, frame) into `manual_torso_constraints.csv`
and `head_anchor_calibration.csv`. Exiting with corrections saved makes
`run_unet.py calibrate` re-export the dataset and retrain automatically
(`exit 0` = an actual edit was saved; closing without edits skips retraining).
The GUI shows the
annotation statistics per video (screened / labelled / remaining / excluded /
background cache).

## Direct commands (without the launcher)

```powershell
# Screen one video
python unet/screen_frames.py --video "data/video.avi" --roi "traditional/basic_rois/video_roi.json" --output "traditional/results/screening.csv" --per-video 10 --junk 20

# Annotate using the screened candidates
python traditional/annotate_torso_constraints.py --input "data/video.avi" --comparison-csv "traditional/results/basic_recognition/video/head_method_comparison.csv" --roi-json "traditional/basic_rois/video_roi.json" --output "traditional/results/manual_torso_constraints.csv" --arena-width-cm 25 --arena-height-cm 30 --candidate-csv "traditional/results/screening.csv"

# Export labels for one video into the shared dataset
python unet/prepare_dataset.py --video "data/video.avi" --labels "traditional/results/manual_torso_constraints.csv" --heads "traditional/results/head_anchor_calibration.csv" --roi-json "traditional/basic_rois/video_roi.json" --arena-width-cm 25 --arena-height-cm 30 --output-dir "unet/datasets/project" --exclude-csv "traditional/results/screening.csv"

# Train
python unet/train.py --dataset "unet/datasets/project" --output-dir "unet/models/project" --epochs 80 --batch-size 8

# Predict a new video (excluded frames -> NaN + EXCLUDED overlay)
python unet/infer.py --video "data/video.avi" --model "unet/models/project/best_unet.pt" --output-dir "results/video_unet" --threshold 0.5 --exclude-csv "traditional/results/screening.csv"
```

When a legacy Mask+Head checkpoint is upgraded with a new Reflection decoder,
training runs in reflection-isolated mode. The complete inherited network and
BatchNorm statistics are frozen, while a versioned two-convolution Reflection
refinement branch trains independently. Mask and Head therefore remain exactly
the source model rather than being softly encouraged to stay close. Reflection
heatmaps also receive an explicit coordinate loss so that a plausible blob on
the fibre or tail is penalized. Checkpoint v4 records this branch explicitly;
v1 mask-only, v2 Mask+Head, and v3 single-layer Reflection files remain directly
loadable.

Promotion is regression-gated. A candidate is selected automatically only if
its comparable Mask+Head score is at least the baseline score, Dice drops by no
more than 0.002, Head error increases by no more than 1 px, and all active
keypoint errors remain within their readiness limits. A rejected checkpoint is
retained as `candidate_unet_reflection_*.pt` for inspection; the selected source
model and every pre-existing `.pt` file remain unchanged.

Outputs of inference include `mouse_miniscope_mask.mp4`,
`mouse_miniscope_overlay.mp4`, and `unet_trajectory.csv`. The trajectory
contains the CNN mask centroid; it is a clean body input for the later
head/reflection pipeline.

### Coordinate calibration & wall-band exclusion

The ROI often includes the arena walls because the camera sees the mouse's
shadow projected on them; wall content changes with the mouse and must not
influence the segmentation.

- `unet/calibrate.py` (automatic, no user input):
  - `detect_floor_bounds`: finds the floor rectangle inside the rectified
    background (Otsu bimodal split, solidity check). Everything outside it —
    the wall band — is zeroed in the mask before computing the body centroid
    and the reflection search, so wall projections can never shift the body
    or create fake heads. Works with bright or dark walls; returns None when
    there is no wall contrast (whole arena treated as floor).
  - `refine_corners`: snaps the four clicked arena corners to the detected
    arena edges (Hough lines), turning a 10-15 px click error into ~1 px —
    this is what keeps every video on the same cm coordinate system.
- `make_roi.py`: after clicking the 4 corners press **A** for auto-snap.
- `head_track.py`: runs the floor detection automatically, logs the result
  (e.g. `Floor detected (50, 40, 270, 200) (77% of rectified arena)`), draws
  the floor outline in the overlay video, and records `floor_bounds` in the
  metadata JSON. Check the overlay once per video: the orange rectangle must
  match the real floor edges.

### Arena turned? (rotation robustness)

If a new recording looks like the arena was rotated a quarter-turn vs the
training videos, two layers fix it:

1. **Inference-side alignment (no retraining)**: `--rotate 90/180/270` on
   `infer`/`head` rotates the frames before the CNN and rotates the mask /
   centroid / ROI corners back, so the model always sees the training
   orientation. In the GUI: the "画面旋转校正" dropdown in the processing tab.
2. **Training-side augmentation (root cause)**: `train.py` now rotates
   training pairs by 90/180/270 degrees (zero border artifacts), adds gamma
   lighting curves, sensor noise and mild blur. Retrain once
   (`python unet/run_unet.py train`) and the model handles any quarter-turn
   plus small angular tilts and lighting differences on its own.

### Head mode (body + miniscope reflection tracking)

`python unet/run_unet.py head` runs `unet/head_track.py` on the first video
(override with `--video`, needs `--roi`). Per frame it:

1. runs the U-Net once to get the clean mouse+miniscope mask (fibre/tail and
   floor reflections are already suppressed by the trained model);
2. removes thin fibre branches and selects the compact, temporally consistent
   body component, then transforms it into rectified arena space;
3. **body** = mask centroid, converted to cm;
4. **head** = the learned head heatmap restricted to a dilated clean mask;
   checkpoints without a head decoder fall back to `ReflectionTracker`;
5. excluded frames (from `screening.csv`) are NaN rows with an EXCLUDED border.

Outputs in the inference folder:

- `head_track_trajectory.csv` — `frame, timestamp_sec, body_x_cm, body_y_cm,
  head_x_cm, head_y_cm, head_confidence`;
- `head_track_overlay.mp4` — green U-Net mask, red body dot, yellow head dot,
  white body→head line;
- `mouse_miniscope_mask.mp4` — binary clean mask from the same single pass;
- `head_track_metadata.json` — device, head-valid percentage, excluded frames.

## Important scope

The binary model is deliberately the first production step because current
polygon labels specify the desired mouse/miniscope region. A four-class model
(`background`, `mouse`, `miniscope`, `fibre`) requires separate fibre and
miniscope pixel labels; the data format here is intentionally extensible for
that next iteration.
