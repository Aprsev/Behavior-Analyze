# U-Net mouse / miniscope segmentation

This is the GPU-oriented CNN path. It replaces dark-background segmentation
with a supervised pixel classifier. The initial release trains a **binary**
mask: `mouse + head-mounted miniscope = 1`; `fibre + tail + arena = 0`.
This directly solves the current failure mode where a dark fibre is mistaken
for the mouse.

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

# 1. Status overview: what exists, what is missing
python unet/run_unet.py check

# 2. (re)generate head_method_comparison.csv for every video (needed by annotator)
python unet/run_unet.py compare

# 3. Screen frames: farthest-point sampling picks the most different frames of
#    every video; a montage lets you exclude junk (mouse absent, human
#    intervention, motion blur). Junk is auto-flagged and shown last.
python unet/run_unet.py screen

# 4. Annotate torso polygons for the screened frames of each video
#    (external OpenCV window; frames already labelled are skipped)
python unet/run_unet.py annotate

# 5. Export all videos into one dataset (excluded frames are dropped;
#    image names are prefixed per video so nothing collides)
python unet/run_unet.py prepare

# 6. Train on the combined dataset
python unet/run_unet.py train

# 7. Predict a new video; excluded frames become NaN + EXCLUDED overlay
python unet/run_unet.py infer
```

### Frame selection and screening

- The old low-confidence + uniform picking produced many near-identical
  frames. `scan_candidates` now scans the whole video, keeps frames with a
  plausible automatic mouse segmentation, and `farthest_pick` greedily picks
  the frames whose normalized arena appearance (position, posture) differs
  most from everything already picked.
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

## Direct commands (without the launcher)

```powershell
# Screen one video
python unet/screen_frames.py --video "data/video.avi" --roi "traditional/basic_rois/video_roi.json" --output "traditional/results/screening.csv" --per-video 40 --junk 20

# Annotate using the screened candidates
python traditional/annotate_torso_constraints.py --input "data/video.avi" --comparison-csv "traditional/results/basic_recognition/video/head_method_comparison.csv" --roi-json "traditional/basic_rois/video_roi.json" --output "traditional/results/manual_torso_constraints.csv" --arena-width-cm 25 --arena-height-cm 30 --candidate-csv "traditional/results/screening.csv"

# Export labels for one video into the shared dataset
python unet/prepare_dataset.py --video "data/video.avi" --labels "traditional/results/manual_torso_constraints.csv" --output-dir "unet/datasets/project" --exclude-csv "traditional/results/screening.csv"

# Train
python unet/train.py --dataset "unet/datasets/project" --output-dir "unet/models/project" --epochs 80 --batch-size 8

# Predict a new video (excluded frames -> NaN + EXCLUDED overlay)
python unet/infer.py --video "data/video.avi" --model "unet/models/project/best_unet.pt" --output-dir "results/video_unet" --threshold 0.5 --exclude-csv "traditional/results/screening.csv"
```

Outputs of inference include `mouse_miniscope_mask.mp4`,
`mouse_miniscope_overlay.mp4`, and `unet_trajectory.csv`. The trajectory
contains the CNN mask centroid; it is a clean body input for the later
head/reflection pipeline.

## Important scope

The binary model is deliberately the first production step because current
polygon labels specify the desired mouse/miniscope region. A four-class model
(`background`, `mouse`, `miniscope`, `fibre`) requires separate fibre and
miniscope pixel labels; the data format here is intentionally extensible for
that next iteration.