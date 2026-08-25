# Mouse Pose Hybrid Workbench

This is the recommended GUI for the project. It implements the complete
pipeline:

```text
traditional background subtraction
        ↓ pseudo-label mouse boxes
visual contact-sheet review + draggable box editor
        ↓ reviewed YOLO detection dataset
Ultralytics YOLO mouse detector
        ↓ high-SNR dynamic square crop
neural super-resolution (EDSR / ESPCN / FSRCNN / LapSRN)
        ↓ fixed-size enhanced crop video
DeepLabCut SuperAnimal-TopViewMouse
        ↓ inverse per-frame crop transform
source-pixel and arena-centimetre head/body trajectories
```

The workbench is entirely in English. The older `dlc/gui.py` is retained only
for compatibility; launch `hybrid_gui.py` for new experiments.

## Installation on the execution computer

Create a separate environment. Install the CUDA build of PyTorch/torchvision
that matches that computer first, then:

```powershell
python -m pip install -r dlc/requirements-hybrid.txt
python dlc/hybrid_gui.py
```

On first launch the GUI loads `config.hybrid.example.json`. Saving creates
`dlc/hybrid_config.json`, which is the per-computer configuration. Relative
paths are resolved relative to the JSON file rather than the terminal working
directory.

The first YOLO and DLC run downloads their pretrained weights. OpenCV neural
super-resolution models are not embedded in this Git repository; download the
desired `.pb` model on the execution computer and select it under
`Super-resolution crop stage`.

## Workflow

### 1. Setup

Select one or more source videos, the existing four-corner arena ROI JSON, the
analysis output directory, and physical arena dimensions. Run `Check
environment and files` before starting.

### 2. Generate and review YOLO boxes

`Generate automatic boxes` samples temporally distributed frames from every
video. It builds the same bright-floor percentile background used by the
traditional pipeline, rectifies each frame using the ROI, segments the mouse,
and maps the torso rectangle back to source pixels.

Every pseudo-label is stored in `box_labels.csv` with:

- source video and frame;
- source image and pixel rectangle;
- `source=traditional_background` or `traditional_missing`;
- an automatic confidence score;
- manual `reviewed` and `exclude` flags.

`Review / edit all boxes` opens a paginated contact sheet. Orange boxes are
unreviewed, green boxes were manually reviewed, and red samples are excluded.
Click a thumbnail to open the large editor:

- drag inside the box to move it;
- drag a corner to resize it;
- drag outside to replace it;
- mark unusable frames as excluded;
- save to change the source to `manual_review` with confidence 1.0.

Review low-confidence boxes and all failure modes: obstacle contact, complete
and partial occlusion, wall contact, miniscope/fibre overlap, grooming,
rearing, and human intervention. `Export reviewed YOLO dataset` writes the
official normalized YOLO detection format and a deterministic train/validation
split.

### 3. Train and validate YOLO

The default base checkpoint is `yolo26n.pt`. Train and validation parameters
are exposed in the GUI. The best checkpoint path is automatically copied into
the `Trained best.pt` field when training completes.

Validation mAP must be paired with visual inspection. The final pipeline also
reports direct YOLO detections, temporal fallbacks, and missing-box percentages
over the complete video.

### 4. Super-resolution and DLC

Choose a neural OpenCV super-resolution model:

- `edsr`: highest-quality default, usually slower;
- `espcn` or `fsrcnn`: faster alternatives;
- `lapsrn`: pyramid reconstruction;
- `bicubic`: non-neural ablation/fallback, not equivalent to learned SR;
- `none`: resize-only baseline.

The trained YOLO detector runs on every source frame. Its box is enlarged by
`crop_scale`, converted to a square, super-resolved, and written to a fixed-size
`dlc_input.mp4`. A short missing detection can reuse the last box for no more
than `max_fallback_sec`; longer misses produce a blank crop and remain missing
in the final trajectory.

Each frame records `x0`, `y0`, `crop_size`, and `output_size`. Therefore every
DLC keypoint is transformed back using:

```text
source_x = x0 + crop_x * crop_size / output_size
source_y = y0 + crop_y * crop_size / output_size
```

Only after this inverse transform are head/body points fused and mapped through
the arena homography to centimetres.

### 5. Optional supervised DLC fine-tuning

The same English workbench retains the complete DeepLabCut project workflow:
project creation, frame extraction, DLC labeling, label checking, SuperAnimal
transfer-dataset construction, training, evaluation, batch inference, outlier
extraction, refinement, merging, and iterative retraining.

### 6. Advanced parameters

Every primary parameter is shown directly. The `Advanced API` tab accepts JSON
objects keyed by action name. These objects are merged into the corresponding
Ultralytics or DeepLabCut Python call and override visible defaults, so the GUI
does not prevent access to newly added library parameters.

## Output structure

```text
output_dir/
  hybrid/<video_stem>/
    yolo_detection.mp4       full-frame detector QA
    dlc_input.mp4             SR crop video sent to DLC
    crop_transforms.csv       exact per-frame inverse transform
    hybrid_manifest.json
    dlc_input*.h5             raw crop-space DLC predictions
    final/
      trajectory.csv          final source/cm trajectory
      annotated_output.mp4    final overlay on the source video
      dlc_crop_keypoints.h5   original crop-space audit data
      dlc_source_keypoints.csv
      quality_report.json
```

`quality_report.json` includes YOLO direct/fallback/missing rates and final
head/body valid rates. Long complete occlusions remain `NaN`; they are not
fabricated by the crop transform or DLC interpolation.

## Reproducibility and licensing

- GUI tasks call `dlc/hybrid_jobs.py`, so every action can also be reproduced
  from a terminal with `python dlc/hybrid_jobs.py ACTION --config ...`.
- Training images, CSV labels, YOLO text labels, split manifest, intermediate
  video, transforms, raw DLC output, and final output are all retained.
- Ultralytics software/model licensing should be reviewed for the intended
  deployment. Its open-source distribution uses AGPL-3.0; commercial or
  externally deployed use may require a different license from Ultralytics.

## CPU tests

```powershell
python -m unittest dlc.test_hybrid_pipeline dlc.test_postprocess -v
```

Full YOLO, neural SR, and DLC inference must be validated on the execution GPU
computer using representative recordings.
