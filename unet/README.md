# U-Net mouse / miniscope segmentation

This is the GPU-oriented CNN path. It replaces dark-background segmentation
with a supervised pixel classifier. The initial release trains a **binary**
mask: `mouse + head-mounted miniscope = 1`; `fibre + tail + arena = 0`.
This directly solves the current failure mode where a dark fibre is mistaken
for the mouse.

The existing polygon labels from `../traditional/annotate_torso_constraints.py`
are accepted as positive masks. Draw them tightly around mouse plus miniscope,
and explicitly exclude fibre/tail. Label 80–150 diverse frames before the
first training run.

## GPU-host setup

Install CUDA-compatible PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```powershell
python -m pip install -r unet/requirements-gpu.txt
```

## 1. Export polygon labels to U-Net dataset

```powershell
python unet/prepare_dataset.py --video "data/video.avi" --labels "traditional/results/project/manual_torso_constraints.csv" --output-dir "unet/datasets/project"
```

## 2. Train

```powershell
python unet/train.py --dataset "unet/datasets/project" --output-dir "unet/models/project" --epochs 80 --batch-size 8
```

## 3. Predict a new video

```powershell
python unet/infer.py --video "data/video.avi" --model "unet/models/project/best_unet.pt" --output-dir "results/video_unet" --threshold 0.5
```

Outputs include `mouse_miniscope_mask.mp4`, `mouse_miniscope_overlay.mp4`, and
`unet_trajectory.csv`. The trajectory contains the CNN mask centroid; it is a
clean body input for the later head/reflection pipeline.

## Important scope

The binary model is deliberately the first production step because current
polygon labels specify the desired mouse/miniscope region. A four-class model
(`background`, `mouse`, `miniscope`, `fibre`) requires separate fibre and
miniscope pixel labels; the data format here is intentionally extensible for
that next iteration.
