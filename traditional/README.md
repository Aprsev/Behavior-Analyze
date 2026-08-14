# Traditional tracking pipeline

This folder contains the complete rule-based pipeline and its existing results: dark
background subtraction, compact-body constraints, reflected-miniscope anchor,
manual head calibration, and temporal regularisation.

The actual shared implementation is in `traditional/code/`. Raw videos remain
at repository root in `../data/`; Git metadata remains at repository root.

## Main commands

```powershell
cd traditional
python process_new_video.py --input "../data/video.avi" --model "results/HQ312_manual_calibrated/head_calibrator_v2.joblib" --arena-width-cm 30 --arena-height-cm 25
python annotate_torso_constraints.py --help
python annotate_head_anchors.py --help
```

Use this pipeline for the current classical/hand-calibrated method. For CNN
segmentation that explicitly separates mouse from fibre, use `../unet/`.
