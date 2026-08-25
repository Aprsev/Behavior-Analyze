# Hybrid YOLO + super-resolution + DeepLabCut workflow

This is the primary, all-English workbench for occlusion-aware mouse tracking.
It follows the sequence below:

1. Estimate traditional foreground boxes as YOLO pseudo-labels.
2. Review and manually correct every proposed box in the visual editor.
3. Export and train an Ultralytics YOLO detector.
4. Crop the detected mouse and optionally apply OpenCV DNN super-resolution.
5. Run a pretrained DeepLabCut SuperAnimal model on the enhanced crops.
6. Map keypoints back to source-video coordinates and export head/centroid data.

## Install and launch

Use the environment doctor from the repository root:

```powershell
python dlc/environment_doctor.py
python dlc/environment_doctor.py --install --torch cu126
python dlc/hybrid_gui.py
```

The complete hybrid requirement group is stored at
`dlc/requirements/requirements-hybrid.txt`. The GUI loads
`dlc/hybrid/config.hybrid.example.json` on first launch and writes the
machine-local configuration to `dlc/hybrid/hybrid_config.json`.

## Reproducible command-line jobs

Every GUI operation runs a fresh worker process to isolate CUDA and Qt state.
The same actions can be launched directly:

```powershell
python -m dlc.hybrid.hybrid_jobs hybrid_check --config dlc/hybrid/hybrid_config.json
python -m dlc.hybrid.hybrid_jobs full_hybrid --config dlc/hybrid/hybrid_config.json
```

Run `python -m dlc.hybrid.hybrid_jobs --help` to see the current actions. Paths
inside the JSON are resolved relative to the JSON file, which keeps execution
consistent across computers.

## Data and model policy

Videos, generated datasets, downloaded weights, and trained checkpoints stay on
the execution computer. Source code, example configuration, and small metadata
are synchronized through Git. Completely opaque occlusion cannot be inferred as
ground truth; long low-confidence gaps remain missing rather than being silently
fabricated.
