# DLC environment setup

Run the standard-library-only doctor after activating the Conda environment
that will execute the project. Diagnostic mode never changes the environment.

```powershell
conda activate YOUR_ENVIRONMENT
python dlc/environment_doctor.py
```

DeepLabCut supports Python 3.10-3.12. Create a dedicated environment when the
doctor reports a different Python version:

```powershell
conda create -n Behavior-DLC python=3.12 pip -y
conda activate Behavior-DLC
python -m pip install --upgrade pip
```

Preview the repair before installing:

```powershell
python dlc/environment_doctor.py --install --dry-run --torch cu126
```

Apply it after reviewing the commands:

```powershell
python dlc/environment_doctor.py --install --torch cu126
```

Use the wheel index produced by the official PyTorch selector when another CUDA
build is required:

```powershell
python dlc/environment_doctor.py --install --torch-index-url https://download.pytorch.org/whl/CUDA_INDEX
```

The doctor checks Conda/Python identity, NVIDIA visibility, PyTorch CUDA tensor
allocation, DeepLabCut Model Zoo, Ultralytics, PySide6, OpenCV super-resolution,
pandas/PyTables HDF5 support, and `pip check`. It refuses Conda `base` by default
and never replaces PyTorch unless a build/index is explicitly selected.

If multiple OpenCV wheels are installed, create a fresh environment whenever
possible. The explicit cleanup path can be previewed with:

```powershell
python dlc/environment_doctor.py --install --dry-run --fix-opencv
```

Add `--json-report environment-report.json` for a machine-readable report.
