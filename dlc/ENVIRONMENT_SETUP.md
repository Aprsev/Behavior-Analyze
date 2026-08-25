# Hybrid DLC Environment Doctor

Run the doctor **after activating the Conda environment that will execute the
project**. The command-line interface and all diagnostic output are in English.

## 1. Diagnose without changing the environment

```powershell
conda activate YOUR_ENVIRONMENT
python dlc/environment_doctor.py
```

The doctor checks the active interpreter and Conda prefix, the supported Python
range, NVIDIA visibility, PyTorch/CUDA tensor allocation, DeepLabCut Model Zoo,
Ultralytics, PySide6, OpenCV super-resolution support, HDF5 support, and `pip
check`. Diagnostic mode never installs or removes anything.

## 2. Preview a repair

Choose the PyTorch build using the official PyTorch selector. For example, to
preview the CUDA 12.6 plan:

```powershell
python dlc/environment_doctor.py --install --dry-run --torch cu126
```

CPU-only preview:

```powershell
python dlc/environment_doctor.py --install --dry-run --torch cpu
```

If the official selector gives another wheel index, pass it directly:

```powershell
python dlc/environment_doctor.py --install --dry-run --torch-index-url https://download.pytorch.org/whl/CUDA_INDEX
```

## 3. Apply the repair

Remove `--dry-run` and review the commands before answering `y`:

```powershell
python dlc/environment_doctor.py --install --torch cu126
```

For unattended execution, add `--yes`. The script uses the active Python as
`python -m pip`, refuses Conda `base` by default, and does not replace PyTorch
unless a `--torch` build or `--torch-index-url` is explicitly supplied.

OpenCV publishes mutually exclusive wheel variants. If the report says that
`cv2.dnn_superres` is unavailable or lists multiple OpenCV wheels, preview the
explicit cleanup first:

```powershell
python dlc/environment_doctor.py --install --dry-run --fix-opencv
```

Use `--json-report environment-report.json` to keep a machine-readable report.

Exit codes are `0` (ready), `1` (issues remain), `2` (unsafe/unsupported target),
and `3` (an installation command failed).
