# DeepLabCut mouse tracking

The DLC implementation is organized as a Python package. The primary workflow
uses background-based pseudo boxes, reviewed YOLO detection, optional
super-resolution, and DeepLabCut SuperAnimal pose estimation.

## Layout

```text
dlc/
├── hybrid/        Primary English GUI, jobs, box editor, and pipeline
├── legacy/        Original direct-DLC workflow retained for compatibility
├── tools/         Environment diagnostics and installation helper
├── configs/       Reserved for shared configuration schemas
├── requirements/  Reproducible dependency groups
├── docs/          Workflow and environment documentation
├── tests/         Unit tests
├── hybrid_gui.py          Stable GUI launcher
└── environment_doctor.py  Stable environment-doctor launcher
```

Implementation code belongs in the workflow subpackages. The two root launcher
scripts intentionally remain stable so commands already used on the GPU machine
continue to work after a Git pull. Small root compatibility modules preserve old
Python imports while downstream code migrates to the package paths.

## Quick start

Activate a Python 3.10-3.12 Conda environment, then run:

```powershell
python dlc/environment_doctor.py
python dlc/environment_doctor.py --install --dry-run --torch cu126
python dlc/environment_doctor.py --install --torch cu126
python dlc/hybrid_gui.py
```

The GUI can also be launched as a module:

```powershell
python -m dlc
```

See [the Hybrid workflow](docs/README_HYBRID.md) and
[environment setup](docs/ENVIRONMENT_SETUP.md) for details.

## Tests

```powershell
python -m unittest discover -s dlc/tests -p "test_*.py" -v
```

The pipeline tests require NumPy, pandas, and OpenCV but do not download DLC
weights.
