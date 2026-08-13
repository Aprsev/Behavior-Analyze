# Code layout

`behavior_analyze/` is the shared implementation package:

- `geometry.py`: video ingestion, arena geometry, coordinate conversion
- `segmentation.py`: static-background mouse/body segmentation
- `head_tracking.py`: silhouette and reflection anchor tracking
- `calibration_model.py`: manually calibrated head-model features

The root-level Python files remain supported command-line entry points. They
will be moved into `code/` progressively without changing existing commands.
