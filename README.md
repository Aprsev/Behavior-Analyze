# Behavior Analyze

Automated single-mouse tracking for a static-camera open-field assay. The
pipeline detects and rectifies the arena, builds a median background, segments
the mouse, estimates body and head positions, converts them to centimetres,
and produces a CSV plus an annotated MP4.

## Usage

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place exactly one input MP4 in the project directory and run:

```powershell
python mouse_behavior_pipeline.py
```

The default input path is intentionally blank and the sole MP4 is discovered
automatically. An explicit file can instead be supplied with `--input`.

Generated outputs are `calibration.json`, `trajectory.csv`,
`annotated_output.mp4`, and `pipeline_log.txt`. These and raw MP4 recordings
are excluded from Git because video/data artifacts may be large or sensitive.

If arena detection needs correction, pass ordered top-left, top-right,
bottom-right, and bottom-left points:

```powershell
python mouse_behavior_pipeline.py --corners "100,60;540,80;520,420;80,400"
```
