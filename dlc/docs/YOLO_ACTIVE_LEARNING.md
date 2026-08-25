# YOLO active-learning workflow

This workflow reuses previous labels and a previous `best.pt` while adding a
small, information-rich set of frames from new videos.

## Supported imports

The **Existing labeled dataset** selector accepts either:

- a native Hybrid Workbench dataset containing `box_labels.csv`; or
- a standard one-class YOLO detection dataset containing `images/` and
  `labels/` subdirectories.

Imported images are copied into the current dataset and deduplicated by SHA-256
content hash. Native manual-review metadata is retained. For standard YOLO
labels, the largest class-0 box is used because this project assumes one mouse
per frame.

## Low-confidence mining

Select the previous checkpoint in **Trained best.pt**, add new unlabeled videos,
and choose:

- candidate frames scored per video;
- total lowest-confidence frames to review;
- minimum time spacing between selected frames; and
- minimum prediction confidence used during candidate inference.

Missing detections receive confidence zero and are selected first. Temporal
spacing reduces near-duplicate adjacent frames. The selected images are added to
`box_labels.csv` with a timestamped review batch. Original model confidence is
stored separately and remains available after a manual box correction.

## Sequential correction

The review window displays one full-size frame at a time. Drag inside a box to
move it, drag a corner to resize it, or drag outside to replace/create it. Use
**Save + Next** for continuous review or **Save + Close** to stop and resume
later. Frames without a usable mouse image can be explicitly excluded.

## Fine-tuning

After review, export the combined dataset and click **Fine-tune old best.pt**.
The workbench loads the old checkpoint as initial weights and starts a new
training run on the expanded dataset. It deliberately does not use
`resume=True`, because resume restores the old optimizer, epoch, and training
configuration rather than starting a new optimization run over changed data.
The new `best.pt` is automatically written back to the GUI configuration; the
old checkpoint and run directory remain untouched.
