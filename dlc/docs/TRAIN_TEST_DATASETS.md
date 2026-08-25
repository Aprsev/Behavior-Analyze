# Training and held-out test videos

The Hybrid Workbench keeps model-development videos separate from evaluation
videos.

- **Training videos** are sampled for traditional pseudo boxes, manually
  reviewed, and exported into the YOLO train/validation dataset. They are also
  the only videos added to an optional supervised DeepLabCut project.
- **Held-out test videos** are never used for pseudo labels or training. They
  run through the trained YOLO detector, super-resolution stage, pretrained or
  fine-tuned DLC model, inverse mapping, trajectory export, and overlay video.

In the Setup tab, use **Add files...** to populate both lists. ROI entries follow
the same order as their videos. One ROI entry is broadcast to every video in
that set; otherwise the number of ROI entries must exactly match the number of
videos.

When at least two training videos exist, the YOLO validation set contains whole
videos rather than random neighboring frames. The selected validation video
names are written to `dataset_manifest.json`. With one training video only, the
workbench falls back to a deterministic frame-level split and records that fact
in the manifest.

After training YOLO, use:

- **Run complete TRAIN-set pipeline** to inspect fit and pipeline integrity.
- **Run complete HELD-OUT TEST pipeline** to inspect generalization on unseen
  videos.

Each video receives an independent output directory under
`output_dir/hybrid/VIDEO_STEM/`, including YOLO detection overlay, enhanced crop
video, transforms, DLC predictions, final trajectory, quality report, and final
source-video overlay.

## Old configuration migration

Configurations created before the `dlc/hybrid/` modular layout used paths
relative to `dlc/`. The loader recognizes the old `videos`/`roi_json` schema and
resolves those paths with the original base. When opened in the GUI, they are
displayed and saved as normalized absolute paths under schema version 2.
