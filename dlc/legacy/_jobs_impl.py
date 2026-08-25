#!/usr/bin/env python3
"""Subprocess jobs used by the DeepLabCut Workbench GUI.

Keeping every DLC call in a fresh process prevents CUDA state and interactive
labeling windows from freezing the main Qt event loop. This file is also a
reproducible CLI: ``python dlc/jobs.py ACTION --config dlc/config.json``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dlc.postprocess import postprocess_predictions


TOPVIEWMOUSE_BODYPARTS = [
    "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
    "mid_backend", "mid_backend2", "mid_backend3", "tail_base", "tail1",
    "tail2", "tail3", "tail4", "tail5", "left_shoulder", "left_midside",
    "left_hip", "right_shoulder", "right_midside", "right_hip", "tail_end",
    "head_midpoint",
]


def load_settings(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    path_keys = ("video", "roi_json", "output_dir", "project_config", "working_directory")
    for key in path_keys:
        if cfg.get(key):
            value = Path(cfg[key]).expanduser()
            cfg[key] = str(value if value.is_absolute() else (base / value).resolve())
    videos = cfg.get("videos") or ([cfg["video"]] if cfg.get("video") else [])
    cfg["videos"] = [str((Path(v).expanduser() if Path(v).is_absolute() else (base / v).resolve())) for v in videos]
    return cfg


def advanced(cfg: dict, name: str) -> dict:
    value = cfg.get("advanced", {}).get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"advanced.{name} must be a JSON object")
    return value


def require_project(cfg: dict) -> str:
    path = Path(cfg.get("project_config", ""))
    if not path.is_file():
        raise FileNotFoundError("DLC project config.yaml is not set or does not exist")
    return str(path)


def require_videos(cfg: dict) -> list[str]:
    videos = [str(Path(v)) for v in cfg.get("videos", [])]
    missing = [v for v in videos if not Path(v).is_file()]
    if not videos:
        raise ValueError("No videos configured")
    if missing:
        raise FileNotFoundError("Missing videos: " + ", ".join(missing))
    return videos


def emit_result(**values: Any) -> None:
    print("DLC_GUI_RESULT " + json.dumps(values, ensure_ascii=False), flush=True)


def dlc_module():
    try:
        import deeplabcut
    except ImportError as exc:
        raise RuntimeError("Install dlc/requirements-dlc.txt in this Python environment") from exc
    return deeplabcut


def job_check(cfg: dict) -> None:
    print(f"Python {sys.version}")
    for name in ("deeplabcut", "torch", "torchvision", "pandas", "opencv-python", "tables", "PySide6"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: MISSING")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    for video in cfg.get("videos", []):
        print(f"video {'OK' if Path(video).is_file() else 'MISSING'}: {video}")
    if cfg.get("project_config"):
        print(f"project {'OK' if Path(cfg['project_config']).is_file() else 'MISSING'}: {cfg['project_config']}")
    if cfg.get("roi_json"):
        print(f"ROI {'OK' if Path(cfg['roi_json']).is_file() else 'MISSING'}: {cfg['roi_json']}")


def job_zero_shot(cfg: dict) -> None:
    from deeplabcut.modelzoo.video_inference import video_inference_superanimal

    model = cfg.get("model", {})
    kwargs = {
        "videos": require_videos(cfg),
        "superanimal_name": model.get("superanimal_name", "superanimal_topviewmouse"),
        "model_name": model.get("model_name", "hrnet_w32"),
        "detector_name": model.get("detector_name", "fasterrcnn_resnet50_fpn_v2"),
        "video_adapt": bool(model.get("video_adapt", True)),
        "max_individuals": int(model.get("max_individuals", 1)),
        "batch_size": int(model.get("batch_size", 4)),
        "detector_batch_size": int(model.get("detector_batch_size", 1)),
        "pcutoff": float(model.get("inference_pcutoff", 0.1)),
        "pseudo_threshold": float(model.get("pseudo_threshold", 0.1)),
        "bbox_threshold": float(model.get("bbox_threshold", 0.9)),
        "detector_epochs": int(model.get("detector_epochs", 1)),
        "pose_epochs": int(model.get("pose_epochs", 1)),
        "video_adapt_batch_size": int(model.get("video_adapt_batch_size", 4)),
        "device": model.get("device", "auto"),
        "plot_bboxes": bool(model.get("plot_bboxes", True)),
        "create_labeled_video": bool(model.get("create_labeled_video", True)),
        "dest_folder": str(Path(cfg.get("output_dir", ROOT / "results" / "dlc_gui")).resolve()),
    }
    kwargs.update(advanced(cfg, "zero_shot"))
    Path(kwargs["dest_folder"]).mkdir(parents=True, exist_ok=True)
    video_inference_superanimal(**kwargs)


def _find_prediction(output_dir: Path, video: Path) -> Path:
    candidates = [p for p in output_dir.glob(f"{video.stem}*.h5") if "_3d" not in p.stem and p.name != "dlc_keypoints.h5"]
    if not candidates:
        raise FileNotFoundError(f"No prediction HDF5 for {video.name} in {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _postprocess(cfg: dict) -> None:
    videos = require_videos(cfg)
    roi = Path(cfg.get("roi_json", ""))
    if not roi.is_file():
        raise FileNotFoundError("A four-corner ROI JSON is required for centimetre output")
    output = Path(cfg.get("output_dir", ROOT / "results" / "dlc_gui")).resolve()
    post = cfg.get("postprocess", {})
    for video_name in videos:
        video = Path(video_name)
        h5 = _find_prediction(output, video)
        destination = output / video.stem
        print(f"Post-processing {video.name} from {h5.name}", flush=True)
        postprocess_predictions(
            predictions_h5=h5,
            video=video,
            roi_json=roi,
            output_dir=destination,
            width_cm=float(cfg.get("arena_width_cm", 25.0)),
            height_cm=float(cfg.get("arena_height_cm", 30.0)),
            pcutoff=float(post.get("pcutoff", 0.35)),
            max_gap_sec=float(post.get("max_gap_sec", 0.2)),
            median_window=int(post.get("median_window", 3)),
            make_overlay=bool(post.get("write_overlay", True)),
            draw_all_keypoints=bool(post.get("draw_all_keypoints", True)),
        )


def job_postprocess(cfg: dict) -> None:
    _postprocess(cfg)


def job_create_project(cfg: dict) -> None:
    dlc = dlc_module()
    project = cfg.get("project", {})
    config_path = dlc.create_new_project(
        project.get("task", "mouse_occlusion"),
        project.get("experimenter", "researcher"),
        require_videos(cfg),
        working_directory=str(Path(cfg.get("working_directory", ROOT / "dlc_projects")).resolve()),
        copy_videos=bool(project.get("copy_videos", False)),
        multianimal=False,
        **advanced(cfg, "create_project"),
    )
    from deeplabcut.utils import auxiliaryfunctions
    bodyparts = project.get("bodyparts") or TOPVIEWMOUSE_BODYPARTS
    auxiliaryfunctions.edit_config(config_path, {
        "bodyparts": bodyparts,
        "engine": "pytorch",
        "TrainingFraction": [float(project.get("training_fraction", 0.9))],
        "numframes2pick": int(project.get("num_frames", 40)),
        "pcutoff": float(cfg.get("postprocess", {}).get("pcutoff", 0.35)),
    })
    print(f"Created project: {config_path}", flush=True)
    emit_result(project_config=str(Path(config_path).resolve()))


def job_add_videos(cfg: dict) -> None:
    dlc_module().add_new_videos(
        require_project(cfg), require_videos(cfg),
        copy_videos=bool(cfg.get("project", {}).get("copy_videos", False)),
        **advanced(cfg, "add_videos"),
    )


def job_extract_frames(cfg: dict) -> None:
    data = cfg.get("dataset", {})
    kwargs = {
        "mode": data.get("mode", "automatic"),
        "algo": data.get("algorithm", "kmeans"),
        "crop": bool(data.get("crop", False)),
        "cluster_step": int(data.get("cluster_step", 1)),
        "cluster_resizewidth": int(data.get("cluster_resize_width", 30)),
        "cluster_color": bool(data.get("cluster_color", False)),
        "opencv": True,
        "userfeedback": False,
    }
    kwargs.update(advanced(cfg, "extract_frames"))
    dlc_module().extract_frames(require_project(cfg), **kwargs)


def job_label_frames(cfg: dict) -> None:
    dlc_module().label_frames(require_project(cfg), **advanced(cfg, "label_frames"))


def job_check_labels(cfg: dict) -> None:
    dlc_module().check_labels(require_project(cfg), **advanced(cfg, "check_labels"))


def job_build_dataset(cfg: dict) -> None:
    dlc = dlc_module()
    from deeplabcut.modelzoo import build_weight_init
    from deeplabcut.utils import auxiliaryfunctions
    model, training = cfg.get("model", {}), cfg.get("training", {})
    config = require_project(cfg)
    project_cfg = auxiliaryfunctions.read_config(config)
    weight_init = build_weight_init(
        cfg=project_cfg,
        super_animal=model.get("superanimal_name", "superanimal_topviewmouse"),
        model_name=model.get("model_name", "hrnet_w32"),
        detector_name=model.get("detector_name", "fasterrcnn_resnet50_fpn_v2"),
        with_decoder=False,
    )
    kwargs = {
        "Shuffles": [int(training.get("shuffle", 1))],
        "net_type": f"top_down_{model.get('model_name', 'hrnet_w32')}",
        "detector_type": model.get("detector_name", "fasterrcnn_resnet50_fpn_v2"),
        "weight_init": weight_init,
        "engine": dlc.Engine.PYTORCH,
        "userfeedback": False,
    }
    kwargs.update(advanced(cfg, "build_dataset"))
    dlc.create_training_dataset(config, **kwargs)


def job_train(cfg: dict) -> None:
    training = cfg.get("training", {})
    kwargs = {
        "shuffle": int(training.get("shuffle", 1)),
        "device": training.get("device", "auto"),
        "batch_size": int(training.get("batch_size", 8)),
        "epochs": int(training.get("epochs", 100)),
        "save_epochs": int(training.get("save_epochs", 10)),
        "detector_batch_size": int(training.get("detector_batch_size", 2)),
        "detector_epochs": int(training.get("detector_epochs", 0)),
        "display_iters": int(training.get("display_iters", 20)),
        "max_snapshots_to_keep": int(training.get("max_snapshots", 5)),
    }
    kwargs.update(advanced(cfg, "train"))
    dlc_module().train_network(require_project(cfg), **kwargs)


def job_evaluate(cfg: dict) -> None:
    training = cfg.get("training", {})
    kwargs = {"Shuffles": [int(training.get("shuffle", 1))], "plotting": bool(training.get("evaluation_plots", True))}
    kwargs.update(advanced(cfg, "evaluate"))
    dlc_module().evaluate_network(require_project(cfg), **kwargs)


def job_analyze(cfg: dict) -> None:
    inference, training = cfg.get("inference", {}), cfg.get("training", {})
    output = Path(cfg.get("output_dir", ROOT / "results" / "dlc_gui")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "shuffle": int(training.get("shuffle", 1)),
        "device": inference.get("device", "auto"),
        "destfolder": str(output),
        "batch_size": int(inference.get("batch_size", 4)),
        "detector_batch_size": int(inference.get("detector_batch_size", 1)),
        "save_as_csv": bool(inference.get("save_as_csv", True)),
        "overwrite": bool(inference.get("overwrite", False)),
    }
    kwargs.update(advanced(cfg, "analyze"))
    dlc_module().analyze_videos(require_project(cfg), require_videos(cfg), **kwargs)


def job_filter_predictions(cfg: dict) -> None:
    training = cfg.get("training", {})
    kwargs = {"shuffle": int(training.get("shuffle", 1)), "destfolder": str(Path(cfg["output_dir"]).resolve())}
    kwargs.update(advanced(cfg, "filter_predictions"))
    dlc_module().filterpredictions(require_project(cfg), require_videos(cfg), **kwargs)


def job_create_labeled_video(cfg: dict) -> None:
    training, inference = cfg.get("training", {}), cfg.get("inference", {})
    kwargs = {
        "shuffle": int(training.get("shuffle", 1)),
        "destfolder": str(Path(cfg["output_dir"]).resolve()),
        "filtered": bool(inference.get("use_filtered", False)),
        "pcutoff": float(inference.get("pcutoff", 0.35)),
        "draw_skeleton": bool(inference.get("draw_skeleton", True)),
        "trailpoints": int(inference.get("trailpoints", 0)),
        "plot_bboxes": bool(inference.get("plot_bboxes", True)),
        "overwrite": bool(inference.get("overwrite", False)),
    }
    kwargs.update(advanced(cfg, "labeled_video"))
    dlc_module().create_labeled_video(require_project(cfg), require_videos(cfg), **kwargs)


def job_plot_trajectories(cfg: dict) -> None:
    training, inference = cfg.get("training", {}), cfg.get("inference", {})
    kwargs = {
        "shuffle": int(training.get("shuffle", 1)),
        "destfolder": str(Path(cfg["output_dir"]).resolve()),
        "filtered": bool(inference.get("use_filtered", False)),
    }
    kwargs.update(advanced(cfg, "plot_trajectories"))
    dlc_module().plot_trajectories(require_project(cfg), require_videos(cfg), **kwargs)


def job_extract_outliers(cfg: dict) -> None:
    tuning, training = cfg.get("tuning", {}), cfg.get("training", {})
    kwargs = {
        "shuffle": int(training.get("shuffle", 1)),
        "outlieralgorithm": tuning.get("outlier_algorithm", "jump"),
        "epsilon": float(tuning.get("epsilon", 20)),
        "p_bound": float(tuning.get("p_bound", 0.1)),
        "extractionalgorithm": tuning.get("extraction_algorithm", "kmeans"),
        "automatic": bool(tuning.get("automatic", True)),
        "destfolder": str(Path(cfg["output_dir"]).resolve()),
    }
    kwargs.update(advanced(cfg, "extract_outliers"))
    dlc_module().extract_outlier_frames(require_project(cfg), require_videos(cfg), **kwargs)


def job_refine_labels(cfg: dict) -> None:
    dlc_module().refine_labels(require_project(cfg), **advanced(cfg, "refine_labels"))


def job_merge_datasets(cfg: dict) -> None:
    dlc_module().merge_datasets(require_project(cfg), **advanced(cfg, "merge_datasets"))


JOBS: dict[str, Callable[[dict], None]] = {
    "check": job_check,
    "zero_shot": job_zero_shot,
    "postprocess": job_postprocess,
    "create_project": job_create_project,
    "add_videos": job_add_videos,
    "extract_frames": job_extract_frames,
    "label_frames": job_label_frames,
    "check_labels": job_check_labels,
    "build_dataset": job_build_dataset,
    "train": job_train,
    "evaluate": job_evaluate,
    "analyze": job_analyze,
    "filter_predictions": job_filter_predictions,
    "create_labeled_video": job_create_labeled_video,
    "plot_trajectories": job_plot_trajectories,
    "extract_outliers": job_extract_outliers,
    "refine_labels": job_refine_labels,
    "merge_datasets": job_merge_datasets,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(JOBS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cfg = load_settings(args.config)
    print(f"=== {args.action} ===", flush=True)
    JOBS[args.action](cfg)
    print(f"=== {args.action} completed ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
