#!/usr/bin/env python3
"""Paginated contact-sheet viewer for existing U-Net annotations.

The viewer prefers exact exported training samples (image/mask/head PNGs).
If the dataset has not been rebuilt, it falls back to the polygon CSV and
source videos. Only one page is decoded at a time.
"""
from __future__ import annotations

import argparse
import math
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk
from label_compat import as_bool, normalize_polygon, video_matches


class ContactSheet:
    COLS, ROWS = 5, 4
    PAGE_SIZE = COLS * ROWS
    THUMBNAIL = (220, 145)

    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root, self.args = root, args
        self.dataset = Path(args.dataset)
        self.items = self._load_items()
        self.page = 0
        self.photos: list[ImageTk.PhotoImage] = []
        self.page_var = tk.StringVar()
        self._build()
        self.root.after(10, self.render)

    def _dataset_items(self) -> list[dict]:
        images = self.dataset / "images"
        if not images.is_dir():
            return []
        names: list[str] = []
        manifest = self.dataset / "dataset.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                for video in data.get("videos", []):
                    names.extend(str(name) for name in video.get("samples", []))
            except (OSError, json.JSONDecodeError, TypeError):
                names = []
        if not names:
            names = [path.name for path in sorted(images.glob("*.png"))]
        items = []
        for name in dict.fromkeys(names):
            if not (images / name).is_file() or not (self.dataset / "masks" / name).is_file():
                continue
            match = re.match(r"(.+)_([0-9]{7})\.png$", name)
            key = (match.group(1), int(match.group(2))) if match else (name, -1)
            items.append({"kind": "dataset", "name": name, "key": key})
        return items

    def _raw_items(self) -> list[dict]:
        labels_path = Path(self.args.labels)
        if not labels_path.is_file():
            return []
        try:
            rows = pd.read_csv(labels_path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return []
        if "polygon_px" not in rows or "frame" not in rows:
            return []
        original = rows.copy()
        total = len(rows)
        excluded = int(rows["exclude"].map(as_bool).sum()) if "exclude" in rows else 0
        if "exclude" in rows:
            rows = rows.loc[~rows["exclude"].map(as_bool)]
        keys = [column for column in ("video", "frame") if column in rows]
        rows = rows.loc[rows["polygon_px"].notna()].drop_duplicates(keys, keep="last")
        videos = [Path(path) for path in self.args.video if Path(path).is_file()]
        manifest = self.dataset / "dataset.json"
        if manifest.is_file():
            try:
                for record in json.loads(manifest.read_text(encoding="utf-8")).get("videos", []):
                    path = Path(record.get("video", ""))
                    if path.is_file() and path not in videos:
                        videos.append(path)
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        only_video = videos[0] if len(videos) == 1 else None
        items = []
        invalid = unmatched = 0
        for _, row in rows.iterrows():
            video = next((path for path in videos if video_matches(row.get("video", ""), path)), only_video)
            if video is None:
                unmatched += 1
                continue
            try:
                polygon = normalize_polygon(row.polygon_px).astype(np.int32)
            except (TypeError, ValueError):
                invalid += 1
                continue
            frame = int(row.frame)
            items.append({"kind": "raw", "name": f"{video.name} · frame {frame}",
                          "video": video, "frame": frame, "polygon": polygon,
                          "key": (video.stem.replace(" ", "_"), frame)})
        excluded_rows = rows.iloc[0:0]
        if "exclude" in original:
            excluded_rows = original.loc[original["exclude"].map(as_bool)]
        self.csv_excluded = [(str(row.get("video", "")), int(row.frame))
                             for _, row in excluded_rows.iterrows()]
        self.csv_stats = f"CSV {total}；排除 {excluded}；显示 {len(items)}；无法匹配视频 {unmatched}；格式错误 {invalid}"
        return items

    def _load_items(self) -> list[dict]:
        raw, dataset = self._raw_items(), self._dataset_items()
        if self.args.source == "dataset":
            self.mode = "上次重建的训练数据（绿色=mask，红色=head）"
            return dataset
        if self.args.source == "all":
            raw_keys = {item["key"] for item in raw}
            excluded = getattr(self, "csv_excluded", [])
            historical = [item for item in dataset if item["key"] not in raw_keys and
                          not any(frame == item["key"][1] and video_matches(value, item["key"][0])
                                  for value, frame in excluded)]
            self.mode = "全部标注：当前 CSV 轮廓 + 未迁移旧格式的历史训练样本"
            self.csv_stats += f"；历史补充 {len(historical)}；合计 {len(raw) + len(historical)}"
            return raw + historical
        if raw:
            self.mode = "当前 CSV 原始标注（绿色=轮廓；保存后立即反映）"
            return raw
        self.mode = "当前 CSV 无法读取，回退到上次重建的训练数据"
        return dataset

    def _build(self) -> None:
        self.root.title("已标注数据 · 分页拼接预览")
        self.root.geometry("1220x790")
        self.root.minsize(980, 650)
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=10, pady=8)
        ttk.Label(toolbar, text=self.mode, font=("Microsoft YaHei UI", 11, "bold")).pack(side="left")
        details = getattr(self, "csv_stats", f"共 {len(self.items)} 张")
        ttk.Label(toolbar, text=details, foreground="#555").pack(side="left", padx=12)
        ttk.Button(toolbar, text="上一页  ←", command=lambda: self.move(-1)).pack(side="right", padx=4)
        ttk.Button(toolbar, text="下一页  →", command=lambda: self.move(1)).pack(side="right", padx=4)
        ttk.Label(toolbar, textvariable=self.page_var).pack(side="right", padx=12)
        self.grid = ttk.Frame(self.root)
        self.grid.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        for column in range(self.COLS):
            self.grid.columnconfigure(column, weight=1)
        for row in range(self.ROWS):
            self.grid.rowconfigure(row, weight=1)
        self.root.bind("<Left>", lambda _event: self.move(-1))
        self.root.bind("<Right>", lambda _event: self.move(1))
        self.root.bind("<Prior>", lambda _event: self.move(-1))
        self.root.bind("<Next>", lambda _event: self.move(1))

    def _dataset_image(self, item: dict) -> np.ndarray | None:
        name = item["name"]
        gray = cv2.imread(str(self.dataset / "images" / name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.dataset / "masks" / name), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            return None
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = image.copy()
        overlay[mask > 127] = (35, 205, 90)
        image = cv2.addWeighted(image, 0.68, overlay, 0.32, 0)
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 80), 2)
        head = cv2.imread(str(self.dataset / "heads" / name), cv2.IMREAD_GRAYSCALE)
        if head is not None and head.max() > 0:
            _, _, _, point = cv2.minMaxLoc(head)
            cv2.circle(image, point, 6, (255, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(image, point, 9, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    @staticmethod
    def _raw_image(item: dict, captures: dict[Path, cv2.VideoCapture]) -> np.ndarray | None:
        video = item["video"]
        cap = captures.setdefault(video, cv2.VideoCapture(str(video)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"])
        ok, frame = cap.read()
        if not ok:
            return None
        cv2.polylines(frame, [item["polygon"]], True, (20, 235, 70), 3, cv2.LINE_AA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def render(self) -> None:
        for child in self.grid.winfo_children():
            child.destroy()
        self.photos.clear()
        total_pages = max(1, math.ceil(len(self.items) / self.PAGE_SIZE))
        self.page = min(max(self.page, 0), total_pages - 1)
        self.page_var.set(f"第 {self.page + 1} / {total_pages} 页")
        if not self.items:
            ttk.Label(self.grid, text="没有找到可显示的标注。\n请确认训练数据集目录、标注 CSV 和视频列表。",
                      justify="center", font=("Microsoft YaHei UI", 12)).grid(
                          row=0, column=0, columnspan=self.COLS, sticky="nsew")
            return
        start = self.page * self.PAGE_SIZE
        captures: dict[Path, cv2.VideoCapture] = {}
        try:
            for index, item in enumerate(self.items[start:start + self.PAGE_SIZE]):
                image = (self._dataset_image(item) if item["kind"] == "dataset"
                         else self._raw_image(item, captures))
                cell = ttk.Frame(self.grid, relief="solid", borderwidth=1)
                cell.grid(row=index // self.COLS, column=index % self.COLS,
                          sticky="nsew", padx=3, pady=3)
                if image is None:
                    ttk.Label(cell, text="读取失败").pack(expand=True)
                    continue
                pil = Image.fromarray(image)
                pil.thumbnail(self.THUMBNAIL, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                self.photos.append(photo)
                ttk.Label(cell, image=photo).pack(expand=True, padx=2, pady=(2, 0))
                title = item["name"]
                if len(title) > 38:
                    title = title[:17] + "…" + title[-18:]
                ttk.Label(cell, text=title, font=("Microsoft YaHei UI", 8)).pack(pady=(0, 2))
        finally:
            for cap in captures.values():
                cap.release()

    def move(self, delta: int) -> None:
        total_pages = max(1, math.ceil(len(self.items) / self.PAGE_SIZE))
        new_page = min(max(self.page + delta, 0), total_pages - 1)
        if new_page != self.page:
            self.page = new_page
            self.render()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--source", choices=("all", "csv", "dataset"), default="all")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        ContactSheet(root, args)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("标注预览失败", str(exc))
        raise


if __name__ == "__main__":
    main()
