from __future__ import annotations

from pathlib import Path
from typing import List

import ffmpeg
import cv2
import numpy as np


def extract_keyframes_ffmpeg(
    video_path: Path,
    output_dir: Path,
    max_fps: float = 1.0,
    scene_threshold: float = 0.45,
    method: str = "scene",  # scene | iframe | interval
    interval_sec: float = 5.0,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "iframe":
        vf = "select='eq(pict_type\\,I)'"
    elif method == "interval":
        eff_interval = interval_sec if interval_sec and interval_sec > 0 else 5.0
        vf = f"fps=1/{eff_interval}"
    else:  # scene
        vf = f"select='gt(scene,{scene_threshold})',fps={max_fps}"

    pattern = str(output_dir / "frame_%06d.jpg")

    (
        ffmpeg
        .input(str(video_path))
        .output(
            pattern,
            vf=vf,
            vsync="vfr",
            **{"qscale:v": 2},
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return sorted(output_dir.glob("frame_*.jpg"))


def _compute_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _resize_if_needed(frame_bgr: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    scale = max_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def extract_keyframes_opencv(
    video_path: Path,
    output_dir: Path,
    max_fps: float = 1.0,
    scene_threshold: float = 0.45,
    method: str = "scene",
    interval_sec: float = 5.0,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if method == "interval":
        eff_interval = interval_sec if interval_sec and interval_sec > 0 else 5.0
        step = max(1, int(fps * eff_interval))
        frame_idx = 0
        saved_paths: List[Path] = []
        keyframe_idx = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            if frame_idx % step == 0:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    frame_resized = _resize_if_needed(frame)
                    out_path = output_dir / f"frame_{keyframe_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    saved_paths.append(out_path)
                    keyframe_idx += 1
            frame_idx += 1
        cap.release()
        return saved_paths

    # Default to scene detection for OpenCV path
    frame_interval = max(1, int(fps / max_fps))

    saved_paths: List[Path] = []
    prev_hist = None
    frame_idx = 0
    keyframe_idx = 0

    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            frame_idx += 1
            continue

        hist = _compute_histogram(frame)
        diff = 1.0
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = 1.0 - float(corr)

        if prev_hist is None or diff >= scene_threshold:
            frame_resized = _resize_if_needed(frame)
            out_path = output_dir / f"frame_{keyframe_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved_paths.append(out_path)
            keyframe_idx += 1
            prev_hist = hist

        frame_idx += 1

    cap.release()
    return saved_paths


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    max_fps: float = 1.0,
    scene_threshold: float = 0.45,
    method: str = "scene",
    interval_sec: float = 5.0,
) -> List[Path]:
    try:
        return extract_keyframes_ffmpeg(
            video_path,
            output_dir,
            max_fps=max_fps,
            scene_threshold=scene_threshold,
            method=method,
            interval_sec=interval_sec,
        )
    except Exception:
        return extract_keyframes_opencv(
            video_path,
            output_dir,
            max_fps=max_fps,
            scene_threshold=scene_threshold,
            method=method,
            interval_sec=interval_sec,
        )
