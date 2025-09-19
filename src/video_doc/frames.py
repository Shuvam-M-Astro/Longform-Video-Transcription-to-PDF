from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Callable, Sequence

import ffmpeg
import cv2
import numpy as np
from tqdm import tqdm
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED


def _probe_duration_seconds(video_path: Path) -> float | None:
    try:
        info = ffmpeg.probe(str(video_path))
        if not info:
            return None
        streams = info.get("streams", [])
        for stream in streams:
            if stream.get("codec_type") == "video":
                # duration may be on format or stream
                dur = stream.get("duration")
                if dur is not None:
                    return float(dur)
        fmt = info.get("format", {})
        if "duration" in fmt:
            return float(fmt["duration"])
    except Exception:
        return None
    return None


def extract_keyframes_ffmpeg(
    video_path: Path,
    output_dir: Path,
    max_fps: float = 1.0,
    scene_threshold: float = 0.45,
    method: str = "scene",  # scene | iframe | interval
    interval_sec: float = 5.0,
    output_format: str = "jpg",
    jpeg_quality: int = 90,
    max_width: int = 1280,
    max_frames: Optional[int] = None,
    *,
    progress_cb: Optional[Callable[[float], None]] = None,
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

    # Optional resize to cap width
    if max_width and max_width > 0:
        # Escape comma inside min(iw,MAX) expression
        vf = f"{vf},scale='min(iw\\,{max_width})':-2"

    # Normalize output format
    ext = output_format.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in {"jpg", "png", "webp"}:
        ext = "jpg"

    pattern = str(output_dir / f"frame_%06d.{ext}")

    # Try live-progress run; if anything fails, fallback to quiet run
    try:
        total_seconds = _probe_duration_seconds(video_path) or 0.0
        print(
            f"[frames] ffmpeg: method={method} vf=\"{vf}\" max_fps={max_fps} "
            f"scene_th={scene_threshold} interval={interval_sec}s duration≈{total_seconds:.1f}s",
            flush=True,
        )
        # Map jpeg_quality (0-100) to qscale (2-31, lower is better)
        out_kwargs = {"vf": vf, "vsync": "vfr"}
        if ext == "jpg":
            qscale = max(2, min(31, int(round(31 - (jpeg_quality / 100.0) * 29))))
            out_kwargs["qscale:v"] = qscale
        if max_frames and max_frames > 0:
            out_kwargs["vframes"] = int(max_frames)

        stream = (
            ffmpeg
            .input(str(video_path))
            .output(
                pattern,
                **out_kwargs,
            )
            .overwrite_output()
            .global_args("-progress", "pipe:1", "-nostats")
        )

        process = stream.run_async(pipe_stdout=True, pipe_stderr=False)

        progress_bar = None
        if total_seconds > 0:
            progress_bar = tqdm(total=total_seconds, unit="s", desc="Keyframes", leave=False)

        out_time_regex = re.compile(rb"out_time_ms=(\d+)")
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                if progress_bar is not None:
                    m = out_time_regex.search(line)
                    if m:
                        ms = int(m.group(1))
                        secs = ms / 1000000.0
                        # Clamp monotonic increase
                        current = progress_bar.n
                        if secs > current:
                            progress_bar.update(secs - current)
                            if progress_cb and total_seconds > 0:
                                pct = max(0.0, min(100.0, (secs / total_seconds) * 100.0))
                                progress_cb(pct)
        finally:
            process.wait()
            if progress_bar is not None:
                # Ensure bar completes
                if progress_bar.n < progress_bar.total:
                    progress_bar.update(progress_bar.total - progress_bar.n)
                progress_bar.close()
            if progress_cb:
                progress_cb(100.0)
    except Exception:
        print("[frames] ffmpeg live-progress unavailable; running without live progress...", flush=True)
        out_kwargs = {"vf": vf, "vsync": "vfr"}
        if ext == "jpg":
            qscale = max(2, min(31, int(round(31 - (jpeg_quality / 100.0) * 29))))
            out_kwargs["qscale:v"] = qscale
        if max_frames and max_frames > 0:
            out_kwargs["vframes"] = int(max_frames)
        (
            ffmpeg
            .input(str(video_path))
            .output(
                pattern,
                **out_kwargs,
            )
            .overwrite_output()
            .run(quiet=True)
        )

    return sorted(output_dir.glob(f"frame_*.{ext}"))


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


def _is_dark(frame_bgr: np.ndarray, *, value_threshold: int = 16, ratio_threshold: float = 0.98) -> bool:
    """Return True if the frame is mostly dark.

    value_threshold: pixels with V (HSV) below this [0-255] are considered dark
    ratio_threshold: if >= this ratio of pixels are dark, consider frame dark
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark = (v < value_threshold)
    ratio = float(np.count_nonzero(dark)) / float(v.size)
    return ratio >= ratio_threshold


def extract_keyframes_opencv(
    video_path: Path,
    output_dir: Path,
    max_fps: float = 1.0,
    scene_threshold: float = 0.45,
    method: str = "scene",
    interval_sec: float = 5.0,
    output_format: str = "jpg",
    jpeg_quality: int = 90,
    max_width: int = 1280,
    max_frames: Optional[int] = None,
    skip_dark: bool = False,
    dark_pixel_value: int = 16,
    dark_ratio_threshold: float = 0.98,
    dedupe: bool = False,
    dedupe_similarity: float = 0.995,
    *,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(
        f"[frames] opencv: method={method} fps={fps:.2f} frames={frame_count} "
        f"max_fps={max_fps} scene_th={scene_threshold} interval={interval_sec}s",
        flush=True,
    )

    # Normalize output format and writer params
    ext = output_format.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in {"jpg", "png", "webp"}:
        ext = "jpg"
    write_params = []
    if ext == "jpg":
        write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(0, min(100, jpeg_quality)))]
    elif ext == "webp":
        write_params = [int(cv2.IMWRITE_WEBP_QUALITY), int(max(1, min(100, jpeg_quality)))]
    elif ext == "png":
        # Map quality 0-100 to PNG compression 9-0 (higher quality -> lower compression)
        compression = int(round((100 - max(0, min(100, jpeg_quality))) / 100.0 * 9))
        write_params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(max(0, min(9, compression)))]

    # Thread pool for parallel disk writes (CPU-bound IO) to speed up saving frames
    env_threads = os.environ.get("FRAME_IO_THREADS")
    try:
        max_io_workers = int(env_threads) if env_threads else min(8, max(2, (os.cpu_count() or 2)))
    except Exception:
        max_io_workers = min(8, max(2, (os.cpu_count() or 2)))
    io_executor = ThreadPoolExecutor(max_workers=max_io_workers)
    # Backlog control
    max_backlog = max_io_workers * 3
    pending: List = []  # list of (future, path)

    if method == "interval":
        eff_interval = interval_sec if interval_sec and interval_sec > 0 else 5.0
        step = max(1, int(fps * eff_interval))
        frame_idx = 0
        saved_paths: List[Path] = []
        keyframe_idx = 0
        last_saved_hist = None
        expected = frame_count // step if frame_count > 0 else None
        bar = tqdm(total=expected, desc="Keyframes", unit="frame", leave=False)
        while True:
            ret = cap.grab()
            if not ret:
                break
            if frame_idx % step == 0:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    if skip_dark and _is_dark(frame, value_threshold=dark_pixel_value, ratio_threshold=dark_ratio_threshold):
                        # Skip dark frames
                        pass
                    else:
                        if dedupe:
                            hist = _compute_histogram(frame)
                            if last_saved_hist is not None:
                                corr = cv2.compareHist(last_saved_hist, hist, cv2.HISTCMP_CORREL)
                                if float(corr) >= dedupe_similarity:
                                    # Too similar to last saved; skip
                                    pass
                                else:
                                    last_saved_hist = hist
                            else:
                                last_saved_hist = hist
                        frame_resized = _resize_if_needed(frame, max_width=max_width)
                        out_path = output_dir / f"frame_{keyframe_idx:06d}.{ext}"
                        # Offload disk write to thread pool
                        fut = io_executor.submit(cv2.imwrite, str(out_path), frame_resized, write_params)
                        pending.append((fut, out_path))
                        # Backpressure to cap memory/backlog
                        if len(pending) >= max_backlog:
                            done, _ = wait([f for f, _ in pending], return_when=FIRST_COMPLETED)
                            # harvest completed
                            still_pending = []
                            for f, pth in pending:
                                if f.done():
                                    try:
                                        ok = bool(f.result())
                                        if ok:
                                            saved_paths.append(pth)
                                    except Exception:
                                        pass
                                else:
                                    still_pending.append((f, pth))
                            pending = still_pending
                        keyframe_idx += 1
                        bar.update(1)
                        if progress_cb and expected:
                            pct = max(0.0, min(100.0, (bar.n / expected) * 100.0))
                            progress_cb(pct)
                        if max_frames and len(saved_paths) >= max_frames:
                            break
            frame_idx += 1
        cap.release()
        # Ensure all pending writes complete before returning paths
        for f, pth in pending:
            try:
                ok = bool(f.result())
                if ok:
                    saved_paths.append(pth)
            except Exception:
                pass
        io_executor.shutdown(wait=True)
        bar.close()
        if progress_cb:
            progress_cb(100.0)
        return saved_paths

    # Default to scene detection for OpenCV path
    frame_interval = max(1, int(fps / max_fps))

    saved_paths: List[Path] = []
    prev_hist = None
    last_saved_hist = None
    frame_idx = 0
    keyframe_idx = 0
    expected = (frame_count // frame_interval) if frame_count > 0 else None
    bar = tqdm(total=expected, desc="Keyframes", unit="frame", leave=False)

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
            if skip_dark and _is_dark(frame, value_threshold=dark_pixel_value, ratio_threshold=dark_ratio_threshold):
                # Skip dark frames
                pass
            else:
                if dedupe and last_saved_hist is not None:
                    corr_saved = cv2.compareHist(last_saved_hist, hist, cv2.HISTCMP_CORREL)
                    if float(corr_saved) >= dedupe_similarity:
                        # Too similar to last saved; skip
                        pass
                    else:
                        last_saved_hist = hist
                else:
                    last_saved_hist = hist
                frame_resized = _resize_if_needed(frame, max_width=max_width)
                out_path = output_dir / f"frame_{keyframe_idx:06d}.{ext}"
                # Offload disk write to thread pool
                fut = io_executor.submit(cv2.imwrite, str(out_path), frame_resized, write_params)
                pending.append((fut, out_path))
                # Backpressure to cap memory/backlog
                if len(pending) >= max_backlog:
                    done, _ = wait([f for f, _ in pending], return_when=FIRST_COMPLETED)
                    still_pending = []
                    for f, pth in pending:
                        if f.done():
                            try:
                                ok = bool(f.result())
                                if ok:
                                    saved_paths.append(pth)
                            except Exception:
                                pass
                        else:
                            still_pending.append((f, pth))
                    pending = still_pending
                keyframe_idx += 1
                prev_hist = hist
                if max_frames and len(saved_paths) >= max_frames:
                    break

        frame_idx += 1
        if expected is not None:
            bar.update(1)
            if progress_cb and expected:
                pct = max(0.0, min(100.0, (bar.n / expected) * 100.0))
                progress_cb(pct)

    cap.release()
    # Ensure all pending writes complete before returning paths
    for f, pth in pending:
        try:
            ok = bool(f.result())
            if ok:
                saved_paths.append(pth)
        except Exception:
            pass
    io_executor.shutdown(wait=True)
    bar.close()
    if progress_cb:
        progress_cb(100.0)
    return saved_paths


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    max_fps: float = 1.0,
    scene_threshold: float = 0.45,
    method: str = "scene",
    interval_sec: float = 5.0,
    output_format: str = "jpg",
    jpeg_quality: int = 90,
    max_width: int = 1280,
    max_frames: Optional[int] = None,
    skip_dark: bool = False,
    dark_pixel_value: int = 16,
    dark_ratio_threshold: float = 0.98,
    dedupe: bool = False,
    dedupe_similarity: float = 0.995,
    *,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> List[Path]:
    # Prefer ffmpeg path unless dark-skip or dedupe is requested (handled in OpenCV).
    prefer_opencv = skip_dark or dedupe
    if not prefer_opencv:
        try:
            return extract_keyframes_ffmpeg(
                video_path,
                output_dir,
                max_fps=max_fps,
                scene_threshold=scene_threshold,
                method=method,
                interval_sec=interval_sec,
                output_format=output_format,
                jpeg_quality=jpeg_quality,
                max_width=max_width,
                max_frames=max_frames,
                progress_cb=progress_cb,
            )
        except Exception:
            pass

    return extract_keyframes_opencv(
        video_path,
        output_dir,
        max_fps=max_fps,
        scene_threshold=scene_threshold,
        method=method,
        interval_sec=interval_sec,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
        max_width=max_width,
        max_frames=max_frames,
        skip_dark=skip_dark,
        dark_pixel_value=dark_pixel_value,
        dark_ratio_threshold=dark_ratio_threshold,
        dedupe=dedupe,
        dedupe_similarity=dedupe_similarity,
        progress_cb=progress_cb,
    )


def build_contact_sheet(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    columns: int = 5,
    thumb_width: int = 320,
    padding: int = 10,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    title: Optional[str] = None,
    title_height: int = 0,
) -> Path:
    """Create a contact sheet image from a list of image paths.

    Returns the path to the generated contact sheet.
    """
    paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not paths:
        raise ValueError("No valid image paths provided")
    columns = max(1, int(columns))
    thumb_width = max(16, int(thumb_width))
    padding = max(0, int(padding))

    # Load and resize thumbs keeping aspect ratio (parallelized IO + resize)
    thumbs: List[np.ndarray] = []
    max_thumb_height = 0
    def _load_and_resize(p: Path):
        img = cv2.imread(str(p))
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = thumb_width / float(w)
        new_size = (thumb_width, int(round(h * scale)))
        thumb = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return thumb

    # Allow overriding contact sheet parallelism via env var
    cs_env = os.environ.get("CONTACT_SHEET_THREADS")
    try:
        max_workers = int(cs_env) if cs_env else min(8, max(2, (os.cpu_count() or 2)))
    except Exception:
        max_workers = min(8, max(2, (os.cpu_count() or 2)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_and_resize, p) for p in paths]
        for fut in futures:
            try:
                thumb = fut.result()
                if thumb is None:
                    continue
                thumbs.append(thumb)
                if thumb.shape[0] > max_thumb_height:
                    max_thumb_height = thumb.shape[0]
            except Exception:
                continue
    if not thumbs:
        raise ValueError("Failed to load any images for contact sheet")

    rows = (len(thumbs) + columns - 1) // columns
    cell_w = thumb_width
    cell_h = max_thumb_height
    sheet_w = padding + columns * (cell_w + padding)
    sheet_h = padding + rows * (cell_h + padding) + title_height

    sheet = np.full((sheet_h, sheet_w, 3), bg_color, dtype=np.uint8)

    if title and title_height > 0:
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            size, _ = cv2.getTextSize(title, font, font_scale, thickness)
            tx = max(0, (sheet_w - size[0]) // 2)
            ty = min(title_height - 5, max(5, (title_height + size[1]) // 2))
            cv2.putText(sheet, title, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        except Exception:
            pass

    y = padding + title_height
    x = padding
    col = 0
    for thumb in thumbs:
        h, w = thumb.shape[:2]
        # Vertically center inside the cell
        y_offset = y + max(0, (cell_h - h) // 2)
        sheet[y_offset:y_offset + h, x:x + w] = thumb
        col += 1
        if col >= columns:
            col = 0
            x = padding
            y += cell_h + padding
        else:
            x += cell_w + padding

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return output_path
