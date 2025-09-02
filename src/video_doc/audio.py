from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable
import ffmpeg


def _probe_duration_seconds(media_path: Path) -> float | None:
    try:
        info = ffmpeg.probe(str(media_path))
        if not info:
            return None
        fmt = info.get("format", {})
        if "duration" in fmt:
            return float(fmt["duration"])  # type: ignore[arg-type]
    except Exception:
        return None
    return None


def extract_audio_wav(video_path: Path, audio_path: Path, *, progress_cb: Optional[Callable[[float], None]] = None) -> Path:
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    total_seconds = _probe_duration_seconds(video_path) or 0.0

    try:
        stream = (
            ffmpeg
            .input(str(video_path))
            .output(
                str(audio_path),
                ac=1,  # mono
                ar=16000,  # 16kHz
                format="wav",
            )
            .overwrite_output()
            .global_args("-progress", "pipe:1", "-nostats")
        )
        process = stream.run_async(pipe_stdout=True, pipe_stderr=False)
        if progress_cb and total_seconds > 0:
            import re
            out_time_regex = re.compile(rb"out_time_ms=(\d+)")
            last = 0.0
            try:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    m = out_time_regex.search(line)
                    if m:
                        ms = int(m.group(1))
                        secs = ms / 1_000_000.0
                        pct = max(0.0, min(100.0, (secs / total_seconds) * 100.0))
                        if pct > last:
                            progress_cb(pct)
                            last = pct
            finally:
                process.wait()
        else:
            process.communicate()
    except Exception:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(audio_path),
                ac=1,
                ar=16000,
                format="wav",
            )
            .overwrite_output()
            .run(quiet=True)
        )

    if progress_cb:
        progress_cb(100.0)

    return audio_path
