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


def extract_audio_wav(
    video_path: Path,
    audio_path: Path,
    *,
    progress_cb: Optional[Callable[[float], None]] = None,
    start_time: float = 0.0,
    end_trim: float = 0.0,
    volume_gain_db: float = 0.0,
) -> Path:
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    total_seconds = _probe_duration_seconds(video_path) or 0.0
    eff_start = max(0.0, float(start_time or 0.0))
    eff_end_trim = max(0.0, float(end_trim or 0.0))
    eff_duration = None
    if total_seconds > 0 and (eff_start > 0 or eff_end_trim > 0):
        eff_duration = max(0.0, total_seconds - eff_start - eff_end_trim)
    print(
        f"[audio] Extracting WAV: duration≈{total_seconds:.1f}s start={eff_start:.2f}s end_trim={eff_end_trim:.2f}s -> {audio_path}",
        flush=True,
    )

    try:
        out_kwargs = {
            "ac": 1,
            "ar": 16000,
            "format": "wav",
        }
        if eff_duration is not None and eff_duration > 0:
            out_kwargs["t"] = eff_duration
        afilter = None
        if volume_gain_db and abs(volume_gain_db) > 0.001:
            afilter = f"volume={volume_gain_db}dB"

        stream = (
            ffmpeg
            .input(str(video_path), ss=eff_start if eff_start > 0 else None)
            .output(
                str(audio_path),
                af=afilter,
                **out_kwargs,
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
        out_kwargs = {
            "ac": 1,
            "ar": 16000,
            "format": "wav",
        }
        if eff_duration is not None and eff_duration > 0:
            out_kwargs["t"] = eff_duration
        afilter = None
        if volume_gain_db and abs(volume_gain_db) > 0.001:
            afilter = f"volume={volume_gain_db}dB"
        (
            ffmpeg
            .input(str(video_path), ss=eff_start if eff_start > 0 else None)
            .output(
                str(audio_path),
                af=afilter,
                **out_kwargs,
            )
            .overwrite_output()
            .run(quiet=True)
        )

    if progress_cb:
        progress_cb(100.0)

    return audio_path
