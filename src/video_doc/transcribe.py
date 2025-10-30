from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Callable
import wave

from faster_whisper import WhisperModel
import os
import time

from src.video_doc.monitoring import metrics


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def _select_compute_type() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "float16"
    except Exception:
        pass
    return "int8"


def _init_model(model_size: str, prefer_cuda: bool, *, cpu_threads: Optional[int] = None, num_workers: Optional[int] = None) -> WhisperModel:
    if prefer_cuda:
        try:
            return WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[transcribe] GPU init failed ({e}). Falling back to CPU...")
    # Enable CPU parallelism where possible
    # Respect CLI overrides first, then environment variables
    if cpu_threads is None:
        cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", str(max(2, (os.cpu_count() or 2)))))
    if num_workers is None:
        num_workers = int(os.environ.get("WHISPER_NUM_WORKERS", "1"))
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )


def transcribe_audio(
    audio_path: Path,
    transcript_txt_path: Path,
    segments_json_path: Path,
    language: str = "auto",
    beam_size: int = 5,
    model_size: str = "medium",
    *,
    progress_cb: Optional[Callable[[float], None]] = None,
    cpu_threads: Optional[int] = None,
    num_workers: Optional[int] = None,
    srt_path: Optional[Path] = None,
) -> List[TranscriptSegment]:
    transcript_txt_path.parent.mkdir(parents=True, exist_ok=True)
    segments_json_path.parent.mkdir(parents=True, exist_ok=True)

    compute_type = _select_compute_type()
    prefer_cuda = compute_type == "float16"
    model = _init_model(model_size, prefer_cuda, cpu_threads=cpu_threads, num_workers=num_workers)
    print(
        f"[transcribe] Initialized model: size={model_size} device={'cuda' if prefer_cuda else 'cpu'} compute_type={compute_type}",
        flush=True,
    )

    lang_arg: Optional[str] = None if language == "auto" else language

    def _do_transcribe(m: WhisperModel):
        return m.transcribe(
            str(audio_path),
            beam_size=beam_size,
            language=lang_arg,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

    try:
        _t0 = time.time()
        segments_iter, info = _do_transcribe(model)
        _duration = time.time() - _t0
        try:
            metrics.observe_model_inference(model=model_size, device=('cuda' if prefer_cuda else 'cpu'), duration_seconds=_duration)
        except Exception:
            pass
    except Exception as e:
        msg = str(e).lower()
        if prefer_cuda and ("cublas" in msg or "cuda" in msg or "cudnn" in msg):
            print(f"[transcribe] CUDA error during transcription ({e}). Retrying on CPU...")
            model = _init_model(model_size, prefer_cuda=False, cpu_threads=cpu_threads, num_workers=num_workers)
            _t0 = time.time()
            segments_iter, info = _do_transcribe(model)
            _duration = time.time() - _t0
            try:
                metrics.observe_model_inference(model=model_size, device='cpu', duration_seconds=_duration)
            except Exception:
                pass
        else:
            raise

    segments: List[TranscriptSegment] = []
    total_duration = None
    try:
        # faster-whisper returns info.duration sometimes
        if info is not None and hasattr(info, "duration"):
            total_duration = float(getattr(info, "duration"))
    except Exception:
        total_duration = None

    # Fallback: probe WAV duration if model did not provide duration
    if (total_duration is None or total_duration <= 0) and str(audio_path).lower().endswith(".wav"):
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or 16000
                if rate > 0:
                    total_duration = float(frames) / float(rate)
        except Exception:
            total_duration = None

    last_pct = 0.0
    with open(transcript_txt_path, "w", encoding="utf-8") as f_txt:
        for seg in segments_iter:
            segment = TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
            segments.append(segment)
            f_txt.write(f"[{segment.start:7.2f} -> {segment.end:7.2f}] {segment.text}\n")
            if progress_cb and total_duration and total_duration > 0:
                pct = max(0.0, min(100.0, (segment.end / total_duration) * 100.0))
                if pct > last_pct:
                    progress_cb(pct)
                    last_pct = pct

    with open(segments_json_path, "w", encoding="utf-8") as f_json:
        json.dump([asdict(s) for s in segments], f_json, ensure_ascii=False, indent=2)

    # Optional SRT export
    if srt_path is not None:
        try:
            srt_path = Path(srt_path)
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            def _fmt_time(t: float) -> str:
                # HH:MM:SS,mmm
                hours = int(t // 3600)
                minutes = int((t % 3600) // 60)
                seconds = int(t % 60)
                millis = int(round((t - int(t)) * 1000))
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
            lines: List[str] = []
            for idx, seg in enumerate(segments, start=1):
                lines.append(str(idx))
                lines.append(f"{_fmt_time(seg.start)} --> { _fmt_time(seg.end)}")
                lines.append(seg.text or "")
                lines.append("")
            srt_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"[transcribe] Failed to write SRT: {e}")

    if progress_cb:
        progress_cb(100.0)
    return segments
