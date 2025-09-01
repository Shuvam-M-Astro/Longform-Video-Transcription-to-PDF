from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from faster_whisper import WhisperModel


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


def _init_model(model_size: str, prefer_cuda: bool) -> WhisperModel:
    if prefer_cuda:
        try:
            return WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[transcribe] GPU init failed ({e}). Falling back to CPU...")
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(
    audio_path: Path,
    transcript_txt_path: Path,
    segments_json_path: Path,
    language: str = "auto",
    beam_size: int = 5,
    model_size: str = "medium",
) -> List[TranscriptSegment]:
    transcript_txt_path.parent.mkdir(parents=True, exist_ok=True)
    segments_json_path.parent.mkdir(parents=True, exist_ok=True)

    compute_type = _select_compute_type()
    prefer_cuda = compute_type == "float16"
    model = _init_model(model_size, prefer_cuda)

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
        segments_iter, _info = _do_transcribe(model)
    except Exception as e:
        msg = str(e).lower()
        if prefer_cuda and ("cublas" in msg or "cuda" in msg or "cudnn" in msg):
            print(f"[transcribe] CUDA error during transcription ({e}). Retrying on CPU...")
            model = _init_model(model_size, prefer_cuda=False)
            segments_iter, _info = _do_transcribe(model)
        else:
            raise

    segments: List[TranscriptSegment] = []
    with open(transcript_txt_path, "w", encoding="utf-8") as f_txt:
        for seg in segments_iter:
            segment = TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
            segments.append(segment)
            f_txt.write(f"[{segment.start:7.2f} -> {segment.end:7.2f}] {segment.text}\n")

    with open(segments_json_path, "w", encoding="utf-8") as f_json:
        json.dump([asdict(s) for s in segments], f_json, ensure_ascii=False, indent=2)

    return segments
