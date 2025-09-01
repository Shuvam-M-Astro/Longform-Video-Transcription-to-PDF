from __future__ import annotations

from pathlib import Path
import ffmpeg


def extract_audio_wav(video_path: Path, audio_path: Path) -> Path:
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg
        .input(str(video_path))
        .output(
            str(audio_path),
            ac=1,  # mono
            ar=16000,  # 16kHz
            format="wav",
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return audio_path
