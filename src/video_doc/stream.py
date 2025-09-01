from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import ffmpeg
from yt_dlp import YoutubeDL

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def _headers_to_ffmpeg_header_string(headers: Optional[Dict[str, str]]) -> Optional[str]:
    if not headers:
        return None
    lines = [f"{k}: {v}" for k, v in headers.items()]
    return "\r\n".join(lines)


def _build_ydl_opts(
    *,
    cookies_from_browser: Optional[str] = None,
    browser_profile: Optional[str] = None,
    cookies_file: Optional[Path] = None,
    use_android_client: bool = False,
    extra: Optional[Dict] = None,
) -> Dict:
    extractor_args = {"youtube": {"player_client": ["android"]}} if use_android_client else {}
    ydl_opts: Dict = {
        "quiet": False,
        "noplaylist": True,
        "no_warnings": False,
        "http_headers": {"User-Agent": _DEFAULT_UA},
        "extractor_args": extractor_args,
        "retries": 3,
        "fragment_retries": 3,
    }
    if cookies_from_browser:
        tup = (cookies_from_browser,) if not browser_profile else (cookies_from_browser, browser_profile)
        ydl_opts["cookiesfrombrowser"] = tup
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)
    if extra:
        ydl_opts.update(extra)
    return ydl_opts


def resolve_stream_urls(
    url: str,
    *,
    cookies_from_browser: Optional[str] = None,
    browser_profile: Optional[str] = None,
    cookies_file: Optional[Path] = None,
    use_android_client: bool = False,
) -> Dict[str, Optional[str]]:
    ydl_opts = _build_ydl_opts(
        cookies_from_browser=cookies_from_browser,
        browser_profile=browser_profile,
        cookies_file=cookies_file,
        use_android_client=use_android_client,
        extra={"quiet": True, "no_warnings": True},
    )

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    best_audio = None
    best_video = None
    if "formats" in info:
        formats = info["formats"]
        audio_candidates = [f for f in formats if f.get("acodec") != "none" and f.get("vcodec") == "none"]
        audio_candidates.sort(key=lambda f: (
            0 if (f.get("ext") == "m4a") else 1,
            -(f.get("tbr") or 0),
        ))
        best_audio = audio_candidates[0]["url"] if audio_candidates else None

        video_candidates = [f for f in formats if f.get("vcodec") != "none" and f.get("acodec") == "none"]
        video_candidates.sort(key=lambda f: (
            0 if (f.get("ext") == "mp4") else 1,
            -(f.get("height") or 0),
            -(f.get("tbr") or 0),
        ))
        best_video = video_candidates[0]["url"] if video_candidates else None

    headers = info.get("http_headers") or {"User-Agent": _DEFAULT_UA}

    return {
        "audio_url": best_audio,
        "video_url": best_video,
        "headers": _headers_to_ffmpeg_header_string(headers),
    }


def stream_extract_audio(audio_url: str, audio_path: Path, headers: Optional[str] = None) -> Path:
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    inp = ffmpeg.input(audio_url, user_agent=_DEFAULT_UA, headers=headers) if headers else ffmpeg.input(audio_url, user_agent=_DEFAULT_UA)
    (
        inp
        .output(
            str(audio_path),
            ac=1,
            ar=16000,
            format="wav",
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path


def fallback_download_audio_via_ytdlp(
    page_url: str,
    audio_path: Path,
    *,
    cookies_from_browser: Optional[str] = None,
    browser_profile: Optional[str] = None,
    cookies_file: Optional[Path] = None,
    use_android_client: bool = False,
) -> Path:
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    base_tmpl = str(audio_path.with_suffix(""))
    # Force standard web client in fallback to avoid PO token issues
    ydl_opts = _build_ydl_opts(
        cookies_from_browser=cookies_from_browser,
        browser_profile=browser_profile,
        cookies_file=cookies_file,
        use_android_client=False,
        extra={
            "format": "bestaudio/best",
            "outtmpl": {"default": base_tmpl + ".%(ext)s"},
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",
                }
            ],
            "keepvideo": False,
        },
    )

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([page_url])

    if audio_path.exists():
        return audio_path
    candidates = list(audio_path.parent.glob(audio_path.stem + ".*"))
    if candidates:
        for c in candidates:
            if c.suffix.lower() == ".wav":
                return c
    raise FileNotFoundError(f"Audio extraction via yt-dlp failed: {audio_path}")


def fallback_download_small_video(
    page_url: str,
    video_path: Path,
    *,
    max_height: int = 144,
    cookies_from_browser: Optional[str] = None,
    browser_profile: Optional[str] = None,
    cookies_file: Optional[Path] = None,
    use_android_client: bool = False,
) -> Path:
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Force standard web client in fallback to avoid PO token issues
    ydl_opts = _build_ydl_opts(
        cookies_from_browser=cookies_from_browser,
        browser_profile=browser_profile,
        cookies_file=cookies_file,
        use_android_client=False,
        extra={
            "format": f"bestvideo[ext=mp4][height<={max_height}]+bestaudio[ext=m4a]/best[ext=mp4][height<={max_height}]/best[height<={max_height}]",
            "outtmpl": {"default": str(video_path.with_suffix("")) + ".%(ext)s"},
            "merge_output_format": "mp4",
            "postprocessors": [
                {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}
            ],
        },
    )

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([page_url])

    if video_path.exists():
        return video_path
    candidates = list(video_path.parent.glob(video_path.stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"Small video download failed: {video_path}")
    best = None
    for c in candidates:
        if c.suffix.lower() == ".mp4":
            best = c
            break
    if best is None:
        best = candidates[0]
    if best != video_path:
        best.replace(video_path)
    return video_path


def stream_extract_keyframes(
    video_url: str,
    output_dir: Path,
    max_fps: float = 1.0,
    scene_threshold: float = 0.45,
    headers: Optional[str] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vf = f"select='gt(scene,{scene_threshold})',fps={max_fps}"
    pattern = str(output_dir / "frame_%06d.jpg")

    inp = ffmpeg.input(video_url, user_agent=_DEFAULT_UA, headers=headers) if headers else ffmpeg.input(video_url, user_agent=_DEFAULT_UA)
    (
        inp
        .output(
            pattern,
            vf=vf,
            vsync="vfr",
            **{"qscale:v": 2},
        )
        .overwrite_output()
        .run(quiet=True)
    )
