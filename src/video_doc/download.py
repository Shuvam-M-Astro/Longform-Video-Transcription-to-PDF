from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

from yt_dlp import YoutubeDL


_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def _cookiesfrombrowser_tuple(browser: Optional[str], profile: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not browser:
        return None
    browser = browser.strip()
    if not browser:
        return None
    if profile and profile.strip():
        return (browser, profile.strip())
    return (browser,)


def download_video(
    url: str,
    output_path: Path,
    *,
    cookies_from_browser: Optional[str] = None,
    browser_profile: Optional[str] = None,
    cookies_file: Optional[Path] = None,
    use_android_client: bool = False,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Download a video from the given URL to the specified MP4 path.

    Supports cookie auth via a browser profile or a cookies.txt file.
    Optionally uses the YouTube Android player client as a fallback.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    outtmpl = str(output_path.with_suffix(""))

    extractor_args: Dict[str, Any] = {}
    if use_android_client:
        extractor_args = {"youtube": {"player_client": ["android"]}}

    ydl_opts: Dict[str, Any] = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": {"default": outtmpl + ".%(ext)s"},
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "http_headers": {"User-Agent": _DEFAULT_UA},
        "extractor_args": extractor_args,
        "retries": 3,
        "fragment_retries": 3,
    }

    cfb = _cookiesfrombrowser_tuple(cookies_from_browser, browser_profile)
    if cfb:
        ydl_opts["cookiesfrombrowser"] = cfb
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)

    def _hook(d: Dict[str, Any]) -> None:
        if not progress_cb:
            return
        try:
            if d.get('status') == 'downloading':
                # Estimate 0-90% for download; postprocessing 90-100
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                downloaded = d.get('downloaded_bytes') or 0
                if total:
                    pct = max(0.0, min(100.0, float(downloaded) / float(total) * 90.0))
                    progress_cb(pct)
            elif d.get('status') == 'finished':
                progress_cb(95.0)
        except Exception:
            pass

    if progress_cb:
        ydl_opts['progress_hooks'] = [_hook]

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if output_path.exists():
        return output_path

    candidates = list(output_path.parent.glob(output_path.stem + ".*"))
    if not candidates:
        raise FileNotFoundError("Download finished but file not found: " + str(output_path))

    final_candidate: Optional[Path] = None
    for c in candidates:
        if c.suffix.lower() == ".mp4":
            final_candidate = c
            break
    if final_candidate is None:
        final_candidate = candidates[0]

    if final_candidate != output_path:
        final_candidate.replace(output_path)

    if progress_cb:
        progress_cb(100.0)
    return output_path
