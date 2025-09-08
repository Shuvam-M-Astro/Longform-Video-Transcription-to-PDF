from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Callable

import cv2
import numpy as np
import shutil
import easyocr
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os


_CODE_TOKENS = {
    "def", "class", "import", "from", "return", "try", "except", "raise",
    "for", "while", "if", "elif", "else", "switch", "case",
    "function", "const", "let", "var", "=>",
    "public", "private", "protected", "static", "void", "int", "string", "boolean",
    ";", "{}", "()", "[]", "::", "#include", "printf", "cout", "System.out",
}

_PLOT_TOKENS = {
    "plt", "figure", "matplotlib", "seaborn", "epoch", "accuracy", "loss", "ROC",
    "precision", "recall", "F1", "confusion", "matrix", "chart", "graph", "axis", "axes",
}


def _text_density(text: str) -> float:
    if not text:
        return 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    avg_len = sum(len(ln) for ln in lines) / len(lines)
    return min(1.0, (len(lines) / 20.0) * 0.5 + (avg_len / 80.0) * 0.5)


def _contains_any(text: str, tokens: set[str]) -> bool:
    lower = text.lower()
    for t in tokens:
        if t.lower() in lower:
            return True
    return False


def _detect_plot_lines(image_path: Path) -> float:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=60, maxLineGap=10)
    count = 0 if lines is None else len(lines)
    return min(1.0, count / 40.0)


def classify_frames(
    frame_paths: List[Path],
    classified_root: Path,
    snippets_dir: Path,
    *,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, List[Path]]:
    classified_root = Path(classified_root)
    (classified_root / "code").mkdir(parents=True, exist_ok=True)
    (classified_root / "plots").mkdir(parents=True, exist_ok=True)
    (classified_root / "images").mkdir(parents=True, exist_ok=True)

    snippets_dir = Path(snippets_dir)
    snippets_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = False
    try:
        import torch  # type: ignore
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    result: Dict[str, List[Path]] = {"code": [], "plots": [], "images": []}
    lock = threading.Lock()

    thread_local: threading.local = threading.local()

    def get_reader() -> easyocr.Reader:
        rdr = getattr(thread_local, "reader", None)
        if rdr is None:
            rdr = easyocr.Reader(["en"], gpu=use_gpu)
            thread_local.reader = rdr
        return rdr

    def process_one(args: tuple[int, Path]) -> None:
        idx, frame_path = args
        reader = get_reader()
        ocr_result = reader.readtext(str(frame_path), detail=1, paragraph=True)
        texts = [entry[1] for entry in ocr_result if isinstance(entry, (list, tuple)) and len(entry) >= 2]
        text = "\n".join(t.strip() for t in texts if t and t.strip())

        density = _text_density(text)
        looks_like_code = density > 0.2 and _contains_any(text, _CODE_TOKENS)
        looks_like_plot = _contains_any(text, _PLOT_TOKENS) or _detect_plot_lines(frame_path) > 0.4

        if looks_like_code and not looks_like_plot:
            target = classified_root / "code" / frame_path.name
            shutil.copy2(frame_path, target)
            with lock:
                result["code"].append(target)
            snippet_file = snippets_dir / f"snippet_{idx:04d}.txt"
            snippet_file.write_text(text, encoding="utf-8")
        elif looks_like_plot:
            target = classified_root / "plots" / frame_path.name
            shutil.copy2(frame_path, target)
            with lock:
                result["plots"].append(target)
        else:
            target = classified_root / "images" / frame_path.name
            shutil.copy2(frame_path, target)
            with lock:
                result["images"].append(target)

    total = len(frame_paths) if frame_paths else 0
    if total == 0:
        if progress_cb:
            progress_cb(100.0)
        return result

    max_workers = 1 if use_gpu else min(8, max(2, (os.cpu_count() or 2)))
    tasks = [(idx, fp) for idx, fp in enumerate(frame_paths)]

    completed = 0
    bar = tqdm(total=total, desc="Classifying", unit="frame", leave=False)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for _ in as_completed(futures):
            completed += 1
            bar.update(1)
            if progress_cb and total:
                pct = max(0.0, min(100.0, (completed / total) * 100.0))
                progress_cb(pct)
    bar.close()

    if progress_cb:
        progress_cb(100.0)
    return result
