from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import shutil
import easyocr


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

    reader = easyocr.Reader(["en"], gpu=use_gpu)

    result: Dict[str, List[Path]] = {"code": [], "plots": [], "images": []}

    for idx, frame_path in enumerate(frame_paths):
        ocr_result = reader.readtext(str(frame_path), detail=1, paragraph=True)
        texts = [entry[1] for entry in ocr_result if isinstance(entry, (list, tuple)) and len(entry) >= 2]
        text = "\n".join(t.strip() for t in texts if t and t.strip())

        density = _text_density(text)
        looks_like_code = density > 0.2 and _contains_any(text, _CODE_TOKENS)
        looks_like_plot = _contains_any(text, _PLOT_TOKENS) or _detect_plot_lines(frame_path) > 0.4

        if looks_like_code and not looks_like_plot:
            target = classified_root / "code" / frame_path.name
            shutil.copy2(frame_path, target)
            result["code"].append(target)
            snippet_file = snippets_dir / f"snippet_{idx:04d}.txt"
            snippet_file.write_text(text, encoding="utf-8")
        elif looks_like_plot:
            target = classified_root / "plots" / frame_path.name
            shutil.copy2(frame_path, target)
            result["plots"].append(target)
        else:
            target = classified_root / "images" / frame_path.name
            shutil.copy2(frame_path, target)
            result["images"].append(target)

    return result
