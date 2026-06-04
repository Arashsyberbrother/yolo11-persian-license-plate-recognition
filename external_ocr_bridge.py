from __future__ import annotations

from typing import List, Tuple


class EasyOCRBridge:
    """Optional OCR bridge powered by the EasyOCR GitHub project."""

    def __init__(self, use_gpu: bool = False, languages: List[str] | None = None):
        try:
            import easyocr  # type: ignore
        except Exception as exc:  # pragma: no cover - import failure is runtime/environment-specific
            raise RuntimeError("EasyOCR نصب نیست. لطفاً easyocr را نصب کنید.") from exc

        self._reader = easyocr.Reader(languages or ["fa", "en"], gpu=bool(use_gpu), verbose=False)

    def read_candidates(self, image_bgr) -> List[Tuple[str, float]]:
        results = self._reader.readtext(image_bgr, detail=1, paragraph=False)
        candidates: List[Tuple[str, float]] = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                text = str(item[1]).strip()
                conf = float(item[2])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                text = str(item[1]).strip()
                conf = 0.0
            else:
                text = str(item).strip()
                conf = 0.0
            if text:
                candidates.append((text, conf))
        return candidates
