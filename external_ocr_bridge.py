from __future__ import annotations

from typing import List, Tuple


class EasyOCRBridge:
    """Optional OCR bridge powered by EasyOCR.

    Args:
        use_gpu: Enables GPU inference in EasyOCR when available.
        languages: EasyOCR language codes, defaults to Persian + English.
    """

    def __init__(self, use_gpu: bool = False, languages: List[str] | None = None):
        try:
            import easyocr  # type: ignore
        except Exception as exc:  # pragma: no cover - import failure is runtime/environment-specific
            raise RuntimeError("EasyOCR نصب نیست. لطفاً easyocr را نصب کنید.") from exc

        self._reader = easyocr.Reader(languages or ["fa", "en"], gpu=bool(use_gpu), verbose=False)

    def read_candidates(self, image_bgr) -> List[Tuple[str, float]]:
        """Read OCR candidates from a BGR image.

        Args:
            image_bgr: Plate crop in OpenCV BGR ndarray format.

        Returns:
            List of (recognized_text, confidence_score) pairs.
        """
        results = self._reader.readtext(image_bgr, detail=1, paragraph=False)
        candidates: List[Tuple[str, float]] = []
        for item in results:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                continue
            _, text, conf = item
            text = str(text).strip()
            if text:
                candidates.append((text, conf))
        return candidates
