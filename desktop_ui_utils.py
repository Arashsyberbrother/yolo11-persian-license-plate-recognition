import os
import tempfile


_DIGIT_TRANSLATION = str.maketrans(
    {
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)

_CHAR_TRANSLATION = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
        "ە": "ه",
        "ة": "ه",
        "ۀ": "ه",
        "إ": "ا",
        "أ": "ا",
        "ؤ": "و",
    }
)


def normalize_plate_text(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text).strip().translate(_DIGIT_TRANSLATION).translate(_CHAR_TRANSLATION)
    cleaned = cleaned.replace(" ", "").replace("-", "").replace("_", "")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ("\u0600" <= ch <= "\u06ff"))
    return cleaned.upper()


def register_plate_event(last_seen: dict, duplicate_counts: dict, plate_key: str, now: float, interval_seconds: int):
    if interval_seconds <= 0 or not plate_key:
        return True, 0

    prev = last_seen.get(plate_key)
    if prev is not None and (now - prev) < interval_seconds:
        duplicate_counts[plate_key] = duplicate_counts.get(plate_key, 0) + 1
        return False, duplicate_counts[plate_key]

    skipped = duplicate_counts.pop(plate_key, 0)
    last_seen[plate_key] = now
    return True, skipped


def ensure_output_dir_writable(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        return False, f"ایجاد پوشه خروجی ممکن نیست: {exc}"

    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path, delete=True):
            pass
    except OSError as exc:
        return False, f"پوشه خروجی قابل نوشتن نیست: {exc}"

    return True, ""
