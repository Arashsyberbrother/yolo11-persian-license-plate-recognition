import unittest

from desktop_ui_utils import (
    extract_iranian_plate_candidate,
    is_strict_iranian_plate_text,
    is_plausible_plate_text,
    is_readable_plate_text,
    localize_plate_text_for_display,
    normalize_plate_text,
    register_plate_event,
)


class DesktopUiUtilsSmokeTests(unittest.TestCase):
    def test_normalize_plate_text_unifies_digits_and_separators(self):
        self.assertEqual(normalize_plate_text("۱۲-٣ ٤"), "1234")

    def test_localize_plate_text_for_display_uses_persian_digits(self):
        self.assertEqual(localize_plate_text_for_display("12ب34567"), "۱۲ب۳۴۵۶۷")

    def test_register_plate_event_applies_interval_dedup(self):
        last_seen = {}
        duplicate_counts = {}

        emitted, skipped = register_plate_event(last_seen, duplicate_counts, "12ABC34", 100.0, 2)
        self.assertTrue(emitted)
        self.assertEqual(skipped, 0)

        emitted, skipped = register_plate_event(last_seen, duplicate_counts, "12ABC34", 101.0, 2)
        self.assertFalse(emitted)
        self.assertEqual(skipped, 1)

        emitted, skipped = register_plate_event(last_seen, duplicate_counts, "12ABC34", 103.5, 2)
        self.assertTrue(emitted)
        self.assertEqual(skipped, 1)

    def test_is_plausible_plate_text_accepts_reasonable_plate_pattern(self):
        self.assertTrue(is_plausible_plate_text("12ب34567"))

    def test_is_strict_iranian_plate_text_accepts_standard_pattern(self):
        self.assertTrue(is_strict_iranian_plate_text("12ب34567"))

    def test_is_strict_iranian_plate_text_rejects_wrong_structure(self):
        self.assertFalse(is_strict_iranian_plate_text("1ب234567"))

    def test_is_strict_iranian_plate_text_rejects_non_persian_letter(self):
        self.assertFalse(is_strict_iranian_plate_text("12A34567"))

    def test_extract_iranian_plate_candidate_recovers_from_noise(self):
        self.assertEqual(extract_iranian_plate_candidate("XX12ب34567YY"), "12ب34567")

    def test_extract_iranian_plate_candidate_returns_empty_when_missing(self):
        self.assertEqual(extract_iranian_plate_candidate("12345اب"), "")

    def test_is_plausible_plate_text_rejects_repeated_gibberish(self):
        self.assertFalse(is_plausible_plate_text("قققققق5"))

    def test_is_readable_plate_text_allows_less_strict_candidate(self):
        self.assertTrue(is_readable_plate_text("123ب45"))

    def test_is_readable_plate_text_allows_upper_boundary(self):
        self.assertTrue(is_readable_plate_text("1234بپتث5678"))

    def test_is_readable_plate_text_rejects_insufficient_digits(self):
        self.assertFalse(is_readable_plate_text("اااا"))

    def test_is_readable_plate_text_rejects_too_many_letters(self):
        self.assertFalse(is_readable_plate_text("1234ابپتث"))


if __name__ == "__main__":
    unittest.main()
