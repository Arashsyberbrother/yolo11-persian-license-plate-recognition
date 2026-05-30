import unittest

from desktop_ui_utils import is_plausible_plate_text, is_readable_plate_text, normalize_plate_text, register_plate_event


class DesktopUiUtilsSmokeTests(unittest.TestCase):
    def test_normalize_plate_text_unifies_digits_and_separators(self):
        self.assertEqual(normalize_plate_text("۱۲-٣ ٤"), "1234")

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
