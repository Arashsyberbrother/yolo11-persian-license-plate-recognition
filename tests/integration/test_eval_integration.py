import os
import unittest


class EvalPipelineIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_INTEGRATION_TESTS") == "1",
        "Set RUN_INTEGRATION_TESTS=1 to run model-dependent integration tests.",
    )
    def test_eval_weights_are_available_for_optional_integration_runs(self):
        self.assertTrue(os.path.exists("yolo11_anpr_ghd.pt"))
        self.assertTrue(os.path.exists("persian_digit_classifier.pt"))


if __name__ == "__main__":
    unittest.main()
