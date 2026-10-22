"""Locale directory resolution no longer relies on the deprecated
``ovos_utils.lang.get_language_dir`` helper."""
import unittest
import warnings

import ocp_pipeline.opm as opm
from ocp_pipeline.opm import OCPPipelineMatcher


class TestLocaleResolution(unittest.TestCase):
    def test_deprecated_helper_not_imported(self):
        self.assertFalse(hasattr(opm, "get_language_dir"))

    def test_load_resource_files_resolves_locale_without_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            intents = OCPPipelineMatcher.load_resource_files()
        self.assertIn("en-US", intents)
        self.assertTrue(any(intents["en-US"].values()))
        self.assertEqual(
            [w for w in caught if "get_language_dir" in str(w.message)], [])


if __name__ == "__main__":
    unittest.main()
