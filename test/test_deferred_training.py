"""Only the primary language trains while OCP loads; every other language
trains on its first query."""
import tempfile
import unittest
from unittest.mock import patch

from ocp_pipeline.opm import OCPPipelineMatcher

try:
    import ovos_padatious  # noqa: F401
    IS_PADATIOUS = True
except ImportError:
    IS_PADATIOUS = False

CONFIG = {"lang": "en-US", "secondary_langs": ["fr-FR", "de-DE"]}


def _load():
    with patch("ocp_pipeline.opm.Configuration", return_value=CONFIG):
        OCPPipelineMatcher.load_intent_files()
    return OCPPipelineMatcher.intent_matchers


@unittest.skipUnless(IS_PADATIOUS, "only a padatious container trains")
class TestDeferredTraining(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache.cleanup)
        patcher = patch.object(OCPPipelineMatcher, "intent_cache", self.cache.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(OCPPipelineMatcher.intent_matchers.clear)
        self.matchers = _load()

    def test_only_the_primary_language_trains_while_loading(self):
        self.assertFalse(self.matchers["en-US"].needs_compile)
        self.assertTrue(self.matchers["fr-FR"].needs_compile)
        self.assertTrue(self.matchers["de-DE"].needs_compile)

    def test_an_untrained_language_still_matches_its_first_query(self):
        # a restart: the previous boot left every language's models cached
        for matcher in self.matchers.values():
            matcher.train()
        fr = _load()["fr-FR"]
        self.assertTrue(fr.needs_compile)
        self.assertEqual(fr.calc_intent("mettre en pause la musique").name, "pause")
        self.assertFalse(fr.needs_compile)


if __name__ == "__main__":
    unittest.main()
