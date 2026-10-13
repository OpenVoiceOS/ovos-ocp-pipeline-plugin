"""Only the primary language trains while OCP loads; the rest wait for their
first query or for ``mycroft.ready``."""
import tempfile
import unittest
from unittest.mock import patch

from ovos_utils.fakebus import FakeBus
from ovos_bus_client.message import Message

from ocp_pipeline.opm import OCPPipelineMatcher

CONFIG = {"lang": "en-US", "secondary_langs": ["fr-FR", "de-DE"]}


class TestDeferredTraining(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache.cleanup)
        patcher = patch.object(OCPPipelineMatcher, "intent_cache", self.cache.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(OCPPipelineMatcher.intent_matchers.clear)
        with patch("ocp_pipeline.opm.Configuration", return_value=CONFIG):
            OCPPipelineMatcher.load_intent_files()
        self.matchers = OCPPipelineMatcher.intent_matchers

    def test_only_the_primary_language_trains_while_loading(self):
        self.assertFalse(self.matchers["en-US"].needs_compile)
        self.assertTrue(self.matchers["fr-FR"].needs_compile)
        self.assertTrue(self.matchers["de-DE"].needs_compile)
        self.assertEqual(sorted(OCPPipelineMatcher.untrained_langs), ["de-DE", "fr-FR"])

    def test_an_untrained_language_still_matches_its_first_query(self):
        # a restart: the previous boot left every language's models cached
        for matcher in self.matchers.values():
            matcher.train()
        with patch("ocp_pipeline.opm.Configuration", return_value=CONFIG):
            OCPPipelineMatcher.load_intent_files()
        fr = OCPPipelineMatcher.intent_matchers["fr-FR"]
        self.assertTrue(fr.needs_compile)
        self.assertEqual(fr.calc_intent("mettre en pause la musique").name, "pause")

    def test_ready_trains_the_rest_in_the_background(self):
        trained = []
        with patch("ocp_pipeline.opm.threading.Thread") as thread:
            OCPPipelineMatcher.train_untrained_matchers(Message("mycroft.ready"))
            target = thread.call_args.kwargs["target"]
            self.assertTrue(thread.call_args.kwargs["daemon"])
        self.assertEqual(OCPPipelineMatcher.untrained_langs, [])
        target()
        self.assertFalse(self.matchers["fr-FR"].needs_compile)
        self.assertFalse(self.matchers["de-DE"].needs_compile)


class TestReadyHook(unittest.TestCase):
    def test_register_ocp_intents_trains_the_rest_on_ready(self):
        bus = FakeBus()
        with patch.object(OCPPipelineMatcher, "load_intent_files"), \
                patch.object(OCPPipelineMatcher, "train_untrained_matchers") as train, \
                patch.object(OCPPipelineMatcher, "add_event"):
            matcher = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
            matcher.bus = bus
            matcher.register_ocp_intents()
            bus.emit(Message("mycroft.ready"))
            bus.emit(Message("mycroft.ready"))
        train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
