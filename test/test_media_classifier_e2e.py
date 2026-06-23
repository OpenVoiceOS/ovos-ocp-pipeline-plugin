"""End-to-end tests for media-type classification via ovos-media-classifier.

These exercise the *real* classifier (not mocked) end to end: the pipeline
hands its own ``voc_match`` + locale ``.voc`` files to
``ovos-media-classifier``'s keyword backend (which is the extraction of the
pipeline's former embedded voc logic), and the result is bridged from
``mediavocab.MediaType`` back to ``ovos_utils.ocp.MediaType``.

Two taxonomies are involved and converge here: e.g. ``anime``/``cartoon`` now
classify through mediavocab ``EPISODIC_SERIES`` and surface as ocp
``VIDEO_EPISODES``; ``documentary`` and ``news`` collapse onto ``MOVIE`` /
``RADIO`` respectively. See ``ocp_pipeline.bridge`` for the exact mapping.
"""
import unittest

from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType

from ocp_pipeline.opm import OCPPipelineMatcher


class TestMediaClassifierE2E(unittest.TestCase):
    """Real classifier, real locale .voc files, real mediavocab<->ocp bridge."""

    @classmethod
    def setUpClass(cls):
        cls.ocp = OCPPipelineMatcher(bus=FakeBus())

    def test_real_classifier_is_wired(self):
        """The matcher delegates to a real ovos-media-classifier instance."""
        from ovos_media_classifier.base import AbstractMediaClassifier
        self.assertIsInstance(self.ocp.media_clf, AbstractMediaClassifier)

    def test_classify_media_real_utterances(self):
        """Real utterances classify to the expected ocp.MediaType after the
        mediavocab -> ocp bridge."""
        cases = {
            # query                       -> expected ovos_utils.ocp.MediaType
            "play some music": MediaType.MUSIC,
            "watch a movie": MediaType.MOVIE,
            "play a podcast": MediaType.PODCAST,
            # taxonomy convergence: ANIME/CARTOON -> mediavocab EPISODIC_SERIES
            # -> ocp VIDEO_EPISODES
            "i want to watch an anime": MediaType.VIDEO_EPISODES,
            "play a cartoon": MediaType.VIDEO_EPISODES,
            # NEWS -> mediavocab RADIO -> ocp RADIO
            "play the news": MediaType.RADIO,
            # DOCUMENTARY -> mediavocab MOVIE -> ocp MOVIE
            "show me a documentary": MediaType.MOVIE,
        }
        for query, expected in cases.items():
            media, conf = self.ocp.classify_media(query, "en-US")
            self.assertEqual(media, expected,
                             f"{query!r} -> {media} (expected {expected})")
            self.assertIsInstance(conf, float)
            self.assertGreater(conf, 0.0)

    def test_classify_media_no_keyword_is_generic(self):
        """A bare artist name with no media keyword stays GENERIC."""
        media, conf = self.ocp.classify_media("play metallica", "en-US")
        self.assertEqual(media, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_media_respects_valid_labels(self):
        """A query whose mapped type is excluded by valid_labels -> GENERIC."""
        # "play some music" maps to MUSIC, but MUSIC is not allowed here
        media, _ = self.ocp.classify_media(
            "play some music", "en-US", valid_labels=[MediaType.PODCAST])
        self.assertNotEqual(media, MediaType.MUSIC)

    def test_is_ocp_query_true_for_media(self):
        for query in ("play some music", "watch a movie", "play a podcast",
                      "i want to watch an anime", "play the news"):
            is_ocp, _ = self.ocp.is_ocp_query(query, "en-US")
            self.assertTrue(is_ocp, f"{query!r} should be an OCP query")

    def test_is_ocp_query_false_for_non_media(self):
        is_ocp, _ = self.ocp.is_ocp_query("what time is it", "en-US")
        self.assertFalse(is_ocp)

    def test_classification_extracted_from_pipeline(self):
        """The embedded voc_match_media elif chain is gone; classification now
        lives in the package. The pipeline keeps only the thin adapters."""
        # the dead method must no longer exist on the matcher
        self.assertFalse(hasattr(self.ocp, "voc_match_media"))
        # the thin adapters remain and delegate to the package
        self.assertTrue(hasattr(self.ocp, "classify_media"))
        self.assertTrue(hasattr(self.ocp, "is_ocp_query"))


if __name__ == "__main__":
    unittest.main()
