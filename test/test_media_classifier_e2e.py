"""End-to-end tests for media-type classification via ovos-media-classifier.

These exercise the *real* classifier (not mocked) end to end: the pipeline
hands its own ``voc_match`` + locale ``.voc`` files to
``ovos-media-classifier``'s keyword backend (which is the extraction of the
pipeline's former embedded voc logic), and the result -- a
:class:`mediavocab.MediaType` -- is returned directly. The pipeline is
mediavocab-native; there is no ``ovos_utils.ocp`` taxonomy translation.

Some legacy keyword buckets converge onto the mediavocab taxonomy: e.g.
``anime``/``cartoon`` classify to ``EPISODIC_SERIES``; ``documentary`` collapses
onto ``MOVIE``; ``news`` onto ``RADIO``.
"""
import unittest

from ovos_utils.fakebus import FakeBus
from mediavocab import MediaType

from ocp_pipeline.opm import OCPPipelineMatcher


class TestMediaClassifierE2E(unittest.TestCase):
    """Real classifier, real locale .voc files, mediavocab-native results."""

    @classmethod
    def setUpClass(cls):
        cls.ocp = OCPPipelineMatcher(bus=FakeBus())

    def test_real_classifier_is_wired(self):
        """The matcher delegates to a real ovos-media-classifier instance."""
        from ovos_media_classifier.base import AbstractMediaClassifier
        self.assertIsInstance(self.ocp.media_clf, AbstractMediaClassifier)

    def test_classify_media_real_utterances(self):
        """Real utterances classify to the expected mediavocab.MediaType."""
        cases = {
            # query                       -> expected mediavocab.MediaType
            "play some music": MediaType.MUSIC,
            "watch a movie": MediaType.MOVIE,
            "play a podcast": MediaType.PODCAST,
            # ANIME/CARTOON -> mediavocab EPISODIC_SERIES
            "i want to watch an anime": MediaType.EPISODIC_SERIES,
            "play a cartoon": MediaType.EPISODIC_SERIES,
            # NEWS -> mediavocab RADIO
            "play the news": MediaType.RADIO,
            # DOCUMENTARY -> mediavocab MOVIE
            "show me a documentary": MediaType.MOVIE,
        }
        for query, expected in cases.items():
            media, conf = self.ocp.classify_media(query, "en-US")
            self.assertEqual(media, expected,
                             f"{query!r} -> {media} (expected {expected})")
            self.assertIsInstance(media, MediaType)
            self.assertIsInstance(conf, float)
            self.assertGreater(conf, 0.0)

    def test_classify_media_no_keyword_is_generic(self):
        """A bare artist name with no media keyword stays GENERIC."""
        media, conf = self.ocp.classify_media("play metallica", "en-US")
        self.assertEqual(media, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_media_respects_valid_labels(self):
        """A query whose type is excluded by valid_labels is not returned."""
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
