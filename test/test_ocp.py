"""Integration tests for OCPPipelineMatcher using the real __init__.

These exercise the full match_high / match_medium / match_low flow. Media-type
detection is delegated to the standalone ``ovos-media-classifier`` zero-dependency
keyword backend (the pipeline's native voc chain was replaced by it); detection
fires on the classifier's own media cues, not on free-text artist/title guesses.
"""
import os.path
import unittest

from ovos_utils.ocp import MediaType

import ocp_pipeline.opm
from ocp_pipeline.opm import OCPPipelineMatcher


class TestOCPPipelineMatcher(unittest.TestCase):

    def setUp(self):
        self.ocp = OCPPipelineMatcher(config={})
        # pretend a skill is loaded so media2skill is non-empty and all
        # MediaType labels are considered valid during classification
        self.ocp.skill_aliases["test"] = ["Test Skill"]

    # ------------------------------------------------------------------ #
    # match_high (padatious/padacioso intents)
    # ------------------------------------------------------------------ #
    def test_match_high(self):
        result = self.ocp.match_high(["play metallica"], "en-US")
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, 'ocp:play')

    def test_match_high_with_invalid_input(self):
        result = self.ocp.match_high(["put on some music"], "en-US")
        self.assertIsNone(result)

    # ------------------------------------------------------------------ #
    # match_medium (is_ocp_query gate via media keywords)
    # ------------------------------------------------------------------ #
    def test_match_medium(self):
        # "movie" is a media keyword -> is_ocp_query True
        result = self.ocp.match_medium(["put on some movie"], "en-US")
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, 'ocp:play')

    def test_match_medium_with_invalid_input(self):
        # no media cue present -> not an OCP query
        result = self.ocp.match_medium(["what time is it"], "en-US")
        self.assertIsNone(result)

    # ------------------------------------------------------------------ #
    # match_low (requires NER entity + media keyword)
    # ------------------------------------------------------------------ #
    def test_match_fallback_with_invalid_input(self):
        result = self.ocp.match_low(["do the thing"], "en-US")
        self.assertIsNone(result)

    def test_match_fallback_without_known_entity(self):
        # match_low requires a NER entity hit; with no registered skill
        # keywords/entities there is nothing to extract -> None
        result = self.ocp.match_low(["i want music"], "en-US")
        self.assertIsNone(result)

    def test_match_fallback_with_registered_entity(self):
        # register a MUSIC skill alias as a NER entity, then a query that
        # contains both the entity and a media keyword should match
        self.ocp.handle_skill_keyword_register(
            ocp_pipeline.opm.Message("", {
                "skill_id": "fake",
                "label": "music_streaming_service",
                "media_type": int(MediaType.MUSIC),
                "samples": ["spotify"],
            })
        )
        result = self.ocp.match_low(["play music on spotify"], "en-US")
        self.assertIsNotNone(result)
        self.assertEqual(result.match_type, 'ocp:play')

    # ------------------------------------------------------------------ #
    # is_ocp_query (keyword based)
    # ------------------------------------------------------------------ #
    def test_predict(self):
        # explicit media keywords -> OCP query
        self.assertTrue(self.ocp.is_ocp_query("play a song", "en-US")[0])
        self.assertTrue(self.ocp.is_ocp_query("play a movie", "en-US")[0])
        self.assertTrue(self.ocp.is_ocp_query("play a podcast", "en-US")[0])
        # no media keyword -> not an OCP query
        self.assertFalse(self.ocp.is_ocp_query("tell me a joke", "en-US")[0])
        self.assertFalse(self.ocp.is_ocp_query("who are you", "en-US")[0])
        self.assertFalse(self.ocp.is_ocp_query("you suck", "en-US")[0])

    # ------------------------------------------------------------------ #
    # classify_media (ovos-media-classifier keyword backend)
    # ------------------------------------------------------------------ #
    def test_classify_media_by_keyword(self):
        self.assertEqual(
            self.ocp.classify_media("play some music", "en-US")[0],
            MediaType.MUSIC)
        self.assertEqual(
            self.ocp.classify_media("play a movie", "en-US")[0],
            MediaType.MOVIE)
        self.assertEqual(
            self.ocp.classify_media("play a podcast", "en-US")[0],
            MediaType.PODCAST)
        self.assertIsInstance(
            self.ocp.classify_media("play some music", "en-US")[1], float)

    def test_classify_media_no_keyword_is_generic(self):
        # bare artist name with no media keyword -> GENERIC
        self.assertEqual(
            self.ocp.classify_media("play metallica", "en-US")[0],
            MediaType.GENERIC)


if __name__ == '__main__':
    unittest.main()
