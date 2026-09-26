"""Tests for the context-aware classifier integration.

The pipeline builds the standalone classifier's two minimal context inputs —
player_status (from the now-playing proxy) and ner_list (from registered skill
keywords) — and passes them to ``classify_full``.  These cover the bridge
(context builders, the unified ``media_type_map`` mapping, the context-aware
``classify`` / ``domain_of``, rich ``to_signals``, the content filter) and the
wiring into ``classify_media``.
"""
import unittest

from ovos_utils.ocp import MediaType, PlayerState

from ahocorasick_ner import AhocorasickNER
from ocp_pipeline.context_classify import (
    ContextAwareClassifier,
    build_ner_list,
    build_player_status,
    legacy_to_mv,
    mv_to_legacy,
)
# the single shared mapping lives in media_type_map; context_classify re-exports it
from ocp_pipeline.media_type_map import (
    legacy_to_mv as map_legacy_to_mv,
    mv_to_legacy as map_mv_to_legacy,
)
from ocp_pipeline.opm import OCPPipelineMatcher
from mediavocab import MediaType as MVMediaType


class TestMappingUnified(unittest.TestCase):
    """context_classify must reuse the single media_type_map (no duplicate)."""

    def test_mapping_helpers_are_the_shared_ones(self):
        self.assertIs(mv_to_legacy, map_mv_to_legacy)
        self.assertIs(legacy_to_mv, map_legacy_to_mv)

    def test_legacy_to_mv(self):
        self.assertEqual(legacy_to_mv(MediaType.MUSIC), MVMediaType.MUSIC)
        self.assertEqual(legacy_to_mv(MediaType.VIDEO_EPISODES),
                         MVMediaType.EPISODIC_SERIES)
        self.assertIsNone(legacy_to_mv(None))


class _FakePlayer:
    def __init__(self, state, media_type):
        self.player_state = state
        self.media_type = media_type


class TestContextBuilders(unittest.TestCase):
    def test_build_player_status_playing(self):
        ps = build_player_status(_FakePlayer(PlayerState.PLAYING, MediaType.MUSIC))
        self.assertTrue(ps.is_active)
        self.assertEqual(ps.now_playing, MVMediaType.MUSIC)

    def test_build_player_status_none(self):
        self.assertIsNone(build_player_status(None))

    def test_build_player_status_generic_is_no_now_playing(self):
        ps = build_player_status(_FakePlayer(PlayerState.PLAYING, MediaType.GENERIC))
        self.assertIsNone(ps.now_playing)

    def test_build_ner_list_empty(self):
        self.assertEqual(build_ner_list(AhocorasickNER()), {})

    def test_build_ner_list_none(self):
        self.assertEqual(build_ner_list(None), {})

    def test_build_ner_list_from_registered_keywords(self):
        ner = AhocorasickNER()
        ner.add_word("anime_title", "Attack on Titan")
        ner.add_word("artist_name", "Radiohead")
        out = build_ner_list(ner)
        self.assertIn("anime_title", out)
        self.assertIn("artist_name", out)


class TestContextAwareClassify(unittest.TestCase):
    def setUp(self):
        self.c = ContextAwareClassifier()

    def _playing(self, mt=MediaType.MUSIC):
        return build_player_status(_FakePlayer(PlayerState.PLAYING, mt))

    def test_relative_control_with_active_session(self):
        domain = self.c.domain_of("next", "en-us",
                                  player_status=self._playing())
        self.assertEqual(domain, "ocp_control")

    def test_play_something_else_requeries_current_type(self):
        domain = self.c.domain_of("play something else", "en-us",
                                  player_status=self._playing(MediaType.MUSIC))
        self.assertEqual(domain, "ocp_play")
        mt, _conf = self.c.classify("play something else", "en-us",
                                    player_status=self._playing(MediaType.MUSIC))
        self.assertEqual(mt, MediaType.MUSIC)

    def test_explicit_route_wins_over_context(self):
        mt, _conf = self.c.classify("play a movie", "en-us",
                                    player_status=self._playing(MediaType.MUSIC))
        self.assertEqual(mt, MediaType.MOVIE)

    def test_ner_list_threaded_without_error(self):
        # the keyword default backend has no entity stream, so ner_list is inert,
        # but threading it must not raise and must keep the explicit route.
        mt, _c = self.c.classify("play some music", "en-us",
                                 ner_list={"artist_name": ["whoever"]})
        self.assertEqual(mt, MediaType.MUSIC)

    def test_classify_returns_two_tuple(self):
        out = self.c.classify("play some music", "en-us")
        self.assertEqual(len(out), 2)
        mt, conf = out
        self.assertEqual(mt, MediaType.MUSIC)
        self.assertIsInstance(conf, float)


class TestSignals(unittest.TestCase):
    def setUp(self):
        self.c = ContextAwareClassifier()

    def test_to_signals_carries_medium(self):
        signals = self.c.to_signals("play a movie", "en-us")
        # mediavocab.Signals.medium is the leaf MediaType (mediavocab vocab)
        self.assertEqual(getattr(signals, "medium", None), MVMediaType.MOVIE)

    def test_to_signals_has_axis_fields(self):
        signals = self.c.to_signals("play some music", "en-us")
        for axis in ("playback_type", "content_genres", "content_form",
                     "programme_format", "variant_kind", "accessibility",
                     "picture_format"):
            self.assertTrue(hasattr(signals, axis), f"missing Signals.{axis}")


class TestContentFilter(unittest.TestCase):
    def test_adult_blocked_by_default(self):
        c = ContextAwareClassifier()
        blocked, _reason = c.is_blocked("play some porn", "en-us")
        self.assertTrue(blocked)

    def test_adult_allowed_when_configured(self):
        c = ContextAwareClassifier(config={"allow_adult_content": True})
        blocked, _reason = c.is_blocked("play some porn", "en-us")
        self.assertFalse(blocked)

    def test_clean_query_not_blocked(self):
        c = ContextAwareClassifier()
        blocked, _reason = c.is_blocked("play some music", "en-us")
        self.assertFalse(blocked)

    def test_filter_can_be_disabled(self):
        c = ContextAwareClassifier(
            config={"media_content_filter": {"enabled": False}})
        blocked, _reason = c.is_blocked("play some porn", "en-us")
        self.assertFalse(blocked)


class TestConfigToggles(unittest.TestCase):
    def test_default_backend_is_keyword(self):
        from ovos_media_classifier.keyword import KeywordMediaClassifier
        c = ContextAwareClassifier()
        self.assertIsInstance(c.clf, KeywordMediaClassifier)

    def test_unknown_plugin_falls_back_to_keyword(self):
        # an unresolvable plugin name must not crash; load_media_classifier falls
        # back to the zero-ML keyword floor.
        from ovos_media_classifier.keyword import KeywordMediaClassifier
        c = ContextAwareClassifier(
            config={"media_classifier_plugin": "does-not-exist"})
        self.assertIsInstance(c.clf, KeywordMediaClassifier)

    def test_voc_match_func_is_honored(self):
        calls = []

        def fake_voc(phrase, vocab, lang="en-us"):
            calls.append((vocab, lang))
            return False

        c = ContextAwareClassifier(voc_match_func=fake_voc)
        c.classify("play something", "en-us")
        self.assertTrue(calls, "the injected voc_match_func was never used")


class TestClassifyMediaIntegration(unittest.TestCase):
    def setUp(self):
        self.ocp = OCPPipelineMatcher(config={})
        self.ocp.skill_aliases["test"] = ["Test Skill"]

    def test_classify_media_routes_explicit(self):
        self.assertEqual(
            self.ocp.classify_media("play some music", "en-US")[0],
            MediaType.MUSIC)
        self.assertEqual(
            self.ocp.classify_media("play a movie", "en-US")[0],
            MediaType.MOVIE)

    def test_classify_media_generic_when_no_keyword(self):
        self.assertEqual(
            self.ocp.classify_media("play metallica", "en-US")[0],
            MediaType.GENERIC)

    def test_classifier_context_builds_inputs(self):
        self.ocp.ner.add_word("anime_title", "Cowboy Bebop")
        player_status, ner_list = self.ocp._classifier_context()
        self.assertIn("anime_title", ner_list)
        # player_status may be None (no active session in a bare matcher)

    def test_media_signals_dict_exposed(self):
        signals = self.ocp.media_signals("play a movie", "en-us")
        self.assertIsInstance(signals, dict)
        # the legacy leaf folds movie -> movie; signals carry the mediavocab medium
        self.assertIn("medium", signals)

    def test_is_blocked_content_blocks_adult_by_default(self):
        blocked, _reason = self.ocp.is_blocked_content("play some porn", "en-us")
        self.assertTrue(blocked)


if __name__ == "__main__":
    unittest.main()
