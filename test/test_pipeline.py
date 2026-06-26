"""Tests for OCPPipelineMatcher helpers and OCPPlayerProxy.

The full OCPPipelineMatcher.__init__ is very heavy (intent loading, bus events,
padatious training). All tests here bypass it with __new__ and inject only the
attributes each method under test actually reads.
"""
import unittest
from unittest.mock import MagicMock, patch

from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import PlayerState, MediaState, TrackState, MediaEntry
from mediavocab import MediaType


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_pipeline():
    """Create OCPPipelineMatcher bypassing its __init__."""
    from ocp_pipeline.opm import OCPPipelineMatcher
    from ahocorasick_ner import AhocorasickNER

    p = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
    p.bus = FakeBus()
    p.ocp_sessions = {}
    p.skill_aliases = {}
    p.media2skill = {m: [] for m in MediaType}
    p.ner = AhocorasickNER()
    p.config = {}
    # NOTE: ``lang`` is a read-only property on the base skill class (derived
    # from config/session), so it must not be assigned here. The methods under
    # test receive ``lang`` as an explicit argument and never read ``self.lang``.
    # stub vocab methods so no resource files are needed
    p.voc_match = MagicMock(return_value=False)
    p.remove_voc = MagicMock(side_effect=lambda phrase, _voc, **kw: phrase)
    return p


def _make_message(session_id="default", data=None, context=None):
    from ovos_bus_client.message import Message
    ctx = context or {}
    ctx.setdefault("session", {"session_id": session_id})
    return Message("test", data=data or {}, context=ctx)


# ---------------------------------------------------------------------------
# OCPPlayerProxy
# ---------------------------------------------------------------------------

class TestOCPPlayerProxy(unittest.TestCase):

    def test_required_fields(self):
        from ocp_pipeline.opm import OCPPlayerProxy
        proxy = OCPPlayerProxy(
            session_id="default",
            available_extractors=["yt-dlp"],
            ocp_available=True,
        )
        self.assertEqual(proxy.session_id, "default")
        self.assertEqual(proxy.available_extractors, ["yt-dlp"])
        self.assertTrue(proxy.ocp_available)

    def test_default_states(self):
        from ocp_pipeline.opm import OCPPlayerProxy
        proxy = OCPPlayerProxy(
            session_id="s1",
            available_extractors=[],
            ocp_available=False,
        )
        self.assertEqual(proxy.player_state, PlayerState.STOPPED)
        self.assertEqual(proxy.media_state, MediaState.UNKNOWN)
        self.assertEqual(proxy.media_type, MediaType.GENERIC)
        self.assertIsNone(proxy.skill_id)

    def test_custom_state_stored(self):
        from ocp_pipeline.opm import OCPPlayerProxy
        proxy = OCPPlayerProxy(
            session_id="s2",
            available_extractors=[],
            ocp_available=True,
            player_state=PlayerState.PLAYING,
            media_type=MediaType.MUSIC,
            skill_id="ovos-skill-spotify",
        )
        self.assertEqual(proxy.player_state, PlayerState.PLAYING)
        self.assertEqual(proxy.media_type, MediaType.MUSIC)
        self.assertEqual(proxy.skill_id, "ovos-skill-spotify")


# ---------------------------------------------------------------------------
# _normalize_media_enum
# ---------------------------------------------------------------------------

class TestNormalizeMediaEnum(unittest.TestCase):

    def test_already_enum_is_returned_unchanged(self):
        from ocp_pipeline.opm import OCPPipelineMatcher
        result = OCPPipelineMatcher._normalize_media_enum(MediaType.MUSIC)
        self.assertEqual(result, MediaType.MUSIC)

    def test_value_converted_to_enum(self):
        """mediavocab is a str-enum: its value (e.g. 'music') round-trips."""
        from ocp_pipeline.opm import OCPPipelineMatcher
        result = OCPPipelineMatcher._normalize_media_enum(MediaType.MUSIC.value)
        self.assertEqual(result, MediaType.MUSIC)

    def test_name_converted_to_enum(self):
        from ocp_pipeline.opm import OCPPipelineMatcher
        result = OCPPipelineMatcher._normalize_media_enum("MUSIC")
        self.assertEqual(result, MediaType.MUSIC)

    def test_invalid_token_raises(self):
        from ocp_pipeline.opm import OCPPipelineMatcher
        with self.assertRaises((ValueError, Exception)):
            OCPPipelineMatcher._normalize_media_enum(99999)


# ---------------------------------------------------------------------------
# normalize_results
# ---------------------------------------------------------------------------

class TestNormalizeResults(unittest.TestCase):

    def test_media_entry_passed_through(self):
        from ocp_pipeline.opm import OCPPipelineMatcher
        entry = MediaEntry(uri="http://example.com/t.mp3", title="Test")
        results = [entry]
        out = OCPPipelineMatcher.normalize_results(results)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], entry)

    def test_valid_dict_converted_to_entry(self):
        from ocp_pipeline.opm import OCPPipelineMatcher
        d = {"uri": "http://example.com/t.mp3", "title": "Track",
             "media_type": MediaType.MUSIC.value,
             "playback": 2, "match_confidence": 75}
        out = OCPPipelineMatcher.normalize_results([d])
        self.assertEqual(len(out), 1)

    def test_invalid_dict_dropped(self):
        from ocp_pipeline.opm import OCPPipelineMatcher
        out = OCPPipelineMatcher.normalize_results([{"not_a_track": True}])
        self.assertEqual(out, [])

    def test_none_values_filtered_out(self):
        from ocp_pipeline.opm import OCPPipelineMatcher
        entry = MediaEntry(uri="http://x.com/t.mp3", title="X")
        out = OCPPipelineMatcher.normalize_results([entry, None])
        self.assertEqual(len(out), 1)

    def test_mixed_list(self):
        from ocp_pipeline.opm import OCPPipelineMatcher
        entry = MediaEntry(uri="http://x.com/t.mp3", title="X")
        valid_dict = {"uri": "http://y.com/t.mp3", "title": "Y",
                      "media_type": MediaType.MUSIC.value,
                      "playback": 2, "match_confidence": 60}
        out = OCPPipelineMatcher.normalize_results([entry, valid_dict, {"bad": True}])
        self.assertEqual(len(out), 2)


# ---------------------------------------------------------------------------
# classify_media (embedded mediavocab-native voc_match_media classifier)
# ---------------------------------------------------------------------------

class TestClassifyMedia(unittest.TestCase):

    def test_single_valid_label_returned_immediately(self):
        """When only one media type is valid, classify_media returns it at 1.0."""
        p = _make_pipeline()
        p.media2skill = {MediaType.MUSIC: ["skill-a"]}
        media, conf = p.classify_media("play jazz", "en-us",
                                       valid_labels=[MediaType.MUSIC])
        self.assertEqual(media, MediaType.MUSIC)
        self.assertEqual(conf, 1.0)

    def test_no_voc_match_returns_generic(self):
        p = _make_pipeline()
        # voc_match always returns False → no keyword hit → GENERIC
        p.media2skill = {m: ["skill-x"] for m in MediaType}
        media, conf = p.classify_media("play something", "en-us")
        self.assertEqual(media, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_music_keyword_returns_music(self):
        p = _make_pipeline()
        p.media2skill = {m: ["skill-x"] for m in MediaType}

        def _voc(phrase, vocab, **kw):
            return vocab == "MusicKeyword"

        p.voc_match = _voc
        media, conf = p.classify_media("play some music", "en-us")
        self.assertEqual(media, MediaType.MUSIC)
        self.assertGreater(conf, 0)

    def test_podcast_keyword_returns_podcast(self):
        p = _make_pipeline()
        p.media2skill = {m: ["skill-x"] for m in MediaType}

        def _voc(phrase, vocab, **kw):
            return vocab == "PodcastKeyword"

        p.voc_match = _voc
        media, conf = p.classify_media("play a podcast", "en-us")
        self.assertEqual(media, MediaType.PODCAST)

    def test_radio_keyword_returns_radio(self):
        p = _make_pipeline()
        p.media2skill = {m: ["skill-x"] for m in MediaType}

        def _voc(phrase, vocab, **kw):
            return vocab == "RadioKeyword"

        p.voc_match = _voc
        media, conf = p.classify_media("play radio", "en-us")
        self.assertEqual(media, MediaType.RADIO)

    def test_anime_keyword_converges_to_episodic_series(self):
        """Taxonomy convergence: an AnimeKeyword hit classifies to mediavocab
        EPISODIC_SERIES (mediavocab has no separate ANIME label)."""
        p = _make_pipeline()
        p.media2skill = {m: ["skill-x"] for m in MediaType}

        def _voc(phrase, vocab, **kw):
            return vocab == "AnimeKeyword"

        p.voc_match = _voc
        media, _ = p.classify_media("i want to watch an anime", "en-us")
        self.assertEqual(media, MediaType.EPISODIC_SERIES)

    def test_documentary_keyword_converges_to_movie(self):
        """DocumentaryKeyword -> mediavocab MOVIE (no DOCUMENTARY label)."""
        p = _make_pipeline()
        p.media2skill = {m: ["skill-x"] for m in MediaType}

        def _voc(phrase, vocab, **kw):
            return vocab == "DocumentaryKeyword"

        p.voc_match = _voc
        media, _ = p.classify_media("show me a documentary", "en-us")
        self.assertEqual(media, MediaType.MOVIE)

    def test_valid_labels_filter_limits_candidates(self):
        """If valid_labels excludes a type, that type must not be returned."""
        p = _make_pipeline()
        # Only PODCAST is a valid label
        valid = [MediaType.PODCAST]
        # voc_match returns True for everything
        p.voc_match = MagicMock(return_value=True)
        media, _ = p.classify_media("play music", "en-us", valid_labels=valid)
        self.assertNotEqual(media, MediaType.MUSIC)


# ---------------------------------------------------------------------------
# is_ocp_query
# ---------------------------------------------------------------------------

class TestIsOcpQuery(unittest.TestCase):

    def test_generic_result_means_not_ocp(self):
        p = _make_pipeline()
        # no voc match → GENERIC → False
        p.media2skill = {m: ["skill-x"] for m in MediaType}
        is_ocp, _ = p.is_ocp_query("hello world", "en-us")
        self.assertFalse(is_ocp)

    def test_specific_media_means_is_ocp(self):
        p = _make_pipeline()
        p.media2skill = {m: ["skill-x"] for m in MediaType}

        def _voc(phrase, vocab, **kw):
            return vocab == "MusicKeyword"

        p.voc_match = _voc
        is_ocp, conf = p.is_ocp_query("play some music", "en-us")
        self.assertTrue(is_ocp)
        self.assertGreater(conf, 0)


# ---------------------------------------------------------------------------
# handle_skill_register
# ---------------------------------------------------------------------------

class TestHandleSkillRegister(unittest.TestCase):

    def _register_skill(self, pipeline, skill_id="test.skill",
                        media_types=None, aliases=None):
        from ovos_bus_client.message import Message
        msg = Message("ovos.common_play.announce", data={
            "skill_id": skill_id,
            "skill_name": "Test Skill",
            "media_types": media_types or [MediaType.MUSIC.value],
            "aliases": aliases or ["Test Skill"],
            "featured_tracks": False,
            "thumbnail": "",
        })
        pipeline.handle_skill_register(msg)

    def test_skill_added_to_media2skill(self):
        p = _make_pipeline()
        self._register_skill(p, skill_id="music.skill",
                             media_types=[MediaType.MUSIC.value])
        self.assertIn("music.skill", p.media2skill[MediaType.MUSIC])

    def test_skill_aliases_stored(self):
        p = _make_pipeline()
        self._register_skill(p, skill_id="jazz.skill",
                             aliases=["Jazz", "The Jazz App"])
        self.assertEqual(p.skill_aliases["jazz.skill"], ["Jazz", "The Jazz App"])

    def test_multiple_media_types_registered(self):
        p = _make_pipeline()
        self._register_skill(p, skill_id="av.skill",
                             media_types=[MediaType.MUSIC.value,
                                          MediaType.PODCAST.value])
        self.assertIn("av.skill", p.media2skill[MediaType.MUSIC])
        self.assertIn("av.skill", p.media2skill[MediaType.PODCAST])

    def test_music_alias_added_to_ner(self):
        p = _make_pipeline()
        # Confirm that an alias for a MUSIC skill is added to the NER
        self._register_skill(p, skill_id="spotify.skill",
                             media_types=[MediaType.MUSIC.value],
                             aliases=["Spotify"])
        # The NER should be able to tag "Spotify" as music_streaming_service
        tags = p.ner.tag("play Spotify")
        labels = {t["label"] for t in tags}
        self.assertIn("music_streaming_service", labels)

    def test_invalid_media_type_skipped(self):
        p = _make_pipeline()
        from ovos_bus_client.message import Message
        msg = Message("ovos.common_play.announce", data={
            "skill_id": "bad.skill",
            "skill_name": "Bad Skill",
            "media_types": [99999],  # invalid
            "aliases": ["Bad Skill"],
            "featured_tracks": False,
            "thumbnail": "",
        })
        # Must not raise even with an invalid media type
        p.handle_skill_register(msg)


# ---------------------------------------------------------------------------
# handle_skill_keyword_register
# ---------------------------------------------------------------------------

class TestHandleSkillKeywordRegister(unittest.TestCase):

    def test_samples_added_to_ner(self):
        p = _make_pipeline()
        from ovos_bus_client.message import Message
        msg = Message("ovos.common_play.register_keyword", data={
            "skill_id": "music.skill",
            "label": "music_streaming_service",
            "media_type": MediaType.MUSIC.value,
            "samples": ["BandCamp", "SoundCloud"],
        })
        p.handle_skill_keyword_register(msg)
        tags = p.ner.tag("play BandCamp")
        labels = {t["label"] for t in tags}
        self.assertIn("music_streaming_service", labels)

    def test_empty_samples_no_error(self):
        p = _make_pipeline()
        from ovos_bus_client.message import Message
        msg = Message("ovos.common_play.register_keyword", data={
            "skill_id": "music.skill",
            "label": "music_streaming_service",
            "media_type": MediaType.MUSIC.value,
            "samples": [],
        })
        p.handle_skill_keyword_register(msg)  # must not raise


# ---------------------------------------------------------------------------
# _extract_entities (NER guard)
# ---------------------------------------------------------------------------

class TestExtractEntities(unittest.TestCase):

    def test_empty_ner_returns_empty_dict(self):
        """An empty NER must not raise 'Not an Aho-Corasick automaton yet'."""
        p = _make_pipeline()
        # nothing registered -> empty automaton
        self.assertEqual(p._extract_entities("play some music"), {})

    def test_registered_entity_extracted(self):
        p = _make_pipeline()
        p.ner.add_word("music_streaming_service", "spotify")
        ents = p._extract_entities("play music on spotify")
        self.assertEqual(ents.get("music_streaming_service"), "spotify")


# ---------------------------------------------------------------------------
# get_player / update_player_proxy
# ---------------------------------------------------------------------------

class TestGetPlayer(unittest.TestCase):

    def test_new_session_creates_proxy(self):
        p = _make_pipeline()
        p.config = {"legacy": True}  # skip SEI sync
        msg = _make_message(session_id="sess-abc")
        player = p.get_player(msg)
        self.assertEqual(player.session_id, "sess-abc")
        self.assertIn("sess-abc", p.ocp_sessions)

    def test_existing_session_returned(self):
        from ocp_pipeline.opm import OCPPlayerProxy
        p = _make_pipeline()
        p.config = {"legacy": True}
        existing = OCPPlayerProxy(
            session_id="sess-xyz",
            available_extractors=[],
            ocp_available=True,
            player_state=PlayerState.PLAYING,
        )
        p.ocp_sessions["sess-xyz"] = existing
        msg = _make_message(session_id="sess-xyz")
        player = p.get_player(msg)
        self.assertIs(player, existing)

    def test_update_player_proxy_stores(self):
        from ocp_pipeline.opm import OCPPlayerProxy
        p = _make_pipeline()
        proxy = OCPPlayerProxy(
            session_id="s99",
            available_extractors=[],
            ocp_available=False,
        )
        p.update_player_proxy(proxy)
        self.assertIn("s99", p.ocp_sessions)
        self.assertIs(p.ocp_sessions["s99"], proxy)

    def test_default_session_id_used_without_message(self):
        p = _make_pipeline()
        p.config = {"legacy": True}
        player = p.get_player(None)
        self.assertEqual(player.session_id, "default")


# ---------------------------------------------------------------------------
# handle_player_state_update
# ---------------------------------------------------------------------------

class TestHandlePlayerStateUpdate(unittest.TestCase):

    def test_player_state_updated(self):
        from ocp_pipeline.opm import OCPPlayerProxy
        p = _make_pipeline()
        p.config = {"legacy": True}
        proxy = OCPPlayerProxy(session_id="default", available_extractors=[],
                               ocp_available=True,
                               player_state=PlayerState.STOPPED)
        p.ocp_sessions["default"] = proxy

        msg = _make_message(session_id="default",
                            data={"player_state": int(PlayerState.PLAYING)})
        p.handle_player_state_update(msg)
        self.assertEqual(p.ocp_sessions["default"].player_state,
                         PlayerState.PLAYING)

    def test_media_state_updated(self):
        from ocp_pipeline.opm import OCPPlayerProxy
        p = _make_pipeline()
        p.config = {"legacy": True}
        proxy = OCPPlayerProxy(session_id="default", available_extractors=[],
                               ocp_available=True)
        p.ocp_sessions["default"] = proxy

        msg = _make_message(session_id="default",
                            data={"media_state": int(MediaState.BUFFERED_MEDIA)})
        p.handle_player_state_update(msg)
        self.assertEqual(p.ocp_sessions["default"].media_state,
                         MediaState.BUFFERED_MEDIA)

    def test_missing_field_not_applied(self):
        """A state update with only player_state must not touch media_state."""
        from ocp_pipeline.opm import OCPPlayerProxy
        p = _make_pipeline()
        p.config = {"legacy": True}
        proxy = OCPPlayerProxy(session_id="default", available_extractors=[],
                               ocp_available=True,
                               media_state=MediaState.UNKNOWN)
        p.ocp_sessions["default"] = proxy

        msg = _make_message(session_id="default",
                            data={"player_state": int(PlayerState.PLAYING)})
        p.handle_player_state_update(msg)
        self.assertEqual(p.ocp_sessions["default"].media_state,
                         MediaState.UNKNOWN)


# ---------------------------------------------------------------------------
# handle_track_state_update
# ---------------------------------------------------------------------------

class TestHandleTrackStateUpdate(unittest.TestCase):

    def test_playing_audio_sets_player_playing(self):
        from ocp_pipeline.opm import OCPPlayerProxy
        p = _make_pipeline()
        p.config = {"legacy": True}
        proxy = OCPPlayerProxy(session_id="default", available_extractors=[],
                               ocp_available=True,
                               player_state=PlayerState.STOPPED)
        p.ocp_sessions["default"] = proxy

        msg = _make_message(session_id="default",
                            data={"state": int(TrackState.PLAYING_AUDIO)})
        p.handle_track_state_update(msg)
        self.assertEqual(p.ocp_sessions["default"].player_state,
                         PlayerState.PLAYING)

    def test_missing_state_raises(self):
        p = _make_pipeline()
        p.config = {"legacy": True}
        msg = _make_message(data={})
        with self.assertRaises(ValueError):
            p.handle_track_state_update(msg)


# ---------------------------------------------------------------------------
# _update_player_skill_id
# ---------------------------------------------------------------------------

class TestUpdatePlayerSkillId(unittest.TestCase):

    def test_skill_id_from_message_data(self):
        from ocp_pipeline.opm import OCPPipelineMatcher, OCPPlayerProxy
        proxy = OCPPlayerProxy(session_id="s", available_extractors=[],
                               ocp_available=True)
        msg = _make_message(data={"skill_id": "ovos-skill-spotify"})
        result = OCPPipelineMatcher._update_player_skill_id(proxy, msg)
        self.assertEqual(result.skill_id, "ovos-skill-spotify")

    def test_skill_id_from_context(self):
        from ocp_pipeline.opm import OCPPipelineMatcher, OCPPlayerProxy
        proxy = OCPPlayerProxy(session_id="s", available_extractors=[],
                               ocp_available=True)
        from ovos_bus_client.message import Message
        msg = Message("test", data={},
                      context={"skill_id": "ovos-skill-youtube",
                               "session": {"session_id": "default"}})
        result = OCPPipelineMatcher._update_player_skill_id(proxy, msg)
        self.assertEqual(result.skill_id, "ovos-skill-youtube")

    def test_ocp_id_not_stored_as_skill_id(self):
        from ocp_pipeline.opm import OCPPipelineMatcher, OCPPlayerProxy, OCP_ID
        proxy = OCPPlayerProxy(session_id="s", available_extractors=[],
                               ocp_available=True, skill_id="original-skill")
        msg = _make_message(data={"skill_id": OCP_ID})
        result = OCPPipelineMatcher._update_player_skill_id(proxy, msg)
        # OCP_ID must not overwrite the real skill_id
        self.assertEqual(result.skill_id, "original-skill")


if __name__ == "__main__":
    unittest.main()
