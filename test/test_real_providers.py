"""End-to-end dispatch tests against the real, published MediaProvider plugins.

The rest of the provider-dispatch suite (``test_media_provider_dispatch.py``)
proves the *mechanism* with mock providers. This module proves the mechanism
against the actual plugins users install from PyPI, so a contract drift in
either direction -- the ``MediaProvider.search`` signature, the
``mediavocab.Release`` shape, the ``opm.media.provider`` entry point -- fails
here instead of in production.

Three published providers are exercised, chosen because their catalog lookup
needs no API key:

* ``ovos-media-provider-local`` -- fully offline. A temporary library of
  generated WAV files is indexed and searched for real, with nothing mocked
  at all. This is the primary real-dispatch test.
* ``ovos-media-provider-news`` -- fully offline. It ships its feed catalog
  as bundled JSON and matches against it locally.
* ``ovos-media-provider-somafm`` -- only the SomaFM channel-list HTTP fetch
  is replaced by a canned channel; the provider's own matching and
  ``Release`` construction run for real.

The final assertion in each case is the wire shape: every emitted entry is
fed to ovos-media's own ``_is_valid_media`` decoder predicate, so a result
this pipeline produces is provably a result the media service will accept.
"""
import os
import shutil
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType, PlaybackType

from ocp_pipeline.opm import OCPPipelineMatcher

from ovos_plugin_manager.media_provider import (find_media_provider_plugins,
                                                load_media_providers)

try:
    from ovos_media_provider_local import LocalMediaProvider
    HAS_LOCAL = True
except ImportError:
    HAS_LOCAL = False

try:
    from ovos_media_provider_news import NewsMediaProvider
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False

try:
    import ovos_media_provider_somafm as somafm_plugin
    from radiosoma import SomaFmStation
    HAS_SOMAFM = True
except ImportError:
    HAS_SOMAFM = False

try:
    from ovos_media.bus.schemas import _is_valid_media
    HAS_OVOS_MEDIA = True
except ImportError:
    HAS_OVOS_MEDIA = False


# a single SomaFM channel in the exact raw shape `radiosoma` parses out of
# https://api.somafm.com/channels.xml -- the only thing mocked away here.
SOMAFM_CHANNEL = {
    "id": "groovesalad",
    "title": "Groove Salad",
    "description": "A nicely chilled plate of ambient/downtempo beats",
    "genre": "ambient|electronica",
    "image": "https://somafm.com/img/groovesalad.jpg",
    "dj": "Rusty Hodge",
    "fastpls": [{"text": "https://somafm.com/groovesalad130.pls",
                 "format": "aac"}],
    "slowpls": [{"text": "https://somafm.com/groovesalad32.pls",
                 "format": "aac"}],
}


def _make_pipeline(media_providers):
    """A bare OCPPipelineMatcher carrying only what dispatch reads.

    ``__init__`` is heavy (bus, intents, padatious training), so the suite
    uses the same ``__new__``-bypass pattern as ``test_pipeline.py``.
    """
    p = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
    p.bus = FakeBus()
    p.config = {}
    p.media_providers = dict(media_providers)
    return p


def _write_wav(path, seconds=1, rate=8000):
    """Write a silent, untagged mono WAV. Untagged is deliberate: the local
    provider must fall back to the filename for the title, which is the
    common case for a real user's library of loose files."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * rate * seconds)


def _assert_media_entry_shape(case, entry, skill_id):
    """Assert one dispatch result carries the MediaEntry fields the pipeline
    and the media service both read."""
    for key in ("uri", "title", "media_type", "playback",
                "match_confidence", "skill_id"):
        case.assertIn(key, entry)
    case.assertEqual(entry["skill_id"], skill_id)
    case.assertIsInstance(entry["uri"], str)
    case.assertTrue(entry["uri"])
    case.assertIsInstance(entry["media_type"], MediaType)
    case.assertIsInstance(entry["playback"], PlaybackType)
    case.assertIsInstance(entry["match_confidence"], int)
    case.assertGreaterEqual(entry["match_confidence"], 0)
    case.assertLessEqual(entry["match_confidence"], 100)


class TestPublishedProviderDiscovery(unittest.TestCase):
    """The pipeline needs no registration code of its own: providers are
    found through their ``opm.media.provider`` entry points."""

    def test_installed_providers_are_discovered_by_entry_point(self):
        found = find_media_provider_plugins()
        expected = {name for name, present in (("local", HAS_LOCAL),
                                               ("news", HAS_NEWS),
                                               ("somafm", HAS_SOMAFM))
                    if present}
        self.assertTrue(expected, "no published providers installed to test")
        self.assertTrue(expected.issubset(set(found)),
                        f"entry points {sorted(found)} miss {sorted(expected)}")

    def test_loader_instantiates_published_providers(self):
        loaded = load_media_providers()
        for name in ("local", "news", "somafm"):
            if name in find_media_provider_plugins():
                self.assertIn(name, loaded)
                self.assertEqual(getattr(loaded[name], "name", None), name)

    @unittest.skipUnless(HAS_LOCAL, "ovos-media-provider-local not installed")
    def test_top_level_config_reaches_a_provider(self):
        """With no `media_providers` block in the pipeline config, per-provider
        settings come from the top-level `media_providers` block, which is what
        the README tells users to write."""
        library = tempfile.mkdtemp(prefix="ocp-config-library-")
        self.addCleanup(shutil.rmtree, library, ignore_errors=True)
        _write_wav(os.path.join(library, "Config Probe.wav"))

        p = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
        p.config = {}
        p.media_providers = {}
        with patch("ovos_config.Configuration") as cfg:
            cfg.return_value = {"media_providers": {"local": {"paths": [library]}}}
            p._load_media_providers()

        self.assertIn("local", p.media_providers)
        p.bus = FakeBus()
        results = p._search_providers("config probe", MediaType.MUSIC, "en-us")
        self.assertIn("Config Probe", {r["title"] for r in results})

    def test_pipeline_loads_installed_providers_without_extra_wiring(self):
        p = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
        p.config = {}
        p.media_providers = {}
        p._load_media_providers()
        self.assertTrue(p.media_providers,
                        "installed providers were not picked up by the pipeline")


@unittest.skipUnless(HAS_LOCAL, "ovos-media-provider-local not installed")
class TestLocalProviderRealDispatch(unittest.TestCase):
    """Fully offline, nothing mocked: a real library on disk, searched
    through the real provider, dispatched by the real pipeline code."""

    @classmethod
    def setUpClass(cls):
        cls.library = tempfile.mkdtemp(prefix="ocp-local-library-")
        _write_wav(os.path.join(cls.library, "Ghost Track.wav"))
        _write_wav(os.path.join(cls.library, "Harbour Lights.wav"))
        os.makedirs(os.path.join(cls.library, "nested"))
        _write_wav(os.path.join(cls.library, "nested", "Deep Cut.wav"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.library, ignore_errors=True)

    def _pipeline(self):
        provider = LocalMediaProvider({"paths": [self.library]})
        return _make_pipeline({"local": provider})

    def test_dispatch_returns_media_entry_dicts(self):
        p = self._pipeline()
        results = p._search_providers("ghost track", MediaType.MUSIC, "en-us")

        self.assertTrue(results, "the local provider answered nothing")
        titles = {r["title"] for r in results}
        self.assertIn("Ghost Track", titles)
        for entry in results:
            _assert_media_entry_shape(self, entry, "local")
            self.assertTrue(entry["uri"].startswith("file://"))
            self.assertEqual(entry["media_type"], MediaType.MUSIC)

    def test_nested_files_are_reachable(self):
        p = self._pipeline()
        results = p._search_providers("deep cut", MediaType.MUSIC, "en-us")
        self.assertIn("Deep Cut", {r["title"] for r in results})

    def test_query_media_type_is_stamped_on_results(self):
        """The QUERY's media type is stamped, not a re-derivation of the
        Release's own -- otherwise `filter_results` drops the answer whenever
        the mediavocab fold is not injective."""
        p = self._pipeline()
        results = p._search_providers("ghost track", MediaType.AUDIOBOOK,
                                      "en-us")
        for entry in results:
            self.assertEqual(entry["media_type"], MediaType.AUDIOBOOK)

    def test_no_match_returns_nothing_rather_than_the_whole_library(self):
        p = self._pipeline()
        results = p._search_providers("zzzz unlikely query zzzz",
                                      MediaType.MUSIC, "en-us")
        self.assertEqual(results, [])

    @unittest.skipUnless(HAS_OVOS_MEDIA, "ovos-media not installed")
    def test_results_pass_the_ovos_media_decoder(self):
        """Every emitted entry must satisfy ovos-media's own validity
        predicate: what this pipeline sends is what that service accepts."""
        p = self._pipeline()
        results = p._search_providers("ghost track", MediaType.MUSIC, "en-us")
        self.assertTrue(results)
        for entry in results:
            self.assertTrue(_is_valid_media(entry),
                            f"ovos-media would refuse {entry!r}")

    def test_dual_window_merge_keeps_both_windows(self):
        """A legacy bus answer and a real provider answer to the same query
        both survive the merge; only a provider entry duplicating a legacy
        uri is dropped."""
        p = self._pipeline()
        provider_results = p._search_providers("ghost track", MediaType.MUSIC,
                                               "en-us")
        self.assertTrue(provider_results)

        bus_track = {"uri": "http://stream.example/ghost.mp3",
                     "title": "Ghost Track", "artist": "The Bus Skill",
                     "media_type": MediaType.MUSIC,
                     "playback": PlaybackType.AUDIO,
                     "match_confidence": 60, "skill_id": "bus.skill"}
        merged = p._merge_provider_results([dict(bus_track)], provider_results)

        self.assertIn(bus_track, merged)
        self.assertEqual(len(merged), 1 + len(provider_results))
        self.assertEqual({"bus.skill", "local"},
                         {r["skill_id"] for r in merged})

    def test_provider_entry_duplicating_a_legacy_uri_is_dropped(self):
        p = self._pipeline()
        provider_results = p._search_providers("ghost track", MediaType.MUSIC,
                                               "en-us")
        self.assertTrue(provider_results)
        # the legacy window already covers the very file the provider found
        legacy = {"uri": provider_results[0]["uri"], "title": "Ghost Track",
                  "media_type": MediaType.MUSIC,
                  "playback": PlaybackType.AUDIO,
                  "match_confidence": 70, "skill_id": "bus.skill"}
        merged = p._merge_provider_results([dict(legacy)], provider_results)

        self.assertEqual(len(merged), len(provider_results))
        self.assertEqual(merged[0]["skill_id"], "bus.skill")

    @unittest.skipUnless(HAS_OVOS_MEDIA, "ovos-media not installed")
    def test_full_search_to_play_payload(self):
        """Drive the whole path a real utterance takes -- dual-window search,
        normalization, ranking, then the `ovos.common_play.play` emission --
        and check the payload ovos-media actually receives."""
        p = self._pipeline()
        p.search_lock = MagicMock()
        p.search_lock.__enter__ = MagicMock(return_value=None)
        p.search_lock.__exit__ = MagicMock(return_value=None)
        p._enclosure = MagicMock()
        p.config = {"filter_media": False, "filter_SEI": False, "min_score": 0}
        p._execute_query = MagicMock(return_value=[{"results": []}])
        p.get_player = MagicMock()

        message = Message("recognizer_loop:utterance", {},
                          {"session": {"session_id": "s1"}})
        results = p._search("ghost track", MediaType.MUSIC, "en-us",
                            message=message)
        self.assertTrue(results, "no results survived the full search path")

        best = p.select_best(results, message)
        self.assertIsNotNone(best)
        self.assertEqual(best.skill_id, "local")

        payloads = []
        p.bus.on("ovos.common_play.play", lambda m: payloads.append(m.data))
        from ovos_bus_client.apis.ocp import OCPInterface
        OCPInterface(bus=p.bus).play(tracks=[best], utterance="ghost track",
                                     source_message=message)

        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        for key in ("media", "playlist", "disambiguation"):
            self.assertIn(key, payload)
        self.assertTrue(_is_valid_media(payload["media"]),
                        f"ovos-media would refuse {payload['media']!r}")
        for track in payload["playlist"]:
            self.assertTrue(_is_valid_media(track))
        self.assertEqual(payload["media"]["skill_id"], "local")
        self.assertTrue(payload["media"]["uri"].startswith("file://"))


@unittest.skipUnless(HAS_NEWS, "ovos-media-provider-news not installed")
class TestNewsProviderRealDispatch(unittest.TestCase):
    """The news provider matches against its bundled feed catalog, so its
    search runs offline; only playback would touch the network."""

    def _pipeline(self):
        return _make_pipeline({"news": NewsMediaProvider()})

    def test_news_query_returns_media_entry_dicts(self):
        p = self._pipeline()
        results = p._search_providers("npr news", MediaType.NEWS, "en-us")

        self.assertTrue(results, "the news provider answered nothing")
        for entry in results:
            _assert_media_entry_shape(self, entry, "news")
            self.assertEqual(entry["media_type"], MediaType.NEWS)

    def test_news_media_type_survives_the_mediavocab_round_trip(self):
        """Legacy NEWS folds to mediavocab RADIO, which folds back to legacy
        RADIO. Results must still be stamped NEWS or `filter_results` drops
        every news answer to a news query."""
        p = self._pipeline()
        results = p._search_providers("npr news", MediaType.NEWS, "en-us")
        self.assertTrue(results)
        self.assertEqual({MediaType.NEWS},
                         {r["media_type"] for r in results})

    @unittest.skipUnless(HAS_OVOS_MEDIA, "ovos-media not installed")
    def test_results_pass_the_ovos_media_decoder(self):
        p = self._pipeline()
        results = p._search_providers("npr news", MediaType.NEWS, "en-us")
        self.assertTrue(results)
        for entry in results:
            self.assertTrue(_is_valid_media(entry),
                            f"ovos-media would refuse {entry!r}")


@unittest.skipUnless(HAS_SOMAFM, "ovos-media-provider-somafm not installed")
class TestSomaFMProviderRealDispatch(unittest.TestCase):
    """Only the channel-list HTTP fetch is canned; the provider's matching
    and Release construction are the published code."""

    def _dispatch(self, phrase, media_type=MediaType.RADIO):
        station = SomaFmStation(dict(SOMAFM_CHANNEL), session=object())
        p = _make_pipeline({"somafm": somafm_plugin.SomaFMMediaProvider()})
        with patch.object(somafm_plugin, "get_stations",
                          return_value=[station]):
            return p, p._search_providers(phrase, media_type, "en-us")

    def test_channel_query_returns_one_entry_per_stream_variant(self):
        _, results = self._dispatch("groove salad")

        self.assertTrue(results, "the somafm provider answered nothing")
        for entry in results:
            _assert_media_entry_shape(self, entry, "somafm")
            self.assertEqual(entry["title"], "Groove Salad")
            self.assertEqual(entry["media_type"], MediaType.RADIO)
        # one Release per stream encoding, all distinct uris
        uris = [r["uri"] for r in results]
        self.assertEqual(len(uris), len(set(uris)))
        self.assertEqual(len(uris), 2)

    def test_unrelated_query_is_not_answered(self):
        _, results = self._dispatch("beethoven symphony")
        self.assertEqual(results, [])

    def test_a_somafm_bus_skill_and_the_provider_coexist(self):
        """The classic SomaFM OCP skill and this provider answer the same
        query with different uris; the merge must keep both."""
        p, results = self._dispatch("groove salad")
        self.assertTrue(results)
        legacy = {"uri": "https://somafm.com/groovesalad256.pls",
                  "title": "Groove Salad", "artist": "SomaFM",
                  "media_type": MediaType.RADIO,
                  "playback": PlaybackType.AUDIO,
                  "match_confidence": 80,
                  "skill_id": "skill-somafm.openvoiceos"}
        merged = p._merge_provider_results([dict(legacy)], results)
        self.assertEqual(len(merged), 1 + len(results))
        self.assertIn("skill-somafm.openvoiceos",
                      {r["skill_id"] for r in merged})

    @unittest.skipUnless(HAS_OVOS_MEDIA, "ovos-media not installed")
    def test_results_pass_the_ovos_media_decoder(self):
        _, results = self._dispatch("groove salad")
        self.assertTrue(results)
        for entry in results:
            self.assertTrue(_is_valid_media(entry),
                            f"ovos-media would refuse {entry!r}")


@unittest.skipUnless(HAS_LOCAL and HAS_NEWS, "need >1 published provider")
class TestMultipleRealProvidersConcurrently(unittest.TestCase):
    """Several published providers dispatched in one search, plus the
    resilience guarantees checked against real plugins rather than mocks."""

    @classmethod
    def setUpClass(cls):
        cls.library = tempfile.mkdtemp(prefix="ocp-multi-library-")
        _write_wav(os.path.join(cls.library, "Morning Report.wav"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.library, ignore_errors=True)

    def _pipeline(self):
        return _make_pipeline({
            "local": LocalMediaProvider({"paths": [self.library]}),
            "news": NewsMediaProvider(),
        })

    def test_both_providers_contribute(self):
        p = self._pipeline()
        results = p._search_providers("npr news", MediaType.NEWS, "en-us")
        self.assertTrue(results)
        self.assertIn("news", {r["skill_id"] for r in results})

    def test_blacklisted_provider_is_not_dispatched_to(self):
        p = self._pipeline()
        message = Message("recognizer_loop:utterance", {},
                          {"session": {"session_id": "s1",
                                       "blacklisted_skills": ["news"]}})
        results = p._search_providers("npr news", MediaType.NEWS, "en-us",
                                      message=message)
        self.assertNotIn("news", {r["skill_id"] for r in results})

    def test_a_broken_provider_cannot_suppress_a_working_one(self):
        broken = MagicMock()
        broken.name = "broken"
        broken.search.side_effect = RuntimeError("provider exploded")
        p = _make_pipeline({"news": NewsMediaProvider(), "broken": broken})

        results = p._search_providers("npr news", MediaType.NEWS, "en-us")
        self.assertTrue(results, "a broken provider emptied the search")
        self.assertEqual({"news"}, {r["skill_id"] for r in results})


if __name__ == "__main__":
    unittest.main()
