"""Regressions from a provider-only live deployment.

Three defects only a real deployment with in-process ``MediaProvider``
plugins and no legacy OCP search skills exposes:

1. Classification was derived from registered skills alone. ovos-media's own
   favorites skill registers ``GENERIC``, so the single-label shortcut
   answered ``(GENERIC, 1.0)`` for every utterance and the classifier never
   ran.
2. A bare typed request ("play some music") was forwarded as a title, so
   providers fuzzy matched "some music" against real titles and scored below
   the confidence floor, instead of browsing their catalog.
3. A tagged artist reached the pipeline as an empty string, and a search with
   no possible bus responder still waited out the legacy window.
4. The two search windows ran one after the other, so a deployment with legacy
   skills paid the bus window and the provider window in series (12-29s from
   utterance to playback).
5. A provider that declares no media types answered a MUSIC request with
   confidence 100 and outranked the local music library.
"""
import time
import unittest
from unittest.mock import MagicMock

from ovos_bus_client.message import Message

from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType

from mediavocab import MediaType as MVMediaType, Release, Work

from ocp_pipeline.bridge import release_to_ocp_result
from ocp_pipeline.opm import OCPPipelineMatcher


class _AnsweringProvider:
    """MediaProvider stand-in that answers with canned releases."""

    def __init__(self, name, releases, served=None, delay=0.0):
        self.name = name
        if served is not None:
            self.SERVED_MEDIA = served
        self.releases = list(releases)
        self.delay = delay
        self.queries = []

    def search(self, signals, lang="en-us", **kwargs):
        self.queries.append(signals)
        if self.delay:
            time.sleep(self.delay)
        return list(self.releases)


def _release(title, media_type, uri, confidence):
    return Release(work=Work(title=title, media_type=media_type), uri=uri,
                   image="", match_confidence=confidence)


def _wire_search(p, legacy_results=(), legacy_delay=0.0, config=None):
    """Wire the collaborators `_search` needs around a fake legacy window."""
    p.search_lock = MagicMock()
    p._enclosure = MagicMock()
    p.config = config if config is not None else {"min_score": 50,
                                                  "filter_media": True,
                                                  "filter_SEI": False,
                                                  "max_timeout": 5}

    def _legacy(*args, **kwargs):
        if legacy_delay:
            time.sleep(legacy_delay)
        return [{"results": [dict(r) for r in legacy_results]}]

    p._execute_query = MagicMock(side_effect=_legacy)
    player = MagicMock()
    player.available_extractors = []
    player.ocp_available = True
    p.get_player = MagicMock(return_value=player)
    return p


class _Provider:
    """Minimal MediaProvider stand-in declaring what it serves."""

    def __init__(self, name, served=None):
        self.name = name
        if served is not None:
            self.SERVED_MEDIA = served

    def search(self, signals, lang="en-us", **kwargs):
        return []


def _make_pipeline(media_providers=None):
    p = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
    p.bus = FakeBus()
    p.ocp_sessions = {}
    p.skill_aliases = {}
    p.media2skill = {m: [] for m in MediaType}
    p.config = {}
    p.media_providers = media_providers or {}
    p.provider_media_types = []
    from ahocorasick_ner import AhocorasickNER
    p.ner = AhocorasickNER()
    return p


class TestProvidersDriveClassification(unittest.TestCase):

    def test_generic_registering_skill_does_not_bypass_the_classifier(self):
        """A GENERIC-only skill registration must not shortcut classification."""
        p = _make_pipeline()
        p.media2skill[MediaType.GENERIC] = ["ovos-skill-favorites"]
        media, conf = p.classify_media("play some music", "en-us")
        self.assertEqual(media, MediaType.MUSIC)
        self.assertGreater(conf, 0.0)

    def test_provider_media_types_join_the_valid_labels(self):
        p = _make_pipeline()
        p.media2skill[MediaType.GENERIC] = ["ovos-skill-favorites"]
        p.provider_media_types = [MediaType.MUSIC, MediaType.NEWS,
                                  MediaType.RADIO]
        self.assertIn(MediaType.MUSIC, p._default_valid_labels())
        self.assertIn(MediaType.GENERIC, p._default_valid_labels())
        media, _ = p.classify_media("play some music", "en-us")
        self.assertEqual(media, MediaType.MUSIC)

    def test_declared_provider_types_are_collected(self):
        p = _make_pipeline({
            "local": _Provider("local", {MVMediaType.MUSIC, MVMediaType.MOVIE}),
        })
        types = p._collect_provider_media_types()
        self.assertIn(MediaType.MUSIC, types)
        self.assertIn(MediaType.MOVIE, types)
        self.assertNotIn(MediaType.PODCAST, types)

    def test_provider_declaring_nothing_serves_everything(self):
        p = _make_pipeline({"news": _Provider("news")})
        self.assertEqual(p._collect_provider_media_types(), list(MediaType))

    def test_force_generic_downgrade_considers_providers(self):
        """No skill for a type is not GENERIC when a provider serves it.

        With the type still live, an empty first pass falls back to a second
        GENERIC search; a downgraded type skips that fallback entirely.
        """
        def _run(provider_types):
            p = _make_pipeline({"local": _Provider("local")})
            p.provider_media_types = provider_types
            p.skill_aliases = {"ovos-skill-favorites": ["favorites"]}
            p.config = {}
            p.search_lock = MagicMock()
            p.search_lock.__enter__ = MagicMock(return_value=None)
            p.search_lock.__exit__ = MagicMock(return_value=None)
            sends = []

            class _Query:
                def __init__(self, query=None, media_type=None, config=None,
                             bus=None):
                    self.media_type = media_type
                    self.results = []

                def send(self, *args, **kwargs):
                    sends.append(self.media_type)

                def wait(self):
                    pass

                def reset(self):
                    pass

            import ocp_pipeline.opm as opm
            real = opm.OCPQuery
            opm.OCPQuery = _Query
            try:
                p._execute_query("some music", media_type=MediaType.MUSIC,
                                 message=Message("test"))
            finally:
                opm.OCPQuery = real
            return sends

        self.assertEqual(_run([MediaType.MUSIC]),
                         [MediaType.MUSIC, MediaType.GENERIC])
        self.assertEqual(_run([]), [MediaType.MUSIC])


class TestBareRequestsBrowse(unittest.TestCase):

    def test_bare_typed_request_sends_empty_title(self):
        p = _make_pipeline()
        signals = p._provider_signals("some music", MediaType.MUSIC, "en-us")
        self.assertFalse(signals.title)

    def test_real_title_is_forwarded_untouched(self):
        p = _make_pipeline()
        signals = p._provider_signals("rain song", MediaType.MUSIC, "en-us")
        self.assertEqual(signals.title, "rain song")

    def test_empty_query_is_bare(self):
        p = _make_pipeline()
        self.assertTrue(p._is_bare_media_request("", "en-us"))


class TestArtistSurvivesTheBridge(unittest.TestCase):

    def test_tagged_artist_reaches_the_media_entry(self):
        """Providers reading file/stream tags carry the artist on Work.extra."""
        work = Work(title="Zen Ambience", media_type=MVMediaType.MUSIC)
        work.extra = {"artist": "Zen Garden"}
        release = Release(work=work, uri="file:///music/zen.mp3")
        entry = release_to_ocp_result(release, "local",
                                      media_type=MediaType.MUSIC)
        self.assertEqual(entry["artist"], "Zen Garden")


class TestZeroSkillSearchIsFast(unittest.TestCase):

    def test_legacy_bus_window_is_skipped_without_skills(self):
        """No registered OCP skill means no possible bus responder."""
        p = _make_pipeline({"local": _Provider("local")})
        p.skill_aliases = {}
        p.config = {"max_timeout": 10}
        p.search_lock = MagicMock()

        class _Query:
            def __init__(self, *args, **kwargs):
                raise AssertionError("legacy bus window must not run")

        import ocp_pipeline.opm as opm
        real = opm.OCPQuery
        opm.OCPQuery = _Query
        start = time.monotonic()
        try:
            results = p._execute_query("some music",
                                       media_type=MediaType.MUSIC,
                                       message=Message("test"))
        finally:
            opm.OCPQuery = real
        self.assertEqual(results, [])
        self.assertLess(time.monotonic() - start, 1)


class TestSearchWindowsRunConcurrently(unittest.TestCase):
    """The dual-window search merges two independent sources, so it must cost
    the slower window, not the sum of both."""

    LEGACY_DELAY = 2.0
    PROVIDER_DELAY = 1.0

    def _pipeline(self):
        provider = _AnsweringProvider(
            "local",
            [_release("Ghost Track", MVMediaType.MUSIC,
                      "file:///music/ghost.mp3", 0.9)],
            served={MVMediaType.MUSIC}, delay=self.PROVIDER_DELAY)
        p = _make_pipeline({"local": provider})
        p.skill_aliases = {"bus.skill": ["bus skill"]}
        return _wire_search(p, legacy_results=[
            {"uri": "file:///music/bus.mp3", "title": "Bus Track",
             "artist": "", "media_type": MediaType.MUSIC, "playback": 2,
             "match_confidence": 60, "skill_id": "bus.skill"}],
            legacy_delay=self.LEGACY_DELAY)

    def test_wall_clock_is_the_slower_window_not_the_sum(self):
        p = self._pipeline()
        start = time.monotonic()
        results = p._search("ghost track", MediaType.MUSIC, "en-us",
                            message=Message("test"))
        elapsed = time.monotonic() - start
        self.assertTrue(results)
        self.assertLess(elapsed, self.LEGACY_DELAY + self.PROVIDER_DELAY - 0.5,
                        "the windows ran in series")

    def test_merged_output_is_unchanged(self):
        """Same inputs, same merged pool as the sequential search produced."""
        p = self._pipeline()
        concurrent = p._search("ghost track", MediaType.MUSIC, "en-us",
                               message=Message("test"))

        seq = self._pipeline()
        sequential = seq._merge_provider_results(
            [r for batch in seq._execute_query("ghost track",
                                               media_type=MediaType.MUSIC,
                                               message=Message("test"))
             for r in batch["results"]],
            seq._search_providers("ghost track", MediaType.MUSIC, "en-us"))
        sequential = seq.filter_results(seq.normalize_results(sequential),
                                        "ghost track", "en-us",
                                        MediaType.MUSIC,
                                        message=Message("test"))

        self.assertEqual([(r.skill_id, r.uri, r.match_confidence)
                          for r in concurrent],
                         [(r.skill_id, r.uri, r.match_confidence)
                          for r in sequential])


class TestDeclaredProvidersRankFirst(unittest.TestCase):
    """A provider that declares nothing may answer anything, but its answer is
    not evidence that it fits a typed request."""

    def _pipeline(self, providers):
        p = _make_pipeline(providers)
        p.skill_aliases = {}
        return _wire_search(p)

    @staticmethod
    def _providers():
        return {
            "local": _AnsweringProvider(
                "local", [_release("Some Music", MVMediaType.MUSIC,
                                   "file:///music/browse.mp3", 0.5)],
                served={MVMediaType.MUSIC}),
            "news": _AnsweringProvider(
                "news", [_release("NPR News Now", MVMediaType.RADIO,
                                  "https://npr.example/news.mp3", 1.0)]),
        }

    def test_declared_provider_wins_a_typed_request(self):
        p = self._pipeline(self._providers())
        results = p._search("some music", MediaType.MUSIC, "en-us",
                            message=Message("test"))
        best = p.select_best(results, Message("test"))
        self.assertEqual(best.skill_id, "local")

    def test_undeclared_answer_is_dropped_when_a_declared_one_exists(self):
        p = self._pipeline(self._providers())
        results = p._search("some music", MediaType.MUSIC, "en-us",
                            message=Message("test"))
        self.assertEqual({"local"}, {r.skill_id for r in results})

    def test_undeclared_provider_still_plays_when_it_is_the_only_answer(self):
        providers = self._providers()
        providers.pop("local")
        p = self._pipeline(providers)
        results = p._search("some music", MediaType.MUSIC, "en-us",
                            message=Message("test"))
        best = p.select_best(results, Message("test"))
        self.assertEqual(best.skill_id, "news")
        self.assertEqual(best.match_confidence, 50)

    def test_declaring_provider_is_not_dispatched_off_its_types(self):
        providers = self._providers()
        p = self._pipeline(providers)
        p._search("the news", MediaType.NEWS, "en-us", message=Message("test"))
        self.assertEqual(providers["local"].queries, [],
                         "a MUSIC-only provider was asked about NEWS")
        self.assertTrue(providers["news"].queries)

    def test_generic_request_caps_nothing(self):
        """"play something" asks for no type, so no provider is off-topic and
        the existing arbitration decides."""
        p = self._pipeline(self._providers())
        results = p._search("something", MediaType.GENERIC, "en-us",
                            message=Message("test"))
        best = p.select_best(results, Message("test"))
        self.assertEqual(best.skill_id, "news")
        self.assertEqual(best.match_confidence, 100)


if __name__ == "__main__":
    unittest.main()
