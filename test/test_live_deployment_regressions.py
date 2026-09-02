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


if __name__ == "__main__":
    unittest.main()
