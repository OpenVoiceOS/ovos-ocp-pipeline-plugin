"""Unit tests for in-process MediaProvider dispatch and the Release->OCP seam.

These use mock ``MediaProvider`` subclasses returning synthetic
``mediavocab.Release`` objects. They cover:

* the three-axis routing gate (a provider that can't serve a query is skipped),
* concurrent dispatch returning ranked results,
* the ``Release`` -> OCP result field mapping (mediavocab-native ``media_type``,
  ``PlaybackType`` backend selector derived from mediavocab routing),
* that bus-path behaviour is unchanged when no providers are installed.

The pipeline is mediavocab-native: routing and result ``media_type`` use
``mediavocab.MediaType`` directly. The only translation left is the playback
*backend selector* (``ovos_utils.ocp.PlaybackType``), which is ``MediaEntry``
structure, not a media-type taxonomy.
"""
import unittest
from typing import List, Set

from mediavocab import MediaType
from mediavocab import EntityKind, Release, Signals, Work
from mediavocab.models.entity import Credit, EntityRef
from mediavocab.taxonomy import PlaybackType as MVPlaybackType

from ovos_utils.ocp import PlaybackType as OCPPlaybackType

from ocp_pipeline.bridge import (release_to_ocp_result, media_type_to_signals,
                                 mediavocab_playback_to_ocp)
from ocp_pipeline.opm import OCPPipelineMatcher

try:
    from ovos_plugin_manager.templates.media_provider import MediaProvider
    HAS_MP = True
except ImportError:
    MediaProvider = object
    HAS_MP = False


def _make_release(title: str, media_type: MediaType, uri: str,
                  conf: float, artist: str = "", runtime: float = 0) -> Release:
    credits = []
    if artist:
        credits = [Credit(entity=EntityRef(name=artist, kind=EntityKind.GROUP),
                          role="artist")]
    work = Work(title=title, media_type=media_type, runtime=runtime,
                credits=credits)
    return Release(work=work, uri=uri, image=f"{title}.png",
                   match_confidence=conf)


if HAS_MP:
    class MusicProvider(MediaProvider):
        name = "mock.music"
        media: Set[MediaType] = {MediaType.MUSIC}

        def is_available(self) -> bool:
            return True

        def search(self, signals: Signals, lang: str = "en-us") -> List[Release]:
            return [
                _make_release("Black Album", MediaType.MUSIC,
                              "https://music/black", 0.9, artist="Metallica",
                              runtime=3600),
                _make_release("Ride the Lightning", MediaType.MUSIC,
                              "https://music/ride", 0.6, artist="Metallica"),
            ]

    class MovieProvider(MediaProvider):
        name = "mock.movie"
        media: Set[MediaType] = {MediaType.MOVIE}

        def is_available(self) -> bool:
            return True

        def search(self, signals: Signals, lang: str = "en-us") -> List[Release]:
            return [
                _make_release("Some Movie", MediaType.MOVIE,
                              "https://movie/x", 0.75, runtime=7200),
            ]

    class ExplodingProvider(MediaProvider):
        name = "mock.boom"
        media: Set[MediaType] = {MediaType.MUSIC}

        def is_available(self) -> bool:
            return True

        def search(self, signals: Signals, lang: str = "en-us") -> List[Release]:
            raise RuntimeError("boom")


@unittest.skipUnless(HAS_MP, "ovos-plugin-manager MediaProvider type unavailable")
class TestBridge(unittest.TestCase):
    def test_release_to_ocp_result_fields(self):
        rel = _make_release("Black Album", MediaType.MUSIC,
                            "https://music/black", 0.85, artist="Metallica",
                            runtime=3600)
        d = release_to_ocp_result(rel, "mock.music")
        self.assertEqual(d["uri"], "https://music/black")
        self.assertEqual(d["title"], "Black Album")
        self.assertEqual(d["image"], "Black Album.png")
        self.assertEqual(d["artist"], "Metallica")
        self.assertEqual(d["length"], 3600)
        # media_type is the mediavocab value directly
        self.assertEqual(d["media_type"], MediaType.MUSIC)
        self.assertEqual(d["playback"], OCPPlaybackType.AUDIO)
        # 0.85 float -> 85 int
        self.assertEqual(d["match_confidence"], 85)
        self.assertEqual(d["skill_id"], "mock.music")

    def test_confidence_clamped_and_scaled(self):
        rel = _make_release("x", MediaType.MUSIC, "u", 1.0)
        self.assertEqual(release_to_ocp_result(rel, "p")["match_confidence"], 100)
        rel0 = _make_release("x", MediaType.MUSIC, "u", 0.0)
        self.assertEqual(release_to_ocp_result(rel0, "p")["match_confidence"], 0)

    def test_no_credit_artist_empty(self):
        rel = _make_release("x", MediaType.MOVIE, "u", 0.5)
        self.assertEqual(release_to_ocp_result(rel, "p")["artist"], "")

    def test_playback_type_mapping(self):
        self.assertEqual(mediavocab_playback_to_ocp(MVPlaybackType.AUDIO),
                         OCPPlaybackType.AUDIO)
        self.assertEqual(mediavocab_playback_to_ocp(MVPlaybackType.VIDEO),
                         OCPPlaybackType.VIDEO)
        self.assertEqual(mediavocab_playback_to_ocp(MVPlaybackType.INTERACTIVE),
                         OCPPlaybackType.SKILL)
        self.assertEqual(mediavocab_playback_to_ocp(MVPlaybackType.PAGED),
                         OCPPlaybackType.WEBVIEW)

    def test_interactive_release_maps_to_skill(self):
        rel = _make_release("A Game", MediaType.GAME, "u", 0.5)
        d = release_to_ocp_result(rel, "p")
        self.assertEqual(d["playback"], OCPPlaybackType.SKILL)
        self.assertEqual(d["media_type"], MediaType.GAME)

    def test_paged_release_maps_to_webview(self):
        rel = _make_release("A Comic", MediaType.COMIC, "u", 0.5)
        d = release_to_ocp_result(rel, "p")
        self.assertEqual(d["playback"], OCPPlaybackType.WEBVIEW)

    def test_media_type_to_signals(self):
        sig = media_type_to_signals(MediaType.MUSIC, "metallica")
        self.assertEqual(sig.medium, MediaType.MUSIC)
        self.assertEqual(sig.title, "metallica")
        self.assertEqual(sig.playback_type, MVPlaybackType.AUDIO)

    def test_generic_signals_typeless(self):
        sig = media_type_to_signals(MediaType.GENERIC, "anything")
        self.assertIsNone(sig.medium)


@unittest.skipUnless(HAS_MP, "ovos-plugin-manager MediaProvider type unavailable")
class TestProviderDispatch(unittest.TestCase):
    def setUp(self):
        self.ocp = OCPPipelineMatcher(config={})

    def _install(self, *providers):
        self.ocp.media_providers = {p.name: p for p in providers}

    def test_routing_gate_skips_non_matching(self):
        # only MovieProvider installed; a MUSIC query must not reach it
        self._install(MovieProvider())
        results = self.ocp._search_providers("metallica", MediaType.MUSIC,
                                             "en-us")
        self.assertEqual(results, [])

    def test_routing_gate_matches(self):
        self._install(MovieProvider())
        results = self.ocp._search_providers("some movie", MediaType.MOVIE,
                                             "en-us")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Some Movie")

    def test_dispatch_returns_bridged_results(self):
        self._install(MusicProvider())
        results = self.ocp._search_providers("metallica", MediaType.MUSIC,
                                             "en-us")
        self.assertEqual(len(results), 2)
        titles = {r["title"] for r in results}
        self.assertEqual(titles, {"Black Album", "Ride the Lightning"})
        for r in results:
            self.assertEqual(r["skill_id"], "mock.music")
            self.assertEqual(r["media_type"], MediaType.MUSIC)

    def test_multiple_providers_concurrent(self):
        # both gate on different media; a generic query (no medium) reaches both
        self._install(MusicProvider(), MovieProvider())
        results = self.ocp._search_providers("anything", MediaType.GENERIC,
                                             "en-us")
        uris = {r["uri"] for r in results}
        self.assertIn("https://music/black", uris)
        self.assertIn("https://movie/x", uris)

    def test_exploding_provider_is_isolated(self):
        # search_safe swallows the error; other providers still return
        self._install(MusicProvider(), ExplodingProvider())
        results = self.ocp._search_providers("metallica", MediaType.MUSIC,
                                             "en-us")
        # only the 2 music results, boom contributed nothing and didn't abort
        self.assertEqual(len(results), 2)

    def test_no_providers_returns_empty(self):
        self.ocp.media_providers = {}
        self.assertEqual(
            self.ocp._search_providers("x", MediaType.MUSIC, "en-us"), [])

    def test_dispatch_results_rank_via_select_best(self):
        self._install(MusicProvider())
        results = self.ocp._search_providers("metallica", MediaType.MUSIC,
                                             "en-us")
        normalized = self.ocp.normalize_results(results)
        best = self.ocp.select_best(normalized, message=None)
        # 0.9 (Black Album) outranks 0.6 (Ride the Lightning)
        self.assertEqual(best.title, "Black Album")
        self.assertEqual(best.match_confidence, 90)


class TestProviderConfigGate(unittest.TestCase):
    def test_no_providers_installed_behaves_as_today(self):
        # default init with nothing installed/monkeypatched -> empty dict,
        # dispatch is a no-op and the bus path is untouched.
        ocp = OCPPipelineMatcher(config={})
        if not HAS_MP:
            self.assertEqual(ocp.media_providers, {})
        # _search_providers must be safe to call regardless
        self.assertEqual(
            ocp._search_providers("x", MediaType.MUSIC, "en-us"), [])


if __name__ == "__main__":
    unittest.main()
