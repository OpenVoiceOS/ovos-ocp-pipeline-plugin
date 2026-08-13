"""Tests for the in-process MediaProvider dispatch + additive dual-window merge.

Follows the ``__new__``-bypass pattern used across ``test/test_pipeline.py``:
``OCPPipelineMatcher.__init__`` is heavy (bus/intents/padatious), so these
tests construct a bare instance and inject only the attributes the methods
under test actually read.

Covers the guarantees the design ruling requires:

1. Zero MediaProviders installed -> ``_search`` output is identical to the
   legacy-only bus path (the back-compat guarantee).
2. Provider results are ADDED to the pool; a legacy bus entry is never
   replaced or deleted. The only de-dup is provider -> pool, by canonical uri.
3. A provider raising during ``search`` does not break or empty out the
   legacy bus results.
4. A slow provider cannot stall the search past ``max_timeout``.
5. The provider query context comes from the GLOBAL config, not the pipeline
   sub-config.
6. Provider results carry the QUERY's legacy media type, so ``filter_results``
   does not drop them when the mediavocab fold is not injective (NEWS).
7. Session ``blacklisted_skills`` suppresses a provider of that name.
8. Real-shape SomaFM coexistence: the bus answer and the provider answer both
   survive all the way to ``select_best``.
"""
import time
import unittest
from unittest.mock import MagicMock, patch

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType, PlaybackType

from ocp_pipeline.opm import OCPPipelineMatcher

try:
    from mediavocab import MediaType as MVMediaType, Release, Work
    HAS_MEDIAVOCAB = True
except ImportError:
    HAS_MEDIAVOCAB = False


def _make_pipeline(media_providers=None):
    """Create OCPPipelineMatcher bypassing its __init__."""
    p = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
    p.bus = FakeBus()
    p.ocp_sessions = {}
    p.skill_aliases = {}
    p.media2skill = {m: [] for m in MediaType}
    p.config = {}
    p.media_providers = media_providers or {}
    return p


def _wire_search(p, bus_track, config=None):
    """Wire the collaborators `_search` needs, with a single legacy result."""
    p.search_lock = MagicMock()
    p.search_lock.__enter__ = MagicMock(return_value=None)
    p.search_lock.__exit__ = MagicMock(return_value=None)
    p._enclosure = MagicMock()
    p.config = config if config is not None else {"filter_media": False,
                                                  "filter_SEI": False,
                                                  "min_score": 0}
    p._execute_query = MagicMock(return_value=[{"results": [dict(bus_track)]}])
    p.normalize_results = MagicMock(side_effect=lambda r: r)
    p.filter_results = MagicMock(side_effect=lambda r, *a, **kw: r)
    p.get_player = MagicMock()
    return p


def _message(session_id="s1", blacklist=None):
    sess = {"session_id": session_id}
    if blacklist is not None:
        sess["blacklisted_skills"] = blacklist
    return Message("recognizer_loop:utterance", {}, {"session": sess})


def _bus_track(uri, title, artist="", confidence=50):
    return {"uri": uri, "title": title, "artist": artist,
            "media_type": int(MediaType.MUSIC), "playback": 2,
            "match_confidence": confidence, "skill_id": "bus.skill"}


def _provider_result(uri, title, artist="", confidence=90, skill_id="mock.provider"):
    return {"uri": uri, "title": title, "artist": artist, "image": "",
            "length": 0, "media_type": MediaType.MUSIC, "playback": 2,
            "match_confidence": confidence, "skill_id": skill_id}


def _release(title, media_type, uri, conf=None):
    work = Work(title=title, media_type=media_type)
    kwargs = {}
    if conf is not None:
        kwargs["match_confidence"] = conf
    return Release(work=work, uri=uri, image="", **kwargs)


def _provider(name, releases=(), delay=0.0, raises=None):
    prov = MagicMock()
    prov.name = name

    def _search(*args, **kwargs):
        if delay:
            time.sleep(delay)
        if raises:
            raise raises
        return list(releases)

    prov.search.side_effect = _search
    return prov


# ---------------------------------------------------------------------------
# 1. Back-compat guarantee: zero providers -> bit-identical to legacy-only
# ---------------------------------------------------------------------------

class TestZeroProvidersBackCompat(unittest.TestCase):
    def test_search_providers_is_noop_with_no_providers(self):
        p = _make_pipeline(media_providers={})
        self.assertEqual(
            p._search_providers("play metallica", MediaType.MUSIC, "en-us"), [])

    def test_search_matches_legacy_only_output_when_no_providers(self):
        """`_search` with an empty `media_providers` dict must produce exactly
        the same `results` list (same objects, same order) that the
        pre-existing legacy-only implementation would have -- i.e. the merge
        step must never even run."""
        p = _wire_search(_make_pipeline(media_providers={}),
                         _bus_track("http://x.com/a.mp3", "Song A"))
        bus_track = _bus_track("http://x.com/a.mp3", "Song A")

        results = p._search("play song a", MediaType.MUSIC, "en-us",
                            message=_message())

        # exactly the legacy bus result, unmodified -- no provider dispatch,
        # no merge, bit-identical to the pre-change behaviour.
        self.assertEqual(results, [bus_track])

    def test_load_media_providers_respects_enabled_flag(self):
        p = _make_pipeline()
        p.config = {"media_providers": {"enabled": False}}
        with patch("ocp_pipeline.opm.load_media_providers") as loader:
            p._load_media_providers()
        loader.assert_not_called()
        self.assertEqual(p.media_providers, {})

    def test_load_media_providers_enabled_by_default(self):
        p = _make_pipeline()
        p.config = {}
        with patch("ocp_pipeline.opm.load_media_providers",
                   return_value={"x": MagicMock()}) as loader:
            p._load_media_providers()
        loader.assert_called_once()
        self.assertEqual(list(p.media_providers), ["x"])


# ---------------------------------------------------------------------------
# 2. Additive merge: provider results are ADDED, never replace a legacy entry
# ---------------------------------------------------------------------------

class TestAdditiveMerge(unittest.TestCase):
    def test_provider_result_appended_when_no_collision(self):
        p = _make_pipeline()
        bus_results = [_bus_track("http://bus/1", "Bus Track")]
        provider_results = [_provider_result("http://prov/1", "Provider Track")]
        merged = p._merge_provider_results(bus_results, provider_results)
        self.assertEqual(len(merged), 2)
        self.assertEqual({r["uri"] for r in merged},
                         {"http://bus/1", "http://prov/1"})

    def test_legacy_entry_is_never_replaced_on_uri_collision(self):
        """The provider entry is DROPPED; the legacy entry stays untouched."""
        p = _make_pipeline()
        bus = _bus_track("http://same/uri", "Old Title", confidence=40)
        provider_results = [_provider_result("http://same/uri", "New Title",
                                             confidence=95)]
        merged = p._merge_provider_results([bus], provider_results)
        self.assertEqual(merged, [bus])
        self.assertEqual(merged[0]["skill_id"], "bus.skill")

    def test_uri_dedup_is_canonical(self):
        p = _make_pipeline()
        bus = _bus_track("http://Example.COM/Track/", "Track")
        for dupe in ("https://example.com/Track",
                     "http://example.com/Track/",
                     "HTTPS://EXAMPLE.com/Track/"):
            merged = p._merge_provider_results(
                [bus], [_provider_result(dupe, "Track")])
            self.assertEqual(merged, [bus], dupe)

    def test_same_title_artist_different_uri_both_survive(self):
        """No (title, artist) de-dup tier: distinct Releases of one Work
        legitimately coexist."""
        p = _make_pipeline()
        bus = _bus_track("http://bus/uri", "Same Song", artist="Metallica")
        prov = _provider_result("http://provider/uri", "Same Song",
                                artist="Metallica")
        merged = p._merge_provider_results([bus], [prov])
        self.assertEqual(merged, [bus, prov])

    def test_playlist_entries_are_never_deduplicated(self):
        p = _make_pipeline()
        bus = {"title": "Mix", "uri": "http://same/uri",
               "playlist": [{"uri": "http://a", "title": "a"}],
               "match_confidence": 50, "skill_id": "bus.skill"}
        prov = {"title": "Mix", "uri": "http://same/uri",
                "playlist": [{"uri": "http://b", "title": "b"}],
                "match_confidence": 90, "skill_id": "mock.provider"}
        merged = p._merge_provider_results([bus], [prov])
        self.assertEqual(merged, [bus, prov])

    def test_empty_provider_results_returns_bus_results_unchanged(self):
        p = _make_pipeline()
        bus_results = [_bus_track("http://bus/uri", "Song")]
        merged = p._merge_provider_results(bus_results, [])
        self.assertEqual(merged, bus_results)

    def test_search_providers_dispatches_to_registered_provider(self):
        prov = _provider("mock.provider")
        p = _make_pipeline(media_providers={"mock.provider": prov})
        p._search_providers("play something", MediaType.GENERIC, "en-us")
        self.assertTrue(prov.search.called)


# ---------------------------------------------------------------------------
# 3. A raising provider must not break or empty out legacy bus results
# ---------------------------------------------------------------------------

class TestProviderExceptionIsolation(unittest.TestCase):
    def test_safe_search_absorbs_exception(self):
        provider = _provider("boom.provider", raises=RuntimeError("boom"))
        result = OCPPipelineMatcher._safe_search(provider, "signals", "en-us")
        self.assertEqual(result, [])

    def test_raising_provider_does_not_empty_dispatch_results(self):
        good = _provider("good.provider")
        boom = _provider("boom.provider", raises=RuntimeError("boom"))
        p = _make_pipeline(media_providers={"boom.provider": boom,
                                            "good.provider": good})
        results = p._search_providers("play something", MediaType.GENERIC,
                                      "en-us")
        self.assertEqual(results, [])
        self.assertTrue(boom.search.called)
        self.assertTrue(good.search.called)

    def test_raising_provider_does_not_break_full_search_bus_results_survive(self):
        boom = _provider("boom.provider", raises=RuntimeError("boom"))
        p = _wire_search(_make_pipeline(media_providers={"boom.provider": boom}),
                         _bus_track("http://x.com/a.mp3", "Legacy Song"))
        bus_track = _bus_track("http://x.com/a.mp3", "Legacy Song")
        results = p._search("play legacy song", MediaType.MUSIC, "en-us",
                            message=_message())
        self.assertEqual(results, [bus_track])


# ---------------------------------------------------------------------------
# 4. A slow provider cannot stall the search past max_timeout
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_MEDIAVOCAB, "mediavocab not installed")
class TestDispatchTimeout(unittest.TestCase):
    def test_slow_provider_is_dropped_at_max_timeout(self):
        fast = _provider("fast.provider",
                         releases=[_release("Fast Song", MVMediaType.MUSIC,
                                            "http://fast/1", conf=0.9)])
        slow = _provider("slow.provider", delay=3.0,
                         releases=[_release("Slow Song", MVMediaType.MUSIC,
                                            "http://slow/1", conf=0.9)])
        p = _make_pipeline(media_providers={"fast.provider": fast,
                                            "slow.provider": slow})
        p.config = {"max_timeout": 1}

        # warm up the config/signals machinery so the measurement below times
        # the dispatch, not a first-call import
        warm = _make_pipeline(media_providers={"fast.provider": fast})
        warm.config = {"max_timeout": 1}
        warm._search_providers("fast song", MediaType.MUSIC, "en-us")

        start = time.monotonic()
        results = p._search_providers("fast song", MediaType.MUSIC, "en-us")
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 1.5,
                        f"search stalled on the slow provider ({elapsed:.2f}s)")
        self.assertEqual([r["title"] for r in results], ["Fast Song"])


# ---------------------------------------------------------------------------
# 5. Query context comes from the GLOBAL config
# ---------------------------------------------------------------------------

class TestQueryContext(unittest.TestCase):
    GLOBAL = {
        "media": {"supported_playback_types": ["audio", "video"]},
        "media_content_filter": {"blocked_genres": ["adult", "horror"]},
        "location": {"city": {"region": {"country": {"code": "PT"}}}},
    }

    def test_context_is_read_from_global_config(self):
        p = _make_pipeline()
        # the pipeline sub-config carries none of these keys
        p.config = {"min_score": 50}
        with patch("ocp_pipeline.opm.Configuration", return_value=dict(self.GLOBAL)):
            ctx = p._build_query_context("en-us", message=_message("sess-42"))
        self.assertEqual(ctx["supported_playback_types"], {"audio", "video"})
        self.assertEqual(ctx["blocked_genres"], {"adult", "horror"})
        self.assertEqual(ctx["region"], "PT")
        self.assertEqual(ctx["session_id"], "sess-42")

    def test_context_defaults_are_permissive_but_block_adult(self):
        p = _make_pipeline()
        p.config = {}
        with patch("ocp_pipeline.opm.Configuration", return_value={}):
            ctx = p._build_query_context("en-us")
        self.assertEqual(ctx["supported_playback_types"], set())
        self.assertEqual(ctx["blocked_genres"], {"adult"})
        self.assertIsNone(ctx["region"])
        self.assertIsNone(ctx["session_id"])

    def test_allow_adult_content_unblocks_adult(self):
        p = _make_pipeline()
        p.config = {}
        cfg = {"media_content_filter": {"allow_adult_content": True}}
        with patch("ocp_pipeline.opm.Configuration", return_value=cfg):
            ctx = p._build_query_context("en-us")
        self.assertEqual(ctx["blocked_genres"], set())


# ---------------------------------------------------------------------------
# 6. MediaType round-trip: provider results carry the QUERY's media type
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_MEDIAVOCAB, "mediavocab not installed")
class TestMediaTypeRoundTrip(unittest.TestCase):
    def test_news_query_results_survive_filter_results(self):
        """legacy NEWS folds onto mediavocab RADIO, which folds back to legacy
        RADIO -- stamping the Release's own type would make filter_results
        drop every provider answer to a NEWS query."""
        prov = _provider("news.provider",
                         releases=[_release("BBC World Service",
                                            MVMediaType.RADIO,
                                            "http://news/stream.mp3",
                                            conf=0.9)])
        p = _make_pipeline(media_providers={"news.provider": prov})
        p.config = {"min_score": 50, "filter_media": True, "filter_SEI": True}

        results = p._search_providers("the news", MediaType.NEWS, "en-us")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["media_type"], MediaType.NEWS)

        player = MagicMock()
        player.available_extractors = []
        player.ocp_available = True
        p.get_player = MagicMock(return_value=player)
        kept = p.filter_results(p.normalize_results(results), "the news",
                                "en-us", MediaType.NEWS, message=_message())
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].media_type, MediaType.NEWS)


# ---------------------------------------------------------------------------
# 7. Session blacklist suppresses a provider of the same name
# ---------------------------------------------------------------------------

class TestSessionBlacklist(unittest.TestCase):
    def test_blacklisted_provider_is_not_dispatched_to(self):
        blocked = _provider("somafm")
        other = _provider("other.provider")
        p = _make_pipeline(media_providers={"somafm": blocked,
                                            "other.provider": other})
        p.config = {}
        p._search_providers("play groove salad", MediaType.RADIO, "en-us",
                            message=_message(blacklist=["somafm"]))
        self.assertFalse(blocked.search.called)
        self.assertTrue(other.search.called)

    def test_no_blacklist_dispatches_to_all(self):
        prov = _provider("somafm")
        p = _make_pipeline(media_providers={"somafm": prov})
        p.config = {}
        p._search_providers("play groove salad", MediaType.RADIO, "en-us",
                            message=_message())
        self.assertTrue(prov.search.called)


# ---------------------------------------------------------------------------
# 8. Real-shape coexistence, merged -> normalize -> filter -> select_best
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_MEDIAVOCAB, "mediavocab not installed")
class TestSomaFMCoexistence(unittest.TestCase):
    """Recorded shapes from the real ecosystem: the skill answers with a
    direct stream uri, artist "SomaFM", confidence 95; the provider answers
    with a .pls playlist uri, no credited artist, and no confidence at all.
    Nothing may collapse and nothing may be dropped."""

    SKILL_RESULT = {"uri": "http://ice2.somafm.com/groovesalad-128-mp3",
                    "title": "Groove Salad",
                    "artist": "SomaFM",
                    "media_type": MediaType.RADIO,
                    "playback": PlaybackType.AUDIO,
                    "match_confidence": 95,
                    "skill_id": "skill-somafm.openvoiceos"}

    def test_both_windows_survive_to_select_best(self):
        release = _release("Groove Salad", MVMediaType.RADIO,
                           "https://somafm.com/groovesalad.pls")
        self.assertFalse(release.match_confidence,
                         "fixture must be an UNSCORED provider release")
        prov = _provider("somafm", releases=[release])
        p = _make_pipeline(media_providers={"somafm": prov})
        p.config = {"min_score": 50, "filter_media": True, "filter_SEI": True}

        provider_results = p._search_providers("groove salad", MediaType.RADIO,
                                               "en-us")
        self.assertEqual(len(provider_results), 1)
        # the unscored release was scored by the bridge fallback, so min_score
        # cannot kill it
        self.assertGreaterEqual(provider_results[0]["match_confidence"], 50)

        merged = p._merge_provider_results([dict(self.SKILL_RESULT)],
                                           provider_results)
        self.assertEqual(len(merged), 2, "different uris must NOT collapse")

        player = MagicMock()
        player.available_extractors = []
        player.ocp_available = True
        p.get_player = MagicMock(return_value=player)

        kept = p.filter_results(p.normalize_results(merged), "groove salad",
                                "en-us", MediaType.RADIO, message=_message())
        self.assertEqual(len(kept), 2, "both windows must reach select_best")
        self.assertEqual({r.skill_id for r in kept},
                         {"skill-somafm.openvoiceos", "somafm"})

        # arbitration is the EXISTING pipeline's, unchanged: here the exact
        # title match scores the provider 100 against the skill's own 95.
        best = p.select_best(kept, _message())
        self.assertEqual(best.skill_id, "somafm")
        self.assertEqual(best.match_confidence, 100)

    def test_blacklisting_the_provider_leaves_the_skill_result(self):
        release = _release("Groove Salad", MVMediaType.RADIO,
                           "https://somafm.com/groovesalad.pls")
        prov = _provider("somafm", releases=[release])
        p = _make_pipeline(media_providers={"somafm": prov})
        p.config = {"min_score": 50}
        provider_results = p._search_providers(
            "groove salad", MediaType.RADIO, "en-us",
            message=_message(blacklist=["somafm"]))
        self.assertEqual(provider_results, [])


if __name__ == "__main__":
    unittest.main()
