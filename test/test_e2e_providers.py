"""End-to-end MediaProvider path through the OCP pipeline.

Drives the in-process ``opm.media.provider`` path from a fake
:class:`~ovos_plugin_manager.templates.media_provider.MediaProvider` subclass —
which implements *only* the single
``search(signals, lang="en-us", *, supported_playback_types, blocked_genres,
region, session_id)`` method — all the way through the pipeline's
``_search_providers`` -> ``normalize_results`` -> ``select_best`` flow, asserting
that:

* a provider's ``Release`` results are collected, bridged to OCP result dicts,
  normalized and ranked (highest ``match_confidence`` wins);
* the request-context kwargs are built (:meth:`_build_query_context`) and passed
  to ``search`` (the fake captures them);
* a provider raising inside ``search`` is absorbed by the ``_safe_search``
  wrapper and cannot abort the multi-provider search (the surviving provider's
  results still come through).

The full ``OCPPipelineMatcher`` instantiates in this environment, so these tests
construct it normally (``config={}``) and inject the fake providers into
``ocp.media_providers``. If a future GUI-rework env blocks full init, the
provider-dispatch methods can be exercised on an instance created via
``OCPPipelineMatcher.__new__(OCPPipelineMatcher)`` with ``config``/
``media_providers`` set by hand (the ``__new__``-bypass pattern already used for
the unbound-method tests in ``test_context_aware_providers.py``).
"""
import unittest
from typing import List

try:
    from ovos_plugin_manager.templates.media_provider import MediaProvider
    HAS_MP = True
except ImportError:
    MediaProvider = object
    HAS_MP = False

from mediavocab import MediaType
from mediavocab import EntityKind, Release, Signals, Work
from mediavocab.models.entity import Credit, EntityRef

from ocp_pipeline.opm import OCPPipelineMatcher


def _release(title, uri, conf, media_type=MediaType.MUSIC, artist="Metallica"):
    credits = [Credit(entity=EntityRef(name=artist, kind=EntityKind.GROUP),
                      role="artist")] if artist else []
    work = Work(title=title, media_type=media_type, credits=credits)
    return Release(work=work, uri=uri, image=f"{title}.png",
                   match_confidence=conf)


if HAS_MP:
    class RecordingMusicProvider(MediaProvider):
        """Fake provider implementing only ``search``; records the kwargs it
        was handed so the test can assert the request context was passed."""
        name = "e2e.music"

        def __init__(self, config=None):
            super().__init__(config)
            self.calls = []

        def search(self, signals: Signals, lang: str = "en-us", *,
                   supported_playback_types=None, blocked_genres=None,
                   region=None, session_id=None) -> List[Release]:
            self.calls.append({
                "signals": signals,
                "lang": lang,
                "supported_playback_types": supported_playback_types,
                "blocked_genres": blocked_genres,
                "region": region,
                "session_id": session_id,
            })
            return [
                _release("Black Album", "https://music/black", 0.9),
                _release("Ride the Lightning", "https://music/ride", 0.6),
            ]

    class BoomProvider(MediaProvider):
        name = "e2e.boom"

        def search(self, signals: Signals, lang: str = "en-us", *,
                   supported_playback_types=None, blocked_genres=None,
                   region=None, session_id=None) -> List[Release]:
            raise RuntimeError("provider blew up")


@unittest.skipUnless(HAS_MP, "ovos-plugin-manager MediaProvider type unavailable")
class TestProviderE2E(unittest.TestCase):
    def setUp(self):
        # full pipeline init (bus/GUI stack) works in this env; if a future
        # env blocks it, swap to the __new__-bypass documented in the module
        # docstring.
        self.ocp = OCPPipelineMatcher(config={})

    def test_provider_results_collected_normalized_ranked(self):
        prov = RecordingMusicProvider()
        self.ocp.media_providers = {prov.name: prov}

        raw = self.ocp._search_providers("metallica", MediaType.MUSIC, "en-us")
        # both releases bridged to OCP result dicts, tagged with the skill_id
        self.assertEqual(len(raw), 2)
        self.assertTrue(all(r["skill_id"] == "e2e.music" for r in raw))

        normalized = self.ocp.normalize_results(raw)
        best = self.ocp.select_best(normalized, message=None)
        # 0.9 outranks 0.6 -> Black Album wins, confidence scaled to int 90
        self.assertEqual(best.title, "Black Album")
        self.assertEqual(best.match_confidence, 90)

    def test_query_context_built_and_passed(self):
        prov = RecordingMusicProvider()
        self.ocp.media_providers = {prov.name: prov}
        # adult blocked by default in the built context
        self.ocp._search_providers("metallica", MediaType.MUSIC, "en-us")

        self.assertEqual(len(prov.calls), 1)
        call = prov.calls[0]
        self.assertEqual(call["lang"], "en-us")
        # the four explicit context kwargs reach the provider
        self.assertIn("adult", call["blocked_genres"])
        self.assertEqual(call["supported_playback_types"], set())
        self.assertIsNone(call["region"])
        # a real mediavocab Signals object was forwarded
        self.assertIsInstance(call["signals"], Signals)

    def test_supported_playback_types_passed_from_config(self):
        prov = RecordingMusicProvider()
        ocp = OCPPipelineMatcher(
            config={"media": {"supported_playback_types": ["audio"]}})
        ocp.media_providers = {prov.name: prov}
        ocp._search_providers("metallica", MediaType.MUSIC, "en-us")
        self.assertEqual(prov.calls[0]["supported_playback_types"], {"audio"})

    def test_raising_provider_is_absorbed(self):
        good = RecordingMusicProvider()
        boom = BoomProvider()
        # boom first so it would abort an unwrapped loop
        self.ocp.media_providers = {boom.name: boom, good.name: good}

        raw = self.ocp._search_providers("metallica", MediaType.MUSIC, "en-us")
        # boom contributed nothing but did not abort the search; good's 2 came
        self.assertEqual(len(raw), 2)
        self.assertTrue(all(r["skill_id"] == "e2e.music" for r in raw))

    def test_only_raising_provider_yields_empty(self):
        boom = BoomProvider()
        self.ocp.media_providers = {boom.name: boom}
        self.assertEqual(
            self.ocp._search_providers("metallica", MediaType.MUSIC, "en-us"), [])


if __name__ == "__main__":
    unittest.main()
