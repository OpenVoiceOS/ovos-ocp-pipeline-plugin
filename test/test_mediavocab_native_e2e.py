"""End-to-end proof that the OCP pipeline is fully mediavocab-native.

Two things are asserted here:

1. **Behaviour** -- the *real* ``ovos-media-classifier`` (not mocked) classifies
   real utterances and ``classify_media`` / ``is_ocp_query`` return the correct
   ``mediavocab.MediaType`` / bool, with no ``ovos_utils.ocp`` taxonomy in the
   loop.
2. **Source hygiene** -- the legacy ``ovos_utils.ocp.MediaType`` taxonomy import
   is gone from ``opm.py``/``legacy.py``, and the old ocp<->mediavocab media-type
   bridge functions/maps no longer exist in ``ocp_pipeline.bridge``.
"""
import inspect
import unittest

from mediavocab import MediaType

from ovos_utils.fakebus import FakeBus

import ocp_pipeline.opm as opm_module
import ocp_pipeline.legacy as legacy_module
import ocp_pipeline.bridge as bridge_module
from ocp_pipeline.opm import OCPPipelineMatcher


class TestMediavocabNativeClassification(unittest.TestCase):
    """Real classifier; assert mediavocab-native results."""

    @classmethod
    def setUpClass(cls):
        cls.ocp = OCPPipelineMatcher(bus=FakeBus())

    def test_classify_media_returns_mediavocab_types(self):
        cases = {
            "play some music": MediaType.MUSIC,
            "watch a movie": MediaType.MOVIE,
            "play a podcast": MediaType.PODCAST,
            # mediavocab has no ANIME label -> EPISODIC_SERIES
            "i want to watch an anime": MediaType.EPISODIC_SERIES,
            # mediavocab folds news into RADIO
            "play the news": MediaType.RADIO,
        }
        for query, expected in cases.items():
            media, conf = self.ocp.classify_media(query, "en-US")
            self.assertIsInstance(media, MediaType)
            self.assertEqual(media, expected,
                             f"{query!r} -> {media} (expected {expected})")
            self.assertIsInstance(conf, float)
            self.assertGreater(conf, 0.0)

    def test_non_media_is_generic_and_not_ocp(self):
        media, conf = self.ocp.classify_media("what time is it", "en-US")
        self.assertEqual(media, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)
        is_ocp, _ = self.ocp.is_ocp_query("what time is it", "en-US")
        self.assertFalse(is_ocp)

    def test_is_ocp_query_true_for_media(self):
        for query in ("play some music", "watch a movie", "play a podcast",
                      "i want to watch an anime", "play the news"):
            is_ocp, _ = self.ocp.is_ocp_query(query, "en-US")
            self.assertTrue(is_ocp, f"{query!r} should be an OCP query")


class TestNoOcpTaxonomyInSource(unittest.TestCase):
    """The legacy ocp.MediaType taxonomy and its bridge must be gone."""

    def test_opm_does_not_import_ocp_mediatype(self):
        src = inspect.getsource(opm_module)
        self.assertNotIn("from ovos_utils.ocp import MediaType", src)
        # also guard against `ovos_utils.ocp ... MediaType` on one import line
        for line in src.splitlines():
            if "import" in line and "ovos_utils.ocp" in line:
                self.assertNotIn("MediaType", line,
                                 f"opm.py still imports ocp.MediaType: {line!r}")
        # the module's MediaType symbol must be the mediavocab one
        self.assertIs(opm_module.MediaType, MediaType)

    def test_legacy_uses_mediavocab_mediatype(self):
        self.assertIs(legacy_module.MediaType, MediaType)
        src = inspect.getsource(legacy_module)
        for line in src.splitlines():
            if "import" in line and "ovos_utils.ocp" in line:
                self.assertNotIn("MediaType", line)

    def test_bridge_has_no_media_type_bridge(self):
        for gone in ("ocp_media_type_to_mediavocab",
                     "mediavocab_media_type_to_ocp",
                     "_OCP_TO_MV_MEDIA", "_MV_TO_OCP_MEDIA"):
            self.assertFalse(hasattr(bridge_module, gone),
                             f"bridge.py still exposes {gone}")

    def test_bridge_keeps_playback_selector_and_signals(self):
        # the playback backend-selector map is MediaEntry structure, kept
        self.assertTrue(hasattr(bridge_module, "mediavocab_playback_to_ocp"))
        self.assertTrue(hasattr(bridge_module, "media_type_to_signals"))
        self.assertTrue(hasattr(bridge_module, "release_to_ocp_result"))


if __name__ == "__main__":
    unittest.main()
