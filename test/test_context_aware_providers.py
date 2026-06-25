"""Context-aware MediaProvider gating in the OCP pipeline.

Tests the QueryContext construction + the provider-search tolerance shim in
isolation (no full OCPPipelineMatcher init, which needs the bus/GUI stack).
"""
import unittest
from unittest.mock import MagicMock

from ocp_pipeline.opm import OCPPipelineMatcher
from ovos_plugin_manager.templates.media_provider import QueryContext


class _Stub:
    """Minimal stand-in exposing only ``config`` for the unbound methods."""
    def __init__(self, config):
        self.config = config


class TestBuildQueryContext(unittest.TestCase):
    def _ctx(self, config):
        return OCPPipelineMatcher._build_query_context(_Stub(config), "en-us")

    def test_adult_blocked_by_default(self):
        ctx = self._ctx({})
        self.assertIsInstance(ctx, QueryContext)
        self.assertIn("adult", ctx.blocked_genres)
        self.assertEqual(ctx.lang, "en-us")
        self.assertEqual(ctx.supported_playback_types, set())  # permissive

    def test_allow_adult_content_lifts_block(self):
        ctx = self._ctx({"allow_adult_content": True})
        self.assertNotIn("adult", ctx.blocked_genres)

    def test_supported_playback_types_from_config(self):
        ctx = self._ctx({"media": {"supported_playback_types": ["audio"]}})
        self.assertEqual(ctx.supported_playback_types, {"audio"})

    def test_custom_blocked_genres(self):
        ctx = self._ctx({"media_content_filter": {"blocked_genres": ["adult", "violence"]}})
        self.assertEqual(ctx.blocked_genres, {"adult", "violence"})


class TestProviderSearchShim(unittest.TestCase):
    def test_passes_context_to_modern_provider(self):
        prov = MagicMock()
        prov.search_safe.return_value = ["r"]
        out = OCPPipelineMatcher._provider_search(prov, "sig", "ctx", "en-us")
        self.assertEqual(out, ["r"])
        prov.search_safe.assert_called_once_with("sig", "ctx", lang="en-us")

    def test_falls_back_for_legacy_provider(self):
        prov = MagicMock()
        # legacy search_safe(signals, lang) raises TypeError on the context arg
        prov.search_safe.side_effect = [TypeError("no context"), ["r"]]
        out = OCPPipelineMatcher._provider_search(prov, "sig", "ctx", "en-us")
        self.assertEqual(out, ["r"])
        self.assertEqual(prov.search_safe.call_count, 2)


if __name__ == "__main__":
    unittest.main()
