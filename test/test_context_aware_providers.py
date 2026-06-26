"""Request-context kwargs + the safe-search wrapper in the OCP pipeline.

Tests :meth:`OCPPipelineMatcher._build_query_context` (which now returns a plain
kwargs ``dict``, not a ``QueryContext`` object) and the :meth:`_safe_search`
wrapper in isolation (no full OCPPipelineMatcher init, which needs the bus/GUI
stack). The MediaProvider contract is a single
``search(signals, lang="en-us", *, supported_playback_types, blocked_genres,
region, session_id)`` call — there is no QueryContext / serves() / matches()
routing API.
"""
import unittest
from unittest.mock import MagicMock

from ocp_pipeline.opm import OCPPipelineMatcher


class _Stub:
    """Minimal stand-in exposing only ``config`` for the unbound methods."""
    def __init__(self, config):
        self.config = config


class TestBuildQueryContext(unittest.TestCase):
    def _ctx(self, config):
        return OCPPipelineMatcher._build_query_context(_Stub(config), "en-us")

    def test_returns_plain_dict(self):
        ctx = self._ctx({})
        self.assertIsInstance(ctx, dict)
        # the four explicit search() context kwargs, no QueryContext object
        self.assertEqual(
            set(ctx),
            {"supported_playback_types", "blocked_genres", "region", "session_id"})

    def test_adult_blocked_by_default(self):
        ctx = self._ctx({})
        self.assertIn("adult", ctx["blocked_genres"])
        self.assertEqual(ctx["supported_playback_types"], set())  # permissive
        self.assertIsNone(ctx["region"])
        self.assertIsNone(ctx["session_id"])

    def test_region_from_location_config(self):
        ctx = self._ctx({"location": {"city": {"region": {"country": {"code": "PT"}}}}})
        self.assertEqual(ctx["region"], "PT")

    def test_allow_adult_content_lifts_block(self):
        ctx = self._ctx({"allow_adult_content": True})
        self.assertNotIn("adult", ctx["blocked_genres"])

    def test_supported_playback_types_from_config(self):
        ctx = self._ctx({"media": {"supported_playback_types": ["audio"]}})
        self.assertEqual(ctx["supported_playback_types"], {"audio"})

    def test_custom_blocked_genres(self):
        ctx = self._ctx({"media_content_filter": {"blocked_genres": ["adult", "violence"]}})
        self.assertEqual(ctx["blocked_genres"], {"adult", "violence"})


class TestSafeSearch(unittest.TestCase):
    def test_forwards_lang_and_context_kwargs(self):
        prov = MagicMock()
        prov.search.return_value = ["r"]
        out = OCPPipelineMatcher._safe_search(
            prov, "sig", "en-us", supported_playback_types={"audio"},
            blocked_genres=set())
        self.assertEqual(out, ["r"])
        prov.search.assert_called_once_with(
            "sig", lang="en-us", supported_playback_types={"audio"},
            blocked_genres=set())

    def test_none_normalised_to_empty_list(self):
        prov = MagicMock()
        prov.search.return_value = None
        self.assertEqual(OCPPipelineMatcher._safe_search(prov, "sig", "en-us"), [])

    def test_exception_absorbed(self):
        prov = MagicMock()
        prov.name = "mock.boom"
        prov.search.side_effect = RuntimeError("boom")
        # one provider raising must not propagate — returns []
        self.assertEqual(OCPPipelineMatcher._safe_search(prov, "sig", "en-us"), [])


if __name__ == "__main__":
    unittest.main()
