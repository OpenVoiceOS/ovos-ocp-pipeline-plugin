"""Shutdown must survive an instance whose ``__init__`` never finished.

``OVOSSkill.__del__`` calls ``shutdown()``. Most of this suite builds the
matcher with a ``__new__`` bypass to skip padatious training, so the garbage
collector reaches ``__del__`` on an object that has no ``intent_service`` and
no ``skill_id``. ``default_shutdown`` then raises ``AttributeError``, and
``ovos_workshop``'s own handler raises a second ``AttributeError`` on
``skill_id`` while logging the first, so nothing readable is ever logged.
"""
import unittest

from ocp_pipeline.opm import OCPPipelineMatcher


class TestShutdownOnHalfBuiltMatcher(unittest.TestCase):

    def test_shutdown_on_a_half_built_matcher_is_quiet(self):
        half = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
        # the two attributes the teardown path reads, neither one set
        self.assertFalse(hasattr(half, "intent_service"))
        self.assertFalse(hasattr(half, "skill_id"))
        half.shutdown()  # raised AttributeError before this fix

    def test_del_on_a_half_built_matcher_is_quiet(self):
        """The real path: the collector, not a direct call."""
        import gc

        raised = []
        half = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
        try:
            half.__del__()
        except AttributeError as e:  # pragma: no cover - the defect
            raised.append(e)
        del half
        gc.collect()
        self.assertEqual([], raised, f"__del__ raised: {raised}")

    def test_a_built_matcher_still_tears_down(self):
        """The guard must not turn a real shutdown into a no-op."""
        built = OCPPipelineMatcher(config={})
        self.assertTrue(hasattr(built, "intent_service"))
        calls = []
        built.default_shutdown = lambda: calls.append("default_shutdown")
        built.shutdown()
        self.assertEqual(["default_shutdown"], calls)


if __name__ == "__main__":
    unittest.main()
