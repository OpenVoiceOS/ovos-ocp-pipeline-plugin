"""Every shipped locale must register under padacioso.

padatious drops a blank sample. padacioso expands it, and ovos-spec-tools
raises MalformedTemplate, which aborts OCPPipelineMatcher for the whole
language: that language then has no high-tier matcher, and only
match_medium answers. A user reported it for nl-NL, where every shipped
intent file carried a blank sample.

No test covered a non-English OCP locale before this one, which is why a
defect in 4 of the 13 shipped locales shipped.
"""
import os
import unittest
from os.path import dirname, isdir, join
from unittest.mock import patch

from padacioso import IntentContainer

import ocp_pipeline.opm as opm
from ocp_pipeline.opm import OCPPipelineMatcher

LOCALE_ROOT = join(dirname(opm.__file__), "locale")
SHIPPED = sorted(d for d in os.listdir(LOCALE_ROOT)
                 if isdir(join(LOCALE_ROOT, d)))


def _load(lang):
    """The real loader, asked for one language."""
    config = {"lang": lang, "secondary_langs": []}
    with patch.object(opm, "Configuration", lambda: config):
        return OCPPipelineMatcher.load_resource_files()[lang]


class TestEveryLocaleRegistersUnderPadacioso(unittest.TestCase):

    def test_at_least_the_known_locales_ship(self):
        # a guard on the guard: an empty list would let the loop below pass
        # without registering anything.
        self.assertGreaterEqual(len(SHIPPED), 10)
        self.assertIn("nl-NL", SHIPPED)
        self.assertIn("en-US", SHIPPED)

    def test_every_shipped_locale_registers(self):
        failures = []
        for lang in SHIPPED:
            container = IntentContainer()
            for name, samples in _load(lang).items():
                try:
                    container.add_intent(name.replace(".intent", ""), samples)
                except Exception as exc:
                    failures.append("%s/%s: %s: %s"
                                    % (lang, name, type(exc).__name__, exc))
        self.assertEqual(failures, [], "locales failed to register: %s"
                         % "; ".join(failures))

    def test_the_loader_drops_a_blank_sample(self):
        # the unit under the behaviour above: whatever a translator's editor
        # leaves at the end of a file, no empty sample reaches the engine.
        for lang in SHIPPED:
            for name, samples in _load(lang).items():
                self.assertTrue(all(s.strip() for s in samples),
                                "%s/%s yields a blank sample" % (lang, name))


if __name__ == "__main__":
    unittest.main()
