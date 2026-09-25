"""A comment line in a .voc is not vocabulary.

OVOS-INTENT-2 section 5 step 3 says a conformant reader skips comment lines.
Every .voc in the fleet opens with an auto-translation header, so a reader that
keeps them puts the header text in the word set that
``_is_bare_media_request`` strips a phrase with.
"""
import os
import unittest
from os.path import join
from tempfile import TemporaryDirectory

import ovos_media_classifier.keyword as clf_keyword

import ocp_pipeline.opm as opm
from ocp_pipeline.opm import OCPPipelineMatcher

HEADER = "# auto translated from en-us to xx-xx"


class TestVocCommentLines(unittest.TestCase):

    def setUp(self):
        self._locale = opm.LOCALE_DIR
        self._clf_locale = clf_keyword._LOCALE_DIR
        self._tmp = TemporaryDirectory()
        root = self._tmp.name
        # one locale the reader will match, holding one commented resource
        folder = join(root, "xx-xx")
        os.makedirs(folder)
        with open(join(folder, "Play.voc"), "w", encoding="utf-8") as f:
            f.write(HEADER + "\n")
            f.write("play\n")
            f.write("  # an indented comment is still a comment\n")
        # an empty second root, so only the fixture above is read
        self._empty = TemporaryDirectory()
        os.makedirs(join(self._empty.name, "xx-xx"))

        opm.LOCALE_DIR = root
        clf_keyword._LOCALE_DIR = self._empty.name
        OCPPipelineMatcher._voc_cache.clear()

    def tearDown(self):
        opm.LOCALE_DIR = self._locale
        clf_keyword._LOCALE_DIR = self._clf_locale
        OCPPipelineMatcher._voc_cache.clear()
        self._tmp.cleanup()
        self._empty.cleanup()

    def test_reader_is_the_spec_reader(self):
        """The word set comes from ovos_spec_tools, not a local line loop."""
        import ocp_pipeline.opm as mod
        calls = []
        real = mod.read_resource_file

        def spy(path):
            calls.append(str(path))
            return real(path)

        mod.read_resource_file = spy
        try:
            OCPPipelineMatcher._voc_cache.clear()
            OCPPipelineMatcher._voc_words("xx-xx")
        finally:
            mod.read_resource_file = real
        self.assertTrue(calls, "_voc_words did not go through read_resource_file")
        self.assertTrue(any(c.endswith("Play.voc") for c in calls), calls)

    def test_bare_request_strips_whole_words_only(self):
        """A keyword inside a longer word is not stripped.

        The old hand-rolled subtraction padded with spaces, so it also matched
        only whole words, but it missed a keyword next to punctuation. The
        spec tools' stripper anchors on word boundaries instead.
        """
        # _is_bare_media_request touches only the _voc_words classmethod, so
        # the class stands in for an instance and no plugin is constructed
        bare = OCPPipelineMatcher._is_bare_media_request
        cls = OCPPipelineMatcher
        self.assertTrue(bare(cls, "play", "xx-xx"))
        self.assertTrue(bare(cls, "  PLAY  ", "xx-xx"))
        self.assertFalse(bare(cls, "playlist", "xx-xx"))
        self.assertFalse(bare(cls, "play thriller", "xx-xx"))

    def test_comment_lines_are_not_words(self):
        words = OCPPipelineMatcher._voc_words("xx-xx")
        self.assertIn("play", words)
        for word in words:
            self.assertFalse(
                word.startswith("#"),
                f"comment line entered the word set: {word!r}")
        self.assertEqual(["play"], words)


if __name__ == "__main__":
    unittest.main()


class TestRepeatedKeyword(unittest.TestCase):
    """The one shape where strip_samples and the old subtraction disagree.

    The subtraction this branch replaces was a non-overlapping
    ``residual.replace(f" {word} ", " ")``. Two adjacent copies of one keyword
    share the space between them, so the first replacement consumed it and the
    second copy was never matched: ``"play play"`` came back as a title.
    ``strip_samples`` anchors on word boundaries and removes both.

    Without this, reverting the ``_is_bare_media_request`` hunk leaves the
    suite green, and the behavioural half of the change is carried by nothing.
    """

    def test_a_keyword_said_twice_is_still_bare(self):
        bare = OCPPipelineMatcher._is_bare_media_request
        cls = OCPPipelineMatcher
        self.assertIn("play", OCPPipelineMatcher._voc_words("en-US"))
        # True here, False under the old non-overlapping replace loop
        self.assertTrue(bare(cls, "play play", "en-US"))
        self.assertTrue(bare(cls, "play play play", "en-US"))
        # unchanged by the swap, so the assertion above is the only new claim
        self.assertTrue(bare(cls, "play", "en-US"))
        self.assertFalse(bare(cls, "play thriller", "en-US"))


class TestTemplateExpansion(unittest.TestCase):
    """OVOS-INTENT-2 section 5 step 4: a .voc line is a template.

    ``read_resource_file`` returns template lines verbatim. Without the
    INTENT-1 expander a line such as ``(play|start)`` becomes one literal
    keyword that no utterance ever contains, so the alternatives it names are
    silently absent from the word set.
    """

    def setUp(self):
        import ovos_media_classifier.keyword as clf_keyword
        import ocp_pipeline.opm as opm
        self._locale = opm.LOCALE_DIR
        self._clf_locale = clf_keyword._LOCALE_DIR
        self._opm, self._clf = opm, clf_keyword
        self._tmp = TemporaryDirectory()
        self._empty = TemporaryDirectory()
        folder = join(self._tmp.name, "xx-xx")
        os.makedirs(folder)
        os.makedirs(join(self._empty.name, "xx-xx"))
        with open(join(folder, "Play.voc"), "w", encoding="utf-8") as f:
            f.write("(play|start)\n")
            f.write("[the] radio\n")
        opm.LOCALE_DIR = self._tmp.name
        clf_keyword._LOCALE_DIR = self._empty.name
        OCPPipelineMatcher._voc_cache.clear()

    def tearDown(self):
        self._opm.LOCALE_DIR = self._locale
        self._clf._LOCALE_DIR = self._clf_locale
        OCPPipelineMatcher._voc_cache.clear()
        self._tmp.cleanup()
        self._empty.cleanup()

    def test_a_template_line_becomes_its_samples(self):
        words = set(OCPPipelineMatcher._voc_words("xx-xx"))
        self.assertEqual({"play", "start", "the radio", "radio"}, words)
        # the unexpanded literal must not be a keyword
        self.assertNotIn("(play|start)", words)

    def test_an_alternative_counts_as_vocabulary(self):
        bare = OCPPipelineMatcher._is_bare_media_request
        cls = OCPPipelineMatcher
        self.assertTrue(bare(cls, "start", "xx-xx"))
        self.assertTrue(bare(cls, "start the radio", "xx-xx"))
        self.assertFalse(bare(cls, "start thriller", "xx-xx"))
