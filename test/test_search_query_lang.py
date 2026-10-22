"""A search query with no session context must use the box's language.

``handle_search_query`` read the language straight out of
``message.context["session"]`` and fell back to a hardcoded ``"en-us"``. A
message that carries no session context is not an English request: it is a
request that did not name a session, and OVOS-SESSION resolves it to the
default session, which carries the language this box is configured for.

The old expression could only ever produce ``en-us`` for such a message, so on
a Dutch box the whole query was classified and searched in English.
``SessionManager.get`` answers correctly for both cases: the message's own
session when it has one, the default session when it does not.
"""
import unittest
from unittest.mock import patch

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session, SessionManager
from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType

from ocp_pipeline.opm import OCPPipelineMatcher


class TestSearchQueryLang(unittest.TestCase):

    def setUp(self):
        self.ocp = OCPPipelineMatcher(bus=FakeBus(), config={})
        self.ocp.skill_aliases["test"] = ["Test Skill"]
        self.ocp.media2skill = {m: ["Test Skill"] for m in MediaType}
        self._default_lang = SessionManager.get_default_session().lang

    def tearDown(self):
        SessionManager.get_default_session().lang = self._default_lang

    def _lang_used(self, message):
        """Run handle_search_query and return the language it searched with."""
        seen = []

        def classify(utterance, lang, message=None):
            seen.append(lang)
            return MediaType.GENERIC, 0.0

        with patch.object(self.ocp, "classify_media", side_effect=classify), \
                patch.object(self.ocp, "_search", return_value=[]), \
                patch.object(self.ocp, "select_best", return_value=None):
            self.ocp.handle_search_query(message)
        return seen[0]

    def test_no_session_context_uses_the_configured_language(self):
        """The regression: a Dutch box must not search in English."""
        SessionManager.get_default_session().lang = "nl-NL"
        message = Message("ovos.common_play.search",
                          {"utterance": "speel wat muziek"})
        self.assertEqual("nl-NL", self._lang_used(message))

    def test_session_language_is_used_when_the_message_carries_one(self):
        """The control: an explicit session still wins over the default."""
        SessionManager.get_default_session().lang = "nl-NL"
        session = Session("test-session", lang="pt-PT")
        message = Message("ovos.common_play.search",
                          {"utterance": "toca musica"},
                          {"session": session.serialize()})
        self.assertEqual("pt-PT", self._lang_used(message))

    def test_explicit_lang_in_data_still_wins(self):
        """The second control: data['lang'] is unchanged by this fix."""
        SessionManager.get_default_session().lang = "nl-NL"
        message = Message("ovos.common_play.search",
                          {"utterance": "play some music", "lang": "en-US"})
        self.assertEqual("en-US", self._lang_used(message))


if __name__ == "__main__":
    unittest.main()
