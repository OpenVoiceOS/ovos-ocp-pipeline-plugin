"""The per-utterance status sync must carry the caller's session.

``match_high`` emits ``ovos.common_play.status`` to sync player state before
it matches. It built a bare ``Message``, so every media utterance put a
message with no ``context`` on the bus. ``SessionManager`` logs

    WARNING - No session context in message:ovos.common_play.status

and resolves such a message to the DEFAULT session rather than the caller's.
Nothing changed language because of it — the default session carries the
configured language — but the reply is then answered against the wrong
session, and OVOS-SESSION requires a derived message to carry the context of
the message it came from.

The launch-time emit is deliberately left context-free and is not tested
here: it runs at load, there is no caller and no session, and the default
session is the right one for a sync that belongs to the service rather than
to any request.
"""
import unittest

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType

from ocp_pipeline.opm import OCPPipelineMatcher


class TestStatusEmitContext(unittest.TestCase):

    def setUp(self):
        self.bus = FakeBus()
        self.ocp = OCPPipelineMatcher(bus=self.bus, config={})
        self.ocp.skill_aliases["test"] = ["Test Skill"]
        self.ocp.media2skill = {m: ["Test Skill"] for m in MediaType}
        self.seen = []
        self.bus.on("ovos.common_play.status", self.seen.append)

    def test_status_sync_carries_the_caller_session(self):
        session = Session("a-session", lang="pt-PT")
        message = Message("recognizer_loop:utterance",
                          {"utterances": ["play metallica"]},
                          {"session": session.serialize()})
        self.ocp.match_high(["play metallica"], "en-US", message)
        self.assertTrue(self.seen, "no status sync was emitted")
        got = self.seen[0].context.get("session") or {}
        self.assertEqual("a-session", got.get("session_id"))
        self.assertEqual("pt-PT", got.get("lang"))

    def test_status_sync_still_emitted_without_a_caller_message(self):
        """The control: dropping the context must not drop the sync. A caller
        that passes no message still gets the bare form rather than nothing."""
        self.ocp.match_high(["play metallica"], "en-US")
        self.assertTrue(self.seen, "no status sync was emitted")
        self.assertEqual("ovos.common_play.status", self.seen[0].msg_type)


    def _match_with_the_message_only_as_an_argument(self, message):
        """Call ``match_high`` with no ``message`` of its own.

        ``dig_for_message`` inspects the ARGUMENTS of the frames on the stack,
        not their other locals, so the caller's message has to arrive here as a
        parameter. That is its production shape: a bus handler is given the
        message and calls on into code that was not.
        """
        self.ocp.match_high(["play metallica"], "en-US")

    def test_status_sync_digs_the_caller_message_out_of_the_stack(self):
        """The ``dig_for_message()`` fallback, pinned.

        A caller that passes no ``message`` may still have one on the stack,
        which is the case ``_get_player`` uses this idiom for. Without the
        fallback the sync goes out bare and resolves to the default session;
        with it the caller's session is carried. Deleting ``or
        dig_for_message()`` from opm.py must fail this test.
        """
        caller = Message("recognizer_loop:utterance",
                         {"utterances": ["play metallica"]},
                         {"session": Session("dug-session", lang="nl-NL").serialize()})
        self._match_with_the_message_only_as_an_argument(caller)
        self.assertTrue(self.seen, "no status sync was emitted")
        got = self.seen[0].context.get("session") or {}
        self.assertEqual("dug-session", got.get("session_id"))
        self.assertEqual("nl-NL", got.get("lang"))


if __name__ == "__main__":
    unittest.main()
