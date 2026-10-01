"""Regression tests for ``handle_play_intent`` -- the REAL bus entry point
for the ``ocp:play`` intent, as opposed to ``_search``/``_search_providers``
which #156's tests drove directly.

A live-run investigation found that ``handle_play_intent`` early-returns with
the "no.media.skills" dialog whenever ``self.skill_aliases`` is empty --
without ever considering ``self.media_providers``. Legacy CommonPlaySkills
populate ``skill_aliases`` when they register over the bus; in-process
MediaProviders never do (they live in ``self.media_providers`` instead).
So a provider-only deployment (zero legacy CPS skills, one or more
MediaProviders) could never reach ``_search``/``_search_providers`` at all --
the dispatch died at the guard before ``_search`` ran.
"""
import unittest
from unittest.mock import MagicMock

from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType

from ocp_pipeline.opm import OCPPipelineMatcher


def _provider(name):
    prov = MagicMock()
    prov.name = name
    return prov


def _provider_result(uri, title, confidence=90, skill_id="mock.provider"):
    return {"uri": uri, "title": title, "artist": "", "image": "",
            "length": 0, "media_type": int(MediaType.MUSIC), "playback": 2,
            "match_confidence": confidence, "skill_id": skill_id}


def _message(query="metallica", media_type=MediaType.MUSIC, session_id="s1"):
    return Message("ocp:play",
                   {"lang": "en-us", "query": query, "media_type": media_type},
                   {"session": {"session_id": session_id}})


def _make_pipeline(skill_aliases=None, media_providers=None):
    """Bare OCPPipelineMatcher, bypassing the heavy bus/padatious __init__ --
    same pattern as test_media_provider_dispatch.py / test_pipeline.py."""
    p = OCPPipelineMatcher.__new__(OCPPipelineMatcher)
    p.bus = FakeBus()
    p.ocp_sessions = {}
    p.skill_aliases = skill_aliases if skill_aliases is not None else {}
    p.media_providers = media_providers if media_providers is not None else {}
    p.media2skill = {m: [] for m in MediaType}
    p.config = {"filter_media": False, "filter_SEI": False, "min_score": 0}
    p.search_lock = MagicMock()
    p.search_lock.__enter__ = MagicMock(return_value=None)
    p.search_lock.__exit__ = MagicMock(return_value=None)
    p._enclosure = MagicMock()
    p.speak_dialog = MagicMock()
    p.get_response = MagicMock(return_value=None)
    p.set_context = MagicMock()
    p.update_player_proxy = MagicMock()
    p.legacy_play = MagicMock()
    p.ocp_api = MagicMock()
    return p


class TestHandlePlayIntentEntryPoint(unittest.TestCase):
    """Drives the REAL ``handle_play_intent`` bus handler, not the private
    search helpers -- this is the path #156's tests never exercised."""

    def test_zero_skills_zero_providers_still_speaks_no_media_skills(self):
        """The genuine "nothing at all is registered" case must still be
        rejected -- the fix must not turn the guard into a no-op."""
        p = _make_pipeline(skill_aliases={}, media_providers={})
        p._search = MagicMock()

        p.handle_play_intent(_message())

        p.speak_dialog.assert_called_once_with("no.media.skills")
        p._search.assert_not_called()

    def test_provider_only_deployment_reaches_provider_dispatch(self):
        """Zero legacy skills (empty skill_aliases), one in-process
        MediaProvider: dispatch must reach ``_search`` -- the guard must not
        die before ``_search_providers`` ever runs."""
        p = _make_pipeline(skill_aliases={},
                           media_providers={"mock.provider": _provider("mock.provider")})
        p._search = MagicMock(return_value=[])

        p.handle_play_intent(_message())

        p._search.assert_called_once()
        for call in p.speak_dialog.call_args_list:
            self.assertNotEqual(call.args[0], "no.media.skills")

    def test_provider_only_happy_path_emits_play(self):
        """A provider-only deployment must be able to actually play something,
        not just clear the "no.media.skills" guard."""
        p = _make_pipeline(skill_aliases={},
                           media_providers={"mock.provider": _provider("mock.provider")})
        p._execute_query = MagicMock(return_value=[])  # no legacy bus results
        p._search_providers = MagicMock(
            return_value=[_provider_result("http://prov/1", "Master of Puppets")])
        # normalize_results/filter_results run for real -- exercises the same
        # dict -> MediaEntry conversion and confidence/SEI filtering a real
        # provider-only search would hit.
        p.config["filter_media"] = False
        p.config["filter_SEI"] = False
        p.config["min_score"] = 0

        player = MagicMock()
        player.ocp_available = True
        player.available_extractors = []
        p.get_player = MagicMock(return_value=player)

        message = _message()
        SessionManager.sessions[message.context["session"]["session_id"]] = \
            SessionManager.get(message)

        p.handle_play_intent(message)

        for call in p.speak_dialog.call_args_list:
            self.assertNotIn(call.args[0], ("no.media.skills", "cant.play"))
        p.ocp_api.play.assert_called_once()
        played_tracks = p.ocp_api.play.call_args.kwargs["tracks"]
        self.assertEqual(len(played_tracks), 1)
        self.assertEqual(played_tracks[0].uri, "http://prov/1")


if __name__ == "__main__":
    unittest.main()
