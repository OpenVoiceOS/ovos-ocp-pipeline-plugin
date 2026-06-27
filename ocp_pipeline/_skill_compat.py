"""Skill-framework conveniences, reconstructed without ovos-workshop.

This module re-implements the handful of ``OVOSAbstractApplication`` /
``OVOSSkill`` conveniences that the OCP pipeline used to inherit, so that the
plugin can depend only on its real requirements (``ovos-plugin-manager`` for
the pipeline base, ``ovos-utils`` for the event machinery, and
``ovos-spec-tools`` for OVOS-INTENT-2 locale resources).

Nothing here needs the skills framework: each helper is a thin wrapper over a
messagebus emit or over ``ovos-utils`` / ``ovos-spec-tools`` primitives.

Precedent: the adapt pipeline (``ovos_adapt/opm.py``) is likewise decoupled
from ovos-workshop and registers its handlers with plain ``self.bus.on(...)``.
"""
import time
from os.path import dirname, join
from typing import Callable, Dict, List, Optional

from ovos_bus_client.message import Message, dig_for_message
from ovos_bus_client.session import SessionManager
from ovos_bus_client.util import get_message_lang
from ovos_config import Configuration
from ovos_spec_tools import DialogRenderer, LocaleResources, standardize_lang
from ovos_utils.events import EventContainer, create_wrapper, get_handler_name
from ovos_utils.log import LOG

# .voc/.dialog resources shipped with this plugin (locale/<lang>/<name>.{voc,dialog}).
# Loaded/matched via ovos-spec-tools (OVOS-INTENT-2) instead of ovos-workshop's
# skill resource machinery.
LOCALE_DIR = join(dirname(__file__), "locale")


class SkillCompatMixin:
    """Reconstructs the OVOSAbstractApplication conveniences OCP relied on.

    Consumers must set ``self.bus``, ``self.config`` (the pipeline base does
    this) and ``self.skill_id`` before using these helpers, then call
    ``self._init_skill_compat()`` from their ``__init__``.
    """

    def _init_skill_compat(self):
        # event bookkeeping (replaces OVOSSkill.events)
        self.events = EventContainer(self.bus)
        # OVOS-INTENT-2 locale resources (.voc/.dialog), replaces skill resources
        self._resources = LocaleResources(skill_locale=LOCALE_DIR)
        # caches DialogRenderer per dialog key
        self._dialog_renderers: Dict[str, DialogRenderer] = {}
        # pending get_response answers, keyed by session_id
        self._responses: Dict[str, Optional[List[str]]] = {}

    # -- language -----------------------------------------------------------
    @property
    def lang(self) -> str:
        """Current BCP-47 language, from the active message or config."""
        message = dig_for_message()
        if message:
            return standardize_lang(get_message_lang(message))
        return standardize_lang(Configuration().get("lang", "en-US"))

    # -- events -------------------------------------------------------------
    def add_event(self, name: str, handler: Callable,
                  is_intent: bool = False):
        """Register a bus handler, tracked for clean shutdown.

        Faithful reconstruction of ``OVOSSkill.add_event`` for this plugin's
        usage: every call here passes ``handler_info=None``, so the only extra
        behaviour ``is_intent=True`` adds is emitting ``ovos.utterance.handled``
        on completion.

        NOTE: ovos-core already emits ``ovos.utterance.handled`` itself after
        handling a pipeline match (intent_services/service.py, stop_service.py),
        so this self-emission is redundant in a normal stack. It is preserved
        for faithfulness (and for stacks that route OCP intents directly).
        """
        skill_data = {"name": get_handler_name(handler)}

        def on_end(message):
            if is_intent:
                self.bus.emit(message.forward("ovos.utterance.handled",
                                              skill_data))

        def on_error(error, message):
            LOG.error(f"Error handling event '{name}' : {error}")

        wrapper = create_wrapper(handler, self.skill_id, None, on_end, on_error)
        return self.events.add(name, wrapper)

    def default_shutdown(self):
        """Remove all events registered via :meth:`add_event`."""
        self.events.clear()

    # -- enclosure ----------------------------------------------------------
    def mouth_think(self):
        """Animate the Mark-1 mouth during search (enclosure.mouth.think)."""
        message = dig_for_message() or Message("")
        self.bus.emit(message.forward("enclosure.mouth.think"))

    # -- vocab --------------------------------------------------------------
    def remove_voc(self, utt: str, voc_filename: str,
                   lang: Optional[str] = None) -> str:
        """Strip any whole-word vocab match from ``utt`` (OVOS-INTENT-2)."""
        if not utt:
            return utt
        lang = standardize_lang(lang or self.lang)
        try:
            return self._resources.remove_voc(utt, voc_filename, lang)
        except Exception as e:
            LOG.debug(f"remove_voc({voc_filename}) failed: {e}")
            return utt

    # -- speech -------------------------------------------------------------
    def speak(self, utterance: str, expect_response: bool = False):
        """Emit a ``speak`` message for the current session."""
        message = dig_for_message() or Message("")
        data = {"utterance": utterance,
                "expect_response": expect_response,
                "meta": {"skill": self.skill_id},
                "lang": self.lang}
        m = message.forward("speak", data)
        m.context["skill_id"] = self.skill_id
        self.bus.emit(m)

    def speak_dialog(self, key: str, data: Optional[dict] = None,
                     expect_response: bool = False):
        """Render and speak a random line from ``<key>.dialog``."""
        renderer = self._dialog_renderers.get(key)
        if renderer is None:
            renderer = DialogRenderer(self._resources, key)
            self._dialog_renderers[key] = renderer
        try:
            utterance = renderer.render(self.lang, slots=data or {})
        except Exception as e:
            LOG.error(f"failed to render dialog '{key}': {e}")
            utterance = key
        self.speak(utterance, expect_response=expect_response)

    # -- context ------------------------------------------------------------
    def set_context(self, context: str, word: str = "", origin: str = ""):
        """Add adapt intent context (mirrors OVOSSkill.set_context)."""
        if not isinstance(context, str):
            raise ValueError("Context should be a string")
        if not isinstance(word, str):
            raise ValueError("Word should be a string")
        message = dig_for_message() or Message("")
        self.bus.emit(message.forward("add_context",
                                      {"context": context, "word": word,
                                       "origin": origin or self.skill_id}))

    # -- get_response -------------------------------------------------------
    def _handle_get_response(self, message: Message):
        """Capture an utterance routed back from the converse subsystem."""
        sess = SessionManager.get(message)
        if sess.session_id not in self._responses:
            return  # not for us
        self._responses[sess.session_id] = message.data.get("utterances", [])

    def get_response(self, dialog: str = "", data: Optional[dict] = None,
                     num_retries: int = -1,
                     message: Optional[Message] = None) -> Optional[str]:
        """Prompt the user and wait for a spoken response.

        Reconstructs the essential ``OVOSSkill.get_response`` flow without the
        skills framework: enable response mode for the session, speak the
        prompt, listen, and wait for ovos-core's converse subsystem to route
        the user's utterance back via ``<skill_id>.converse.get_response``.
        """
        message = message or dig_for_message() or \
            Message("mycroft.mic.listen", context={"skill_id": self.skill_id})
        session = SessionManager.get(message)
        session.enable_response_mode(self.skill_id)
        message.context["session"] = session.serialize()

        self._responses[session.session_id] = []
        ev_name = f"{self.skill_id}.converse.get_response"
        self.bus.on(ev_name, self._handle_get_response)
        self.bus.emit(message.forward("skill.converse.get_response.enable",
                                      {"skill_id": self.skill_id}))

        timeout = Configuration().get("skills", {}).get("get_response_timeout", 20)
        retries = 0
        answer = None
        try:
            while True:
                if dialog:
                    self.speak_dialog(dialog, data, expect_response=True)
                else:
                    self.bus.emit(message.forward("mycroft.mic.listen"))

                start = time.time()
                while time.time() - start <= timeout:
                    resp = self._responses.get(session.session_id)
                    if resp is None:  # aborted
                        break
                    if resp:
                        answer = resp[0]
                        break
                    time.sleep(0.1)

                if answer is not None or self._responses.get(session.session_id) is None:
                    break
                retries += 1
                if 0 <= num_retries < retries:
                    break
                self._responses[session.session_id] = []  # re-prompt
        finally:
            self.bus.remove(ev_name, self._handle_get_response)
            self._responses.pop(session.session_id, None)
            session.disable_response_mode(self.skill_id)
            message.context["session"] = session.serialize()
            self.bus.emit(message.forward("skill.converse.get_response.disable",
                                          {"skill_id": self.skill_id}))
        return answer
