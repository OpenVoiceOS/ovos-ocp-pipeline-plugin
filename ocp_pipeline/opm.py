import os
import random
import threading
from dataclasses import dataclass
from os.path import join, dirname
from threading import RLock
from typing import Tuple, Optional, Dict, List, Union, Any

from langcodes import closest_match
from ovos_bus_client.apis.ocp import ClassicAudioServiceInterface
from ovos_bus_client.apis.ocp import OCPInterface, OCPQuery
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message, dig_for_message
from ovos_bus_client.session import SessionManager
from ovos_config import Configuration
from ovos_plugin_manager.ocp import available_extractors
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, ConfidenceMatcherPipeline, PipelinePlugin
from ovos_utils.lang import standardize_lang_tag, get_language_dir
from ovos_utils.log import LOG, deprecated, log_deprecation
from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType, PlaybackType, PlaybackMode, PlayerState, OCP_ID, \
    MediaEntry, Playlist, MediaState, TrackState, dict2entry, PluginStream
from ovos_workshop.app import OVOSAbstractApplication
from ovos_utils.xdg_utils import xdg_data_home
from ovos_config.meta import get_xdg_base
from ahocorasick_ner import AhocorasickNER
from ocp_pipeline.legacy import LegacyCommonPlay


@dataclass
class OCPPlayerProxy:
    """proxy object tracking the state of connected player devices (Sessions)"""
    session_id: str
    available_extractors: List[str]
    ocp_available: bool
    player_state: PlayerState = PlayerState.STOPPED
    media_state: MediaState = MediaState.UNKNOWN
    media_type: MediaType = MediaType.GENERIC
    skill_id: Optional[str] = None


# for easier typing
RawResultsList = List[Union[MediaEntry, Playlist, PluginStream, Dict[str, Any]]]
NormalizedResultsList = List[Union[MediaEntry, Playlist, PluginStream]]


class OCPPipelineMatcher(ConfidenceMatcherPipeline, OVOSAbstractApplication):
    intents = ["play.intent", "open.intent", "media_stop.intent",
               "next.intent", "prev.intent", "pause.intent",
               #  "play_favorites.intent", "like_song.intent",  # handled by ovos-media not ovos-audio, re-enable later
               "resume.intent", "save_game.intent", "load_game.intent"]
    intent_matchers = {}
    intent_cache = f"{xdg_data_home()}/{get_xdg_base()}/intent_cache"

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the OCPPipelineMatcher, setting up OCP and legacy audio interfaces, intent and event registration, player session tracking, skill and media mappings, and the AhocorasickNER for entity recognition.
        
        Parameters:
            bus (Optional[Union[MessageBusClient, FakeBus]]): The message bus for event communication. If not provided, a fake bus is used.
            config (Optional[Dict]): Optional configuration dictionary for pipeline and entity keyword setup.
        """
        OVOSAbstractApplication.__init__(
            self, bus=bus or FakeBus(), skill_id=OCP_ID, resources_dir=f"{dirname(__file__)}")
        ConfidenceMatcherPipeline.__init__(self, bus, config)

        self.ocp_api = OCPInterface(self.bus)
        self.legacy_api = ClassicAudioServiceInterface(self.bus)

        self.search_lock = RLock()
        self.ocp_sessions = {}  # session_id: PlaybackCapabilities

        self.skill_aliases = {
            # "skill_id": ["names"]
        }
        self.media2skill = {
            m: [] for m in MediaType
        }
        self.entity_csvs = self.config.get("entity_csvs", [])  # user defined keyword csv files
        self.ner = AhocorasickNER()

        self.register_ocp_api_events()
        self.register_ocp_intents()
        # request available Stream extractor plugins from OCP
        self.bus.emit(Message("ovos.common_play.SEI.get"))

    @classmethod
    def load_resource_files(cls):
        intents = {}
        langs = Configuration().get('secondary_langs', []) + [Configuration().get('lang', "en-US")]
        langs = set([standardize_lang_tag(l) for l in langs])
        for lang in langs:
            lang = standardize_lang_tag(lang)
            intents[lang] = {}
            locale_folder = get_language_dir(join(dirname(__file__), "locale"), lang)
            if locale_folder is not None:
                for f in os.listdir(locale_folder):
                    path = join(locale_folder, f)
                    if f in cls.intents:
                        with open(path) as intent:
                            samples = intent.read().split("\n")
                            for idx, s in enumerate(samples):
                                samples[idx] = s.replace("{{", "{").replace("}}", "}")
                            intents[lang][f] = samples
        return intents

    def register_ocp_api_events(self):
        """
        Register messagebus handlers for OCP events
        """
        self.add_event("ovos.common_play.search", self.handle_search_query)
        self.add_event("ovos.common_play.play_search", self.handle_play_search)
        self.add_event('ovos.common_play.status.response', self.handle_player_state_update)
        self.add_event('ovos.common_play.track.state', self.handle_track_state_update)

        self.add_event('ovos.common_play.register_keyword', self.handle_skill_keyword_register)
        self.add_event('ovos.common_play.deregister_keyword', self.handle_skill_keyword_deregister)
        self.add_event('ovos.common_play.announce', self.handle_skill_register)

        self.add_event("mycroft.audio.playing_track", self._handle_legacy_audio_start)
        self.add_event("mycroft.audio.queue_end", self._handle_legacy_audio_end)
        self.add_event("mycroft.audio.service.pause", self._handle_legacy_audio_pause)
        self.add_event("mycroft.audio.service.resume", self._handle_legacy_audio_resume)
        self.add_event("mycroft.audio.service.stop", self._handle_legacy_audio_stop)
        self.bus.emit(Message("ovos.common_play.status"))  # sync player state on launch

    @classmethod
    def load_intent_files(cls):
        intent_files = cls.load_resource_files()

        try:
            from ovos_padatious import IntentContainer
            is_padatious = True
        except ImportError:
            from padacioso import IntentContainer
            is_padatious = False
            LOG.warning("Padatious not available, using padacioso. intent matching will be orders of magnitude slower!")

        for lang, intent_data in intent_files.items():
            lang = standardize_lang_tag(lang)
            if is_padatious:
                cache = f"{cls.intent_cache}/{lang}"
                cls.intent_matchers[lang] = IntentContainer(cache)
            else:
                cls.intent_matchers[lang] = IntentContainer()
            for intent_name in cls.intents:
                samples = intent_data.get(intent_name)
                if samples:
                    LOG.debug(f"registering OCP intent: {intent_name}")
                    cls.intent_matchers[lang].add_intent(
                        intent_name.replace(".intent", ""), samples)
            if is_padatious:
                cls.intent_matchers[lang].train()

    def register_ocp_intents(self):
        self.load_intent_files()
        self.add_event("ocp:play", self.handle_play_intent, is_intent=True)
        self.add_event("ocp:play_favorites", self.handle_play_favorites_intent, is_intent=True)
        self.add_event("ocp:open", self.handle_open_intent, is_intent=True)
        self.add_event("ocp:next", self.handle_next_intent, is_intent=True)
        self.add_event("ocp:prev", self.handle_prev_intent, is_intent=True)
        self.add_event("ocp:pause", self.handle_pause_intent, is_intent=True)
        self.add_event("ocp:resume", self.handle_resume_intent, is_intent=True)
        self.add_event("ocp:media_stop", self.handle_stop_intent, is_intent=True)
        self.add_event("ocp:search_error", self.handle_search_error_intent, is_intent=True)
        self.add_event("ocp:like_song", self.handle_like_intent, is_intent=True)
        self.add_event("ocp:save_game", self.handle_save_intent, is_intent=True)
        self.add_event("ocp:load_game", self.handle_load_intent, is_intent=True)

    def update_player_proxy(self, player: OCPPlayerProxy):
        """remember OCP session state"""
        self.ocp_sessions[player.session_id] = player

    def handle_skill_register(self, message: Message):
        """
        Registers a skill's names and aliases as keywords for media type matching.
        
        Associates the skill's aliases with appropriate media type labels in the named entity recognizer, enabling accurate media intent classification and routing. Updates internal mappings of skills to media types and aliases.
        """
        skill_id = message.data["skill_id"]
        media = message.data.get("media_types") or \
                message.data.get("media_type") or []
        has_featured_media = message.data.get("featured_tracks", False)
        thumbnail = message.data.get("thumbnail", "")
        display_name = message.data["skill_name"].replace(" Skill", "")
        aliases = message.data.get("aliases", [display_name])
        LOG.info(f"Registering OCP Keyword for {skill_id} : {aliases}")
        self.skill_aliases[skill_id] = aliases

        for idx, m in enumerate(media):
            try:
                m = self._normalize_media_enum(m)
                self.media2skill[m].append(skill_id)
                media[idx] = m
            except:
                LOG.error(f"{skill_id} reported an invalid media_type: {m}")

        # TODO - review below and add missing
        # set bias in classifier
        # aliases -> {type}_streaming_service bias
        for a in aliases:
            if MediaType.MUSIC in media:
                self.ner.add_word("music_streaming_service", a)
            if MediaType.MOVIE in media:
                self.ner.add_word("movie_streaming_service", a)
            # if MediaType.SILENT_MOVIE in media:
            #    self.ner.add_word("silent_movie_streaming_service", a)
            # if MediaType.BLACK_WHITE_MOVIE in media:
            #    self.ner.add_word("bw_movie_streaming_service", a)
            if MediaType.SHORT_FILM in media:
                self.ner.add_word("shorts_streaming_service", a)
            if MediaType.PODCAST in media:
                self.ner.add_word("podcast_streaming_service", a)
            if MediaType.AUDIOBOOK in media:
                self.ner.add_word("audiobook_streaming_service", a)
            if MediaType.NEWS in media:
                self.ner.add_word("news_provider", a)
            if MediaType.TV in media:
                self.ner.add_word("tv_streaming_service", a)
            if MediaType.RADIO in media:
                self.ner.add_word("radio_streaming_service", a)
            if MediaType.ADULT in media:
                self.ner.add_word("porn_streaming_service", a)

    def handle_skill_keyword_register(self, message: Message):
        """
        Register skill-provided keywords and samples for entity recognition.
        
        Adds keywords from a CSV file and/or provided samples to the named entity recognizer for the specified skill and media type.
        """
        skill_id = message.data["skill_id"]
        kw_label = message.data["label"]
        media = message.data["media_type"]
        samples = message.data.get("samples", [])
        csv_path = message.data.get("csv")

        if csv_path:
            with open(csv_path) as f:
                lines = f.read().split("\n")[1:]
                for l in lines:
                    if not l.strip():
                        continue
                    label, value = l.split(",", 1)
                    self.ner.add_word(label, value)

        for s in samples:
            self.ner.add_word(kw_label, s)


    def handle_skill_keyword_deregister(self, message: Message):
        """
        Placeholder for deregistering skill-provided keywords from the entity recognizer.
        
        Currently not implemented.
        """
        skill_id = message.data["skill_id"]
        kw_label = message.data["label"]
        media = message.data["media_type"]
        # TODO

    def handle_track_state_update(self, message: Message):
        """
        Handles track state update messages and updates the player proxy to reflect active playback when a playing state is detected.
        
        Raises:
            ValueError: If the message does not contain a 'state' field.
        """
        state = message.data.get("state")
        if state is None:
            raise ValueError(f"Got state update message with no state: "
                             f"{message}")
        if isinstance(state, int):
            state = TrackState(state)
        player = self.get_player(message)
        if player.player_state != PlayerState.PLAYING and \
                state in [TrackState.PLAYING_AUDIO, TrackState.PLAYING_AUDIOSERVICE,
                          TrackState.PLAYING_VIDEO, TrackState.PLAYING_WEBVIEW,
                          TrackState.PLAYING_MPRIS]:
            player = self.get_player(message)
            player.player_state = PlayerState.PLAYING
            player = self._update_player_skill_id(player, message)
            LOG.info(f"Session: {player.session_id} OCP PlayerState: PlayerState.PLAYING")
            self.update_player_proxy(player)

    def handle_player_state_update(self, message: Message):
        """
        Handles 'ovos.common_play.status' messages with player status updates
        @param message: Message providing new "state" data
        """
        player = self.get_player(message)
        pstate: int = message.data.get("player_state")
        mstate: int = message.data.get("media_state")
        mtype: int = message.data.get("media_type")
        if pstate is not None:
            player.player_state = PlayerState(pstate)
            LOG.debug(f"Session: {player.session_id} PlayerState: {player.player_state}")
        if mstate is not None:
            player.media_state = MediaState(mstate)
            LOG.debug(f"Session: {player.session_id} MediaState: {player.media_state}")
        if mtype is not None:
            player.media_type = MediaType(pstate)
            LOG.debug(f"Session: {player.session_id} MediaType: {player.media_type}")
        player = self._update_player_skill_id(player, message)
        self.update_player_proxy(player)

    # pipeline
    def match_high(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """ exact matches only, handles playback control
        recommended after high confidence intents pipeline stage """

        if not len(self.skill_aliases):  # skill_id registered when skills load
            return None  # dont waste compute cycles, no media skills -> no match

        lang = self._get_closest_lang(lang)
        if lang is None:  # no intents registered for this lang
            return None

        utterance = utterances[0].lower()

        # avoid common confusion with alerts and parrot skill
        if (self.voc_match(utterance, "Alerts") or
                self.voc_match(utterance, "SoundIntents") or
                self.voc_match(utterance, "Parrot")):
            return None

        self.bus.emit(Message("ovos.common_play.status"))  # sync

        match = self.intent_matchers[lang].calc_intent(utterance)

        if hasattr(match, "name"):  # padatious
            match = {
                "name": match.name,
                "conf": match.conf,
                "entities": match.matches
            }

        if match["name"] is None:
            return None

        if match.get("conf", 1.0) < 0.7:
            LOG.debug(f"Ignoring low confidence OCP match: {match}")
            return None

        LOG.info(f"OCP match: {match}")

        player = self.get_player(message)

        if player.media_type == MediaType.GAME:
            # if the user is currently playing a game
            # disable: next/prev/shuffle/... intents
            # enable: load/save intents
            game_blacklist = ["next", "prev", "open", "like_song", "play_favorites"]
            if match["name"] in game_blacklist:
                LOG.info(f'Ignoring OCP intent match {match["name"]}, playing MediaType.GAME')
                return None
        else:
            # if no game is being played, disable game specific intents
            game_only = ["save_game", "load_game"]
            # TODO - allow load_game without being in game already
            #  this can only be done if we match skill_id
            if match["name"] in game_only:
                LOG.info(f'Ignoring OCP intent match {match["name"]}, not playing MediaType.GAME')
                return None

        if match["name"] == "play":
            query = match["entities"].pop("query")
            return self._process_play_query(query, utterance, lang, match)

        if match["name"] == "like_song" and player.media_type != MediaType.MUSIC:
            LOG.debug("Ignoring like_song intent, current media is not MediaType.MUSIC")
            return None

        if match["name"] not in ["open", "play_favorites"] and player.player_state == PlayerState.STOPPED:
            LOG.info(f'Ignoring OCP intent match {match["name"]}, OCP Virtual Player is not active')
            # next / previous / pause / resume not targeted
            # at OCP if playback is not happening / paused
            if match["name"] == "resume":
                # TODO - handle resume for last_played query, eg, previous day
                return None
            else:
                return None

        return IntentHandlerMatch(match_type=f'ocp:{match["name"]}',
                                  match_data=match,
                                  skill_id=OCP_ID,
                                  utterance=utterance)

    def match_medium(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """
        Performs medium-confidence intent matching for media playback queries using classifiers and entity extraction.
        
        Analyzes the first utterance to determine if it is an OCP (Open Common Play) query, classifies the requested media type, and extracts relevant entities. Returns an `IntentHandlerMatch` with extracted information if a match is found; otherwise, returns `None`.
        
        Returns:
            Optional[IntentHandlerMatch]: An intent match object containing media type, entities, query string, and confidence, or `None` if no match is found.
        """
        lang = standardize_lang_tag(lang)

        utterance = utterances[0].lower()
        # is this a OCP query ?
        is_ocp, bconf = self.is_ocp_query(utterance, lang)

        if not is_ocp:
            return None

        # classify the query media type
        media_type, confidence = self.classify_media(utterance, lang)

        # extract entities
        try:
            ents = {e["label"]: e["word"] for e in self.ner.tag(utterance)}
        except Exception as e:
            LOG.error(f"failed to extract media entities: ({e})")
            ents = {}

        # extract the query string
        query = self.remove_voc(utterance, "Play", lang).strip()

        return IntentHandlerMatch(match_type="ocp:play",
                                  match_data={"media_type": media_type,
                                              "entities": ents,
                                              "query": query,
                                              "is_ocp_conf": bconf,
                                              "conf": confidence},
                                  skill_id=OCP_ID,
                                  utterance=utterance)

    def match_low(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """
        Perform low-confidence matching of an utterance based on the presence of known OCP media keywords.
        
        Attempts to extract media-related entities from the utterance using the internal NER. If entities are found and the media type classification confidence meets a minimum threshold, returns an intent match for OCP playback; otherwise, returns None.
        
        Returns:
            IntentHandlerMatch: An intent match object if a suitable media keyword is found and classified with sufficient confidence, otherwise None.
        """
        utterance = utterances[0].lower()
        # extract entities
        try:
            ents = {e["label"]: e["word"] for e in self.ner.tag(utterance)}
        except Exception as e:
            LOG.error(f"failed to extract media entities: ({e})")
            ents = {}

        if not ents:
            return None

        lang = standardize_lang_tag(lang)

        # classify the query media type
        media_type, confidence = self.classify_media(utterance, lang)

        if confidence < 0.3:
            return None

        # extract the query string
        query = self.remove_voc(utterance, "Play", lang).strip()

        return IntentHandlerMatch(match_type="ocp:play",
                                  match_data={"media_type": media_type,
                                              "entities": ents,
                                              "query": query,
                                              "conf": float(confidence)},
                                  skill_id=OCP_ID,
                                  utterance=utterance)

    def _process_play_query(self, query:str, utterance: str, lang: str, match: dict = None,
                            message: Optional[Message] = None) -> Optional[IntentHandlerMatch]:
        """
        Process a play query to determine the appropriate playback action or search intent.
        
        If the query indicates a resume action (e.g., "play" while paused), returns a resume intent. Otherwise, prompts for missing queries, identifies explicitly requested skills, classifies the media type, extracts relevant entities, and constructs an intent match for playback.
        
        Parameters:
            query (str): The user's spoken or typed query.
            utterance (str): The original utterance from the user.
            lang (str): The language code for processing.
            match (dict, optional): Existing match data to include in the result.
            message (Message, optional): The message context for the request.
        
        Returns:
            Optional[IntentHandlerMatch]: An intent match object for playback, resume, or search error, or None if no action is determined.
        """
        lang = standardize_lang_tag(lang)
        match = match or {}
        player = self.get_player(message)
        # if media is currently paused, empty string means "resume playback"
        if player.player_state == PlayerState.PAUSED and \
                self._should_resume(query, lang, message=message):
            return IntentHandlerMatch(match_type="ocp:resume",
                                      match_data=match,
                                      skill_id=OCP_ID,
                                      utterance=utterance)

        if not query:
            # user just said "play", we are missing the search query
            phrase = self.get_response("play.what", num_retries=2)
            if not phrase:
                # let the error intent handler take action
                return IntentHandlerMatch(match_type="ocp:search_error",
                                          match_data=match,
                                          skill_id=OCP_ID,
                                          utterance=utterance)

        sess = SessionManager.get(message)
        # if a skill was explicitly requested, search it first
        valid_skills = [
            skill_id for skill_id, samples in self.skill_aliases.items()
            if skill_id not in sess.blacklisted_skills and
               any(s.lower() in utterance for s in samples)
        ]
        valid_labels = []
        if valid_skills:
            LOG.info(f"OCP specific skill names matched: {valid_skills}")
            for mtype, skills in self.media2skill.items():
                if any([s in skills for s in valid_skills]):
                    valid_labels.append(mtype)

        # classify the query media type
        media_type, conf = self.classify_media(utterance, lang, valid_labels=valid_labels)

        # remove play verb from the query string
        query = self.remove_voc(query, "Play", lang).strip()

        # extract entities
        try:
            ents = {e["label"]: e["word"] for e in self.ner.tag(utterance)}
        except Exception as e:
            LOG.error(f"failed to extract media entities: ({e})")
            ents = {}

        return IntentHandlerMatch(match_type="ocp:play",
                                  match_data={"media_type": media_type,
                                              "query": query,
                                              "entities": ents,
                                              "skills": valid_skills,
                                              "conf": match["conf"],
                                              "media_conf": float(conf),
                                              # "results": results,
                                              "lang": lang},
                                  skill_id=OCP_ID,
                                  utterance=utterance)

    # bus api
    def handle_search_query(self, message: Message):
        utterance = message.data["utterance"].lower()
        phrase = message.data.get("query", "") or utterance
        lang = message.data.get("lang") or message.context.get("session", {}).get("lang", "en-us")
        LOG.debug(f"Handle {message.msg_type} request: {phrase}")
        num = message.data.get("number", "")
        if num:
            phrase += " " + num

        lang = standardize_lang_tag(lang)
        # classify the query media type
        media_type, prob = self.classify_media(utterance, lang)
        # search common play skills
        results = self._search(phrase, media_type, lang, message=message)
        best = self.select_best(results, message)
        results = [r.as_dict if isinstance(best, (MediaEntry, Playlist)) else r
                   for r in results]
        if isinstance(best, (MediaEntry, Playlist)):
            best = best.as_dict
        self.bus.emit(message.response(data={"results": results,
                                             "best": best,
                                             "media_type_conf": float(prob)}))

    def handle_play_search(self, message: Message):
        LOG.info("searching and playing best OCP result")
        utterance = message.data["utterance"].lower()
        query = utterance
        match = self._process_play_query(query, utterance, self.lang, {"conf": 1.0})
        self.bus.emit(message.forward(match.match_type, match.match_data))

    def handle_play_favorites_intent(self, message: Message):
        LOG.info("playing favorite tracks")
        self.bus.emit(message.forward("ovos.common_play.liked_tracks.play"))

    # intent handlers
    @staticmethod
    def _normalize_media_enum(m: Union[int, MediaType]):
        if isinstance(m, MediaType):
            return m
        # convert int to enum
        for e in MediaType:
            if e == m:
                return e
        raise ValueError(f"{m} is not a valid media type")

    def handle_save_intent(self, message: Message):
        skill_id = self.get_player(message).skill_id
        self.bus.emit(message.forward(f"ovos.common_play.{skill_id}.save"))

    def handle_load_intent(self, message: Message):
        skill_id = self.get_player(message).skill_id
        self.bus.emit(message.forward(f"ovos.common_play.{skill_id}.load"))

    def handle_play_intent(self, message: Message):

        if not len(self.skill_aliases):  # skill_id registered when skills load
            self.speak_dialog("no.media.skills")
            return

        self.speak_dialog("just.one.moment")

        lang = message.data["lang"]
        query = message.data["query"]
        media_type = message.data["media_type"]
        skills = message.data.get("skills", [])
        sess = SessionManager.get(message)

        # search common play skills
        lang = standardize_lang_tag(lang)
        results = self._search(query, media_type, lang,
                               skills=skills, message=message)

        # tell OCP to play
        self.bus.emit(message.forward('ovos.common_play.reset'))
        if not results:
            self.speak_dialog("cant.play",
                              data={"phrase": query,
                                    "media_type": media_type})
        else:
            LOG.debug(f"Playing {len(results)} results for: {query}")
            best = self.select_best(results, message)
            if best is None:
                self.speak_dialog("cant.play",
                                  data={"phrase": query,
                                        "media_type": media_type})
                return
            LOG.debug(f"OCP Best match: {best}")
            results = [r for r in results if r.as_dict != best.as_dict]
            results.insert(0, best)
            self.set_context("Playing", origin=OCP_ID)

            # ovos-PHAL-plugin-mk1 will display music icon in response to play message
            player = self.get_player(message)
            player.skill_id = best.skill_id
            player.player_state = PlayerState.PLAYING
            player.media_type = best.media_type
            self.update_player_proxy(player)
            # add active skill to session
            sess.activate_skill(best.skill_id)
            message.context["session"] = sess.serialize()
            if not player.ocp_available:
                self.legacy_play(results, query, message=message)
            else:
                self.ocp_api.play(tracks=[best], utterance=query, source_message=message)
            self.ocp_api.populate_search_results(tracks=results,
                                                 replace=True,
                                                 sort_by_conf=False,  # already sorted
                                                 source_message=message)

    def handle_open_intent(self, message: Message):
        LOG.info("Requesting OCP homescreen")
        # let ovos-media handle it
        self.bus.emit(message.forward('ovos.common_play.home'))

    def handle_like_intent(self, message: Message):
        LOG.info("Requesting OCP to like current song")
        # let ovos-media handle it
        self.bus.emit(message.forward("ovos.common_play.like"))

    def handle_stop_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to stop")
            self.legacy_api.stop(source_message=message)
        else:
            LOG.info("Requesting OCP to stop")
            self.ocp_api.stop(source_message=message)
        player = self.get_player(message)
        player.player_state = PlayerState.STOPPED
        player.skill_id = None
        self.update_player_proxy(player)

    def handle_next_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to go to next track")
            self.legacy_api.next(source_message=message)
        else:
            LOG.info("Requesting OCP to go to next track")
            self.ocp_api.next(source_message=message)

    def handle_prev_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to go to prev track")
            self.legacy_api.prev(source_message=message)
        else:
            LOG.info("Requesting OCP to go to prev track")
            self.ocp_api.prev(source_message=message)

    def handle_pause_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to pause")
            self.legacy_api.pause(source_message=message)
        else:
            LOG.info("Requesting OCP to go to pause")
            self.ocp_api.pause(source_message=message)
        player = self.get_player(message)
        player.player_state = PlayerState.PAUSED
        player = self._update_player_skill_id(player, message)
        self.update_player_proxy(player)

    def handle_resume_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to resume")
            self.legacy_api.resume(source_message=message)
        else:
            LOG.info("Requesting OCP to go to resume")
            self.ocp_api.resume(source_message=message)
        player = self.get_player(message)
        player.player_state = PlayerState.PLAYING
        player = self._update_player_skill_id(player, message)
        self.update_player_proxy(player)

    def handle_search_error_intent(self, message: Message):
        self.bus.emit(message.forward("mycroft.audio.play_sound",
                                      {"uri": "snd/error.mp3"}))
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to stop")
            self.legacy_api.stop(source_message=message)
        else:
            LOG.info("Requesting OCP to stop")
            self.ocp_api.stop(source_message=message)

    # NLP
    def voc_match_media(self, query: str, lang: str, valid_labels: Optional[List[MediaType]] = None) -> Tuple[MediaType, float]:
        lang = standardize_lang_tag(lang)
        valid_labels = valid_labels or [m for m, s in self.media2skill.items() if s] or list(MediaType)
        # simplistic approach via voc_match, works anywhere
        # and it's easy to localize, but isn't very accurate
        if MediaType.DOCUMENTARY in valid_labels and self.voc_match(query, "DocumentaryKeyword", lang=lang):
            return MediaType.DOCUMENTARY, 0.6
        elif MediaType.AUDIOBOOK in valid_labels and self.voc_match(query, "AudioBookKeyword", lang=lang):
            return MediaType.AUDIOBOOK, 0.6
        elif MediaType.NEWS in valid_labels and self.voc_match(query, "NewsKeyword", lang=lang):
            return MediaType.NEWS, 0.6
        elif MediaType.ANIME in valid_labels and  self.voc_match(query, "AnimeKeyword", lang=lang):
            return MediaType.ANIME, 0.6
        elif MediaType.CARTOON in valid_labels and self.voc_match(query, "CartoonKeyword", lang=lang):
            return MediaType.CARTOON, 0.6
        elif MediaType.PODCAST in valid_labels and self.voc_match(query, "PodcastKeyword", lang=lang):
            return MediaType.PODCAST, 0.6
        elif MediaType.RADIO_THEATRE in valid_labels and self.voc_match(query, "AudioDramaKeyword", lang=lang):
            # NOTE - before "radio" to allow "radio theatre"
            return MediaType.RADIO_THEATRE, 0.6
        elif MediaType.RADIO in valid_labels and self.voc_match(query, "RadioKeyword", lang=lang):
            return MediaType.RADIO, 0.6
        elif MediaType.MUSIC in valid_labels and self.voc_match(query, "MusicKeyword", lang=lang):
            # NOTE - before movie to handle "{movie_name} soundtrack"
            return MediaType.MUSIC, 0.6
        elif MediaType.TV in valid_labels and self.voc_match(query, "TVKeyword", lang=lang):
            return MediaType.TV, 0.6
        elif MediaType.VIDEO_EPISODES in valid_labels and self.voc_match(query, "SeriesKeyword", lang=lang):
            return MediaType.VIDEO_EPISODES, 0.6
        elif any([s in valid_labels for s in [MediaType.MOVIE, MediaType.SHORT_FILM, MediaType.SILENT_MOVIE, MediaType.BLACK_WHITE_MOVIE]]) and \
                self.voc_match(query, "MovieKeyword", lang=lang):
            if MediaType.SHORT_FILM in valid_labels and self.voc_match(query, "ShortKeyword", lang=lang):
                return MediaType.SHORT_FILM, 0.7
            elif MediaType.SILENT_MOVIE in valid_labels and self.voc_match(query, "SilentKeyword", lang=lang):
                return MediaType.SILENT_MOVIE, 0.7
            elif MediaType.BLACK_WHITE_MOVIE in valid_labels and self.voc_match(query, "BWKeyword", lang=lang):
                return MediaType.BLACK_WHITE_MOVIE, 0.7
            return MediaType.MOVIE, 0.6
        elif MediaType.VISUAL_STORY in valid_labels and self.voc_match(query, "ComicBookKeyword", lang=lang):
            return MediaType.VISUAL_STORY, 0.4
        elif MediaType.GAME in valid_labels and self.voc_match(query, "GameKeyword", lang=lang):
            return MediaType.GAME, 0.4
        elif MediaType.AUDIO_DESCRIPTION in valid_labels and self.voc_match(query, "ADKeyword", lang=lang):
            return MediaType.AUDIO_DESCRIPTION, 0.4
        elif MediaType.ASMR in valid_labels and self.voc_match(query, "ASMRKeyword", lang=lang):
            return MediaType.ASMR, 0.4
        elif any([s in valid_labels for s in [MediaType.ADULT, MediaType.HENTAI, MediaType.ADULT_AUDIO]]) and self.voc_match(query, "AdultKeyword", lang=lang):
            if MediaType.HENTAI in valid_labels and self.voc_match(query, "CartoonKeyword", lang=lang) or \
                    self.voc_match(query, "AnimeKeyword", lang=lang) or \
                    self.voc_match(query, "HentaiKeyword", lang=lang):
                return MediaType.HENTAI, 0.4
            elif MediaType.ADULT_AUDIO in valid_labels and  self.voc_match(query, "AudioKeyword", lang=lang) or \
                    self.voc_match(query, "ASMRKeyword", lang=lang):
                return MediaType.ADULT_AUDIO, 0.4
            return MediaType.ADULT, 0.4
        elif MediaType.HENTAI in valid_labels and self.voc_match(query, "HentaiKeyword", lang=lang):
            return MediaType.HENTAI, 0.4
        elif MediaType.VIDEO in valid_labels and self.voc_match(query, "VideoKeyword", lang=lang):
            return MediaType.VIDEO, 0.4
        elif MediaType.AUDIO in valid_labels and self.voc_match(query, "AudioKeyword", lang=lang):
            return MediaType.AUDIO, 0.4
        return MediaType.GENERIC, 0.0

    def classify_media(self, query: str, lang: str, valid_labels: Optional[List[MediaType]] = None) -> Tuple[MediaType, float]:
        """ determine what media type is being requested """
        lang = standardize_lang_tag(lang)
        valid_labels = valid_labels or [m for m, s in self.media2skill.items() if s] or list(MediaType)
        LOG.debug(f"valid media types: {valid_labels}")
        if len(valid_labels) == 1:
            return valid_labels[0], 1.0

        return self.voc_match_media(query, lang, valid_labels)

    def is_ocp_query(self, query: str, lang: str) -> Tuple[bool, float]:
        """ determine if a playback question is being asked"""
        lang = standardize_lang_tag(lang)
        m, p = self.voc_match_media(query, lang)
        return m != MediaType.GENERIC, p

    def _should_resume(self, phrase: str, lang: str, message: Optional[Message] = None) -> bool:
        """
        Check if a "play" request should resume playback or be handled as a new
        session.
        @param phrase: Extracted playback phrase
        @return: True if player should resume, False if this is a new request
        """
        lang = standardize_lang_tag(lang)
        player = self.get_player(message)
        if player.player_state == PlayerState.PAUSED:
            if not phrase.strip() or \
                    self.voc_match(phrase, "Resume", lang=lang, exact=True) or \
                    self.voc_match(phrase, "Play", lang=lang, exact=True):
                return True
        return False

    # search
    def _player_sync(self, player: OCPPlayerProxy, message: Optional[Message] = None, timeout=1) -> OCPPlayerProxy:

        if not self.config.get("legacy"):  # force legacy audio in config
            ev = threading.Event()

            def handle_m(m):
                nonlocal player
                s = SessionManager.get(m)
                if s.session_id == player.session_id:
                    player.available_extractors = m.data["SEI"]
                    player.ocp_available = True
                    self.update_player_proxy(player)
                    ev.set()
                    LOG.debug(f"Session: {player.session_id} Available stream extractor plugins: {m.data['SEI']}")

            self.bus.on("ovos.common_play.SEI.get.response", handle_m)
            message = message or dig_for_message() or Message("")  # get message.context to forward
            self.bus.emit(message.forward("ovos.common_play.SEI.get"))
            ev.wait(timeout)
            self.bus.remove("ovos.common_play.SEI.get.response", handle_m)

            if not ev.is_set():
                LOG.warning(f"Player synchronization timed out after {timeout} seconds")

        return player

    def get_player(self, message: Optional[Message] = None, timeout=1) -> OCPPlayerProxy:
        """get a PlayerProxy object, containing info such as player state and the available stream extractors from OCP
        this is tracked per Session, if needed requests the info from the client"""
        sess = SessionManager.get(message)
        if sess.session_id not in self.ocp_sessions:
            player = OCPPlayerProxy(available_extractors=available_extractors(),
                                    ocp_available=False,
                                    session_id=sess.session_id)
            self.update_player_proxy(player)
        else:
            player = self.ocp_sessions[sess.session_id]
        if not player.ocp_available and not self.config.get("legacy"):
            # OCP might have loaded meanwhile
            player = self._player_sync(player, message, timeout)
        return player

    @staticmethod
    def _update_player_skill_id(player, message):
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if skill_id and skill_id != OCP_ID:
            player.skill_id = skill_id
        return player

    @staticmethod
    def normalize_results(results: RawResultsList) -> NormalizedResultsList:
        # support Playlist and MediaEntry objects in tracks
        for idx, track in enumerate(results):
            if isinstance(track, dict):
                try:
                    results[idx] = dict2entry(track)
                except Exception as e:
                    LOG.error(f"got an invalid track: {track}")
                    results[idx] = None
        return [r for r in results if r]

    def filter_results(self, results: list, phrase: str, lang: str,
                       media_type: MediaType = MediaType.GENERIC,
                       message: Optional[Message] = None) -> list:
        lang = standardize_lang_tag(lang)
        # ignore very low score matches
        l1 = len(results)
        results = [r for r in results
                   if r.match_confidence >= self.config.get("min_score", 50)]
        LOG.debug(f"filtered {l1 - len(results)} low confidence results")

        # filter based on MediaType
        if self.config.get("filter_media", True) and media_type != MediaType.GENERIC:
            l1 = len(results)
            # TODO - also check inside playlists
            results = [r for r in results
                       if isinstance(r, Playlist) or r.media_type == media_type]
            LOG.debug(f"filtered {l1 - len(results)} wrong MediaType results")

        # filter based on available stream extractors
        player = self.get_player(message)
        valid_starts = ["/", "http://", "https://", "file://"] + \
                       [f"{sei}//" for sei in player.available_extractors]
        if self.config.get("filter_SEI", True):
            # TODO - also check inside playlists
            bad_seis = [r for r in results if isinstance(r, MediaEntry) and
                        not any(r.uri.startswith(sei) for sei in valid_starts)]

            results = [r for r in results if r not in bad_seis]
            plugs = set([s.uri.split('//')[0] for s in bad_seis if '//' in s.uri])
            if bad_seis:
                LOG.debug(f"filtered {len(bad_seis)} results that require "
                          f"unavailable plugins: {plugs}")

        # filter by media type
        audio_only = self.voc_match(phrase, "audio_only", lang=lang)
        video_only = self.voc_match(phrase, "video_only", lang=lang)
        if self.config.get("playback_mode") == PlaybackMode.VIDEO_ONLY:
            # select only from VIDEO results if preference is set
            audio_only = True
        elif self.config.get("playback_mode") == PlaybackMode.AUDIO_ONLY:
            # select only from AUDIO results if preference is set
            video_only = True

        # check if user said "play XXX audio only"
        if audio_only or not player.ocp_available:
            l1 = len(results)
            # TODO - also check inside playlists
            results = [r for r in results
                       if (isinstance(r, Playlist) and player.ocp_available)
                       or r.playback == PlaybackType.AUDIO]
            LOG.debug(f"filtered {l1 - len(results)} non-audio results")

        # check if user said "play XXX video only"
        elif video_only:
            l1 = len(results)
            results = [r for r in results
                       if isinstance(r, Playlist) or r.playback == PlaybackType.VIDEO]
            LOG.debug(f"filtered {l1 - len(results)} non-video results")

        return results

    def _search(self, phrase: str, media_type: MediaType, lang: str,
                skills: Optional[List[str]] = None,
                message: Optional[Message] = None) -> list:
        self.bus.emit(message.reply("ovos.common_play.search.start"))
        self.enclosure.mouth_think()  # animate mk1 mouth during search

        # Now we place a query on the messsagebus for anyone who wants to
        # attempt to service a 'play.request' message.
        results = []
        for r in self._execute_query(phrase,
                                     media_type=media_type,
                                     skills=skills,
                                     message=message):
            results += r["results"]

        results = self.normalize_results(results)

        if not skills:
            LOG.debug(f"Got {len(results)} results")
            results = self.filter_results(results, phrase, lang, media_type,
                                          message=message)
            LOG.debug(f"Got {len(results)} usable results")
        else:  # no filtering if skill explicitly requested
            LOG.debug(f"Got {len(results)} usable results from {skills}")

        self.bus.emit(message.reply("ovos.common_play.search.end"))
        return results

    def _execute_query(self, phrase: str,
                       media_type: MediaType = Union[int, MediaType],
                       skills: Optional[List[str]] = None,
                       message: Optional[Message] = None) -> list:
        """ actually send the search to OCP skills"""
        media_type = self._normalize_media_enum(media_type)

        with self.search_lock:
            # stop any search still happening
            self.bus.emit(message.reply("ovos.common_play.search.stop"))

            query = OCPQuery(query=phrase, media_type=media_type,
                             config=self.config, bus=self.bus)
            # search individual skills first if user specifically asked for it
            results = []
            if skills:
                for skill_id in skills:
                    if skill_id not in self.media2skill[media_type]:
                        LOG.debug(f"{skill_id} can't handle {media_type} queries")
                        continue
                    LOG.debug(f"Searching OCP Skill: {skill_id}")
                    query.send(skill_id, source_message=message)
                    query.wait()
                    results += query.results

            if not len(self.media2skill[media_type]):
                LOG.info(f"No skills available to handle {media_type} queries, "
                         f"forcing MediaType.GENERIC")
                media_type = MediaType.GENERIC

            # search all skills
            if not results:
                if skills:
                    LOG.info(f"No specific skill results from {skills}, "
                             f"performing global OCP search")
                query.reset()
                query.send()
                query.wait()
                results = query.results

            # fallback to generic search type
            if not results and \
                    self.config.get("search_fallback", True) and \
                    media_type != MediaType.GENERIC:
                LOG.debug("OVOSCommonPlay falling back to MediaType.GENERIC")
                query.media_type = MediaType.GENERIC
                query.reset()
                query.send()
                query.wait()
                results = query.results

        LOG.debug(f'Returning {len(results)} search results')
        return results

    @staticmethod
    def select_best(results: list, message: Message) -> Union[MediaEntry, Playlist, PluginStream]:

        sess = SessionManager.get(message)

        # Look at any replies that arrived before the timeout
        # Find response(s) with the highest confidence
        best = None
        ties = []

        for res in results:
            if isinstance(res, dict):
                res = dict2entry(res)
            if res.skill_id in sess.blacklisted_skills:
                LOG.debug(f"ignoring match, skill_id '{res.skill_id}' blacklisted by Session '{sess.session_id}'")
                continue
            if not best or res.match_confidence > best.match_confidence:
                best = res
                ties = [best]
            elif res.match_confidence == best.match_confidence:
                ties.append(res)

        if ties:
            # select randomly
            selected = random.choice(ties)
            # TODO: Ask user to pick between ties or do it automagically
        else:
            selected = best
        if selected:
            LOG.info(f"OVOSCommonPlay selected: {selected.skill_id} - {selected.match_confidence}")
            LOG.debug(str(selected))
        else:
            LOG.error("No valid OCP matches")
        return selected

    ##################
    # Legacy Audio subsystem API
    def legacy_play(self, results: NormalizedResultsList, phrase="",
                    message: Optional[Message] = None):
        player = self.get_player(message)
        player.media_state = MediaState.LOADING_MEDIA
        playing = False
        for idx, r in enumerate(results):
            real_uri = None
            if not (r.playback == PlaybackType.AUDIO or r.media_type in OCPQuery.cast2audio):
                # we need to filter video results
                continue
            if isinstance(r, Playlist):
                # get internal entries from the playlist
                real_uri = [e.uri for e in r.entries]
            elif isinstance(r, MediaEntry):
                real_uri = r.uri
            elif isinstance(r, PluginStream):
                # for legacy audio service we need to do stream extraction here
                LOG.debug(f"extracting uri: {r.stream}")
                # TODO - apparently it can hang here forever ???
                # happens with https://www.cbc.ca/podcasting/includes/hourlynews.xml from news skill
                try:
                    real_uri = r.extract_uri(video=False)
                except Exception as e:
                    LOG.exception(f"extraction failed: {r}")
            if not real_uri:
                continue
            if not playing:
                playing = True
                self.legacy_api.play(real_uri, utterance=phrase, source_message=message)
                player.player_state = PlayerState.PLAYING
                player.skill_id = r.skill_id
                self.update_player_proxy(player)
            else:
                self.legacy_api.queue(real_uri, source_message=message)

    def _handle_legacy_audio_stop(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            player.player_state = PlayerState.STOPPED
            player.media_state = MediaState.NO_MEDIA
            player.skill_id = None
            self.update_player_proxy(player)

    def _handle_legacy_audio_pause(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available and player.player_state == PlayerState.PLAYING:
            player.player_state = PlayerState.PAUSED
            player.media_state = MediaState.LOADED_MEDIA
            player = self._update_player_skill_id(player, message)
            self.update_player_proxy(player)

    def _handle_legacy_audio_resume(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available and player.player_state == PlayerState.PAUSED:
            player.player_state = PlayerState.PLAYING
            player.media_state = MediaState.LOADED_MEDIA
            player = self._update_player_skill_id(player, message)
            self.update_player_proxy(player)

    def _handle_legacy_audio_start(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            player.player_state = PlayerState.PLAYING
            player.media_state = MediaState.LOADED_MEDIA
            player = self._update_player_skill_id(player, message)
            self.update_player_proxy(player)

    def _handle_legacy_audio_end(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            player.player_state = PlayerState.STOPPED
            player.media_state = MediaState.END_OF_MEDIA
            player.skill_id = None
            self.update_player_proxy(player)

    @classmethod
    def _get_closest_lang(cls, lang: str) -> Optional[str]:
        if cls.intent_matchers:
            lang = standardize_lang_tag(lang)
            closest, score = closest_match(lang, list(cls.intent_matchers.keys()))
            # https://langcodes-hickford.readthedocs.io/en/sphinx/index.html#distance-values
            # 0 -> These codes represent the same language, possibly after filling in values and normalizing.
            # 1- 3 -> These codes indicate a minor regional difference.
            # 4 - 10 -> These codes indicate a significant but unproblematic regional difference.
            if score < 10:
                return closest
        return None

    def shutdown(self):
        self.default_shutdown()  # remove events registered via self.add_event

    # deprecated
    @property
    def mycroft_cps(self) -> LegacyCommonPlay:
        log_deprecation("self.mycroft_cps is deprecated, use MycroftCPSLegacyPipeline instead", "2.0.0")
        return LegacyCommonPlay(self.bus)

    @deprecated("match_fallback has been renamed match_low", "2.0.0")
    def match_fallback(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        return self.match_low(utterances, lang, message)

    @deprecated("match_legacy is deprecated! use MycroftCPSLegacyPipeline class directly instead", "2.0.0")
    def match_legacy(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """ match legacy mycroft common play skills  (must import from deprecated mycroft module)
        not recommended, legacy support only

        legacy base class at mycroft/skills/common_play_skill.py marked for removal in ovos-core 0.1.0
        """
        return MycroftCPSLegacyPipeline(self.bus, self.config).match(utterances, lang, message)


class MycroftCPSLegacyPipeline(PipelinePlugin, OVOSAbstractApplication):
    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        OVOSAbstractApplication.__init__(self, bus=bus or FakeBus(),
                                         skill_id=OCP_ID, resources_dir=f"{dirname(__file__)}")
        PipelinePlugin.__init__(self, bus, config)
        self.mycroft_cps = LegacyCommonPlay(self.bus)
        OCPPipelineMatcher.load_intent_files()
        self.add_event("ocp:legacy_cps", self.handle_legacy_cps, is_intent=True)

    ############
    # Legacy Mycroft CommonPlay skills
    def match(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """ match legacy mycroft common play skills  (must import from deprecated mycroft module)
        not recommended, legacy support only

        legacy base class at mycroft/skills/common_play_skill.py marked for removal in ovos-core 0.1.0
        """
        if not self.config.get("legacy_cps", True):
            # needs to be explicitly enabled in pipeline config
            return None

        utterance = utterances[0].lower()

        lang = OCPPipelineMatcher._get_closest_lang(lang)
        if lang is None:  # no intents registered for this lang
            return None

        match = OCPPipelineMatcher.intent_matchers[lang].calc_intent(utterance)
        if hasattr(match, "name"):  # padatious
            match = {
                "name": match.name,
                "conf": match.conf,
                "entities": match.matches
            }

        if match["name"] is None:
            return None
        if match["name"] == "play":
            LOG.info(f"Legacy Mycroft CommonPlay match: {match}")
            utterance = match["entities"].pop("query")
            return IntentHandlerMatch(match_type="ocp:legacy_cps",
                                      match_data={"query": utterance,
                                                  "conf": 0.7},
                                      skill_id=OCP_ID,
                                      utterance=utterance)

    def handle_legacy_cps(self, message: Message):
        """intent handler for legacy CPS matches"""
        utt = message.data["query"]
        res = self.mycroft_cps.search(utt, message=message)
        if res:
            best = OCPPipelineMatcher.select_best([r[0] for r in res], message)
            if best:
                callback = [r[1] for r in res if r[0].uri == best.uri][0]
                self.mycroft_cps.skill_play(skill_id=best.skill_id,
                                            callback_data=callback,
                                            phrase=utt,
                                            message=message)
                return
        self.bus.emit(message.forward("mycroft.audio.play_sound",
                                      {"uri": "snd/error.mp3"}))

    def shutdown(self):
        self.mycroft_cps.shutdown()
