import time
from typing import Tuple, Optional

from ovos_bus_client.message import Message, dig_for_message
from ovos_bus_client.util import wait_for_reply
from ovos_utils.ocp import MediaType, PlaybackType, MediaEntry


class LegacyCommonPlay:
    """ interface for mycroft common play
    1 - emit 'play:query'
    2 - gather 'play:query.response' from legacy skills
    3 - emit 'play:start' for selected skill

    legacy base class at mycroft/skills/common_play_skill.py
    marked for removal in ovos-core 0.1.0
    """

    def __init__(self, bus):
        self.bus = bus
        self.query_replies = {}
        self.query_extensions = {}
        self.waiting = False
        self.start_ts = 0
        self.bus.on("play:query.response", self.handle_cps_response)

    def skill_play(self, skill_id: str, callback_data: dict,
                   phrase: Optional[str] = "",
                   message: Optional[Message] = None):
        """tell legacy CommonPlaySkills they were selected and should handle playback"""
        message = message or Message("ocp:legacy_cps")
        self.bus.emit(message.forward(
            'play:start',
            {"skill_id": skill_id,
             "phrase": phrase,
             "callback_data": callback_data}
        ))

    def shutdown(self):
        self.bus.remove("play:query.response", self.handle_cps_response)

    @property
    def cps_status(self):
        return wait_for_reply('play:status.query',
                              reply_type="play:status.response",
                              bus=self.bus).data

    def handle_cps_response(self, message):
        """receive matches from legacy skills"""
        search_phrase = message.data["phrase"]

        if ("searching" in message.data and
                search_phrase in self.query_extensions):
            # Manage requests for time to complete searches
            skill_id = message.data["skill_id"]
            if message.data["searching"]:
                # extend the timeout by N seconds
                # IGNORED HERE, used in mycroft-playback-control skill
                if skill_id not in self.query_extensions[search_phrase]:
                    self.query_extensions[search_phrase].append(skill_id)
            else:
                # Search complete, don't wait on this skill any longer
                if skill_id in self.query_extensions[search_phrase]:
                    self.query_extensions[search_phrase].remove(skill_id)

        elif search_phrase in self.query_replies:
            # Collect all replies until the timeout
            self.query_replies[message.data["phrase"]].append(message.data)

    def send_query(self, phrase, message: Optional[Message] = None):
        self.query_replies[phrase] = []
        self.query_extensions[phrase] = []
        message = message or dig_for_message() or Message("")
        self.bus.emit(message.forward('play:query',{"phrase": phrase}))

    def get_results(self, phrase):
        if self.query_replies.get(phrase):
            return [self.cps2media(r) for r in self.query_replies[phrase]]
        return []

    def search(self, phrase, timeout=5, message: Optional[Message] = None):
        self.send_query(phrase, message)
        self.waiting = True
        start_ts = time.time()
        while self.waiting and time.time() - start_ts <= timeout:
            time.sleep(0.2)
        self.waiting = False
        return self.get_results(phrase)

    @staticmethod
    def cps2media(res: dict, media_type=MediaType.GENERIC) -> Tuple[MediaEntry, dict]:
        """convert a cps result into a modern result"""
        entry = MediaEntry(title=res["phrase"],
                           artist=res["skill_id"],
                           uri=f"callback:{res['skill_id']}",
                           media_type=media_type,
                           playback=PlaybackType.SKILL,
                           match_confidence=res["conf"] * 100,
                           skill_id=res["skill_id"])
        return entry, res['callback_data']
