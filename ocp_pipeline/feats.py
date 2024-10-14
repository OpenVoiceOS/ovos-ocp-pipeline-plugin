import os
from os.path import dirname

from ovos_utils import classproperty
from ovos_utils.log import LOG
from ovos_utils.ocp import MediaType


class OCPFeaturizer:
    """used by the experimental media type classifier,
    API should be considered unstable"""
    ocp_keywords = None
    # defined at training time
    _clf_labels = ['ad_keyword', 'album_name', 'anime_genre', 'anime_name', 'anime_streaming_service',
                   'artist_name', 'asmr_keyword', 'asmr_trigger', 'audio_genre', 'audiobook_narrator',
                   'audiobook_streaming_service', 'book_author', 'book_genre', 'book_name',
                   'bw_movie_name', 'cartoon_genre', 'cartoon_name', 'cartoon_streaming_service',
                   'comic_name', 'comic_streaming_service', 'comics_genre', 'country_name',
                   'documentary_genre', 'documentary_name', 'documentary_streaming_service',
                   'film_genre', 'film_studio', 'game_genre', 'game_name', 'gaming_console_name',
                   'generic_streaming_service', 'hentai_name', 'hentai_streaming_service',
                   'media_type_adult', 'media_type_adult_audio', 'media_type_anime', 'media_type_audio',
                   'media_type_audiobook', 'media_type_bts', 'media_type_bw_movie', 'media_type_cartoon',
                   'media_type_documentary', 'media_type_game', 'media_type_hentai', 'media_type_movie',
                   'media_type_music', 'media_type_news', 'media_type_podcast', 'media_type_radio',
                   'media_type_radio_theatre', 'media_type_short_film', 'media_type_silent_movie',
                   'media_type_sound', 'media_type_trailer', 'media_type_tv', 'media_type_video',
                   'media_type_video_episodes', 'media_type_visual_story', 'movie_actor',
                   'movie_director', 'movie_name', 'movie_streaming_service', 'music_genre',
                   'music_streaming_service', 'news_provider', 'news_streaming_service',
                   'play_verb_audio', 'play_verb_video', 'playback_device', 'playlist_name',
                   'podcast_genre', 'podcast_name', 'podcast_streaming_service', 'podcaster',
                   'porn_film_name', 'porn_genre', 'porn_streaming_service', 'pornstar_name',
                   'radio_drama_actor', 'radio_drama_genre', 'radio_drama_name', 'radio_program',
                   'radio_program_name', 'radio_streaming_service', 'radio_theatre_company',
                   'radio_theatre_streaming_service', 'record_label', 'series_name',
                   'short_film_name', 'shorts_streaming_service', 'silent_movie_name',
                   'song_name', 'sound_name', 'soundtrack_keyword', 'tv_channel', 'tv_genre',
                   'tv_streaming_service', 'video_genre', 'video_streaming_service', 'youtube_channel']

    def __init__(self, base_clf=None):
        self.clf_feats = None
        from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier
        from ovos_classifiers.skovos.features import ClassifierProbaVectorizer
        self.init_keyword_matcher()
        if base_clf:
            if isinstance(base_clf, str):
                clf_path = f"{dirname(__file__)}/models/{base_clf}.clf"
                assert os.path.isfile(clf_path)
                base_clf = SklearnOVOSClassifier.from_file(clf_path)
            self.clf_feats = ClassifierProbaVectorizer(base_clf)
        for l in self._clf_labels:  # no samples, just to ensure featurizer has right number of feats
            self.ocp_keywords.register_entity(l, [])

    @classmethod
    def init_keyword_matcher(cls):
        from ovos_classifiers.skovos.features import KeywordFeaturesVectorizer
        if OCPFeaturizer.ocp_keywords is None:
            # ignore_list accounts for "noise" keywords in the csv file
            OCPFeaturizer.ocp_keywords = KeywordFeaturesVectorizer(ignore_list=["play", "stop"])

    @classmethod
    def load_csv(cls, entity_csvs: list):
        for csv in entity_csvs or []:
            if not os.path.isfile(csv):
                # check for bundled files
                if os.path.isfile(f"{dirname(__file__)}/models/{csv}"):
                    csv = f"{dirname(__file__)}/models/{csv}"
                else:
                    LOG.error(f"Requested OCP entities file does not exist? {csv}")
                    continue
            OCPFeaturizer.ocp_keywords.load_entities(csv)
            LOG.info(f"Loaded OCP keywords: {csv}")

    @classproperty
    def labels(cls):
        """
        in V0 classifier using synth dataset - this is tied to the classifier model"""
        return cls._clf_labels

    @staticmethod
    def label2media(label: str) -> MediaType:
        if isinstance(label, MediaType):
            return label
        if label == "ad":
            mt = MediaType.AUDIO_DESCRIPTION
        elif label == "adult":
            mt = MediaType.ADULT
        elif label == "adult_asmr":
            mt = MediaType.ADULT_AUDIO
        elif label == "anime":
            mt = MediaType.ANIME
        elif label == "audio":
            mt = MediaType.AUDIO
        elif label == "asmr":
            mt = MediaType.ASMR
        elif label == "audiobook":
            mt = MediaType.AUDIOBOOK
        elif label == "bts":
            mt = MediaType.BEHIND_THE_SCENES
        elif label == "bw_movie":
            mt = MediaType.BLACK_WHITE_MOVIE
        elif label == "cartoon":
            mt = MediaType.CARTOON
        elif label == "comic":
            mt = MediaType.VISUAL_STORY
        elif label == "documentary":
            mt = MediaType.DOCUMENTARY
        elif label == "game":
            mt = MediaType.GAME
        elif label == "hentai":
            mt = MediaType.HENTAI
        elif label == "movie":
            mt = MediaType.MOVIE
        elif label == "music":
            mt = MediaType.MUSIC
        elif label == "news":
            mt = MediaType.NEWS
        elif label == "podcast":
            mt = MediaType.PODCAST
        elif label == "radio":
            mt = MediaType.RADIO
        elif label == "radio_drama":
            mt = MediaType.RADIO_THEATRE
        elif label == "series":
            mt = MediaType.VIDEO_EPISODES
        elif label == "short_film":
            mt = MediaType.SHORT_FILM
        elif label == "silent_movie":
            mt = MediaType.SILENT_MOVIE
        elif label == "trailer":
            mt = MediaType.TRAILER
        elif label == "tv_channel":
            mt = MediaType.TV
        elif label == "video":
            mt = MediaType.VIDEO
        else:
            LOG.error(f"bad label {label}")
            mt = MediaType.GENERIC
        return mt

    def transform(self, X):
        if self.clf_feats:
            from sklearn.pipeline import FeatureUnion
            vec = FeatureUnion([
                ("kw", self.ocp_keywords),
                ("clf", self.clf_feats)
            ])
            return vec.transform(X)
        return self.ocp_keywords.transform(X)

    @classmethod
    def extract_entities(cls, utterance) -> dict:
        return cls.ocp_keywords._transformer.wordlist.extract(utterance)
