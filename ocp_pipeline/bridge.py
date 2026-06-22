"""Bridge between in-process ``MediaProvider`` plugins and the OCP pipeline.

``MediaProvider`` plugins (``opm.media.provider``) speak the *mediavocab*
catalog model: they consume a :class:`mediavocab.Signals` query and return
:class:`mediavocab.Release` candidates. The rest of the OCP pipeline speaks the
``ovos_utils.ocp`` playback model (``MediaEntry`` dicts / ``MediaType`` /
``PlaybackType``).

This module is the seam between the two:

* :func:`media_type_to_signals` builds a ``Signals`` query from the pipeline's
  classified ``ovos_utils.ocp.MediaType`` + text query, mapping the playback
  taxonomy across to mediavocab's ``MediaType``.
* :func:`release_to_ocp_result` maps a ``Release`` back into the OCP playback
  result dict the pipeline already emits and ranks.

Two distinct ``PlaybackType`` enums are involved and are mapped explicitly:

* ``mediavocab.taxonomy.PlaybackType`` — *routing* (audio/video/paged/
  interactive), describes how a work is consumed.
* ``ovos_utils.ocp.PlaybackType`` — *backend selector* (AUDIO/VIDEO/SKILL/
  WEBVIEW/…), tells OCP which player backend to hand the track to.
"""
from typing import Optional

from ovos_utils.ocp import MediaType as OCPMediaType
from ovos_utils.ocp import PlaybackType as OCPPlaybackType

import mediavocab
from mediavocab import MediaType as MVMediaType
from mediavocab import Release, Signals
from mediavocab.taxonomy import PlaybackType as MVPlaybackType


# ovos_utils.ocp.MediaType  ->  mediavocab.MediaType
# Only the playback taxonomy the providers actually route on; anything not
# listed (or GENERIC) is left as a typeless query so providers self-select via
# their own three-axis gate.
_OCP_TO_MV_MEDIA = {
    OCPMediaType.MUSIC: MVMediaType.MUSIC,
    OCPMediaType.AUDIO: MVMediaType.MUSIC,
    OCPMediaType.PODCAST: MVMediaType.PODCAST,
    OCPMediaType.AUDIOBOOK: MVMediaType.AUDIOBOOK,
    OCPMediaType.RADIO: MVMediaType.RADIO,
    OCPMediaType.RADIO_THEATRE: MVMediaType.AUDIO_DRAMA,
    OCPMediaType.ADULT_AUDIO: MVMediaType.MUSIC,
    OCPMediaType.ASMR: MVMediaType.PROCEDURAL_AMBIENT,
    OCPMediaType.MOVIE: MVMediaType.MOVIE,
    OCPMediaType.SILENT_MOVIE: MVMediaType.MOVIE,
    OCPMediaType.BLACK_WHITE_MOVIE: MVMediaType.MOVIE,
    OCPMediaType.SHORT_FILM: MVMediaType.SHORT_FILM,
    OCPMediaType.TV: MVMediaType.TV,
    OCPMediaType.VIDEO_EPISODES: MVMediaType.EPISODIC_SERIES,
    OCPMediaType.CARTOON: MVMediaType.EPISODIC_SERIES,
    OCPMediaType.ANIME: MVMediaType.EPISODIC_SERIES,
    OCPMediaType.DOCUMENTARY: MVMediaType.MOVIE,
    OCPMediaType.VIDEO: MVMediaType.MOVIE,
    OCPMediaType.ADULT: MVMediaType.MOVIE,
    OCPMediaType.HENTAI: MVMediaType.EPISODIC_SERIES,
    OCPMediaType.GAME: MVMediaType.GAME,
    OCPMediaType.VISUAL_STORY: MVMediaType.COMIC,
    OCPMediaType.NEWS: MVMediaType.RADIO,
}

# reverse direction: mediavocab.MediaType -> ovos_utils.ocp.MediaType, used when
# a Release comes back so the pipeline can keep filtering/ranking on its own
# taxonomy. Picks the closest OCP label.
_MV_TO_OCP_MEDIA = {
    MVMediaType.MOVIE: OCPMediaType.MOVIE,
    MVMediaType.SHORT_FILM: OCPMediaType.SHORT_FILM,
    MVMediaType.EPISODIC_SERIES: OCPMediaType.VIDEO_EPISODES,
    MVMediaType.TV: OCPMediaType.TV,
    MVMediaType.MUSIC: OCPMediaType.MUSIC,
    MVMediaType.MUSIC_VIDEO: OCPMediaType.VIDEO,
    MVMediaType.PODCAST: OCPMediaType.PODCAST,
    MVMediaType.AUDIOBOOK: OCPMediaType.AUDIOBOOK,
    MVMediaType.AUDIO_DRAMA: OCPMediaType.RADIO_THEATRE,
    MVMediaType.RADIO: OCPMediaType.RADIO,
    MVMediaType.BOOK: OCPMediaType.AUDIOBOOK,
    MVMediaType.COMIC: OCPMediaType.VISUAL_STORY,
    MVMediaType.GAME: OCPMediaType.GAME,
    MVMediaType.INTERACTIVE_FICTION: OCPMediaType.GAME,
    MVMediaType.SOUND_EFFECT: OCPMediaType.AUDIO,
    MVMediaType.PROCEDURAL_AMBIENT: OCPMediaType.ASMR,
    MVMediaType.PLAYLIST: OCPMediaType.AUDIO,
    MVMediaType.GENERIC: OCPMediaType.GENERIC,
}

# mediavocab routing taxonomy -> ovos_utils.ocp backend selector
_MV_PLAYBACK_TO_OCP = {
    MVPlaybackType.AUDIO: OCPPlaybackType.AUDIO,
    MVPlaybackType.VIDEO: OCPPlaybackType.VIDEO,
    MVPlaybackType.INTERACTIVE: OCPPlaybackType.SKILL,
    MVPlaybackType.PAGED: OCPPlaybackType.WEBVIEW,
    MVPlaybackType.UNKNOWN: OCPPlaybackType.UNDEFINED,
}


def ocp_media_type_to_mediavocab(media_type: OCPMediaType) -> Optional[MVMediaType]:
    """Map an ``ovos_utils.ocp.MediaType`` to the closest ``mediavocab.MediaType``.

    Returns ``None`` for ``GENERIC`` (and anything unmapped) so the query stays
    typeless and providers self-select via their own routing gate.
    """
    if media_type == OCPMediaType.GENERIC:
        return None
    return _OCP_TO_MV_MEDIA.get(media_type)


def mediavocab_media_type_to_ocp(media_type: MVMediaType) -> OCPMediaType:
    """Map a ``mediavocab.MediaType`` back to the closest ``ovos_utils.ocp.MediaType``."""
    return _MV_TO_OCP_MEDIA.get(media_type, OCPMediaType.GENERIC)


def mediavocab_playback_to_ocp(pb: MVPlaybackType) -> OCPPlaybackType:
    """Map a ``mediavocab.taxonomy.PlaybackType`` (routing) to an
    ``ovos_utils.ocp.PlaybackType`` (backend selector)."""
    return _MV_PLAYBACK_TO_OCP.get(pb, OCPPlaybackType.UNDEFINED)


def media_type_to_signals(media_type: OCPMediaType, query: str,
                          artist: Optional[str] = None) -> Signals:
    """Build a query-role :class:`mediavocab.Signals` from the pipeline's
    classified media type and free-text query."""
    medium = ocp_media_type_to_mediavocab(media_type)
    kwargs = {"title": query or None}
    if artist:
        kwargs["artist"] = artist
    if medium is not None:
        kwargs["medium"] = medium
        kwargs["playback_type"] = mediavocab.infer_playback_type(medium)
    return Signals.as_query(**kwargs)


def _first_credit_name(release: Release) -> str:
    """Best-effort 'artist' string: name of the first work credit."""
    try:
        credits = release.work.credits or []
        if credits:
            entity = credits[0].entity
            return getattr(entity, "name", "") or ""
    except Exception:
        pass
    return ""


def release_to_ocp_result(release: Release, provider_id: str) -> dict:
    """Map a :class:`mediavocab.Release` to the OCP playback result dict the
    pipeline normalizes, ranks and plays.

    @param release: the catalog candidate returned by a provider.
    @param provider_id: registry name of the provider; becomes ``skill_id``.
    @return: a dict in the ``ovos_utils.ocp`` ``MediaEntry`` shape.
    """
    work = release.work
    mv_media = work.media_type
    ocp_media = mediavocab_media_type_to_ocp(mv_media)
    playback = mediavocab_playback_to_ocp(mediavocab.infer_playback_type(mv_media))

    # match_confidence: mediavocab is 0.0..1.0 float, OCP MediaEntry is 0..100 int
    conf = release.match_confidence or 0.0
    match_conf = int(round(max(0.0, min(1.0, conf)) * 100))

    result = {
        "uri": release.uri,
        "title": work.title,
        "image": release.image,
        "artist": _first_credit_name(release),
        "length": work.runtime or 0,
        "media_type": ocp_media,
        "playback": playback,
        "match_confidence": match_conf,
        "skill_id": provider_id,
    }
    return result
