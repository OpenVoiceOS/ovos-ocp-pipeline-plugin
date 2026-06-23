"""Seam between in-process ``MediaProvider`` plugins and the OCP pipeline.

``MediaProvider`` plugins (``opm.media.provider``) speak the *mediavocab*
catalog model: they consume a :class:`mediavocab.Signals` query and return
:class:`mediavocab.Release` candidates. The pipeline is now *mediavocab-native*
too — it classifies and routes on :class:`mediavocab.MediaType` directly, so
there is no media-type taxonomy translation here anymore.

The only thing this module still maps is the **playback backend selector**:

* ``mediavocab.taxonomy.PlaybackType`` — *routing* (audio/video/paged/
  interactive), describes how a work is consumed. This is part of the catalog
  model and is mediavocab's own taxonomy.
* ``ovos_utils.ocp.PlaybackType`` — *backend selector* (AUDIO/VIDEO/SKILL/
  WEBVIEW/…), tells OCP which player backend to hand the track to. This is part
  of the ``MediaEntry`` container structure, not the media-type taxonomy, so it
  stays and is derived from the routing playback type.
"""
from typing import Optional

from ovos_utils.ocp import PlaybackType as OCPPlaybackType

import mediavocab
from mediavocab import MediaType
from mediavocab import Release, Signals
from mediavocab.taxonomy import PlaybackType as MVPlaybackType


# mediavocab routing taxonomy -> ovos_utils.ocp backend selector.
# This is NOT a media-type bridge: it maps mediavocab's playback routing onto
# the OCP MediaEntry backend selector (which player to hand the track to).
_MV_PLAYBACK_TO_OCP = {
    MVPlaybackType.AUDIO: OCPPlaybackType.AUDIO,
    MVPlaybackType.VIDEO: OCPPlaybackType.VIDEO,
    MVPlaybackType.INTERACTIVE: OCPPlaybackType.SKILL,
    MVPlaybackType.PAGED: OCPPlaybackType.WEBVIEW,
    MVPlaybackType.UNKNOWN: OCPPlaybackType.UNDEFINED,
}


def mediavocab_playback_to_ocp(pb: MVPlaybackType) -> OCPPlaybackType:
    """Map a ``mediavocab.taxonomy.PlaybackType`` (routing) to an
    ``ovos_utils.ocp.PlaybackType`` (``MediaEntry`` backend selector)."""
    return _MV_PLAYBACK_TO_OCP.get(pb, OCPPlaybackType.UNDEFINED)


def media_type_to_signals(media_type: MediaType, query: str,
                          artist: Optional[str] = None) -> Signals:
    """Build a query-role :class:`mediavocab.Signals` from the pipeline's
    classified :class:`mediavocab.MediaType` and free-text query.

    ``GENERIC`` (and the ``NOT_MEDIA``/``CONTROL`` sentinels) stay typeless so
    providers self-select via their own three-axis routing gate.
    """
    kwargs = {"title": query or None}
    if artist:
        kwargs["artist"] = artist
    if media_type not in (None, MediaType.GENERIC,
                          MediaType.NOT_MEDIA, MediaType.CONTROL):
        kwargs["medium"] = media_type
        kwargs["playback_type"] = mediavocab.infer_playback_type(media_type)
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

    The result's ``media_type`` is the release's :class:`mediavocab.MediaType`
    directly (the pipeline is mediavocab-native). Only ``playback`` is derived,
    mapping mediavocab's routing onto the OCP ``MediaEntry`` backend selector.

    @param release: the catalog candidate returned by a provider.
    @param provider_id: registry name of the provider; becomes ``skill_id``.
    @return: a dict in the ``ovos_utils.ocp`` ``MediaEntry`` shape, carrying a
             ``mediavocab.MediaType`` in ``media_type``.
    """
    work = release.work
    media_type = work.media_type
    playback = mediavocab_playback_to_ocp(mediavocab.infer_playback_type(media_type))

    # match_confidence: mediavocab is 0.0..1.0 float, OCP MediaEntry is 0..100 int
    conf = release.match_confidence or 0.0
    match_conf = int(round(max(0.0, min(1.0, conf)) * 100))

    result = {
        "uri": release.uri,
        "title": work.title,
        "image": release.image,
        "artist": _first_credit_name(release),
        "length": work.runtime or 0,
        "media_type": media_type,
        "playback": playback,
        "match_confidence": match_conf,
        "skill_id": provider_id,
    }
    return result
