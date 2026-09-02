"""Seam between in-process ``MediaProvider`` plugins and the legacy OCP pipeline.

``MediaProvider`` plugins (``opm.media.provider``) are the in-process
replacement for the old bus-broadcast OCP search skills. They speak
*mediavocab*: a provider consumes a :class:`mediavocab.Signals` query and
returns :class:`mediavocab.Release` candidates.

The OCP pipeline itself is **not** mediavocab-native (that classification
migration is tracked separately, see PRs #142/#143) — it still classifies and
routes on the legacy :class:`ovos_utils.ocp.MediaType` taxonomy. This module is
therefore a best-effort two-way bridge between the two taxonomies, used only by
the in-process provider dispatch path (:mod:`ocp_pipeline.opm`'s
``_search_providers``); it does not touch ``classify_media``/``voc_match_media``.

``mediavocab`` and the ``MediaProvider`` plugin type are required dependencies
of this module.
"""
from typing import Optional

from ovos_utils.ocp import MediaType as OCPMediaType
from ovos_utils.ocp import PlaybackType as OCPPlaybackType
from ovos_utils.parse import fuzzy_match

import mediavocab
from mediavocab import MediaType as MVMediaType
from mediavocab import Release, Signals
from mediavocab.taxonomy import PlaybackType as MVPlaybackType

# ovos_utils.ocp.MediaType -> mediavocab.MediaType, best-effort. mediavocab
# is a coarser/cleaner taxonomy, so several legacy buckets fold onto a
# single mediavocab type. AUDIO/VIDEO (generic, untyped) have no good
# target and stay unmapped (None) -- the Signals stays typeless so
# providers self-select via their own routing.
_OCP_TO_MV = {
    OCPMediaType.MUSIC: MVMediaType.MUSIC,
    OCPMediaType.AUDIOBOOK: MVMediaType.AUDIOBOOK,
    OCPMediaType.GAME: MVMediaType.GAME,
    OCPMediaType.PODCAST: MVMediaType.PODCAST,
    OCPMediaType.RADIO: MVMediaType.RADIO,
    OCPMediaType.NEWS: MVMediaType.RADIO,
    OCPMediaType.TV: MVMediaType.TV,
    OCPMediaType.MOVIE: MVMediaType.MOVIE,
    OCPMediaType.TRAILER: MVMediaType.MOVIE,
    OCPMediaType.VISUAL_STORY: MVMediaType.COMIC,
    OCPMediaType.DOCUMENTARY: MVMediaType.MOVIE,
    OCPMediaType.RADIO_THEATRE: MVMediaType.AUDIO_DRAMA,
    OCPMediaType.SHORT_FILM: MVMediaType.SHORT_FILM,
    OCPMediaType.SILENT_MOVIE: MVMediaType.MOVIE,
    OCPMediaType.VIDEO_EPISODES: MVMediaType.EPISODIC_SERIES,
    OCPMediaType.BLACK_WHITE_MOVIE: MVMediaType.MOVIE,
    OCPMediaType.CARTOON: MVMediaType.EPISODIC_SERIES,
    OCPMediaType.ANIME: MVMediaType.EPISODIC_SERIES,
    OCPMediaType.ASMR: MVMediaType.PROCEDURAL_AMBIENT,
    OCPMediaType.ADULT: MVMediaType.MOVIE,
    OCPMediaType.HENTAI: MVMediaType.EPISODIC_SERIES,
    OCPMediaType.ADULT_AUDIO: MVMediaType.MUSIC,
}

# mediavocab.MediaType -> ovos_utils.ocp.MediaType, for folding a
# provider's Release.work.media_type back into the MediaEntry shape the
# rest of the pipeline (filter_results, select_best, ...) expects.
_MV_TO_OCP = {
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
    MVMediaType.BOOK: OCPMediaType.VISUAL_STORY,
    MVMediaType.COMIC: OCPMediaType.VISUAL_STORY,
    MVMediaType.GAME: OCPMediaType.GAME,
    MVMediaType.INTERACTIVE_FICTION: OCPMediaType.GAME,
    MVMediaType.SOUND_EFFECT: OCPMediaType.AUDIO,
    MVMediaType.PROCEDURAL_AMBIENT: OCPMediaType.ASMR,
    MVMediaType.PLAYLIST: OCPMediaType.GENERIC,
    MVMediaType.GENERIC: OCPMediaType.GENERIC,
    MVMediaType.NOT_MEDIA: OCPMediaType.GENERIC,
    MVMediaType.CONTROL: OCPMediaType.GENERIC,
}

# mediavocab routing taxonomy -> ovos_utils.ocp backend selector. This is
# NOT a media-type bridge: it maps mediavocab's playback routing
# (audio/video/paged/interactive) onto the OCP MediaEntry backend selector
# (which player to hand the track to).
_MV_PLAYBACK_TO_OCP = {
    MVPlaybackType.AUDIO: OCPPlaybackType.AUDIO,
    MVPlaybackType.VIDEO: OCPPlaybackType.VIDEO,
    MVPlaybackType.INTERACTIVE: OCPPlaybackType.SKILL,
    MVPlaybackType.PAGED: OCPPlaybackType.WEBVIEW,
    MVPlaybackType.UNKNOWN: OCPPlaybackType.UNDEFINED,
}

def ocp_media_type_to_mediavocab(media_type: OCPMediaType) -> Optional["MVMediaType"]:
    """Best-effort map of a legacy ``ovos_utils.ocp.MediaType`` onto a
    ``mediavocab.MediaType``, for building the provider query ``Signals``.

    Returns ``None`` for ``GENERIC``/untyped/unmapped types -- the
    ``Signals`` stays typeless so providers self-select via their own
    routing, exactly like an unclassified bus query would.
    """
    return _OCP_TO_MV.get(media_type)

def mediavocab_media_type_to_ocp(media_type) -> OCPMediaType:
    """Best-effort map of a ``mediavocab.MediaType`` (a Release's media
    type) back onto the legacy ``ovos_utils.ocp.MediaType`` taxonomy, so a
    provider result fits the ``MediaEntry`` shape the rest of the pipeline
    expects."""
    return _MV_TO_OCP.get(media_type, OCPMediaType.GENERIC)

def mediavocab_playback_to_ocp(pb: "MVPlaybackType") -> OCPPlaybackType:
    """Map a ``mediavocab.taxonomy.PlaybackType`` (routing) to an
    ``ovos_utils.ocp.PlaybackType`` (``MediaEntry`` backend selector)."""
    return _MV_PLAYBACK_TO_OCP.get(pb, OCPPlaybackType.UNDEFINED)

def media_type_to_signals(media_type: OCPMediaType, query: str,
                          artist: Optional[str] = None) -> "Signals":
    """Build a query-role :class:`mediavocab.Signals` from the pipeline's
    classified legacy :class:`ovos_utils.ocp.MediaType` and free-text
    query.

    ``GENERIC`` (and anything with no mediavocab equivalent) stays
    typeless so providers self-select via their own routing.
    """
    kwargs = {"title": query or None}
    if artist:
        kwargs["artist"] = artist
    mv_type = ocp_media_type_to_mediavocab(media_type)
    if mv_type is not None:
        kwargs["medium"] = mv_type
        kwargs["playback_type"] = mediavocab.infer_playback_type(mv_type)
    return Signals.as_query(**kwargs)

def _first_credit_name(release: "Release") -> str:
    """Best-effort 'artist' string: name of the first work credit."""
    try:
        credits = release.work.credits or []
        if credits:
            entity = credits[0].entity
            return getattr(entity, "name", "") or ""
    except Exception:
        pass
    return ""

def fallback_confidence(release: "Release",
                        signals: Optional["Signals"] = None) -> float:
    """Score a ``Release`` that a provider did not score itself.

    Providers **should** set ``Release.match_confidence`` themselves: only
    the provider knows how well its catalog hit matches the request. When
    it does not (unset or ``0.0``), forwarding ``0`` would be fatal --
    :meth:`OCPPipelineMatcher.filter_results` drops everything below
    ``min_score`` (50 by default), so an unscored provider result could
    never be played. This computes a best-effort fuzzy score of the
    release title against the query signals instead, using the same
    ``ovos_utils.parse.fuzzy_match`` utility the OVOS ecosystem ranks
    with. The artist is folded in only when the query asked for one.

    @param release: the unscored candidate.
    @param signals: the query ``Signals`` the candidate was fetched for.
    @return: a 0.0..1.0 confidence.
    """
    if signals is None:
        return 0.0
    title = (getattr(release.work, "title", "") or "").strip()
    q_title = (getattr(signals, "title", "") or "").strip()
    if not title or not q_title:
        return 0.0
    score = fuzzy_match(title.lower(), q_title.lower())
    q_artist = (getattr(signals, "artist", "") or "").strip()
    if q_artist:
        artist = (_first_credit_name(release) or "").strip()
        a_score = fuzzy_match(artist.lower(), q_artist.lower()) if artist else 0.0
        score = (score + a_score) / 2
    return max(0.0, min(1.0, score))

def release_to_ocp_result(release: "Release", provider_id: str,
                          media_type: Optional[OCPMediaType] = None,
                          signals: Optional["Signals"] = None) -> dict:
    """Map a :class:`mediavocab.Release` to the OCP playback result dict
    the pipeline normalizes, ranks and plays.

    ``media_type`` is folded back to the legacy
    :class:`ovos_utils.ocp.MediaType` the rest of the pipeline (e.g.
    :meth:`OCPPipelineMatcher.filter_results`) still expects. Only
    ``playback`` is otherwise derived, mapping mediavocab's routing onto
    the OCP ``MediaEntry`` backend selector.

    @param release: the catalog candidate returned by a provider.
    @param provider_id: registry name of the provider; becomes ``skill_id``.
    @param media_type: legacy media type to stamp on the result. Callers
        pass the media type of the QUERY, because the result was fetched
        for that query: the mediavocab fold is not injective (legacy NEWS
        and RADIO both map to mediavocab RADIO, which folds back to
        RADIO), so folding the Release's own type back would make
        ``filter_results`` drop the result for a NEWS query. When
        ``None``, the Release's own type is folded back.
    @param signals: the query signals, used only to score a Release the
        provider left unscored (see :func:`fallback_confidence`).
    @return: a dict in the ``ovos_utils.ocp`` ``MediaEntry`` shape.
    """
    work = release.work
    mv_media_type = work.media_type
    if media_type is None:
        media_type = mediavocab_media_type_to_ocp(mv_media_type)
    playback = mediavocab_playback_to_ocp(mediavocab.infer_playback_type(mv_media_type))

    # match_confidence: mediavocab is 0.0..1.0 float, OCP MediaEntry is 0..100 int
    conf = release.match_confidence or 0.0
    if not conf:
        conf = fallback_confidence(release, signals)
    match_conf = int(round(max(0.0, min(1.0, conf)) * 100))

    return {
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
