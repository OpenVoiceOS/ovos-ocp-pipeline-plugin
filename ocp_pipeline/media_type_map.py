"""Translate between the standalone classifier's vocabulary and the legacy enum.

``ovos-media-classifier`` speaks ``mediavocab`` (a str-enum ``MediaType`` plus a
set of orthogonal axes — ``ProgrammeFormat`` / ``ContentForm`` / ``PictureFormat``
/ ``AccessibilityKind`` / genres).  This pipeline still emits the legacy
``ovos_utils.ocp.MediaType`` (an ``IntEnum``) downstream (``media2skill`` routing,
the ``ocp:play`` match data).  The legacy enum is *flatter*: several of its
members are really an axis folded onto a leaf — e.g. ``DOCUMENTARY`` is a
``MOVIE`` with ``ProgrammeFormat.DOCUMENTARY``, ``TRAILER`` is a ``MOVIE`` with
``ContentForm.TRAILER``, ``SILENT_MOVIE`` is a ``MOVIE`` with
``PictureFormat.SILENT``.

:func:`mv_to_legacy` re-folds those axes back onto the leaf so the pipeline keeps
emitting exactly the legacy ``MediaType`` set the old native ``voc_match_media``
chain produced.  :func:`legacy_to_mv` is the inverse used to translate caller
``valid_labels`` (legacy enum) into the ``mediavocab`` labels the classifier
gates on.

The pipeline adapts to the classifier's contract here — the classifier is never
bent toward the pipeline.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ovos_utils.ocp import MediaType as LegacyMediaType

from mediavocab import MediaType as MVMediaType
from mediavocab.taxonomy import (
    AccessibilityKind,
    ContentForm,
    PictureFormat,
    ProgrammeFormat,
)

# ---------------------------------------------------------------------------
# leaf mediavocab.MediaType -> legacy ovos_utils.ocp.MediaType
#
# The names that align map 1:1; the diverging mediavocab leaves map to the
# closest legacy member so downstream ``media2skill`` routing keeps working.
# The genre / programme-format / content-form / picture-format / accessibility
# axes are folded on *afterwards* by ``mv_to_legacy`` (they refine this leaf).
# ---------------------------------------------------------------------------
_MV_LEAF_TO_LEGACY = {
    MVMediaType.MUSIC: LegacyMediaType.MUSIC,
    MVMediaType.MOVIE: LegacyMediaType.MOVIE,
    MVMediaType.TV: LegacyMediaType.TV,
    MVMediaType.PODCAST: LegacyMediaType.PODCAST,
    MVMediaType.RADIO: LegacyMediaType.RADIO,
    MVMediaType.AUDIOBOOK: LegacyMediaType.AUDIOBOOK,
    MVMediaType.GAME: LegacyMediaType.GAME,
    MVMediaType.SHORT_FILM: LegacyMediaType.SHORT_FILM,
    MVMediaType.GENERIC: LegacyMediaType.GENERIC,
    # diverging leaves -> closest legacy member
    MVMediaType.EPISODIC_SERIES: LegacyMediaType.VIDEO_EPISODES,
    MVMediaType.AUDIO_DRAMA: LegacyMediaType.RADIO_THEATRE,
    MVMediaType.COMIC: LegacyMediaType.VISUAL_STORY,
    MVMediaType.PROCEDURAL_AMBIENT: LegacyMediaType.ASMR,
    MVMediaType.MUSIC_VIDEO: LegacyMediaType.VIDEO,
    MVMediaType.PLAYLIST: LegacyMediaType.MUSIC,
    MVMediaType.BOOK: LegacyMediaType.AUDIOBOOK,
    MVMediaType.SOUND_EFFECT: LegacyMediaType.AUDIO,
    MVMediaType.INTERACTIVE_FICTION: LegacyMediaType.GAME,
    # sentinels -> no media leaf
    MVMediaType.NOT_MEDIA: LegacyMediaType.GENERIC,
    MVMediaType.CONTROL: LegacyMediaType.GENERIC,
}

# ---------------------------------------------------------------------------
# legacy ovos_utils.ocp.MediaType -> mediavocab.MediaType (for ``valid_labels``).
#
# Covers every legacy member that can be a ``media2skill`` key.  The legacy-only
# axis members (DOCUMENTARY / NEWS / TRAILER / SILENT_MOVIE / ... / ANIME /
# CARTOON / ADULT / HENTAI) translate to their carrier leaf so the classifier's
# ``valid_labels`` gate still admits the query that produces them.
# ---------------------------------------------------------------------------
_LEGACY_TO_MV = {
    LegacyMediaType.GENERIC: MVMediaType.GENERIC,
    LegacyMediaType.AUDIO: MVMediaType.SOUND_EFFECT,
    LegacyMediaType.MUSIC: MVMediaType.MUSIC,
    LegacyMediaType.VIDEO: MVMediaType.MUSIC_VIDEO,
    LegacyMediaType.AUDIOBOOK: MVMediaType.AUDIOBOOK,
    LegacyMediaType.GAME: MVMediaType.GAME,
    LegacyMediaType.PODCAST: MVMediaType.PODCAST,
    LegacyMediaType.RADIO: MVMediaType.RADIO,
    LegacyMediaType.NEWS: MVMediaType.RADIO,
    LegacyMediaType.TV: MVMediaType.TV,
    LegacyMediaType.MOVIE: MVMediaType.MOVIE,
    LegacyMediaType.TRAILER: MVMediaType.MOVIE,
    LegacyMediaType.AUDIO_DESCRIPTION: MVMediaType.MOVIE,
    LegacyMediaType.VISUAL_STORY: MVMediaType.COMIC,
    LegacyMediaType.BEHIND_THE_SCENES: MVMediaType.MOVIE,
    LegacyMediaType.DOCUMENTARY: MVMediaType.MOVIE,
    LegacyMediaType.RADIO_THEATRE: MVMediaType.AUDIO_DRAMA,
    LegacyMediaType.SHORT_FILM: MVMediaType.SHORT_FILM,
    LegacyMediaType.SILENT_MOVIE: MVMediaType.MOVIE,
    LegacyMediaType.VIDEO_EPISODES: MVMediaType.EPISODIC_SERIES,
    LegacyMediaType.BLACK_WHITE_MOVIE: MVMediaType.MOVIE,
    LegacyMediaType.CARTOON: MVMediaType.EPISODIC_SERIES,
    LegacyMediaType.ANIME: MVMediaType.EPISODIC_SERIES,
    LegacyMediaType.ASMR: MVMediaType.PROCEDURAL_AMBIENT,
    LegacyMediaType.ADULT: MVMediaType.MOVIE,
    LegacyMediaType.HENTAI: MVMediaType.EPISODIC_SERIES,
    LegacyMediaType.ADULT_AUDIO: MVMediaType.MUSIC,
}


def legacy_to_mv(mt: Optional[LegacyMediaType]) -> Optional[MVMediaType]:
    """Map a legacy ``ovos_utils.ocp.MediaType`` to a ``mediavocab.MediaType``."""
    if mt is None:
        return None
    return _LEGACY_TO_MV.get(mt, MVMediaType.GENERIC)


# ---------------------------------------------------------------------------
# leaf mediavocab.MediaType -> the broad legacy VIDEO/AUDIO bucket it belongs
# to.  This is the coarsest fallback: when neither the axis-refined member nor
# the plain leaf mapping is a registered ``media2skill`` label, a skill that
# only declared the legacy VIDEO or AUDIO bucket should still be reachable.
# ---------------------------------------------------------------------------
_LEGACY_PARENT = {
    # ``EPISODIC_SERIES`` folds straight to ``VIDEO_EPISODES`` (its closest
    # single legacy member — see ``_MV_LEAF_TO_LEGACY``), but ``VIDEO_EPISODES``
    # is itself a refinement of the general ``TV`` bucket in the legacy
    # taxonomy, same as ``TRAILER`` refines ``MOVIE``. A skill that only
    # registered ``TV`` must stay reachable for "watch tv show ...".
    LegacyMediaType.VIDEO_EPISODES: LegacyMediaType.TV,
    LegacyMediaType.CARTOON: LegacyMediaType.TV,
    LegacyMediaType.ANIME: LegacyMediaType.TV,
    LegacyMediaType.HENTAI: LegacyMediaType.TV,
}

_MV_LEAF_TO_BROAD = {
    MVMediaType.MOVIE: LegacyMediaType.VIDEO,
    MVMediaType.TV: LegacyMediaType.VIDEO,
    MVMediaType.EPISODIC_SERIES: LegacyMediaType.VIDEO,
    MVMediaType.SHORT_FILM: LegacyMediaType.VIDEO,
    MVMediaType.MUSIC_VIDEO: LegacyMediaType.VIDEO,
    MVMediaType.COMIC: LegacyMediaType.VIDEO,
    MVMediaType.MUSIC: LegacyMediaType.AUDIO,
    MVMediaType.PLAYLIST: LegacyMediaType.AUDIO,
    MVMediaType.PODCAST: LegacyMediaType.AUDIO,
    MVMediaType.RADIO: LegacyMediaType.AUDIO,
    MVMediaType.AUDIOBOOK: LegacyMediaType.AUDIO,
    MVMediaType.BOOK: LegacyMediaType.AUDIO,
    MVMediaType.AUDIO_DRAMA: LegacyMediaType.AUDIO,
    MVMediaType.SOUND_EFFECT: LegacyMediaType.AUDIO,
    MVMediaType.PROCEDURAL_AMBIENT: LegacyMediaType.AUDIO,
}


def mv_to_legacy_candidates(
    classification,
    content_form: Optional[ContentForm] = None,
    programme_format: Optional[ProgrammeFormat] = None,
    picture_format: Optional[Sequence[PictureFormat]] = None,
    accessibility: Optional[Sequence[AccessibilityKind]] = None,
) -> List[LegacyMediaType]:
    """Preference-ordered legacy ``MediaType`` candidates for a classification.

    Mirrors the fall-through the old native ``voc_match_media`` cascade did:
    the most specific axis-refined member (e.g. ``TRAILER``) is tried first,
    then its axis-blind parent leaf (``MOVIE``), then the broadest legacy
    bucket the leaf belongs to (``VIDEO``/``AUDIO``).  Each entry's parent is
    simply the fold of the same ``mediavocab`` leaf with one less axis
    applied, so the chain is derived from :func:`mv_to_legacy`'s own fold
    structure rather than hardcoded per-member.

    De-duplicated, order preserved.  Callers pick the first candidate that is
    a registered ``valid_labels`` member.
    """
    refined = mv_to_legacy(
        classification,
        content_form=content_form,
        programme_format=programme_format,
        picture_format=picture_format,
        accessibility=accessibility,
    )
    leaf = _MV_LEAF_TO_LEGACY.get(classification.media_type, LegacyMediaType.GENERIC)
    parent = _LEGACY_PARENT.get(leaf)
    broad = _MV_LEAF_TO_BROAD.get(classification.media_type)

    out: List[LegacyMediaType] = []
    for candidate in (refined, leaf, parent, broad):
        if candidate is not None and candidate not in out:
            out.append(candidate)
    return out


def mv_to_legacy(
    classification,
    content_form: Optional[ContentForm] = None,
    programme_format: Optional[ProgrammeFormat] = None,
    picture_format: Optional[Sequence[PictureFormat]] = None,
    accessibility: Optional[Sequence[AccessibilityKind]] = None,
) -> LegacyMediaType:
    """Fold a classifier result + its axes back onto the legacy ``MediaType``.

    ``classification`` is the
    :class:`~ovos_media_classifier.axes.MediaClassification` (carrying the leaf
    ``media_type`` and ``genres``); the remaining arguments are the orthogonal
    axes the classifier predicts on separate heads (the legacy enum folds them
    into dedicated members, so they must be re-applied here).

    Precedence mirrors the old native ``voc_match_media`` chain: the most
    specific axis-derived member wins, falling through to the plain leaf mapping.
    """
    media_type = classification.media_type
    genres = set(classification.genres or [])
    picture_format = list(picture_format or [])
    accessibility = list(accessibility or [])

    # --- 1. adult family (genre-driven, most specific) --------------------
    # adult + anime  -> HENTAI ; adult on an audio leaf -> ADULT_AUDIO ; else ADULT
    if "adult" in genres:
        if "anime" in genres:
            return LegacyMediaType.HENTAI
        if media_type in (MVMediaType.MUSIC, MVMediaType.PROCEDURAL_AMBIENT,
                          MVMediaType.SOUND_EFFECT, MVMediaType.AUDIOBOOK,
                          MVMediaType.AUDIO_DRAMA, MVMediaType.PODCAST,
                          MVMediaType.RADIO):
            return LegacyMediaType.ADULT_AUDIO
        return LegacyMediaType.ADULT

    # --- 2. programme format (documentary / news) -------------------------
    if programme_format is ProgrammeFormat.DOCUMENTARY:
        return LegacyMediaType.DOCUMENTARY
    if programme_format is ProgrammeFormat.NEWS:
        return LegacyMediaType.NEWS

    # --- 3. content form (trailer) ----------------------------------------
    if content_form is ContentForm.TRAILER:
        return LegacyMediaType.TRAILER

    # --- 4. picture format (silent / black & white) -----------------------
    if PictureFormat.SILENT in picture_format:
        return LegacyMediaType.SILENT_MOVIE
    if PictureFormat.BLACK_AND_WHITE in picture_format:
        return LegacyMediaType.BLACK_WHITE_MOVIE

    # --- 5. accessibility (audio description) -----------------------------
    if AccessibilityKind.AUDIO_DESCRIPTION in accessibility:
        return LegacyMediaType.AUDIO_DESCRIPTION

    # --- 6. genre tags folded onto a leaf ---------------------------------
    if "anime" in genres:
        return LegacyMediaType.ANIME
    if "animation" in genres:
        return LegacyMediaType.CARTOON
    if "asmr" in genres:
        return LegacyMediaType.ASMR

    # --- 7. the plain leaf mapping ----------------------------------------
    return _MV_LEAF_TO_LEGACY.get(media_type, LegacyMediaType.GENERIC)
