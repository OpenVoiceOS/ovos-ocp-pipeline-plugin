"""Bridge to the standalone ``ovos-media-classifier`` context-aware contract.

The classifier is the timeless standalone product; this pipeline is a consumer.
It speaks the legacy ``ovos_utils.ocp.MediaType`` (an IntEnum) while the
classifier speaks ``mediavocab.MediaType`` (a str-enum), so this module:

* builds the classifier's two minimal context inputs from the pipeline's own
  state — :class:`~ovos_media_classifier.context.PlayerStatus` from the per
  -session now-playing proxy, and ``ner_list`` (``{label: [entity]}``) from the
  skill-registered keywords already held in the pipeline's NER;
* selects the backend from the OCP config via
  :func:`~ovos_media_classifier.load_media_classifier` (keyword by default; a
  config may opt into an ``opm.media.classifier`` plugin / ONNX / embedding
  -router bundle / online metadatarr layer);
* calls :meth:`classify_full(query, lang, player_status, ner_list)` and folds
  the ``mediavocab`` result + its orthogonal axes back onto the legacy enum via
  the single shared :mod:`ocp_pipeline.media_type_map`;
* exposes :meth:`to_signals` so the rich provider-ready ``mediavocab.Signals``
  reach the MediaProviders, and applies the
  :class:`~ovos_media_classifier.ContentFilter` so blocked content (adult by
  default; configurable genres / types) never routes.

The pipeline adapts to the classifier's contract here — the classifier is never
bent toward the pipeline.  The mapping between vocabularies lives *only* in
:mod:`ocp_pipeline.media_type_map`; this module imports it (no duplicate).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ovos_utils.log import LOG
from ovos_utils.ocp import MediaType as LegacyMediaType, PlayerState as LegacyPlayerState

from mediavocab import MediaType as MVMediaType

from ovos_media_classifier import ContentFilter, load_media_classifier
from ovos_media_classifier.context import PlayerState, PlayerStatus

# the single shared vocabulary bridge (no duplicate mapping lives here)
from ocp_pipeline.media_type_map import legacy_to_mv, mv_to_legacy, mv_to_legacy_candidates

_LEGACY_STATE_TO_CTX = {
    LegacyPlayerState.PLAYING: PlayerState.PLAYING,
    LegacyPlayerState.PAUSED: PlayerState.PAUSED,
    LegacyPlayerState.STOPPED: PlayerState.STOPPED,
}


def build_player_status(player) -> Optional[PlayerStatus]:
    """Build the classifier's :class:`PlayerStatus` from an ``OCPPlayerProxy``.

    Returns ``None`` when there is no player / nothing to derive context from.
    The proxy reports a legacy ``MediaType``; it is translated into the
    classifier's ``mediavocab`` vocabulary via the shared mapping.
    """
    if player is None:
        return None
    state = _LEGACY_STATE_TO_CTX.get(
        getattr(player, "player_state", None), PlayerState.STOPPED)
    now_playing = legacy_to_mv(getattr(player, "media_type", None))
    if now_playing == MVMediaType.GENERIC:
        now_playing = None
    return PlayerStatus(now_playing=now_playing, state=state)


def build_ner_list(ner) -> Dict[str, List[str]]:
    """Build the classifier's ``ner_list`` ({label: [entity]}) from the NER trie.

    The available entities are exactly the skill-registered keywords already in
    the pipeline's :class:`~ahocorasick_ner.AhocorasickNER`.  Returns ``{}`` when
    nothing has been registered.
    """
    out: Dict[str, List[str]] = {}
    if ner is None:
        return out
    try:
        automaton = getattr(ner, "automaton", None)
        if not automaton or not len(automaton):
            return out
        # ahocorasick stores (label, word) tuples as the value of each key
        for _key, value in automaton.items():
            try:
                label, word = value
            except (ValueError, TypeError):
                continue
            if label and word:
                out.setdefault(str(label), []).append(str(word))
    except Exception as e:  # noqa: BLE001 - context is best-effort, never fatal
        LOG.debug(f"could not build ner_list from NER trie: {e}")
        return {}
    return out


class ContextAwareClassifier:
    """Lazy wrapper around the standalone classifier's context-aware contract.

    The backend is selected by :func:`~ovos_media_classifier.load_media_classifier`
    from the OCP config block.  With nothing configured this is the zero-ML
    keyword (``.voc``) backend — today's behaviour.  Recognised config keys
    (all optional; defaults reproduce the bare keyword floor):

    Backend selection
        ``media_classifier_plugin``
            name of an external ``opm.media.classifier`` entry-point plugin.
        ``media_classifier_onnx_model``
            path to an opt-in ONNX trained bundle (needs the ``[onnx]`` extra).
        ``media_classifier_embedding_router``
            path to a learned embedding-router bundle (needs the ``[onnx]``
            extra); wired as a keyword+router *hybrid* by default.
        ``media_classifier_embedding_router_hybrid`` (default ``True``)
            ``False`` runs the router standalone instead of the hybrid.

    Gazetteer / entity library (embedding-router only)
        ``media_classifier_gazetteer`` (default ``True``)
            inject the bundled offline gazetteer of common real titles so bare
            titles route without a network call.
        ``media_classifier_gazetteer_size``
            cap titles per media type (``None`` = the classifier default).
        ``media_classifier_entity_library``
            ``{label: [titles]}`` of the user's own library, injected at runtime.

    Online (last-resort metadatarr layer, embedding-router hybrid only)
        ``media_classifier_online_metadatarr`` (default ``False``)
            opt into the online metadatarr last-resort layer (adds latency).
        ``media_classifier_online_timeout`` (default ``4.0``)
            per-request timeout, seconds.
        ``media_classifier_online_min_confidence`` (default ``0.5``)
            minimum confidence to accept an online answer.

    Content filter (applied at routing, see :meth:`is_blocked`)
        ``allow_adult_content`` (default ``False``)
            top-level convenience flag; lifts the default adult block.
        ``media_content_filter.blocked_genres`` / ``.blocked_media_types``
            additional genres / media types to block.
    """

    def __init__(self, config: Optional[dict] = None,
                 voc_match_func=None) -> None:
        self._config = config or {}
        self._voc_match_func = voc_match_func
        self._clf = None
        # the content filter is cheap + dependency-free; build it eagerly so the
        # routing gate is always available (adult blocked by default).
        self.content_filter = ContentFilter(self._config)

    @property
    def clf(self):
        if self._clf is None:
            self._clf = load_media_classifier(self._config,
                                              voc_match_func=self._voc_match_func)
        return self._clf

    def classify(self, query: str, lang: str,
                 player_status: Optional[PlayerStatus] = None,
                 ner_list: Optional[Dict[str, List[str]]] = None,
                 valid_labels: Optional[List[LegacyMediaType]] = None
                 ) -> Tuple[LegacyMediaType, float]:
        """Context-aware classify → ``(legacy MediaType, confidence)``.

        Runs the full multi-axis context-aware classification and folds the leaf
        + its orthogonal axes (content-form / programme-format / picture-format /
        accessibility) back onto the legacy enum through the shared
        :func:`~ocp_pipeline.media_type_map.mv_to_legacy_candidates`. A relative
        control follow-up ("next" / "pause") collapses to ``GENERIC`` here (it
        carries no media leaf) — :meth:`domain_of` exposes the control domain
        separately.

        ``valid_labels``, when given, gates the result: the axis-refined leaf
        (e.g. ``TRAILER``) is preferred, but a caller that only registered a
        coarser parent (``MOVIE``, or the broad ``VIDEO``/``AUDIO`` bucket)
        still gets a match instead of ``GENERIC`` — same fallback chain
        ``classify_media`` uses for the keyword backend. Falling back past the
        refined leaf carries a small confidence penalty.
        """
        clf = self.clf
        full = clf.classify_full(query, lang,
                                 player_status=player_status,
                                 ner_list=ner_list)
        candidates = mv_to_legacy_candidates(
            full,
            content_form=clf.classify_content_form(query, lang),
            programme_format=clf.classify_programme_format(query, lang),
            picture_format=clf.classify_picture_format(query, lang),
            accessibility=clf.classify_accessibility(query, lang),
        )
        if valid_labels is None:
            return candidates[0], float(full.confidence)
        for i, media_type in enumerate(candidates):
            if media_type == LegacyMediaType.GENERIC:
                break
            if media_type in valid_labels:
                confidence = float(full.confidence)
                if i > 0:
                    confidence *= 0.9
                return media_type, confidence
        return LegacyMediaType.GENERIC, 0.0

    def domain_of(self, query: str, lang: str,
                  player_status: Optional[PlayerStatus] = None,
                  ner_list: Optional[Dict[str, List[str]]] = None) -> str:
        """Return the ``OCPDomain`` value ("ocp_play"/"ocp_control"/"not_ocp").

        Lets the caller route a relative control follow-up ("next", "pause",
        "something else") that carries no media leaf.
        """
        full = self.clf.classify_full(query, lang,
                                      player_status=player_status,
                                      ner_list=ner_list)
        return full.domain.value

    def to_signals(self, query: str, lang: str = "en-us"):
        """Build the provider-ready :class:`mediavocab.Signals` for *query*.

        This is the classifier's primary, lossless output for the providers:
        ``medium`` / ``playback_type`` / ``content_genres`` / ``content_form`` /
        ``programme_format`` / ``variant_kind`` / ``accessibility`` /
        ``picture_format`` are all emitted in mediavocab's own vocabulary.  The
        pipeline forwards these straight to the MediaProviders instead of only a
        ``(media_type, confidence)`` pair.
        """
        return self.clf.to_signals(query, lang)

    def is_blocked(self, query: str, lang: str = "en-us") -> Tuple[bool, str]:
        """Apply the content filter → ``(blocked, reason)``.

        Adult content is blocked by default; ``allow_adult_content`` and the
        ``media_content_filter`` sub-dict (``blocked_genres`` /
        ``blocked_media_types``) tune it.  Blocked queries must not route.
        """
        if not self.content_filter.enabled:
            return False, ""
        try:
            return self.content_filter.check(self.clf, query, lang)
        except Exception as e:  # noqa: BLE001 - filtering is best-effort, fail open
            LOG.debug(f"content filter unavailable: {e}")
            return False, ""
