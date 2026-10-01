"""Unit tests for ocp_pipeline.bridge (mediavocab <-> legacy ocp.MediaType)."""
import unittest

from ovos_utils.ocp import MediaType as OCPMediaType
from ovos_utils.ocp import PlaybackType as OCPPlaybackType

import mediavocab
from mediavocab import EntityKind, MediaType as MVMediaType, Release, Work
from mediavocab.models.entity import Credit, EntityRef
from mediavocab.taxonomy import PlaybackType as MVPlaybackType

from ocp_pipeline import bridge


def _make_release(title, media_type, uri, conf, artist="", runtime=0):
    credits = []
    if artist:
        credits = [Credit(entity=EntityRef(name=artist, kind=EntityKind.GROUP),
                          role="artist")]
    work = Work(title=title, media_type=media_type, runtime=runtime,
               credits=credits)
    return Release(work=work, uri=uri, image=f"{title}.png", match_confidence=conf)


class TestBridge(unittest.TestCase):
    def test_release_to_ocp_result_fields(self):
        rel = _make_release("Black Album", MVMediaType.MUSIC,
                            "https://music/black", 0.85, artist="Metallica",
                            runtime=3600)
        d = bridge.release_to_ocp_result(rel, "mock.music")
        self.assertEqual(d["uri"], "https://music/black")
        self.assertEqual(d["title"], "Black Album")
        self.assertEqual(d["artist"], "Metallica")
        self.assertEqual(d["length"], 3600)
        # media_type is folded back to the legacy ovos_utils.ocp taxonomy
        self.assertEqual(d["media_type"], OCPMediaType.MUSIC)
        self.assertEqual(d["playback"], OCPPlaybackType.AUDIO)
        self.assertEqual(d["match_confidence"], 85)
        self.assertEqual(d["skill_id"], "mock.music")

    def test_confidence_clamped_and_scaled(self):
        rel = _make_release("x", MVMediaType.MUSIC, "u", 1.0)
        self.assertEqual(bridge.release_to_ocp_result(rel, "p")["match_confidence"], 100)
        rel0 = _make_release("x", MVMediaType.MUSIC, "u", 0.0)
        self.assertEqual(bridge.release_to_ocp_result(rel0, "p")["match_confidence"], 0)

    def test_no_credit_artist_empty(self):
        rel = _make_release("x", MVMediaType.MOVIE, "u", 0.5)
        self.assertEqual(bridge.release_to_ocp_result(rel, "p")["artist"], "")

    def test_playback_type_mapping(self):
        self.assertEqual(bridge.mediavocab_playback_to_ocp(MVPlaybackType.AUDIO),
                         OCPPlaybackType.AUDIO)
        self.assertEqual(bridge.mediavocab_playback_to_ocp(MVPlaybackType.VIDEO),
                         OCPPlaybackType.VIDEO)
        self.assertEqual(bridge.mediavocab_playback_to_ocp(MVPlaybackType.INTERACTIVE),
                         OCPPlaybackType.SKILL)
        self.assertEqual(bridge.mediavocab_playback_to_ocp(MVPlaybackType.PAGED),
                         OCPPlaybackType.WEBVIEW)

    def test_media_type_to_signals_music(self):
        sig = bridge.media_type_to_signals(OCPMediaType.MUSIC, "metallica")
        self.assertEqual(sig.medium, MVMediaType.MUSIC)
        self.assertEqual(sig.title, "metallica")

    def test_media_type_to_signals_generic_is_typeless(self):
        sig = bridge.media_type_to_signals(OCPMediaType.GENERIC, "anything")
        self.assertIsNone(sig.medium)

    def test_ocp_to_mediavocab_movie(self):
        self.assertEqual(bridge.ocp_media_type_to_mediavocab(OCPMediaType.MOVIE),
                         MVMediaType.MOVIE)

    def test_mediavocab_to_ocp_roundtrip_music(self):
        self.assertEqual(bridge.mediavocab_media_type_to_ocp(MVMediaType.MUSIC),
                         OCPMediaType.MUSIC)

    def test_mediavocab_unmapped_falls_back_to_generic(self):
        self.assertEqual(bridge.mediavocab_media_type_to_ocp(MVMediaType.PLAYLIST),
                         OCPMediaType.GENERIC)

    def test_media_type_override_wins(self):
        """Callers stamp the QUERY's legacy media type: the mediavocab fold is
        not injective (NEWS -> RADIO -> RADIO)."""
        rel = _make_release("BBC News", MVMediaType.RADIO, "u", 0.9)
        d = bridge.release_to_ocp_result(rel, "p", media_type=OCPMediaType.NEWS)
        self.assertEqual(d["media_type"], OCPMediaType.NEWS)
        # without the override the Release's own type is folded back
        self.assertEqual(bridge.release_to_ocp_result(rel, "p")["media_type"],
                         OCPMediaType.RADIO)

    def test_unscored_release_gets_fuzzy_fallback_confidence(self):
        rel = _make_release("Groove Salad", MVMediaType.RADIO,
                            "https://somafm.com/groovesalad.pls", 0.0)
        sig = bridge.media_type_to_signals(OCPMediaType.RADIO, "groove salad")
        d = bridge.release_to_ocp_result(rel, "somafm", signals=sig)
        # a good title match must survive the default min_score of 50
        self.assertGreaterEqual(d["match_confidence"], 50)

    def test_unscored_release_with_bad_title_stays_low(self):
        rel = _make_release("Totally Unrelated", MVMediaType.RADIO, "u", 0.0)
        sig = bridge.media_type_to_signals(OCPMediaType.RADIO, "groove salad")
        d = bridge.release_to_ocp_result(rel, "somafm", signals=sig)
        self.assertLess(d["match_confidence"], 50)

    def test_provider_score_is_never_overridden(self):
        rel = _make_release("Totally Unrelated", MVMediaType.RADIO, "u", 0.8)
        sig = bridge.media_type_to_signals(OCPMediaType.RADIO, "groove salad")
        d = bridge.release_to_ocp_result(rel, "somafm", signals=sig)
        self.assertEqual(d["match_confidence"], 80)

    def test_fallback_without_signals_is_zero(self):
        rel = _make_release("Groove Salad", MVMediaType.RADIO, "u", 0.0)
        self.assertEqual(bridge.release_to_ocp_result(rel, "p")["match_confidence"], 0)


if __name__ == "__main__":
    unittest.main()
