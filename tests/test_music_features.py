import numpy as np

from music_features import MAJOR_PROFILE, MINOR_PROFILE, SongFeatures, estimate_key


def test_detects_g_major():
    assert estimate_key(np.roll(MAJOR_PROFILE, 7)) == "G major"


def test_detects_a_minor():
    assert estimate_key(np.roll(MINOR_PROFILE, 9)) == "A minor"


def test_flat_chroma_is_unknown():
    assert estimate_key(np.ones(12)) == "Unknown"


def test_summary_contains_descriptive_labels():
    summary = SongFeatures("x.wav", 200.0, 128.0, "A minor").to_summary()
    assert "upbeat" in summary
    assert "minor key" in summary
