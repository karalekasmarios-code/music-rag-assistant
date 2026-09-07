"""
test_music.py

Tests for audio feature extraction and music retrieval.

Note: key and tempo detection are estimation algorithms, not ground
truth extraction -- tests check that the pipeline runs correctly and
produces internally consistent, sane output, not that every estimate
is perfectly accurate (tempo octave errors and relative major/minor
ambiguity are known, expected limitations documented in music_features.py).
"""

import pytest
from music_features import extract_features, estimate_key, SongFeatures
from music_rag import MusicRetriever
import numpy as np


def test_extract_features_runs_on_test_song():
    features = extract_features("songs/test_song_am_f_c_g.wav")
    assert features.duration_sec > 0
    assert features.tempo_bpm >= 0
    assert features.estimated_key != ""


def test_estimate_key_handles_flat_chroma_gracefully():
    # A perfectly flat chroma vector (e.g. silence) has zero variance --
    # correlation against any key profile is mathematically undefined.
    # The function should return "Unknown" cleanly, not crash or produce
    # a misleading result from a NaN comparison.
    flat_chroma = np.ones(12)
    key = estimate_key(flat_chroma)
    assert key == "Unknown"


def test_estimate_key_returns_valid_key_for_real_chroma():
    # A chroma vector with actual tonal variation should return a real
    # major/minor key guess.
    varied_chroma = np.array([6.0, 1.0, 2.0, 1.0, 4.0, 3.0, 1.0, 5.0, 1.0, 3.0, 1.0, 2.0])
    key = estimate_key(varied_chroma)
    assert "major" in key or "minor" in key


def test_song_summary_includes_derived_labels():
    features = SongFeatures(
        filename="test.wav", duration_sec=10.0, tempo_bpm=140, estimated_key="C minor"
    )
    summary = features.to_summary()
    assert "upbeat" in summary
    assert "melancholic" in summary or "darker" in summary


def test_song_summary_slow_tempo_label():
    features = SongFeatures(
        filename="test.wav", duration_sec=10.0, tempo_bpm=60, estimated_key="C major"
    )
    summary = features.to_summary()
    assert "ballad-like" in summary


def test_music_retriever_loads_songs_directory():
    retriever = MusicRetriever("songs")
    assert len(retriever.songs) >= 1


def test_music_retriever_raises_on_empty_directory(tmp_path):
    with pytest.raises(ValueError):
        MusicRetriever(str(tmp_path))


def test_retrieval_finds_minor_key_song_for_melancholic_query():
    retriever = MusicRetriever("songs")
    results = retriever.retrieve("melancholic minor key song", top_k=1)
    top_song, _, score = results[0]
    assert "minor" in top_song.estimated_key.lower()
