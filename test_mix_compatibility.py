"""
test_mix_compatibility.py

Tests for the exact pairwise mix-compatibility computation and the
query router that decides between retrieval and direct computation.
"""

from music_features import SongFeatures
from mix_compatibility import camelot_distance, find_best_mix_pairs
from music_rag import is_comparison_query


def make_song(filename, tempo, key):
    return SongFeatures(filename=filename, duration_sec=180.0, tempo_bpm=tempo, estimated_key=key)


def test_camelot_distance_identical_key_is_zero():
    assert camelot_distance("C major", "C major") == 0


def test_camelot_distance_relative_minor_is_one():
    # C major and A minor are relative major/minor -- same Camelot number
    assert camelot_distance("C major", "A minor") == 1


def test_camelot_distance_unknown_key_returns_sentinel():
    assert camelot_distance("Unknown", "C major") == 99


def test_find_best_mix_pairs_respects_bpm_gap():
    songs = [
        make_song("a.wav", 120, "C major"),
        make_song("b.wav", 200, "C major"),  # too far in tempo, should be excluded
    ]
    pairs = find_best_mix_pairs(songs, top_n=5, max_bpm_gap=8.0)
    assert pairs == []


def test_find_best_mix_pairs_ranks_best_key_match_first():
    songs = [
        make_song("close_key.wav", 120, "A minor"),      # relative minor of C major
        make_song("far_key.wav", 121, "F# major"),        # harmonically distant from C major
        make_song("reference.wav", 122, "C major"),
    ]
    pairs = find_best_mix_pairs(songs, top_n=5, max_bpm_gap=8.0)
    # the reference + close_key pair should rank ahead of reference + far_key
    top_pair_files = {pairs[0][0].filename, pairs[0][1].filename}
    assert top_pair_files == {"reference.wav", "close_key.wav"}


def test_router_detects_comparison_queries():
    assert is_comparison_query("Which songs mix well together?") is True
    assert is_comparison_query("What tempo is this track?") is False
    assert is_comparison_query("Best back to back pairing?") is True
    assert is_comparison_query("Is this song in a minor key?") is False
