from mix_compatibility import camelot_distance, find_best_mix_pairs
from music_features import SongFeatures


def song(name, bpm, key):
    return SongFeatures(filename=name, duration_sec=180.0, tempo_bpm=bpm, estimated_key=key)


def test_same_key_is_zero():
    assert camelot_distance("C major", "C major") == 0


def test_relative_major_minor_is_one():
    assert camelot_distance("C major", "A minor") == 1


def test_adjacent_same_mode_is_one():
    assert camelot_distance("C major", "G major") == 1


def test_wheel_wraps_around():
    assert camelot_distance("E major", "B major") == 1


def test_adjacent_different_mode_is_two():
    assert camelot_distance("C major", "E minor") == 2


def test_unknown_key_is_penalised():
    assert camelot_distance("Unknown", "C major") == 99


def test_pairs_outside_bpm_gap_are_excluded():
    songs = [song("a", 120, "C major"), song("b", 140, "C major")]
    assert find_best_mix_pairs(songs) == []


def test_key_match_ranks_above_tempo_match():
    clash = song("clash", 120, "F# major")
    match = song("match", 126, "C major")
    anchor = song("anchor", 120, "C major")
    best = find_best_mix_pairs([anchor, clash, match], top_n=1)
    a, b, _, dist = best[0]
    assert {a.filename, b.filename} == {"anchor", "match"}
    assert dist == 0


def test_checks_every_pair_in_library():
    songs = [song(str(i), 120 + i, "C major") for i in range(6)]
    assert len(find_best_mix_pairs(songs, top_n=100)) == 15
