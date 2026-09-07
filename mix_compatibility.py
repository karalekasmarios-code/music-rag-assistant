"""
mix_compatibility.py

Direct, exact computation for "which songs go well together" style
questions -- BPM closeness and key compatibility across every pair in
the library.

This deliberately does NOT use retrieval. Retrieval (TF-IDF or any
other similarity search) fetches a small subset of "relevant" items
for a query -- it is the right tool when the question is about
*finding* something in a large space. But "which pair mixes best" is
a full-dataset aggregate/comparison question: the correct answer
requires checking every pair, not a sample of a few. Feeding a
retrieval-selected subset into an LLM for this kind of question means
the true best pair might never even be shown to it. This module
computes the exact answer directly instead.
"""

import itertools
from music_features import SongFeatures

# Camelot wheel: maps each musical key to a position used by DJs to judge
# harmonic compatibility. Keys that are adjacent on the wheel (or share a
# number with major/minor swapped) mix harmonically; keys far apart clash.
CAMELOT = {
    "C major": "8B", "A minor": "8A",
    "G major": "9B", "E minor": "9A",
    "D major": "10B", "B minor": "10A",
    "A major": "11B", "F# minor": "11A",
    "E major": "12B", "C# minor": "12A",
    "B major": "1B", "G# minor": "1A",
    "F# major": "2B", "D# minor": "2A",
    "C# major": "3B", "A# minor": "3A",
    "G# major": "4B", "F minor": "4A",
    "D# major": "5B", "C minor": "5A",
    "A# major": "6B", "G minor": "6A",
    "F major": "7B", "D minor": "7A",
}


def camelot_distance(key_a: str, key_b: str) -> int:
    """
    Rough harmonic distance between two keys on the Camelot wheel.
    0 = identical key. 1 = adjacent on the wheel or relative major/minor
    (generally considered a smooth mix). Larger = more harmonically
    distant / more likely to clash.

    Returns a large sentinel value if either key is unrecognized
    (e.g. "Unknown" from a low-confidence key estimate), so unmatched
    songs sort to the bottom rather than being silently treated as a
    perfect match.
    """
    pos_a = CAMELOT.get(key_a)
    pos_b = CAMELOT.get(key_b)
    if pos_a is None or pos_b is None:
        return 99

    if pos_a == pos_b:
        return 0

    num_a, letter_a = int(pos_a[:-1]), pos_a[-1]
    num_b, letter_b = int(pos_b[:-1]), pos_b[-1]

    if num_a == num_b:  # same number, relative major/minor
        return 1

    # circular distance around the 12-position wheel
    diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    return diff if letter_a == letter_b else diff + 1


def find_best_mix_pairs(songs: list[SongFeatures], top_n: int = 3, max_bpm_gap: float = 8.0):
    """
    Compare every pair of songs in the library and rank them by mix
    compatibility: BPM closeness first (large tempo gaps make beatmatching
    hard regardless of key), then harmonic (Camelot) distance as a
    tiebreaker among tempo-compatible pairs.

    Returns the top_n pairs as (song_a, song_b, bpm_gap, camelot_dist).
    """
    pairs = []
    for song_a, song_b in itertools.combinations(songs, 2):
        bpm_gap = abs(song_a.tempo_bpm - song_b.tempo_bpm)
        if bpm_gap > max_bpm_gap:
            continue
        dist = camelot_distance(song_a.estimated_key, song_b.estimated_key)
        pairs.append((song_a, song_b, bpm_gap, dist))

    # sort by harmonic distance first, then tempo gap -- a near-identical
    # key match with a slightly larger (but still viable) tempo gap makes
    # for a smoother mix than a perfect tempo match in a clashing key
    pairs.sort(key=lambda p: (p[3], p[2]))
    return pairs[:top_n]


def format_pair_result(song_a: SongFeatures, song_b: SongFeatures, bpm_gap: float, dist: int) -> str:
    quality = (
        "excellent harmonic match" if dist == 0 else
        "strong match (relative major/minor or adjacent key)" if dist == 1 else
        "workable, some key tension" if dist <= 2 else
        "harmonically distant -- likely to clash"
    )
    return (
        f"{song_a.filename} ({song_a.tempo_bpm:.0f} BPM, {song_a.estimated_key}) + "
        f"{song_b.filename} ({song_b.tempo_bpm:.0f} BPM, {song_b.estimated_key})\n"
        f"  BPM gap: {bpm_gap:.1f} | Camelot distance: {dist} ({quality})"
    )


if __name__ == "__main__":
    from music_features import extract_all

    songs = extract_all("songs")
    if len(songs) < 2:
        print("Need at least 2 songs to compare pairs.")
    else:
        best = find_best_mix_pairs(songs, top_n=5)
        if not best:
            print("No pairs found within the BPM gap threshold. Try increasing max_bpm_gap.")
        for song_a, song_b, gap, dist in best:
            print(format_pair_result(song_a, song_b, gap, dist))
            print()
