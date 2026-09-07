"""
music_features.py

Extracts structured, textual/numerical features from audio files using
librosa. This is the piece that makes RAG-over-music possible at all:
Claude cannot process raw audio, so this module converts each song into
a text summary (tempo, key estimate, chord estimate, structure) that
CAN be embedded, retrieved, and reasoned about by an LLM.

Feature extraction is inherently approximate -- key and chord detection
in particular are estimates based on pitch-class energy, not ground
truth. Treat the output as a useful signal, not a certified transcription.
"""

import glob
import os
from dataclasses import dataclass, field

import librosa
import numpy as np

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class SongFeatures:
    filename: str
    duration_sec: float
    tempo_bpm: float
    estimated_key: str
    chroma_mean: list = field(default_factory=list)  # 12-dim pitch class energy profile

    def to_summary(self) -> str:
        tempo_label = (
            "slow tempo, ballad-like" if self.tempo_bpm < 80 else
            "moderate tempo" if self.tempo_bpm < 115 else
            "fast tempo, upbeat"
        )
        mode_label = "minor key, melancholic/darker mood" if "minor" in self.estimated_key else \
                     "major key, brighter mood"

        return (
            f"Track: {self.filename}\n"
            f"Duration: {self.duration_sec:.1f} seconds\n"
            f"Tempo: {self.tempo_bpm:.0f} BPM ({tempo_label})\n"
            f"Estimated key: {self.estimated_key} ({mode_label})\n"
        )


def estimate_key(chroma_mean: np.ndarray) -> str:
    if np.std(chroma_mean) == 0:
        return "Unknown"

    best_score = -np.inf
    best_key = "Unknown"

    for shift in range(12):
        major_corr = np.corrcoef(chroma_mean, np.roll(MAJOR_PROFILE, shift))[0, 1]
        minor_corr = np.corrcoef(chroma_mean, np.roll(MINOR_PROFILE, shift))[0, 1]

        if major_corr > best_score:
            best_score = major_corr
            best_key = f"{NOTE_NAMES[shift]} major"
        if minor_corr > best_score:
            best_score = minor_corr
            best_key = f"{NOTE_NAMES[shift]} minor"

    return best_key


def extract_features(filepath: str) -> SongFeatures:
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    key = estimate_key(chroma_mean)

    return SongFeatures(
        filename=os.path.basename(filepath),
        duration_sec=duration,
        tempo_bpm=tempo,
        estimated_key=key,
        chroma_mean=chroma_mean.tolist(),
    )


def extract_all(songs_dir: str, extensions=(".wav", ".mp3", ".flac", ".m4a")) -> list[SongFeatures]:
    filepaths = []
    for ext in extensions:
        filepaths.extend(glob.glob(os.path.join(songs_dir, f"*{ext}")))

    return [extract_features(fp) for fp in sorted(filepaths)]


if __name__ == "__main__":
    features = extract_all("songs")
    if not features:
        print("No audio files found in songs/. Add some .wav/.mp3/.flac files and rerun.")
    for f in features:
        print(f.to_summary())
