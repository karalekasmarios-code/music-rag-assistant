# Music RAG Assistant

Ask questions about your own music library in plain English, answered by Claude using features extracted from the audio itself.

Claude has no audio input, so the pipeline never sends audio to the model. It extracts tempo and key from each track with `librosa`, turns them into short text summaries, and gives Claude only that data to reason about.

## The key idea: retrieval is not always the right tool

A standard RAG pipeline retrieves the few most similar items and passes them to the model. That works for descriptive questions ("what key are my slowest songs in?"), but fails for comparison questions ("which two songs mix best?"): the true best pair may never be retrieved, so the model can't find it.

This project routes the two kinds of question differently:

| Question type | Example | Path |
|---|---|---|
| Descriptive | "What mood are my songs?" | TF-IDF retrieval → Claude |
| Comparison | "Which songs mix well together?" | Exact pairwise comparison of the whole library → Claude explains the result |

The comparison path scores every pair of tracks by BPM gap and harmonic distance on the Camelot wheel, the system DJs use to judge which keys mix cleanly.

## How it works

1. **Feature extraction** (`music_features.py`): tempo via beat tracking and key via chroma features matched against Krumhansl–Schmuckler key profiles. Each track becomes a summary with descriptive labels ("fast tempo, upbeat", "minor key, darker mood") so word-based retrieval can match natural queries.
2. **Routing** (`music_rag.py`): keyword-based detection of comparison questions.
3. **Retrieval** (`music_rag.py`, `retriever.py`): TF-IDF embeddings and cosine similarity, fully offline.
4. **Exact comparison** (`mix_compatibility.py`): every pair is checked, filtered by BPM gap and ranked by key compatibility first, tempo second.
5. **Generation**: Claude answers from the extracted data only and is instructed not to claim it has heard the audio.

`rag.py` and `retriever.py` are the same pattern applied to text documents in `docs/`.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key-here
```

Put audio files (`.wav`, `.mp3`, `.flac`, `.m4a`) in a `songs/` folder.

## Usage

```bash
python music_rag.py "What key are my slowest songs in?"
python music_rag.py "Which songs mix well together?"
python music_features.py
python mix_compatibility.py
python rag.py "How does SAP Joule reduce manual review of contracts?"
```

`music_features.py` and `mix_compatibility.py` run without an API key.

## Tests

```bash
python -m pytest -v
```

The suite covers Camelot distance, pair ranking across the full library, key detection, query routing and text retrieval. None of it needs audio files or an API key.

## Project structure

```
music-rag-assistant/
├── music_features.py
├── mix_compatibility.py
├── music_rag.py
├── retriever.py
├── rag.py
├── docs/
├── tests/
├── requirements.txt
└── pytest.ini
```

## Known limitations

- **Tempo octave errors**: beat tracking can return half or double the true BPM, which can hide pairs that would mix well.
- **Relative major/minor ambiguity**: C major and A minor share the same notes, so key detection from pitch-class energy alone can confuse them.
- **Keyword routing**: comparison questions phrased without the expected keywords fall back to retrieval.
- **No audio output**: Claude gives analysis and suggestions in text; generating sound would need a separate audio model.
