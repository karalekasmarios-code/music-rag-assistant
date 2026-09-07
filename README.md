# RAG Pipeline with Claude

A small but complete Retrieval-Augmented Generation (RAG) pipeline: it
retrieves relevant chunks from a local document set and uses the Claude
API to generate an answer grounded in that retrieved context, rather
than relying on the model's training data alone.

## How it works

1. **Chunking** (`retriever.py`) — documents in `docs/` are split into
   overlapping word-based chunks, so relevant information isn't lost at
   a chunk boundary.
2. **Retrieval** (`retriever.py`) — chunks are embedded with TF-IDF and
   ranked against the query using cosine similarity. TF-IDF was chosen
   over a dense embedding model to keep the pipeline fully offline and
   reproducible with no external model downloads — the `Retriever`
   class is written so the embedding method can be swapped out
   (e.g. for `sentence-transformers` or an embeddings API) without
   changing the rest of the pipeline.
3. **Generation** (`rag.py`) — the retrieved chunks are inserted into a
   prompt and sent to the Claude API, with a system prompt that
   instructs the model to answer only from the provided context and
   flag when the context is insufficient.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key-here
```

## Usage

```bash
python rag.py "How does SAP Joule reduce manual review of contracts?"
```

Test the retrieval step on its own (no API key needed):

```bash
python retriever.py
```

Run the test suite:

```bash
python -m pytest test_retriever.py -v
```

## Project structure

```
rag-project/
├── docs/              # Source documents (plain .txt) -- text RAG demo
├── songs/             # Audio files (.wav/.mp3/.flac) -- music RAG demo
├── retriever.py       # Chunking + TF-IDF retrieval (text)
├── rag.py             # Text RAG: prompt construction + Claude API call
├── music_features.py  # Audio feature extraction (tempo, key) via librosa
├── music_rag.py        # Music RAG: features -> retrieval -> Claude API call
├── test_retriever.py  # Unit tests for text chunking and retrieval
├── test_music.py       # Unit tests for feature extraction and music retrieval
└── requirements.txt
```

## Music RAG: retrieving over a personal song library

`music_rag.py` extends the same RAG pattern to audio. **Claude cannot
process raw audio** -- there is no audio input modality in the API --
so this pipeline never sends audio to the model. Instead:

1. `music_features.py` extracts tempo (BPM) and an estimated musical key
   from each song using `librosa`, and renders them as a text summary
   with derived descriptive labels (e.g. "fast tempo, upbeat", "minor
   key, melancholic/darker mood"). The descriptive labels matter for
   retrieval quality: TF-IDF only matches literal words, so a query like
   "upbeat songs" would never match a summary that only said "123 BPM" --
   there's no shared vocabulary between the two.
2. Those summaries are embedded and retrieved with the same TF-IDF +
   cosine similarity approach as the text pipeline.
3. Retrieved song summaries are given to Claude, which is explicitly
   instructed to reason about the *extracted data*, not to claim it has
   listened to the audio.

```bash
python music_features.py          # test feature extraction on songs/
python music_rag.py "What key are my slowest songs in?"
```

### Known limitations (real, not hidden)

- **Tempo octave errors**: beat-tracking can lock onto every other beat
  (or every second beat) instead of the true tempo, returning half or
  double the actual BPM. This is a well-known limitation of onset-based
  beat trackers generally, not specific to this implementation.
- **Relative major/minor ambiguity**: C major and A minor share the
  exact same notes, so pitch-class-based key detection alone cannot
  always distinguish between them without additional analysis (e.g.
  weighting which chord the song starts or ends on).
- **No actual audio generation**: this pipeline gives Claude structured
  data to reason about (tempo, key, composition suggestions in text/
  notation form). Generating actual audio would require a separate,
  dedicated audio generation model (e.g. MusicGen, Stable Audio) --
  no text-based LLM, including Claude, can output sound.

## Notes / possible extensions

- Swap TF-IDF for a dense embedding model to capture semantic
  similarity beyond keyword overlap.
- Add a vector database (e.g. FAISS, Chroma) if the document set grows
  beyond what fits comfortably in memory.
- Add citation verification: check that the model's cited source
  actually contains the claim being attributed to it.
