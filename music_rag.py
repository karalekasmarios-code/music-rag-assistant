"""
music_rag.py

Extends the text-based RAG pipeline (retriever.py) to work over a
personal song library instead of text documents.

Since Claude cannot process raw audio, this pipeline never sends audio
to the model. Instead:

  1. music_features.py extracts tempo, estimated key, and other
     numerical/textual features from each song.
  2. Those features are rendered as text summaries and treated exactly
     like the document chunks in retriever.py -- embedded with TF-IDF,
     retrieved by cosine similarity against the user's query.
  3. The retrieved song summaries are given to Claude as context, and
     Claude is asked for composition/theory feedback grounded in that
     data -- NOT asked to generate actual audio, which no LLM can do.

Query routing:
    Plain TF-IDF retrieval fetches a small subset of "relevant" songs
    for a query -- correct for descriptive/open-ended questions
    ("what mood are my songs"), but wrong for aggregate/comparison
    questions ("which pair mixes best"), where the correct answer
    requires checking the WHOLE library, not a retrieved sample. A
    simple keyword router sends comparison-style questions down a
    direct, exact computation path (mix_compatibility.py) instead,
    and only calls retrieval for genuinely open-ended questions.

Usage:
    python music_rag.py "What key are my upbeat songs in?"
    python music_rag.py "Which songs mix well together?"
"""

import sys
import os

from dotenv import load_dotenv
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from music_features import extract_all, SongFeatures
from mix_compatibility import find_best_mix_pairs, format_pair_result

SYSTEM_PROMPT = (
    "You are a music theory and composition assistant. You will be given "
    "structured data (tempo, estimated key, duration) extracted from the "
    "user's own song library, NOT the actual audio. Use this data to answer "
    "questions about their music and suggest composition or arrangement "
    "ideas. Be explicit that your feedback is based on the extracted "
    "features, not on listening to the audio. Do not claim to have heard "
    "the songs. If the data doesn't support a confident answer, say so."
)

# Keywords that signal a full-library comparison/aggregate question rather
# than a descriptive lookup. Deliberately simple (a real system might use
# an LLM call to classify intent instead) -- but transparent and cheap,
# and easy to extend as new comparison phrasing comes up.
COMPARISON_KEYWORDS = [
    "mix well", "mix together", "back to back", "back-to-back",
    "pair", "pairs", "blend", "transition between", "which two",
    "best combination", "go together",
]


def is_comparison_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in COMPARISON_KEYWORDS)


class MusicRetriever:
    def __init__(self, songs_dir: str):
        self.songs: list[SongFeatures] = extract_all(songs_dir)
        if not self.songs:
            raise ValueError(f"No audio files found in {songs_dir}")

        self.summaries = [s.to_summary() for s in self.songs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors = self.vectorizer.fit_transform(self.summaries)

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [(self.songs[i], self.summaries[i], float(scores[i])) for i in ranked]


def build_prompt(query: str, retrieved) -> str:
    context = "\n\n---\n\n".join(summary for _, summary, _ in retrieved)
    return (
        f"Song data from the user's library:\n\n{context}\n\n"
        f"User question: {query}\n\n"
        f"Answer using only the song data above."
    )


def build_comparison_prompt(query: str, songs: list[SongFeatures]) -> str:
    """
    For comparison-style questions, the best pairs are computed exactly
    (see mix_compatibility.py) across the WHOLE library, not a retrieved
    subset. Claude's job here is narrower: explain/contextualize an
    already-correct computed answer, not discover it from a sample.
    """
    best_pairs = find_best_mix_pairs(songs, top_n=3)

    if not best_pairs:
        computed_results = (
            "No pairs were found within a reasonable BPM gap across the "
            "entire library -- the songs may be too spread out in tempo "
            "for a smooth back-to-back mix."
        )
    else:
        computed_results = "\n\n".join(
            format_pair_result(a, b, gap, dist) for a, b, gap, dist in best_pairs
        )

    return (
        f"The following pairs were computed by comparing EVERY song in the "
        f"user's library against every other song, ranked by tempo closeness "
        f"and harmonic (Camelot wheel) key compatibility:\n\n"
        f"{computed_results}\n\n"
        f"User question: {query}\n\n"
        f"Explain why the top pair(s) above work well together musically, "
        f"in a sentence or two. Do not suggest a different pair -- the "
        f"ranking above was computed exactly across the full library, not "
        f"a sample, so it is already correct."
    )


def answer_query(query: str, songs_dir: str = "songs", top_k: int = 3, model: str = "claude-sonnet-5") -> str:
    load_dotenv()
    client = Anthropic(default_headers={"Accept-Encoding": "gzip, deflate"})

    if is_comparison_query(query):
        songs = extract_all(songs_dir)
        if not songs:
            raise ValueError(f"No audio files found in {songs_dir}")
        prompt = build_comparison_prompt(query, songs)
    else:
        retriever = MusicRetriever(songs_dir)
        retrieved = retriever.retrieve(query, top_k=top_k)
        prompt = build_prompt(query, retrieved)

    message = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    for block in message.content:
        if block.type == "text":
            return block.text
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python music_rag.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}")
    print(f"Routed to: {'direct comparison' if is_comparison_query(query) else 'TF-IDF retrieval'}\n")
    print(answer_query(query))

