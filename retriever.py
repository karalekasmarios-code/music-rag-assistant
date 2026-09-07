"""
retriever.py

Handles the "R" in RAG: loading documents, splitting them into chunks,
embedding those chunks, and retrieving the most relevant ones for a
given query using cosine similarity.

Uses TF-IDF as the embedding method. This is a sparse, fully offline
retrieval method (no external model downloads or API calls required),
which makes the pipeline reproducible and runnable in any environment
with just scikit-learn installed. Swapping in a dense embedding model
would only require changing the `embed` step below -- the chunking and
retrieval interface stays the same.
"""

import os
import glob
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int

def load_documents(docs_dir: str) -> list[tuple[str, str]]:
    documents = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            documents.append((os.path.basename(path), f.read()))
    return documents

def chunk_text(text: str, max_words: int = 80, overlap: int = 20) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += max_words - overlap
    return chunks

class Retriever:
    def __init__(self, docs_dir: str, max_words: int = 80, overlap: int = 20):
        self.chunks: list[Chunk] = []

        for filename, text in load_documents(docs_dir):
            for i, chunk_text_ in enumerate(chunk_text(text, max_words, overlap)):
                self.chunks.append(Chunk(text=chunk_text_, source=filename, chunk_id=i))

        if not self.chunks:
            raise ValueError(f"No .txt documents found in {docs_dir}")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.chunk_vectors)[0]

        ranked_indices = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]
        print(f"\nQuery: {q}")
        for chunk, score in retriever.retrieve(q, top_k=2):
            print(f"  [{score:.3f}] ({chunk.source} #{chunk.chunk_id}) {chunk.text[:100]}...")
