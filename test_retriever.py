"""
test_retriever.py

Basic tests for the chunking and retrieval logic. Run with:
    python -m pytest test_retriever.py -v
"""

from retriever import chunk_text, Retriever


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, max_words=80)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_text_splits_into_multiple_chunks():
    text = " ".join(["word"] * 200)
    chunks = chunk_text(text, max_words=80, overlap=20)
    assert len(chunks) > 1


def test_chunk_text_respects_overlap():
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_text(text, max_words=80, overlap=20)

    # The last 20 words of chunk 0 should reappear at the start of chunk 1
    chunk0_words = chunks[0].split()
    chunk1_words = chunks[1].split()
    assert chunk0_words[-20:] == chunk1_words[:20]


def test_retriever_returns_top_k_results():
    retriever = Retriever(docs_dir="docs")
    results = retriever.retrieve("SAP Joule procurement", top_k=2)
    assert len(results) == 2


def test_retriever_ranks_relevant_chunk_first():
    retriever = Retriever(docs_dir="docs")
    results = retriever.retrieve("What is retrieval augmented generation?", top_k=1)
    top_chunk, score = results[0]
    assert top_chunk.source == "rag_overview.txt"


def test_retriever_raises_on_empty_directory(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        Retriever(docs_dir=str(tmp_path))
