from retriever import Retriever, chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("one two three", max_words=10) == ["one two three"]


def test_chunks_overlap():
    words = " ".join(str(i) for i in range(100))
    chunks = chunk_text(words, max_words=40, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_retrieves_most_relevant_document(tmp_path):
    (tmp_path / "boats.txt").write_text("sailing yachts harbour marina boats")
    (tmp_path / "music.txt").write_text("tempo key chords melody songs")
    retriever = Retriever(str(tmp_path))
    chunk, score = retriever.retrieve("which songs have a fast tempo", top_k=1)[0]
    assert chunk.source == "music.txt"
    assert score > 0
