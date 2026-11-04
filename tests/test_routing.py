import pytest

from music_rag import is_comparison_query


@pytest.mark.parametrize("query", [
    "Which songs mix well together?",
    "What are the best pairs for a back-to-back set?",
    "Which two tracks blend best?",
])
def test_comparison_queries_use_exact_path(query):
    assert is_comparison_query(query)


@pytest.mark.parametrize("query", [
    "What key are my slowest songs in?",
    "What mood are my songs?",
])
def test_descriptive_queries_use_retrieval(query):
    assert not is_comparison_query(query)
