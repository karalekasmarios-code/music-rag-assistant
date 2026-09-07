"""
rag.py

The full RAG pipeline: retrieve relevant chunks for a query, build a
prompt that grounds the model in that context, and call the Claude API
to generate an answer.

Usage:
    python rag.py "How does SAP Joule reduce manual review of contracts?"

Requires ANTHROPIC_API_KEY to be set (e.g. via a .env file loaded with
python-dotenv, or exported in your shell).
"""

import sys
import os

from dotenv import load_dotenv
from anthropic import Anthropic

from retriever import Retriever

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the context does not contain enough information "
    "to answer, say so explicitly rather than guessing. Cite which source "
    "file each part of your answer comes from."
)


def build_prompt(query: str, retrieved_chunks: list) -> str:
    context_blocks = []
    for chunk, score in retrieved_chunks:
        context_blocks.append(f"[Source: {chunk.source}]\n{chunk.text}")

    context = "\n\n---\n\n".join(context_blocks)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the context above."
    )


def answer_query(query: str, docs_dir: str = "docs", top_k: int = 3, model: str = "claude-sonnet-5") -> str:
    retriever = Retriever(docs_dir=docs_dir)
    retrieved = retriever.retrieve(query, top_k=top_k)

    prompt = build_prompt(query, retrieved)

    load_dotenv()
    client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

    message = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n")
    answer = answer_query(query)
    print("Answer:")
    print(answer)
