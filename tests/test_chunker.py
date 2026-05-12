import pytest

from src.chunker import ChunkConfig, chunk_document, chunk_documents


def test_short_doc_yields_single_chunk():
    doc = {"id": "d", "title": "t", "text": "one two three"}
    chunks = chunk_document(doc, ChunkConfig(max_tokens=10, overlap=2))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "one two three"
    assert chunks[0]["parent_doc_id"] == "d"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["id"] == "d::chunk-0"


def test_long_doc_chunks_with_overlap():
    tokens = [f"w{i}" for i in range(20)]
    doc = {"id": "d", "title": "t", "text": " ".join(tokens)}
    chunks = chunk_document(doc, ChunkConfig(max_tokens=8, overlap=2))

    # stride = 6, so chunks start at 0, 6, 12; the chunk at 12 spans [12:20]
    # which reaches end-of-text → loop terminates with 3 chunks.
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
    assert chunks[0]["text"].split()[:3] == ["w0", "w1", "w2"]
    # Overlap: chunk 1 should start at token index 6.
    assert chunks[1]["text"].split()[0] == "w6"
    # Last chunk reaches the end of the source.
    assert chunks[-1]["text"].split()[-1] == "w19"
    # All chunks point at the same parent.
    assert {c["parent_doc_id"] for c in chunks} == {"d"}


def test_empty_text_yields_no_chunks():
    chunks = chunk_document({"id": "d", "title": "t", "text": "   "})
    assert chunks == []


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ChunkConfig(max_tokens=0)
    with pytest.raises(ValueError):
        ChunkConfig(max_tokens=5, overlap=5)
    with pytest.raises(ValueError):
        ChunkConfig(max_tokens=5, overlap=-1)


def test_chunk_documents_preserves_order():
    docs = [
        {"id": "a", "title": "A", "text": " ".join(["x"] * 30)},
        {"id": "b", "title": "B", "text": "short"},
    ]
    chunks = chunk_documents(docs, ChunkConfig(max_tokens=10, overlap=2))
    parent_ids_in_order = [c["parent_doc_id"] for c in chunks]
    # All "a" chunks come before any "b" chunk.
    assert parent_ids_in_order == sorted(parent_ids_in_order)
