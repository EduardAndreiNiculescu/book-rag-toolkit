from pathlib import Path

from book_rag.chunk import chunk, sentence_stream, split_sentences
from book_rag.generate import FRAMEWORKS, build_prompt
from book_rag.search import ChunkSearch

BOOK = (Path(__file__).parent.parent / "examples" / "sample_book.md").read_text(encoding="utf-8")


def test_sentences_and_structure():
    assert split_sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]
    sents = sentence_stream(BOOK)
    assert {s.chapter for s in sents} >= {"CHAPTER 1: WHY PEOPLE COME BACK", "CHAPTER 3: REWARDS THAT WORK"}
    assert max(s.page for s in sents) == 3


def test_chunks_carry_context_and_overlap():
    chunks = chunk(sentence_stream(BOOK), "Sample Guide", size=200, overlap=60)
    assert len(chunks) >= 4
    first = chunks[0]
    assert first["text"].startswith("Document: Sample Guide | Section:")
    assert first["metadata"]["start_page"] == 1
    # consecutive chunks share at least one sentence (overlap)
    assert any(s in chunks[1]["raw_text"] for s in split_sentences(first["raw_text"])[-2:])


def test_strip_removes_watermarks():
    sents = sentence_stream("Body text here. WATERMARK.COM More text.", strip=["WATERMARK.COM"])
    assert all("WATERMARK" not in s.text for s in sents)


def test_bm25_ranks_the_right_chapter():
    engine = ChunkSearch(chunk(sentence_stream(BOOK), "Sample Guide", size=250, overlap=50))
    top = engine.search("guarantee risk", limit=1)[0][1]
    assert top["metadata"]["chapter"].startswith("CHAPTER 2")
    assert engine.search("stamp card progress", limit=1)[0][1]["metadata"]["chapter"].startswith("CHAPTER 3")


def test_prompt_contains_retrieved_passages():
    engine = ChunkSearch(chunk(sentence_stream(BOOK), "Sample Guide", size=250, overlap=50))
    system, user = build_prompt(engine, "offer", "digital loyalty cards", "coffee shops", "29 EUR/month", k=2)
    assert "Grand Slam Offer" in system
    assert "digital loyalty cards" in user and "Sample Guide" in user
    assert set(FRAMEWORKS) == {"offer", "outreach", "viral", "persuasion"}
