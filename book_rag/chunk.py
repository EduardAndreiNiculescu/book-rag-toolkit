"""Structure-aware chunking for long-form books.

Most RAG chunkers split blindly every N characters. Books have structure, and
a retrieved passage is far more useful when the model also knows *where* it
came from. This chunker:

1. walks the document page by page (form feeds from the PDF conversion),
2. tracks the current section/chapter from heading lines,
3. streams sentences into ~1,200-character chunks with ~250 characters of overlap,
4. stamps every chunk with its dominant section, chapter and page range, and
   prepends that as a one-line header to the text that gets embedded/searched.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

SENTENCE_END = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s")
DEFAULT_HEADING = re.compile(r"^(#{1,3}\s+\S|(CHAPTER|SECTION|PART)\b|\d+\.\s+[A-Z])", re.I)


@dataclass
class Sentence:
    text: str
    section: str
    chapter: str
    page: int


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def is_heading(block: str, known: list[str], pattern: re.Pattern) -> bool:
    if len(block) > 120 or "\n" in block.strip():
        return False
    up = block.upper().strip()
    return any(k.upper() in up or up in k.upper() for k in known if up) or bool(pattern.match(block.strip()))


def sentence_stream(content: str, sections: list[str] = (), chapters: list[str] = (),
                    heading_pattern: re.Pattern = DEFAULT_HEADING, strip: list[str] = ()) -> list[Sentence]:
    """Turns raw Markdown into sentences annotated with section/chapter/page."""
    section, chapter, out = "Front matter", "Front matter", []
    for page_num, page in enumerate(content.split("\x0c"), 1):
        for noise in strip:
            page = page.replace(noise, "")
        page = "\n".join(line.strip() for line in page.splitlines())
        for block in (b.strip() for b in page.split("\n\n") if b.strip()):
            if sections and is_heading(block, list(sections), re.compile(r"$^")):
                section, chapter = block.lstrip("# ").strip(), "Introduction"
                continue
            if is_heading(block, list(chapters), heading_pattern):
                chapter = block.lstrip("# ").strip()
                continue
            out += [Sentence(s, section, chapter, page_num) for s in split_sentences(block)]
    return out


def chunk(sentences: list[Sentence], title: str, size: int = 1200, overlap: int = 250) -> list[dict]:
    chunks, i, n = [], 0, 0
    while i < len(sentences):
        start, text, picked = i, "", []
        while i < len(sentences) and len(text) < size:
            text = f"{text} {sentences[i].text}".strip()
            picked.append(sentences[i])
            i += 1
        section = Counter(s.section for s in picked).most_common(1)[0][0]
        chapter = Counter(s.chapter for s in picked).most_common(1)[0][0]
        pages = [s.page for s in picked]
        n += 1
        chunks.append({
            "chunk_id": f"chunk_{n:04d}",
            "text": f"Document: {title} | Section: {section} | Chapter: {chapter}\n\n{text}",
            "raw_text": text,
            "metadata": {"title": title, "section": section, "chapter": chapter,
                         "start_page": min(pages), "end_page": max(pages),
                         "char_count": len(text), "word_count": len(text.split())},
        })
        # step back far enough to re-use ~`overlap` characters, but always move forward
        back, acc = 0, 0
        while i - back > start + 1 and acc < overlap:
            back += 1
            acc += len(sentences[i - back].text) + 1
        i = max(i - back, start + 1) if i < len(sentences) else i
    return chunks
