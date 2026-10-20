"""book-rag: convert | chunk | search | generate"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(prog="book-rag", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="PDF(s) -> Markdown")
    c.add_argument("src", type=Path)
    c.add_argument("--out", type=Path, default=Path("books_md"))

    k = sub.add_parser("chunk", help="Markdown -> context-tagged chunks JSON")
    k.add_argument("markdown", type=Path)
    k.add_argument("--title", required=True, help='e.g. "Influence by Robert Cialdini"')
    k.add_argument("--sections", type=Path, help="text file, one top-level section heading per line")
    k.add_argument("--chapters", type=Path, help="text file, one chapter heading per line")
    k.add_argument("--heading-regex", help="regex for chapter heading lines")
    k.add_argument("--strip", action="append", default=[], help="watermark/boilerplate text to remove (repeatable)")
    k.add_argument("--size", type=int, default=1200)
    k.add_argument("--overlap", type=int, default=250)
    k.add_argument("--out", type=Path, default=Path("rag_chunks"))

    s = sub.add_parser("search", help="BM25 search over chunk files")
    s.add_argument("query")
    s.add_argument("--chunks", type=Path, default=Path("rag_chunks"))
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--book", help="filter by title substring")

    g = sub.add_parser("generate", help="retrieve passages and brief an LLM with a marketing framework")
    g.add_argument("--framework", default="offer")
    g.add_argument("--product", required=True)
    g.add_argument("--audience", required=True)
    g.add_argument("--details", default="")
    g.add_argument("--chunks", type=Path, default=Path("rag_chunks"))
    g.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    if args.cmd == "convert":
        from .convert import convert
        convert(args.src, args.out)

    elif args.cmd == "chunk":
        from .chunk import DEFAULT_HEADING, chunk, sentence_stream
        lines = lambda p: [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p else []
        sents = sentence_stream(args.markdown.read_text(encoding="utf-8"), lines(args.sections), lines(args.chapters),
                                re.compile(args.heading_regex) if args.heading_regex else DEFAULT_HEADING, args.strip)
        chunks = chunk(sents, args.title, args.size, args.overlap)
        args.out.mkdir(parents=True, exist_ok=True)
        out = args.out / f"{args.markdown.stem}.json"
        out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(sents)} sentences -> {len(chunks)} chunks -> {out}")

    elif args.cmd == "search":
        from .search import ChunkSearch
        for rank, (score, ch) in enumerate(ChunkSearch.from_dir(args.chunks).search(args.query, args.limit, args.book), 1):
            m = ch["metadata"]
            print(f"\n[{rank}] {score:.2f}  {m['title']} · {m['chapter']} · p.{m['start_page']}-{m['end_page']}\n{ch['raw_text'][:600]}")

    else:
        from .generate import FRAMEWORKS, build_prompt, call_llm
        from .search import ChunkSearch
        if args.framework not in FRAMEWORKS:
            sys.exit(f"framework must be one of: {', '.join(FRAMEWORKS)}")
        system, user = build_prompt(ChunkSearch.from_dir(args.chunks), args.framework, args.product, args.audience, args.details, args.k)
        answer = call_llm(system, user)
        if answer:
            print(answer)
        else:
            out = Path(f"prompt_{args.framework}.txt")
            out.write_text(f"SYSTEM:\n{system}\n\nUSER:\n{user}", encoding="utf-8")
            print(f"No GEMINI_API_KEY / OPENAI_API_KEY set. Compiled prompt with retrieved passages -> {out}")


if __name__ == "__main__":
    main()
