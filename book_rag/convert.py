"""PDF (or DOCX/EPUB/HTML) -> Markdown with Microsoft MarkItDown, keeping form feeds as page breaks."""
from __future__ import annotations

from pathlib import Path


def convert(src: Path, out_dir: Path) -> list[Path]:
    from markitdown import MarkItDown

    md = MarkItDown()
    files = sorted(src.glob("*.pdf")) if src.is_dir() else [src]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in files:
        target = out_dir / f"{f.stem}.md"
        target.write_text(md.convert(str(f)).text_content, encoding="utf-8")
        written.append(target)
        print(f"{f.name} -> {target}")
    return written
