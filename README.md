# Book RAG Toolkit

**Turn the business books you own into a searchable knowledge base, then use it to brief an LLM with real frameworks instead of generic advice.**

```
PDF ──convert──▶ Markdown ──chunk──▶ context-tagged chunks ──search──▶ top passages ──generate──▶ framework prompt / LLM answer
```

No vector database, no embeddings bill. Retrieval is a dependency-free BM25 implementation that is fast enough for a shelf of books on a laptop.

## Why it works better than "split every 1,000 characters"

A passage is only useful if the model knows where it came from. The chunker:

- walks the converted book **page by page** (form feeds from the PDF conversion are kept as page breaks),
- tracks the current **section and chapter** from heading lines (Markdown `#`, `CHAPTER …`, `1. Title`, or your own list or regex),
- streams **sentences** (not characters) into ~1,200-character chunks with ~250 characters of overlap, so no idea is cut in half,
- stamps every chunk with its **dominant section, chapter and page range**, and prepends `Document: … | Section: … | Chapter: …` to the searchable text.

BM25 indexes the title, section and chapter together with the body, so a query like `pricing guarantee` also surfaces passages from a chapter *named* "Pricing".

## Quick start

```bash
pip install -r requirements.txt

python -m book_rag convert my_books/ --out books_md
python -m book_rag chunk books_md/influence.md --title "Influence by Robert Cialdini" \
    --chapters chapters.txt --strip "Some Watermark Text"
python -m book_rag search "reciprocity free value first" --limit 3
python -m book_rag generate --framework offer \
    --product "digital loyalty cards in Apple Wallet" --audience "independent coffee shops" --details "29 EUR/month"
```

Try it right away on the bundled sample (an original 3-page text, no copyrighted content):

```bash
python -m book_rag chunk examples/sample_book.md --title "A Small Guide to Loyal Customers" --size 300
python -m book_rag search "guarantee risk"
```

```
[1] 0.61  A Small Guide to Loyal Customers · CHAPTER 2: PRICING WITHOUT DISCOUNTS · p.2-3
Instead, raise the value of the offer. Add a bonus, shorten the wait, or remove a risk with a clear guarantee. …
```

## Frameworks

`generate` retrieves the passages that matter for a framework and fills a structured brief:

| framework | retrieval query | asks the model for |
|---|---|---|
| `offer` | value equation, pricing, guarantees, bonuses | core offer, pricing, value equation, bonuses, guarantee, scarcity |
| `outreach` | warm/cold outreach, lead magnets | lead magnet, warm and cold scripts, scaling plan |
| `viral` | STEPPS: social currency, triggers, emotion, stories | hooks for each STEPPS lever |
| `persuasion` | Cialdini's principles | copy sections for each principle |

With `GEMINI_API_KEY` or `OPENAI_API_KEY` set it calls the model (override with `LLM_MODEL`). Without a key it writes `prompt_<framework>.txt`, ready to paste into any chat model.

## Layout

```
book_rag/convert.py   MarkItDown wrapper (PDF/DOCX/EPUB/HTML -> Markdown)
book_rag/chunk.py     structure-aware sentence chunking with page/section/chapter metadata
book_rag/search.py    BM25 (k1=1.5, b=0.75), metadata-aware
book_rag/generate.py  framework prompts + optional LLM call
book_rag/cli.py       convert | chunk | search | generate
tests/                chunking, overlap, watermark stripping, ranking, prompt assembly
```

```bash
pytest -q
```

## Use your own copies

Only index books you have the right to use. This repository contains no book text, and chunk files and converted books are git-ignored.

## License

MIT
