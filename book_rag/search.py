"""Dependency-free BM25 search over chunk JSON files (k1=1.5, b=0.75).

Section, chapter and title are indexed together with the text, so a query like
"pricing guarantee" also matches chunks from a chapter *named* Pricing.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9ăâîșț]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class ChunkSearch:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf, self.lengths, self.df = [], [], {}
        for c in chunks:
            meta = c.get("metadata", {})
            tokens = tokenize(f"{c.get('raw_text', c.get('text', ''))} {meta.get('title', '')} "
                              f"{meta.get('section', '')} {meta.get('chapter', '')}")
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
            self.tf.append(counts)
            self.lengths.append(len(tokens))
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))

    @classmethod
    def from_dir(cls, chunks_dir: Path) -> "ChunkSearch":
        chunks = []
        for f in sorted(chunks_dir.glob("*.json")):
            for c in json.loads(f.read_text(encoding="utf-8")):
                c.setdefault("source_file", f.name)
                chunks.append(c)
        return cls(chunks)

    def search(self, query: str, limit: int = 5, title_filter: str | None = None) -> list[tuple[float, dict]]:
        q, n, scored = tokenize(query), len(self.chunks), []
        for idx, c in enumerate(self.chunks):
            if title_filter and title_filter.lower() not in (c.get("metadata", {}).get("title", "") + c.get("source_file", "")).lower():
                continue
            tf, norm, score = self.tf[idx], 1 - self.b + self.b * self.lengths[idx] / self.avg_len, 0.0
            for t in q:
                if t in tf:
                    idf = math.log((n - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0)
                    score += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * norm)
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]
