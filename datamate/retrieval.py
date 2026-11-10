"""Few-shot exemplar retrieval from a bank built ONLY from training data (see scripts/build_exemplars.py)."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .textutil import vi_tokens


class ExemplarBank:
    def __init__(self, items: list[dict]):
        self.items = items
        self.tokens = [Counter(vi_tokens(it["question"])) for it in items]
        df = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = max(len(items), 1)
        self.idf = {w: math.log(1 + n / (1 + c)) for w, c in df.items()}

    @classmethod
    def load(cls, path: str | Path | None, extra: str | Path | None = None) -> "ExemplarBank":
        """`path`: the bank built from TRAIN (may be absent). `extra`: hand-written seed examples (idioms of the helpers), always available."""
        items: list[dict] = []
        for p in (extra, path):
            if p and str(p).lower() != "none" and Path(p).exists():
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))
        return cls(items)

    def __len__(self) -> int:
        return len(self.items)

    def search(self, question: str, k: int = 3) -> list[dict]:
        if not self.items or k <= 0:
            return []
        q = Counter(vi_tokens(question))
        scored = []
        for i, t in enumerate(self.tokens):
            common = set(q) & set(t)
            if not common:
                continue
            num = sum(self.idf.get(w, 1.0) for w in common)
            den = math.sqrt(sum(self.idf.get(w, 1.0) for w in q) * sum(self.idf.get(w, 1.0) for w in t)) or 1.0
            scored.append((num / den, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out, seen = [], set()
        for _, i in scored:
            key = self.items[i]["question"]
            if key in seen:
                continue
            seen.add(key)
            out.append(self.items[i])
            if len(out) >= k:
                break
        return out
