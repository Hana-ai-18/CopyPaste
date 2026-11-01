"""Rulebook handling: keep the general rules + only the industry section(s) that match the shop, so that the long real
rulebook (~24k chars) fits the prompt. The rulebook of the split is the source of truth (it differs between splits and
contains industries that are not in the train rulebook) - nothing here hard-codes an answer."""
from __future__ import annotations

import re

from .textutil import strip_accents

# industry heading -> keywords looked for in the shop README (accent-stripped, lower case)
INDUSTRY_KEYWORDS = {
    "ca phe": ["ca phe", "quan cafe", "tra sua", "do uong pha che", "topping", "size m", "size l"],
    "nha thuoc": ["nha thuoc", "thuoc", "hoat chat", "ke don", "han dung", "lo hang"],
    "vat lieu xay dung": ["vat lieu", "xay dung", "giao nhieu dot", "dot giao", "xi mang", "gach"],
    "me va be": ["me va be", "bim", "ta ", "sua bot", "diem thanh vien", "voucher"],
    "sua xe": ["sua xe", "luot xe", "bien so", "tho ", "bao hanh"],
    "cua hang dien thoai": ["dien thoai", "imei", "tra gop", "may cu", "iphone"],
}


def _key(s: str) -> str:
    return strip_accents(s).lower()


def split_rulebook(text: str) -> tuple[list[str], dict[str, str], list[str]]:
    """-> (blocks before the industry chapter, {industry heading: text}, blocks after it). Falls back to ([text], {}, [])."""
    lines = text.splitlines(keepends=True)
    top: list[tuple[str, list[str]]] = []
    for ln in lines:
        if ln.startswith("## "):
            top.append((ln, [ln]))
        elif top:
            top[-1][1].append(ln)
        else:
            top.append(("", [ln]))
    ind_idx = next((i for i, (h, _) in enumerate(top) if _key(h).startswith("## 6") or "theo nganh" in _key(h)), None)
    if ind_idx is None:
        return [text], {}, []
    before = ["".join(b) for _, b in top[:ind_idx]]
    after = ["".join(b) for _, b in top[ind_idx + 1:]]
    industries: dict[str, str] = {}
    intro, cur = [top[ind_idx][1][0]], None
    for ln in top[ind_idx][1][1:]:
        if ln.startswith("### "):
            cur = ln[4:].strip()
            industries[cur] = ln
        elif cur is None:
            intro.append(ln)
        else:
            industries[cur] += ln
    before.append("".join(intro))
    return before, industries, after


def select_rulebook(text: str, readme: str, max_chars: int = 16000) -> str:
    """Drop the industry sections that do not match the shop (README keywords). If nothing matches, keep all of them."""
    if not text or len(text) <= max_chars:
        return text
    before, industries, after = split_rulebook(text)
    if not industries:
        return text[:max_chars]
    rk = _key(readme[:1500]) + " "   # the intro paragraph says what kind of shop it is
    scores = {}
    for name in industries:
        k = _key(name)
        kws = next((v for key, v in INDUSTRY_KEYWORDS.items() if key in k), [k])
        scores[name] = sum(rk.count(w) for w in kws) + (5 if k in rk else 0)
    best = max(scores.values(), default=0)
    if best < 3:   # README does not say what kind of shop it is -> never guess, keep every industry
        best = 0
    keep = [n for n, sc in scores.items() if best > 0 and sc >= max(1, best * 0.5)] or list(industries)
    out = "".join(before) + "".join(industries[n] for n in industries if n in keep) + "".join(after)
    if len(out) > max_chars and best > 0:  # still too long: keep only the single best industry
        top = max(scores, key=scores.get)
        out = "".join(before) + industries[top] + "".join(after)
    if len(out) > max_chars:
        out = out[: max_chars // 2] + "\n...\n" + out[-(max_chars // 2):]
    return out
