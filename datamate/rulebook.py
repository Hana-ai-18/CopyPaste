"""Rulebook handling: keep the general rules + only the industry section(s) that match the shop, so that the long real
rulebook (~24k chars) fits the prompt. The rulebook of the split is the source of truth (it differs between splits and
contains industries that are not in the train rulebook) - nothing here hard-codes an answer."""
from __future__ import annotations

import re

from .textutil import strip_accents

# industry heading -> keywords looked for in the shop README (accent-stripped, lower case)
INDUSTRY_KEYWORDS = {
    "ca phe": ["ca phe", "cafe", "quan cafe", "tra sua", "pha che", "topping", "size m", "size l", "(m)", "(l)", "phi phuc vu"],
    "nha thuoc": ["nha thuoc", "hoat chat", "ke don", "han dung", "han su dung", "thuoc ke don", "danhsachthuoc", "danh sach thuoc"],
    "vat lieu xay dung": ["vat lieu", "xay dung", "giao nhieu dot", "dot giao", "xi mang", "gach", "cat xay", "thep"],
    "me va be": ["me va be", "bim", "sua bot", "diem thanh vien", "voucher", "ngay dang ky"],
    "sua xe": ["sua xe", "luot xe", "bien so", "phieu bao hanh", "lam gi"],
    "cua hang dien thoai": ["cua hang dien thoai", "imei", "tra gop", "may cu", "iphone", "samsung", "thu may cu"],
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


_WORD = re.compile(r"[a-z0-9]{4,}")
_ALWAYS = ("khong tra loi", "cach ghi", "dap an va cham", "tu ngu")      # chapters that are never pruned (format, unanswerable, reading rules)


def _words(s: str) -> set[str]:
    return set(_WORD.findall(_key(s)))


def _prune_rows(blocks: list[str], evidence: str, max_chars: int, fixed: int) -> list[str]:
    """Drop whole table rows (never characters) with the least overlap with the shop's own evidence (README, column names, ITS questions)
    until the rulebook fits. Headings, table headers and the protected chapters are untouched."""
    ev = _words(evidence)
    rows = []                                    # (score, -len, block_idx, line_idx)
    split = [b.splitlines(keepends=True) for b in blocks]
    for bi, lines in enumerate(split):
        head = _key(lines[0]) if lines else ""
        if any(a in head for a in _ALWAYS):
            continue
        for li, ln in enumerate(lines):
            if ln.startswith("|") and not re.match(r"^\|\s*(-|:)", ln) and li >= 2:     # data rows only (not header + separator)
                rows.append((len(_words(ln) & ev), -len(ln), bi, li))
    total = sum(len(b) for b in blocks) + fixed
    drop = set()
    for sc, nl, bi, li in sorted(rows):
        if total <= max_chars:
            break
        drop.add((bi, li))
        total -= -nl
    return ["".join(ln for li, ln in enumerate(lines) if (bi, li) not in drop) for bi, lines in enumerate(split)]


def select_rulebook(text: str, readme: str, max_chars: int = 16000, evidence: str = "") -> str:
    """Fit the real rulebook into the prompt WITHOUT cutting it blindly: (1) keep only the industry chapters that match the shop
    (README + table/column names + the shop's questions = `evidence`), (2) if still too long drop the least relevant table rows.
    The chapters on unanswerable questions, how to read the files and the answer format are always kept whole."""
    if not text or len(text) <= max_chars:
        return text
    before, industries, after = split_rulebook(text)
    if not industries:
        return text[:max_chars]
    rk = _key(readme[:3000]) + " " + _key(evidence[:20000]) + " "
    scores = {}
    for name in industries:
        k = _key(name)
        kws = next((v for key, v in INDUSTRY_KEYWORDS.items() if key in k), [k])
        scores[name] = sum(rk.count(w) for w in kws) + (5 if k in rk else 0)
    best = max(scores.values(), default=0)
    if best < 4:   # nothing says what kind of shop it is -> never guess, keep every industry
        best = 0
    keep = [n for n, sc in scores.items() if best > 0 and sc >= max(1, best * 0.4)] or list(industries)
    ind_txt = "".join(industries[n] for n in industries if n in keep)
    out = "".join(before) + ind_txt + "".join(after)
    if len(out) > max_chars and best > 0 and len(keep) > 1:      # still too long: only the single best industry
        top = max(scores, key=scores.get)
        ind_txt = industries[top]
        out = "".join(before) + ind_txt + "".join(after)
    if len(out) > max_chars:
        fixed = len(ind_txt) + sum(len(a) for a in after)
        before = _prune_rows(before, readme + " " + evidence, max_chars, fixed)
        out = "".join(before) + ind_txt + "".join(after)
    if len(out) > max_chars:                                     # extreme budget: keep both ends (format chapters are at the end)
        out = out[: max_chars // 3] + "\n...\n" + out[-(max_chars * 2 // 3):]
    return out
