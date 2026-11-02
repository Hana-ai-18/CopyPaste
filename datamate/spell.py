"""Conservative spelling correction for question tokens (runs inside qnorm.normalize_question, after Telex/slang rules).

Kinds of mistakes handled, each with its own guard:
 1. tone / diacritic / missing-letter errors in an ACCENTED token that is not a valid Vietnamese syllable (đươc->được, nhât->nhất, bìh->bình, tổg->tổng)
 2. Gen Z initial swaps on accented tokens (zô->vô, júp->giúp, wán->quán, dzạ->dạ)
 3. glued words (nhiềuđơn, caonhất, ngàynào)
 4. typos in an unaccented token that is not a word (nhug, bnhiu are handled elsewhere): corrected only to the unaccented spelling
Nothing is changed when the token is a known word, a shop word, a valid syllable, a mid-sentence capitalised name, or when two candidates are
about equally likely (precision first: a wrong "fix" costs more than leaving the typo for the model to read).
"""
from __future__ import annotations

import math
import unicodedata

from .textutil import strip_accents

_TONES = {"̀", "́", "̉", "̃", "̣"}
_MIN_COUNT = 3
# very common colloquial / regional / typo spellings of ONE clear word (they occur in the material itself, so they look like "known words")
COMMON = {"kòn": "còn", "kủa": "của", "kua": "của", "ngừi": "người", "nhứt": "nhất", "zị": "vậy", "zậy": "vậy", "dzậy": "vậy", "dzị": "vậy", "dzạ": "dạ", "zạ": "dạ", "zô": "vô", "dzô": "vô", "zìa": "về", "dzìa": "về",
          "đươc": "được", "đuoc": "được", "đưoc": "được", "nhìu": "nhiều", "nhiu": "nhiều", "bnhiêu": "bao nhiêu", "ngày mô": "ngày nào"}
_RATIO = 2.0


def _toneless(w: str) -> str:
    """Remove tone marks only (keeps ă â ê ô ơ ư đ)."""
    d = unicodedata.normalize("NFD", w)
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in _TONES))


def _tone_of(w: str) -> str:
    for c in unicodedata.normalize("NFD", w):
        if c in _TONES:
            return c
    return ""


def _within1(a: str, b: str) -> bool:
    """Damerau edit distance <= 1 (insert / delete / substitute / adjacent swap)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _valid(w: str) -> bool:
    from .qnorm import split_tone, valid_syllable

    b, t = split_tone(w)
    return valid_syllable(b, t)


class Speller:
    def __init__(self, lex):
        self.lex = lex
        self.forms: dict[str, int] = {}                    # accented/plain form -> count
        for key, cnt in lex.words.items():
            for f, c in cnt.items():
                self.forms[f] = self.forms.get(f, 0) + c
        self.by_len: dict[int, list[str]] = {}
        for f in self.forms:
            if f.isalpha() and (_valid(f) or strip_accents(f) == f):
                self.by_len.setdefault(len(_toneless(f)), []).append(f)
        self.keys = set(lex.words)

    # ---- helpers -----------------------------------------------------------
    def count(self, w: str) -> int:
        return self.forms.get(w, 0)

    def _pick(self, scored: list[tuple[float, str]], ratio: float = _RATIO) -> str | None:
        if not scored:
            return None
        best: dict[str, float] = {}
        for s, w in scored:
            best[w] = max(best.get(w, 0), s)
        ranked = sorted(best.items(), key=lambda x: -x[1])
        top_w, top = ranked[0]
        if self.count(top_w) < _MIN_COUNT:
            return None
        if len(ranked) > 1 and top < ratio * ranked[1][1]:
            return None
        return top_w

    # ---- 1. accented, invalid syllable ---------------------------------------
    def fix_accented(self, w: str, strict: bool = False) -> str | None:
        tl = _toneless(w)
        tone = _tone_of(w)
        scored = []
        for L in (len(tl) - 1, len(tl), len(tl) + 1):
            for c in self.by_len.get(L, ()):
                if not _valid(c) or c == w:
                    continue
                if not _within1(_toneless(c), tl):
                    continue
                n = self.forms[c]
                if strict and (n < 30 or not tone or _tone_of(c) != tone):
                    continue
                same_tone = 1 if tone and _tone_of(c) == tone else 0
                exact_base = 1 if _toneless(c) == tl else 0
                exact_ins = 1 if len(c) != len(w) and _within1(c, w) else 0          # c is w with ONE letter added/removed, all marks identical
                scored.append((n * (1 + 3 * same_tone) * (1 + 2 * exact_base) * (1 + 60 * exact_ins), c))
        return self._pick(scored, 3.0 if strict else _RATIO)

    # ---- 2. Gen Z initial swaps on accented tokens --------------------------
    _SWAPS = [("dz", ["d", "v", "gi"]), ("z", ["d", "v", "gi", "r"]), ("j", ["gi"]), ("w", ["qu"]), ("f", ["ph"])]

    def swap_accented(self, w: str) -> str | None:
        for a, reps in self._SWAPS:
            if w.startswith(a):
                cands = []
                for r in reps:
                    c = r + w[len(a):]
                    if _valid(c) and self.count(c) >= _MIN_COUNT:
                        cands.append((self.count(c), c))
                return self._pick(cands)
        return None

    # ---- 3. glued words ------------------------------------------------------
    def split_glued(self, w: str) -> str | None:
        if len(w) < 6:
            return None
        best: list[tuple[float, str]] = []
        for i in range(2, len(w) - 1):
            l, r = w[:i], w[i:]
            if not (_valid(l) and _valid(r)):
                continue
            cl, cr = self.count(l), self.count(r)
            if cl < 10 or cr < 10:
                continue
            best.append((math.log(cl) + math.log(cr), f"{l} {r}"))
        if not best:
            return None
        best.sort(reverse=True)
        if len(best) > 1 and best[0][0] - best[1][0] < 1.0:
            return None
        return best[0][1]

    # ---- 4. unaccented typos ---------------------------------------------------
    def fix_plain(self, w: str) -> str | None:
        if len(w) < 4 or w in self.keys:
            return None
        cands = []
        for k in self.keys:
            if abs(len(k) - len(w)) > 1 or not _within1(w, k) or not _valid(k):
                continue
            if len(w) < 5 and len(k) == len(w):                       # short tokens: substitutions are too risky
                continue
            cands.append((sum(self.lex.words[k].values()), k))
        return self._pick([(c, k) for c, k in cands]) if cands else None


_SPELLER: dict[int, Speller] = {}


def speller(lex) -> Speller:
    s = _SPELLER.get(id(lex))
    if s is None:
        _SPELLER.clear()
        s = _SPELLER[id(lex)] = Speller(lex)
    return s


def correct_token(tok: str, lex, known: set[str]) -> str | None:
    """Return a corrected spelling of `tok` or None. `known` = every word that must be left alone (vocab, shop words, protected)."""
    t = unicodedata.normalize("NFC", tok.lower())
    if len(t) < 2 or not t.isalpha():
        return None
    if t in COMMON:
        return COMMON[t]
    if t in known:
        return None
    if t in COMMON:
        return COMMON[t]
    sp = speller(lex)
    accented = strip_accents(t) != t
    swapped = t[0] in "zjwf" or t.startswith("dz")
    if sp.count(t) >= _MIN_COUNT and _valid(t):
        return None
    if accented:
        if swapped:
            return sp.swap_accented(t)                              # never fall back to other edits: 'zô' is vô, not 'số'
        if not _valid(t):
            return sp.fix_accented(t) or sp.split_glued(t)
        return None
    if _valid(t) and not swapped:
        return None                                                  # a plausible unaccented syllable: names, words
    if swapped:
        out = sp.swap_accented(t)
        return strip_accents(out) if out else None
    return sp.split_glued(t) or sp.fix_plain(t)
