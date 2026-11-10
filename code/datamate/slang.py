"""Gen Z / chat-style Vietnamese: curated slang plus GENERIC repair rules, so words never seen before are still handled.

Generic rules (each only fires on a token that is NOT a known word of the material and is validated against the vocabulary):
  1. letter stretching        'nhieeeeu', 'hnayyyy'             -> collapse repeated letters
  2. letter swaps             'zô' 'fải' 'jữ' 'wa'              -> z/f/j/w rewritten to d|gi|v / ph / gi|g / qu and looked up
  3. initial + skeleton       'dthu' 'bnhiu' 'trc' 'hnay'       -> matched to a multi-word phrase of the material (first letter of every
                                                                  syllable kept, other letters optional, order kept)
  4. one-letter typos         'tuaan' 'khachh'                  -> the unique known word at edit distance 1
The result is advisory ("Reading: ..."): the original question always stays in the prompt."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from .textutil import strip_accents

PHRASE_FILE = Path(__file__).with_name("vi_phrases.txt")

# curated slang / shorthand (lower-case token -> reading). Only items that are unambiguous inside a question about a shop.
GENZ = {
    # questions / fillers that carry meaning
    "ko": "không", "k": "không", "hok": "không", "hông": "không", "hong": "không", "hem": "không", "khum": "không", "khong": "không", "kh0ng": "không",
    "hổng": "không", "hk": "không", "kp": "không phải", "kg": "kg", "ko0": "không", "kô": "không", "k0": "không",
    "j": "gì", "gj": "gì", "zì": "gì", "jì": "gì", "gì": "gì", "z": "vậy", "zậy": "vậy", "zay": "vậy", "zạ": "vậy", "dậy": "vậy", "dzậy": "vậy", "vại": "vậy",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được", "đx": "được", "dzô": "vô", "zô": "vô", "zào": "vào", "zui": "vui",
    "bít": "biết", "bik": "biết", "biet": "biết", "bjt": "biết", "wa": "quá", "wá": "quá", "qá": "quá", "wen": "quen", "wan": "quan",
    "cx": "cũng", "cũg": "cũng", "đg": "đang", "dg": "đang", "đag": "đang", "r": "rồi", "rùi": "rồi", "rui": "rồi", "ròi": "rồi", "roài": "rồi",
    "lun": "luôn", "luon": "luôn", "lunn": "luôn", "nx": "nữa", "nữ": "nữa", "thui": "thôi", "thoi": "thôi", "thuj": "thôi", "nhẹ": "nhé",
    "ntn": "như thế nào", "nthe": "như thế nào", "ntnao": "như thế nào", "tn": "thế nào", "sao": "sao", "vs": "với", "vơi": "với", "vsơ": "với",
    "mik": "mình", "mìk": "mình", "mk": "mình", "mih": "mình", "minh": "mình", "tui": "tôi", "tuii": "tôi", "t": "tôi", "mình": "mình", "e": "em", "a": "anh",
    "ngta": "người ta", "ng": "người", "ngta": "người ta", "mn": "mọi người", "ae": "anh em", "ad": "admin", "bro": "anh", "sis": "chị",
    "trc": "trước", "truoc": "trước", "trk": "trước", "sau": "sau", "nay": "nay", "bữa": "bữa", "hnay": "hôm nay", "hqua": "hôm qua", "hqa": "hôm qua",
    "hnai": "hôm nay", "hum nay": "hôm nay", "hum qua": "hôm qua", "humqua": "hôm qua", "bnay": "bữa nay", "bqua": "bữa qua", "mai": "mai", "mốt": "mốt",
    "tuần r": "tuần rồi", "tgian": "thời gian", "tg": "thời gian", "tgian": "thời gian",
    # business words
    "dthu": "doanh thu", "dt": "doanh thu", "doanhthu": "doanh thu", "ds": "doanh số", "doanh số": "doanh thu", "dsố": "doanh số", "rev": "doanh thu", "revenue": "doanh thu",
    "bnhiu": "bao nhiêu", "bnhieu": "bao nhiêu", "bnh": "bao nhiêu", "bn": "bao nhiêu", "bnhiêu": "bao nhiêu", "bao nhiu": "bao nhiêu", "nhiu": "nhiêu",
    "nhìu": "nhiều", "nhiuu": "nhiều", "nhju": "nhiều", "nhiu nhat": "nhiều nhất", "nhất": "nhất", "nhat": "nhất", "nhìu": "nhiều",
    "sp": "sản phẩm", "spham": "sản phẩm", "kh": "khách hàng", "khach": "khách", "khg": "khách hàng", "nv": "nhân viên", "ncc": "nhà cung cấp",
    "dh": "đơn hàng", "đh": "đơn hàng", "hd": "hóa đơn", "hđ": "hóa đơn", "bill": "hóa đơn", "sl": "số lượng", "slg": "số lượng", "tt": "thanh toán",
    "ck": "chuyển khoản", "ctk": "chuyển khoản", "tm": "tiền mặt", "tmat": "tiền mặt", "cod": "cod", "tk": "tài khoản", "gd": "giao dịch",
    "cty": "công ty", "ctr": "chương trình", "lq": "liên quan", "kq": "kết quả", "sdt": "số điện thoại", "đt": "điện thoại", "dtdd": "điện thoại",
    "nl": "nguyên liệu", "vd": "ví dụ", "bt": "bình thường", "bthg": "bình thường", "tl": "trả lời", "ns": "nói", "wtf": "gì vậy", "ib": "nhắn tin",
    "tr": "triệu", "trieu": "triệu", "củ": "triệu", "lít": "lít", "oke": "ok", "okie": "ok", "oki": "ok", "okela": "ok", "ô kê": "ok",
    "đỉnh": "cao nhất", "xịn": "tốt", "ế": "bán không được", "chốt": "bán được", "chốt đơn": "đơn bán thành công", "sale": "khuyến mãi", "flash sale": "khuyến mãi",
    "freeship": "miễn phí vận chuyển", "ship": "phí vận chuyển", "off": "giảm", "sold": "đã bán", "best seller": "bán chạy", "bestseller": "bán chạy",
    "top": "top", "hot": "bán chạy", "trend": "xu hướng", "check": "xem", "coi": "xem", "chek": "xem", "xem": "xem", "cho hỏi": "cho hỏi", "cho xin": "cho xin",
    "ơi": "", "nha": "", "nhen": "", "nhé": "", "nè": "", "nhỉ": "", "á": "", "ạ": "", "dạ": "", "ha": "", "hen": "", "hén": "", "ná": "", "nào": "nào", "ik": "", "đi": "",
}
_FILLERS = {k for k, v in GENZ.items() if v == "" and k not in ("đi",)}
_FILLERS = _FILLERS - {"nào"}
GENZ = {k: v for k, v in GENZ.items() if v != ""}                         # fillers are left in place (harmless); only meaningful items are rewritten
# items that are real words in another reading must NOT be rewritten when they stand alone as ordinary Vietnamese
_RISKY_SINGLE = {"a", "e", "t", "r", "minh", "nữ", "nhẹ", "sau", "nay", "mai", "bữa", "mốt", "sao", "top", "lít", "đi", "nhất", "nhat", "gì", "mình", "ds", "rev", "tn", "kp", "off", "ship",
                 "hot", "sale", "check", "coi", "xem", "bill", "củ", "tr", "dt"}
GENZ_SAFE = {k: v for k, v in GENZ.items() if k not in _RISKY_SINGLE and k != v}


def collapse_repeats(tok: str) -> str:
    """'nhieeeeu' -> 'nhieeu' (three or more equal letters become two: Telex needs 'ee', 'oo', 'aa'); 'hnayyyy' -> 'hnayy' -> 'hnay'."""
    t = re.sub(r"(.)\1{2,}", r"\1\1", tok)
    return t


def _dedupe_double(tok: str) -> str:
    """Also try single letters for a double that Telex does not need: 'hnayy' -> 'hnay'."""
    return re.sub(r"([^aeoud])\1", r"\1", tok)


_SWAPS = [("z", ["d", "gi", "v"]), ("f", ["ph"]), ("j", ["gi", "g"]), ("w", ["qu"]), ("dz", ["d", "v", "gi"])]


def swap_variants(tok: str) -> list[str]:
    out = []
    for src, dst in _SWAPS:
        if tok.startswith(src):
            out += [d + tok[len(src):] for d in dst]
    return out


def is_subsequence(part: str, syl: str) -> bool:
    it = iter(syl)
    return all(c in it for c in part)


def skeleton_match(tok: str, phrase_syl: list[str]) -> bool:
    """tok is built from non-empty pieces, one per syllable, in order; each piece starts with the syllable's first letter and is a subsequence of it."""
    k = len(phrase_syl)
    if k != 2 or len(tok) < 4 or len(re.findall(r"[aeiouy]+", tok)) > 1:        # real shorthand keeps few vowels: 'dthu', 'bnhiu', 'hnay'
        return False
    if tok == "".join(phrase_syl):                               # just a missing space ('saigon'): not shorthand
        return False

    def rec(pos: int, i: int) -> bool:
        if i == k:
            return pos == len(tok)
        s = phrase_syl[i]
        if pos >= len(tok) or tok[pos] != s[0]:
            return False
        if i == k - 1:
            return is_subsequence(tok[pos:], s)
        for end in range(pos + 1, len(tok) - (k - i - 1) + 1):
            if not is_subsequence(tok[pos:end], s):
                break
            if rec(end, i + 1):
                return True
        return False

    return rec(0, 0)


class Lexicon:
    """Phrases (2-4 syllables) and a stripped-word index built from the material."""

    def __init__(self):
        self.phrases: dict[tuple[str, ...], tuple[str, int]] = {}      # stripped syllables -> (accented phrase, count)
        self.words: dict[str, Counter] = {}                           # stripped word -> Counter(accented forms)

    def add_phrase(self, phrase: str, count: int = 1) -> None:
        syl = phrase.split()
        if not 2 <= len(syl) <= 4:
            return
        key = tuple(strip_accents(s).lower() for s in syl)
        if not all(k.isalpha() for k in key):
            return
        old = self.phrases.get(key)
        self.phrases[key] = (phrase, (old[1] if old else 0) + count)

    def add_word(self, word: str, count: int = 1) -> None:
        w = unicodedata.normalize("NFC", word.lower())
        if w.isalpha():
            self.words.setdefault(strip_accents(w).lower(), Counter())[w] += count

    def add_text(self, text: str) -> None:
        for line in re.split(r"[\n.;:!?()|]", unicodedata.normalize("NFC", text.lower())):
            toks = re.findall(r"[^\W\d_]+", line)
            for t in toks:
                self.add_word(t)
            for n in (2, 3):
                for i in range(len(toks) - n + 1):
                    self.add_phrase(" ".join(toks[i:i + n]))

    @classmethod
    def load(cls, path: Path = PHRASE_FILE) -> "Lexicon":
        lx = cls()
        vf = path.with_name("vi_vocab.txt")                     # single words: the vocabulary of the material
        if vf.exists():
            for line in vf.read_text(encoding="utf-8").splitlines():
                w, _, c = line.partition("\t")
                if w and len(w) >= 2:
                    lx.add_word(w, int(c or 1))
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                p, _, c = line.partition("\t")
                if p:
                    lx.add_phrase(p, int(c or 1))
                    for w in p.split():
                        lx.add_word(w, int(c or 1))
        return lx

    def best_word(self, stripped: str) -> str | None:
        c = self.words.get(stripped)
        return c.most_common(1)[0][0] if c else None

    def phrase_for(self, tok: str) -> str | None:
        best: list[tuple[int, str]] = []
        for key, (phrase, cnt) in self.phrases.items():
            if skeleton_match(tok, list(key)):
                best.append((cnt, phrase))
        if not best:
            return None
        best.sort(reverse=True)
        if len(best) > 1 and best[0][0] < 2 * best[1][0] and best[0][1] != best[1][1]:
            return None                                                  # ambiguous: never guess
        return best[0][1]

    def skeleton_word(self, tok: str) -> str | None:
        """'trc' -> 'trước', 'tbinh' style single words: first and last letter kept, others optional (3+ letters, shorter than the word)."""
        if len(tok) < 3 or len(tok) > 5 or any(ch in "aeiouyăâêôơư" for ch in tok):      # pure consonant skeletons only: 'trc', 'ntn'
            return None
        hits = []
        for w, c in self.words.items():
            if len(w) > len(tok) and len(w) >= 4 and w[0] == tok[0] and w[-1] == tok[-1] and is_subsequence(tok, w):
                hits.append((sum(c.values()), w))
        if not hits:
            return None
        hits.sort(reverse=True)
        if len(hits) > 1 and hits[0][0] < 2 * hits[1][0]:
            return None
        return self.best_word(hits[0][1])

    def fuzzy_word(self, tok: str) -> str | None:
        """Unique known word at edit distance 1 (substitution / insertion / deletion), tokens of 5+ letters only."""
        if len(tok) < 5:
            return None
        cands = []
        for w, c in self.words.items():
            if abs(len(w) - len(tok)) > 1 or len(w) < 4:
                continue
            if len(tok) == 5 and len(w) == len(tok):
                continue                                   # a one-letter substitution inside a 5-letter token is too ambiguous
            if _edit1(tok, w):
                cands.append((sum(c.values()), w))
        if not cands:
            return None
        cands.sort(reverse=True)
        if len(cands) > 1 and cands[0][0] < 2 * cands[1][0]:
            return None
        return self.best_word(cands[0][1])


def _edit1(a: str, b: str) -> bool:
    if a == b:
        return False
    la, lb = len(a), len(b)
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


_LEX: Lexicon | None = None


def default_lexicon() -> Lexicon:
    global _LEX
    if _LEX is None:
        _LEX = Lexicon.load()
    return _LEX


def repair_token(tok: str, lex: Lexicon, vocab_plain: set[str]) -> str | None:
    """Generic repair of a token that is not a known word. Returns a replacement or None."""
    t = tok.lower()
    if not t.isalpha() or len(t) < 2 or t in vocab_plain:
        return None
    if strip_accents(t).lower() != t:                            # carries Vietnamese diacritics: properly typed (names, words) -> leave it
        return None
    t = t
    if t in GENZ_SAFE:
        return GENZ_SAFE[t]
    if len(t) < 3:
        return None
    if t in lex.words:
        return None                                              # an ordinary accent-less word: the model reads that fine
    from .qnorm import _split_onset

    if _split_onset(t) is not None:                              # a possible Vietnamese syllable ('linh', 'vinh', 'chanh'): a name / word, not shorthand
        return None
    c = collapse_repeats(t)
    forms = [t]
    for x in (c, _dedupe_double(c), re.sub(r"(.)\1+", r"\1", t)):      # stretched letters: 3+ -> 2, doubles -> 1, all repeats -> 1
        if x not in forms:
            forms.append(x)
    for x in (forms[1:] if re.search(r"[aeiouy]", t) else []):    # stretched letters of a known word / slang ('nhieeeeu', 'khachh')
        if x in GENZ_SAFE:
            return GENZ_SAFE[x]
        if x in lex.words:
            return lex.best_word(x)
    for v in swap_variants(t):                                   # zui -> vui, fai -> phai, wa -> qua
        if v in lex.words and sum(lex.words[v].values()) >= 3:
            return lex.best_word(v)
    for x in forms:
        ph = lex.phrase_for(x)
        if ph:
            return ph
    return None
