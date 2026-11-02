"""Question normaliser: expands chat shorthand and decodes raw Telex keystrokes ('tieenf' -> 'tiền', 'dduwowcj' -> 'được'), which the real
questions contain. Decoding is validated against a vocabulary of Vietnamese words that really occur in the provided material (questions,
rulebooks, READMEs, column names), so ordinary words, brands and product names are never 'repaired' by mistake."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

_TONE_MARKS = {"̀": "f", "́": "s", "̉": "r", "̃": "x", "̣": "j"}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
VOCAB_FILE = Path(__file__).with_name("vi_vocab.txt")


def split_tone(word: str) -> tuple[str, str]:
    """'tiền' -> ('tiên', 'f'): the word without its tone mark (modifiers ê â ă ô ơ ư đ are kept) and the Telex tone letter ('' if none)."""
    d = unicodedata.normalize("NFD", word.lower())
    tone = ""
    keep = []
    for ch in d:
        if ch in _TONE_MARKS:
            tone = _TONE_MARKS[ch]
        else:
            keep.append(ch)
    return unicodedata.normalize("NFC", "".join(keep)), tone


def _is_vi(word: str) -> bool:
    return unicodedata.normalize("NFC", word) != "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn").replace("đ", "d")


class Vocab:
    def __init__(self, counts: Counter | None = None):
        self.by_key: dict[tuple[str, str], Counter] = {}
        self.plain: set[str] = set()                    # every lower-case word seen as-is (protects brands / English / unaccented words)
        self.shop: set[str] = set()                     # words of THIS shop's own text (README, profile): they win over any shorthand rule
        if counts:
            self.add_counts(counts)

    def add_counts(self, counts: Counter) -> None:
        for w, c in counts.items():
            w = unicodedata.normalize("NFC", w.lower())
            self.plain.add(w)
            if _is_vi(w):
                self.by_key.setdefault(split_tone(w), Counter())[w] += c

    def add_text(self, text: str) -> None:
        words = _WORD.findall(unicodedata.normalize("NFC", text.lower()))
        self.shop.update(words)
        self.add_counts(Counter(words))

    def lookup(self, base: str, tone: str) -> str | None:
        c = self.by_key.get((base, tone))
        return c.most_common(1)[0][0] if c else None

    @classmethod
    def load(cls, path: Path = VOCAB_FILE) -> "Vocab":
        cnt: Counter = Counter()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                w, _, c = line.partition("\t")
                if w:
                    cnt[w] += int(c or 1)
        return cls(cnt)


_MODS = [("uwow", "ươ"), ("uow", "ươ"), ("dd", "đ"), ("aa", "â"), ("ee", "ê"), ("oo", "ô"), ("aw", "ă"), ("ow", "ơ"), ("uw", "ư")]


def telex_candidates(tok: str) -> list[tuple[str, str]]:
    """(base, tone) readings of a raw Telex token, most plausible first."""
    s = tok.lower()
    out = []
    forms = [(s[:-1], s[-1])] if len(s) > 2 and s[-1] in "sfrxj" else []
    forms.append((s, ""))
    for body, tone in forms:
        for a, b in _MODS:
            body = body.replace(a, b)
        if body != tok.lower() or tone:
            out.append((unicodedata.normalize("NFC", body), tone))
    return out



# ---------------------------------------------------------------- phonotactics (decode Telex words that are not in the vocabulary)
_ONSETS = ["ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr", "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x", ""]
_RIMES = set("""a ac ach ai am an ang anh ao ap at au ay ăc ăm ăn ăng ăp ăt âc âm ân âng âp ât âu ây e ec em en eng eo ep et ê êch êm ên ênh êp êt êu
i ia ich im in inh ip it iu iêc iêm iên iêng iêp iêt iêu yêm yên yêng yêt yêu o oa oac oach oai oan oang oanh oao oap oat oay oc oe oen oeo oet oi om on ong op ot
ô ôc ôi ôm ôn ông ôp ôt ơ ơi ơm ơn ơp ơt u ua uc ui um un ung up ut uy uya uych uyên uyêt uyn uynh uyp uyt uyu uân uât uây uê uêch uênh uôc uôi uôm uôn uông uôt
ư ưa ưc ưi ưng ưt ưu ươc ươi ươm ươn ương ươp ươt ươu y""".split())
_VOWELS = set("aăâeêiouôơưy")
_MODIFIED = set("ăâêôơư")
_TONE_CHAR = {"f": "\u0300", "s": "\u0301", "r": "\u0309", "x": "\u0303", "j": "\u0323"}


def _split_onset(base: str) -> tuple[str, str] | None:
    for o in _ONSETS:
        if base.startswith(o):
            r = base[len(o):]
            if r in _RIMES:
                return o, r
            if o == "gi" and ("i" + r) in _RIMES:      # 'gì', 'gìn': the i belongs to the rime
                return "g", "i" + r
    return None


def valid_syllable(base: str, tone: str) -> bool:
    sp = _split_onset(base)
    if not sp:
        return False
    rime = sp[1]
    if rime and rime[-1] in "cpt" and tone not in ("s", "j"):    # stopped syllables (c, ch, p, t) only take sắc / nặng
        return False
    return True


def place_tone(base: str, tone: str) -> str:
    if not tone:
        return base
    sp = _split_onset(base)
    if not sp:
        return base
    onset, rime = sp
    idx = [i for i, ch in enumerate(rime) if ch in _VOWELS]
    nuc = rime[idx[0]: idx[-1] + 1] if idx else ""
    first = idx[0]
    coda = rime[idx[-1] + 1:]
    if any(ch in _MODIFIED for ch in nuc):
        mod = [i for i in idx if rime[i] in _MODIFIED]
        pos = mod[-1] if nuc in ("ươ", "uô", "iê", "yê", "ươi", "ươu", "uôi", "iêu", "yêu") or len(mod) > 1 else mod[0]
        if nuc in ("uô", "iê", "yê"):
            pos = idx[-1]
    elif len(idx) == 1:
        pos = first
    elif len(idx) == 2:
        pos = idx[1] if (coda or nuc in ("oa", "oe", "uy")) else first
    else:
        pos = idx[1]
    ch = unicodedata.normalize("NFC", rime[pos] + _TONE_CHAR[tone])
    return onset + rime[:pos] + ch + rime[pos + 1:]


_MODIFIER_PAT = re.compile(r"dd|aa|ee|oo|aw|ow|uw")


def phonotactic_decode(tok: str, tone_only: bool = False) -> str | None:
    t = tok.lower()
    if not _MODIFIER_PAT.search(t) and not (tone_only and len(t) > 3 and t[-1] in "sfrxj"):
        return None
    for base, tone in telex_candidates(t):
        if tone == "" and base == t:
            continue
        if valid_syllable(base, tone):
            return place_tone(base, tone)
    return None


def telex_decode(tok: str, vocab: Vocab, protected: set[str] | frozenset = frozenset(), allow_tone_only: bool = True) -> str | None:
    t = tok.lower()
    if len(t) < 3 or t in protected or not t.isalpha():
        return None
    if t in vocab.plain:                        # seen as-is in the material (a proper word, brand, English word...): never 'repair' it
        return None
    for base, tone in telex_candidates(t):
        if not any(m in base for m in "đâêôăơư") and not tone:
            continue
        if not allow_tone_only and not any(m in base for m in "đâêôăơư"):
            continue                                  # 'max' -> 'mã', 'bans' -> 'bán': only trusted when the sentence is typed in Telex
        w = vocab.lookup(base, tone)
        if w and w != t:
            return w
    if t in vocab.plain:                        # seen as-is in the material (brand, English word...): leave it
        return None
    return phonotactic_decode(t, tone_only=allow_tone_only)


# unambiguous chat shorthand (lower-case token -> expansion)
ABBR = {
    "dthu": "doanh thu", "bnhiu": "bao nhiêu", "bnhieu": "bao nhiêu", "bnh": "bao nhiêu", "bn": "bao nhiêu", "nhiu": "nhiêu",
    "hnay": "hôm nay", "hqua": "hôm qua", "hqa": "hôm qua", "bnay": "bữa nay", "trc": "trước", "truoc": "trước", "vs": "với",
    "z": "vậy", "zậy": "vậy", "zay": "vậy", "zạ": "vậy", "mik": "mình", "mk": "mình", "ng": "người", "ngta": "người ta",
    "sp": "sản phẩm", "kh": "khách hàng", "nv": "nhân viên", "ncc": "nhà cung cấp", "dh": "đơn hàng", "đh": "đơn hàng", "hd": "hóa đơn", "hđ": "hóa đơn",
    "sl": "số lượng", "tt": "thanh toán", "ck": "chuyển khoản", "tm": "tiền mặt", "đc": "được", "dc": "được", "dk": "được", "cko": "cho",
    "trog": "trong", "ko": "không", "hok": "không", "hong": "không", "hem": "không", "khum": "không", "kg": "kg",
    "tg": "thời gian", "gd": "giao dịch", "cty": "công ty", "wa": "quá", "ns": "nói", "bt": "bình thường",
    "lq": "liên quan", "tl": "trả lời", "ctr": "chương trình", "đt": "điện thoại", "dtdd": "điện thoại", "nl": "nguyên liệu",
    "ae": "anh em", "cx": "cũng", "đg": "đang", "ntn": "như thế nào", "vd": "ví dụ", "mn": "mọi người", "sdt": "số điện thoại", "kq": "kết quả",
    "nhìu": "nhiều", "k": "không", "kô": "không", "khong": "không", "ok": "ok", "oke": "ok", "nx": "nữa", "j": "gì", "gj": "gì", "rùi": "rồi", "rui": "rồi", "lun": "luôn", "luon": "luôn",
}
_TH = re.compile(r"\b(?:th|thg)\s*(\d{1,2})\b|\bt\s*(\d{1,2})(?=\s*/\s*\d{2,4}\b)", re.I)
_Q = re.compile(r"\bq\s*(\d)\b", re.I)


def normalize_question(q: str, vocab: Vocab | None = None, protected: set[str] | frozenset = frozenset(), lex=None) -> str:
    """Return the question with shorthand expanded and Telex artefacts decoded. Digits, dates and unknown words are left untouched."""
    from .slang import default_lexicon, repair_token

    vocab = vocab or default_vocab()
    lex = lex or default_lexicon()
    q = unicodedata.normalize("NFC", q)
    q = _TH.sub(lambda m: f"tháng {m.group(1) or m.group(2)}", q)
    q = _Q.sub(lambda m: f"quý {m.group(1)}", q)

    hits = {"n": 0}

    def fix(m: re.Match, tone_only: bool) -> str:
        w = m.group(0)
        lw = w.lower()
        if lw in protected or (m.start() > 0 and src['s'][m.start() - 1].isdigit()):     # '50k', '3tr', '10ml', '2l' are amounts, not words
            return w
        if lw in ABBR and ABBR[lw] != lw and lw not in vocab.shop:
            hits["n"] += 1
            return ABBR[lw]
        d = telex_decode(w, vocab, protected, allow_tone_only=tone_only)
        proper_noun = w[0].isupper() and not w.isupper() and m.start() > 0 and not src['s'][:m.start()].rstrip().endswith((".", "?", "!"))
        if not d and not tone_only and not proper_noun and not telex_decode(w, vocab, protected, allow_tone_only=True):   # Telex words wait for pass 2
            try:
                d = repair_token(w, lex, vocab.plain | vocab.shop | set(protected))      # Gen Z spellings never seen before (generic rules, see slang.py)
            except Exception:  # noqa: BLE001
                d = None
            if not d:
                try:
                    from .spell import correct_token

                    known = vocab.plain | vocab.shop | set(protected)
                    if "đ" in lw and re.search(r"ow|uw|aw|ee|oo|aa", lw):          # half-Telex 'đuowc'
                        x = lw
                        for a_, b_ in _MODS:
                            x = x.replace(a_, b_)
                        d = correct_token(x, lex, known) if x != lw else None
                    if not d:
                        d = correct_token(w, lex, known)       # misspellings: wrong/missing tone or letter, glued words
                except Exception:  # noqa: BLE001
                    d = None
        if d:
            hits["n"] += 1
        return d if d else w

    src = {"s": q}
    first = _WORD.sub(lambda m: fix(m, False), q)          # pass 1: unmistakable evidence (modifiers dd/aa/ee/oo/aw/ow/uw, shorthand)
    src["s"] = first
    return _WORD.sub(lambda m: fix(m, True), first) if hits["n"] else first


_AMT = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*(k|nghìn|nghin|ngàn|ngan|tr|triệu|trieu|củ|cu)(\d{1,3})?(?![\w])", re.I)


def amounts_note(q: str) -> str:
    """'đơn trên 500k', '1tr5', '2 củ' -> '500k = 500000; 1tr5 = 1500000' (Vietnamese money shorthand, read the same way by everybody)."""
    out = []
    for m in _AMT.finditer(unicodedata.normalize("NFC", q)):
        num, unit, tail = m.group(1), m.group(2).lower(), m.group(3)
        try:
            val = float(num.replace(",", ".")) if not (num.count(".") == 1 and len(num.split(".")[1]) == 3) else float(num.replace(".", ""))
        except ValueError:
            continue
        mult = 1_000 if unit in ("k", "nghìn", "nghin", "ngàn", "ngan") else 1_000_000
        if tail:                                    # 1tr5 = 1.5 triệu, 2k5 = 2.5k
            val = int(num) + float("0." + tail)
        total = val * mult
        out.append(f"{m.group(0).strip()} = {int(round(total))}")
    return "; ".join(dict.fromkeys(out))


_DEFAULT: Vocab | None = None


def default_vocab() -> Vocab:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Vocab.load()
    return _DEFAULT
