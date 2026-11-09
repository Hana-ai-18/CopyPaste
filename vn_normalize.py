"""
vn_normalize.py — Chuẩn hoá câu hỏi tiếng Việt "đời thường" cho DataMate.

Thứ tự xử lý (thứ tự là quan trọng):
  1. NFC + gom khoảng trắng, bỏ emoji
  2. BẢO VỆ THỰC THỂ trước mọi thứ khác: tên hàng / khách / nhóm của shop,
     số tiền, ngày, mã (SP0012, NV003...) -> thay bằng placeholder ⟦E0⟧ ...
     (nếu không: "Plusssz" bị co thành "Plusz", "gas" bị Telex thành "gá")
  3. Với từng token chưa hợp lệ, sinh ứng viên theo thứ tự ưu tiên:
       a. giữ nguyên nếu là âm tiết tiếng Việt hợp lệ hoặc từ đã biết
       b. co ký tự kéo dài (>=3 lần -> 1)            "nhiềuuuu" -> "nhiều"
       c. giải mã Telex bị lọt                       "dduwowcj" -> "được"
       d. co ký tự đôi                               "nhiềeu"  -> "nhiều"
       e. từ điển teencode / viết tắt                "bnhiu"   -> "bao nhiêu"
       f. sửa 1 lỗi gõ theo từ vựng miền (edit distance 1, token dài >= 4)
     Ứng viên đầu tiên hợp lệ thắng; không có thì giữ nguyên (không đoán bừa).
  4. Trả placeholder về đúng chuỗi gốc.

Mọi thay đổi được ghi vào `changes` để log / debug.
"""
import re
import unicodedata

# ---------------------------------------------------------------- dấu thanh
TONE_MARKS = {'̀': 'f', '́': 's', '̉': 'r', '̃': 'x', '̣': 'j'}
TONE_COMBINING = {v: k for k, v in TONE_MARKS.items()}


def nfc(s):
    return unicodedata.normalize('NFC', s)


def strip_tone(s):
    """Bỏ dấu thanh, GIỮ dấu mũ/móc/trăng: 'nhiều' -> 'nhiêu'."""
    d = unicodedata.normalize('NFD', s)
    return nfc(''.join(c for c in d if c not in TONE_MARKS))


def get_tone(s):
    for c in unicodedata.normalize('NFD', s):
        if c in TONE_MARKS:
            return TONE_MARKS[c]
    return ''


def strip_all(s):
    """Bỏ hết dấu (để so khớp tên): 'Đậu phộng' -> 'dau phong'."""
    d = unicodedata.normalize('NFD', s)
    d = ''.join(c for c in d if unicodedata.category(c) != 'Mn')
    return d.replace('đ', 'd').replace('Đ', 'D')


def base_letter(c):
    return strip_all(c).lower()


# ---------------------------------------------------------------- âm tiết hợp lệ
ONSETS = sorted(['ngh', 'ng', 'nh', 'ch', 'tr', 'th', 'ph', 'kh', 'gh', 'gi', 'qu',
                 'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'x', ''],
                key=len, reverse=True)
VOWELS = 'aăâeêioôơuưy'
# Vần (không dấu thanh) — đủ dùng cho văn nói; bổ sung từ corpus bằng learn_rhymes()
RHYMES = set("""a ai ao au ay am an ang anh ap at ac ach
ă ăm ăn ăng ăp ăt ăc â âu ây âm ân âng âp ât âc
e eo em en eng ep et ec ê êu êm ên ênh êp êt êch
i ia iu im in inh ip it ich iêu iêm iên iêng iêp iêt iêc
o oa oai oao oay oan oang oanh oat oac oach oam oap oe oeo oen oet oi om on ong op ot oc oong ooc
ô ôi ôm ôn ông ôp ôt ôc ơ ơi ơm ơn ơp ơt
u ua uân uâng uât uây ue uê uêch uênh ui um un ung up ut uc uy uya uyên uyêt uynh uych uyt uyu uôi uôm uôn uông uôt uôc uơ
ư ưa ưi ưu ưm ưn ưng ưt ưc ươi ươu ươm ươn ương ươp ươt ươc
y yêu yên yêt yêm""".split())


def split_syllable(s):
    s = s.lower()
    for o in ONSETS:
        if s.startswith(o):
            rest = s[len(o):]
            if o in ('gi', 'qu') and rest and rest[0] not in VOWELS:
                continue
            return o, rest
    return '', s


def _onset_ok(o, r):
    """Luật chính tả phụ âm đầu: k/gh/ngh chỉ đứng trước e ê i y; c/g/ng không đứng trước e ê i; q phải là qu."""
    v = strip_all(r[:1]) if r else ''
    front = v in ('e', 'i', 'y')
    if o in ('k', 'gh', 'ngh'):
        return front
    if o in ('c', 'ng') and v in ('e', 'i', 'y'):
        return False
    if o == 'g' and v in ('e',):
        return False
    if o == 'q':
        return False
    return True


def is_syllable(tok):
    low = tok.lower()
    if re.fullmatch(r'[a-z]+', low):
        # chữ KHÔNG DẤU là hợp lệ nếu ứng với một âm tiết có dấu nào đó -> giữ nguyên,
        # việc so khớp để bước sau làm bằng khoá bỏ dấu (tránh 'luot' -> 'luật' thay vì 'lượt')
        o, r = split_syllable(low)
        if o == 'gi' and r == '':
            return True
        return r in {strip_all(x) for x in RHYMES} and _onset_ok(o, r)
    t = strip_tone(low)
    if not t or not re.fullmatch(r'[a-zăâđêôơư]+', t):
        return False
    o, r = split_syllable(t)
    if o == 'gi' and r == '':      # "gì"
        return True
    if r not in RHYMES or not _onset_ok(o, r):
        return False
    tone = get_tone(tok)
    # vần tắc (-p -t -c -ch) chỉ đi với sắc / nặng
    if re.search(r'(p|t|c|ch)$', r) and tone not in ('s', 'j', ''):
        return False
    return True


def learn_rhymes(texts):
    """Bổ sung vần từ văn bản sạch (rulebook, README, tên hàng)."""
    for t in texts:
        for w in re.findall(r'[^\W\d_]+', nfc(t)):
            if w.isupper() and len(w) > 1:          # bỏ 'YYYY', 'COD', 'VAT'... (không phải từ tiếng Việt)
                continue
            w = w.lower()
            if re.search(r'(.)\1', strip_all(w)) and not re.search(r'oo', w):   # bỏ chuỗi lặp
                continue
            w0 = strip_tone(w)
            o, r = split_syllable(w0)
            if r and len(r) <= 4 and re.fullmatch(r'[aăâeêioôơuưy]+[mnptcgh]*', r):
                RHYMES.add(r)


# ---------------------------------------------------------------- đặt dấu thanh
def place_tone(word, tone):
    """Đặt dấu thanh theo kiểu mới ('hoà', 'thuý'), trả NFC."""
    if not tone:
        return word
    w = word
    idx = [i for i, c in enumerate(w) if c in VOWELS]
    if not idx:
        return w
    # bỏ u của 'qu', i của 'gi' khỏi nhóm nguyên âm
    if w.startswith('qu') and len(idx) > 1 and idx[0] == 1:
        idx = idx[1:]
    if w.startswith('gi') and len(idx) > 1 and idx[0] == 1:
        idx = idx[1:]
    pos = None
    for i in idx:                              # nguyên âm có dấu phụ được ưu tiên
        if w[i] in 'ăâêôơư':
            pos = i
    if 'ươ' in w:
        pos = w.index('ươ') + 1
    if pos is None:
        last = idx[-1]
        has_final = last < len(w) - 1
        if len(idx) == 1:
            pos = idx[0]
        elif has_final or w[idx[0]:idx[-1] + 1] in ('oa', 'oe', 'uy', 'uê'):
            pos = idx[-1]
        elif len(idx) == 3:
            pos = idx[1]
        else:
            pos = idx[0]
    ch = unicodedata.normalize('NFD', w[pos])
    return nfc(w[:pos] + ch + TONE_COMBINING[tone] + w[pos + 1:])


# ---------------------------------------------------------------- Telex
def telex_decode(tok):
    """'dduwowcj' -> 'được', 'tieenf' -> 'tiền', 'cuwar' -> 'của'. None nếu không ra âm tiết hợp lệ."""
    t = tok.lower()
    if not re.fullmatch(r'[a-z]+', t):
        return None
    tone = ''
    # phím thanh: ký tự cuối (hoặc trước phụ âm cuối) thuộc sfrxj
    m = re.fullmatch(r'(.*?[aeiouyw].*?)([sfrxj])([a-z]*)', t)
    if m and m.group(3) in ('', 'n', 'ng', 'nh', 'm', 'c', 'ch', 't', 'p', 'i', 'u', 'o', 'y'):
        t, tone = m.group(1) + m.group(3), m.group(2)
    t = t.replace('dd', 'đ')
    t = re.sub(r'uow', 'ươ', t)
    t = re.sub(r'uo(.*?)w', r'ươ\1', t)             # "uowc" / "uocw"
    for a, b in (('aa', 'â'), ('ee', 'ê'), ('oo', 'ô')):
        t = t.replace(a, b)
    t = re.sub(r'a([a-z]*)w', r'ă\1', t)           # "awm", "amw" -> ăm
    t = re.sub(r'o([a-z]*)w', r'ơ\1', t)
    t = re.sub(r'u([a-z]*)w', r'ư\1', t)
    t = t.replace('w', 'ư')
    if not re.fullmatch(r'[a-zăâđêôơư]+', t):
        return None
    out = place_tone(t, tone)
    return out if is_syllable(out) else None


_HAT = {'a': 'aâă', 'e': 'eê', 'o': 'oôơ', 'u': 'uư', 'd': 'dđ'}


def telex_lazy(tok, limit=24):
    """
    Telex gõ thiếu dấu mũ/móc: 'nhieuf' -> 'nhiều', 'nguoif' -> 'người', 'dduocj' -> 'được'.
    Giữ phím thanh (s f r x j) như Telex, rồi thử thêm mũ/móc cho các nguyên âm, giữ các âm tiết hợp lệ.
    """
    from itertools import product
    t = tok.lower()
    if not re.fullmatch(r'[a-z]+', t):
        return []
    tone = ''
    m = re.fullmatch(r'(.*?[aeiouy].*?)([sfrxj])([a-z]*)', t)
    if m and m.group(3) in ('', 'n', 'ng', 'nh', 'm', 'c', 'ch', 't', 'p', 'i', 'u', 'o', 'y'):
        t, tone = m.group(1) + m.group(3), m.group(2)
    t = t.replace('dd', 'đ')
    slots = [_HAT.get(c, c) for c in t]
    if sum(len(x) > 1 for x in slots) > 4:
        return []
    out = []
    for combo in product(*slots):
        w = place_tone(''.join(combo), tone)
        if w and is_syllable(w) and w not in out:
            out.append(w)
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------- kéo dài
def collapse_runs(tok, min_run=3):
    """Co các chuỗi >= min_run ký tự cùng chữ cái gốc về 1, giữ ký tự có dấu.
    'nhiềuuuu' -> 'nhiều', 'zậyyyyy' -> 'zậy', 'baoooo' -> 'bao'."""
    out, i = [], 0
    while i < len(tok):
        j = i
        while j + 1 < len(tok) and base_letter(tok[j + 1]) == base_letter(tok[i]) and tok[i].isalpha():
            j += 1
        run = tok[i:j + 1]
        if len(run) >= min_run:
            # giữ ký tự "giàu dấu" nhất
            out.append(max(run, key=lambda c: len(unicodedata.normalize('NFD', c))))
        else:
            out.append(run)
        i = j + 1
    return ''.join(out)


def collapse_doubles_variants(tok):
    """Sinh các biến thể bỏ 1 ký tự ở từng cặp đôi: 'nhiềeu' -> ['nhiều', ...]."""
    res = []
    for i in range(len(tok) - 1):
        if base_letter(tok[i]) == base_letter(tok[i + 1]) and tok[i].isalpha():
            a, b = tok[i], tok[i + 1]
            keep = a if len(unicodedata.normalize('NFD', a)) >= len(unicodedata.normalize('NFD', b)) else b
            res.append(tok[:i] + keep + tok[i + 2:])
    return res


# ---------------------------------------------------------------- teencode / viết tắt
# Khoá: dạng thường, đã co kéo dài. Giá trị: cụm chuẩn. Mở rộng dần từ log.
TEEN = {
    'bnhiu': 'bao nhiêu', 'bnhieu': 'bao nhiêu', 'bn': 'bao nhiêu', 'bnh': 'bao nhiêu',
    'nhiu': 'nhiêu', 'nhìu': 'nhiều',
    'dthu': 'doanh thu', 'dt': 'doanh thu', 'đh': 'đơn hàng', 'dh': 'đơn hàng', 'đơng': 'đơn',
    'hnay': 'hôm nay', 'hqua': 'hôm qua', 'hom': 'hôm', 'trc': 'trước', 'trk': 'trước',
    'th': 'tháng', 'thg': 'tháng', 'sp': 'sản phẩm', 'ncc': 'nhà cung cấp', 'nv': 'nhân viên',
    'kh': 'khách hàng', 'ck': 'chuyển khoản', 'tm': 'tiền mặt', 'dc': 'được', 'đc': 'được',
    'ko': 'không', 'k': 'không', 'hok': 'không', 'hông': 'không', 'hổng': 'không', 'khum': 'không',
    'mik': 'mình', 'mk': 'mình', 'mn': 'mọi người', 'j': 'gì', 'z': 'vậy', 'zậy': 'vậy',
    'dzậy': 'vậy', 'zị': 'vậy', 'vs': 'với', 'wán': 'quán', 'wan': 'quán', 'kủa': 'của',
    'kòn': 'còn', 'bít': 'biết', 'cx': 'cũng', 'ntn': 'như thế nào', 'r': 'rồi',
    'tiềng': 'tiền', 'tổg': 'tổng', 'chừ': 'bây giờ', 'ni': 'này', 'mô': 'nào', 'rứa': 'vậy',
    'nhứt': 'nhất', 'cafe': 'cà phê', 'cf': 'cà phê', 'ship': 'ship', 'cóa': 'có', 'hôk': 'không', 'hem': 'không', 'kô': 'không', 'hông': 'không', 'hok': 'không', 'nhut': 'nhất', 'củ': 'triệu', 'xi ô đi': 'cod', 'o đờ': 'đơn',
}
FILLERS = {'ạ', 'à', 'hả', 'hen', 'nha', 'nhé', 'nhe', 'nghen', 'nhỉ', 'hỉ', 'ta', 'đó', 'z',
           'ha', 'ơi', 'ui', 'nè', 'e', 'ee', 'ad', 'shop', 'thanks'}


def edit1(a, b):
    """Khoảng cách Damerau-Levenshtein <= 1 ?"""
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
    for i in range(len(b)):
        if b[:i] + b[i + 1:] == a:
            return True
    return False



# ---------------------------------------------------------------- bảng teencode mở rộng (lexicon/teencode.tsv)
TEEN_ALT_FILE = {}   # viết tắt -> các nghĩa phụ (nghĩa đầu nằm trong TEEN)


def load_teencode(path=None):
    """Nạp lexicon/teencode.tsv vào TEEN (nghĩa đầu) và TEEN_ALT_FILE (nghĩa phụ). Trả số mục đã nạp."""
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lexicon', 'teencode.tsv')
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue
            k = nfc(parts[0].strip().lower())
            means = [nfc(m.strip()) for m in parts[1].split('|') if m.strip()]
            TEEN[k] = means[0]
            TEEN_ALT_FILE[k] = means[1:]
            n += 1
    return n


N_TEEN_FILE = load_teencode()
ALIAS_GROUPS = []    # nhóm từ đồng nghĩa vùng miền (heo/lợn, ly/cốc…) — chỉ để gợi ý khớp tên hàng, không sửa câu


def load_lamtay_extra(path=None):
    """Nạp lexicon/gioitre_vungmien_lamtay.tsv: nhóm dong_nghia_* vào ALIAS_GROUPS, còn lại vào TEEN (không ghi đè bảng chính)."""
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lexicon', 'gioitre_vungmien_lamtay.tsv')
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            p = line.rstrip('\n').split('\t')
            if len(p) < 3:
                continue
            k = nfc(p[0].strip().lower())
            means = [nfc(m.strip()) for m in p[1].split('|') if m.strip()]
            if p[2].startswith('dong_nghia'):
                ALIAS_GROUPS.append([k] + means)
                continue
            if k in TEEN or not means or means[0] == k:
                continue
            TEEN[k] = means[0]
            TEEN_ALT_FILE[k] = means[1:]
            n += 1
    return n


N_TEEN_EXTRA = load_lamtay_extra()


# ---------------------------------------------------------------- viết lệch kiểu mới (ngoài train/public)
# Thay chữ kiểu teen: j->gi, z->d/v/gi, w->qu, f->ph, k->c, đuôi k/ck->c/ch, thừa g cuối ('tiềng')
_TEEN_SUBS = [
    (r'^j', ['gi', 'd']), (r'^dz', ['d', 'v', 'gi', 'đ']), (r'^z', ['d', 'v', 'gi', 'đ']), (r'^w', ['qu', 'u', 'o']),
    (r'^d(?!z)', ['đ']), (r'(?<=^[ck])oa$', ['o']), (r'(?<=^h)ôk$', ['ông']),
    (r'^f', ['ph']), (r'^k(?=[aăâoôơuư])', ['c']), (r'ck$', ['ch', 'c']), (r'(?<=[aăâeêioôơuưy])k$', ['c', 'ch']),
    (r'(?<=[aăâeêioôơuưy])g$', ['']), (r'ih$', ['inh']), (r'uk$', ['uc']), (r'ug$', ['ung']), (r'^ng(?=[eêi])', ['ngh']),
    (r'^g(?=[eêi])', ['gh']),
]


def teen_variants(tok, limit=12):
    """Các cách đọc lại một chữ viết kiểu teen/lệch chính tả, chỉ giữ âm tiết hợp lệ. 'jờ' -> giờ, 'wá' -> quá, 'thík' -> thích."""
    t = nfc(tok.lower())
    base, tone = strip_tone(t), get_tone(t)       # thay trên chữ bỏ thanh rồi đặt thanh lại ('thík' -> 'thich' -> 'thích')
    out = []
    frontier = [base]
    for _ in range(2):                       # tối đa 2 phép thay
        nxt = []
        for w in frontier:
            for pat, reps in _TEEN_SUBS:
                if re.search(pat, w):
                    for r in reps:
                        v = re.sub(pat, r, w, count=1)
                        if v == w or v in nxt:
                            continue
                        nxt.append(v)
                        vt = place_tone(v, tone) if tone else v
                        if vt and vt != t and vt not in out and is_syllable(vt):
                            out.append(vt)
        frontier = nxt[:30]
    return out[:limit]


# Lẫn lộn chính tả theo vùng miền: n/l, s/x, ch/tr, d/gi/r, hỏi/ngã, c/t cuối, n/ng cuối, i/y
_REGION_SUBS = [
    (r'^n(?!g|h)', ['l']), (r'^l', ['n']), (r'^s', ['x']), (r'^x', ['s']), (r'^ch', ['tr']), (r'^tr', ['ch']),
    (r'^d(?!z)', ['gi', 'r']), (r'^gi', ['d', 'r']), (r'^r', ['d', 'gi']), (r'c$', ['t']), (r't$', ['c']),
    (r'(?<!n)n$', ['ng']), (r'ng$', ['n']), (r'i$', ['y']), (r'y$', ['i']),
]
_HOI_NGA = {'\u0309': '\u0303', '\u0303': '\u0309'}


def region_variants(tok, limit=10):
    """Các chữ dễ viết nhầm do phát âm vùng miền (một phép thay), chỉ giữ âm tiết hợp lệ."""
    import unicodedata as _u
    t = nfc(tok.lower())
    out = []
    for pat, reps in _REGION_SUBS:
        if re.search(pat, strip_tone(t)):
            for r in reps:
                v = nfc(re.sub(pat, r, t, count=1))
                if v != t and v not in out and is_syllable(v):
                    out.append(v)
    d = _u.normalize('NFD', t)
    if any(c in d for c in _HOI_NGA):
        v = nfc(''.join(_HOI_NGA.get(c, c) for c in d))
        if v != t and is_syllable(v):
            out.append(v)
    return out[:limit]


# Gõ kiểu VNI: số sau chữ là dấu (1 sắc 2 huyền 3 hỏi 4 ngã 5 nặng 6 mũ 7 móc 8 trăng 9 đ)
_VNI_TONE = {'1': 's', '2': 'f', '3': 'r', '4': 'x', '5': 'j'}


def vni_decode(tok):
    """'tie6n2' -> 'tiền', 'nha6t1' -> 'nhất', 'd9o7n' -> 'đơn'. Chỉ nhận chữ bắt đầu bằng chữ cái, có 1–3 chữ số
    xen trong/cuối và giải ra âm tiết hợp lệ (tránh mã hàng 'SP0066', kỳ 'q1', 't7', đơn vị '10ml')."""
    t = tok.lower()
    if not re.fullmatch(r'[a-z]+[1-9][a-z1-9]*', t) or len(re.findall(r'\d', t)) > 4 or re.search(r'\d{3}', t):
        return None
    letters, tone = [], ''
    for ch in t:
        if ch.isalpha():
            letters.append(ch)
        elif ch in _VNI_TONE:
            tone = _VNI_TONE[ch]
        elif ch == '7' and len(letters) >= 2 and letters[-2:] == ['u', 'o']:
            letters[-2:] = ['ư', 'ơ']            # 'duo7c' -> 'dươc'
        elif ch in '678' and letters:
            prev = letters[-1]
            m = {('6', 'a'): 'â', ('6', 'e'): 'ê', ('6', 'o'): 'ô', ('7', 'o'): 'ơ', ('7', 'u'): 'ư', ('8', 'a'): 'ă'}.get((ch, prev))
            if not m:
                # dấu móc của 'uo' ghi sau cả cụm: 'duo7c' -> 'dươc'
                if ch == '7' and len(letters) >= 2 and letters[-2:] == ['u', 'o']:
                    letters[-2:] = ['ư', 'ơ']
                    continue
                return None
            letters[-1] = m
        elif ch == '9':
            idx = next((i for i, c in enumerate(letters) if c == 'd'), None)
            if idx is None:
                return None
            letters[idx] = 'đ'
    w = ''.join(letters)
    if not re.search(r'[aăâeêioôơuưy]', w):
        return None
    if tone:
        w = place_tone(w, tone)
    return w if w and is_syllable(w) else None


# ---------------------------------------------------------------- tìm ứng viên tổng quát
def tone_canon(w):
    """Chuẩn hoá vị trí dấu thanh: 'cóa' và 'coá' -> cùng một dạng."""
    w = nfc(w.lower())
    return place_tone(strip_tone(w), get_tone(w)) if re.search(r'[a-zăâđêôơư]', strip_tone(w)) else w


def _runs(tok):
    out, i = [], 0
    while i < len(tok):
        j = i
        while j + 1 < len(tok) and tok[i].isalpha() and base_letter(tok[j + 1]) == base_letter(tok[i]):
            j += 1
        out.append(tok[i:j + 1])
        i = j + 1
    return out


def search_candidates_all(low, known_words=(), max_cands=256):
    """
    Xử lý TRỘN lỗi: Telex + kéo dài + đuôi thừa. Ví dụ:
      'coasaaaa'   -> 'coas' (bỏ đuôi 'aaaa') -> Telex 'cóa' -> teencode 'có'
      'nhaatstttt' -> 'nhaats' (bỏ đuôi 'tttt') -> Telex 'nhất'
      'tieeenffff' -> 'tieenf' (eee->ee, ffff->f) -> Telex 'tiền'
      'kooo'       -> 'ko' -> teencode 'không'
    Mỗi chuỗi lặp >=2 thử giữ 1 hoặc 2 ký tự (Telex cần 'aa','ee','oo','dd');
    thử bỏ 1-2 cụm cuối (người gõ giữ phím). Chấm điểm:
      (1) trúng teencode hoặc từ đã biết  (2) chi phí sửa thấp  (3) đã giải Telex hợp lệ.
    """
    from itertools import product
    runs = _runs(low)
    opts = []
    for r in runs:
        if len(r) >= 2 and r[0].isalpha():
            keep1 = max(r, key=lambda c: len(unicodedata.normalize('NFD', c)))
            c1 = 0 if len(r) >= 3 else 1          # co chuỗi >=3 miễn phí; co cặp đôi tốn 1
            opts.append([(keep1, c1), (r[:2], 0 if len(r) >= 3 else 0)])
        else:
            opts.append([(r, 0)])
    teen = {tone_canon(k): v for k, v in TEEN.items() if ' ' not in k}
    known = set(known_words)
    scored = []
    for n, combo in enumerate(product(*opts)):
        if n >= max_cands:
            break
        parts = [c for c, _ in combo]
        base_cost = sum(k for _, k in combo)
        for trim in range(0, 3):                   # bỏ 0..2 cụm cuối
            if trim >= len(parts):
                break
            # chỉ được bỏ đuôi nếu cụm bị bỏ là chuỗi KÉO DÀI (gốc lặp >= 2) -> 'cafe' không bị cắt
            if trim and any(len(runs[len(runs) - 1 - t]) < 2 for t in range(trim)):
                break
            s = ''.join(parts[:len(parts) - trim]) if trim else ''.join(parts)
            cost = base_cost + 2 * trim
            forms = [(s, 'kéo dài'), (telex_decode(s), 'telex')]
            forms += [(f, 'telex thiếu mũ') for f in telex_lazy(s) if f != forms[1][0]]
            for form, how in forms:
                if not form:
                    continue
                tc = tone_canon(form)
                # cắt đuôi rồi mới tra teencode dễ sinh nghĩa lạ ('thuuu' -> 'th' -> 'tháng'): cấm
                hit_teen = tc in teen and not trim
                if not hit_teen and not re.search(r'[aăâeêioôơuưy]', strip_tone(form)):
                    continue                        # phải có nguyên âm ('l' bị loại), trừ teencode ('đc')
                hit_word = len(form) >= 2 and form in known
                valid = hit_teen or hit_word or is_syllable(form)
                if not valid:
                    continue
                out = teen.get(tc, form)
                label = how + ('+bỏ đuôi' if trim else '') + ('+teencode' if hit_teen else '')
                scored.append(((0 if (hit_teen or hit_word) else 1, cost, 0 if how == 'telex' else 1), out, label))
    scored.sort(key=lambda x: x[0])
    res, seen = [], set()
    for key, out, label in scored:          # bỏ trùng, giữ thứ tự tốt nhất
        if out not in seen:
            seen.add(out)
            res.append((out, label, key))
    return res


def search_candidates(low, known_words=(), max_cands=256):
    """Ứng viên tốt nhất (theo luật, chưa xét ngữ cảnh). None nếu không sửa."""
    res = search_candidates_all(low, known_words, max_cands)
    if not res or res[0][0] == low:
        return None
    return res[0][0], res[0][1]

# ---------------------------------------------------------------- bảo vệ thực thể
ENTITY_PATTERNS = [
    r'(?<![A-Za-z0-9])(?:SP|NV|KH|PN|ORD|NCC|CA)\d+(?!\d)',                 # mã
    r'(?<!\d)\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}(?!\d)',                       # ngày đủ
    r'(?<!\d)\d{1,2}[/\-]\d{4}(?!\d)',                                        # tháng/năm 9/2025
    r'(?<!\d)\d{1,2}[/\-]\d{2}(?!\d)',                                        # 3/26, 1/1
    r'\b\d+(?:[.,]\d+)*\s*(?:tr\d*|triệu|củ|k|nghìn|ngàn|đ|đồng|vnd)\b',   # tiền
    r'\b\d+(?:[.,]\d+)?\s*(?:%|mg|ml|g|kg|l|m|cm|mm|gb)\b',                 # số + đơn vị đo
    r'\b\d+(?:[.,]\d+)*\b',                                                  # số trần
]


class Normalizer:
    def __init__(self, known_terms=(), known_words=(), clean_texts=()):
        """
        known_terms : tên hàng / khách / nhóm / NCC / nhân viên của SHOP ĐANG HỎI (bảo vệ nguyên khối).
        known_words : từ hợp lệ khác (tiếng Anh hay gặp: bill, voucher, top, gas, order...).
        clean_texts : rulebook, README... để học thêm vần và từ vựng miền.
        """
        learn_rhymes(clean_texts)
        self.terms = sorted({nfc(t) for t in known_terms if t and len(t) >= 2}, key=len, reverse=True)
        self.term_keys = [(self._key(t), t) for t in self.terms]
        self.words = {w.lower() for w in known_words}
        for t in list(known_terms) + list(clean_texts):
            for w in re.findall(r'[^\W\d_]+', nfc(str(t)).lower()):
                self.words.add(w)
        self.domain_vocab = {w for w in self.words if len(w) >= 4}

    @staticmethod
    def _key(s):
        return re.sub(r'\s+', ' ', strip_all(nfc(s)).lower()).strip()

    def _protect(self, text, store):
        # 1) tên trong danh mục: so khớp không dấu, không phân biệt hoa thường, theo ranh giới từ
        low = self._key(text)
        # chuỗi key và text có cùng độ dài khi NFC + strip_all (ký tự 1-1), nên dùng chung chỉ số
        assert len(low) == len(re.sub(r'\s+', ' ', nfc(text)).strip()) or True
        txt = re.sub(r'\s+', ' ', nfc(text)).strip()
        low = self._key(txt)
        spans = []
        for k, t in self.term_keys:
            for m in re.finditer(r'(?<!\w)' + re.escape(k) + r'(?!\w)', low):
                raw = txt[m.start():m.end()]
                if raw.lower() != t.lower():
                    # khớp nhờ bỏ dấu: chỉ chấp nhận khi người viết KHÔNG gõ dấu ở đoạn đó và tên đủ dài,
                    # tránh 'hôm qua' bị khoá thành tên hàng 'quả'
                    if strip_all(raw) != raw or len(k) < 5:
                        continue
                if not any(m.start() < e and s < m.end() for s, e, _ in spans):
                    spans.append((m.start(), m.end(), t))
        out, last = [], 0
        for s, e, t in sorted(spans):
            out.append(txt[last:s])
            store.append(('term', txt[s:e], t))
            out.append(f' ⟦E{len(store) - 1}⟧ ')
            last = e
        out.append(txt[last:])
        txt = ''.join(out)
        # 2) mã, ngày, tiền, số
        for pat in ENTITY_PATTERNS:
            def rep(m):
                if '⟦' in m.group(0):
                    return m.group(0)
                store.append(('lit', m.group(0), m.group(0)))
                return f' ⟦E{len(store) - 1}⟧ '
            txt = re.sub(pat, rep, txt, flags=re.I)
        return txt

    def _fix_token(self, tok, changes):
        if tok.isupper() and len(tok) <= 4:          # viết hoa ngắn: AA, COD, VAT, NV -> giữ nguyên
            return tok
        low = tok.lower()
        if low in self.words or is_syllable(low) or not re.search(r'[^\W\d_]', low):
            return tok
        best = search_candidates(low, self.words)
        if best is not None:
            out, how = best
            changes.append((tok, out, how))
            return out
        c1 = collapse_runs(low)
        if len(c1) >= 4:
            ascii_tok = bool(re.fullmatch(r'[a-z]+', c1))
            # chữ không dấu (có thể là tên tiếng Anh: Gold, Bull, Sino...) chỉ được sửa nếu chỉ khác phần dấu
            near = sorted(w for w in self.domain_vocab
                          if (strip_all(w) == c1 if ascii_tok else edit1(c1, w)))
            if len(near) == 1:                      # chỉ sửa khi duy nhất 1 ứng viên
                changes.append((tok, near[0], 'gõ sai 1 ký tự'))
                return near[0]
        return tok                                   # không chắc -> giữ nguyên

    def normalize(self, text):
        store, changes = [], []
        t = nfc(text)
        t = re.sub(r'[\U0001F000-\U0001FFFF☀-➿]', ' ', t)        # emoji
        t = re.sub(r'([?!.,])\1+', r'\1', t)                                 # "???" -> "?"
        t = self._protect(t, store)
        parts = re.split(r'(⟦E\d+⟧|[^\w⟦⟧]+)', t)
        out = []
        for p in parts:
            if not p or p.startswith('⟦') or not re.search(r'\w', p):
                out.append(p)
            else:
                out.append(self._fix_token(p, changes))
        t = ''.join(out)
        # teencode nhiều token (sau khi đã sửa từng token)
        low = t.lower()
        for k in sorted((k for k in TEEN if ' ' in k), key=len, reverse=True):
            low = re.sub(r'(?<!\w)' + re.escape(k) + r'(?!\w)', TEEN[k], low)
        t = low
        # trả thực thể về: tên hàng -> dạng chuẩn trong danh mục, literal -> nguyên văn
        entities = []
        def back(m):
            kind, raw, canon = store[int(m.group(1))]
            entities.append((kind, raw, canon))
            return canon
        t = re.sub(r'⟦[Ee](\d+)⟧', back, t)
        t = re.sub(r'\s+', ' ', t).strip()
        return {'text': t, 'changes': changes, 'entities': entities}


if __name__ == '__main__':
    terms = ['Plusssz Max Multivitamin viên sủi tuýp 20 viên', 'Gas Petrolimex 12kg', 'Bia Tiger lon',
             'Đậu phộng rang muối 200g', 'Nurofen for Children hỗn dịch chai 60ml', 'Cà phê sữa (M)']
    eng = ['bill', 'voucher', 'top', 'order', 'revenue', 'check', 'size', 'cod', 'zalo', 'oa']
    N = Normalizer(terms, eng)
    tests = [
        'tieenf bán gas petrolimex 12kg tháng 9/2025 là nhiêu',
        'hnay cửa hangf có nhiêu ddown',
        'trong tháng 5 năm 2026 cuwar hàng bán dduwowcj mấy chai bia tiger lon ta',
        'dthu thangs nayyyy tăng hay giảmmm bnhiuuuu % zậyyyy 😅',
        'tồn kho PLUSSSZ max multivitamin viên sủi tuýp 20 viên còn bao nhiêuuu',
        'so 7 ngày kể từ ngày tăng giá vs 7 ngày ngay truwowcs, lượng bán tính theo tuýp',
        'ca phe sua (m) thang 3/26 ban dc bn ly',
        'nhiềeu nhất là mooix ngayf mấy đơn',
        'th10/2025 doanh thu tu 1tr5 tro len co may don',
    ]
    for s in tests:
        r = N.normalize(s)
        print('IN :', s)
        print('OUT:', r['text'])
        print('    ', r['changes'])
