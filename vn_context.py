"""
vn_context.py — sửa chữ THEO NGỮ CẢNH CẢ CÂU (không dùng LLM).

Vấn đề: sửa từng chữ riêng lẻ không biết chữ đó có hợp với câu không.
  'teenn khach hang'  ->  từng chữ: 'tên' / 'ten' / 'teen' ?   cả câu: 'tên khách hàng'
  'baan chayj nhaat'  ->  từng chữ: 'bân' / 'ban' ?             cả câu: 'bán chạy nhất'

Cách làm (kiểu "kênh nhiễu" trong sửa chính tả):
  1. Mỗi chữ có một DANH SÁCH ỨNG VIÊN do luật sinh ra:
       - chính nó; các cách gộp chữ kéo dài / giải Telex (vn_normalize.search_candidates_all);
       - teencode; các dạng CÓ DẤU cùng mặt chữ trong từ điển ('khach' -> 'khách').
     Mỗi ứng viên có CHI PHÍ SỬA (sửa càng nhiều càng đắt).
  2. Một MÔ HÌNH NGÔN NGỮ 2-GRAM (học từ rulebook, README, tên hàng, câu hỏi TRAIN)
     chấm xem chuỗi chữ nào "nghe giống câu thật".
  3. Viterbi chọn tổ hợp ứng viên có điểm = log P(câu) − λ·chi phí sửa cao nhất.
  4. Mỗi vị trí có ĐỘ CHẮC = chênh điểm giữa ứng viên tốt nhất và nhì.
     Chỗ nào chênh ít -> đánh dấu `unsure` để (tuỳ chọn) hỏi LLM dạng trắc nghiệm.

Không bịa chữ: kết quả luôn nằm trong danh sách ứng viên. Số, ngày, tiền, tên hàng
đã được Normalizer khoá trước (⟦E..⟧) nên không bị đụng.
"""
import math
import re
from collections import Counter, defaultdict

from vn_normalize import (TEEN, TEEN_ALT_FILE, Normalizer, is_syllable, nfc, region_variants,
                          search_candidates_all, strip_all, teen_variants, tone_canon, vni_decode)

# chữ tiếng Anh / tên riêng hay gặp trong câu hỏi bán hàng: KHÔNG sửa, KHÔNG thêm dấu ('bill' không thành 'bị')
ENGLISH = set('''bill best seller top sale sales ship shipper order voucher combo size online offline shop menu topping
cod ok app grab shopee lazada tiktok momo zalo zalopay vip free freeship deal flash pos sku code team feedback review kpi
mini max pro plus set pack box gift card live stream check sold out new hot trend tea milk coffee cake cookie
latte cappuccino espresso matcha socola chocolate pizza burger gas spa nail hair makeup skincare serum toner
iphone samsung oppo xiaomi laptop tablet sim usb wifi tv inch gb mah'''.split())

# teencode có nhiều nghĩa: để ngữ cảnh chọn ('bao nhiu' -> nhiêu, 'nhiu nhất' -> nhiều)
TEEN_ALT = {'nhiu': ['nhiêu', 'nhiều'], 'bn': ['bao nhiêu', 'bạn'], 'nhìu': ['nhiều'], 'dt': ['doanh thu', 'điện thoại'],
            'k': ['không', 'nghìn'], 'ck': ['chuyển khoản', 'chồng'], 'th': ['tháng', 'thì'], 'r': ['rồi'],
            'mn': ['mọi người'], 'vs': ['với', 'so với']}

WORD_RE = re.compile(r'[^\W\d_]+')
BOS, EOS, ENT, UNK = '<s>', '</s>', '<E>', '<unk>'


_LEX_PHRASES = None
USE_WORDLIST_LM = True      # đưa ~22.000 từ thông dụng vào mô hình ngôn ngữ (1 lần đếm/từ)
USE_TSV_EXTRA = True        # dùng cả mục nhiều nghĩa + cụm nhiều chữ của các bảng tsv (xem load_variant_extra)
SOFT_COST, PHRASE_COST = 0.8, 0.05


def lexicon_phrases():
    """Từ/cụm chuẩn trong lexicon (tu_thong_dung.txt + nghĩa của bảng teencode) làm câu ngắn cho mô hình ngôn ngữ:
    để cụm bung ra từ viết tắt ('bnt' -> bao nhiêu tiền) không bị phạt chỉ vì shop/rulebook chưa từng viết."""
    global _LEX_PHRASES
    if _LEX_PHRASES is None:
        import os
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lexicon')
        out = []
        p = os.path.join(here, 'tu_thong_dung.txt')
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                if not line.startswith('#'):
                    out += [w.replace('_', ' ') for w in line.split()]
        out += sorted({v for v in TEEN.values()} | {m for ms in TEEN_ALT_FILE.values() for m in ms})
        p = os.path.join(here, 'gpl', 'tu_dien_22k.txt')          # từ điển Viet22K (GPL v2), tuỳ chọn
        if USE_WORDLIST_LM and os.path.exists(p):
            out += [l.strip() for l in open(p, encoding='utf-8') if l.strip() and not l.startswith('#')]
        _LEX_PHRASES = out
    return _LEX_PHRASES


LOOKUP_KINDS = ('viet_tat', 'dinh_lien', 'dinh_lien_bo_dau', 'viet_tat_gioi_tre', 'gioi_tre', 'gioi_tre_khong_dau', 'vung_mien', 'teen')


def load_variant_lookup(paths=None):
    """Dạng lệch AN TOÀN (không phải âm tiết hợp lệ / từ thật, chỉ một nghĩa) từ các bộ sinh tự động:
    lexicon/teencode_variants.tsv và lexicon/teencode_gioitre_vungmien.tsv.
    Ví dụ: 'khachhang' -> khách hàng, 'tck' -> tồn cuối kỳ, 'djã' -> dĩa, 'bnt' -> bao nhiêu tiền."""
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lexicon')
    paths = paths or [os.path.join(here, 'teencode_variants.tsv'), os.path.join(here, 'teencode_gioitre_vungmien.tsv'),
                      # mở rộng từ từ điển Viet22K (GPL v2, thư mục riêng; không có thì bỏ qua)
                      os.path.join(here, 'gpl', 'teencode_variants_tudien.tsv'),
                      os.path.join(here, 'gpl', 'teencode_gioitre_vungmien_tudien.tsv')]
    out = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4 and p[3] == '1' and p[2] in LOOKUP_KINDS:
                    if re.fullmatch(r'\w+', p[0]):
                        out.setdefault(tone_canon(p[0]), p[1])
                    elif re.fullmatch(r'\w\.\w+', p[0]):                  # 'g.nhập' -> giá nhập (thay trước khi tách chữ)
                        out.setdefault('.' + nfc(p[0].lower()), p[1])
    return out


_VARIANT_EXTRA = None


def load_variant_extra(paths=None):
    """Phần CÒN LẠI của các bảng tsv (trước chỉ dùng mục an toàn 1 chữ):
    - soft:    dạng lệch -> [các nghĩa] cho mục NHIỀU NGHĨA (cột an_toan = 0: 'j' = gì/giờ, vùng miền mơ hồ…) và mục an toàn
               thuộc kiểu luật đã tự sinh (kéo dài, telex, vni, bỏ dấu) — thành ỨNG VIÊN, mô hình ngôn ngữ chọn theo ngữ cảnh;
    - phrases: cụm NHIỀU CHỮ an toàn cùng số chữ ('biên lợi nhuan' -> 'biên lợi nhuận') — ưu tiên khi cả cụm xuất hiện.
    Đọc một lần, dùng chung."""
    global _VARIANT_EXTRA
    if _VARIANT_EXTRA is not None and paths is None:
        return _VARIANT_EXTRA
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lexicon')
    paths = paths or [os.path.join(here, 'teencode_variants.tsv'), os.path.join(here, 'teencode_gioitre_vungmien.tsv'),
                      os.path.join(here, 'gpl', 'teencode_variants_tudien.tsv'),
                      os.path.join(here, 'gpl', 'teencode_gioitre_vungmien_tudien.tsv')]
    soft, phrases = {}, {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                p = line.rstrip('\n').split('\t')
                if len(p) < 4 or not p[0] or not p[1]:
                    continue
                var, canon, kind, safe = nfc(p[0].lower()), nfc(p[1].lower()), p[2], p[3]
                vt, ct = var.split(), canon.split()
                if len(vt) >= 2:
                    if safe == '1' and len(vt) == len(ct) and len(vt) <= 5:
                        phrases.setdefault(tuple(vt), tuple(ct))
                    continue
                if not re.fullmatch(r'\w+', var):
                    continue
                if safe == '1' and kind in LOOKUP_KINDS:
                    continue                                   # đã dùng trong load_variant_lookup (tra thẳng)
                lst = soft.setdefault(tone_canon(var), [])
                for m in canon.split('|'):
                    m = m.strip()
                    # nghĩa NHIỀU chữ chỉ cho dạng KHÔNG phải âm tiết hợp lệ ('bnh' -> bao nhiêu); 'banh' là âm tiết thật,
                    # bung ra 'bạn hữu' sẽ được thưởng theo số chữ và thắng bừa (đo trên train: làm tụt kiểu bỏ dấu 93% -> 90%)
                    if not m or m in lst or len(lst) >= 4 or (' ' in m and is_syllable(var)):
                        continue
                    lst.append(m)
    _VARIANT_EXTRA = (soft, phrases)
    return _VARIANT_EXTRA


def tokenize(text):
    return WORD_RE.findall(nfc(str(text)).lower())


def accent_ratio(line):
    """Tỉ lệ chữ có dấu. Câu gần như không dấu không được dùng để học mô hình ngôn ngữ
    (train có ~26% câu hỏi viết không dấu; học từ chúng sẽ dạy model rằng 'khach hang' là chuẩn)."""
    w = tokenize(line)
    if not w:
        return 0.0
    return sum(1 for x in w if strip_all(x) != x) / len(w)


class BigramLM:
    """2-gram nội suy, đủ nhẹ để học trong < 1 giây trên vài nghìn câu."""

    def __init__(self, texts, lam=0.75, alpha=0.1, min_accent=0.4, weak_texts=()):
        """texts: văn bản thật (rulebook, README, tên hàng, câu hỏi train). weak_texts: từ điển/lexicon — góp vào xác suất
        nhưng KHÔNG tính là 'đã gặp thật' (self.real), để chữ viết lệch trùng một từ hiếm trong từ điển vẫn được sửa."""
        self.min_accent = min_accent
        self.real = Counter()
        self.uni, self.bi, self.ctx = Counter(), Counter(), Counter()
        self.tri, self.ctx2 = Counter(), Counter()
        n_strong = len(texts)
        texts = list(texts) + list(weak_texts)
        for i_t, t in enumerate(texts):
            for line in re.split(r'[\n.?!;:]+', str(t)):
                toks = tokenize(line)
                if not toks or (len(toks) >= 3 and accent_ratio(line) < self.min_accent):
                    continue
                # chữ gõ hỏng trong câu hỏi train ('hangf', 'ddown', 'cuar') KHÔNG được thành từ vựng chuẩn
                toks = [w if is_syllable(w) else UNK for w in toks]
                seq = [BOS] + toks + [EOS]
                self.uni.update(seq)
                if i_t < n_strong:
                    self.real.update(toks)
                for a, b in zip(seq, seq[1:]):
                    self.bi[(a, b)] += 1
                    self.ctx[a] += 1
                for a, b, c in zip(seq, seq[1:], seq[2:]):
                    self.tri[(a, b, c)] += 1
                    self.ctx2[(a, b)] += 1
        self.N = sum(self.uni.values())
        self.V = len(self.uni) + 1
        self.lam, self.alpha = lam, alpha
        # chỉ mục: mặt chữ không dấu -> các dạng có dấu đã gặp (để khôi phục dấu theo ngữ cảnh)
        # bỏ dạng viết sai hiếm ('đươc' cạnh 'được'): chỉ giữ dạng có số lần >= 2 và >= 5% dạng phổ biến nhất
        groups = defaultdict(list)
        for w, c in self.uni.items():
            if w not in (BOS, EOS, UNK):
                groups[strip_all(w)].append((c, w))
        self.by_base = defaultdict(set)
        for base_, lst in groups.items():
            top = max(c for c, _ in lst)
            for c, w in lst:
                if top < 100 or c >= 0.01 * top:      # dạng hiếm cạnh dạng rất phổ biến = gõ sai
                    self.by_base[base_].add(w)

    def p_uni(self, w):
        return (self.uni.get(w, 0) + self.alpha) / (self.N + self.alpha * self.V)

    def logp(self, a, b):
        if a == ENT or b == ENT:                 # thực thể bị khoá: chỉ dùng xác suất đơn
            return math.log(self.p_uni(b)) if b != ENT else 0.0
        pu = self.p_uni(b)
        c = self.ctx.get(a, 0)
        pb = self.bi.get((a, b), 0) / c if c else 0.0
        return math.log(self.lam * pb + (1 - self.lam) * pu) if c else math.log(pu)

    def logp3(self, z, a, b):
        """3-gram nội suy với 2-gram: P(b | z a)."""
        p2 = math.exp(self.logp(a, b))
        c = self.ctx2.get((z, a), 0) if ENT not in (z, a, b) else 0
        if not c:
            return math.log(p2)
        p3 = self.tri.get((z, a, b), 0) / c
        return math.log(0.6 * p3 + 0.4 * p2)

    def seq_logp(self, words, state):
        """state = (chữ trước nữa, chữ trước)."""
        z, a = state
        s = 0.0
        for w in words:
            s += self.logp3(z, a, w)
            z, a = a, w
        return s, (z, a)


class ContextNormalizer:
    """
    Dùng:
        cn = ContextNormalizer(Normalizer(terms, eng, [rulebook, readme]), lm_texts=[rulebook, readme, *train_questions])
        r = cn.normalize('teenn khach hang mua nhieuf nhaatt')
        r['text']    -> 'tên khách hàng mua nhiều nhất'
        r['unsure']  -> [(vị_trí, chữ_gốc, [ứng viên...]), ...]   # chỗ nên hỏi LLM trắc nghiệm
    """

    def __init__(self, base: Normalizer, lm_texts=(), cost_weight=1.0, accent_cost=0.3,
                 max_cands=6, unsure_margin=1.0):
        self.base = base
        # lexicon chỉ góp 1 lần đếm mỗi cụm: đủ để cụm bung ra từ viết tắt không bị phạt, nhưng chữ 'đã gặp thật' cần >= 2
        self.lm = BigramLM(list(lm_texts) + list(base.terms), weak_texts=lexicon_phrases())
        self.w, self.acc_cost = cost_weight, accent_cost
        self.max_cands, self.margin = max_cands, unsure_margin
        self.word_bonus = 4.5   # chọn bằng dò lưới trên train: câu sạch giữ 99,2%, dạng ghép/viết tắt 92%
        self.careful_accent_cost, self.careful_region_cost = 3.0, 3.5   # chọn bằng dò lưới trên train (giữ câu sạch ≥ 99%)
        self.teen1 = {tone_canon(k): v for k, v in TEEN.items() if ' ' not in k}
        lk = load_variant_lookup()
        self.compound = {k: v for k, v in lk.items() if not k.startswith('.')}
        self.dotted = {k[1:]: v for k, v in lk.items() if k.startswith('.')}
        self.teen_multi = sorted(((k, v) for k, v in TEEN.items() if ' ' in k and k != v), key=lambda kv: -len(kv[0]))
        self.soft, self.phrases = load_variant_extra() if USE_TSV_EXTRA else ({}, {})
        self.phrase_max = max((len(k) for k in self.phrases), default=0)

    def _dotted(self, w):
        """'g.nhập' (có trong bảng) -> giá nhập; 'd.thu' (không có) -> ghép 'dthu' rồi tra teencode / bảng ghép -> doanh thu.
        Không khớp gì thì giữ nguyên ('TP.HCM', 'Q.1')."""
        k = nfc(w.lower())
        if k in self.dotted:
            return self.dotted[k]
        j = tone_canon(k.replace('.', ''))
        if len(k.split('.')[0]) == 1 and not k.split('.')[1].isdigit():
            if j in self.teen1:
                return self.teen1[j]
            if j in self.compound:
                return self.compound[j]
        return w

    # ------------------------------------------------------------ ứng viên cho 1 chữ
    def candidates(self, tok, restore_accents=True, careful=False):
        """careful=True (câu phần lớn đã có dấu): chữ đã là âm tiết hợp lệ thì GIỮ NGUYÊN, chỉ sửa chữ hỏng.
        Đo trên câu train sạch: không bật chế độ này thì ~7% câu đúng bị sửa nhầm ('do' -> 'đó', 'Long' -> 'lon')."""
        low = tok.lower()
        if careful and (is_syllable(low) or low in self.lm.uni) and tone_canon(low) not in self.teen1:
            # chữ hợp lệ trong câu có dấu: chỉ cho (a) thêm dấu nếu chữ không dấu, (b) lẫn vùng miền nếu chữ chưa từng gặp;
            # cả hai đều đắt, chỉ chọn khi ngữ cảnh ủng hộ rất rõ
            opts = [([low], 0.0, 'giữ')]
            if strip_all(low) == low:
                for acc in self.lm.by_base.get(low, ()):
                    if acc != low:
                        opts.append(([acc], self.careful_accent_cost, 'thêm dấu (câu có dấu)'))
            if self.lm.real.get(low, 0) < 2:
                for w in region_variants(low):
                    if self.lm.real.get(w, 0) >= 5:
                        opts.append(([w], self.careful_region_cost, 'vùng miền'))
                for w in teen_variants(low):
                    if self.lm.real.get(w, 0) >= 5:
                        opts.append(([w], 1.5, 'teen'))
            return opts
        cands = {}

        def add(words, cost, how):
            key = tuple(words)
            if key not in cands or cands[key][0] > cost:
                cands[key] = (cost, how)

        if low in ENGLISH or (low in self.base.words and len(low) >= 3 and tone_canon(low) not in self.teen1
                              and re.fullmatch(r'[a-z]+', low) and not is_syllable(low)):
            return [([low], 0.0, 'giữ')]                     # tiếng Anh / từ đã khai báo: giữ nguyên
        if low in self.lm.uni and strip_all(low) != low:
            keep_cost = 0.0                                   # chữ có dấu đã gặp trong văn bản sạch
        elif low in self.lm.uni or low in self.base.words or is_syllable(low):
            keep_cost = 0.3 if strip_all(low) == low else 0.0  # chữ không dấu: giữ được nhưng ngữ cảnh có thể thêm dấu
        else:
            keep_cost = 3.0
        add([low], keep_cost, 'giữ')
        if tok.isupper() and len(tok) <= 4:               # COD, VAT, NV: chỉ giữ
            return [(list(k), c, h) for k, (c, h) in cands.items()]
        tc = tone_canon(low)
        is_teen = tc in self.teen1
        if is_teen:
            add(self.teen1[tc].split(), 0.5, 'teencode')
            for alt in TEEN_ALT.get(tc, []) + TEEN_ALT_FILE.get(tc, []):
                add(alt.split(), 0.6, 'teencode')
        if tc in self.compound and not is_teen:
            add(self.compound[tc].split(), 0.7, 'ghép/viết tắt')
        # mục nhiều nghĩa / kiểu luật đã sinh: ứng viên, ngữ cảnh chọn — CHỈ cho chữ lạ (không phải âm tiết hợp lệ, chưa gặp
        # trong văn bản sạch). Đo A/B trên train: thêm cho mọi chữ làm tụt teen 95,5% -> 94,8%; giới hạn chữ lạ thì không tụt.
        oov = len(low) >= 2 and not is_syllable(low) and low not in self.lm.uni
        for m in (self.soft.get(tc, ()) if oov else ()):
            if ' ' in m:
                continue      # nghĩa nhiều chữ từ mục mơ hồ ('dh' -> đơn huỷ, 'th' -> trả hàng) đổi hẳn ý câu: không dùng
            m = self.teen1.get(tone_canon(m), m)                  # nghĩa trong tsv lại là dạng teen ('nhìu') -> chuẩn ('nhiều')
            if m != low:
                add(m.split(), SOFT_COST + (0.0 if all(w in self.lm.uni for w in m.split()) else 0.8), 'từ điển tsv')
        # gõ kiểu VNI ('tie6n2' -> tiền): chỉ khi cả câu có >= 2 chữ kiểu VNI (tránh 'ca1' = ca 1 thành 'cá')
        v = vni_decode(low) if getattr(self, '_vni_sentence', False) else None
        if v:
            add([v], 0.3, 'vni')
        # viết kiểu teen chưa có trong bảng ('jờ' -> giờ, 'thík' -> thích, 'fải' -> phải)
        if not low.isascii() or not is_syllable(low) or self.lm.real.get(low, 0) < 2:
            for w in teen_variants(low):
                add([w], 0.9 + (0.0 if w in self.lm.uni else 0.8), 'teen')
            # thay chữ rồi mới ra teencode ('fantram' -> 'phantram' -> phần trăm, 'kuahang' -> 'cuahang' -> cửa hàng)
            for w in self._teen_letter_forms(low):
                tcw = tone_canon(w)
                if tcw in self.teen1:
                    add(self.teen1[tcw].split(), 1.0, 'teen+teencode')
        # lẫn chính tả vùng miền (n/l, s/x, ch/tr, d/gi/r, hỏi/ngã…): chỉ xét chữ CHƯA từng gặp trong văn bản sạch
        if self.lm.real.get(low, 0) < 2:
            for w in region_variants(low):
                if self.lm.real.get(w, 0) >= 5:
                    add([w], 1.5, 'vùng miền')
        for rank, (out, how, key) in enumerate(search_candidates_all(low, self.base.words)[: self.max_cands]):
            # key = (0 nếu trúng teencode/từ đã biết, chi phí sửa, ưu tiên telex)
            cost = 0.5 + 0.5 * key[0] + 0.5 * key[1] + 0.2 * rank
            if how.startswith('telex') and 'thiếu mũ' not in how and re.search(r'[sfrxjw]|aa|ee|oo|dd', low):
                cost -= 0.4      # gõ Telex ĐỦ phím là bằng chứng mạnh: ưu tiên giải đúng như phím gõ
            if out != low and any(w not in self.lm.uni for w in out.split()):
                cost += 1.5      # chữ chưa từng gặp trong văn bản sạch ('đươc'): chỉ chọn khi ngữ cảnh ép
            add(out.split(), cost, how)
        # khôi phục dấu theo ngữ cảnh: chỉ cho chữ KHÔNG dấu, chỉ lấy dạng đã gặp trong văn bản sạch
        for k in (list(cands) if restore_accents else []):
            if (len(k) == 1 and strip_all(k[0]) == k[0] and not (is_teen and k[0] == low)
                    and cands[k][1] not in ('teen', 'vùng miền', 'vni', 'teencode')):
                for acc in self.lm.by_base.get(k[0], ()):
                    if acc != k[0]:
                        add([acc], cands[k][0] + self.acc_cost + getattr(self, '_acc_extra', 0.0), cands[k][1] + '+thêm dấu')
        return [(list(k), c, h) for k, (c, h) in cands.items()]

    @staticmethod
    def _teen_letter_forms(low):
        """Thay chữ đầu kiểu teen (f->ph, k->c, z->d/v/gi, j->gi, w->qu) KHÔNG đòi kết quả là âm tiết (để tra bảng chữ ghép)."""
        out = []
        for pat, rep in ((r'^f', 'ph'), (r'^k', 'c'), (r'^z', 'd'), (r'^z', 'v'), (r'^j', 'gi'), (r'^w', 'qu'), (r'^dz', 'd')):
            if re.search(pat, low):
                out.append(re.sub(pat, rep, low, count=1))
        return out

    # ------------------------------------------------------------ Viterbi cả câu
    def decode(self, slots):
        """slots: list các ('fix', [(words, cost, how)...]) hoặc ('lock', text)."""
        beams = {(BOS, BOS): (0.0, [])}          # trạng thái = 2 chữ cuối (3-gram)
        margins, opt_scores = [], []
        self._opt_scores = opt_scores
        for kind, val in slots:
            new = {}
            opts = [([ENT], 0.0, 'khoá')] if kind == 'lock' else val
            local = []
            for prev, (score, path) in beams.items():
                for words, cost, how in opts:
                    lp, last = self.lm.seq_logp(words, prev)
                    # cụm nhiều chữ (viết tắt bung ra) bị mô hình ngôn ngữ phạt vì dài hơn: bù mỗi chữ thêm
                    s = score + lp - self.w * cost + self.word_bonus * (len(words) - 1)
                    local.append((s, words))
                    if last not in new or new[last][0] < s:
                        new[last] = (s, path + [(words, how)])
            if kind == 'fix':
                best_by_opt = {}
                for s, words in local:
                    k = tuple(words)
                    best_by_opt[k] = max(best_by_opt.get(k, -1e18), s)
                opt_scores.append(best_by_opt)
                top = sorted(best_by_opt.values(), reverse=True)
                margins.append(top[0] - top[1] if len(top) > 1 else 99.0)
            else:
                margins.append(99.0)
                opt_scores.append({})
            if len(new) > 64:                    # cắt chùm: giữ 64 trạng thái tốt nhất (đủ cho câu hỏi ngắn)
                new = dict(sorted(new.items(), key=lambda kv: kv[1][0], reverse=True)[:64])
            beams = new
        final = max(beams.items(), key=lambda kv: kv[1][0] + self.lm.logp3(kv[0][0], kv[0][1], EOS))
        return final[1][1], margins

    # ------------------------------------------------------------ API chính
    def normalize(self, text, forced=None):
        """forced: {vị_trí_slot: 'chữ đã chọn'} — dùng khi LLM đã chọn ở chỗ chưa chắc (xem resolve_unsure)."""
        forced = forced or {}
        store = []
        t = nfc(text)
        t = re.sub(r'[\U0001F000-\U0001FFFF☀-➿]', ' ', t)
        t = re.sub(r'([?!.,])\1+', r'\1', t)
        t = self.base._protect(t, store)
        # teencode nhiều chữ ('o đờ' -> đơn, 'xi ô đi' -> cod): thay trước khi tách chữ
        for k, v in self.teen_multi:
            t = re.sub(r'(?<!\w)' + re.escape(k) + r'(?!\w)', v, t, flags=re.I)
        # viết tắt có dấu chấm đã biết ('g.nhập', 'd.thu'); không đụng 'TP.HCM', 'Q.1' vì không có trong bảng
        t = re.sub(r'(?<![\w.])(\w\.\w+)(?![\w.])', lambda m: self._dotted(m.group(1)), t)
        parts = [p for p in re.split(r'(⟦E\d+⟧|[^\w⟦⟧]+)', t) if p]
        # câu phần lớn đã có dấu -> người viết gõ dấu: chữ hợp lệ giữ nguyên, chỉ sửa chữ hỏng
        careful = accent_ratio(t) >= 0.5
        self._vni_sentence = sum(1 for w in re.findall(r'\w+', t) if vni_decode(w)) >= 2
        self._acc_extra = 0.0
        restore = not careful
        slots, layout, origs = [], [], {}         # layout: ('slot', i) | ('sep', text)
        for p in parts:
            if p.startswith('⟦'):
                layout.append(('slot', len(slots)))
                slots.append(('lock', p))
            elif re.search(r'[^\W\d_]', p) and re.fullmatch(r'\w+', p):
                layout.append(('slot', len(slots)))
                i_slot = len(slots)
                cands = self.candidates(p, restore, careful)
                if i_slot in forced:
                    cands = [(forced[i_slot].split(), 0.0, 'llm chọn')]
                slots.append(('fix', cands))
                origs[i_slot] = p.lower()
            else:
                layout.append(('sep', p))
        # cụm nhiều chữ an toàn trong tsv: cả cụm khớp (chữ liền nhau, chỉ cách khoảng trắng) -> thêm ứng viên rẻ cho từng chữ
        if self.phrase_max >= 2:
            seq, prev_layout = [], None
            for li, item in enumerate(layout):
                if item[0] == 'slot' and slots[item[1]][0] == 'fix':
                    seq.append(item[1])
                elif item[0] == 'sep' and item[1].strip() == '':
                    continue
                else:
                    seq.append(None)
            for L in range(min(self.phrase_max, 5), 1, -1):
                for a in range(0, len(seq) - L + 1):
                    win = seq[a:a + L]
                    if None in win or any(w in forced for w in win):
                        continue
                    tgt = self.phrases.get(tuple(origs[w] for w in win))
                    if tgt:
                        for w, tw in zip(win, tgt):
                            cands = [c for c in slots[w][1] if c[0] != [tw]]
                            slots[w] = ('fix', [([tw], PHRASE_COST, 'cụm tsv')] + cands)
        path, margins = self.decode(slots)
        out, changes, unsure = [], [], []
        for item in layout:
            if item[0] == 'sep':
                out.append(item[1])
                continue
            i = item[1]
            kind, val = slots[i]
            if kind == 'lock':
                out.append(val)
                continue
            words, how = path[i]
            orig = origs[i]
            chosen = ' '.join(words)
            out.append(chosen)
            if chosen != orig:
                changes.append((orig, chosen, how))
            if margins[i] < self.margin and len(val) > 1:
                sc = self._opt_scores[i]
                # (A) luôn là chữ đang chọn; các phương án khác theo điểm cả câu
                others = [k for k in sorted(sc, key=lambda k: -sc[k]) if ' '.join(k) != chosen][:3]
                unsure.append((i, orig, [chosen] + [' '.join(k) for k in others]))
        t = ''.join(out)
        entities = []

        def back(m):
            kind, raw, canon = store[int(m.group(1))]
            entities.append((kind, raw, canon))
            return canon
        t = re.sub(r'⟦[Ee](\d+)⟧', back, t)
        t = re.sub(r'\s+', ' ', t).strip()
        return {'text': t, 'changes': changes, 'unsure': unsure, 'entities': entities}


MC_SYSTEM = ("Bạn là chuyên gia tiếng Việt, hiểu teencode, viết tắt, gõ sai dấu và tiếng địa phương trong câu hỏi bán hàng. "
             "Nhiệm vụ: chọn nghĩa đúng nhất của MỘT chữ theo ngữ cảnh câu. Chỉ trả lời đúng một chữ cái A, B, C hoặc D.")


def mc_messages(raw, sentence, word, options):
    """Tin nhắn trắc nghiệm cho LLM: câu gốc + câu đã sửa tạm + các phương án (do luật sinh, LLM chỉ được chọn)."""
    letters = 'ABCD'
    body = [f'Câu gốc người dùng gõ: «{raw}»', f'Câu đã sửa tạm: «{sentence}»',
            f'Chữ "{word}" trong câu gốc nghĩa là:']
    body += [f'({letters[i]}) {o}' for i, o in enumerate(options[:4])]
    body.append('Trả lời một chữ cái.')
    return [{"role": "system", "content": MC_SYSTEM}, {"role": "user", "content": '\n'.join(body)}]


def parse_choice(text, n_options):
    m = re.search(r'\b([A-D])\b', (text or '').strip().upper()[:20]) or re.match(r'\(?([A-D])', (text or '').strip().upper())
    if not m:
        return None
    k = 'ABCD'.index(m.group(1))
    return k if k < n_options else None


def resolve_unsure(cn, raws, ask, max_per_sentence=3):
    """Sửa chữ cho một lô câu, rồi hỏi LLM ở những chỗ chưa chắc.
    cn: ContextNormalizer; raws: list câu gốc; ask: hàm(list[messages]) -> list[str] (gọi LLM theo lô).
    Trả list kết quả normalize (đã áp lựa chọn của LLM) + log từng lựa chọn."""
    first = [cn.normalize(r) for r in raws]
    jobs = []
    for qi, (raw, r) in enumerate(zip(raws, first)):
        for slot, word, opts in r['unsure'][:max_per_sentence]:
            if len(opts) >= 2:
                jobs.append((qi, slot, word, opts, mc_messages(raw, r['text'], word, opts)))
    if not jobs:
        return first, []
    answers = ask([j[4] for j in jobs])
    forced, log = {}, []
    for (qi, slot, word, opts, _), ans in zip(jobs, answers):
        k = parse_choice(ans, len(opts))
        log.append({'q': qi, 'slot': slot, 'chu': word, 'phuong_an': opts, 'llm': (ans or '')[:10], 'chon': opts[k] if k is not None else None})
        if k is not None:
            forced.setdefault(qi, {})[slot] = opts[k]
    out = [cn.normalize(r, forced=forced[i]) if i in forced else first[i] for i, r in enumerate(raws)]
    return out, log


def mc_prompt(sentence, word, options):
    """Câu hỏi trắc nghiệm cho LLM ở chỗ chưa chắc (model chỉ trả 1 chữ cái)."""
    letters = 'ABCD'
    lines = [f'Trong câu: «{sentence}»', f'chữ "{word}" nghĩa là:']
    lines += [f'({letters[i]}) {o}' for i, o in enumerate(options[:4])]
    lines.append('Chỉ trả lời một chữ cái.')
    return '\n'.join(lines)
