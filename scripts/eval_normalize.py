"""
Đo bộ sửa chữ trên câu hỏi TRAIN bị làm bẩn có kiểm soát (không dùng public/private).

Cách đo công bằng: chia shop train làm 2 nửa. Mô hình ngôn ngữ chỉ học câu hỏi của nửa A,
rồi đo trên câu hỏi nửa B (và ngược lại). Chỉ lấy câu gốc CÓ DẤU làm đáp án chuẩn.

Các kiểu làm bẩn:
  bo_dau      : bỏ dấu CẢ CÂU ("khách hàng" -> "khach hang")      ~26% câu train thật như vậy
  telex       : gõ Telex sót (tháng -> thangs, được -> dduwowcj)
  telex_luoi  : Telex thiếu mũ/móc (nhiều -> nhieuf, người -> nguoif)
  keo_dai     : kéo dài chữ (có -> cóoooo, nhất -> nhaatstttt)
  tron        : trộn ngẫu nhiên các kiểu trên theo từng chữ
  sach        : KHÔNG làm bẩn — đo xem bộ sửa có sửa nhầm câu đúng không (phải ~100%)
  teen        : viết tắt (ko, dc, bn, dthu…) + thay chữ kiểu teen (gi->j, qu->w, ph->f, c->k, v->z)
  vni         : gõ kiểu VNI (tiền -> tie6n2)
  vung_mien   : lẫn chính tả vùng miền (n/l, s/x, ch/tr, d/gi/r, hỏi/ngã)

Chạy: python scripts/eval_normalize.py --train /path/train
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vn_normalize import TEEN, Normalizer, is_syllable, nfc, strip_all, tone_canon  # noqa: E402
from vn_context import ContextNormalizer, accent_ratio, tokenize  # noqa: E402

TONE_KEY = {'́': 's', '̀': 'f', '̉': 'r', '̃': 'x', '̣': 'j'}
HAT_KEY = {'â': 'aa', 'ê': 'ee', 'ô': 'oo', 'ă': 'aw', 'ơ': 'ow', 'ư': 'uw', 'đ': 'dd'}


def to_telex(w, lazy=False):
    d = unicodedata.normalize('NFD', w)
    tone = ''.join(TONE_KEY.get(c, '') for c in d)
    base = unicodedata.normalize('NFC', ''.join(c for c in d if c not in TONE_KEY))
    out = ''
    for c in base:
        if c in HAT_KEY:
            out += (c if c == 'đ' else strip_all(c)) if lazy and c != 'đ' else HAT_KEY[c]
        else:
            out += c
    if lazy:
        out = out.replace('đ', 'd')
    return out + tone


def elongate(w, rng):
    if len(w) < 2:
        return w
    i = rng.randrange(len(w))
    return w[:i + 1] + w[i] * rng.randint(2, 4) + w[i + 1:]


def corrupt_word(w, kind, rng):
    if strip_all(w) == w and kind != 'keo_dai':
        return w
    if kind == 'bo_dau':
        return strip_all(w)
    if kind == 'telex':
        return to_telex(w)
    if kind == 'telex_luoi':
        return to_telex(w, lazy=True)
    if kind == 'keo_dai':
        return elongate(to_telex(w) if strip_all(w) != w and rng.random() < 0.5 else w, rng)
    if kind == 'tron':
        return corrupt_word(w, rng.choice(['bo_dau', 'telex', 'telex_luoi', 'keo_dai', 'giu']), rng)
    return w


VNI_TONE = {'\u0301': '1', '\u0300': '2', '\u0309': '3', '\u0303': '4', '\u0323': '5'}
VNI_HAT = {'â': 'a6', 'ê': 'e6', 'ô': 'o6', 'ă': 'a8', 'ơ': 'o7', 'ư': 'u7', 'đ': 'd9'}


def to_vni(w):
    d = unicodedata.normalize('NFD', w)
    tone = ''.join(VNI_TONE.get(c, '') for c in d)
    base = unicodedata.normalize('NFC', ''.join(c for c in d if c not in VNI_TONE))
    return ''.join(VNI_HAT.get(c, c) for c in base) + tone


TEEN_REV = {}
for _k, _v in TEEN.items():
    if _k.isascii() or len(_k) <= 4:
        TEEN_REV.setdefault(_v, []).append(_k)
TEEN_REV = {k: v for k, v in TEEN_REV.items() if k not in v}
LETTER_TEEN = [(r'^gi', 'j'), (r'^qu', 'w'), (r'^ph', 'f'), (r'^c(?=[aáàảãạăâoóòỏõọôơuúùủũụư])', 'k'), (r'ch$', 'k'), (r'^v', 'z'), (r'^d(?=[aâăoôơuư])', 'z')]
REGION = [(r'^n(?!g|h)', 'l'), (r'^l', 'n'), (r'^s', 'x'), (r'^x', 's'), (r'^ch', 'tr'), (r'^tr', 'ch'), (r'^d', 'gi'), (r'^gi', 'd'), (r'^r', 'd')]


def corrupt_teen(sentence, rng):
    """Như người gõ thật: một số chữ bị thay kiểu teen (gi->j, ph->f, c->k…), một số cụm bị viết tắt.
    Không chồng hai kiểu lên cùng một chữ (không có 'kustomer')."""
    t = nfc(sentence).lower()
    toks = re.split(r'(\W+)', t)
    abbrev_done = set()
    phrases = sorted(TEEN_REV, key=len, reverse=True)
    for phrase in phrases:
        if rng.random() < 0.6 and re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', t):
            ab = rng.choice(TEEN_REV[phrase])
            t = re.sub(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', ab, t)
            abbrev_done.add(ab)

    def rep(m):
        w = m.group(0)
        if w in abbrev_done or w.isascii() and not re.search(r'[aeiouy]', w):
            return w
        if rng.random() < 0.3 and strip_all(w) != w:          # chỉ thay chữ tiếng Việt có dấu
            for pat, r in LETTER_TEEN:
                if re.search(pat, w):
                    return re.sub(pat, r, w, count=1)
        return w
    del toks
    return re.sub(r'[^\W\d_]+', rep, t)


def corrupt_region(sentence, rng):
    def rep(m):
        w = m.group(0).lower()
        if rng.random() < 0.3:
            for pat, r in rng.sample(REGION, len(REGION)):
                if re.search(pat, w):
                    return re.sub(pat, r, w, count=1)
            d = unicodedata.normalize('NFD', w)
            if '\u0309' in d or '\u0303' in d:
                return unicodedata.normalize('NFC', d.replace('\u0309', '#').replace('\u0303', '\u0309').replace('#', '\u0303'))
        return w
    return re.sub(r'[^\W\d_]+', rep, nfc(sentence))


def corrupt(sentence, kind, rng, p=0.6):
    if kind == 'sach':
        return sentence
    if kind == 'teen':
        return corrupt_teen(sentence, rng)
    if kind == 'vung_mien':
        return corrupt_region(sentence, rng)
    if kind == 'vni':
        return re.sub(r'[^\W\d_]+', lambda m: to_vni(m.group(0).lower()) if rng.random() < p else m.group(0).lower(), nfc(sentence))
    def rep(m):
        w = m.group(0)
        return corrupt_word(w.lower(), kind, rng) if rng.random() < p else w.lower()
    return re.sub(r'[^\W\d_]+', rep, nfc(sentence))


TEEN_KEYS = {tone_canon(k) for k, v in TEEN.items() if tone_canon(k) != tone_canon(v)}   # bỏ mục đồng nhất ('tháng' -> 'tháng')


def is_clean(q):
    """Câu gốc dùng làm đáp án phải SẠCH: có dấu, không có teencode ('đc', 'nhiu', 'vs').
    Nếu không, bộ sửa chữ sửa đúng mà vẫn bị tính sai."""
    toks = tokenize(q)
    return (accent_ratio(q) >= 0.6 and toks and
            all(len(t) > 1 and is_syllable(t) and tone_canon(t) not in TEEN_KEYS for t in toks))


def word_acc(pred, gold):
    p, g = tokenize(pred), tokenize(gold)
    if len(p) != len(g):
        return 0, len(g), 0
    ok = sum(a == b for a, b in zip(p, g))
    return ok, len(g), int(ok == len(g))


def eval_variants(path, rb, questions, rng, per_kind=400):
    """Từng dạng lệch đứng MỘT MÌNH (không có ngữ cảnh câu) -> có đúng từ chuẩn không. Kiểu bo_dau một chữ vốn mơ hồ
    (ban = bán/bàn/bạn) nên điểm thấp là bình thường; kiểu 'viet_tat/dinh_lien' an toàn là tra bảng nên gần 100%."""
    from collections import defaultdict
    rows = defaultdict(list)
    for line in open(path, encoding='utf-8'):
        if line.startswith('#'):
            continue
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4:
            rows[f"{p[2]} ({'an toàn' if p[3] == '1' else 'mơ hồ'})"].append((p[0], p[1]))
    base = Normalizer([], ['bill'], [rb])
    cn = ContextNormalizer(base, [rb] + questions)
    print(f"{'kiểu (đứng một mình)':32s} | số mẫu | đúng%")
    for kind in sorted(rows):
        sample = rng.sample(rows[kind], min(per_kind, len(rows[kind])))
        ok = sum(tokenize(cn.normalize(v)['text']) == tokenize(c) for v, c in sample)
        print(f"{kind:32s} | {len(sample):6d} | {100 * ok / len(sample):5.1f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--n', type=int, default=400, help='số câu đo mỗi nửa')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variants', default=None,
                    help='đo thêm trên lexicon/teencode_variants.tsv: sửa từng dạng lệch đứng một mình, theo kiểu')
    ap.add_argument('--with_shop', action='store_true',
                    help='giống lúc chạy thật: thêm tên hàng + README của shop vào từ vựng (chậm hơn, đọc dữ liệu shop)')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tr = Path(a.train)
    qs = [json.loads(l) for l in open(tr / 'questions.jsonl', encoding='utf-8')]
    rb = (tr / 'rulebook.md').read_text(encoding='utf-8')
    shops = sorted({q['business_id'] for q in qs})
    halves = [set(shops[0::2]), set(shops[1::2])]
    kinds = ['sach', 'bo_dau', 'telex', 'telex_luoi', 'keo_dai', 'tron', 'teen', 'vni', 'vung_mien']
    tot = {k: {m: [0, 0, 0, 0] for m in ('khong_sua', 'tung_chu', 'ngu_canh')} for k in kinds}
    for h in range(2):
        learn = [q['question'] for q in qs if q['business_id'] in halves[h]]
        test = [(q['business_id'], q['question']) for q in qs
                if q['business_id'] not in halves[h] and is_clean(q['question'])]
        rng.shuffle(test)
        test = test[: a.n]
        eng = ['bill', 'voucher', 'top', 'cod', 'ok', 'sale', 'size', 'order']
        shop_cn = {}

        def get_cn(bid):
            if bid not in shop_cn:
                terms, readme = [], ''
                if a.with_shop:
                    import datamate_pipeline as dp
                    b = dp.Business(str(tr / 'data' / bid), cache_dir='/tmp/eval_norm_cache')
                    terms, readme = b.terms, b.readme
                base = Normalizer(terms, eng, [rb, readme])
                shop_cn[bid] = (base, ContextNormalizer(base, [rb, readme] + learn))
            return shop_cn[bid]
        for kind in kinds:
            for bid, gold in test:
                base, cn = get_cn(bid)
                noisy = corrupt(gold, kind, rng, p=1.0 if kind == 'bo_dau' else 0.6)  # câu không dấu thật thường bỏ dấu cả câu
                for m, pred in (('khong_sua', noisy), ('tung_chu', base.normalize(noisy)['text']),
                                ('ngu_canh', cn.normalize(noisy)['text'])):
                    ok, n, sent = word_acc(pred, gold.lower())
                    t = tot[kind][m]
                    t[0] += ok; t[1] += n; t[2] += sent; t[3] += 1
    if a.variants:
        eval_variants(a.variants, rb, [q['question'] for q in qs], rng)
    print(f"{'kiểu bẩn':12s} | {'không sửa':>18s} | {'sửa từng chữ':>18s} | {'theo ngữ cảnh':>18s}")
    print(f"{'':12s} | {'chữ%   câu%':>18s} | {'chữ%   câu%':>18s} | {'chữ%   câu%':>18s}")
    for kind in kinds:
        row = [f'{kind:12s}']
        for m in ('khong_sua', 'tung_chu', 'ngu_canh'):
            ok, n, sent, c = tot[kind][m]
            row.append(f'{100 * ok / max(n, 1):7.1f} {100 * sent / max(c, 1):7.1f}   ')
        print(' | '.join(row))


if __name__ == '__main__':
    main()
