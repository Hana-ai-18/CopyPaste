"""
Sinh bộ teencode GIỚI TRẺ + VÙNG MIỀN (>= 10.000 dòng) — không dùng public/private test.

    python scripts/build_gioitre_dataset.py --train /path/train --out lexicon/teencode_gioitre_vungmien.tsv

Từ chuẩn: lexicon/tu_thong_dung.txt (viết tay, ~700 từ giao tiếp/mua bán/chat) + âm tiết có dấu trong câu hỏi/rulebook/README
TRAIN + nghĩa trong lexicon/gioitre_vungmien_lamtay.tsv.
Dòng làm tay (lexicon/gioitre_vungmien_lamtay.tsv) được chép vào đầu file với kiểu lam_tay_<nhóm>.

Quy tắc viết của giới trẻ (gioi_tre):  c->k, qu->w, ph->f, gi->j, d->z, v->z/dz, ch cuối->k/ck, ng cuối->g/q,
  i->j (sau phụ âm), ôi->ui, ươ->uo, u cuối->w, oa->wa, yêu->iu, kéo dài nguyên âm cuối.
Quy tắc phát âm vùng miền (vung_mien):  Nam: v->d/dz, gi->d, n<->ng cuối, t<->c cuối, ươu->ưu, ưu->iu;
  Trung: hỏi<->ngã; Bắc: l<->n, s<->x, tr<->ch, r<->d/gi.
Viết tắt giới trẻ cho cụm (viet_tat_gioi_tre): chữ cái đầu, chữ đầu + chữ cái đầu, dính liền có/không dấu.

Cột: dạng <TAB> từ_chuẩn <TAB> kiểu <TAB> an_toàn (1 = không trùng chữ thật, một nghĩa -> tra thẳng được)
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vn_normalize import is_syllable, nfc, strip_all  # noqa: E402

TONES = '̣́̀̉̃'


def _tone(w):
    d = unicodedata.normalize('NFD', w)
    return ''.join(c for c in d if c in TONES)


def _base(w):
    d = unicodedata.normalize('NFD', w)
    return unicodedata.normalize('NFC', ''.join(c for c in d if c not in TONES))


def _retone(base, tone):
    """Đặt lại dấu thanh lên nguyên âm chính (gần đúng: nguyên âm có mũ/móc, hoặc nguyên âm cuối trước phụ âm cuối)."""
    if not tone:
        return base
    vs = [i for i, c in enumerate(base) if strip_all(c) in 'aeiouy']
    if not vs:
        return base
    hat = [i for i in vs if base[i] in 'âăêôơư']
    i = hat[-1] if hat else (vs[-1] if (len(vs) == 1 or vs[-1] < len(base) - 1) else vs[-2])
    return unicodedata.normalize('NFC', base[:i + 1] + tone + base[i + 1:])


YOUTH = [
    (r'^c(?!h)', 'k'), (r'^qu', 'w'), (r'^ph', 'f'), (r'^gi', 'j'), (r'^d(?!z)', 'z'), (r'^v', 'z'), (r'^v', 'dz'),
    (r'ch$', 'k'), (r'ch$', 'ck'), (r'ng$', 'g'), (r'ng$', 'q'), (r'(?<=[bcdđghklmnprstvx])i', 'j'), (r'ôi$', 'ui'),
    (r'ươ', 'uo'), (r'(?<=[aâeêiơư])u$', 'w'), (r'^h?oa', 'wa'), (r'nh$', 'nk'), (r'^kh', 'k'), (r'^tr', 'ch'),
]
REGION = [
    (r'^v', 'd'), (r'^v', 'dz'), (r'^gi', 'd'), (r'(?<!n)n$', 'ng'), (r'ng$', 'n'), (r't$', 'c'), (r'c$', 't'),
    (r'ươu$', 'ưu'), (r'ưu$', 'iu'), (r'^n(?!g|h)', 'l'), (r'^l', 'n'), (r'^s', 'x'), (r'^x', 's'), (r'^tr', 'ch'),
    (r'^ch', 'tr'), (r'^r', 'd'), (r'^r', 'gi'), (r'inh$', 'in'), (r'ênh$', 'ơn'),
]


def variants(word):
    out = []
    base, tone = _base(word), _tone(word)
    for kind, rules in (('gioi_tre', YOUTH), ('vung_mien', REGION)):
        for pat, rep in rules:
            if re.search(pat, base):
                v = _retone(re.sub(pat, rep, base, count=1), tone)
                if v != word:
                    out.append((v, kind))
    if word == 'yêu':
        out.append(('iu', 'gioi_tre'))
    if '̉' in tone or '̃' in tone:
        out.append((_retone(base, '̃' if '̉' in tone else '̉'), 'vung_mien'))
    # hai quy tắc giới trẻ chồng nhau ('không' -> 'khôg' -> 'kôg') và viết không dấu ('cái' -> 'kái' -> 'kai')
    youth1 = [v for v, k in out if k == 'gioi_tre']
    for v1 in youth1:
        b1, t1 = _base(v1), _tone(v1)
        for pat, rep in YOUTH:
            if re.search(pat, b1):
                v2 = _retone(re.sub(pat, rep, b1, count=1), t1)
                if v2 not in (v1, word):
                    out.append((v2, 'gioi_tre'))
        if strip_all(v1) != v1:
            out.append((strip_all(v1), 'gioi_tre_khong_dau'))
    # kéo dài nguyên âm cuối kiểu chat
    m = re.search(r'([aăâeêioôơuưy])(?=[^aăâeêioôơuưy]*$)', strip_all(word))
    if m:
        i = m.start()
        out.append((word[:i + 1] + word[i] * 2 + word[i + 1:], 'gioi_tre'))
    return out


def phrase_rows(p):
    ws = p.split()
    rows = []
    if len(ws) >= 2:
        rows.append((''.join(strip_all(w)[0] for w in ws), 'viet_tat_gioi_tre'))
        rows.append((strip_all(ws[0]) + ''.join(strip_all(w)[0] for w in ws[1:]), 'viet_tat_gioi_tre'))
        rows.append((''.join(ws), 'dinh_lien'))
        rows.append((strip_all(''.join(ws)), 'dinh_lien'))
        for k, w in enumerate(ws):
            for v, kind in variants(w)[:6]:
                ph = ' '.join(ws[:k] + [v] + ws[k + 1:])
                rows.append((ph, kind))
                if kind == 'gioi_tre' and strip_all(ph) != ph:
                    rows.append((strip_all(ph), 'gioi_tre_khong_dau'))   # cả cụm gõ không dấu + một chữ kiểu teen
    else:
        rows.extend(variants(p))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--out', default='lexicon/teencode_gioitre_vungmien.tsv')
    ap.add_argument('--with_shop_terms', action='store_true', help='thêm chữ trong tên hàng/nhóm của shop TRAIN')
    ap.add_argument('--cache', default='/tmp/gioitre_cache')
    a = ap.parse_args()
    tr = Path(a.train)

    canon = []
    for line in open(ROOT / 'lexicon' / 'tu_thong_dung.txt', encoding='utf-8'):
        if not line.startswith('#'):
            canon += [nfc(w.replace('_', ' ')) for w in line.split()]
    texts = [json.loads(l)['question'] for l in open(tr / 'questions.jsonl', encoding='utf-8')]
    texts.append((tr / 'rulebook.md').read_text(encoding='utf-8'))
    texts += [p.read_text(encoding='utf-8', errors='ignore') for p in (tr / 'data').glob('*/*.md')]
    cnt = Counter()
    for t in texts:
        ws = re.findall(r'[^\W\d_]+', nfc(t).lower())
        if ws and sum(strip_all(w) != w for w in ws) / len(ws) >= 0.5:
            cnt.update(w for w in ws if is_syllable(w) and strip_all(w) != w)
    canon += [w for w, c in cnt.most_common() if c >= 2]
    if a.with_shop_terms:                                    # tên hàng/nhóm/khách trong dữ liệu TRAIN (chậm hơn)
        import datamate_pipeline as dp
        for d in sorted((tr / 'data').glob('*')):
            b = dp.Business(str(d), cache_dir=a.cache)
            for t in b.terms:
                for w in re.findall(r'[^\W\d_]+', nfc(t).lower()):
                    if is_syllable(w) and strip_all(w) != w:
                        canon.append(w)
    lam_tay = []
    for line in open(ROOT / 'lexicon' / 'gioitre_vungmien_lamtay.tsv', encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        p = line.rstrip('\n').split('\t')
        for m in p[1].split('|'):
            lam_tay.append((nfc(p[0].lower()), nfc(m.strip()), 'lam_tay_' + p[2]))
            if not p[2].startswith('dong_nghia'):
                canon.append(nfc(m.strip()))
    canon = list(dict.fromkeys(c for c in canon if c))

    rows = list(lam_tay)
    for c in canon:
        for v, kind in phrase_rows(c):
            if v and v != c:
                rows.append((nfc(v), c, kind))

    vocab = set(cnt) | {w for c in canon for w in c.split()}
    targets = defaultdict(set)
    for v, c, _ in rows:
        targets[v].add(c)
    seen, out = set(), []
    for v, c, kind in rows:
        if (v, c) in seen:
            continue
        seen.add((v, c))
        real = v in vocab or all(w in vocab for w in v.split())
        safe = int(len(targets[v]) == 1 and not real and not kind.startswith('lam_tay') and len(v) >= 3
                   and not any(is_syllable(w) for w in v.split()))     # chữ hợp lệ (dù chưa gặp) có thể là từ thật
        out.append((v, c, kind, safe))
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write('# Sinh bởi scripts/build_gioitre_dataset.py — KHÔNG sửa tay; sửa lexicon/gioitre_vungmien_lamtay.tsv hoặc tu_thong_dung.txt rồi sinh lại.\n')
        fh.write('# dạng\ttừ_chuẩn\tkiểu\tan_toàn(1 = tra thẳng được)\n')
        for r in sorted(out, key=lambda r: (r[2], r[1], r[0])):
            fh.write('\t'.join(map(str, r)) + '\n')
    k = Counter(r[2] for r in out)
    print(f'{len(out)} dòng ({sum(r[3] for r in out)} an toàn), {len(canon)} từ chuẩn')
    print(dict(sorted(k.items(), key=lambda x: -x[1])))


if __name__ == '__main__':
    main()
