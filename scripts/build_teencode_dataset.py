"""
Sinh bộ dữ liệu teencode / viết lệch lớn (>= 10.000 dòng) từ từ vựng chuẩn, KHÔNG dùng public/private test.

    python scripts/build_teencode_dataset.py --train /path/train --out lexicon/teencode_variants.tsv

Nguồn từ chuẩn:
  - cụm từ nghiệp vụ viết sẵn (BUSINESS_PHRASES, ~300 cụm)
  - nghĩa trong lexicon/teencode.tsv (bảng làm tay)
  - chữ / cặp chữ hay gặp trong câu hỏi TRAIN có dấu và rulebook train

Mỗi từ chuẩn sinh nhiều dạng lệch, mỗi dòng: dạng_lệch <TAB> từ_chuẩn <TAB> kiểu <TAB> an_toàn
  kiểu: bo_dau | telex | telex_luoi | vni | keo_dai | teen | vung_mien | viet_tat | dinh_lien | dinh_lien_bo_dau | lam_tay
  an_toàn = 1 khi dạng lệch KHÔNG phải chữ/từ thật và chỉ ứng với MỘT từ chuẩn -> tra thẳng được.
  an_toàn = 0: dạng lệch trùng chữ thật hoặc nhiều nghĩa -> chỉ dùng để đo / huấn luyện, bộ sửa chữ phải xét ngữ cảnh.
  nguồn: nghiep_vu (cụm viết sẵn) | lam_tay (bảng teencode.tsv) | train_chu / train_cap (chữ / cặp chữ hay gặp trong train)
  vn_context chỉ tra thẳng: an_toàn = 1 và kiểu viet_tat / dinh_lien / dinh_lien_bo_dau (dạng mà quy tắc không tự sinh được).
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vn_normalize import TEEN, TEEN_ALT_FILE, is_syllable, nfc, strip_all  # noqa: E402

BUSINESS_PHRASES = """
doanh thu|doanh số|lợi nhuận|lãi gộp|lãi ròng|giá vốn|giá bán|giá nhập|đơn giá|thành tiền|tổng tiền|tổng cộng|tiền hàng
giảm giá|chiết khấu|khuyến mãi|phí giao|phí ship|phí phục vụ|tiền mặt|chuyển khoản|thanh toán|công nợ|còn nợ|ghi nợ
trả nợ|thu nợ|nợ thêm|tuổi nợ|quá hạn|hạn thanh toán|phiếu thu|phiếu chi|phiếu nhập|phiếu xuất|hoá đơn|đơn hàng|số đơn
đơn huỷ|đơn trả|trả hàng|hoàn hàng|hoàn tiền|đang giao|đã giao|hoàn thành|đã huỷ|bị huỷ|huỷ đơn|tỉ lệ huỷ|tỉ lệ
phần trăm|tăng trưởng|tăng|giảm|chênh lệch|so với|so sánh|cùng kỳ|kỳ trước|trung bình|bình quân|giá trị|giá trị trung bình
khách hàng|khách lẻ|khách vãng lai|khách quen|khách mới|nhóm khách|thân thiết|thành viên|nhân viên|nhân viên bán hàng
ca sáng|ca chiều|ca tối|quản lý|chủ shop|cửa hàng|chi nhánh|kênh bán|bán online|bán tại quầy|sàn thương mại điện tử
sản phẩm|mặt hàng|nhóm hàng|danh mục|mã hàng|tên hàng|món|đồ uống|topping|size|tồn kho|tồn đầu kỳ|tồn cuối kỳ
nhập hàng|nhập kho|xuất kho|nhà cung cấp|lô hàng|hạn sử dụng|hết hạn|sắp hết hạn|hết hàng|còn hàng|bán chạy|bán ế
đơn vị|đơn vị tính|đơn vị cơ bản|đơn vị lớn|quy đổi|số lượng|thùng|hộp|cái|lon|chai|gói|bao|vỉ|viên|ly|đĩa|phần|kg
hôm nay|hôm qua|hôm kia|ngày mai|tuần này|tuần trước|tuần rồi|tháng này|tháng trước|tháng rồi|năm nay|năm ngoái
năm trước|quý này|quý trước|đầu tháng|cuối tháng|đầu năm|từ đầu năm|cuối tuần|ngày thường|dịp tết|trước tết|sau tết
gần nhất|gần đây|bao nhiêu|mấy|nhiều nhất|ít nhất|cao nhất|thấp nhất|lớn nhất|nhỏ nhất|đứng đầu|xếp hạng|top
có|không|được|chưa|đã|đang|sẽ|vậy|nhé|nhỉ|ạ|hả|à|giùm|giúp|cho hỏi|cho mình hỏi|xem giúp|tính giúp|liệt kê|kể tên
thuốc kê đơn|không kê đơn|hoạt chất|thực phẩm chức năng|vật liệu xây dựng|xi măng|sắt thép|cát|gạch|sơn|cà phê
trà sữa|bánh mì|nước ngọt|bia|nước suối|đợt giao|giao hàng|vận chuyển|địa chỉ|số điện thoại|ghi chú|trạng thái
lãi|lời|lỗ|vốn|chi phí|sửa chữa|tiền điện|tiền nước|tiền nhà|lương|mặt bằng|bảo hiểm|thuế
""".replace("\n", "|").split("|")
BUSINESS_PHRASES = [nfc(p.strip()) for p in BUSINESS_PHRASES if p.strip()]

TONE_KEY = {'́': 's', '̀': 'f', '̉': 'r', '̃': 'x', '̣': 'j'}
VNI_TONE = {'́': '1', '̀': '2', '̉': '3', '̃': '4', '̣': '5'}
HAT_TELEX = {'â': 'aa', 'ê': 'ee', 'ô': 'oo', 'ă': 'aw', 'ơ': 'ow', 'ư': 'uw', 'đ': 'dd'}
HAT_VNI = {'â': 'a6', 'ê': 'e6', 'ô': 'o6', 'ă': 'a8', 'ơ': 'o7', 'ư': 'u7', 'đ': 'd9'}
LETTER_TEEN = [(r'^gi', 'j'), (r'^qu', 'w'), (r'^ph', 'f'), (r'^c(?=[^h])', 'k'), (r'ch$', 'k'), (r'^v', 'z'),
               (r'^d(?=[aăâoôơuư])', 'z'), (r'ng$', 'g'), (r'nh$', 'nk'), (r'^kh', 'k'), (r'^tr', 'ch'), (r'^gi', 'z')]
REGION = [(r'^n(?!g|h)', 'l'), (r'^l', 'n'), (r'^s', 'x'), (r'^x', 's'), (r'^ch', 'tr'), (r'^tr', 'ch'),
          (r'^d', 'gi'), (r'^gi', 'd'), (r'^r', 'd'), (r'^r', 'g'), (r'^v', 'd'), (r'c$', 't'), (r't$', 'c'), (r'n$', 'ng'), (r'ng$', 'n')]


def _split_tone(w):
    d = unicodedata.normalize('NFD', w)
    tone = ''.join(c for c in d if c in TONE_KEY)
    base = unicodedata.normalize('NFC', ''.join(c for c in d if c not in TONE_KEY))
    return base, tone


def telex(w, lazy=False):
    base, tone = _split_tone(w)
    out = ''.join((strip_all(c) if lazy and c != 'đ' else HAT_TELEX[c]) if c in HAT_TELEX else c for c in base)
    if lazy:
        out = out.replace('đ', 'd')
    return out + ''.join(TONE_KEY[t] for t in tone)


def vni(w):
    base, tone = _split_tone(w)
    return ''.join(HAT_VNI.get(c, c) for c in base) + ''.join(VNI_TONE[t] for t in tone)


def word_variants(w, rng):
    """Dạng lệch cho MỘT chữ."""
    out = []
    if strip_all(w) != w:
        out.append((strip_all(w), 'bo_dau'))
        out.append((telex(w), 'telex'))
        lz = telex(w, lazy=True)
        if lz != telex(w):
            out.append((lz, 'telex_luoi'))
        out.append((vni(w), 'vni'))
    if len(w) >= 2:
        i = max(i for i, c in enumerate(w) if c.isalpha())
        out.append((w + w[i] * rng.randint(2, 4), 'keo_dai'))
        vs = [i for i, c in enumerate(w) if strip_all(c) in 'aeiouy']
        if vs:
            j = rng.choice(vs)
            out.append((w[:j + 1] + w[j] * rng.randint(1, 3) + w[j + 1:], 'keo_dai'))
    for pat, rep in LETTER_TEEN:
        if re.search(pat, strip_all(w)):
            v = re.sub(pat, rep, w, count=1) if re.search(pat, w) else None
            if v and v != w:
                out.append((v, 'teen'))
    for pat, rep in REGION:
        if re.search(pat, w):
            v = re.sub(pat, rep, w, count=1)
            if v != w:
                out.append((v, 'vung_mien'))
    base, tone = _split_tone(w)
    if '̉' in tone or '̃' in tone:
        d = unicodedata.normalize('NFD', w)
        out.append((unicodedata.normalize('NFC', d.replace('̉', '#').replace('̃', '̉').replace('#', '̃')), 'vung_mien'))
    return out


def phrase_variants(p, rng):
    """Dạng lệch cho CỤM nhiều chữ: viết tắt, dính liền, dính liền bỏ dấu, và lệch từng chữ."""
    ws = p.split()
    out = []
    if len(ws) >= 2:
        ini = ''.join(strip_all(x)[0] for x in ws)
        out.append((ini, 'viet_tat'))                                        # dt, kh, nv
        out.append((strip_all(ws[0])[0] + strip_all(ws[1]) + ''.join(strip_all(x)[0] for x in ws[2:]), 'viet_tat'))  # dthu
        out.append((strip_all(ws[0]) + ''.join(strip_all(x)[0] for x in ws[1:]), 'viet_tat'))      # doanht
        out.append((''.join(ws), 'dinh_lien'))                               # doanhthu (có dấu)
        out.append((strip_all(''.join(ws)), 'dinh_lien_bo_dau'))             # doanhthu
        out.append((ws[0][0] + '.' + ws[1], 'viet_tat'))                     # d.thu
        k = rng.randrange(len(ws))
        for v, kind in word_variants(ws[k], rng)[:6]:
            out.append((' '.join(ws[:k] + [v] + ws[k + 1:]), kind))
    else:
        out.extend(word_variants(p, rng))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--out', default='lexicon/teencode_variants.tsv')
    ap.add_argument('--min_rows', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tr = Path(a.train)
    qs = [json.loads(l)['question'] for l in open(tr / 'questions.jsonl', encoding='utf-8')]
    rb = (tr / 'rulebook.md').read_text(encoding='utf-8')

    # từ chuẩn: chỉ lấy từ câu CÓ DẤU (câu không dấu/gõ hỏng không phải chuẩn)
    uni, bi = Counter(), Counter()
    for t in qs + rb.split('\n'):
        ws = re.findall(r'[^\W\d_]+', nfc(t).lower())
        if not ws or sum(strip_all(w) != w for w in ws) / len(ws) < 0.5:
            continue
        ws = [w for w in ws if is_syllable(w)]
        uni.update(ws)
        bi.update(' '.join(p) for p in zip(ws, ws[1:]))
    src = {}
    for p in BUSINESS_PHRASES:
        src.setdefault(p, 'nghiep_vu')
    for p in [v for v in TEEN.values() if re.fullmatch(r'[^\W\d_]+( [^\W\d_]+)*', v)] + [m for ms in TEEN_ALT_FILE.values() for m in ms]:
        src.setdefault(p, 'lam_tay')
    for w, c in uni.most_common(1500):
        if c >= 3:
            src.setdefault(w, 'train_chu')
    for b, c in bi.most_common(1500):
        x, y = b.split()
        if c >= 5 and x != y:
            src.setdefault(b, 'train_cap')
    canon = list(src)

    rows = []
    for k, v in TEEN.items():                               # bảng làm tay
        if k != v:
            rows.append((k, v, 'lam_tay'))
        for m in TEEN_ALT_FILE.get(k, []):
            rows.append((k, m, 'lam_tay'))
    rounds = 0
    while True:
        for c in canon:
            for v, kind in phrase_variants(c, rng):
                v = nfc(v)
                # viết tắt chỉ sinh cho cụm nghiệp vụ / bảng làm tay (cặp chữ ghép ngẫu nhiên từ train không ai viết tắt)
                if kind == 'viet_tat' and src[c] == 'train_cap':
                    continue
                if v and v != c:
                    rows.append((v, c, kind))
        rounds += 1
        uniq = {(r[0], r[1]) for r in rows}
        if len(uniq) >= a.min_rows or rounds >= 5:
            break

    # độ an toàn để tra thẳng
    targets = defaultdict(set)
    for v, c, _ in rows:
        targets[v].add(c)
    vocab = set(uni) | {w for p in canon for w in p.split()}
    seen, out = set(), []
    for v, c, kind in rows:
        if (v, c) in seen:
            continue
        seen.add((v, c))
        is_real = all(is_syllable(w) and w in vocab for w in v.split()) or v in vocab
        safe = int(len(targets[v]) == 1 and not is_real and kind != 'lam_tay' and len(v) >= 3)
        out.append((v, c, kind, safe, src.get(c, 'lam_tay')))
    out.sort(key=lambda r: (r[2], r[1], r[0]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write('# Sinh bởi scripts/build_teencode_dataset.py — KHÔNG sửa tay; sửa lexicon/teencode.tsv rồi sinh lại.\n')
        fh.write('# dạng_lệch\ttừ_chuẩn\tkiểu\tan_toàn(1 = tra thẳng được)\tnguồn_từ_chuẩn(nghiep_vu|lam_tay|train_chu|train_cap)\n')
        for r in out:
            fh.write('\t'.join(map(str, r)) + '\n')
    kinds = Counter(r[2] for r in out)
    print(f'{len(out)} dòng, {sum(r[3] for r in out)} dòng an toàn, {len(canon)} từ chuẩn')
    print(dict(kinds))


if __name__ == '__main__':
    main()
