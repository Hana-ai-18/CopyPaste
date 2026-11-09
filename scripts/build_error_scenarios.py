"""
Sinh MA TRẬN TÌNH HUỐNG LỖI (>= 20.000 dòng) từ bảng loại lỗi lexicon/error_catalog.yaml.

    python scripts/build_error_scenarios.py --rows 20000 --out lexicon/error_scenarios.tsv

Bảng loại lỗi (~170 loại) là thứ viết tay. File này nhân mỗi loại với số đo × kỳ × đối tượng × biến thể câu hỏi để
có danh sách tình huống CỤ THỂ dùng làm:
  - đề bài cho xưởng shop giả lập (sinh câu hỏi + đáp án đúng + đáp án sai theo đúng loại lỗi),
  - bộ kiểm cho trình biên dịch / kiểm toán (mỗi tình huống phải bị bắt),
  - danh sách việc khi viết metric chuẩn.
Mọi tên hàng / khách / số trong câu ví dụ là BỊA (không lấy từ public/private).

Cột: ma | loai | ten_loai | nhom | so_do | ky | doi_tuong | cau_vi_du | cach_doc_sai | cach_chan | luat | muc_do | nguon
"""
import argparse
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

METRICS = {  # số đo -> nhóm
    'doanh thu': 'tien', 'số đơn': 'dem', 'số lượng bán': 'luong', 'tỉ lệ huỷ đơn': 'ti_le', 'giá trị trung bình mỗi đơn': 'tien',
    'lãi gộp': 'tien', 'lãi ròng': 'tien', 'số khách': 'dem', 'số khách mới': 'dem', 'tổng nợ khách': 'no', 'tuổi nợ': 'no',
    'tiền nhập hàng': 'kho', 'giá nhập bình quân': 'kho', 'tồn kho': 'kho', 'giá trị tồn kho': 'kho', 'số mặt hàng hết hàng': 'kho',
    'số mặt hàng ế': 'kho', 'tỉ trọng doanh thu nhóm hàng': 'ti_le', 'sản phẩm bán chạy nhất': 'xep_hang',
    'top 3 khách mua nhiều nhất': 'xep_hang', 'nhân viên bán giỏi nhất': 'xep_hang', 'tỉ lệ doanh thu thuốc kê đơn': 'ti_le',
    'số đợt giao': 'dem', 'doanh thu ca sáng': 'tien', 'giá bán': 'gia', 'tăng trưởng doanh thu': 'so_sanh',
    'chênh lệch nhập trừ bán': 'kho', 'số đơn trả hàng': 'dem', 'tiền giảm giá': 'tien', 'phí giao hàng': 'tien',
}
PERIODS = ['hôm nay', 'hôm qua', 'tuần này', 'tuần trước', 'tháng này', 'tháng trước', 'tháng 3/2026', 'quý 1/2026', 'Q2',
           'năm nay', 'năm ngoái', 'từ đầu năm', '7 ngày gần nhất', '30 ngày qua', 'dịp Tết 2026', 'trước Tết', 'ngày 15/5',
           'cuối tuần trước', 'thứ 7 tuần trước', 'tháng 2']
ENTITIES = ['cả shop', 'mặt hàng "Bánh quy bơ hộp 200g"', 'nhóm "Đồ uống"', 'khách "Nguyễn Văn An"', 'nhân viên "Thu"',
            'kênh online', 'size L', 'thuốc kê đơn', 'xi măng PCB40', 'khách lẻ']
STYLE = ['chuẩn', 'không dấu', 'teencode', 'tiếng Anh lẫn', 'vùng miền']

# nhóm loại lỗi -> nhóm số đo áp dụng (None = mọi số đo)
APPLY = {
    'H01': ['so_sanh', 'tien', 'dem'], 'H02': None, 'H03': ['dem', 'xep_hang'], 'H04': ['luong', 'kho'], 'H05': ['tien', 'dem', 'so_sanh'],
    'H06': None, 'H11': ['dem', 'luong', 'xep_hang'], 'H12': ['no', 'kho'], 'H13': ['tien'], 'H14': ['xep_hang'], 'H15': ['ti_le'],
    'H17': ['xep_hang'], 'H18': ['xep_hang'], 'H19': ['dem', 'xep_hang', 'no'], 'H20': ['no'], 'H21': ['no'], 'H22': None,
}


ENT_OK = {  # đối tượng -> nhóm số đo hợp lý (tránh câu vô nghĩa như 'số mặt hàng ế của khách')
    'cả shop': None,
    'mặt hàng "Bánh quy bơ hộp 200g"': ['tien', 'luong', 'kho', 'gia', 'so_sanh', 'dem'],
    'nhóm "Đồ uống"': ['tien', 'luong', 'kho', 'ti_le', 'so_sanh', 'xep_hang'],
    'khách "Nguyễn Văn An"': ['tien', 'dem', 'no'],
    'nhân viên "Thu"': ['tien', 'dem', 'so_sanh'],
    'kênh online': ['tien', 'dem', 'ti_le', 'so_sanh'],
    'size L': ['tien', 'luong', 'ti_le'],
    'thuốc kê đơn': ['tien', 'luong', 'ti_le', 'kho'],
    'xi măng PCB40': ['tien', 'luong', 'kho', 'gia', 'dem'],
    'khách lẻ': ['tien', 'dem'],
}
# loại lỗi về quy trình / tuân thủ / code không sinh từ câu hỏi -> không đưa vào ma trận câu hỏi
SKIP_GROUPS = {'quy_trinh', 'tuan_thu'}


def q_template(t, metric, period, ent, rng):
    """Câu ví dụ có chủ đích kích hoạt loại lỗi t."""
    tid = t['id']
    if tid in ('H01', 'L08', 'T04'):
        return f"{metric} của {ent} {period} {rng.choice(['tăng', 'giảm', 'chênh'])} bao nhiêu % so với kỳ trước?"
    if tid == 'H02':
        return f"{metric} của {ent} Q{rng.randint(1, 4)} là bao nhiêu?"
    if tid in ('H03', 'L04'):
        return f"Có bao nhiêu khách vừa mua {ent} {period} và {rng.choice(['đã thanh toán đủ', 'mua từ 2 lần trở lên', 'không trả hàng'])}?"
    if tid in ('H04', 'L10'):
        return f"{period} bán được bao nhiêu {rng.choice(['thùng', 'hộp', 'cái', 'lon', 'vỉ'])} {ent}?"
    if tid == 'H11':
        return f"Có bao nhiêu mặt hàng bán {rng.choice(['từ 10 cái trở lên', 'hơn 10 cái', 'dưới 5 cái'])} {period}?"
    if tid in ('H24', 'H25'):
        return f"{metric} của {ent} trong {rng.randint(3, 30)} ngày gần nhất là bao nhiêu?"
    if tid == 'H10':
        return f"{metric} của {ent} {rng.choice(['tuần này', 'tuần trước'])} là bao nhiêu?"
    if tid in ('H22', 'H29', 'H30'):
        return f"{metric} của {ent} {rng.choice(['dịp Tết 2026', 'trước Tết 2026', 'Tết năm nay'])} so với cùng số ngày trước đó?"
    if tid in ('L01', 'K06'):
        return f"Những mặt hàng nào {rng.choice(['không bán được', 'chưa từng có đơn', 'không có ai mua'])} {period}?"
    if tid == 'L03':
        return f"{metric} {period} {rng.choice(['trừ', 'không tính', 'ngoại trừ'])} {ent} là bao nhiêu?"
    if tid == 'L05':
        return f"Khách nào mua {rng.randint(2, 4)} tháng liên tiếp {period}?"
    if tid == 'L06':
        return f"{metric} và số đơn của {ent} {period} là bao nhiêu?"
    if tid in ('L09',):
        return f"Nếu giảm giá 10% thì {metric} {period} sẽ là bao nhiêu?"
    if tid in ('U01', 'U02'):
        return f"{metric} của {ent} {rng.choice(['tháng 1/2019', 'năm 2030', 'tháng trước ngày đầu dữ liệu'])} là bao nhiêu?"
    if tid in ('U03',):
        return f"{ent} {rng.choice(['là nam hay nữ', 'ở quận nào', 'vì sao huỷ đơn'])}?"
    if tid in ('H14', 'S04', 'H18'):
        return f"{rng.choice(['Mặt hàng', 'Khách', 'Nhân viên'])} đứng {rng.choice(['thứ hai', 'đầu', 'thứ 3'])} về {metric} {period} là ai?"
    return f"{metric} của {ent} {period} là bao nhiêu?"


def styled(q, style, rng):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from vn_normalize import strip_all
    if style == 'không dấu':
        return strip_all(q)
    if style == 'teencode':
        for a, b in (('bao nhiêu', 'bn'), ('không', 'ko'), ('được', 'dc'), ('doanh thu', 'dthu'), ('tháng', 'thg'), ('khách', 'kh'),
                     ('sản phẩm', 'sp'), ('trước', 'trc'), ('hôm nay', 'hnay'), ('hôm qua', 'hqua')):
            q = q.replace(a, b)
        return q
    if style == 'tiếng Anh lẫn':
        return q.replace('doanh thu', 'revenue').replace('tổng', 'total').replace('bán chạy nhất', 'best seller')
    if style == 'vùng miền':
        return q.replace('không', 'hông').replace('vậy', 'dzậy').replace(' là bao nhiêu', ' bi nhiêu dị').replace('đắt', 'mắc')
    return q


def severity(t):
    if t['id'][0] in 'HTKNU':
        return 'cao'
    if t['id'][0] in 'DLAEM':
        return 'trung bình'
    return 'thấp'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=20000)
    ap.add_argument('--out', default=str(ROOT / 'lexicon' / 'error_scenarios.tsv'))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cat = yaml.safe_load(open(ROOT / 'lexicon' / 'error_catalog.yaml', encoding='utf-8'))
    types = [(g, t) for g, items in cat.items() for t in items]
    combos = []
    for g, t in types:
        if g in SKIP_GROUPS:
            continue
        groups = APPLY.get(t['id'])
        ms = [m for m, mg in METRICS.items() if groups is None or mg in groups]
        for m in ms:
            for p in PERIODS:
                for e in ENTITIES:
                    ok = ENT_OK[e]
                    if ok is None or METRICS[m] in ok:
                        combos.append((g, t, m, p, e))
    rng.shuffle(combos)
    # chia đều theo loại lỗi để loại nào cũng có tình huống
    per_type = {}
    for c in combos:
        per_type.setdefault(c[1]['id'], []).append(c)
    pick, seen = [], set()
    while len(pick) < a.rows and any(per_type.values()):
        for tid in sorted(per_type):
            if per_type[tid] and len(pick) < a.rows:
                g, t, m, p, e = per_type[tid].pop()
                st = rng.choice(STYLE)
                q = styled(q_template(t, m, p, e, rng), st, rng)
                key = (t['id'], q)
                if key in seen:
                    continue
                seen.add(key)
                pick.append((g, t, m, p, e, st, q))
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write('# Sinh bởi scripts/build_error_scenarios.py từ lexicon/error_catalog.yaml — KHÔNG sửa tay.\n')
        fh.write('# Câu ví dụ, tên hàng, tên khách, số liệu đều BỊA; không lấy từ public/private test.\n')
        fh.write('ma\tloai\tten_loai\tnhom\tso_do\tky\tdoi_tuong\tkieu_viet\tcau_vi_du\tcach_doc_sai\tcach_chan\tluat\tmuc_do\tnguon\n')
        for i, (g, t, m, p, e, st, q) in enumerate(pick, 1):
            row = [f'S{i:05d}', t['id'], t['ten'], g, m, p, e, st, q, str(t.get('tai_tao', '')), str(t.get('chan', '')),
                   str(t.get('luat', '')), severity(t), t.get('nguon', '')]
            fh.write('\t'.join(x.replace('\t', ' ').replace('\n', ' ') for x in row) + '\n')
    print(f'{len(pick)} tình huống từ {len({x[1]["id"] for x in pick})} loại lỗi (bỏ nhóm {sorted(SKIP_GROUPS)}) -> {a.out}')


if __name__ == '__main__':
    main()
