"""
Mở rộng 2 bộ sinh tự động bằng TỪ ĐIỂN TỪ THÔNG DỤNG (Viet22K, ~22.000 từ, Hồ Ngọc Đức — giấy phép GPL-2.0).

    git clone --depth 1 https://github.com/duyet/vietnamese-wordlist /tmp/vnwl
    python scripts/build_wordlist_variants.py --wordlist /tmp/vnwl/Viet22K.txt --extra /tmp/vnwl/Viet74K.txt --rows 20000

Ra 3 file trong lexicon/gpl/ (TÁCH RIÊNG vì giấy phép GPL; xoá thư mục này thì pipeline vẫn chạy):
  teencode_variants_tudien.tsv          +20.000 dòng: bỏ dấu / Telex / Telex thiếu mũ / VNI / kéo dài / teen / vùng miền / viết tắt / dính liền
  teencode_gioitre_vungmien_tudien.tsv  +20.000 dòng: kiểu viết giới trẻ, phát âm vùng miền, viết tắt giới trẻ, dính liền
  tu_dien_22k.txt                       từ chuẩn (đưa vào mô hình ngôn ngữ với 1 lần đếm mỗi từ)
Cột giống file gốc: dạng <TAB> từ_chuẩn <TAB> kiểu <TAB> an_toàn (1 = không phải âm tiết hợp lệ, không trùng từ nào, một nghĩa).
"""
import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from vn_normalize import is_syllable, nfc, strip_all  # noqa: E402
import build_teencode_dataset as B1  # noqa: E402
import build_gioitre_dataset as B2  # noqa: E402

HEADER = ('# Sinh bởi scripts/build_wordlist_variants.py từ Viet22K (Hồ Ngọc Đức, https://github.com/duyet/vietnamese-wordlist).\n'
          '# Giấy phép dữ liệu nguồn: GNU GPL v2 — file này là sản phẩm phái sinh, phân phối theo GPL v2.\n'
          '# dạng\ttừ_chuẩn\tkiểu\tan_toàn\n')


def read_words(path):
    out = []
    for line in open(path, encoding='utf-8', errors='ignore'):
        w = nfc(line.strip().lower())
        if w and all(is_syllable(x) for x in w.split()) and len(w.split()) <= 4:
            out.append(w)
    return list(dict.fromkeys(out))


def finish(rows, vocab, existing, n, rng, path):
    targets = defaultdict(set)
    for v, c, _ in rows:
        targets[v].add(c)
    seen, out = set(existing), []
    for v, c, kind in rows:
        if (v, c) in seen or not v or v == c:
            continue
        seen.add((v, c))
        real = v in vocab or all(w in vocab for w in v.split())
        safe = int(len(targets[v]) == 1 and not real and len(v) >= 3 and not any(is_syllable(w) for w in v.split()))
        out.append((v, c, kind, safe))
    # lấy đều theo kiểu cho đủ n dòng
    by = defaultdict(list)
    for r in out:
        by[r[2]].append(r)
    for k in by:
        rng.shuffle(by[k])
    pick, i = [], 0
    while len(pick) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(pick) < n:
                pick.append(by[k].pop())
        i += 1
    pick.sort(key=lambda r: (r[2], r[1], r[0]))
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(HEADER)
        for r in pick:
            fh.write('\t'.join(map(str, r)) + '\n')
    print(path.name, len(pick), 'dòng,', sum(r[3] for r in pick), 'an toàn |', dict(Counter(r[2] for r in pick)))


def existing_pairs(path):
    s = set()
    if path.exists():
        for line in open(path, encoding='utf-8'):
            if not line.startswith('#'):
                p = line.rstrip('\n').split('\t')
                if len(p) >= 2:
                    s.add((p[0], p[1]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wordlist', required=True, help='Viet22K.txt (từ thông dụng: nguồn từ chuẩn)')
    ap.add_argument('--extra', default=None, help='Viet74K.txt (chỉ dùng để biết đâu là TỪ THẬT khi xét an toàn)')
    ap.add_argument('--rows', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    words = read_words(a.wordlist)
    vocab_words = set(words) | (set(read_words(a.extra)) if a.extra else set())
    vocab = vocab_words | {s for w in vocab_words for s in w.split()}
    out_dir = ROOT / 'lexicon' / 'gpl'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'tu_dien_22k.txt').write_text(HEADER.split('# dạng')[0] + '\n'.join(words) + '\n', encoding='utf-8')
    print('từ chuẩn:', len(words))

    r1 = []
    for w in words:
        for v, kind in B1.phrase_variants(w, rng):
            r1.append((nfc(v), w, kind))
    finish(r1, vocab, existing_pairs(ROOT / 'lexicon' / 'teencode_variants.tsv'), a.rows, rng,
           out_dir / 'teencode_variants_tudien.tsv')
    r2 = []
    for w in words:
        for v, kind in B2.phrase_rows(w):
            r2.append((nfc(v), w, kind))
    finish(r2, vocab, existing_pairs(ROOT / 'lexicon' / 'teencode_gioitre_vungmien.tsv'), a.rows, rng,
           out_dir / 'teencode_gioitre_vungmien_tudien.tsv')
    (out_dir / 'LICENSE-GPL-2.0.txt').write_text(
        'Các file trong thư mục này phái sinh từ Viet22K/Viet74K (Hồ Ngọc Đức), giấy phép GNU GPL v2.\n'
        'Toàn văn: https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt\n'
        'Nếu không muốn ràng buộc GPL: xoá thư mục lexicon/gpl/ — pipeline tự bỏ qua.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
