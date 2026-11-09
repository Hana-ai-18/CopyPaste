"""
Đo bộ sửa chữ (vn_normalize + vn_context) trên ViLexNorm (EACL 2024, 10.467 cặp câu MXH tiếng Việt — bộ dữ liệu
CÔNG KHAI, luật 6 cho phép dùng dữ liệu công khai). Giấy phép CC BY-NC-SA 4.0: script này CHỈ ĐO, không chép dữ liệu vào repo.

    git clone --depth 1 https://github.com/ngxtnhi/ViLexNorm.git /tmp/ViLexNorm
    python scripts/eval_vilexnorm.py --data /tmp/ViLexNorm/data/test.csv [--limit 1000] [--mapping out.tsv --train_csv .../train.csv]

In: độ đúng theo từ (câu có cùng số từ trước/sau chuẩn hoá) của Leave-As-Is (giữ nguyên) và của bộ sửa chữ, ERR
(Error Reduction Rate, van der Goot 2019) = (đúng_hệ − đúng_LAI) / (1 − đúng_LAI); tỉ lệ từ ĐÚNG SẴN bị sửa hỏng.
--mapping: dựng bảng ánh xạ từ-đơn (dạng viết -> chuẩn) từ phần TRAIN của ViLexNorm (tần suất ≥ 3, nhất quán ≥ 90%) để nhóm
cân nhắc thêm vào lexicon (thư mục riêng, kèm giấy phép) — chỉ thêm nếu đo trên câu hỏi train của DataMate không tụt.
"""
import argparse
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vn_normalize import Normalizer, nfc  # noqa: E402

try:
    from vn_context import ContextNormalizer  # noqa: E402
except Exception:  # pragma: no cover
    ContextNormalizer = None


def toks(s):
    return re.findall(r"\w+", nfc(s).lower())


def load(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["original"], r["normalized"]))
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_ctx", action="store_true")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--train_csv", default=None)
    a = ap.parse_args()
    rows = load(a.data, a.limit)
    nz = Normalizer()
    if not a.no_ctx and ContextNormalizer is not None:
        nz = ContextNormalizer(nz, [])
    n_tok = lai = sys_ok = broke = need = fixed = skipped = 0
    for orig, norm in rows:
        try:
            out = nz.normalize(orig)["text"]
        except Exception:
            out = orig
        to, tn, ts = toks(orig), toks(norm), toks(out)
        if not (len(to) == len(tn) == len(ts)):
            skipped += 1
            continue
        for o, g, s in zip(to, tn, ts):
            n_tok += 1
            lai += o == g
            sys_ok += s == g
            if o == g and s != g:
                broke += 1
            if o != g:
                need += 1
                fixed += s == g
    acc_lai, acc_sys = lai / max(1, n_tok), sys_ok / max(1, n_tok)
    err = (acc_sys - acc_lai) / max(1e-9, 1 - acc_lai)
    print(f"câu dùng được: {len(rows) - skipped}/{len(rows)} (bỏ {skipped} câu lệch số từ)")
    print(f"từ: {n_tok} | LAI {acc_lai:.4f} | bộ sửa chữ {acc_sys:.4f} | ERR {err:+.4f}")
    print(f"từ cần sửa: {need}, sửa đúng {fixed} ({fixed / max(1, need):.1%}) | từ đúng sẵn bị sửa hỏng: {broke} ({broke / max(1, n_tok - need):.2%})")
    if a.mapping and a.train_csv:
        cnt = collections.defaultdict(collections.Counter)
        for o, g in load(a.train_csv):
            to, tn = toks(o), toks(g)
            if len(to) == len(tn):
                for x, y in zip(to, tn):
                    cnt[x][y] += 1
        keep = []
        for x, c in cnt.items():
            y, k = c.most_common(1)[0]
            tot = sum(c.values())
            if y != x and k >= 3 and k / tot >= 0.9:
                keep.append((x, y, k, round(k / tot, 3)))
        with open(a.mapping, "w", encoding="utf-8") as fh:
            fh.write("# Dẫn xuất từ ViLexNorm (Nguyen et al., EACL 2024), CC BY-NC-SA 4.0 — giữ nguyên giấy phép khi phân phối\n")
            fh.write("viet\tchuan\tso_lan\tnhat_quan\n")
            for r in sorted(keep, key=lambda r: -r[2]):
                fh.write("\t".join(map(str, r)) + "\n")
        print(f"bảng ánh xạ: {len(keep)} dạng -> {a.mapping}")


if __name__ == "__main__":
    main()
