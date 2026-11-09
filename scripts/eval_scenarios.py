"""
Dùng lexicon/error_scenarios.tsv (20.000 câu ví dụ BỊA, 5 kiểu viết: chuẩn / không dấu / tiếng Anh lẫn / vùng miền / teencode)
để soát BƯỚC 3: sau khi sửa chữ, luật chữ có nhận NHẦM dạng câu không. Không cần đáp án, không dùng public/private.

    python scripts/eval_scenarios.py --train /path/train [--biz biz_106] [--n 4000]

Mỗi câu có 'so_do' (đại lượng được hỏi). Nhận nhầm = luật chữ trả một intent KHÁC đại lượng đó (nguy hiểm: máy tính sẽ trả lời
tự tin một con số sai). Không nhận (None) là an toàn — câu chuyển sang đường model viết code.
"""
import argparse
import collections
import csv
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import compiler as C  # noqa: E402
import datamate_pipeline as d  # noqa: E402
from vn_context import ContextNormalizer  # noqa: E402
from vn_normalize import Normalizer  # noqa: E402

# đại lượng -> intent được phép (None / unanswerable luôn được phép)
ALLOWED = {
    "doanh thu": {"doanh_thu"}, "giá trị trung bình mỗi đơn": {"doanh_thu_binh_quan_don"}, "lãi gộp": {"lai_gop"},
    "lãi ròng": {"lai_rong"}, "số đơn": {"so_don"}, "tỉ lệ huỷ đơn": {"so_don_huy"}, "chênh lệch nhập trừ bán": {"nhap_tru_ban"},
    "giá nhập bình quân": {"gia_nhap_binh_quan"}, "giá trị tồn kho": {"gia_tri_ton_kho"}, "tăng trưởng doanh thu": {"doanh_thu_doi"},
    "tiền nhập hàng": {"tien_nhap"}, "sản phẩm bán chạy nhất": {"ban_chay"}, "tuổi nợ": {"tuoi_no"}, "tổng nợ khách": {"tong_no", "so_khach_no", "top_khach_no"},
    "số mặt hàng hết hàng": {"so_mat_hang_het_hang"}, "tồn kho": {"ton", "ton_don_vi_lon"}, "số lượng bán": {"so_luong_ban"},
    "tỉ trọng doanh thu nhóm hàng": {"doanh_thu"}, "tỉ lệ doanh thu thuốc kê đơn": {"doanh_thu"},
    "top 3 khách mua nhiều nhất": {"khach_mua_nhieu_tien_nhat"},
}
OK_ALWAYS = {None, "unanswerable"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--biz", default="biz_106")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lexicon", "error_scenarios.tsv")
    rows = list(csv.DictReader([x for x in open(here, encoding="utf-8") if not x.startswith("#")], delimiter="\t"))
    random.Random(0).shuffle(rows)
    rows = rows[: a.n]
    rb = open(os.path.join(a.train, "rulebook.md"), encoding="utf-8").read()
    B = d.Business(os.path.join(a.train, "data", a.biz), cache_dir=a.cache)
    cn = ContextNormalizer(Normalizer(B.terms, ["bill", "voucher", "top", "order", "ship", "cod", "size", "zalo", "ok", "sale"], [rb, B.readme]), [rb, B.readme])
    by_kind = collections.defaultdict(collections.Counter)
    bad = []
    for r in rows:
        q, sd, kv = r["cau_vi_du"], r["so_do"], r["kieu_viet"]
        qn = cn.normalize(q)["text"]
        try:
            sp = C.spec_rules(q, "2026-06-15", B, qn)
        except Exception as e:
            sp = {"intent": f"LỖI {e}"}
        it = (sp or {}).get("intent")
        allowed = ALLOWED.get(sd, set()) | OK_ALWAYS
        if it in OK_ALWAYS:
            lab = "khong_nhan"
        elif it in allowed:
            lab = "dung_dang"
        else:
            lab = "NHAM"
            bad.append((kv, sd, it, q[:70], qn[:70]))
        by_kind[kv][lab] += 1
    print(f"{'kiểu viết':<16} {'đúng dạng':>10} {'không nhận':>11} {'NHẬN NHẦM':>10}")
    for kv, c in sorted(by_kind.items()):
        n = sum(c.values())
        print(f"{kv:<16} {c['dung_dang'] / n:10.1%} {c['khong_nhan'] / n:11.1%} {c['NHAM'] / n:10.1%}")
    tot = collections.Counter(x[1] + " -> " + str(x[2]) for x in bad)
    print("\nnhầm hay gặp:", tot.most_common(12))
    for x in bad[: a.show]:
        print("  ✗", x)


if __name__ == "__main__":
    main()
