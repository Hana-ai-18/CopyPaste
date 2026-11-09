"""
Đo độ bền của đường máy tính với CÁCH GHI KỲ viết tắt — chỉ dùng TRAIN (luật 6).

Lấy câu train có đúng MỘT cụm 'tháng M/YYYY' (hoặc 'tháng M năm YYYY'), viết lại cụm đó theo các kiểu gõ tắt hay gặp
('T5/2026', 'Th5/26', '5/26', '5-2026', '05/2026', 'thg 5-26', và 'tháng 5' / 'th5' không ghi năm khi năm suy ra được
theo rulebook §2), rồi so đáp án máy tính với đáp án train. Kiểu viết lại nào làm tụt đúng / tăng sai là lỗ hổng.

    python scripts/eval_period_variants.py --train /path/train --cache /tmp/c [--norm] [--limit_biz 30]
"""
import argparse
import collections
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd  # noqa: E402

import compiler as C  # noqa: E402
import datamate_pipeline as d  # noqa: E402

MONTH_PAT = re.compile(r"(?i)\b(tháng|thang|thangs|th)\s*(\d{1,2})\s*(?:/|\s+năm\s+|\s+nam\s+|\s+nawm\s+)(\d{4})\b")


def year_for_month_without_year(mo, asof):
    """Định nghĩa ĐỘC LẬP với compiler (để bài đo không tự chấm mình, và chạy được cả trên code cũ):
    rulebook §2 — tháng không ghi năm là của năm nay nếu không sau hôm nay, ngược lại năm trước."""
    asof = pd.Timestamp(asof).normalize()
    return asof.year - 1 if pd.Timestamp(asof.year, mo, 1) > asof else asof.year


def variants(q, as_of):
    ms = list(MONTH_PAT.finditer(q))
    if len(ms) != 1:
        return {}
    m = ms[0]
    mo, y = int(m.group(2)), int(m.group(3))
    yy = y % 100
    forms = {
        "T5/2026": f"T{mo}/{y}", "Th5/26": f"Th{mo}/{yy:02d}", "5/26": f"{mo}/{yy:02d}", "5-2026": f"{mo}-{y}",
        "05/2026": f"{mo:02d}/{y}", "thg 5-26": f"thg {mo}-{yy:02d}", "th 5/2026": f"th {mo}/{y}",
    }
    a = pd.Timestamp(as_of)
    if year_for_month_without_year(mo, a) == y:                 # năm suy ra theo rulebook trùng năm thật -> được bỏ năm
        forms["tháng 5 (không năm)"] = f"tháng {mo}"
        forms["th5 (không năm)"] = f"th{mo}"
    return {k: q[:m.start()] + v + q[m.end():] for k, v in forms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--show_bad", type=int, default=25)
    ap.add_argument("--per_shop", type=int, default=0, help="chỉ lấy N câu đầu (theo id) mỗi shop để đo nhanh; 0 = tất cả")
    a = ap.parse_args()
    qs = [json.loads(x) for x in open(os.path.join(a.train, "questions.jsonl"), encoding="utf-8")]
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8")}
    rb = open(os.path.join(a.train, "rulebook.md"), encoding="utf-8").read()
    d.RULE_PARAMS, _ = d.extract_rule_params(rb)
    C.WEEK_START = d.RULE_PARAMS.get("week_start", 0)
    by = collections.defaultdict(list)
    for q in qs:
        if variants(q["question"], q["as_of_date"]):
            by[q["business_id"]].append(q)
    res = collections.defaultdict(collections.Counter)
    bad = []
    for bid, items in sorted(by.items()):
        if a.per_shop:
            items = sorted(items, key=lambda q: q["id"])[:a.per_shop]
        print(f"[{bid}] {len(items)} câu", flush=True)
        B = d.Business(os.path.join(a.train, "data", bid), cache_dir=a.cache)
        nz = None
        if a.norm and d.Normalizer:
            nz = d.Normalizer(B.terms, ["bill", "voucher", "top", "order", "ship", "cod", "size", "zalo", "ok", "sale"], [rb, B.readme])
            if d.ContextNormalizer is not None:
                nz = d.ContextNormalizer(nz, [rb, B.readme])
        for q in items:
            forms = {"(gốc)": q["question"], **variants(q["question"], q["as_of_date"])}
            for name, text in forms.items():
                qn = None
                if nz is not None:
                    try:
                        qn = nz.normalize(text)["text"]
                    except Exception:
                        qn = None
                try:
                    sp = C.spec_rules(text, q["as_of_date"], B, qn)
                    ans = C.compile_spec(sp, B, q["as_of_date"]) if sp else None
                except Exception as e:
                    ans = None
                    res[name]["lỗi"] += 1
                if ans is None or ans.get("_tie"):
                    res[name]["không nhận"] += 1
                    continue
                s = d.score_one(ans, gold[q["id"]])
                lab = {1: "đúng", -1: "sai", 0: "bỏ trống"}[s]
                res[name][lab] += 1
                if s == -1 and name != "(gốc)":
                    bad.append((name, bid, text[:90], str(gold[q["id"]].get("answer"))[:40], str(ans.get("answer"))[:40]))
    print(f"{'kiểu ghi kỳ':<22} {'đúng':>6} {'sai':>5} {'không nhận':>11}")
    for name, c in res.items():
        print(f"{name:<22} {c['đúng']:>6} {c['sai']:>5} {c['không nhận']:>11}")
    for x in bad[:a.show_bad]:
        print("  ✗", x)


if __name__ == "__main__":
    main()
