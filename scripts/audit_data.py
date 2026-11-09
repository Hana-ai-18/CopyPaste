"""
Soát chất lượng xử lý DỮ LIỆU SHOP và CÂU CHỮ CÂU HỎI (chỉ dùng cho TRAIN để phát triển).

    python scripts/audit_data.py --split_dir /path/train --out audit_train.md

Báo cáo các dấu hiệu làm sạch chưa kỹ:
  D1  cột 'chữ' nhưng phần lớn giá trị trông như số/tiền/ngày  -> bỏ sót định kiểu
  D2  cột tiền lẫn độ lớn (vừa < 1.000 vừa >= 100.000)          -> có thể thiếu đơn vị k/tr
  D3  cột ngày có giá trị ngoài 2015–2030 hoặc sau as_of lớn nhất -> đọc sai ngày/tháng hoặc năm
  D4  cùng một tên sau key() nhưng viết khác nhau                -> gộp nhóm sai nếu so bằng chuỗi gốc
  D5  dòng trùng hoàn toàn                                        -> cộng trùng
  D6  cột trạng thái: liệt kê giá trị để đối chiếu luật huỷ/trả/hoàn thành/đang giao
  D7  cột đơn vị: liệt kê đơn vị (hộp/cái/thùng…) để kiểm quy đổi
  D8  ô 'hỗn hợp' kiểu khi gộp bảng                               -> các file cùng cấu trúc nhưng định dạng khác
  Q1  sửa chữ làm đổi chữ số trong câu (không được phép)
  Q2  tên hàng có trong câu gốc bị mất sau khi sửa
  Q3  tỉ lệ câu được sửa / có chỗ chưa chắc
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datamate_pipeline as dp  # noqa: E402
from vn_normalize import Normalizer  # noqa: E402
from vn_context import ContextNormalizer  # noqa: E402

STATUS_HINT = re.compile(r"(trạng thái|trang thai|tình trạng|tinh trang|status|tt\b)", re.I)
UNIT_HINT = re.compile(r"(đvt|dvt|đơn vị|don vi|unit|__dv)", re.I)
NUMLIKE = re.compile(r"^\s*-?\(?[\d.,]+\s*(k|tr\d*|đ|d|vnd|vnđ|triệu|nghìn|ngàn)?\)?\s*$", re.I)
EXPECTED_ODD_DATE = re.compile(r"(hạn|han\b|hsd|sinh|birth|due|hết hạn|het han)", re.I)   # hạn dùng/ngày sinh: ngoài khoảng là bình thường
DATELIKE = re.compile(r"^\s*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/\-.]\d{1,2}([/\-.]\d{2,4})?)")


def audit_shop(b, max_asof):
    out = []

    def add(code, table, msg):
        out.append((code, table, msg))
    for name, df in b.T.items():
        ty = b.types.get(name, {})
        if len(df) and (dup := int(df.duplicated().sum())):
            add("D5", name, f"{dup} dòng trùng hoàn toàn / {len(df)}")
        for c in df.columns:
            t = str(ty.get(c, ""))
            s = df[c].dropna()
            if not len(s):
                continue
            if t.startswith("hỗn hợp"):
                add("D8", name, f"cột '{c}': {t}")
            if t == "chữ":
                sv = s.astype(str)
                num = sv.str.match(NUMLIKE).mean()
                dat = sv.str.match(DATELIKE).mean()
                if 0.5 <= num < 1 and not dp.ID_HINT.search(str(c)):
                    bad = sv[~sv.str.match(NUMLIKE)].value_counts().head(5).index.tolist()
                    add("D1", name, f"cột '{c}' để dạng chữ nhưng {num:.0%} trông như số; ví dụ không đọc được: {bad}")
                if 0.5 <= dat < 1:
                    bad = sv[~sv.str.match(DATELIKE)].value_counts().head(5).index.tolist()
                    add("D1", name, f"cột '{c}' để dạng chữ nhưng {dat:.0%} trông như ngày; ví dụ lạ: {bad}")
                keys = defaultdict(set)
                if s.nunique() <= 5000:
                    for v in sv.unique():
                        keys[dp.key(v)].add(v)
                    var = [sorted(v) for v in keys.values() if len(v) > 1]
                    if var:
                        add("D4", name, f"cột '{c}': {len(var)} tên viết nhiều kiểu, ví dụ {var[:3]}")
                if STATUS_HINT.search(str(c)) and s.nunique() <= 30:
                    add("D6", name, f"cột '{c}' giá trị: {dict(s.value_counts().head(15))}")
                if UNIT_HINT.search(str(c)) and s.nunique() <= 30:
                    add("D7", name, f"cột '{c}' đơn vị: {dict(s.value_counts().head(15))}")
            if t == "đơn vị" and s.nunique() <= 30:
                add("D7", name, f"cột '{c}' đơn vị: {dict(s.value_counts().head(15))}")
            if t.startswith("tiền") and pd.api.types.is_numeric_dtype(s):
                pos = s[s > 0]
                # giá lẻ vài trăm đồng là thật (thuốc theo viên); nghi ngờ khi có giá trị < 100đ cạnh giá trị >= 10.000đ
                small, big = (pos < 100).mean() if len(pos) else 0, (pos >= 10_000).mean() if len(pos) else 0
                if small > 0.005 and big > 0.05:
                    add("D2", name, f"cột '{c}': {small:.1%} giá trị < 100đ và {big:.0%} giá trị >= 10.000đ (thiếu đơn vị k/tr?) "
                                    f"ví dụ nhỏ: {pos[pos < 100].head(5).tolist()}")
                if (neg := int((s < 0).sum())):
                    add("D2", name, f"cột '{c}': {neg} giá trị âm (kiểm có phải hoàn tiền/trả hàng)")
            if t.startswith("ngày") and pd.api.types.is_datetime64_any_dtype(s) and not EXPECTED_ODD_DATE.search(dp._colnorm(c)):
                lo, hi = s.min(), s.max()
                odd = s[(s.dt.year < 2015) | (s.dt.year > 2030)]
                if len(odd):
                    add("D3", name, f"cột '{c}': {len(odd)} ngày ngoài 2015–2030, ví dụ {odd.head(3).dt.date.tolist()}")
                if max_asof is not None and hi > max_asof + pd.Timedelta(days=1):
                    n_after = int((s > max_asof + pd.Timedelta(days=1)).sum())
                    add("D3", name, f"cột '{c}': {n_after} ngày sau as_of lớn nhất ({max_asof.date()}), max = {hi.date()}"
                                    f" (dữ liệu tương lai hoặc đọc nhầm ngày/tháng)")
                if ty.get(c, "").endswith("(tháng/ngày)"):
                    add("D3", name, f"cột '{c}' được đọc kiểu THÁNG/NGÀY (Mỹ), khoảng {lo.date()} → {hi.date()}: kiểm lại")
    return out


def audit_questions(split, shops, rulebook, lm_texts):
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    by = defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)
    probs, stat = [], Counter()
    for bid in shops:
        b = shops[bid]
        base = Normalizer(b.terms, ["bill", "voucher", "top", "order", "ship", "cod", "size", "zalo", "ok", "sale"], [rulebook, b.readme])
        cn = ContextNormalizer(base, [rulebook, b.readme] + lm_texts)
        tset = [t for t in b.terms if len(t) >= 4]
        for q in by.get(bid, []):
            raw = q["question"]
            r = cn.normalize(raw)
            stat["câu"] += 1
            stat["câu được sửa"] += bool(r["changes"])
            stat["câu có chỗ chưa chắc"] += bool(r["unsure"])
            d0, d1 = re.findall(r"\d+", raw), re.findall(r"\d+", r["text"])
            if d0 != d1:
                probs.append(("Q1", q["id"], f"{raw!r} -> {r['text']!r}"))
            low_t = dp.key(r["text"])          # so theo key(): đổi sang dạng có dấu của CÙNG tên không phải là mất tên
            for t in tset:
                if t.lower() in raw.lower() and dp.key(t) not in low_t:
                    probs.append(("Q2", q["id"], f"mất tên '{t}': {raw!r} -> {r['text']!r}"))
                    break
    return probs, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out", default="audit_report.md")
    ap.add_argument("--cache", default="/tmp/audit_cache")
    ap.add_argument("--lm_texts", default=None, help="câu hỏi TRAIN để học sửa chữ")
    a = ap.parse_args()
    split = a.split_dir.rstrip("/")
    rulebook = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    max_asof = defaultdict(lambda: None)
    for q in qs:
        t = pd.Timestamp(q["as_of_date"])
        if max_asof[q["business_id"]] is None or t > max_asof[q["business_id"]]:
            max_asof[q["business_id"]] = t
    shops, lines, counts = {}, [], Counter()
    for d in sorted(glob.glob(os.path.join(split, "data", "*"))):
        b = dp.Business(d, cache_dir=a.cache)
        shops[b.id] = b
        found = audit_shop(b, max_asof[b.id])
        lines.append(f"\n## {b.id}  ({len(b.T)} bảng)\n")
        for name, ns in b.notes.items():
            for n in ns:
                lines.append(f"- [ghi chú làm sạch] `{name}`: {n}")
        for code, table, msg in found:
            counts[code] += 1
            lines.append(f"- **{code}** `{table}`: {msg}")
    lm = []
    if a.lm_texts:
        lm = [json.loads(l)["question"] for l in open(a.lm_texts, encoding="utf-8")]
    probs, stat = audit_questions(split, shops, rulebook, lm)
    lines.append("\n## Câu hỏi\n")
    lines.append(f"- {dict(stat)}")
    for code, qid, msg in probs[:200]:
        counts[code] += 1
        lines.append(f"- **{code}** {qid}: {msg}")
    head = ["# Báo cáo soát dữ liệu và câu chữ", "", f"Split: `{split}`", "", "Số vấn đề theo loại: " + json.dumps(dict(counts), ensure_ascii=False), ""]
    Path(a.out).write_text("\n".join(head + lines), encoding="utf-8")
    print("Số vấn đề theo loại:", dict(counts))
    print("Câu hỏi:", dict(stat))
    print("Báo cáo:", a.out)


if __name__ == "__main__":
    main()
