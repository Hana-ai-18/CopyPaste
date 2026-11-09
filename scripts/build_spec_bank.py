"""
Dựng kho ví dụ đặc tả lexicon/spec_bank.jsonl từ TRAIN (luật 6: chỉ train).

    python scripts/build_spec_bank.py --train /path/train [--cache /tmp/c] [--out lexicon/spec_bank.jsonl]

Mỗi dòng: câu train + as_of + đặc tả (dạng JSON giống LLM phải viết) — CHỈ giữ câu mà máy tính theo đặc tả đó
ra ĐÚNG đáp án train. Dùng làm ví dụ mẫu (few-shot) cho LLM viết đặc tả và để đo khả năng đoán dạng câu.
Kèm đánh giá: đoán intent bằng câu train gần nhất của SHOP KHÁC (leave-one-shop-out) — đo độ tổng quát.
"""
import argparse
import collections
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd  # noqa: E402

import compiler as C  # noqa: E402
import datamate_pipeline as d  # noqa: E402
from retriever import Retriever  # noqa: E402

FIELDS = ("tu", "den", "X", "thang", "mat_hang", "don_vi", "n", "N", "khach", "loc", "ti_trong", "ti_le", "theo_hang", "dem")


def to_json_spec(sp):
    out = {"intent": sp["intent"]}
    for k in FIELDS:
        v = sp.get(k)
        if v in (None, False, {}, []):
            continue
        if isinstance(v, pd.Timestamp):
            v = str(v.date())
        if k == "thang":
            v = [f"{y:04d}-{m:02d}" for y, m in v]
        out[k] = v
    out["doc_khac"] = []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lexicon", "spec_bank.jsonl"))
    a = ap.parse_args()
    qs = [json.loads(x) for x in open(os.path.join(a.train, "questions.jsonl"), encoding="utf-8")]
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8")}
    bank = []
    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)
    for bid, items in sorted(by.items()):
        B = d.Business(os.path.join(a.train, "data", bid), cache_dir=a.cache)
        for q in items:
            try:
                sp = C.spec_rules(q["question"], q["as_of_date"], B)
                ans = C.compile_spec(sp, B, q["as_of_date"]) if sp else None
            except Exception:
                continue
            if ans is None or sp["intent"] == "unanswerable" or d.score_one(ans, gold[q["id"]]) != 1:
                continue
            bank.append({"question": q["question"], "as_of": q["as_of_date"], "biz": bid, "intent": sp["intent"], "spec": to_json_spec(sp)})
    with open(a.out, "w", encoding="utf-8") as f:
        for b in bank:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"kho đặc tả: {len(bank)} câu, {len(set(b['intent'] for b in bank))} intent -> {a.out}")
    # đánh giá tổng quát: đoán intent bằng câu gần nhất của SHOP KHÁC
    R = Retriever([b["question"] for b in bank], bank)
    ok = tot = 0
    conf = collections.Counter()
    for i, b in enumerate(bank):
        top = R.topk(b["question"], 3, exclude=lambda m, bid=b["biz"]: m["biz"] == bid)
        votes = collections.Counter(bank[j]["intent"] for _, j in top)
        guess = votes.most_common(1)[0][0]
        tot += 1
        ok += guess == b["intent"]
        if guess != b["intent"]:
            conf[(b["intent"], guess)] += 1
    print(f"đoán intent bằng câu train shop khác (top-3 bỏ phiếu, {R.kind}): {ok}/{tot} = {ok / tot:.3f}")
    print("nhầm nhiều nhất:", conf.most_common(6))


if __name__ == "__main__":
    main()
