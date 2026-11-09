"""
Độ bền của ĐƯỜNG MÁY TÍNH khi câu hỏi bị viết bẩn (teencode, bỏ dấu, telex, vùng miền…) — chỉ TRAIN.

    python scripts/eval_spec_noise.py --train /path/train [--cache /tmp/c] [--n 300]

Lấy câu train SẠCH mà máy tính đã trả lời đúng, làm bẩn theo từng kiểu (bộ sinh của eval_normalize),
sửa chữ (ContextNormalizer học trên NỬA SHOP KHÁC), rồi chạy lại đặc tả + máy tính và so với đáp án train.
In: % vẫn nhận ra + đúng, % nhận ra nhưng SAI (nguy hiểm), % không nhận (về đường code).
"""
import argparse
import collections
import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import compiler as C  # noqa: E402
import datamate_pipeline as d  # noqa: E402
from eval_normalize import corrupt, is_clean  # noqa: E402
from vn_context import ContextNormalizer  # noqa: E402
from vn_normalize import Normalizer  # noqa: E402

KINDS = ["sach", "bo_dau", "telex", "keo_dai", "tron", "teen", "vung_mien"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    qs = [json.loads(x) for x in open(os.path.join(a.train, "questions.jsonl"), encoding="utf-8")]
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8")}
    rb = open(os.path.join(a.train, "rulebook.md"), encoding="utf-8").read()
    shops = sorted({q["business_id"] for q in qs})
    half = {s: i % 2 for i, s in enumerate(shops)}
    BS, CN = {}, {}

    def biz(bid):
        if bid not in BS:
            BS[bid] = d.Business(os.path.join(a.train, "data", bid), cache_dir=a.cache)
        return BS[bid]

    def cn(bid):
        if bid not in CN:
            B = biz(bid)
            learn = [q["question"] for q in qs if half[q["business_id"]] != half[bid]]
            base = Normalizer(B.terms, ["bill", "voucher", "top", "order", "ship", "cod", "size", "zalo", "ok", "sale"], [rb, B.readme])
            CN[bid] = ContextNormalizer(base, [rb, B.readme] + learn)
        return CN[bid]

    # câu sạch máy tính trả lời đúng
    pool = []
    for q in qs:
        if not is_clean(q["question"]):
            continue
        B = biz(q["business_id"])
        sp = C.spec_rules(q["question"], q["as_of_date"], B)
        ans = C.compile_spec(sp, B, q["as_of_date"]) if sp else None
        if ans is not None and d.score_one(ans, gold[q["id"]]) == 1:
            pool.append(q)
    rng.shuffle(pool)
    pool = pool[: a.n]
    print(f"{len(pool)} câu sạch máy tính đang trả lời đúng")
    print(f"{'kiểu bẩn':<10} | {'đúng':>6} | {'SAI':>6} | {'không nhận':>10}")
    for kind in KINDS:
        c = collections.Counter()
        for q in pool:
            B = biz(q["business_id"])
            noisy = corrupt(q["question"], kind, rng, p=1.0 if kind == "bo_dau" else 0.6)
            try:
                fixed = cn(q["business_id"]).normalize(noisy)["text"]
                sp = C.spec_rules(noisy, q["as_of_date"], B, fixed)
                ans = C.compile_spec(sp, B, q["as_of_date"]) if sp else None
            except Exception:
                ans, sp = None, None
            if ans is None:
                c["khong"] += 1
            else:
                s = d.score_one(ans, gold[q["id"]])
                c["dung" if s == 1 else "sai"] += 1
                if s != 1 and sp and sp.get("tin_cay") == "cao" and c["sai"] <= 3:
                    print(f"     ✗ [{kind}] {noisy[:70]} -> {fixed[:70]} | {sp['intent']}")
        n = len(pool)
        print(f"{kind:<10} | {c['dung'] / n:6.1%} | {c['sai'] / n:6.1%} | {c['khong'] / n:10.1%}")


if __name__ == "__main__":
    main()
