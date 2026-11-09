"""
Đo trên GPU: LLM viết ĐẶC TẢ (bước 3) có đúng không — chỉ TRAIN, theo từng shop bỏ ra (ví dụ mẫu lấy từ SHOP KHÁC).

    python scripts/eval_llm_spec.py --train /path/train --model /kaggle/input/qwen... [--embed_model /kaggle/input/bge-m3] [--n 400]

Lấy câu train, cho LLM viết đặc tả JSON (kèm 4 ví dụ gần nhất của shop khác), chạy máy tính, so đáp án train.
In: % đặc tả hợp lệ, % đúng / sai khi máy tính trả lời, tách theo câu luật chữ nhận ra / không nhận ra,
và theo có / không có ví dụ mẫu — để chọn cấu hình (--llm_spec, --embed_model) cho bài nộp.
"""
import argparse
import collections
import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd  # noqa: E402

import compiler as C  # noqa: E402
import datamate_pipeline as d  # noqa: E402
import retriever as RT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--llm", default="vllm", choices=["vllm", "hf", "dummy"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--embed_model", default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cache", default="/tmp/eval_llm_spec")
    a = ap.parse_args()
    qs = [json.loads(x) for x in open(os.path.join(a.train, "questions.jsonl"), encoding="utf-8")]
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8")}
    rb = open(os.path.join(a.train, "rulebook.md"), encoding="utf-8").read()
    d.RULE_PARAMS, _ = d.extract_rule_params(rb)
    bank = RT.load_bank(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lexicon", "spec_bank.jsonl"))
    R = RT.Retriever([b["question"] for b in bank], bank, model_path=a.embed_model) if bank else None
    random.Random(0).shuffle(qs)
    qs = qs[: a.n]
    BS = {}
    llm = d.LLM(a.llm, a.model, a.cache, max_model_len=a.max_model_len, tp=a.tp)
    rows = []
    for use_ex in (False, True):
        msgs, meta = [], []
        for q in qs:
            bid = q["business_id"]
            if bid not in BS:
                BS[bid] = d.Business(os.path.join(a.train, "data", bid), cache_dir=os.path.join(a.cache, "biz"))
            B = BS[bid]
            P = d.resolve_periods(q["question"], q["as_of_date"])
            ex = ""
            if use_ex and R is not None:
                lines = []
                for sc, i in R.topk(q["question"], 4, exclude=lambda m, bid=bid: m["biz"] == bid):
                    b = bank[i]
                    lines.append(f"- Câu: {b['question']}\n  Hôm nay: {b['as_of']}\n  Đặc tả: {json.dumps(b['spec'], ensure_ascii=False)}")
                ex = "\n".join(lines)
            msgs.append(C.spec_messages(q["question"], q["question"], q["as_of_date"], d.periods_text(P), ex))
            rule_sp = C.spec_rules(q["question"], q["as_of_date"], B)
            meta.append((q, B, rule_sp is not None))
        outs = llm.generate(msgs)
        c = collections.Counter()
        for (q, B, by_rule), out in zip(meta, outs):
            grp = "luat_nhan" if by_rule else "luat_khong_nhan"
            sp = C.parse_llm_spec(out, B, q["as_of_date"])
            c[(grp, "tong")] += 1
            if sp is None:
                c[(grp, "khong_hop_le_or_khac")] += 1
                continue
            try:
                ans = C.compile_spec(sp, B, q["as_of_date"])
            except Exception:
                ans = None
            if ans is None:
                c[(grp, "may_khong_tinh")] += 1
                continue
            s = d.score_one(ans, gold[q["id"]])
            c[(grp, "dung" if s == 1 else "sai")] += 1
        for grp in ("luat_nhan", "luat_khong_nhan"):
            n = c[(grp, "tong")] or 1
            print(f"ví dụ mẫu={'có' if use_ex else 'không'} | {grp:<16} | n={c[(grp, 'tong')]:4d} | đúng {c[(grp, 'dung')] / n:6.1%} | "
                  f"SAI {c[(grp, 'sai')] / n:6.1%} | không hợp lệ/khác {c[(grp, 'khong_hop_le_or_khac')] / n:6.1%} | máy không tính {c[(grp, 'may_khong_tinh')] / n:6.1%}",
                  flush=True)


if __name__ == "__main__":
    main()
