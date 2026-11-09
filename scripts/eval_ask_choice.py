"""
Đo trên GPU: câu hỏi trắc nghiệm A/B/C ('cách tính nào đúng theo rulebook') có chọn đúng không — chỉ TRAIN.

    python scripts/eval_ask_choice.py --train /path/train --model /kaggle/input/qwen... [--n 300]

Lấy câu train mà máy tính (đặc tả bằng luật) trả lời ĐÚNG, sinh một CÁCH HIỂU SAI hay gặp (alternative_readings: tính cả đơn
trả/huỷ, sai mẫu số tỉ lệ huỷ, không quy đổi đơn vị, gồm cả mốc N, đảo kê đơn) ra con số khác, rồi hỏi model chọn.
Đảo vị trí đúng/sai (nửa câu đúng ở A, nửa ở B) để tránh model thiên về một chữ cái.
In: % chọn đúng, % chọn sai, % 'không chắc' — dùng để quyết định bật / tắt --ask_choice.
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
import compiler as C  # noqa: E402
import datamate_pipeline as d  # noqa: E402

ALT_TEXT = {
    "tinh_ca_don_tra_huy": "tính cả đơn trả hàng và đơn huỷ",
    "mau_so_chi_don_tinh_doanh_thu": "mẫu số chỉ gồm đơn tính doanh thu",
    "khong_quy_doi_don_vi": "cộng số lượng theo đơn vị ghi trên dòng, không quy đổi",
    "gom_ca_N": "tính cả đơn có tuổi nợ đúng bằng N ngày",
    "dao_ke_don": "đảo điều kiện thuốc kê đơn",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--llm", default="vllm", choices=["vllm", "hf", "dummy"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cache", default="/tmp/eval_ask_choice")
    a = ap.parse_args()
    qs = [json.loads(x) for x in open(os.path.join(a.train, "questions.jsonl"), encoding="utf-8")]
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8")}
    rb = open(os.path.join(a.train, "rulebook.md"), encoding="utf-8").read()
    d.RULE_INDEX = d.RuleIndex(rb)
    d.RULE_PARAMS, _ = d.extract_rule_params(rb)
    random.Random(0).shuffle(qs)
    BS, cases = {}, []
    for q in qs:
        if len(cases) >= a.n:
            break
        bid = q["business_id"]
        if bid not in BS:
            BS[bid] = d.Business(os.path.join(a.train, "data", bid), cache_dir=os.path.join(a.cache, "biz"))
        B = BS[bid]
        sp = C.spec_rules(q["question"], q["as_of_date"], B)
        ans = C.compile_spec(sp, B, q["as_of_date"]) if sp else None
        if ans is None or d.score_one(ans, gold[q["id"]]) != 1:
            continue
        for name, alt in C.alternative_readings(sp):
            wa = C.compile_spec(alt, B, q["as_of_date"])
            if wa is not None and not C.same_answer(ans, wa):
                cases.append((q, B, sp, ans, name, wa))
                break
    print(f"{len(cases)} cặp (cách đúng, cách hiểu sai)")
    llm = d.LLM(a.llm, a.model, a.cache, max_model_len=a.max_model_len, tp=a.tp)
    c = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    for k, (q, B, sp, ans, name, wa) in enumerate(cases):
        item = {"q": q["question"], "q_norm": q["question"], "as_of": q["as_of_date"]}
        rules = "\n".join(f"- {r['text'][:300]}" for r in d.RULE_INDEX.select(q["question"], B.readme or "", k=4))
        right = f"{d.spec_text(sp)} → {json.dumps(ans.get('answer'), ensure_ascii=False)[:200]}"
        wrong = f"{d.spec_text(sp)}; nhưng {ALT_TEXT.get(name, name)} → {json.dumps(wa.get('answer'), ensure_ascii=False)[:200]}"
        swap = k % 2 == 1
        ch = d.ask_choice(llm, item, rules, wrong if swap else right, right if swap else wrong)
        res = "khong_chac" if ch is None else ("dung" if (ch == "A") != swap else "sai")
        c[res] += 1
        by[name][res] += 1
    n = max(1, sum(c.values()))
    print(f"chọn đúng {c['dung'] / n:.1%} | chọn SAI {c['sai'] / n:.1%} | không chắc {c['khong_chac'] / n:.1%}")
    for name, cc in by.items():
        m = sum(cc.values())
        print(f"  {name:<32} n={m:4d} đúng {cc['dung'] / m:.1%} sai {cc['sai'] / m:.1%} không chắc {cc['khong_chac'] / m:.1%}")
    print("Quy tắc: bật --ask_choice khi chọn đúng > chọn sai rõ rệt (PA chỉ lợi khi đúng > sai).")


if __name__ == "__main__":
    main()
