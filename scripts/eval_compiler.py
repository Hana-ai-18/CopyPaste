"""
Đo ĐƯỜNG 1 (đặc tả bằng luật + máy tính theo công thức) trên TRAIN — không cần GPU, không dùng LLM.

    python scripts/eval_compiler.py --train /path/train [--cache /tmp/c] [--norm] [--show_bad 40] [--out /tmp/compiler_train.jsonl]

In: số câu máy tính nhận (độ phủ), đúng / sai trên số đó (precision), PA nếu CHỈ dùng đường này (câu khác bỏ trống),
và bảng theo intent. --norm: sửa chữ câu hỏi trước (giống pipeline, không có LLM).
Chỉ dùng train (luật 6).
"""
import argparse
import collections
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import datamate_pipeline as d  # noqa: E402
import compiler as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--show_bad", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rulebook", default=None, help="mô phỏng chạy với rulebook KHÁC (vd public) để xem Rule-Delta chặn intent nào; mặc định rulebook train")
    a = ap.parse_args()
    qs = [json.loads(x) for x in open(os.path.join(a.train, "questions.jsonl"), encoding="utf-8")]
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8")}
    rb = open(os.path.join(a.train, "rulebook.md"), encoding="utf-8").read()
    d.RULE_PARAMS, _ = d.extract_rule_params(rb)
    C.WEEK_START = d.RULE_PARAMS.get("week_start", 0)
    rd = None
    try:
        import rule_delta as R
        rd = R.RuleDelta(open(a.rulebook, encoding="utf-8").read() if a.rulebook else rb)
    except Exception as e:
        print("(không dựng được Rule-Delta:", e, ")")
    by_biz = collections.defaultdict(list)
    for q in qs:
        by_biz[q["business_id"]].append(q)
    res = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    bad = []
    fout = open(a.out, "w", encoding="utf-8") if a.out else None
    for bid, items in sorted(by_biz.items()):
        B = d.Business(os.path.join(a.train, "data", bid), cache_dir=a.cache)
        norm = None
        if a.norm and d.Normalizer:
            norm = d.Normalizer(B.terms, ["bill", "voucher", "top", "order", "ship", "cod", "size", "zalo", "ok", "sale"], [rb, B.readme])
            if d.ContextNormalizer is not None:
                norm = d.ContextNormalizer(norm, [rb, B.readme])
        for q in items:
            qn = None
            if norm is not None:
                try:
                    qn = norm.normalize(q["question"])["text"]
                except Exception:
                    qn = None
            try:
                sp = C.spec_rules(q["question"], q["as_of_date"], B, qn)
                ans = C.compile_spec(sp, B, q["as_of_date"]) if sp else None
            except Exception as e:
                sp, ans = {"intent": "LỖI"}, None
                res["lỗi"] += 1
                bad.append(("LỖI", bid, q["question"][:80], str(e)[:80], ""))
            it = (sp or {}).get("intent", "-") + ("(thấp)" if (sp or {}).get("tin_cay") == "thap" else "")
            if ans is not None and rd is not None and sp and rd.block_reason(sp, C.key((qn or "") + " " + q["question"])):
                res["rule_delta chặn"] += 1
                per[it]["rule_delta chặn"] += 1
                ans = None
                continue
            if ans is not None and ans.get("_tie"):
                s_ = d.score_one(ans, gold[q["id"]])
                res["hoà ở biên (bỏ trống)"] += 1
                per[it]["hoà: " + {1: "đúng", -1: "sai", 0: "?"}[s_]] += 1
                continue
            if ans is None:
                res["không nhận"] += 1
                per[it]["không nhận"] += 1
                continue
            s = d.score_one(ans, gold[q["id"]])
            lab = {1: "đúng", -1: "sai", 0: "bỏ trống"}[s]
            res[lab] += 1
            per[it][lab] += 1
            if s == -1:
                bad.append((it, bid, q["question"][:80], json.dumps(gold[q["id"]].get("answer"), ensure_ascii=False)[:60],
                            json.dumps(ans.get("answer", ans.get("type")), ensure_ascii=False)[:60]))
            if fout:
                fout.write(json.dumps({"id": q["id"], **ans}, ensure_ascii=False, default=str) + "\n")
    n = len(qs)
    ans_n = res["đúng"] + res["sai"]
    print(f"Câu: {n} | máy tính trả lời {ans_n} ({ans_n / n:.1%}) | đúng {res['đúng']} | sai {res['sai']} | precision {res['đúng'] / max(1, ans_n):.4f}")
    print(f"PA nếu CHỈ dùng đường này: {(res['đúng'] - res['sai']) / n:.4f}  (lỗi code: {res['lỗi']}; hoà ở biên -> bỏ trống: {res['hoà ở biên (bỏ trống)']}; Rule-Delta chặn: {res['rule_delta chặn']})")
    hi = collections.Counter()
    for it, c in per.items():
        if not it.endswith("(thấp)"):
            hi.update(c)
    print(f"Chỉ intent tin cậy CAO: đúng {hi['đúng']} | sai {hi['sai']} | precision {hi['đúng'] / max(1, hi['đúng'] + hi['sai']):.4f} | PA {(hi['đúng'] - hi['sai']) / n:.4f}")
    print("\nTheo intent:")
    for it, c in sorted(per.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {it:<28} {dict(c)}")
    for x in bad[:a.show_bad]:
        print("  ✗", x)


if __name__ == "__main__":
    main()
