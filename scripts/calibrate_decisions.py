"""
Hiệu chỉnh ngưỡng BỎ TRỐNG theo điểm PA = (đúng − sai) / N, từ một lần chạy pipeline trên TRAIN (luật 6: chỉ train).

    python scripts/calibrate_decisions.py --train /path/train --run /path/out_train [--min_n 15] [--out lexicon/decision_calibration.json]

Với mỗi nhãn quyết định (compiler+agree, compiler_wins, agree, single, choice_code…) và mỗi cặp (nhãn, intent),
tính tỉ lệ đúng p khi trả lời. Trả lời có lợi khi p > 0,5 (mỗi câu sai trừ đúng bằng một câu đúng).
Nhãn có p ≤ 0,5 với đủ mẫu -> pipeline đổi thành bỏ trống (đọc file này khi có `--calibration`).
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import datamate_pipeline as d  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--run", required=True, help="thư mục --out của lần chạy pipeline trên train (có logs/)")
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lexicon", "decision_calibration.json"))
    a = ap.parse_args()
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8")}
    st = collections.defaultdict(lambda: [0, 0])          # nhãn -> [đúng, sai]
    for f in glob.glob(os.path.join(a.run, "logs", "*.json")):
        L = json.load(open(f, encoding="utf-8"))
        if L.get("id") not in gold or not L.get("final") or L["final"].get("type") == "abstain":
            continue
        s = d.score_one(L["final"], gold[L["id"]])
        if s == 0:
            continue
        dec = L.get("decision", "?")
        it = (L.get("spec") or {}).get("intent", "-")
        for k in (dec, f"{dec}|{it}"):
            st[k][0 if s == 1 else 1] += 1
    table = {}
    print(f"{'nhãn':<45} {'đúng':>6} {'sai':>6} {'p':>6}  quyết")
    for k, (r, w) in sorted(st.items(), key=lambda x: -sum(x[1])):
        n = r + w
        p = r / n
        abstain = n >= a.min_n and p <= 0.5
        table[k] = {"dung": r, "sai": w, "p": round(p, 4), "bo_trong": abstain}
        if "|" not in k or abstain:
            print(f"{k:<45} {r:>6} {w:>6} {p:6.3f}  {'BỎ TRỐNG' if abstain else 'trả lời'}")
    json.dump({"min_n": a.min_n, "nguon": "train", "bang": table}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
