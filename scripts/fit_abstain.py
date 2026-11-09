"""
Mô hình XÁC SUẤT ĐÚNG cho bỏ trống theo PA, fit trên log một lần chạy pipeline với model THẬT trên TRAIN (luật 6: chỉ train).

    python scripts/fit_abstain.py --train /path/train --run /path/out_train [--out lexicon/abstain_model.json] [--cv 5]

Đặc trưng: answer_features() trong datamate_pipeline (nhãn quyết định, nguồn đặc tả, số đường khớp, đã sửa lỗi, cờ tự soát,
kiểu đáp án…). Hồi quy logistic (có chuẩn hoá L2), kiểm chéo THEO SHOP: so PA "trả lời hết" với PA "bỏ trống khi p ≤ ngưỡng",
chọn ngưỡng tốt nhất trên phần kiểm chéo (không chọn trên chính dữ liệu fit). Lưu trọng số dạng JSON (không pickle).
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import datamate_pipeline as d  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lexicon", "abstain_model.json"))
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--group", default="family", choices=["family", "shop"],
                    help="kiểm chéo giữ lại cả một HỌ định dạng dữ liệu (mặc định, mô phỏng shop lạ ở private) hoặc từng shop")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    gold = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(a.train, "answers.jsonl"), encoding="utf-8")}
    X, y, shops, names = [], [], [], None
    for f in sorted(glob.glob(os.path.join(a.run, "logs", "*.json"))):
        L = json.load(open(f, encoding="utf-8"))
        fin = L.get("final") or {}
        if L.get("id") not in gold or fin.get("type") in (None, "abstain"):
            continue
        s = d.score_one(fin, gold[L["id"]])
        if s == 0:
            continue
        feat = d.answer_features(L)
        names = names or sorted(feat)
        X.append([feat[k] for k in names])
        y.append(1 if s == 1 else 0)
        shops.append(L.get("business_id"))
    if len(set(y)) < 2:
        print("Chưa đủ dữ liệu (cần cả câu đúng và câu sai) — chạy pipeline với model thật trên train trước.")
        return
    X, y = np.array(X), np.array(y)
    if a.group == "family":
        fam = {b: d.data_family(os.path.join(a.train, "data", b)) for b in set(shops)}
        groups = [fam[s] for s in shops]
        print(f"kiểm chéo theo HỌ dữ liệu: {len(set(groups))} họ / {len(set(shops))} shop")
    else:
        groups = shops
    uniq = sorted(set(groups))
    folds = {s: i % a.cv for i, s in enumerate(uniq)}
    fold = np.array([folds[s] for s in groups])
    p_cv = np.zeros(len(y))
    for k in range(a.cv):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        m = LogisticRegression(C=a.C, max_iter=2000, fit_intercept=False).fit(X[tr], y[tr])
        p_cv[te] = m.predict_proba(X[te])[:, 1]
    n = len(y)
    base = (y.sum() - (n - y.sum())) / n
    best = (0.0, base)
    for th in np.arange(0.30, 0.81, 0.02):
        keep = p_cv > th
        pa = (y[keep].sum() - (keep.sum() - y[keep].sum())) / n
        if pa > best[1] + 1e-9:
            best = (float(th), float(pa))
    m = LogisticRegression(C=a.C, max_iter=2000, fit_intercept=False).fit(X, y)
    model = {"nguon": "train", "n": int(n), "nguong": round(max(0.5, best[0]) if best[0] else 0.5, 2),
             "w": {k: float(v) for k, v in zip(names, m.coef_[0])},
             "kiem_cheo": {"pa_tra_loi_het_tren_cau_da_tra_loi": round(float(base), 4), "pa_voi_nguong": round(best[1], 4), "nguong_cv": best[0]}}
    json.dump(model, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(model["kiem_cheo"], ensure_ascii=False), "->", a.out)
    for k, v in sorted(model["w"].items(), key=lambda x: -abs(x[1]))[:10]:
        print(f"  {k:<22} {v:+.3f}")


if __name__ == "__main__":
    main()
