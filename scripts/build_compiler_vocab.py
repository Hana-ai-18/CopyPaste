"""
Từ vựng câu hỏi cho "vocab guard" của máy tính (chỉ dựng từ TRAIN — luật 6):

    python scripts/build_compiler_vocab.py --train /path/train --cache /tmp/c [--min_shops 2] [--out lexicon/compiler_vocab.json]

Một chữ được coi là "đã hiểu" nếu nó xuất hiện trong câu hỏi train ở ≥ min_shops shop khác nhau, sau khi bỏ chữ thuộc
tên thực thể đã gắn vào đặc tả (tên món, khách…) và mọi chữ có chữ số (mã: nv001, d50…). Lúc chạy, câu có chữ ngoài từ vựng (và không thuộc thực thể đã gắn) -> máy tính KHÔNG được tự quyết.
File ra chỉ chứa CHỮ ĐƠN (không có câu hỏi, không có đáp án).
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


def make_normalizer(B, rb):
    if not d.Normalizer:
        return None
    nz = d.Normalizer(B.terms, ["bill", "voucher", "top", "order", "ship", "cod", "size", "zalo", "ok", "sale"], [rb, B.readme])
    if d.ContextNormalizer is not None:
        nz = d.ContextNormalizer(nz, [rb, B.readme])
    return nz


def collect_specs(train, cache, norm=True):
    """[(business_id, question_row, q_norm, spec)] cho mọi câu train (spec có thể None)."""
    qs = [json.loads(x) for x in open(os.path.join(train, "questions.jsonl"), encoding="utf-8")]
    rb = open(os.path.join(train, "rulebook.md"), encoding="utf-8").read()
    d.RULE_PARAMS, _ = d.extract_rule_params(rb)
    C.WEEK_START = d.RULE_PARAMS.get("week_start", 0)
    by = collections.defaultdict(list)
    for q in qs:
        by[q["business_id"]].append(q)
    rows = []
    for bid, items in sorted(by.items()):
        B = d.Business(os.path.join(train, "data", bid), cache_dir=cache)
        nz = make_normalizer(B, rb) if norm else None
        for q in items:
            qn = None
            if nz is not None:
                try:
                    qn = nz.normalize(q["question"])["text"]
                except Exception:
                    qn = None
            try:
                sp = C.spec_rules(q["question"], q["as_of_date"], B, qn)
            except Exception:
                sp = None
            rows.append((bid, q, qn, sp))
    return rows


def token_shops(rows):
    """chữ -> tập shop có câu (được máy tính nhận) chứa chữ đó, đã trừ chữ của thực thể gắn trong đặc tả."""
    ts = collections.defaultdict(set)
    for bid, q, qn, sp in rows:
        # MỌI câu train (kể cả câu máy tính chưa nhận): từ vựng nói chung của người hỏi; trừ tên thực thể đã gắn
        ent = C.spec_entity_tokens(sp) if sp else set()
        for w in C.question_tokens(C.key(qn or q["question"])):
            if w not in ent:
                ts[w].add(bid)
    return ts


def vocab_from(ts, min_shops=2, exclude_shop=None):
    # chữ có chữ số (nv001, d50, 25w…) luôn là mã/tên riêng -> không bao giờ coi là 'từ thường'
    return {w for w, s in ts.items() if len(s - {exclude_shop}) >= min_shops and not any(ch.isdigit() for ch in w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--min_shops", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lexicon", "compiler_vocab.json"))
    a = ap.parse_args()
    rows = collect_specs(a.train, a.cache)
    ts = token_shops(rows)
    V = sorted(vocab_from(ts, a.min_shops))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"min_shops": a.min_shops, "nguon": "câu hỏi TRAIN, trừ tên thực thể đã gắn và chữ có chữ số", "tokens": V}, f,
                  ensure_ascii=False, indent=0)
    print(f"từ vựng: {len(V)} chữ (≥ {a.min_shops} shop) -> {a.out}")


if __name__ == "__main__":
    main()
