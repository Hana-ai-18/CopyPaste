"""Build the few-shot exemplar bank from a LABELLED TRAINING run (no re-running needed).

    1) run the agent on the train split:   python -m datamate.run --data_dir train --out out/train_sub.jsonl --logs out/logs
    2) python scripts/build_exemplars.py --log out/logs/log.jsonl --gold train/answers.jsonl --out exemplars/exemplars.jsonl

For every training question, the first generated program whose answer scores +1 against the gold answer
is stored (question text + code). At inference the most similar questions are shown to the LLM.

Safety: this refuses gold files whose path mentions public_test / private_test - test data must never be
used for training, tuning or labelling (competition rule).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.scoring import score_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="exemplars/exemplars.jsonl")
    ap.add_argument("--max_code_chars", type=int, default=1500)
    ap.add_argument("--max_per_question_text", type=int, default=1)
    a = ap.parse_args()

    low = (str(a.gold) + str(a.log)).lower()
    if "public_test" in low or "private_test" in low or "public-test" in low:
        raise SystemExit("REFUSED: exemplars may only be built from the TRAINING split (rules forbid using test data).")

    gold = {}
    for l in open(a.gold, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            gold[r["id"]] = r
    out, seen = [], {}
    for l in open(a.log, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        g = gold.get(d["id"])
        if g is None or seen.get(d["question"], 0) >= a.max_per_question_text:
            continue
        for c in d.get("candidates", []):
            if not c.get("ok") or not c.get("code"):
                continue
            pred = {"id": d["id"], "type": c["answer_type"], "answer": c.get("answer")}
            s, _ = score_one(pred, g)
            if s == 1 and len(c["code"]) <= a.max_code_chars:
                out.append({"question": d["question"], "code": c["code"], "answer_type": c["answer_type"]})
                seen[d["question"]] = seen.get(d["question"], 0) + 1
                break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(out)} exemplars from {len(gold)} gold answers -> {a.out}")


if __name__ == "__main__":
    main()
