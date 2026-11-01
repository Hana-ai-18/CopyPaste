"""Score a submission against answers.jsonl and calibrate the abstain thresholds.

python -m datamate.evaluate --pred result/submission.jsonl --gold train/answers.jsonl
python -m datamate.evaluate --pred ... --gold ... --calibrate result/logs/log.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scoring import penalized_accuracy, score_one


def _read(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def report(pred_path, gold_path, questions_ids: set | None = None) -> dict:
    preds = {r["id"]: r for r in _read(pred_path)}
    refs = _read(gold_path)
    if questions_ids is not None:
        refs = [r for r in refs if r["id"] in questions_ids]
    m = penalized_accuracy(preds, refs)
    print(f"\nPenalized Accuracy = {m['PA']:.4f}  (n={m['n']}  correct={m['correct']}  wrong={m['wrong']}  abstain={m['abstain']})")
    print(f"{'type':<14}{'n':>5}{'correct':>9}{'wrong':>7}{'abstain':>9}")
    for t, d in sorted(m["by_type"].items()):
        n = sum(d.values())
        print(f"{t:<14}{n:>5}{d.get('correct', 0):>9}{d.get('wrong', 0):>7}{d.get('abstain', 0):>9}")
    return m


def calibrate(log_path, gold_path) -> dict:
    """Sweep the vote-share thresholds using the `top_candidate` + `confidence` stored in the run log."""
    gold = {r["id"]: r for r in _read(gold_path)}
    rows = []
    for l in _read(log_path):
        if l["id"] in gold and l.get("top_candidate") is not None:
            top = dict(l["top_candidate"], id=l["id"])
            s, _ = score_one(top, gold[l["id"]])
            rows.append((float(l.get("confidence", 0.0)), top["type"] == "unanswerable", s))
    if not rows:
        print("no usable rows in the log (need run logs of a labelled split)")
        return {}
    grid = [round(0.3 + 0.1 * i, 1) for i in range(8)] + [1.01]  # 0.3 .. 1.0, and 1.01 = never answer
    best_pa, best_t = -9.0, (grid[0], grid[0])
    for ta in grid:
        for tu in grid:
            total = 0
            for conf, is_un, s in rows:
                total += s if conf >= (tu if is_un else ta) else 0
            pa = total / len(rows)
            # iterate thresholds ascending and accept ties => among equal PA the HIGHER (safer) thresholds win
            if pa >= best_pa - 1e-12:
                best_pa, best_t = max(pa, best_pa), (ta, tu)
    pa, (ta, tu) = best_pa, best_t
    print(f"\nBest on {len(rows)} logged questions: PA={pa:.4f} with threshold_answer={ta:.2f}  threshold_unanswerable={tu:.2f}")
    print('Use:  --set agent.threshold_answer=%.2f --set agent.threshold_unanswerable=%.2f' % (ta, tu))
    print("(1.01 means 'never answer' for that kind. Small samples overfit: sanity-check against the defaults 0.6 / 0.8.)")
    return {"threshold_answer": ta, "threshold_unanswerable": tu, "PA": pa}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--calibrate", default=None, help="path of logs/log.jsonl of the same run")
    ap.add_argument("--all", action="store_true", help="score against EVERY gold row (missing predictions count as abstain); default: only the ids present in --pred")
    a = ap.parse_args(argv)
    report(a.pred, a.gold, None if a.all else {r["id"] for r in _read(a.pred)})
    if a.calibrate:
        calibrate(a.calibrate, a.gold)


if __name__ == "__main__":
    main()
