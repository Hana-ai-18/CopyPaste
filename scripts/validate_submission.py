"""Validate a DataMate submission.jsonl BEFORE uploading it.

    python scripts/validate_submission.py --sub submission.jsonl --questions public_test/questions.jsonl

Checks: UTF-8 JSON lines, unique ids, ids == questions, allowed types, value shapes
(number/percent finite, date yyyy-mm-dd or yyyy-mm, list of str/number, table columns/rows).
Exit code 1 if any error is found.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter

TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}
DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_record(r: dict) -> list[str]:
    errs = []
    t = r.get("type")
    if t not in TYPES:
        return [f"unknown type {t!r}"]
    if t in ("abstain", "unanswerable"):
        return errs
    if "answer" not in r:
        return ["missing 'answer'"]
    a = r["answer"]
    if t in ("number", "percent") and not _num(a):
        errs.append(f"{t} must be a finite number, got {a!r}")
    elif t == "string" and not (isinstance(a, str) and a.strip()):
        errs.append("string must be a non-empty string")
    elif t == "date" and not (isinstance(a, str) and DATE.match(a)):
        errs.append(f"date must be yyyy-mm-dd or yyyy-mm, got {a!r}")
    elif t == "boolean" and not isinstance(a, bool):
        errs.append(f"boolean must be true/false, got {a!r}")
    elif t == "list":
        if not isinstance(a, list):
            errs.append("list answer must be a JSON array")
        elif not all((isinstance(x, str) and x.strip()) or _num(x) for x in a):
            errs.append("list elements must be non-empty strings or numbers")
    elif t == "table":
        if not (isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list)):
            errs.append('table must be {"columns": [...], "rows": [[...]]}')
        else:
            n = len(a["columns"])
            for i, row in enumerate(a["rows"]):
                if not isinstance(row, list) or len(row) != n:
                    errs.append(f"table row {i} must be a list of {n} cells")
                    break
                if not all(isinstance(c, str) or _num(c) for c in row):
                    errs.append(f"table row {i} has a non string/number cell")
                    break
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--questions", required=True)
    a = ap.parse_args()

    qids = [json.loads(l)["id"] for l in open(a.questions, encoding="utf-8") if l.strip()]
    errors, recs = [], []
    with open(a.sub, encoding="utf-8") as f:  # raises on invalid UTF-8
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {ln}: invalid JSON ({e})")
                continue
            if not isinstance(r, dict) or "id" not in r:
                errors.append(f"line {ln}: missing id")
                continue
            for e in check_record(r):
                errors.append(f"line {ln} id={r['id']}: {e}")
            recs.append(r)
    ids = [r["id"] for r in recs]
    dup = [k for k, c in Counter(ids).items() if c > 1]
    if dup:
        errors.append(f"duplicate ids: {dup[:5]}{' ...' if len(dup) > 5 else ''}")
    missing, extra = set(qids) - set(ids), set(ids) - set(qids)
    if missing:
        errors.append(f"{len(missing)} question ids missing (they would be scored as abstain), e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} ids are not in the questions file, e.g. {sorted(extra)[:3]}")
    c = Counter(r.get("type") for r in recs)
    print(f"{len(recs)} lines | questions {len(qids)} | types: {dict(c)}")
    if errors:
        print(f"\n{len(errors)} PROBLEM(S):")
        for e in errors[:30]:
            print("  -", e)
        return 1
    print("OK - submission looks valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
