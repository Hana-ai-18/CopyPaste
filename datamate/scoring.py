"""Local re-implementation of the DataMate per-question scoring and Penalized Accuracy.

Follows the official description:
  * abstain / missing answer -> 0
  * unanswerable reference: only an `unanswerable` answer is correct (+1), any other answer is wrong (-1)
  * otherwise the type must equal the reference type AND the value must match (+1), else wrong (-1)
"""
from __future__ import annotations

import math
from collections import defaultdict

from .answer_format import _is_num, round_half_up
from .textutil import norm_string


# --------------------------------------------------------------------------- matchers
def match_number(p, r, alias=None) -> bool:
    if not (_is_num(p) and _is_num(r)) or not (math.isfinite(p) and math.isfinite(r)):
        return False
    if float(r).is_integer():
        return abs(float(p) - float(r)) < 1e-6
    return round_half_up(p, 2) == round_half_up(r, 2)


def match_percent(p, r, alias=None) -> bool:
    if not (_is_num(p) and _is_num(r)) or not (math.isfinite(p) and math.isfinite(r)):
        return False
    return round_half_up(p, 2) == round_half_up(r, 2)


def match_string(p, r, alias=None) -> bool:
    if not isinstance(p, str):
        return False
    forms = [r] + list(alias or [])
    pn = norm_string(p)
    return any(isinstance(f, (str, int, float)) and norm_string(f) == pn for f in forms)


def match_date(p, r, alias=None) -> bool:
    return isinstance(p, str) and isinstance(r, str) and p.strip() == r.strip()


def match_boolean(p, r, alias=None) -> bool:
    return isinstance(p, bool) and isinstance(r, bool) and p == r


def _match_elem(p, r, alias=None) -> bool:
    if _is_num(r):
        return match_number(p, r)
    return match_string(p, r, alias)


def match_list(p, r, alias=None) -> bool:
    if not isinstance(p, list) or not isinstance(r, list) or len(p) != len(r):
        return False
    per_elem = None
    if alias and len(alias) == len(r) and all(isinstance(a, list) for a in alias):
        per_elem = alias
    return all(_match_elem(pe, re_, per_elem[i] if per_elem else None) for i, (pe, re_) in enumerate(zip(p, r)))


def match_table(p, r, alias=None) -> bool:
    if not (isinstance(p, dict) and isinstance(r, dict)):
        return False
    try:
        pc, pr = list(p["columns"]), list(p["rows"])
        rc, rr = list(r["columns"]), list(r["rows"])
    except (KeyError, TypeError):
        return False
    if len(pc) != len(rc) or len(pr) != len(rr):
        return False
    pn = [norm_string(c) for c in pc]
    rn = [norm_string(c) for c in rc]
    if sorted(pn) == sorted(rn) and len(set(rn)) == len(rn):
        order = [pn.index(c) for c in rn]  # by normalised name
    else:
        order = list(range(len(rc)))  # by position
    try:
        pr = [[row[i] for i in order] for row in pr]
    except (IndexError, TypeError):
        return False
    unused = list(range(len(rr)))
    for prow in pr:
        hit = None
        for j in unused:
            rrow = rr[j]
            if len(rrow) == len(prow) and all(_match_elem(a, b) for a, b in zip(prow, rrow)):
                hit = j
                break
        if hit is None:
            return False
        unused.remove(hit)
    return True


MATCHERS = {
    "number": match_number,
    "percent": match_percent,
    "string": match_string,
    "date": match_date,
    "boolean": match_boolean,
    "list": match_list,
    "table": match_table,
}


# --------------------------------------------------------------------------- scoring
def score_one(pred: dict | None, ref: dict) -> tuple[int, str]:
    """Return (score in {1,0,-1}, outcome in {'correct','wrong','abstain'})."""
    if pred is None or pred.get("type") == "abstain":
        return 0, "abstain"
    rtype, ptype = ref.get("type"), pred.get("type")
    if rtype == "unanswerable":
        return (1, "correct") if ptype == "unanswerable" else (-1, "wrong")
    if ptype != rtype or rtype not in MATCHERS:
        return -1, "wrong"
    ok = MATCHERS[rtype](pred.get("answer"), ref.get("answer"), ref.get("alias") or [])
    return (1, "correct") if ok else (-1, "wrong")


def penalized_accuracy(preds: dict[str, dict], refs: list[dict]) -> dict:
    """PA = mean of per-question scores over all reference questions (missing -> abstain)."""
    total = 0
    counts = defaultdict(int)
    by_type = defaultdict(lambda: defaultdict(int))
    for ref in refs:
        s, outcome = score_one(preds.get(ref["id"]), ref)
        total += s
        counts[outcome] += 1
        by_type[ref.get("type", "?")][outcome] += 1
    n = len(refs)
    return {
        "PA": (total / n) if n else 0.0,
        "n": n,
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "abstain": counts["abstain"],
        "by_type": {t: dict(v) for t, v in by_type.items()},
    }
