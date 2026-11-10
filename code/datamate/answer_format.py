"""Answer sanitising (python object -> valid submission value) and canonical keys for voting."""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd

from .textutil import norm_string

VALUE_TYPES = ("number", "percent", "string", "date", "boolean", "list", "table")
ALL_TYPES = VALUE_TYPES + ("unanswerable", "abstain")

_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}")
_ISO_MONTH = re.compile(r"^\d{4}-\d{2}$")
_DMY = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


# --------------------------------------------------------------------------- rounding
def _dec(x) -> Decimal:
    if isinstance(x, bool):
        raise TypeError("bool is not a number")
    if isinstance(x, (int, np.integer)):
        return Decimal(int(x))
    if isinstance(x, Decimal):
        return x
    return Decimal(repr(float(x)))


def round_half_up(x, nd: int = 0):
    """Round half away from zero (school rounding). Returns int when nd == 0."""
    q = Decimal(1).scaleb(-nd)
    r = _dec(x).quantize(q, rounding=ROUND_HALF_UP)
    return int(r) if nd == 0 else float(r)


def _num_out(v):
    """Number -> int when integral, else 2-decimal float."""
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("number is NaN or infinite")
    if abs(f - round(f)) < 1e-9:
        return int(round(f))
    return round_half_up(f, 2)


# --------------------------------------------------------------------------- python -> json-able
def to_py(v: Any):
    """Convert numpy / pandas / datetime objects to plain JSON-able python values."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v).date().isoformat()
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, _dt.datetime):
        return v.date().isoformat()
    if isinstance(v, _dt.date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, pd.DataFrame):
        return {"columns": [str(c) for c in v.columns], "rows": [[to_py(x) for x in r] for r in v.itertuples(index=False, name=None)]}
    if isinstance(v, (pd.Series, pd.Index, np.ndarray, list, tuple, set)):
        return [to_py(x) for x in (sorted(v) if isinstance(v, set) else list(v))]
    if isinstance(v, dict):
        return {str(k): to_py(x) for k, x in v.items()}
    return v


def _unwrap_scalar(v, lists: bool = True):
    """One-element Series/array (and, for top-level scalar answers, a one-element list) -> the element."""
    if isinstance(v, pd.Series) and len(v) == 1:
        return v.iloc[0]
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.reshape(-1)[0]
    if lists and isinstance(v, (list, tuple)) and len(v) == 1 and not isinstance(v[0], (list, tuple, dict)):
        return v[0]
    return v


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _date_out(v) -> str:
    v = to_py(v)
    if isinstance(v, str):
        s = v.strip()
        if _ISO_MONTH.match(s):
            return s
        if _ISO_DAY.match(s):
            return s[:10]
        m = _DMY.match(s)
        if m:
            d, mo, y = (int(g) for g in m.groups())
            return _dt.date(y, mo, d).isoformat()
    raise ValueError(f"date must be 'YYYY-MM-DD' (or 'YYYY-MM' for month questions), got {v!r}")


def _cell_out(v):
    v = to_py(_unwrap_scalar(v, lists=False))
    if v is None:
        raise ValueError("NaN/None value inside list/table")
    if isinstance(v, bool):
        raise ValueError("bool inside list/table is not supported")
    if _is_num(v):
        return _num_out(v)
    if isinstance(v, str):
        s = v.strip()
        return s
    raise ValueError(f"unsupported element type {type(v).__name__}")


def sanitize_answer(atype: str, value: Any):
    """Validate/convert `value` for `atype`. Raises ValueError with a helpful message."""
    if atype not in VALUE_TYPES:
        raise ValueError(f"answer_type must be one of {VALUE_TYPES} (or use declare_unanswerable), got {atype!r}")
    if atype in ("number", "percent", "string", "date", "boolean"):
        value = _unwrap_scalar(value)
    v = to_py(value)
    if v is None:
        raise ValueError(f"answer is None/NaN for type {atype}")
    if atype == "number":
        if not _is_num(v):
            raise ValueError(f"number answer must be int/float, got {type(v).__name__}: {v!r}")
        return _num_out(v)
    if atype == "percent":
        if not _is_num(v) or not math.isfinite(float(v)):
            raise ValueError(f"percent answer must be a finite float (12.5 means 12.5%), got {v!r}")
        return round_half_up(float(v), 2)
    if atype == "string":
        s = str(v).strip()
        if not s:
            raise ValueError("string answer is empty")
        return s
    if atype == "date":
        return _date_out(v)
    if atype == "boolean":
        if not isinstance(v, bool):
            raise ValueError(f"boolean answer must be True/False, got {v!r}")
        return v
    if atype == "list":
        if not isinstance(v, list):
            raise ValueError(f"list answer must be a list, got {type(v).__name__}")
        return [_cell_out(x) for x in v]
    # table
    if not (isinstance(v, dict) and "columns" in v and "rows" in v):
        raise ValueError('table answer must be {"columns": [...], "rows": [[...], ...]} or a DataFrame')
    cols = [str(c) for c in v["columns"]]
    rows = []
    for r in v["rows"]:
        if not isinstance(r, list) or len(r) != len(cols):
            raise ValueError("every table row must be a list with one value per column")
        rows.append([_cell_out(x) for x in r])
    return {"columns": cols, "rows": rows}


# --------------------------------------------------------------------------- canonical key
def _numkey(v) -> str:
    f = float(v)
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{round_half_up(f, 2):.2f}"


def _elem_key(x) -> str:
    if _is_num(x):
        return "n:" + _numkey(x)
    return "s:" + norm_string(x)


def answer_key(atype: str, answer: Any) -> str:
    """Canonical string used to compare/vote candidate answers (order matters for lists)."""
    if atype == "unanswerable":
        return "unanswerable"
    if atype == "string":
        return "s:" + norm_string(answer)
    if atype == "number":
        return "n:" + _numkey(answer)
    if atype == "percent":
        return "p:" + f"{round_half_up(answer, 2):.2f}"
    if atype == "date":
        return "d:" + str(answer)
    if atype == "boolean":
        return "b:" + str(bool(answer))
    if atype == "list":
        return "l:" + json.dumps([_elem_key(x) for x in answer], ensure_ascii=False)
    if atype == "table":
        cols = [norm_string(c) for c in answer["columns"]]
        rows = sorted(json.dumps([_elem_key(x) for x in r], ensure_ascii=False) for r in answer["rows"])
        return "t:" + json.dumps({"c": cols, "r": rows}, ensure_ascii=False)
    return f"?:{atype}:{answer!r}"


# --------------------------------------------------------------------------- records
def make_record(qid: str, atype: str, answer: Any = None, reason: str | None = None) -> dict:
    if atype == "abstain":
        return {"id": qid, "type": "abstain"}
    if atype == "unanswerable":
        rec = {"id": qid, "type": "unanswerable"}
        if reason:
            rec["reason"] = str(reason)[:300]
        return rec
    return {"id": qid, "type": atype, "answer": answer}
