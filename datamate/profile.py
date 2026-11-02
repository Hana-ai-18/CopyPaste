"""Automatic data profile of a business: schema, value samples, categorical values, date ranges, join hints."""
from __future__ import annotations

import re

import pandas as pd

from .helpers import parse_dates
from .hints import structure_hints

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DMY = re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}")


def _short(v, n=40) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _date_format(sample: pd.Series) -> str | None:
    vals = sample.dropna().astype(str).head(100)
    if vals.empty:
        return None
    iso = vals.map(lambda x: bool(_ISO.match(x))).mean()
    dmy = vals.map(lambda x: bool(_DMY.match(x))).mean()
    if iso >= 0.9:
        return "ISO yyyy-mm-dd"
    if dmy >= 0.9:
        return "d/m/Y (day-first)"
    return None


def _column_line(name: str, s: pd.Series, detail: int) -> str:
    n = len(s)
    nn = int(s.notna().sum())
    null_pct = 0 if n == 0 else round(100 * (n - nn) / n)
    dtype = str(s.dtype)
    parts = [f"- {name!r} ({dtype}"]
    if null_pct:
        parts[0] += f", {null_pct}% null"
    parts[0] += ")"
    if detail == 0:
        return parts[0]
    nun = int(s.nunique(dropna=True))
    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    is_dt = pd.api.types.is_datetime64_any_dtype(s)
    extra = []
    if is_dt:
        extra.append(f"range {s.min()} .. {s.max()}")
    elif is_num:
        if nn:
            extra.append(f"min={_short(s.min())} max={_short(s.max())}")
        if nun <= 8 and nn:
            extra.append("values=" + str(sorted(s.dropna().unique().tolist())[:8]))
    else:
        fmt = _date_format(s)
        if fmt:
            try:
                p = parse_dates(pd.Series(s.dropna().unique()))   # whole column (via distinct values), not a prefix
                extra.append(f"DATE-LIKE as text, format {fmt}, range {p.min()} .. {p.max()} (use parse_dates())")
            except Exception:
                extra.append(f"DATE-LIKE as text, format {fmt} (use parse_dates())")
        if 0 < nun <= 15:
            extra.append("values=" + str([_short(v, 30) for v in s.dropna().unique().tolist()[:15]]))
        elif nn:
            ex = [_short(v) for v in s.dropna().astype(str).drop_duplicates().head(3).tolist()]
            vc = s.dropna().astype(str).value_counts().head(4)
            top = ", ".join(f"{_short(k, 24)!r}x{v}" for k, v in vc.items())     # exposes placeholders like 'Khách lẻ' / 'kl' / '' variants
            extra.append(f"{nun} distinct, e.g. {ex}; most frequent: {top}")
    return parts[0] + (" " + "; ".join(extra) if extra else "")


def _table_block(name: str, df: pd.DataFrame, detail: int, sample_rows: int) -> str:
    lines = [f"### {name}  ({len(df)} rows x {len(df.columns)} cols)"]
    for c in df.columns:
        lines.append(_column_line(c, df[c], detail))
    if detail >= 2 and sample_rows and len(df):
        lines.append("first rows:")
        for r in df.head(sample_rows).itertuples(index=False, name=None):
            lines.append("  " + str([_short(x, 25) for x in r]))
    return "\n".join(lines)


def _join_hints(tables: dict[str, pd.DataFrame]) -> str:
    seen: dict[str, list[str]] = {}
    for t, df in tables.items():
        for c in df.columns:
            seen.setdefault(str(c), []).append(t)
    hints = [f"- column {c!r} appears in: {', '.join(ts)}" for c, ts in seen.items() if len(ts) > 1]
    return "\n".join(hints[:25])


def _names_only(tables: dict[str, pd.DataFrame], per_col: int = 28) -> str:
    """Last resort for very wide schemas: EVERY table is still listed (a dropped table can hold the answer), columns by name only."""
    out = []
    for n, df in tables.items():
        cols = ", ".join(_short(c, per_col) for c in df.columns)
        out.append(f"### {n} ({len(df)} rows): {cols}")
    return "\n".join(out)


def profile_tables(tables: dict[str, pd.DataFrame], max_chars: int = 7000, sample_rows: int = 3) -> str:
    """Return a compact text profile that fits in `max_chars` (detail is reduced progressively).
    The data-derived structure hints go FIRST so tail truncation never drops them."""
    try:
        head = structure_hints(tables, max_chars=max(900, int(0.4 * max_chars)))
    except Exception:
        head = ""
    budget = max_chars - (len(head) + 2 if head else 0)
    for detail, rows in ((2, sample_rows), (1, 0), (0, 0)):
        blocks = [_table_block(n, df, detail, rows) for n, df in tables.items()]
        text = "\n\n".join(blocks)
        hints = _join_hints(tables)
        if hints:
            text += "\n\nPossible join keys (same column name in several tables):\n" + hints
        if len(text) <= budget:
            break
    else:
        text = _names_only(tables)
        if len(text) > budget:
            text = _names_only(tables, per_col=16)
        if len(text) > budget:                       # give each table an equal share so none disappears
            share = max(60, budget // max(len(tables), 1) - 1)
            text = "\n".join(l if len(l) <= share else l[: share - 1] + "…" for l in text.splitlines())
        if len(text) > budget:
            text = text[: budget - 40] + "\n… [profile truncated]"
    return (head + "\n\n" + text) if head else text
