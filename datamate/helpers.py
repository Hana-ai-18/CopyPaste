"""Helper functions injected into the sandbox namespace used by LLM-generated code.

All date ranges are INCLUSIVE `(start_date, end_date)` tuples of `datetime.date`.
The rulebook of the split always has the last word on definitions; these helpers only
implement the usual defaults (week = Monday..Sunday, "this month" = month-to-date, ...).
"""
from __future__ import annotations

import calendar as _cal
import datetime as dt
import difflib
import re
from typing import Callable

import numpy as np
import pandas as pd

from .answer_format import round_half_up
from .textutil import strip_accents, vi_key

Date = dt.date
Range = "tuple[dt.date, dt.date]"


# --------------------------------------------------------------------------- rounding
def round_vnd(x) -> int:
    """Round half-up to an integer (use for VND amounts, counts and quantities)."""
    return int(round_half_up(x, 0))


def round2(x) -> float:
    """Round half-up to 2 decimals (use for ratios, averages, percents)."""
    return float(round_half_up(x, 2))


def pct_change(new, old):
    """Percent change from old to new in percent units (-12.45 means a 12.45% decrease)."""
    if old in (0, None) or pd.isna(old):
        raise ValueError("pct_change: old value is 0/NaN -> undefined")
    return round2((new - old) / abs(old) * 100.0)


# --------------------------------------------------------------------------- date ranges
def _d(x) -> dt.date:
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x)[:10])


def week_range(d, week_start: int = 0):
    """Full week containing `d`. week_start=0 -> Monday..Sunday (default rule)."""
    d = _d(d)
    start = d - dt.timedelta(days=(d.weekday() - week_start) % 7)
    return start, start + dt.timedelta(days=6)


def last_week(d):
    """The complete week before the week containing `d` ("tuần rồi / tuần trước")."""
    return week_range(_d(d) - dt.timedelta(days=7))


def this_week_to_date(d):
    """Monday of d's week .. d ("tuần này" up to the as-of date)."""
    d = _d(d)
    return week_range(d)[0], d


def month_range(year: int, month: int):
    return dt.date(year, month, 1), dt.date(year, month, _cal.monthrange(year, month)[1])


def this_month_to_date(d):
    """First day of d's month .. d ("tháng này")."""
    d = _d(d)
    return d.replace(day=1), d


def last_month(d):
    """The complete month before d's month ("tháng trước / tháng rồi")."""
    d = _d(d)
    first = d.replace(day=1)
    prev_end = first - dt.timedelta(days=1)
    return month_range(prev_end.year, prev_end.month)


def same_period_prev_month(d):
    """Previous month, from day 1 to the same day-of-month as d (clipped to month length)."""
    d = _d(d)
    s, e = last_month(d)
    return s, e.replace(day=min(d.day, e.day))


def quarter_range(year: int, q: int):
    m = (q - 1) * 3 + 1
    return dt.date(year, m, 1), month_range(year, m + 2)[1]


def last_quarter(d):
    d = _d(d)
    q = (d.month - 1) // 3 + 1
    return quarter_range(d.year, q - 1) if q > 1 else quarter_range(d.year - 1, 4)


def this_quarter_to_date(d):
    d = _d(d)
    return quarter_range(d.year, (d.month - 1) // 3 + 1)[0], d


def year_range(year: int):
    return dt.date(year, 1, 1), dt.date(year, 12, 31)


def this_year_to_date(d):
    d = _d(d)
    return dt.date(d.year, 1, 1), d


def last_year(d):
    return year_range(_d(d).year - 1)


def last_n_days(d, n: int, include_today: bool = False):
    """The n days before d (d excluded) or, with include_today=True, the n days ending on d."""
    d = _d(d)
    end = d if include_today else d - dt.timedelta(days=1)
    return end - dt.timedelta(days=n - 1), end


def shift_years(d, years: int) -> dt.date:
    d = _d(d)
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year + years, day=28)


def same_period_last_year(start, end):
    """Shift an inclusive range back one year ("cùng kỳ năm ngoái")."""
    return shift_years(start, -1), shift_years(end, -1)


def days_in(start, end) -> int:
    return (_d(end) - _d(start)).days + 1


# --------------------------------------------------------------------------- dataframe helpers
def parse_dates(s) -> pd.Series:
    """Robustly parse a column to tz-naive datetime64 (ISO first, then day-first d/m/Y)."""
    s = s if isinstance(s, pd.Series) else pd.Series(s)
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif pd.api.types.is_numeric_dtype(s):
        nn = s.dropna()
        if len(nn) and nn.between(20000, 80000).all():  # Excel serial dates
            out = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
        else:
            out = pd.to_datetime(s, errors="coerce")
    else:
        # Every value is parsed in ITS OWN format (never apply day-first to an ISO string: 2026-03-02 would flip to 2 Mar -> 3 Feb).
        st = s.astype("string").str.strip()
        nonnull = (st.notna() & (st != "")).to_numpy(dtype=bool)
        out = pd.to_datetime(st, format="ISO8601", errors="coerce")
        rem = nonnull & out.isna().to_numpy(dtype=bool)
        if rem.any():
            sub = st[rem]
            ymd = sub.str.match(r"^\d{4}[/.]\d{1,2}[/.]\d{1,2}", na=False).to_numpy(dtype=bool)
            if ymd.any():  # 2026/3/4 or 2026.03.04  (year first)
                fixed = sub[ymd].str.replace(r"[/.]", "-", regex=True).str.replace(r"(?<!\d)(\d)(?!\d)", r"0\1", regex=True)
                out.loc[fixed.index] = pd.to_datetime(fixed, format="ISO8601", errors="coerce")
            rest = rem & out.isna().to_numpy(dtype=bool)
            if rest.any():  # Vietnamese style d/m/Y (optionally with time)
                out.loc[st[rest].index] = pd.to_datetime(st[rest], dayfirst=True, format="mixed", errors="coerce")
            rest = nonnull & out.isna().to_numpy(dtype=bool)
            if rest.any():  # '1 thg 1 2025' / '1 tháng 1 năm 2025'  and  Excel serial numbers stored as text ('45658')
                sub = st[rest]
                m = sub.str.extract(r"^(\d{1,2})\s*(?:thg|th|tháng|thang)\s*(\d{1,2})\s*(?:năm|nam)?\s*(\d{4})", expand=True, flags=re.I)
                ok = m.notna().all(axis=1)
                if ok.any():
                    out.loc[m[ok].index] = pd.to_datetime(dict(year=m[ok][2].astype(int), month=m[ok][1].astype(int), day=m[ok][0].astype(int)), errors="coerce")
                serial = sub.str.fullmatch(r"\d{5}(\.0+)?", na=False) & ~ok
                if serial.any():
                    out.loc[sub[serial].index] = pd.to_datetime(sub[serial].astype(float), unit="D", origin="1899-12-30", errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out


def to_num(s) -> pd.Series:
    """Text -> float: '38,000' / '38.000' / '1.250.000đ' (thousand separators) and '12,5' (decimal comma) are handled; junk -> NaN."""
    import re as _re

    def one(v):
        if v is None or (isinstance(v, float) and v != v):
            return float("nan")
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        if len(_re.findall(r"\d[\d.,]*", str(v))) != 1:       # none, or several numbers ('1 bao 25kg') -> not a plain number
            return float("nan")
        t = _re.sub(r"[^0-9,.\-]", "", str(v))
        neg = t.startswith("-")
        t = t.replace("-", "")
        if "," in t and "." in t:
            dec = "," if t.rfind(",") > t.rfind(".") else "."
            t = t.replace("." if dec == "," else ",", "").replace(dec, ".")
        elif t.count(",") + t.count(".") >= 1:
            sep = "," if "," in t else "."
            parts = t.split(sep)
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and parts[0] != "0" and parts[0] != ""):
                t = "".join(parts)                      # thousands separator
            else:
                t = ".".join(parts)                     # decimal separator
        try:
            return -float(t) if neg else float(t)
        except ValueError:
            return float("nan")

    if isinstance(s, pd.Series):
        return s.map(one).astype("float64")
    return one(s)


_ST_HUY = re.compile(r"\bhuy\b|cancel|void")
_ST_TRA = re.compile(r"tra hang|tra lai|hoan tra|hoan hang|doi tra|hang tra ve|returned|refund|khach tra|khach mang tra")
_ST_DANG = re.compile(r"dang giao|dang van chuyen|dang gui|dang ship|cho giao|chua giao|dang xu ly|shipping|pending|processing")
_ST_XONG = re.compile(r"hoan thanh|\bxong\b|thanh cong|da giao|da lay|completed|\bdone\b|da nhap hang")


def norm_status(s) -> pd.Series:
    """Order status spelled in many ways -> 'hoan_thanh' | 'huy' | 'tra_hang' | 'dang_giao' | 'khac' (unknown: look at it yourself).
    e.g. xong/Đã hoàn thành/completed -> hoan_thanh; hủy đơn/Đã huỷ/khách hủy/cancelled -> huy; trả lại/Hoàn trả/Đổi trả/returned -> tra_hang;
    Đang giao dịch/Đang vận chuyển/shipping -> dang_giao. Always print(pd.crosstab(raw, norm)) once to check the mapping."""

    def one(v):
        if not isinstance(v, str):
            return "khac"
        t = strip_accents(v).lower().strip()
        if "khong giao duoc" in t:
            return "khac"
        if _ST_HUY.search(t):
            return "huy"
        if _ST_TRA.search(t):
            return "tra_hang"
        if _ST_DANG.search(t):
            return "dang_giao"
        if _ST_XONG.search(t) or t == "ok":
            return "hoan_thanh"
        return "khac"

    return (s if isinstance(s, pd.Series) else pd.Series(s)).map(one)


def rows_in_period(dates, start, end) -> int:
    """Number of rows whose date lies in [start, end] (inclusive). 0 for the WHOLE period => the shop has no data there =>
    the question is unanswerable (rulebook: no data is not the same as zero). Partly covered periods are computed on what exists."""
    d = parse_dates(dates)
    return int(((d >= pd.Timestamp(_d(start))) & (d < pd.Timestamp(_d(end)) + pd.Timedelta(days=1))).sum())


def data_span(dates):
    """(first_date, last_date) of a date column, as datetime.date."""
    d = parse_dates(dates).dropna()
    return (d.min().date(), d.max().date()) if len(d) else (None, None)


def between(series, start, end) -> pd.Series:
    """Boolean mask: date part of `series` within [start, end] inclusive."""
    d = parse_dates(series).dt.normalize()
    return (d >= pd.Timestamp(_d(start))) & (d <= pd.Timestamp(_d(end)))


def rank_top(df: pd.DataFrame, value_col: str, code_col: str, k: int | None = None, ascending: bool = False):
    """Rank rows by `value_col` (descending unless ascending=True), ties by `code_col` ascending.

    This is the official tie-break rule for rankings. Returns the top-k rows (all if k is None).
    """
    out = df.sort_values([value_col, code_col], ascending=[ascending, True], kind="mergesort")
    return out if k is None else out.head(k)


def contains_vi(series: pd.Series, text: str) -> pd.Series:
    """Accent- and case-insensitive substring match on a text column."""
    key = vi_key(text)
    return series.astype("string").map(lambda x: key in vi_key(x) if isinstance(x, str) else False).astype(bool)


def fuzzy_find(values, query: str, n: int = 5, cutoff: float = 0.5):
    """Best matches of `query` among `values` (accent-insensitive). Returns [(value, score), ...]."""
    uniq = [v for v in pd.Series(list(values)).dropna().astype(str).unique()]
    keys = {v: vi_key(v) for v in uniq}
    q = vi_key(query)
    scored = []
    for v, k in keys.items():
        if not k:
            continue
        r = difflib.SequenceMatcher(None, q, k).ratio()
        if q and q in k:
            r = max(r, 0.9)
        if r >= cutoff:
            scored.append((v, round(r, 3)))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:n]


# --------------------------------------------------------------------------- lunar calendar
def _find_cols(df: pd.DataFrame):
    keys = {c: set(vi_key(str(c).replace("_", " ")).split()) for c in df.columns}
    lun = {"lunar", "am", "al", "amlich"}
    month_t, day_t, leap_t, year_t = {"month", "thang"}, {"day", "ngay"}, {"leap", "nhuan"}, {"year", "nam"}

    def pick(*tests):
        hits = [c for c, k in keys.items() if all(any(t & k for t in [tt]) for tt in tests)]
        return hits[0] if len(hits) == 1 else (hits[0] if hits else None)

    month = pick(lun, month_t)
    day = pick(lun, day_t)
    leap = pick(leap_t)
    year = pick(lun, year_t)
    date = None
    for c, k in keys.items():
        if (k & {"solar", "duong", "gregorian", "date"}) and not (k & lun):
            date = c
            break
    if date is None:
        for c in df.columns:
            if c in (month, day, leap, year):
                continue
            try:
                if parse_dates(df[c]).notna().mean() >= 0.9 and not pd.api.types.is_numeric_dtype(df[c]):
                    date = c
                    break
            except Exception:
                continue
    return date, month, day, leap, year


class LunarCalendar:
    """Solar<->lunar lookups built from the split's `calendar.csv` (format auto-detected)."""

    def __init__(self, df: pd.DataFrame):
        date, month, day, leap, _year = _find_cols(df)
        if date is None or month is None or day is None:
            raise ValueError(f"cannot detect solar-date/lunar-month/lunar-day columns in calendar columns {list(df.columns)}")
        d = parse_dates(df[date]).dt.date
        m = pd.to_numeric(df[month], errors="coerce")
        dd = pd.to_numeric(df[day], errors="coerce")
        if leap is not None:
            lp = df[leap].astype("string").str.strip().str.lower().isin(["1", "true", "t", "yes", "y", "nhuận", "nhuan", "leap"])
        else:
            lp = pd.Series(False, index=df.index)
        rows = [(a, int(b), int(c), bool(e)) for a, b, c, e in zip(d, m, dd, lp) if a is not None and not pd.isna(a) and not pd.isna(b) and not pd.isna(c)]
        rows.sort(key=lambda t: t[0])
        self.rows = rows
        self._by_date = {r[0]: r for r in rows}

    def lunar_of(self, d):
        """(lunar_month, lunar_day, is_leap) of a solar date, or None if not in the table."""
        r = self._by_date.get(_d(d))
        return None if r is None else (r[1], r[2], r[3])

    def lunar_dates(self, month: int, day: int, leap: bool = False):
        """All solar dates whose lunar date is (month, day)."""
        return [r[0] for r in self.rows if r[1] == month and r[2] == day and r[3] == leap]

    def tet_date(self, year: int) -> dt.date:
        """Solar date of mùng 1 Tết (lunar 1/1) falling in Gregorian `year`."""
        c = [x for x in self.lunar_dates(1, 1) if x.year == year]
        if not c:
            raise ValueError(f"Tet {year} not found in calendar")
        return c[0]

    def tet_window(self, year: int):
        """Default Tet window: 23 tháng Chạp (lunar 12/23) of the previous lunar year .. mùng 10 (lunar 1/10)."""
        tet = self.tet_date(year)
        before = [x for x in self.lunar_dates(12, 23) if x < tet]
        if not before:
            raise ValueError(f"23 Chạp before Tet {year} not found in calendar")
        start = max(before)
        after = [x for x in self.lunar_dates(1, 10) if x >= tet]
        end = min(after) if after else tet + dt.timedelta(days=9)
        return start, end


# --------------------------------------------------------------------------- namespace builder
def build_helpers(as_of: dt.date, lunar: LunarCalendar | None) -> dict[str, Callable]:
    """Functions with `as_of`/`lunar` bound; the dict is merged into the sandbox namespace."""

    def tet_window(year: int):
        """(start, end) of the Tet window of Gregorian `year` (23 tháng Chạp .. mùng 10), from calendar.csv."""
        if lunar is None:
            raise ValueError("calendar.csv could not be parsed; use calendar_df directly")
        return lunar.tet_window(year)

    def tet_date(year: int):
        """Solar date of mùng 1 Tết in Gregorian `year`, from calendar.csv."""
        if lunar is None:
            raise ValueError("calendar.csv could not be parsed; use calendar_df directly")
        return lunar.tet_date(year)

    def lunar_of(d):
        """(lunar_month, lunar_day, is_leap) of a solar date, from calendar.csv."""
        if lunar is None:
            raise ValueError("calendar.csv could not be parsed; use calendar_df directly")
        return lunar.lunar_of(d)

    funcs = {
        "round_vnd": round_vnd, "round2": round2, "pct_change": pct_change,
        "week_range": week_range, "last_week": last_week, "this_week_to_date": this_week_to_date,
        "month_range": month_range, "this_month_to_date": this_month_to_date, "last_month": last_month,
        "same_period_prev_month": same_period_prev_month, "quarter_range": quarter_range,
        "last_quarter": last_quarter, "this_quarter_to_date": this_quarter_to_date,
        "year_range": year_range, "this_year_to_date": this_year_to_date, "last_year": last_year,
        "last_n_days": last_n_days, "same_period_last_year": same_period_last_year, "shift_years": shift_years,
        "days_in": days_in, "parse_dates": parse_dates, "to_num": to_num, "rows_in_period": rows_in_period, "norm_status": norm_status, "data_span": data_span, "between": between, "rank_top": rank_top,
        "contains_vi": contains_vi, "fuzzy_find": fuzzy_find, "vi_key": vi_key, "strip_accents": strip_accents,
        "tet_window": tet_window, "tet_date": tet_date, "lunar_of": lunar_of,
    }
    return funcs


def helper_docs(funcs: dict[str, Callable] | None = None) -> str:
    """One line per helper (name(signature) - first docstring line) for the prompt."""
    import inspect

    funcs = funcs or build_helpers(dt.date(2000, 1, 1), None)
    lines = []
    for name, fn in funcs.items():
        doc = (inspect.getdoc(fn) or "").split("\n")[0]
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"
        lines.append(f"- {name}{sig}: {doc}" if doc else f"- {name}{sig}")
    return "\n".join(lines)
