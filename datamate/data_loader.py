"""Load a business data package (csv / xlsx / sqlite + README.md) into pandas DataFrames."""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .textutil import strip_accents

CSV_EXT = {".csv", ".tsv", ".txt"}
XLS_EXT = {".xlsx", ".xlsm", ".xls"}
DB_EXT = {".sqlite", ".sqlite3", ".db"}
JSON_EXT = {".json", ".jsonl", ".ndjson", ".parquet"}


@dataclass
class BusinessData:
    business_id: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    readme: str = ""
    warnings: list[str] = field(default_factory=list)
    merge_notes: list[str] = field(default_factory=list)
    stems: dict[str, str] = field(default_factory=dict)   # table name -> file stem it came from


def safe_identifier(name: str) -> str:
    s = strip_accents(name).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s:
        s = "table"
    if s[0].isdigit():
        s = "t_" + s
    return s


def _unique(name: str, taken: set[str]) -> str:
    base, i = name, 2
    while name in taken:
        name = f"{base}_{i}"
        i += 1
    taken.add(name)
    return name


def _sniff_sep(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(8192).decode("utf-8-sig", errors="replace")
    first = head.splitlines()[0] if head.splitlines() else ""
    counts = {s: first.count(s) for s in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_csv_robust(path: Path) -> pd.DataFrame:
    sep = _sniff_sep(path)
    with open(path, "rb") as f:
        bom = f.read(2)
    encs = ["utf-8-sig", "utf-8"]
    if bom in (b"\xff\xfe", b"\xfe\xff"):
        encs = ["utf-16"] + encs
    encs += ["cp1258", "cp1252"]
    last = None
    for enc in encs:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last = e
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, engine="python", on_bad_lines="skip")
    raise last  # pragma: no cover


def _bad_header(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    return bool(cols) and sum(c.startswith("Unnamed:") for c in cols) >= max(2, len(cols) // 2)


def _fix_header(raw: pd.DataFrame) -> pd.DataFrame:
    """raw was read with header=None; locate the real header row within the first 15 rows."""
    maxn = int(raw.notna().sum(axis=1).max()) if len(raw) else 0
    for i in range(min(15, len(raw))):
        row = raw.iloc[i]
        nn = int(row.notna().sum())
        strs = sum(isinstance(v, str) for v in row.dropna())
        if nn >= 0.6 * maxn and strs >= 0.6 * max(nn, 1):
            df = raw.iloc[i + 1:].copy()
            df.columns = [str(v).strip() if pd.notna(v) else f"col_{j}" for j, v in enumerate(row)]
            df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
            for c in df.columns:
                col = df[c]
                nn = col.dropna()
                if len(nn) and nn.map(lambda v: isinstance(v, (_dt.datetime, _dt.date))).all():
                    df[c] = pd.to_datetime(col)  # real Excel dates: never let to_numeric turn them into integers
                    continue
                try:
                    df[c] = pd.to_numeric(col)
                except (ValueError, TypeError):
                    pass
            return df
    return raw


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        c = str(c).strip()
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    df.columns = cols
    return df


def _read_excel(path: Path, warnings: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except ImportError as e:
        warnings.append(f"{path.name}: cannot read ({e})")
        return out
    for sname, df in sheets.items():
        if _bad_header(df):
            raw = pd.read_excel(path, sheet_name=sname, header=None)
            df = _fix_header(raw)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        out[sname] = _dedupe_columns(df.reset_index(drop=True))
    return out


def _read_json(path: Path) -> dict[str, pd.DataFrame]:
    """Unseen shops might export JSON / JSON-lines / parquet: a list of records, or a dict of {table: records}."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        return {path.stem: pd.read_parquet(path)}
    if ext in (".jsonl", ".ndjson"):
        return {path.stem: pd.read_json(path, lines=True)}
    import json as _json

    obj = _json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    if isinstance(obj, dict) and obj and all(isinstance(v, list) for v in obj.values()):
        return {k: pd.json_normalize(v) for k, v in obj.items() if v}
    if isinstance(obj, dict):                       # {"data": [...]} or a single record
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return {path.stem: pd.json_normalize(v)}
        return {path.stem: pd.json_normalize(obj)}
    return {path.stem: pd.json_normalize(obj)}


def _read_sqlite(path: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for n in names:
            out[n] = _dedupe_columns(pd.read_sql_query(f'SELECT * FROM "{n}"', conn))
    finally:
        conn.close()
    return out


_STEM_NOISE = re.compile(r"(\d+|thang|quy|sheet|nam|^t(?=_)|(?<=_)t(?=_|$))")


def _merged_name(names: list[str], first_col: str, taken: set[str]) -> str:
    from collections import Counter

    stems = [re.sub(r"_+", "_", _STEM_NOISE.sub("", n)).strip("_") for n in names]
    stems = [x for x in stems if len(x) >= 4]
    if stems:
        top = Counter(stems).most_common()
        best = min((s for s, c in top if c == top[0][1]), key=len)
    else:
        best = "all_" + safe_identifier(first_col)
    return _unique(best, taken)


def merge_same_schema(biz: BusinessData, min_tables: int = 2) -> None:
    """Many shops split one list over several sheets/files (monthly sheets, _1/_2 parts): union the tables that have EXACTLY
    the same columns into one table (+ column `_sheet` = original table name), so the LLM does not have to concat them."""
    groups: dict[tuple, list[str]] = {}
    for name, df in biz.tables.items():
        if len(df.columns) >= 3:
            groups.setdefault(tuple(df.columns), []).append(name)
    todo = [names for names in groups.values() if len(names) >= min_tables]
    if not todo:
        return
    taken = {n for names in todo for n in names}
    for n in list(biz.tables):
        if n not in taken:
            pass
    keep_taken = set(biz.tables) - taken
    for names in todo:
        cols = list(biz.tables[names[0]].columns)
        file_stems = {re.sub(r"(_?\d+)+$", "", biz.stems[n]) if biz.stems.get(n) else None for n in names}   # SoBanHang_1/_2026 -> sobanhang
        if len(file_stems) == 1 and None not in file_stems:       # all sheets of ONE file -> the file's name (as in the README)
            name = _unique(file_stems.pop(), keep_taken)
        else:
            name = _merged_name(names, str(cols[0]), keep_taken)
        parts = []
        for n in names:
            df = biz.tables.pop(n)
            df = df.copy()
            df["_sheet"] = n
            parts.append(df)
        biz.tables[name] = pd.concat(parts, ignore_index=True)
        biz.stems[name] = name
        biz.merge_notes.append(f"table `{name}` = {len(names)} sheets/files with identical columns concatenated ({names[0]} ... {names[-1]}); "
                               f"column `_sheet` keeps the original sheet/file name (use it for dates written without a year). Rows: {len(biz.tables[name])}")


_SUMMARY = re.compile(r"^(tong cong|tong|cong|total|grand total|tong ket)\s*:?$|^ghi chu\s*:|^luu y\s*:")


_ERR_TOKENS = {"#n/a", "#div/0!", "#ref!", "#value!", "#name?", "#num!", "#null!", "#n/a!"}
_WSRE = re.compile(r"[\s\u00a0\u200b\u200c\u200d\ufeff]+")


def _clean_text(v: str) -> str | None:
    t = _WSRE.sub(" ", unicodedata.normalize("NFC", v)).strip()
    return None if t.lower() in _ERR_TOKENS else t


def clean_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Unseen exports differ in invisible ways: decomposed (NFD) Vietnamese accents, non-breaking / zero-width spaces, trailing blanks
    ('Cà phê ' vs 'Cà phê' split a groupby), Excel error cells (#N/A, #DIV/0!). Normalise column names and every text cell (once per DISTINCT value)."""
    df = df.copy()
    df.columns = [_WSRE.sub(" ", unicodedata.normalize("NFC", str(c))).strip() for c in df.columns]
    for c in df.columns:
        col = df[c]
        if col.dtype != object and not pd.api.types.is_string_dtype(col):
            continue
        uniq = [u for u in col.dropna().unique() if isinstance(u, str)]
        if not uniq:
            continue
        mp = {u: _clean_text(u) for u in uniq}
        if all(k == v for k, v in mp.items()):
            continue
        df[c] = col.map(lambda x, mp=mp: mp.get(x, x) if isinstance(x, str) else x)
    return df


def drop_summary_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Hand-kept books contain 'Tổng cộng' rows (monthly totals), footnotes ('Ghi chú: ...') and repeated header rows inside the
    data. They are not records: summing over them double counts. Returns (clean df, number of rows dropped)."""
    if df.empty or len(df.columns) < 2:
        return df, 0
    obj = df.select_dtypes(exclude=["number", "datetime"])
    if obj.empty:
        return df, 0
    def _norm_col(c: pd.Series) -> pd.Series:   # normalise each DISTINCT value once (columns repeat a lot) instead of every cell
        s = c.astype("string")
        mp = {v: strip_accents(v).lower().strip() for v in s.dropna().unique()}
        return s.map(mp).fillna("").astype(object)

    keys = obj.apply(_norm_col)
    summ = keys.apply(lambda c: c.str.match(_SUMMARY)).any(axis=1)
    mostly_empty = (df.isna().mean(axis=1) >= 0.6) | (df.notna().sum(axis=1) <= 3)   # label + at most a couple of values
    colnames = {strip_accents(str(c)).lower().strip() for c in df.columns}
    header_like = keys.apply(lambda c: c.isin(colnames) & (c != "")).sum(axis=1) >= 2
    drop = (summ & mostly_empty) | header_like
    n = int(drop.sum())
    return (df[~drop].reset_index(drop=True), n) if n else (df, 0)


_UNIT_LINE = re.compile(r"^\s*[-*]\s*(.+?)\s*:\s*1\s+(.+?)\s*=\s*([\d][\d.,]*)\s*(.+?)\s*$")


def parse_readme_units(readme: str) -> pd.DataFrame:
    """Half of the shops give the unit conversions only as README lines like '- Coca Cola 330ml: 1 thùng = 24 lon'.
    -> DataFrame(ten_hang, don_vi_lon, he_so, don_vi_co_ban). A unit that has no line is the base unit (factor 1)."""
    rows = []
    for line in readme.splitlines():
        m = _UNIT_LINE.match(line)
        if m:
            try:
                f = float(m.group(3).replace(",", "."))
            except ValueError:
                continue
            rows.append((m.group(1).strip(" `*"), m.group(2).strip(), f if f != int(f) else int(f), m.group(4).strip()))
    return pd.DataFrame(rows, columns=["ten_hang", "don_vi_lon", "he_so", "don_vi_co_ban"])


def load_business(directory: str | Path, business_id: str | None = None) -> BusinessData:
    d = Path(directory)
    biz = BusinessData(business_id=business_id or d.name)
    taken: set[str] = set()
    files = sorted(p for p in d.iterdir() if p.is_file())
    pending: list[tuple[str, str, pd.DataFrame]] = []  # (stem, subname, df)
    for p in files:
        ext = p.suffix.lower()
        if p.name.lower().startswith("readme"):
            try:
                biz.readme = p.read_text(encoding="utf-8-sig", errors="replace")
            except OSError as e:  # pragma: no cover
                biz.warnings.append(f"readme unreadable: {e}")
            continue
        try:
            if ext in CSV_EXT:
                pending.append((p.stem, "", read_csv_robust(p)))
            elif ext in XLS_EXT:
                for sname, df in _read_excel(p, biz.warnings).items():
                    pending.append((p.stem, sname, df))
            elif ext in JSON_EXT:
                for tname, df in _read_json(p).items():
                    pending.append((p.stem, tname, df))
            elif ext in DB_EXT:
                for tname, df in _read_sqlite(p).items():
                    pending.append((p.stem, tname, df))
        except Exception as e:  # keep going: one broken file should not kill the business
            biz.warnings.append(f"{p.name}: failed to load ({type(e).__name__}: {e})")
    # csv -> file stem; excel sheet / sqlite table -> its own name (prefixed with the file stem on collision)
    dropped_total = dropped_tables = 0
    for stem, sub, df in pending:
        generic = bool(re.fullmatch(r"sheet\d*", sub.strip().lower())) if sub else False
        name = safe_identifier(sub if (sub and not generic) else stem)    # generic "Sheet1" -> file name (matches the README)
        if sub and not generic and name in taken:
            name = safe_identifier(f"{stem}_{sub}")
        name = _unique(name, taken)
        try:
            df = clean_text_frame(df)
        except Exception:  # noqa: BLE001
            pass
        try:
            df, n_dropped = drop_summary_rows(df)
        except Exception:  # noqa: BLE001
            n_dropped = 0
        dropped_total += n_dropped
        dropped_tables += 1 if n_dropped else 0
        biz.tables[name] = _dedupe_columns(df)
        biz.stems[name] = safe_identifier(stem)
    try:
        units = parse_readme_units(biz.readme)
        if len(units):
            name = _unique("quy_doi_readme", set(biz.tables))
            biz.tables[name] = units
            biz.readme = "\n".join(l for l in biz.readme.splitlines() if not _UNIT_LINE.match(l))   # now in the table: saves ~3k prompt chars
            biz.merge_notes.append(f"table `{name}` was parsed from the README conversion lines ({len(units)} rows: ten_hang, don_vi_lon, he_so, don_vi_co_ban); "
                                   "a unit with no row there is the base unit (factor 1)")
    except Exception as e:  # noqa: BLE001
        biz.warnings.append(f"README unit parsing failed: {e}")
    if dropped_total:
        biz.merge_notes.append(f"{dropped_total} non-record rows were removed from {dropped_tables} sheet(s): monthly 'Tổng cộng' totals, 'Ghi chú:' footnotes and repeated header rows")
    try:
        merge_same_schema(biz)
    except Exception as e:  # noqa: BLE001 - never lose the business over an optimisation
        biz.warnings.append(f"merge_same_schema failed: {type(e).__name__}: {e}")
    if biz.merge_notes:
        biz.readme = (biz.readme.rstrip() + "\n\n## Ghi chú tự động của hệ thống (đã gộp bảng)\n" + "\n".join("- " + x for x in biz.merge_notes) + "\n")
    return biz
