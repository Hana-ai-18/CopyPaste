"""Data-driven structure hints, computed from the actual tables (no hard-coded shop knowledge).

Tells the model, per table: which column is the order id, which columns are ORDER-level (constant inside an order -> count once per order)
vs LINE-level, how the status column is spelled (raw -> normalised, with counts) and which customer values are anonymous placeholders.
Everything is derived by statistics so it generalises to unseen shops; any failure just yields no hint.
"""
from __future__ import annotations

import re

import pandas as pd

from .textutil import vi_key
from .helpers import anon_customer, norm_status, strip_accents, to_num

_ID_NAME = re.compile(r"(^|[^a-z])(ma|so|id|code|no)([^a-z]|$).*(don|hoa don|order|invoice|bill|phieu)|(don|hoa don|order|invoice|bill|phieu).*(ma|so|id|code|no)([^a-z]|$)|^(order|invoice|bill)")
_STATUS_NAME = re.compile(r"trang thai|status|tinh trang")
_CUST_NAME = re.compile(r"khach|customer|\bkh\b|nguoi mua|buyer")
_SAMPLE_IDS = 3000


def _n(name) -> str:
    return strip_accents(str(name)).lower().strip()


def _order_id_col(df: pd.DataFrame) -> str | None:
    best, best_dup = None, 0.0
    for c in df.columns:
        if not _ID_NAME.search(_n(c)):
            continue
        s = df[c].dropna()
        if len(s) < 4:
            continue
        dup = 1 - s.nunique() / len(s)
        if dup > best_dup and s.nunique() >= 2:
            best, best_dup = c, dup
    return best if best is not None and best_dup > 0.05 else None


def _levels(df: pd.DataFrame, oid: str) -> tuple[list[str], list[str]]:
    ids = df[oid].dropna()
    multi = ids[ids.duplicated(keep=False)].unique()
    if len(multi) == 0:
        return [], []
    if len(multi) > _SAMPLE_IDS:
        multi = pd.Series(multi).sample(_SAMPLE_IDS, random_state=0).to_numpy()
    sub = df[df[oid].isin(multi)]
    g = sub.groupby(oid, sort=False)
    order_lv, line_lv = [], []
    for c in df.columns:
        if c == oid:
            continue
        try:
            const = (g[c].nunique(dropna=False) <= 1).mean()
        except Exception:
            continue
        (order_lv if const >= 0.98 else line_lv).append(str(c))
    return order_lv, line_lv


def _status_hint(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if df[c].dtype.kind in "ifbM":
            continue
        s = df[c].dropna().astype(str)
        if s.empty or s.nunique() > 40:
            continue
        vc = s.value_counts()
        mapped = norm_status(pd.Series(vc.index))
        known = float(vc[(mapped != "khac").to_numpy()].sum()) / float(vc.sum())
        if not (_STATUS_NAME.search(_n(c)) or known >= 0.6) or known < 0.3:
            continue
        out.append(f"- status column {str(c)!r}: " + ", ".join(f"{k!r}x{v}->{m}" for (k, v), m in zip(vc.head(14).items(), mapped.head(14))))
        if (mapped == "khac").any():
            unk = [k for k, m in zip(vc.index, mapped) if m == "khac"][:6]
            out.append(f"  unclassified by norm_status (decide yourself): {unk}")
    return out


def _customer_hint(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if df[c].dtype.kind in "ifbM":
            continue
        s = df[c]
        if s.notna().sum() == 0 or s.nunique() < 3:
            continue
        mask = anon_customer(s)
        explicit = float((mask & s.notna()).mean())          # non-blank placeholders such as 'Khách lẻ', 'kl'
        if not mask.any() or not (_CUST_NAME.search(_n(c)) or (explicit >= 0.005 and s.nunique() >= 10)):
            continue
        vc = s[mask].fillna("<blank>").astype(str).value_counts().head(6)
        out.append(f"- customer column {str(c)!r}: {round(100 * mask.mean())}% rows are anonymous placeholders "
                   + str([f"{k}x{v}" for k, v in vc.items()]) + " -> exclude with ~anon_customer() when ranking/counting customers")
    return out


def _quality_hint(df: pd.DataFrame) -> list[str]:
    """Traps that silently change sums/counts: numbers stored as text, unparseable dates, exact duplicate rows, negative amounts (returns)."""
    out = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            v = s.dropna().astype(str)
            if len(v) < 5:
                continue
            smp = v.sample(min(len(v), 400), random_state=0)
            num = to_num(smp)
            frac = float(num.notna().mean())
            if frac >= 0.8 and smp.str.contains(r"\d").mean() >= 0.8 and not _ID_NAME.search(_n(c)) and smp.str.fullmatch(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}.*").mean() < 0.5:
                out.append(f"- {str(c)!r} is NUMERIC stored as text (e.g. {list(smp.head(2))}) -> use to_num() before sum/mean/compare")
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            neg = int((s < 0).sum())
            if neg and neg / max(n, 1) > 0.001 and re.search(r"so luong|sl|qty|quantity|thanh tien|tien|amount|total|gia|price", _n(c)):
                out.append(f"- {str(c)!r} has {neg} negative values (returns/refunds?) - decide whether the question includes them")
    try:
        dup = int(df.duplicated().sum())
        if dup >= 3 and dup / max(n, 1) >= 0.002:
            out.append(f"- {dup} rows are exact duplicates of another row (could be genuine repeated lines or double entry: do NOT drop unless the question implies it)")
    except Exception:
        pass
    return out[:6]


_CAP_RULES = [
    ("stock levels / purchases (inventory, stock value, goods received)", r"nhap hang|phieu nhap|don nhap|nhap kho|ton kho|ton cuoi|ton thuc|so luong ton|the kho|stock|inventory|purchase|sl nhap|gia nhap", "tables"),
    ("debt / unpaid balance (customer owes shop)", r"con no|du no|cong no|phai thu|khach da tra|da thanh toan|con lai|tien no|^no$|^no\.|debt|paid|balance", "cols"),
    ("product cost (cost of goods, profit, margin)", r"gia von|gia nhap|don gia von|cost|von hang", "cols"),
    ("expiry date / lot", r"han su dung|\bhsd\b|han dung|expir|\blo\b|so lo|ma lo|\blot\b", "cols"),
    ("time of day / shift", r"\bgio\b|\bca\b|khung gio|thoi gian|time|hour", "cols"),
    ("product group / category (nhóm hàng)", r"nhom hang|nhom sp|nhom san pham|danh muc|category|loai hang|phan loai|ma nhom|^nhom$|group", "cols"),
    ("payment method", r"phuong thuc|hinh thuc|httt|tra bang|thanh toan bang|payment|pttt", "cols"),
]
_CAP_EXCLUDE = re.compile(r"nhom khach|khach hang group|customer group|dinh muc|ton toi thieu|ton toi da|ton nho nhat|ton lon nhat|min|max")


def _cap_n(name) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", strip_accents(str(name)))        # camelCase -> words
    return re.sub(r"[\s_]+", " ", s).lower().strip()


def capability_hints(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Which kinds of facts the shop actually records, judged ONLY from table/column names. Drives answer-vs-unanswerable: a question about
    stock / debt / cost / expiry / shift when nothing in the data holds it is unanswerable (but check the rulebook and the data before concluding)."""
    out, missing = [], []
    for label, pat, where in _CAP_RULES:
        rx = re.compile(pat)
        found = []
        for t, df in tables.items():
            tn = _cap_n(t)
            if where in ("tables", "both") and rx.search(tn):
                found.append(f"table {t}")
            for c in df.columns:
                cn = _cap_n(c)
                if _CAP_EXCLUDE.search(cn):
                    continue
                if rx.search(cn):
                    found.append(f"{t}.{c}")
        if found:
            out.append(f"- {label}: found {sorted(set(found))[:4]}")
        else:
            missing.append(label.split(" (")[0])
    if missing:
        out.append("- NOT found by name (questions about these are unanswerable unless the rulebook/data derives them): " + "; ".join(missing))
    return out


_HONORIFIC = {"a", "anh", "c", "chi", "e", "em", "ong", "ba", "co", "chu", "bac", "thay", "mr", "ms", "ky", "thuat", "vien"}


def _variant_key(v: str) -> str:
    toks = [t for t in vi_key(v).split() if t]
    while len(toks) > 1 and toks[0] in _HONORIFIC:
        toks = toks[1:]
    return " ".join(toks)


def _variant_hint(df: pd.DataFrame) -> list[str]:
    """Same entity spelled several ways inside one column (worker 'A.Đức' / 'Đức' / 'a Đức', product case/accents): group them so the model
    does not count them as different people/items (rulebook: names for the same thing are the same thing)."""
    out = []
    for c in df.columns:
        s = df[c]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        vals = s.dropna().astype(str)
        nun = vals.nunique()
        if nun < 3 or nun > 80:
            continue
        groups: dict[str, list[str]] = {}
        for v in vals.unique():
            groups.setdefault(_variant_key(v), []).append(v)
        multi = {k: g for k, g in groups.items() if k and len(g) > 1}
        if multi and len(multi) >= max(1, len(groups) // 4):
            ex = "; ".join("/".join(map(repr, g[:4])) for g in list(multi.values())[:4])
            out.append(f"- column {str(c)!r}: {nun} raw values but only {len(groups)} distinct once case/accents/honorific (a, anh, chị...) are ignored, e.g. {ex} -> group them before counting")
    return out[:3]


def structure_hints(tables: dict[str, pd.DataFrame], max_chars: int = 2600) -> str:
    lines: list[str] = []
    for name, df in tables.items():
        try:
            if len(df) == 0:
                continue
            block: list[str] = []
            oid = _order_id_col(df)
            if oid is not None:
                order_lv, line_lv = _levels(df, oid)
                if line_lv or order_lv:
                    block.append(f"- order id column {str(oid)!r} repeats across rows (one row = one order LINE). "
                                 f"ORDER-level (same on every line of an order; count once per order via drop_duplicates({str(oid)!r})): {order_lv[:12]}. "
                                 f"LINE-level (safe to sum per row): {line_lv[:12]}")
                    money = [c for c in line_lv if re.search(r"tien|thanh tien|amount|total|tri gia|gia tri|revenue|doanh", _n(c))]
                    if money:
                        block.append(f"- MONEY per LINE {money[:4]}: revenue = sum over ALL lines of qualifying orders (after filtering rows by period/status). NEVER drop_duplicates({str(oid)!r}) and then sum these: that keeps a single line per order. order_amounts(rows, {str(oid)!r}, col) does it safely.")
            block += _status_hint(df)
            block += _customer_hint(df)
            block += _quality_hint(df)
            block += _variant_hint(df)
            if block:
                lines.append(f"[{name}]")
                lines += block
        except Exception:
            continue
    try:
        caps = capability_hints(tables)
    except Exception:
        caps = []
    if not lines and not caps:
        return ""
    if caps:
        lines = ["[what this shop records - keyword match on names only, verify in the data]"] + caps + lines
    text = "## Detected structure (computed from the data; trust it over guesses)\n" + "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
