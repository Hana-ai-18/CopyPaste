"""Runtime lineage checks inside the sandbox: catch a CLASS of silent mistakes by what the code DOES to the data, not by column names.

drop_duplicates(subset=K) keeps one row per key. If, inside the dropped groups, a numeric-looking column took DIFFERENT values, information was
thrown away (typical: order lines deduplicated on the order id, then a line amount summed -> one line per order is counted). Each such event is
recorded; the audit later flags it when the code also refers to a lost column (it will probably aggregate it).
"""
from __future__ import annotations

import pandas as pd

MAX_ROWS = 400_000
_MIN_GROUPS_VARYING = 0.2


def _numeric_like(s: pd.Series) -> bool:
    from .helpers import to_num

    v = s.dropna()
    if v.empty:
        return False
    if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
        return True
    smp = v.astype(str).head(300)
    try:
        return float(to_num(smp).notna().mean()) >= 0.8
    except Exception:  # noqa: BLE001
        return False


def lost_columns(df: pd.DataFrame, subset) -> list[str]:
    """Numeric-like columns (not in the key) whose values differ between rows that share the key."""
    keys = [subset] if isinstance(subset, str) else list(subset)
    keys = [k for k in keys if k in df.columns]
    if not keys or len(df) > MAX_ROWS:
        return []
    dup = df[df.duplicated(subset=keys, keep=False)]
    if dup.empty:
        return []
    if len(dup) > 60_000:
        dup = dup.sample(60_000, random_state=0)
    g = dup.groupby(keys, sort=False, dropna=False)
    out = []
    for c in df.columns:
        if c in keys:
            continue
        try:
            if float((g[c].nunique(dropna=False) > 1).mean()) >= _MIN_GROUPS_VARYING and _numeric_like(df[c]):
                out.append(str(c))
        except Exception:  # noqa: BLE001
            continue
    return out


def install(sink: list) -> None:
    """Patch DataFrame.drop_duplicates (in the throw-away child process only) so lossy deduplications are appended to `sink`."""
    orig = pd.DataFrame.drop_duplicates

    def drop_duplicates(self, subset=None, **kw):
        out = orig(self, subset, **kw)
        try:
            if subset is not None and not kw.get("inplace") and out is not None and len(out) < len(self):
                lost = lost_columns(self, subset)
                if lost:
                    keys = [subset] if isinstance(subset, str) else list(subset)
                    sink.append({"key": [str(k) for k in keys], "lost": lost[:8], "rows_before": int(len(self)), "rows_after": int(len(out))})
        except Exception:  # noqa: BLE001 - instrumentation must never break the user's code
            pass
        return out

    pd.DataFrame.drop_duplicates = drop_duplicates   # type: ignore[method-assign]


# --------------------------------------------------------------------------- more lineage classes
def _name(x) -> str:
    return str(getattr(x, "name", None) or "series")


def _examples(series: pd.Series, mask: pd.Series, k: int = 3) -> list[str]:
    try:
        return [str(v)[:30] for v in series[mask].dropna().astype(str).unique()[:k]]
    except Exception:  # noqa: BLE001
        return []


def wrap_helpers(ns: dict, sink: list) -> None:
    """Wrap parse_dates / to_num / norm_status of the sandbox namespace: record values they could NOT understand (they would be dropped silently by the next filter)."""
    pd_orig, tn_orig, ns_orig = ns.get("parse_dates"), ns.get("to_num"), ns.get("norm_status")

    def parse_dates(s, *a, **k):
        out = pd_orig(s, *a, **k)
        try:
            src = pd.Series(s)
            nn = src.notna() & (src.astype(str).str.strip() != "")
            bad = nn.to_numpy() & pd.Series(out).isna().to_numpy()
            if int(bad.sum()) >= 3 and bad.sum() / max(int(nn.sum()), 1) >= 0.01:
                sink.append({"kind": "unparsed", "fn": "parse_dates", "name": _name(s), "bad": int(bad.sum()), "total": int(nn.sum()), "examples": _examples(src, pd.Series(bad, index=src.index))})
        except Exception:  # noqa: BLE001
            pass
        return out

    def to_num(s, *a, **k):
        out = tn_orig(s, *a, **k)
        try:
            src = pd.Series(s)
            nn = src.notna() & (src.astype(str).str.strip() != "")
            bad = nn.to_numpy() & pd.Series(out).isna().to_numpy()
            if int(bad.sum()) >= 3 and bad.sum() / max(int(nn.sum()), 1) >= 0.01:
                sink.append({"kind": "unparsed", "fn": "to_num", "name": _name(s), "bad": int(bad.sum()), "total": int(nn.sum()), "examples": _examples(src, pd.Series(bad, index=src.index))})
        except Exception:  # noqa: BLE001
            pass
        return out

    def norm_status(s, *a, **k):
        out = ns_orig(s, *a, **k)
        try:
            o = pd.Series(out)
            unk = (o == "khac").to_numpy()
            if int(unk.sum()) >= 3 and unk.sum() / max(len(o), 1) >= 0.005:
                sink.append({"kind": "unclassified_status", "name": _name(s), "bad": int(unk.sum()), "total": int(len(o)), "examples": _examples(pd.Series(s), pd.Series(unk, index=pd.Series(s).index))})
        except Exception:  # noqa: BLE001
            pass
        return out

    for fn, orig in ((parse_dates, pd_orig), (to_num, tn_orig), (norm_status, ns_orig)):
        if orig is not None:
            ns[fn.__name__] = fn


def install_merge(sink: list) -> None:
    """Record a merge that REPLICATES rows of the left frame (one-to-many): a sum of a left numeric column afterwards is inflated."""
    orig = pd.DataFrame.merge

    def merge(self, right, *a, **k):
        out = orig(self, right, *a, **k)
        try:
            if isinstance(out, pd.DataFrame) and len(out) > len(self) * 1.02 and len(self) > 0:
                num = [str(c) for c in self.columns if c in out.columns and _numeric_like(self[c])][:8]
                sink.append({"kind": "fanout", "left_rows": int(len(self)), "out_rows": int(len(out)), "left_numeric": num})
        except Exception:  # noqa: BLE001
            pass
        return out

    pd.DataFrame.merge = merge   # type: ignore[method-assign]
