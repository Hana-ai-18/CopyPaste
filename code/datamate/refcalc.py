"""Independent second path for the most common question family: revenue / number of orders in ONE period.

Nothing here is written per shop: the roles of the columns are induced from the data (names + values) and the rulebook defaults are coded once
(orders that are not cancelled/returned; money per order = sum of LINE amounts or the order total, whichever the data uses; period by order date).
It answers only when the question is a plain 'metric + one period' and the induction is unambiguous; otherwise it returns None (no opinion).
Use: compare with the voted answer; disagreement means "do not trust the vote".
"""
from __future__ import annotations

import re

import pandas as pd

from . import helpers as H
from .hints import _order_id_col
from .intent import _n as _n0, resolve_periods


def _n(x) -> str:
    return re.sub(r"[_\s]+", " ", _n0(x)).strip()


_REV = re.compile(r"doanh thu|dthu|doanh so|tien ban|ban (duoc )?(bao nhieu|nhieu) tien|thu ve|tong tien ban|tien hang")
_CNT = re.compile(r"(bao nhieu|may|so) (don|hoa don)\b|\bso don\b|tong so don|\bdh\b")
_QUAL = re.compile(
    r"khach|nhom|mat hang|san pham|\bmon\b|\bly\b|size|\bca\b|ca (sang|chieu)|thu (hai|ba|tu|nam|sau|bay)|chu nhat|so voi|%|phan tram|\btop\b|nhieu nhat|it nhat|trung binh|"
    r"ngay nao|thang nao|nho nhat|cao nhat|\bhuy\b|tra hang|hoan hang|\bno\b|ton kho|\bvon\b|\bloi\b|\blai\b|chi phi|giam gia|ship|nhan vien|\btho\b|bao hanh|"
    r"may ngay|bao nhieu ngay|tro len|tro xuong|dat \d|tien mat|chuyen khoan|\bcod\b|the\b|thanh toan|voucher|diem|gio|tang|giam|chenh|hon|kem|\bcao\b|thap|trong so|ti le|ty le|bao nhieu loai|moi|rieng|gom")
_BAD_AMT = re.compile(r"thanh toan|phai tra|da thu|con no|\bno\b|\bship|\bphi\b|giam gia|chiet khau|\bvat\b|thue|\bvon\b|gia von|loi nhuan|\blai\b|\bton\b|so luong|\bsl\b|don gia|gia ban|gia nhap")
_PURCHASE = re.compile(r"nhap|ncc|nha cung cap|mua vao|purchase|supplier|giao hang|phieu giao|cong no|quy doi|bao hanh|tra gop|thanh toan|phieu thu|thu tien|payment|khach hang|san pham")
_PURCHASE_COL = re.compile(r"nha cung cap|\bncc\b|supplier|gia nhap|sl nhap|so luong nhap")
_PRIO = [(re.compile(r"^thanh tien|thanh tien|^t\.? ?tien$|^tri ?gia$|^tri gia$|^amount$"), 3), (re.compile(r"tien hang|gia tri hang|tri gia hang|doanh thu"), 2), (re.compile(r"tong tien|^tong$|^tien$|so tien|amount|total"), 1)]


def _is_dateish(df: pd.DataFrame, c) -> bool:
    if not re.search(r"ngay|date|thoi gian|time|\bgio\b", _n(c)):
        return False
    s = df[c].dropna()
    if s.empty:
        return False
    smp = s.head(400)
    if s.dtype.kind not in "Mm" and float(smp.astype(str).str.contains(r"\d{4}").mean()) < 0.9:
        return False                                                       # '1/1' without a year: the year would be guessed, so this is not an order date
    try:
        return float(H.parse_dates(smp).notna().mean()) >= 0.9
    except Exception:  # noqa: BLE001
        return False


def _status_col(df: pd.DataFrame):
    best, bk = None, 0.0
    for c in df.columns:
        if df[c].dtype.kind in "ifbM":
            continue
        s = df[c].dropna().astype(str)
        if s.empty or s.nunique() > 40:
            continue
        vc = s.value_counts()
        known = float(vc[(H.norm_status(pd.Series(vc.index)) != "khac").to_numpy()].sum()) / float(vc.sum())
        nm = bool(re.search(r"trang thai|status|tinh trang|ket qua", _n(c)))
        if (nm and known >= 0.5) or known >= 0.9:
            if known > bk:
                best, bk = c, known
    return best


def _amount_col(df: pd.DataFrame):
    scored: dict[int, list] = {}
    for c in df.columns:
        n = _n(c)
        if _BAD_AMT.search(n.replace("thanh tien", "TT")):
            continue
        pr = max((w for rx, w in _PRIO if rx.search(n)), default=0)
        if not pr:
            continue
        s = df[c]
        smp = s.dropna().astype(str).head(400)
        if smp.empty or float(H.to_num(smp).notna().mean()) < 0.9:
            continue
        scored.setdefault(pr, []).append(c)
    for pr in sorted(scored, reverse=True):
        return scored[pr][0] if len(scored[pr]) == 1 else None            # ambiguous -> no opinion
    return None


def _idlike_col(df: pd.DataFrame):
    """A column of codes like ORD000123 / HD-0045 that repeats across rows (order id with an unusual name such as 'Mã GD')."""
    best, bd = None, 0.05
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if len(s) < 20:
            continue
        smp = s.head(500)
        if float(smp.str.fullmatch(r"[A-Za-z]{1,6}[-_ ]?\d{3,}").mean()) < 0.9:
            continue
        dup = 1 - s.nunique() / len(s)
        if dup > bd:
            best, bd = c, dup
    return best


def _blocky(df: pd.DataFrame) -> bool:
    """Order code / customer written on the FIRST line of each order only (merged cells): rows are lines, not orders."""
    for c in df.columns:
        na = float(df[c].isna().mean())
        if 0.15 <= na <= 0.9 and re.search(r"ma don|so don|ma hd|so hd|stt|ma phieu|ma bill|so ct|ma hoa don|ma chung tu|order|don hang", _n(c)):
            return True
    return False


def _uniq_id_col(df: pd.DataFrame):
    """The column holding one unique order code per row (ORD000123 ...). None when there is none or it is ambiguous."""
    found = []
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if len(s) < 20 or float(s.head(500).str.fullmatch(r"[A-Za-z]{1,6}[-_ ]?\d{3,}").mean()) < 0.9:
            continue
        if s.nunique() / len(s) >= 0.98 and len(s) >= 0.98 * len(df):      # no gaps: a code on the first line of a block only is NOT one row per order
            found.append(c)
    return found[0] if len(found) == 1 else None


def _date_col(df: pd.DataFrame):
    """The order-date column; several date-like columns are fine when they give the same day on (almost) every row."""
    cols = [c for c in df.columns if _is_dateish(df, c)]
    if len(cols) == 1:
        return cols[0]
    if len(cols) > 1:
        ds = [pd.to_datetime(H.parse_dates(df[c])).dt.normalize() for c in cols]
        if all(float((ds[0] == d).mean()) >= 0.99 for d in ds[1:]):
            return cols[0]
    return None


def induce_star(dfs: dict[str, pd.DataFrame]) -> dict | None:
    """Header table (one row per order: code, date, status) + a separate LINE table (order code, line amount)."""
    heads = []
    for name, df in dfs.items():
        if len(df) < 20 or _PURCHASE.search(_n(name)) or any(_PURCHASE_COL.search(_n(c)) for c in df.columns):
            continue
        oid, dc, sc = _uniq_id_col(df), _date_col(df), _status_col(df)
        if oid is not None and dc is not None and sc is not None:
            heads.append((name, oid, dc, sc, len(df)))
    if len(heads) != 1:
        return None
    hname, hid, hdate, hstat, hrows = heads[0]
    ids = set(dfs[hname][hid].dropna().astype(str))
    details = []
    for name, df in dfs.items():
        if name == hname or len(df) < hrows or _PURCHASE.search(_n(name)) or any(_PURCHASE_COL.search(_n(c)) for c in df.columns):
            continue
        amt = _amount_col(df)
        if amt is None:
            continue
        for c in df.columns:
            s = df[c].dropna().astype(str)
            if len(s) >= 20 and float(s.head(2000).isin(ids).mean()) >= 0.98 and s.nunique() < len(s):
                details.append((name, c, amt))
                break
    if len(details) != 1:
        return None
    dname, did, damt = details[0]
    return {"table": hname, "oid": hid, "date": hdate, "status": hstat, "amount": None, "detail": dname, "did": did, "damount": damt, "rows": hrows}


def induce(dfs: dict[str, pd.DataFrame]) -> dict | None:
    star = induce_star(dfs)                      # rulebook: money = sum of LINE amounts, so a linked line table wins over a header total
    return star if star is not None else _induce_one(dfs)


def _induce_one(dfs: dict[str, pd.DataFrame]) -> dict | None:
    cands = []
    for name, df in dfs.items():
        if len(df) < 20 or _PURCHASE.search(_n(name)) or any(_PURCHASE_COL.search(_n(c)) for c in df.columns):
            continue
        dcols = [c for c in df.columns if _is_dateish(df, c)]
        amt = _amount_col(df)
        if not dcols or amt is None:
            continue
        oid = _uniq_id_col(df) or _order_id_col(df) or _idlike_col(df)
        cands.append({"table": name, "date": dcols[0] if len(dcols) == 1 else None, "status": _status_col(df), "amount": amt, "oid": oid, "rows": len(df), "blocky": oid is None and _blocky(df)})
    if not cands:
        return None
    cands = [c for c in cands if c["date"] is not None and c["status"] is not None]       # no status column -> cannot apply the cancelled/returned rule: no opinion
    if not cands:
        return None
    cands.sort(key=lambda c: -c["rows"])
    if len(cands) > 1 and cands[1]["rows"] > 0.5 * cands[0]["rows"]:
        return None                                                        # two similar candidate tables: ambiguous
    return cands[0]


def _orders_star(dfs, spec) -> pd.DataFrame:
    h = dfs[spec["table"]].copy()
    h = h[h[spec["oid"]].notna()].drop_duplicates(subset=[spec["oid"]])
    o = pd.DataFrame({"_id": h[spec["oid"]].astype(str).to_numpy(),
                      "_d": pd.to_datetime(H.parse_dates(h[spec["date"]])).dt.normalize().to_numpy(),
                      "_s": H.norm_status(h[spec["status"]]).to_numpy()})
    d = dfs[spec["detail"]].copy().drop_duplicates()
    d = d[d[spec["did"]].notna()]
    line = pd.DataFrame({"_id": d[spec["did"]].astype(str).to_numpy(), "_a": H.to_num(d[spec["damount"]]).to_numpy()})
    tot = line.groupby("_id", sort=False)["_a"].sum().rename("amount").reset_index()
    o = o.merge(tot, on="_id", how="left")
    o["amount"] = o["amount"].fillna(0.0)
    return o[["_id", "_d", "_s", "amount"]]


def _orders(dfs, spec) -> pd.DataFrame:
    if spec.get("detail"):
        return _orders_star(dfs, spec)
    df = dfs[spec["table"]].copy()
    df["_d"] = pd.to_datetime(H.parse_dates(df[spec["date"]])).dt.normalize()
    df["_a"] = H.to_num(df[spec["amount"]])
    if spec["status"] is not None:
        df["_s"] = H.norm_status(df[spec["status"]])
    else:
        df["_s"] = "hoan_thanh"
    oid = spec["oid"]
    if oid is None:
        out = df.assign(_id=range(len(df)))
        return out[["_id", "_d", "_s", "_a"]].rename(columns={"_a": "amount"})
    df = df[df[oid].notna()]
    df = df.drop_duplicates()
    amt = H.order_amounts(df.assign(_a=df["_a"]), oid, "_a").rename(columns={oid: "_id"})
    head = df.groupby(oid, sort=False).agg(_d=("_d", "first"), _bad=("_s", lambda s: s.isin(["huy", "tra_hang"]).any()), _s=("_s", "first")).reset_index().rename(columns={oid: "_id"})
    o = head.merge(amt, on="_id")
    o["_s"] = o.apply(lambda r: "huy" if r["_bad"] and r["_s"] not in ("huy", "tra_hang") else r["_s"], axis=1)
    return o[["_id", "_d", "_s", "amount"]]


_GENERIC = set("""z zay zi nhi nha nhe a ak ha vay the roi di ne hen giup xem cho toi minh tiem quan shop cua hang cua hanh ban biet hoi la co duoc bao nhieu nhieu den nay ni qua hom bua
tuan thang nam quy ngay gan nhat tong cong trong ky doanh thu dthu don hoa tinh rieng cai tat ca ve vao lai the thu tien mua duoc may so luot mot tu hai ba bon sau bay tam chin muoi
nua truoc roi vua qua do day ok check kiem tra xin hay lam on ban oi ad em anh chi gium rang gi""".split())


def _unknown_words(question: str) -> bool:
    """True when the question has a word outside a small generic list (product / customer names, other metrics, unusual wording): not a plain 'metric + period' question."""
    from .qnorm import normalize_question

    q = normalize_question(question)
    if re.search(r"\([^)]*\)|\d+\s?(ml|g|kg|l|lit|cai|hop|chai|lon|m2|m3)\b", q.lower()):
        return True
    for w in re.findall(r"[^\W\d_]+", q.lower()):
        k = _n(w)
        if len(k) > 1 and k not in _GENERIC:
            return True
    return False


def parse_simple(question: str, as_of):
    """('revenue'|'count', (start, end)) for a plain 'metric + one period' question, else None."""
    t = _n(question)
    if _QUAL.search(t) or _unknown_words(question):
        return None
    rev, cnt = bool(_REV.search(t)), bool(_CNT.search(t))
    if rev == cnt:
        return None
    per = resolve_periods(question, as_of)
    if len(per) != 1 or "[" in per[0]:
        return None
    m = re.match(r"- .*? = (\d{4}-\d\d-\d\d)(?: \.\. (\d{4}-\d\d-\d\d))?$", per[0])
    if not m:
        return None
    a, b = m.group(1), m.group(2) or m.group(1)
    import datetime as dt

    return ("revenue" if rev else "count"), (dt.date.fromisoformat(a), dt.date.fromisoformat(b))


_PERIOD_LINE = re.compile(r"- (.*?) = (\d{4}-\d\d-\d\d)(?: \.\. (\d{4}-\d\d-\d\d))?$")
_QUAL_FAM = re.compile(
    r"khach|nhom|mat hang|san pham|\bmon\b|\bly\b|size|\bca\b|ca (sang|chieu)|thu (hai|ba|tu|nam|sau|bay)|chu nhat|\btop\b|nhieu nhat|it nhat|"
    r"ngay nao|thang nao|nho nhat|cao nhat|tra hang|hoan hang|\bno\b|ton kho|\bvon\b|\bloi\b|\blai\b|chi phi|giam gia|ship|nhan vien|\btho\b|bao hanh|"
    r"may ngay|bao nhieu ngay|tro len|tro xuong|dat \d|tien mat|chuyen khoan|\bcod\b|\bthe\b|thanh toan|voucher|diem|gio|trong so|bao nhieu loai|moi (ngay|thang|tuan|ca)|rieng|gom|"
    r"ngay \d|tu ngay|den ngay|7 ngay|truoc khi|sau khi|tang gia|giam gia")
_FAM_WORDS = set("""trung binh mot moi gia tri mang ve ti ty le huy bi phan tram thay doi chenh lech tang giam so voi vs sv cung ky lien truoc trc do thang tuan quy nam
ca cua hang bao nhieu nhieu nhiu bnhieu hay khong co la nhi nha vay ha a ak oi hoi xem coi giup gium cho minh toi hen""".split())


def _ranges(per: list[str]):
    """[(label, date, date)] from resolve_periods lines."""
    import datetime as dt

    out = []
    for ln in per:
        m = _PERIOD_LINE.match(ln)
        if m:
            a = dt.date.fromisoformat(m.group(2))
            b = dt.date.fromisoformat(m.group(3) or m.group(2))
            out.append((m.group(1), a, b))
    return out


def _family_text_ok(question: str) -> bool:
    """Only generic words may remain once the family vocabulary is allowed (no product / customer names, no other metric)."""
    from .qnorm import normalize_question

    q = normalize_question(question)
    if re.search(r"\([^)]*\)|\d+\s?(ml|g|kg|l|lit|cai|hop|chai|lon|m2|m3)\b", q.lower()):
        return False
    for w in re.findall(r"[^\W\d_]+", q.lower()):
        k = _n(w)
        if len(k) > 1 and k not in _GENERIC and k not in _FAM_WORDS:
            return False
    return True


def parse_family(question: str, as_of):
    """('avg', win) | ('cancel', win) | ('chg', win, win_prev) for the plain questions of these three families, else None."""
    t = _n(question)
    if _QUAL_FAM.search(t) or not _family_text_ok(question):
        return None
    per = resolve_periods(question, as_of)
    if any("[" in ln for ln in per):
        return None
    rg = _ranges(per)
    if re.search(r"trung binh", t) and re.search(r"(mot|moi|1) (don|hoa don)", t) and not re.search(r"%|phan tram|ti le|so voi", t):
        main = [r for r in rg if "hôm nay" not in r[0] and "so sánh" not in r[0]]
        return ("avg", (main[0][1], main[0][2])) if len(main) == 1 and len(rg) == 1 else None
    if re.search(r"huy", t) and re.search(r"%|phan tram|ti le|ty le", t) and not re.search(r"so voi|tang|giam|chenh", t):
        main = [r for r in rg if "hôm nay" not in r[0]]
        return ("cancel", (main[0][1], main[0][2])) if len(main) == 1 and len(rg) == 1 else None
    if _REV.search(t) and re.search(r"so voi|so vs|\bsv\b", t) and re.search(r"%|phan tram", t) and re.search(r"tang|giam|thay doi|chenh", t):
        cmp_ = [r for r in rg if "so sánh" in r[0]]
        main = [r for r in rg if "hôm nay" not in r[0] and "so sánh" not in r[0] and "cả tháng" not in r[0] and "cả tuần" not in r[0]]
        if len(cmp_) == 1 and len(main) >= 1:
            return ("chg", (main[0][1], main[0][2]), (cmp_[0][1], cmp_[0][2]))
    return None


def _stats(o: pd.DataFrame, a, b):
    """(revenue, valid orders, cancelled, all orders, has_unclassified) in [a, b]."""
    win = o[(o["_d"] >= pd.Timestamp(a)) & (o["_d"] <= pd.Timestamp(b))]
    ok = win[~win["_s"].isin(["huy", "tra_hang"])]
    return float(ok["amount"].sum()), int(len(ok)), int((win["_s"] == "huy").sum()), int(len(win)), bool((win["_s"] == "khac").any())


_CACHE: dict = {}


def compute(dfs: dict[str, pd.DataFrame], question: str, as_of):
    """Independent answer (type, value) or None when the question/data are outside this engine's remit."""
    ps = parse_simple(question, as_of)
    fam = None if ps is not None else parse_family(question, as_of)
    if ps is None and fam is None:
        return None
    key = (id(dfs), tuple((n, len(d), tuple(map(str, d.columns))) for n, d in dfs.items()))     # id() alone can repeat after a business is freed
    if _CACHE.get("key") != key:
        spec = induce(dfs)
        o = _orders(dfs, spec) if spec is not None else None
        _CACHE.update(key=key, spec=spec, o=o)
    spec, o = _CACHE["spec"], _CACHE["o"]
    if spec is None or o is None or o["_d"].isna().mean() > 0.02:
        return None
    if ps is not None:
        rev, cnt, _, _, khac = _stats(o, ps[1][0], ps[1][1])
        if khac:
            return None                                                    # a status nobody classified is inside the period: no opinion
        if ps[0] == "count" and spec.get("blocky"):
            return None
        val = cnt if ps[0] == "count" else int(round(rev + 1e-9))
        return None if val == 0 else ("number", val)     # 0 usually means the period lies outside the data (unanswerable) or a filter broke
    kind = fam[0]
    if spec.get("blocky"):
        return None                                                        # rows are order LINES here: averages / rates per order would be per line
    rev, cnt, canc, tot, khac = _stats(o, *fam[1])
    if khac or tot == 0 or cnt == 0:
        return None
    if kind == "avg":
        return "number", H.round_vnd(rev / cnt)
    if kind == "cancel":
        return "percent", H.round2(100.0 * canc / tot)
    rev0, cnt0, _, tot0, khac0 = _stats(o, *fam[2])
    if khac0 or tot0 == 0 or rev0 <= 0 or rev <= 0:
        return None
    return "percent", H.round2(100.0 * (rev - rev0) / rev0)
