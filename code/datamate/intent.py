"""Deterministic help that does not depend on the LLM reading the rulebook correctly.

1. resolve_periods(): the date ranges named in the question, computed in Python from as_of with the rulebook's rules
   (tuần = thứ Hai-CN, tháng M không ghi năm -> năm nay nếu không sau hôm nay, "tháng này" = từ ngày 1 đến hôm nay ...).
2. rule_checklist(): the rulebook rules that apply to THIS question's metric (revenue, order count, cancel rate, % change, ranking ...).
3. audit_code(): cheap, high-precision static checks on the generated code + result. A finding triggers one targeted repair round.
All of it is advisory: any failure returns "nothing", it never blocks an answer.
"""
from __future__ import annotations

import datetime as dt
import re

from . import helpers as H
from .textutil import strip_accents


def _n(q: str) -> str:
    s = strip_accents(q).lower().replace("đ", "d")
    return re.sub(r"\s+", " ", s)


def _fmt(r) -> str:
    return r[0].isoformat() if r[0] == r[1] else f"{r[0].isoformat()} .. {r[1].isoformat()}"


def _month_no_year(m: int, today: dt.date):
    y = today.year if dt.date(today.year, m, 1) <= today else today.year - 1
    return y


def _quarter_no_year(qn: int, today: dt.date):
    start = H.quarter_range(today.year, qn)[0]
    return today.year if start <= today else today.year - 1


_MY = re.compile(r"(?<![\d/.-])(?:thang\s*)?(\d{1,2})\s*(?:[/-]|\s+nam\s+)\s*(\d{4}|\d{2})(?![\d/])")
_M_ONLY = re.compile(r"\bthang\s*(\d{1,2})(?!\s*(?:[/-]|nam)\s*\d)(?!\d)")
_Q_Y = re.compile(r"\bquy\s*([1-4])\s*(?:[/-]|\s+nam\s+)\s*(\d{4}|\d{2})\b")
_Q_ONLY = re.compile(r"\bquy\s*([1-4])\b(?!\s*[/-]\s*\d)(?!\s+nam\s+\d)")
_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_NDAYS = re.compile(r"\b(\d{1,3})\s*ngay\s*(?:gan nhat|qua|vua qua|gan day|toi nay|truoc)")


def _year(y: str) -> int:
    v = int(y)
    return 2000 + v if v < 100 else v


def resolve_periods(question: str, as_of, span: tuple | None = None) -> list[str]:
    """Human-readable list of resolved date ranges (ISO). Empty list when the question names no period."""
    today = H._d(as_of)
    t = _n(question)
    out: list[str] = []
    seen: set[str] = set()

    def add(label: str, r):
        line = f"- {label} = {_fmt(r)}"
        if span is not None:
            lo, hi = H._d(span[0]), H._d(span[1])
            if r[1] < lo or r[0] > hi:
                line += f"  [ENTIRELY OUTSIDE the data range {lo}..{hi} -> unanswerable, not 0]"
            elif r[0] < lo or r[1] > hi:
                line += f"  [only partly inside the data range {lo}..{hi}: compute on the part that exists]"
        if line not in seen:
            seen.add(line)
            out.append(line)

    main_month = None
    for m in _Q_Y.finditer(t):
        add(f"quý {m.group(1)}/{_year(m.group(2))}", H.quarter_range(_year(m.group(2)), int(m.group(1))))
    t = _Q_Y.sub(" ", t)                       # '4 năm 2025' of 'quý 4 năm 2025' is not a month
    for m in _MY.finditer(t):
        mo, y = int(m.group(1)), _year(m.group(2))
        if 1 <= mo <= 12:
            main_month = (y, mo)
            add(f"tháng {mo}/{y}", H.month_range(y, mo))
    covered_month = bool(out)
    if not covered_month:
        for m in _M_ONLY.finditer(t):
            mo = int(m.group(1))
            if 1 <= mo <= 12:
                y = _month_no_year(mo, today)
                main_month = (y, mo)
                add(f"tháng {mo} (không ghi năm -> {y})", H.month_range(y, mo))
        for m in _Q_ONLY.finditer(t):
            qn = int(m.group(1))
            y = _quarter_no_year(qn, today)
            add(f"quý {qn} (không ghi năm -> {y})", H.quarter_range(y, qn))
    for m in _DATE.finditer(t):
        try:
            d = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            add(f"ngày {d.strftime('%d/%m/%Y')}", (d, d))
        except ValueError:
            pass
    for m in _NDAYS.finditer(t):
        add(f"{m.group(1)} ngày gần nhất (gồm hôm nay)", H.last_n_days(today, int(m.group(1)), include_today=True))
    if re.search(r"\b(hom nay|bua nay|hnay|hom ni|ngay hom nay)\b", t):
        add("hôm nay", (today, today))
    if re.search(r"\b(hom qua|bua qua|hqua|ngay hom qua|hom bua)\b", t):
        y = today - dt.timedelta(days=1)
        add("hôm qua", (y, y))
    if re.search(r"\btuan (nay|ni)\b", t):
        add("tuần này (thứ Hai .. hôm nay)", H.this_week_to_date(today))
    if re.search(r"\btuan (truoc|roi|qua|vua roi)\b", t):
        add("tuần trước (cả tuần, thứ Hai..Chủ nhật)", H.last_week(today))
    if re.search(r"\bthang (nay|ni)\b", t):
        add("tháng này (ngày 1 .. hôm nay)", H.this_month_to_date(today))
        if re.search(r"so voi|thang truoc|cung ky", t):
            add("tháng trước, cùng số ngày để so sánh", H.same_period_prev_month(today))
    if re.search(r"\bthang (truoc|roi|vua roi)\b", t) and not re.search(r"so voi (thang )?(lien )?truoc", t):
        add("tháng trước (cả tháng)", H.last_month(today))
    if re.search(r"\bquy (nay|ni)\b", t):
        add("quý này (đầu quý .. hôm nay)", H.this_quarter_to_date(today))
    if re.search(r"\bquy (truoc|roi)\b", t):
        add("quý trước", H.last_quarter(today))
    if re.search(r"tu dau nam|dau nam (toi|den) (nay|hom nay)", t):
        add("từ đầu năm đến hôm nay", H.this_year_to_date(today))
    if not out:
        m = re.search(r"\bnam (\d{4})\b", t)
        if m:
            add(f"năm {m.group(1)}", H.year_range(int(m.group(1))))
    # comparison baselines
    if main_month and re.search(r"so voi (thang )?(lien )?(truoc|ngay truoc)|thang lien truoc|thang truoc do|so voi thang truoc", t):
        y, mo = main_month
        py, pm = (y, mo - 1) if mo > 1 else (y - 1, 12)
        add(f"tháng liền trước tháng {mo}/{y} (để so sánh)", H.month_range(py, pm))
    if main_month and re.search(r"cung (ky|thang)( nam)? (truoc|ngoai)|nam ngoai|nam truoc", t):
        y, mo = main_month
        add(f"cùng tháng năm trước ({mo}/{y - 1})", H.month_range(y - 1, mo))
    return out


# --------------------------------------------------------------------------- rule checklist
_CHECK = [
    (r"doanh thu|dthu|ban (duoc )?(bao nhieu|nhieu) tien|tien ban|thu ve|chot duoc bao nhieu tien|tong tien|tien hang|doanh so",
     "REVENUE = sum of the LINE amounts (thành tiền after line discount) of orders that are NOT cancelled and NOT returned (an order returned even partly is a returned order: drop the whole order; orders still being delivered count). Do NOT subtract an order-level discount/voucher, do NOT add VAT or shipping. The period is the ORDER date. Count an order-level amount repeated on every line only once per order."),
    (r"bao nhieu don|may don|so don|o do|tong don|don hang|\bdh\b",
     "ORDER COUNT = number of DISTINCT orders (not rows) that are not cancelled and not returned, by order date (drop_duplicates on the order id)."),
    (r"trung binh (moi |mot |1 )?(don|hoa don)|gia tri (trung binh )?(1|mot|moi) don",
     "AVERAGE ORDER VALUE = revenue / number of qualifying orders (same filters as revenue/orders); money is rounded to the dong (half up)."),
    (r"ti le (don |dh )?huy|huy.*(phan tram|%)|% (don )?huy|phan tram don huy",
     "CANCEL RATE = cancelled orders / ALL orders placed in the period (every status: cancelled, returned, in delivery, completed) x 100, 2 decimals. A returned order is NOT a cancelled order."),
    (r"tang truong|thay doi bao nhieu|tang|giam|so voi|chenh lech|hon kem",
     "CHANGE: percent = (new - old) / old x 100 with 2 decimals, NEGATIVE when it fell, old = the earlier/compared period. A difference in absolute terms = asked period - compared period (negative if smaller)."),
    (r"ban chay|nhieu nhat|top \d|mon nao|mat hang|hot nhat|it nhat",
     "RANKING items by QUANTITY in BASE units (convert big units with the conversion table), not by rows or money, excluding cancelled/returned orders. Ties: smaller code first. Ignore service-fee lines. Answer with names exactly as in the data (list in rank order)."),
    (r"khach",
     "CUSTOMERS: skip anonymous placeholders (Khách lẻ, kl, lẻ, blank). Names differing only in case/accents/honorifics are the SAME customer. Trung thành = most distinct purchase DAYS; sộp = highest average order value among customers with >= 3 orders; hay đổi ý = most cancelled orders; 'chi nhiều tiền nhất' = highest revenue."),
    (r"ngay (nao|mo)|ngay dat khach|dat khach|cao diem|gio nao",
     "BEST DAY/HOUR: ngày bán nhiều tiền nhất = day with highest revenue; ngày đắt khách = day with the most orders; ties -> earlier date; giờ cao điểm = start hour of the busiest 1-hour slot (type number). Dates are answered as YYYY-MM-DD (month questions: YYYY-MM)."),
    (r"\bsize (m|l)\b|\(m\)|\(l\)|\bly\b",
     "SIZE M / L = items whose name ENDS with (M) / (L), case-insensitive; quantity in ly counted per item line (a topping glued to the item name with ' + ' is a separate item, it is not part of the item's quantity)."),
    (r"hom nay|bua nay|hnay|hom ni|tuan nay|thang nay|thang ni|tu dau nam|ngay gan nhat|\d+ ngay qua",
     "TODAY = as_of. Rows dated AFTER as_of have not happened yet: ignore them when the question is relative to today (this week/month, year to date, N latest days, today), even though the file contains them. But an order's cancelled/returned status is the one written on the order, even if the return slip is dated after today."),
    (r"nhom (hang|tra|ca phe|sua|[a-z]+)\b|danh muc|loai hang",
     "PRODUCT GROUP: use the group/category column exactly as named in the question; if the shop's data has NO such column the question is unanswerable (do not guess a group from product names)."),
    (r"con no|cong no|no cua|dang no|ai no",
     "DEBT: the state at the END of the stated day, counting every transaction from the start of the data to that day (not just the month); use the shop's debt/paid columns as the README defines them."),
]


def rule_checklist(question: str) -> list[str]:
    t = _n(question)
    out = []
    for pat, text in _CHECK:
        try:
            if re.search(pat, t):
                out.append(text)
        except re.error:
            continue
    return out[:6]


# --------------------------------------------------------------------------- code audit
_TIME_Q = re.compile(r"\b(hom nay|hom qua|bua nay|bua qua|hqua|hnay|tuan|thang|quy|nam|ngay|\d{1,2}/\d{2,4})\b")


def line_columns_from_profile(profile: str) -> set[str]:
    """LINE-level money columns named in the 'Detected structure' block of the profile ('MONEY per LINE [...]')."""
    cols: set[str] = set()
    for m in re.finditer(r"MONEY per LINE \[(.*?)\]:", profile or ""):
        cols.update(x.strip().strip("'\"") for x in m.group(1).split(", ") if x.strip())
    return cols


def audit_code(question: str, code: str, answer_type: str | None, answer, line_cols: set[str] | None = None, lineage: list | None = None) -> list[str]:
    """Return a list of problems that are almost certainly mistakes (empty = nothing to say)."""
    if answer_type in (None, "unanswerable", "abstain"):
        return []
    t = _n(question)
    c = (code or "").lower()
    issues: list[str] = []
    money = re.search(r"doanh thu|dthu|tien ban|ban (duoc )?(bao nhieu|nhieu) tien|thu ve|tong tien", t) is not None
    if money and not re.search(r"norm_status|status|trang_thai|trang thai|huy|hủy|tra_hang|hoan_thanh|ket qua|giao dich", c):
        issues.append("the question asks for revenue but the code never filters the order status: cancelled and returned orders must be excluded (norm_status), and an order returned even partly is dropped whole")
    if _TIME_Q.search(t) and not re.search(r"as_of|as_of_date|date\(|datetime|between\(|month_range|quarter_range|last_|this_|same_period|\bdt\.|pd\.timestamp|to_datetime|\.dt\.", c):
        issues.append("the question names a period but the code has no date filter derived from as_of or the named period")
    for ev in lineage or []:                       # observed at run time
        kind = ev.get("kind")
        if kind == "unparsed":
            issues.append(f"{ev['bad']} of {ev['total']} values of {ev['name']!r} could not be understood by {ev['fn']}() (e.g. {ev['examples']}): a later filter silently drops those rows. Look at them and handle them (other date/number spellings, summary rows) instead of ignoring them")
            continue
        if kind == "unclassified_status":
            issues.append(f"{ev['bad']} of {ev['total']} status values are not classified by norm_status ('khac'; e.g. {ev['examples']}): filters like isin(['huy','tra_hang']) neither exclude nor include them knowingly. Decide each such status explicitly from the rulebook (cancelled / returned / in delivery / completed)")
            continue
        if kind == "fanout":
            used = [col for col in ev.get("left_numeric", []) if col.lower() in c]
            if used and re.search(r"\.(sum|mean)\(", c):
                issues.append(f"a merge multiplied the rows of the left table ({ev['left_rows']} -> {ev['out_rows']}) and the code sums/averages its numeric column {used[0]!r} afterwards: every left row is now repeated. Aggregate before merging, or de-duplicate the left rows (order level) first")
            continue
        used = [col for col in ev.get("lost", []) if col.lower() in c]
        if used:
            issues.append(f"drop_duplicates({ev.get('key')}) removed {ev.get('rows_before', 0) - ev.get('rows_after', 0)} of {ev.get('rows_before')} rows although column {used[0]!r} has DIFFERENT values on rows sharing that key, and the code uses that column: any sum/mean of it now covers only one row per key. Aggregate that column over all the rows (after filtering them), use order_amounts(...) for money per order, and use drop_duplicates only for columns that are constant within the key or when counting distinct keys")
            break
    if line_cols and "drop_duplicates" in c and ".sum(" in c and "order_amounts" not in c:
        hit = [col for col in line_cols if col.lower() in c]
        if hit:
            issues.append(f"the code calls drop_duplicates(...) and then sums the LINE-level column {hit[0]!r}: that keeps ONE line per order and under-counts the money. Filter the rows (period, status), then use order_amounts(rows, order_col, {hit[0]!r}) or sum the column over all lines; drop_duplicates only for ORDER-level columns or to count orders")
    if answer_type == "percent":
        try:
            if 0 < abs(float(answer)) < 1 and "* 100" not in c and "*100" not in c and "pct_change" not in c:
                issues.append("a percent must be written as a percentage number (12.4 for 12.4 %), not a fraction: multiply by 100")
        except Exception:
            pass
    if answer_type == "number" and money:
        try:
            if abs(float(answer) - round(float(answer))) > 1e-9:
                issues.append("money must be a whole number of dong: round_vnd(...) (half up)")
        except Exception:
            pass
    if answer_type == "number" and re.search(r"bao nhieu don|may don|so don", t) and not re.search(r"nunique|drop_duplicates|count\(distinct|\.count\(|len\(|\.shape|size\(\)", c):
        issues.append("an order count should count distinct orders (nunique / drop_duplicates on the order id)")
    return issues
